"""
IdempotentProcessor — wraps any processing function to make it idempotent.

The core problem:
  Message queues (SQS, Kafka) guarantee at-LEAST-once delivery. A submission
  can arrive twice because:
    - A broker retried after a network timeout (got a 200 but didn't see it)
    - The worker crashed after processing but before acknowledging the message
    - An upstream service has a bug that sends duplicates

  Without idempotency, the second delivery creates a duplicate record,
  triggers a duplicate enrichment API call (costs money), and may price the
  same risk twice (financial exposure).

The pattern — Cache-Aside:
  1. Hash the request payload → idempotency key
  2. Check the store: key exists → return cached result immediately
  3. Process the request (first time only)
  4. Store the result under the key
  5. Return the result

The idempotency key:
  SHA-256 of the JSON-serialised payload with keys sorted.
  Sorting keys means {"a": 1, "b": 2} and {"b": 2, "a": 1} produce the
  same hash — dict key ordering is an implementation detail, not a
  semantic difference.

  Fields excluded from the key:
    - Timestamps ("received_at") — same submission at T+5s is still the same
    - Internal metadata ("correlation_id") — broker-generated tracking IDs
      that may differ between retries
  Callers pass `exclude_fields` to customise this.
"""

import hashlib
import json
import logging
import threading
from typing import Callable, Optional, TypeVar

from store import IdempotencyStore

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Fields excluded from the idempotency hash by default.
# These describe the delivery, not the submission itself.
DEFAULT_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"received_at", "correlation_id", "retry_count", "source_ip"}
)


def compute_idempotency_key(
    payload: dict,
    exclude_fields: frozenset[str] = DEFAULT_EXCLUDED_FIELDS,
) -> str:
    """
    Produce a stable, deterministic key for a submission payload.

    The key uniquely identifies the *content* of the request, not the
    delivery event. Two identical payloads received at different times
    will produce the same key.
    """
    filtered = {k: v for k, v in payload.items() if k not in exclude_fields}
    # sort_keys=True ensures field order doesn't affect the hash
    # default=str handles Decimal, datetime, and other non-JSON-native types
    canonical = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ProcessingRecord:
    """
    Wraps a processing result with metadata about whether it was cached.
    Returned by IdempotentProcessor.process() so callers can log/monitor.
    """

    __slots__ = ("result", "was_replay", "idempotency_key")

    def __init__(self, result: T, *, was_replay: bool, idempotency_key: str) -> None:
        self.result = result
        self.was_replay = was_replay
        self.idempotency_key = idempotency_key


class IdempotentProcessor:
    """
    Makes any processing function idempotent.

    Usage:
        store = InMemoryIdempotencyStore()
        processor = IdempotentProcessor(store, triage_engine.process)

        record1 = processor.process(payload)  # processes, stores result
        record2 = processor.process(payload)  # returns cached result

        assert record1.result == record2.result
        assert record1.was_replay is False
        assert record2.was_replay is True
        # The processing function was only called once

    Generic design:
        The processor does not know what `processor_fn` does. It works for
        triage, enrichment, or any other function that takes a dict and returns
        something serialisable. This is the Open/Closed Principle applied to
        operational concerns: adding idempotency to a function does not require
        changing the function.

    Concurrency model — Double-Checked Locking with per-key locks:
        A single global lock would serialise ALL processing, including different
        payloads that have no relation to each other. Instead, we hold a
        per-key lock so only requests for the SAME key block each other.

        The pattern:
          1. Fast path (no lock): if already cached, return immediately.
          2. Acquire the per-key lock.
          3. Re-check under the lock: another thread may have processed while
             we were waiting to acquire the lock.
          4. If still not cached: process, store, release lock.

        This guarantees exactly-once processing per key within a single process.
        For multi-process/distributed systems, replace InMemoryIdempotencyStore
        with a Redis-backed store using SET key value NX (atomic check-and-set).
    """

    def __init__(
        self,
        store: IdempotencyStore,
        processor_fn: Callable[[dict], T],
        exclude_fields: frozenset[str] = DEFAULT_EXCLUDED_FIELDS,
    ) -> None:
        self._store = store
        self._fn = processor_fn
        self._exclude_fields = exclude_fields
        # Per-key locks: different keys don't block each other
        self._key_locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # guards _key_locks dict itself

    def _get_key_lock(self, key: str) -> threading.Lock:
        """Return (and lazily create) a per-key lock."""
        with self._meta_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def process(self, payload: dict) -> ProcessingRecord:
        """
        Process a payload idempotently using double-checked locking.

        The processing function is guaranteed to be called at most once
        per unique payload within this process.
        """
        key = compute_idempotency_key(payload, self._exclude_fields)
        short_key = key[:16]

        # Fast path: common case where the result is already cached.
        # No lock needed — we re-check under lock if this returns None.
        cached = self._store.get(key)
        if cached is not None:
            logger.info("Idempotency replay: returning cached result for key=%s", short_key)
            return ProcessingRecord(cached, was_replay=True, idempotency_key=key)

        # Slow path: acquire per-key lock to serialise concurrent duplicates.
        with self._get_key_lock(key):
            # Re-check: another thread may have processed while we waited.
            cached = self._store.get(key)
            if cached is not None:
                logger.info(
                    "Idempotency replay (post-lock): returning cached result for key=%s",
                    short_key,
                )
                return ProcessingRecord(cached, was_replay=True, idempotency_key=key)

            # We hold the lock and the key still doesn't exist — safe to process.
            logger.info("Processing new submission, key=%s", short_key)
            result = self._fn(payload)
            self._store.set_if_absent(key, result)
            return ProcessingRecord(result, was_replay=False, idempotency_key=key)
