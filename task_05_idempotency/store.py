"""
Idempotency Store — abstract interface + in-memory implementation.

What is an idempotency store?
  A key-value cache that maps "I have seen this exact request before"
  to "here is the result I returned the first time."
  It turns any function into an idempotent one: calling it N times with
  the same input produces the same output as calling it once.

Production backend:
  Replace InMemoryIdempotencyStore with Redis:

      RESULT=$(redis-cli SET idempotency:{key} {result} EX 86400 NX)
      # NX = "set only if Not eXists" — atomic check-and-set
      # EX 86400 = expire after 24 hours (prevents unbounded growth)

  The NX flag is critical. Without it, two simultaneous requests with the
  same key could both pass the "not exists" check and both process — a
  race condition. Redis NX makes the check-and-set atomic.

  A PostgreSQL equivalent:
      INSERT INTO idempotency_keys (key, result) VALUES (?, ?)
      ON CONFLICT (key) DO NOTHING
      RETURNING result;

Thread safety:
  InMemoryIdempotencyStore uses a threading.Lock so it is safe for
  concurrent use within a single process. This is equivalent to Redis NX
  for the single-process case.
"""

import threading
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class IdempotencyStore(ABC, Generic[T]):
    """
    Minimal interface for an idempotency backend.

    Concrete implementations: InMemoryIdempotencyStore (tests/single-process),
    RedisIdempotencyStore (production distributed service).
    """

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Return the stored result for this key, or None if unseen."""
        ...

    @abstractmethod
    def set_if_absent(self, key: str, value: T) -> bool:
        """
        Store the result only if the key does not already exist.

        Returns True if the value was stored (first time seen).
        Returns False if the key already existed (concurrent duplicate).

        This must be ATOMIC. In Redis: SET key value NX.
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether this key has been seen before."""
        ...


class InMemoryIdempotencyStore(IdempotencyStore[T]):
    """
    Thread-safe in-memory implementation. Suitable for:
      - Single-process applications
      - Tests
      - Development environments

    NOT suitable for:
      - Multi-process deployments (each process has its own memory)
      - Restarts (state is lost)
    """

    def __init__(self) -> None:
        self._store: dict[str, T] = {}
        # A single lock covers both the read and write in set_if_absent,
        # making the check-then-set sequence atomic within the process.
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            return self._store.get(key)

    def set_if_absent(self, key: str, value: T) -> bool:
        with self._lock:
            if key in self._store:
                return False   # already exists — do not overwrite
            self._store[key] = value
            return True        # stored for the first time

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
