"""
Tests for the idempotency store and processor.

Senior testing patterns demonstrated:
  - Testing the store in isolation before the processor (unit before integration)
  - Testing concurrency with threads: verifies that the lock works and the
    processing function is only called once even under concurrent load
  - Testing that excluded fields are genuinely excluded from the key
  - Testing the race condition path (set_if_absent returns False)
  - pytest-timeout prevents tests from hanging if a deadlock is introduced
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processor import IdempotentProcessor, compute_idempotency_key
from store import InMemoryIdempotencyStore


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def store() -> InMemoryIdempotencyStore:
    return InMemoryIdempotencyStore()


@pytest.fixture()
def mock_fn():
    """A MagicMock that returns the payload it receives (acts as an identity function)."""
    fn = MagicMock(side_effect=lambda payload: {"processed": True, **payload})
    return fn


@pytest.fixture()
def processor(store, mock_fn) -> IdempotentProcessor:
    return IdempotentProcessor(store, mock_fn)


PAYLOAD = {"company_id": "ACME-001", "revenue": 5_000_000, "industry": "Retail"}


# --------------------------------------------------------------------------- #
# 1. Idempotency key computation                                               #
# --------------------------------------------------------------------------- #


class TestIdempotencyKey:
    def test_same_payload_produces_same_key(self):
        key1 = compute_idempotency_key(PAYLOAD)
        key2 = compute_idempotency_key(PAYLOAD)
        assert key1 == key2

    def test_different_payload_produces_different_key(self):
        other = {**PAYLOAD, "revenue": 9_999_999}
        assert compute_idempotency_key(PAYLOAD) != compute_idempotency_key(other)

    def test_dict_key_order_does_not_affect_key(self):
        """Dict insertion order must not affect the hash — same content = same key."""
        flipped = {"industry": "Retail", "revenue": 5_000_000, "company_id": "ACME-001"}
        assert compute_idempotency_key(PAYLOAD) == compute_idempotency_key(flipped)

    def test_excluded_fields_do_not_affect_key(self):
        """A payload with a 'received_at' timestamp should produce the same key
        as the same payload without it — delivery metadata is excluded."""
        with_timestamp = {**PAYLOAD, "received_at": "2024-01-01T00:00:00Z"}
        without_timestamp = PAYLOAD
        assert (
            compute_idempotency_key(with_timestamp)
            == compute_idempotency_key(without_timestamp)
        )

    def test_custom_excluded_fields_respected(self):
        payload_a = {**PAYLOAD, "broker_note": "first submission"}
        payload_b = {**PAYLOAD, "broker_note": "retry"}
        key_a = compute_idempotency_key(payload_a, exclude_fields=frozenset({"broker_note"}))
        key_b = compute_idempotency_key(payload_b, exclude_fields=frozenset({"broker_note"}))
        assert key_a == key_b

    def test_custom_included_field_creates_different_key(self):
        """Without excluding broker_note, different notes = different keys."""
        payload_a = {**PAYLOAD, "broker_note": "first submission"}
        payload_b = {**PAYLOAD, "broker_note": "retry"}
        assert (
            compute_idempotency_key(payload_a, exclude_fields=frozenset())
            != compute_idempotency_key(payload_b, exclude_fields=frozenset())
        )


# --------------------------------------------------------------------------- #
# 2. InMemoryIdempotencyStore                                                  #
# --------------------------------------------------------------------------- #


class TestInMemoryStore:
    def test_get_returns_none_for_unseen_key(self, store):
        assert store.get("nonexistent") is None

    def test_set_if_absent_stores_value(self, store):
        stored = store.set_if_absent("key1", "result")
        assert stored is True
        assert store.get("key1") == "result"

    def test_set_if_absent_does_not_overwrite(self, store):
        store.set_if_absent("key1", "first")
        stored_again = store.set_if_absent("key1", "second")
        assert stored_again is False
        assert store.get("key1") == "first"  # original value preserved

    def test_exists_reflects_storage(self, store):
        assert store.exists("k") is False
        store.set_if_absent("k", "v")
        assert store.exists("k") is True


# --------------------------------------------------------------------------- #
# 3. IdempotentProcessor — correctness                                         #
# --------------------------------------------------------------------------- #


class TestIdempotentProcessor:
    def test_first_call_processes_and_returns_result(self, processor, mock_fn):
        record = processor.process(PAYLOAD)
        assert record.was_replay is False
        mock_fn.assert_called_once()

    def test_second_call_returns_cached_result(self, processor, mock_fn):
        result1 = processor.process(PAYLOAD)
        result2 = processor.process(PAYLOAD)

        assert result1.result == result2.result
        assert result2.was_replay is True
        mock_fn.assert_called_once()  # processing function called exactly once

    def test_different_payloads_are_processed_independently(self, processor, mock_fn):
        payload_b = {**PAYLOAD, "company_id": "OTHER-002"}
        processor.process(PAYLOAD)
        processor.process(payload_b)
        assert mock_fn.call_count == 2   # each distinct payload processed once

    def test_idempotency_key_is_returned_in_record(self, processor):
        record = processor.process(PAYLOAD)
        assert len(record.idempotency_key) == 64  # SHA-256 hex = 64 chars

    def test_replayed_key_matches_original_key(self, processor):
        record1 = processor.process(PAYLOAD)
        record2 = processor.process(PAYLOAD)
        assert record1.idempotency_key == record2.idempotency_key

    def test_excluded_field_does_not_cause_reprocessing(self, processor, mock_fn):
        """A delivery-level field change must NOT trigger reprocessing."""
        processor.process(PAYLOAD)
        payload_with_timestamp = {**PAYLOAD, "received_at": "2099-01-01T00:00:00Z"}
        record = processor.process(payload_with_timestamp)

        assert record.was_replay is True
        mock_fn.assert_called_once()


# --------------------------------------------------------------------------- #
# 4. Concurrency — the lock must prevent double processing                     #
# --------------------------------------------------------------------------- #


class TestConcurrency:
    @pytest.mark.timeout(5)
    def test_concurrent_identical_requests_process_only_once(self, store):
        """
        10 threads submit the same payload simultaneously.
        The processing function must be called exactly once.

        This is the core guarantee of idempotency under concurrent load.
        Without the threading.Lock in InMemoryIdempotencyStore, multiple
        threads could pass the 'key not found' check before any of them
        writes the result — each would then call the processing function.
        """
        call_count = 0
        call_lock = threading.Lock()

        def slow_processor(payload: dict) -> dict:
            nonlocal call_count
            time.sleep(0.01)   # simulate work — increases chance of race conditions
            with call_lock:
                call_count += 1
            return {"processed": True}

        processor = IdempotentProcessor(store, slow_processor)
        threads = [threading.Thread(target=processor.process, args=(PAYLOAD,)) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1, (
            f"Processing function was called {call_count} times — "
            "the idempotency lock is not working correctly"
        )

    @pytest.mark.timeout(5)
    def test_different_payloads_can_be_processed_concurrently(self, store, mock_fn):
        """
        Different payloads should NOT block each other — only identical payloads
        need to be deduplicated.
        """
        payloads = [{**PAYLOAD, "company_id": f"C-{i}"} for i in range(5)]
        processor = IdempotentProcessor(store, mock_fn)
        threads = [threading.Thread(target=processor.process, args=(p,)) for p in payloads]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mock_fn.call_count == 5   # each unique payload processed exactly once
