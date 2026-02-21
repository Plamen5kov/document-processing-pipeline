# Task 02 — Idempotent Submission Processing

## The Problem

Message queues (SQS, Kafka, RabbitMQ) guarantee **at-least-once delivery**.
A submission can be delivered to your worker more than once because:

- A broker retried after a network timeout (they sent the submission but never
  received our 200 OK, so they assume it failed)
- The worker crashed after processing the submission but before acknowledging
  the message — the queue re-delivers to another worker
- An upstream service has a retry bug and sends the same event twice

Without idempotency, the second delivery:
- Creates a duplicate record in the database (underwriters see the same company twice)
- Triggers a duplicate enrichment API call (costs money, wastes API quota)
- May price the same risk twice (financial exposure)

The solution is **idempotent processing**: processing the same payload N times
must produce the same outcome as processing it once, with no additional side effects.

---

## Solutions Considered

### Option A — Unique constraint on company_id

```sql
INSERT INTO submissions (company_id, ...) VALUES (?, ...)
ON CONFLICT (company_id) DO NOTHING;
```

**Why this is insufficient:**
- `company_id` is assigned by the broker — two different brokers can submit the
  same company with different IDs (the deduplication problem from Task 04).
- It doesn't prevent duplicate processing — the enrichment API is called before
  the INSERT, so two concurrent workers still call it twice.
- It conflates "same company" with "same submission." The unique constraint should
  be on the request, not the business entity.

---

### Option B — Check the database before processing

```python
if db.query("SELECT 1 FROM submissions WHERE company_id = ?", cid):
    return existing_result
process(submission)
```

**Why we rejected this:**
- Same problem: the check is on the business entity, not the request.
- Race condition: two workers can both pass the SELECT before either completes
  the INSERT. This is a classic check-then-act TOCTOU bug.
- Couples the idempotency logic to the submission schema — the check has to
  know what makes a submission "the same."

---

### Option C — Idempotency key + dedicated store ✓

Compute a **content hash** of the request payload, check a dedicated
fast store (Redis in production, in-memory for tests) before processing.
This approach is payload-agnostic — it works for any processing function
without knowing what fields it uses.

---

## Patterns Used

### Idempotency Key — SHA-256 of the Canonical Payload

```python
filtered = {k: v for k, v in payload.items() if k not in exclude_fields}
canonical = json.dumps(filtered, sort_keys=True, default=str)
key = hashlib.sha256(canonical.encode()).hexdigest()
```

Three design decisions:

**`sort_keys=True`:** Dict key insertion order is an implementation detail.
`{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` represent the same submission — they
must hash to the same key.

**`exclude_fields`:** Delivery metadata (timestamps, correlation IDs, retry counts)
must not affect the key. The same submission at T+0 and T+5 seconds — delivered
twice because of a network blip — is the same submission. The business content
is what defines identity, not the envelope.

**`default=str`:** `json.dumps` fails on `Decimal`, `datetime`, `UUID`. Using
`str` as a fallback serialises them as their string representation, which is
stable across processes and Python versions for these types.

---

### Cache-Aside Pattern

The check-then-process-then-store sequence:

```
                        ┌─────────────────────────────┐
Request arrives    ─→   │  compute idempotency key    │
                        └────────────────┬────────────┘
                                         ↓
                        ┌─────────────────────────────┐
                        │  check store (fast path)    │ ─→ HIT: return cached result
                        └────────────────┬────────────┘
                                   MISS  ↓
                        ┌─────────────────────────────┐
                        │  acquire per-key lock       │
                        └────────────────┬────────────┘
                                         ↓
                        ┌─────────────────────────────┐
                        │  re-check store (post-lock) │ ─→ HIT: another thread beat us
                        └────────────────┬────────────┘
                                   MISS  ↓
                        ┌─────────────────────────────┐
                        │  process() — called once    │
                        └────────────────┬────────────┘
                                         ↓
                        ┌─────────────────────────────┐
                        │  store result + release lock│
                        └─────────────────────────────┘
```

---

### Double-Checked Locking for Concurrent Safety

The previous version of this processor had a race condition: 10 threads all
calling `store.get(key)` simultaneously all see `None` (key doesn't exist yet),
all proceed to `processor_fn()`, and call it 10 times. The `set_if_absent`
in the store prevented 9 of the 10 results from being stored, but the processing
function still ran 10 times.

The correct fix is per-key locking in the processor, not in the store:

```python
# Fast path — no lock (handles the common case: result already exists)
cached = self._store.get(key)
if cached is not None:
    return ProcessingRecord(cached, was_replay=True, ...)

# Slow path — acquire per-key lock
with self._get_key_lock(key):
    # Re-check: another thread may have processed while we waited
    cached = self._store.get(key)
    if cached is not None:
        return ProcessingRecord(cached, was_replay=True, ...)

    # We hold the lock, key confirmed absent — safe to process
    result = self._fn(payload)
    self._store.set_if_absent(key, result)
    return ProcessingRecord(result, was_replay=False, ...)
```

This is **Double-Checked Locking**:
1. First check (no lock): eliminates lock overhead for the common cached case.
2. Acquire lock: serialises only requests for the same key.
3. Second check (under lock): handles the window between the first check and
   lock acquisition.

**Per-key locks, not a global lock:** A global lock would serialise all processing,
including completely unrelated submissions. Per-key locks only block concurrent
requests for the *same* key. Different keys run concurrently without interference.

---

### Production Backend: Redis NX

`InMemoryIdempotencyStore` is correct for a single process. For a distributed
service running multiple workers:

```bash
# Redis SET with NX (set only if Not eXists) + EX (expire after N seconds)
# This is atomic — no separate check-and-set race condition
SET idempotency:{key} {result} EX 86400 NX
```

The `NX` flag makes the operation atomic at the Redis level. There is no window
between "check if key exists" and "set the key" — both happen in one command.

PostgreSQL equivalent:
```sql
INSERT INTO idempotency_keys (key, result, expires_at)
VALUES ($1, $2, NOW() + INTERVAL '24 hours')
ON CONFLICT (key) DO NOTHING
RETURNING result;
```

The `store.py` `IdempotencyStore` ABC exists precisely so you can swap the
backend without changing the processor or any callers.

---

## Engineering Deep Dive

### Orthogonality & Separation of Concerns

| Layer | File | Responsibility | Does NOT know about |
|---|---|---|---|
| Persistence | `store.py` | Store and retrieve results | What the results mean, how keys are computed |
| Processing | `processor.py` | Key computation, locking, cache-aside | What `processor_fn` does, storage backend |

A change to the storage backend (Redis instead of in-memory) → implement a new
`IdempotencyStore` subclass, inject it. The processor, the processing function,
and all tests of the processor are unchanged.

A change to what fields are excluded from the key → pass a different `exclude_fields`
to the processor constructor. The store is unchanged.

The processor wraps *any* function — it does not know whether it is wrapping
triage logic, enrichment, or a database write. This is the Open/Closed Principle:
idempotency is an operational concern added on top of existing functions without
modifying them.

---

### Library Trade-offs

**SHA-256 vs UUID as idempotency key**

Some systems accept a client-provided idempotency key (a UUID the caller generates
and sends in a header). This is common in payment APIs (Stripe uses this model).

| Approach | Pros | Cons |
|---|---|---|
| Content hash (SHA-256) | Server-generated — no client cooperation needed | Hash collision risk (negligible for SHA-256) |
| Client-provided UUID | Explicit client intent | Requires the broker to generate and track UUIDs |
| Composite business key | Human-readable (`company_id + broker_id`) | Must know the business schema; brittle if schema changes |

For a submission processing system where brokers don't always send idempotency
keys, content hashing is the most practical approach.

**`threading.Lock` vs `asyncio.Lock`**

`threading.Lock` is the right primitive when:
- The code runs in threads (as it does here — task_05's `ThreadPoolExecutor` pattern)
- The lock may be held across blocking I/O

`asyncio.Lock` is the right primitive when:
- The code is entirely async (coroutines)
- You need to yield to the event loop while waiting for the lock

Mixing them (holding an `asyncio.Lock` in a thread, or a `threading.Lock` in a
coroutine) will either deadlock or not provide mutual exclusion. Know which
concurrency model you are in before choosing a lock type.

---

### System Resiliency

**What happens if the processor_fn raises?**

The current implementation does not catch exceptions from `processor_fn`. An
exception propagates out of `process()` and the key is never stored. This is
intentional: a failed processing attempt should not be cached — the next
delivery should retry processing from scratch.

In production, you would log the exception and re-raise, allowing the message
queue to redeliver (at-least-once semantics work in your favour here).

**What happens if the store.set_if_absent call fails?**

If the store write fails after processing, the result is lost and the key is
not recorded. The next delivery will process again (correct behaviour — better
to process twice than to lose the result). For Redis, this means the `SET NX`
call should be retried on connection error before giving up.

**Monitoring idempotency replay rate:**

```python
record = processor.process(payload)
metrics.increment("submission.processed", tags={"replay": record.was_replay})
```

A high replay rate (>5%) indicates upstream is retrying too aggressively —
investigate the broker integration or the message queue configuration.
A replay rate of 0% in a system that has been running for weeks might indicate
the idempotency key computation is wrong (every delivery looks "new").

---

## File Structure

```
task_02_idempotency/
├── store.py        # IdempotencyStore ABC + InMemoryIdempotencyStore
├── processor.py    # IdempotentProcessor with double-checked locking
└── tests/
    └── test_idempotency.py
```

## Running the Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
```
