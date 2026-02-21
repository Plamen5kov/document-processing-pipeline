# Document Processing Pipeline — Design Patterns Training

A self-contained training project for Senior Backend Engineer interview preparation,
built around a realistic **insurance submission processing** domain. Each task is an
independent, runnable module that introduces one or more production-grade patterns.
Task 06 wires them all into a single end-to-end pipeline.

---

## Why This Domain

Insurance carriers receive thousands of broker submissions per day. Each one must be:

1. **Validated** — required fields present, data in the right format
2. **Triaged** — does it fall within our underwriting appetite?
3. **Deduplicated** — have we seen this company before?
4. **Enriched** — pull external risk data to assist the underwriter
5. **Processed idempotently** — message queues redeliver; we must not act twice

This domain is rich enough to motivate every pattern naturally. None of the patterns
are added for their own sake — each one solves a concrete problem that arises in the
submission flow.

---

## Repository Structure

Folders are numbered in the order they are used in the end-to-end pipeline:

```
task_01_llm_extraction/  LLM extraction with Pydantic schema enforcement
task_02_idempotency/     SHA-256 idempotency key + double-checked locking
task_03_triage/          Pydantic v2 validation + Strategy pattern
task_04_deduplication/   Inverted-index pre-filter + RapidFuzz fuzzy matching
task_05_enrichment/      HTTP client, tenacity retry + ThreadPoolExecutor batching
task_06_pipeline/        Chain of Responsibility wiring all tasks together
```

Each folder contains:
- Source modules with docstrings explaining every design decision
- A `tests/` directory with pytest suites covering normal paths, edge cases, and
  concurrency scenarios
- A `README.md` with problem statement, alternatives considered, and an Engineering
  Deep Dive section answering the "why" behind every choice

---

## Patterns Covered

### Schema Enforcement via LLM — Task 01 (Extraction)

**Problem:** Brokers submit free-text emails and PDFs rather than structured forms.
An LLM can extract fields, but LLM output is probabilistic — it may hallucinate,
omit fields, or return malformed JSON.

**Pattern:** A `Pydantic` schema acts as a contract between the LLM and downstream
code. All fields are `Optional` (the LLM may not find them). A retry loop re-prompts
with a correction suffix on `ValidationError` or `JSONDecodeError`. An
`ExtractionStatus` enum (SUCCESS / PARTIAL / FAILED) lets callers handle degraded
results without crashing.

**Key insight:** The LLM is an unreliable external dependency — treat it the same
way you treat an unreliable API: schema validation, retry, and graceful degradation.

---

### Cache-Aside + Double-Checked Locking — Task 02 (Idempotency)

**Problem:** Message queues guarantee at-least-once delivery. A submission can arrive
twice if the broker retries, the network drops an ACK, or the consumer crashes after
processing but before acknowledging. Processing twice creates duplicate records and
double-charges the enrichment API.

**Pattern:** A SHA-256 hash of the sorted JSON payload is the idempotency key.
Before processing, check the store. On a hit, return the cached result immediately.
On a miss, acquire a per-key lock, re-check under the lock (another thread may have
raced ahead), process exactly once, store the result.

**Key insight:** Double-checked locking eliminates the check-then-act race without
serialising unrelated keys behind a single global lock. The distributed equivalent
is Redis `SET key value NX` (atomic check-and-set).

```python
# Fast path — no lock
if (cached := store.get(key)):
    return cached

# Slow path — per-key lock, re-check
with key_lock:
    if (cached := store.get(key)):   # another thread may have won the race
        return cached
    result = process(payload)
    store.set_if_absent(key, result)
```

---

### Strategy — Task 03 (Triage)

**Problem:** Underwriting appetite rules change frequently. New states get blacklisted,
revenue thresholds shift, sanctioned industries are updated by compliance. Hardcoding
these as `if/elif` chains makes every change a surgery on core logic.

**Pattern:** Each rule is a class implementing a common `Rule` interface with a single
`evaluate(submission) -> RuleResult | None` method. The engine holds a list of rules
and calls them in order — first match wins. Adding a new rule means writing one new
class and appending it to the list. No existing code changes.

**Key insight:** Strategy separates *what algorithm to run* from *when to run it*.
The engine iterates; the rules don't know about each other.

```python
class RevenueOutOfAppetiteRule(Rule):
    def evaluate(self, submission: Submission) -> RuleResult | None:
        if not (self.min_revenue <= submission.revenue <= self.max_revenue):
            return RuleResult(status=TriageStatus.DECLINED, reason="Revenue out of appetite")
        return None
```

---

### Inverted Index + Fuzzy Match — Task 04 (Deduplication)

**Problem:** "Goldman Sachs Group LLC" and "Goldman Sachs Grp" are the same company.
Running fuzzy similarity against every record in the database is O(n) per query —
unacceptable at scale.

**Pattern:** An inverted index on the first 3 digits of the ZIP code pre-filters
candidates to a small geographic bucket (O(1) lookup). RapidFuzz
`token_sort_ratio` then runs only within the bucket — immune to word-order
differences ("New York Life Insurance" vs "Insurance New York Life").

**Key insight:** The index trades memory for time. Pre-filtering converts an O(n)
scan into an O(k) scan where k << n, making fuzzy matching practical at scale.

---

### Retry + Thread Pool — Task 05 (Enrichment)

**Problem:** External risk APIs are unreliable. A single transient failure (network
blip, upstream restart) should not permanently fail a submission. Enriching a batch
one at a time is also slow — each call blocks on network I/O for ~200ms.

**Pattern:** `tenacity` wraps the API call with configurable retry logic:
`stop_after_attempt(3)` caps total attempts, `wait_exponential` spaces them out
(1s → 2s → 4s), `retry_if_exception_type` retries only on retriable errors
(5xx, timeout), and `reraise=True` propagates the final failure rather than
swallowing it. `ThreadPoolExecutor` runs multiple enrichment calls concurrently —
Python's GIL is released during I/O, so threads genuinely run in parallel for
network calls.

**Key insight:** Retry handles *one request hiccuping* (milliseconds). It is not
designed for sustained outages — that is the circuit breaker's job (Task 06).
Threads are the right tool for I/O-bound concurrency; use `multiprocessing` or
`asyncio` for CPU-bound or high-concurrency async work.

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RiskApiServerError),
    reraise=True,
)
def _fetch_risk_data(company_id: str) -> dict: ...
```

---

### Chain of Responsibility + Circuit Breaker — Task 06 (Pipeline)

**Problem:** Tasks 01–05 must run in sequence with specific short-circuit rules:
no enrichment on a declined submission, no triage on invalid data. A single
function with nested `if/return` statements tangles control flow with business
logic, and adding a new stage requires editing and understanding the whole function.
When the enrichment API goes down, every queued submission waits for the timeout
and retries — 15s wasted per submission, 4+ hours at scale.

**Pattern:** Each stage is a `Handler` subclass that receives a `SubmissionContext`,
does its work, then either returns early (short-circuit) or calls `self._next(ctx)`
to continue. Handlers are wired fluently at startup:

```python
head.set_next(ValidationHandler()) \
    .set_next(TriageHandler(rules)) \
    .set_next(DeduplicationHandler(existing_ids)) \
    .set_next(EnrichmentHandler(risk_db, breaker))
```

A `CircuitBreaker` wraps `EnrichmentHandler` — when the API failure rate exceeds a
threshold, `allow_request()` returns `False` and the handler skips the API entirely.

**Hard short-circuits** (further processing is pointless or wasteful):
- `IdempotencyHandler` — cache hit, no work needed
- `ValidationHandler` — missing fields, nothing to triage
- `TriageHandler` — DECLINED, no point enriching a rejected submission

**Soft pass-throughs** (processing continues despite the flag):
- `DeduplicationHandler` — potential duplicate still needs enrichment for review
- `EnrichmentHandler` — failed enrichment is recorded as a warning, not fatal

**Key insight:** CoR and Strategy coexist. `TriageHandler` is one CoR stage; inside
it, a Strategy-based rule chain decides the triage verdict. CoR structures the
*pipeline*; Strategy structures the *decision* inside one stage.

---

## End-to-End Flow

```
Free-text broker email / PDF
        │
        ▼
[Task 01] LLM Extraction          raw dict payload
        │
        ▼
[Task 06] Pipeline ──────────────────────────────────────────────┐
        │                                                         │
        ├─▶ IdempotencyHandler   SHA-256 hash, cache-aside       │ Chain of
        ├─▶ ValidationHandler    required fields, Pydantic        │ Responsibility
        ├─▶ TriageHandler        appetite rules, Strategy pattern │
        ├─▶ DeduplicationHandler fuzzy match, inverted index      │
        └─▶ EnrichmentHandler    retry + circuit breaker         ─┘
                │
                ▼
        SubmissionContext
        status · enrichment_data · warnings · errors
```

The pipeline is the composition layer. Each task is independently testable and
replaceable. Swapping the enrichment API requires changing only `EnrichmentHandler`.
Adding a new processing stage requires writing one new `Handler` subclass and one
`set_next()` call — no existing handler is modified.

---

## Running the Tests

Each task has its own virtual environment:

```bash
cd task_01_llm_extraction && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v

cd task_02_idempotency && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v

# ... same pattern for tasks 03–06
```

All tasks follow the same convention: `task_XX/.venv/bin/pytest tests/ -v`.
