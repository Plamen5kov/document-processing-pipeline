# Task 05 — Data Enrichment

## The Problem

After a submission passes triage, it contains only what the broker sent us. That is
not enough to price the risk accurately. We need to **enrich** each submission with
external signals — loss history, credit scores, industry risk indices — from a
third-party Risk API.

Two hard constraints make this non-trivial:

**The API is flaky.** It has a ~10% failure rate and occasionally takes 5+ seconds
to respond. A single failure must not lose the submission — it must be retried.
After retries are exhausted the submission is marked `enrichment_failed` and moved
on, not dropped.

**Volume.** At peak we enrich hundreds of submissions per minute. Calling the API
sequentially — one at a time, waiting for each to complete before starting the next
— is far too slow.

---

## Solutions Considered

### Option A — Sequential Loop with `time.sleep` Retry

```python
for company_id in company_ids:
    for attempt in range(3):
        try:
            data = requests.get(url).json()
            break
        except Exception:
            time.sleep(2 ** attempt)
```

**Why we rejected this:**
- `time.sleep` in a retry loop blocks the entire thread. No other submissions are
  processed while we wait. At 3 retries × 4s backoff, one bad API call stalls
  everything for 12 seconds.
- The retry logic is entangled with the business logic. Testing it requires
  real time to pass, or careful `unittest.mock.patch` of `time.sleep`.
- No distinction between retriable errors (5xx, timeout) and non-retriable ones
  (4xx). Retrying a 404 wastes time and API quota.

---

### Option B — `asyncio` + `aiohttp`

Asyncio is the correct answer when building a greenfield I/O system from scratch.
A single event loop can have hundreds of in-flight HTTP requests with no threads.

**Why we did not use it here:**
- The standard library's `requests` is synchronous. Mixing `requests` with `asyncio`
  requires either `asyncio.to_thread()` (which still uses threads under the hood) or
  rewriting all HTTP calls with `aiohttp`. That is a larger dependency and API
  surface change.
- In a real codebase that already uses `requests`, `ThreadPoolExecutor` is the
  pragmatic migration path to concurrency. Async is the right long-term target but
  not always the right first step.

---

### Option C — `ThreadPoolExecutor` + `tenacity` ✓

Use threads for concurrency (the right tool when the bottleneck is network I/O) and
`tenacity` for declarative, testable retry logic. This is the approach implemented.

---

## Patterns Used

### Why Threads for Network I/O (and not CPU work)

Python has the **GIL** (Global Interpreter Lock): only one thread runs Python
bytecode at a time. This makes threads useless for CPU-bound work — two threads
computing Fibonacci numbers will not be faster than one.

However, the GIL is **released** whenever a thread performs I/O (reading from a
socket, writing to a file, waiting for a DNS response). This means that while Thread
A is blocked waiting for the Risk API to respond, Thread B can run Python code and
make its own API call. Real concurrency happens at the network level.

```
Sequential (10 calls × 2s each):     ████░░████░░████░░  ≈ 20s
ThreadPoolExecutor(max_workers=5):    ████████████░░░░░░  ≈  4s
asyncio + aiohttp:                    ████░░░░░░░░░░░░░░  ≈  2s
```

Rule of thumb:
- I/O-bound + existing sync library (`requests`) → `ThreadPoolExecutor`
- I/O-bound + greenfield → `asyncio` + `aiohttp`
- CPU-bound (ML inference, parsing) → `ProcessPoolExecutor` (bypasses GIL)

---

### `ThreadPoolExecutor` + `as_completed()`

```python
with ThreadPoolExecutor(max_workers=5) as pool:
    future_to_id = {pool.submit(enrich_submission, cid): cid for cid in ids}
    for future in as_completed(future_to_id):
        result = future.result()
```

`pool.submit()` schedules the function on a free thread and immediately returns a
`Future` — a handle to the eventual result. The main thread does not block.

`as_completed()` yields each `Future` as soon as its thread finishes. This is
important: if we used `[f.result() for f in futures]` instead, we would process
results in submission order, waiting for the slowest call in each position even if
later calls have already finished.

The `with` block guarantees threads are joined (cleaned up) even if an exception
escapes — proper resource management.

---

### Custom Exception Hierarchy

```
RiskApiError (base)
├── RiskApiServerError   ← 5xx, timeouts — transient, safe to retry
└── RiskApiClientError   ← 4xx — our bug, retrying won't fix it
```

This hierarchy lets `tenacity` retry only the right exceptions:

```python
retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, RiskApiServerError))
```

A 404 (`RiskApiClientError`) is not in this list, so tenacity lets it propagate
immediately. Without this distinction, we would waste quota retrying our own bugs.

---

### `tenacity` — Declarative Retry

`tenacity` separates *what to retry* from *the code being retried*. The entire
retry policy is expressed as decorator arguments, making it readable at a glance:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((...)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_risk_data(company_id: str) -> dict:
    ...
```

`reraise=True` means that after all attempts are exhausted, the *original*
exception is raised — not `tenacity.RetryError`. This keeps the call stack clean
and lets `enrich_submission()` catch specific exception types.

**Exponential backoff** (`wait_exponential`): waits 1s after the first failure,
2s after the second, 4s after the third, capped at 10s. This prevents thundering
herd — if the API is struggling, all callers slamming it simultaneously at the same
interval makes it worse. Backoff gives it breathing room.

---

### `enrich_submission()` Never Raises

The public function catches all exceptions and returns an `EnrichmentResult`:

```python
def enrich_submission(company_id: str) -> EnrichmentResult:
    try:
        ...
        return EnrichmentResult(company_id=..., success=True, data=...)
    except (RiskApiServerError, requests.Timeout, ...) as exc:
        return EnrichmentResult(company_id=..., success=False, error=str(exc))
```

A failed enrichment is a **business outcome**, not an unhandled exception. The
batch enricher records failures in a report rather than crashing. In production
these `enrichment_failed` submissions would be sent to a dead-letter queue for
manual review, not silently dropped.

---

### camelCase → snake_case Mapping

The Risk API returns JSON with camelCase keys (`riskScore`, `claimsHistory`). Our
internal system uses snake_case (`risk_score`, `claims_history`). The mapping is
centralised in `mapper.py` and applied once at the API boundary so no downstream
code ever sees camelCase.

The function is recursive so it handles arbitrarily nested structures:

```python
def normalise_keys(data: dict | list) -> dict | list:
    if isinstance(data, dict):
        return {_camel_to_snake(k): normalise_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalise_keys(item) for item in data]
    return data
```

---

### Testing with `responses` and `unittest.mock`

**`responses` library** intercepts `requests.get()` at the HTTP layer and returns
registered mock payloads. No real network calls are made. It is more readable than
patching `requests.get` directly because you declare what URL returns what:

```python
@responses_lib.activate
def test_success(self):
    responses_lib.add(responses_lib.GET, url, json={"riskScore": 72}, status=200)
    result = _fetch_risk_data.__wrapped__(company_id)  # __wrapped__ bypasses tenacity
```

**`__wrapped__`** accesses the original function before tenacity wraps it. This
lets us test a single HTTP call in isolation without triggering retry logic.

**`side_effect` list** simulates a sequence of responses (fail, fail, succeed):

```python
mock_get.side_effect = [server_error, server_error, good_response]
```

Combined with patching `tenacity.nap.time.sleep`, the retry test runs in
milliseconds, not seconds.

---

## Engineering Deep Dive

### Orthogonality & Separation of Concerns

Each file in this task solves exactly one problem. None of them know what the
others are doing at the implementation level.

| Layer | File | Responsibility | Knows about | Does NOT know about |
|---|---|---|---|---|
| Data mapping | `mapper.py` | Key name conversion | String transformation | HTTP, retries, threads |
| Transport | `client.py` | One API call with retry | `requests`, tenacity, exception types | Threads, batch size, downstream consumers |
| Concurrency | `enricher.py` | Parallel execution | `ThreadPoolExecutor`, futures | HTTP details, retry policy, key naming |

**Practical test — "what changes when...?"**

- The Risk API switches from REST to gRPC?
  → Replace the `requests.get()` call inside `client.py`. `mapper.py` and
  `enricher.py` are untouched. Tests for the mapper and enricher still pass without
  modification.
- The key mapping rule changes (e.g., `riskScore` → `risk_rating`)?
  → Only `mapper.py`. No other file references camelCase keys.
- We want to switch from threads to `asyncio`?
  → Replace `enricher.py` with an async equivalent. The `client.py` and `mapper.py`
  contracts don't change.

This is sometimes called **Hexagonal Architecture** (Ports and Adapters): the core
business logic (`enrich_submission`, `normalise_keys`) is insulated from the
transport mechanism. The transport is a detail — swappable without touching the core.

A mid-level implementation would put the thread pool, the retry, and the key mapping
all in one function. It would work until the API changes, the concurrency model
changes, or the key naming convention changes — at which point everything breaks at
once.

---

### Library Trade-offs

**`tenacity` vs the alternatives**

| Option | Retry config location | Distinguishes retriable errors | Testable without sleep? |
|---|---|---|---|
| Manual `for attempt in range(3)` | Inside the function body | Only if you add `if/elif` | Only by patching `time.sleep` |
| `backoff` library | Decorator | No built-in support | Only by patching |
| `tenacity` | Decorator, declarative | `retry_if_exception_type` | Patch `tenacity.nap.time.sleep` |

The key advantage of `tenacity` over `backoff` is `retry_if_exception_type` — the
ability to express "retry on 5xx but not on 4xx" without writing any conditional
logic inside the function. The retry policy is a cross-cutting concern; it belongs
in the decorator, not in the body.

**Trade-off to acknowledge:** `tenacity`'s decorator wraps the function, which
means `_fetch_risk_data` is no longer the original function at import time — it is
a `tenacity`-wrapped object. This is why tests use `_fetch_risk_data.__wrapped__`
to access the raw function for unit tests. It is a minor inconvenience that the
alternative (passing the function as a parameter) would avoid, but the readability
gain of the decorator form is worth it.

**`ThreadPoolExecutor` vs `asyncio`**

This is the most common "trade-off" question in Python backend interviews. The
honest answer has two parts:

*When threads are correct:* When the I/O library is synchronous (`requests`,
`psycopg2`, `boto3`) and you do not want to rewrite it. Threads let you keep the
existing library and get concurrency for free, because the GIL is released during
I/O. The trade-off is thread overhead (~8 MB stack per thread) and the need to
think about shared mutable state.

*When asyncio is correct:* Greenfield I/O-heavy services where you control the full
stack. One event loop can manage thousands of concurrent connections with minimal
memory. The trade-off is that it requires async-native libraries throughout —
mixing sync code into an async event loop blocks it.

The rule: **threads for integrating with existing sync libraries; asyncio for new
systems built from scratch.** Using `asyncio.to_thread()` to run `requests` inside
an async program is just threads with extra steps.

**`responses` library vs `unittest.mock.patch`**

Both are valid. The difference is the abstraction level:

- `mock.patch("requests.get")`: patches Python's object model. Can simulate any
  behaviour. But the test is coupled to the implementation detail that the code
  uses `requests.get` specifically — switch to `httpx` and the patch breaks.
- `responses`: intercepts at the HTTP level (after `requests` but before the
  socket). Tests are coupled to URLs and HTTP semantics, which are more stable
  than library internals. Switching from `requests` to `httpx` would require a
  different interceptor library, not a rewrite of test assertions.

For unit-testing a single function in isolation, `mock.patch` is simpler.
For integration-style tests that verify URL construction and response parsing end
to end, `responses` is more appropriate. This codebase uses both for different
purposes.

---

### System Resiliency

This module has two distinct resilience layers, each with a clear purpose.

**Layer 1 — `client.py`: retry before giving up**

`tenacity` retries transient failures (5xx, timeouts, connection errors) up to 3
times with exponential backoff. The assumption: brief API instability is common and
the right response is to wait and try again. After 3 attempts, the original
exception propagates to the caller.

**Layer 2 — `enrich_submission()`: catch what retry didn't fix**

After all retries are exhausted, `enrich_submission()` catches the exception and
returns a typed failure object:

```python
return EnrichmentResult(company_id=company_id, success=False, error=str(exc))
```

This means **the submission is never lost** — it is marked with `success=False` and
a human-readable error message. In production, these would be written to a
dead-letter queue (SQS, Kafka) for investigation rather than silently dropped.

**Layer 3 — `enricher.py`: isolate per-submission failures**

The thread pool wraps each `future.result()` call in a try/except:

```python
try:
    result = future.result()
except Exception as exc:
    result = EnrichmentResult(company_id=company_id, success=False, error=str(exc))
```

`enrich_submission()` is designed not to raise, so this guard should never fire. It
exists because in a multi-threaded system, defensive programming around `future.result()`
is the correct pattern — unexpected exceptions in threads can otherwise be silently
swallowed.

**What the `BatchEnrichmentReport` provides:**

```python
report = enrich_batch(company_ids)
print(f"Success rate: {report.success_rate:.0%}")  # "Success rate: 94%"
```

The caller gets a structured view of the entire batch outcome, not just whether it
"completed". A 6% failure rate might trigger an alert. A 94% failure rate would
trigger an incident. This is observable failure — the system does not fail silently
or crash; it reports.

---

## File Structure

```
task_05_enrichment/
├── mapper.py       # camelCase → snake_case key normalisation
├── client.py       # HTTP client: retry decorator + exception hierarchy
├── enricher.py     # Batch enrichment: ThreadPoolExecutor + as_completed
└── tests/
    └── test_enrichment.py
```

## Running the Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
```
