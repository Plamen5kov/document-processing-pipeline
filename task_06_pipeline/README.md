# Task 06 — Submission Processing Pipeline

## The Problem

Tasks 01–05 each solve one piece of the submission processing problem in isolation.
In production they run in sequence, and the order matters:

- There is no point enriching a submission that was already processed (idempotency).
- There is no point triaging a submission with missing required fields (validation).
- There is no point enriching a submission that the rules just declined (triage).
- A potential duplicate is worth flagging but not worth stopping for (soft warning).

The naive implementation — a single function calling each task in sequence — has two
problems. First, the short-circuit logic (when to stop) becomes entangled with the
business logic of each step. Second, adding a new processing stage requires editing
that single function and understanding its full context.

The system needs a way to compose sequential stages where each stage is independently
testable, each stage controls its own exit condition, and adding a new stage requires
writing one new class and one new `set_next()` call.

---

## Why Not Just Strategy Again?

Strategy (Task 01) solves a **selection** problem: given a set of alternatives, pick
the one that matches. The engine iterates the list; the rules don't know about each
other; they all answer the same question.

Chain of Responsibility solves a **pipeline** problem: given a sequence of stages,
pass the work through them in order, letting any stage stop the flow early. Each
handler does a *different* job. They are not alternatives — they are sequential.

The simplest way to tell them apart:

> **Strategy**: the engine decides which algorithm to run.
> **Chain of Responsibility**: each handler decides whether to continue.

A committee voting on one question is Strategy. A conveyor belt where each station
either stamps the part and passes it on, or rejects it and sends it back, is Chain
of Responsibility.

Both patterns coexist in this codebase. `TriageHandler` uses CoR to be one stage
in the pipeline, and internally delegates to a Strategy-based rule chain to decide
what that stage's answer is.

---

## Solutions Considered

### Option A — Sequential function calls

```python
def process(payload: dict) -> dict:
    if cache.get(hash(payload)):
        return cache[hash(payload)]
    submission = validate(payload)
    if submission is None:
        return {"status": "error"}
    triage_result = triage(submission)
    if triage_result.status == "declined":
        return {"status": "declined"}
    enriched = enrich(submission)
    ...
```

**Why we rejected this:**
- Every `if/return` that handles short-circuiting is mixed with the function call.
  The control flow and the business logic are inseparable.
- Adding a new stage (e.g., a rate-limit check) requires editing this function,
  understanding every exit condition that already exists, and choosing where to
  insert the new logic.
- Testing any individual stage requires the entire function to run up to that point.
  You cannot test enrichment without also running validation and triage first.

---

### Option B — Middleware Stack (decorator-based)

Frameworks like Django and FastAPI use a middleware stack where each middleware
wraps the next one:

```python
@middleware
def idempotency(request, next):
    if cached: return cached
    return next(request)
```

This is functionally equivalent to Chain of Responsibility but uses function
composition instead of object references. It is the right choice when integrating
with a framework that already provides the middleware abstraction.

**Why we chose explicit CoR instead:**
- We are not inside a framework — we own the pipeline from top to bottom.
- Object-based handlers are more inspectable, serialisable, and testable in
  isolation than wrapped function closures.
- The `SubmissionContext` carrier object (discussed below) cannot easily be
  threaded through a decorator stack without becoming an implicit global.

---

### Option C — Chain of Responsibility with a Context Carrier ✓

Each handler receives and returns a `SubmissionContext` dataclass. The handler
either modifies the context and calls `self._next(ctx)`, or returns `ctx` directly
to short-circuit. This is the approach implemented.

---

## Patterns Used

### Chain of Responsibility — Handler Base Class

```python
class Handler:
    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        return handler          # enables fluent chaining

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        return self._next(ctx)  # subclasses override this

    def _next(self, ctx: SubmissionContext) -> SubmissionContext:
        if self._next_handler:
            return self._next_handler.handle(ctx)
        return ctx              # end of chain
```

`set_next()` returns the handler it was given, enabling fluent construction:

```python
head.set_next(ValidationHandler()) \
    .set_next(TriageHandler(rules)) \
    .set_next(DeduplicationHandler(existing_ids)) \
    .set_next(EnrichmentHandler(risk_db))
```

This reads left-to-right in the same order the handlers execute at runtime.
Compare to the alternative (`a.next = b; b.next = c; c.next = d`) which
requires reading and cross-referencing four separate statements.

---

### SubmissionContext — the Carrier Object

```python
@dataclass
class SubmissionContext:
    payload: dict
    status: PipelineStatus = PipelineStatus.PENDING
    company_id: Optional[str] = None
    enrichment_data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    was_replay: bool = False
```

The context object accumulates information as it travels through the chain.
`ValidationHandler` sets `company_id`. `TriageHandler` reads it for logging.
`DeduplicationHandler` sets a warning. `EnrichmentHandler` reads the company_id
to call the API. Each handler sees the full history of what earlier stages found.

This is the key structural difference from passing arguments: a function call
chain like `enrich(triage(validate(payload)))` requires each function to return
every piece of data that later functions might need. The context object lets
each stage deposit its output independently.

---

### Hard Short-Circuit vs Soft Pass-Through

Not all handlers behave the same way on failure. This is a deliberate design
choice, not an accident:

**Hard short-circuit** (returns immediately, does not call `_next`):

```python
# IdempotencyHandler — cached result, no work needed
if key in self._cache:
    return cached  # stops here

# ValidationHandler — bad data, no further processing makes sense
if missing:
    ctx.status = PipelineStatus.ERROR
    return ctx     # stops here

# TriageHandler — declined, no point enriching a rejected submission
if ctx.status == PipelineStatus.DECLINED:
    return ctx     # stops here
```

**Soft pass-through** (annotates the context and calls `_next`):

```python
# DeduplicationHandler — duplicate flag, but enrichment still runs
if ctx.company_id in self._existing:
    ctx.status = PipelineStatus.DUPLICATE
    ctx.add_warning(...)
return self._next(ctx)   # continues — the underwriter decides what to do
```

The rule: short-circuit when further processing is *pointless* (error, decline)
or *wasteful* (idempotency hit). Pass through when further processing is still
*useful* despite the flag (a potential duplicate still needs enrichment data for
the underwriter who will review it).

---

### SubmissionPipeline as a Builder

`SubmissionPipeline` separates chain construction from chain execution:

```python
class SubmissionPipeline:
    def __init__(self, cache, rules, existing_ids, risk_db):
        head = IdempotencyHandler(cache)
        head.set_next(ValidationHandler()) \
            .set_next(TriageHandler(rules)) \
            ...
        self._head = head

    def run(self, payload: dict) -> SubmissionContext:
        ctx = SubmissionContext(payload=payload)
        result = self._head.handle(ctx)
        # Store AFTER the full chain — never cache a partial result
        if result.idempotency_key and not result.was_replay:
            self._cache[result.idempotency_key] = result
        return result
```

The "store AFTER the full chain completes" decision is important: if
`EnrichmentHandler` fails and we stored the result before it ran, the next
request would get a cached context with empty `enrichment_data`. Storing
at the end ensures the cache only holds complete results.

---

## Engineering Deep Dive

### Orthogonality & Separation of Concerns

Every handler owns exactly one concern and has exactly one reason to change:

| Handler | Changes when... | Does NOT change when... |
|---|---|---|
| `IdempotencyHandler` | The cache backend changes (e.g., Redis) | Validation rules change |
| `ValidationHandler` | Required fields change | Triage rules change |
| `TriageHandler` | Appetite rules change | Enrichment API changes |
| `DeduplicationHandler` | Duplicate detection logic changes | Triage rules change |
| `EnrichmentHandler` | External API changes | Any other stage changes |

The `SubmissionContext` is the only shared surface. If a handler needs to expose
new data to later stages, add a field to `SubmissionContext` — no other handler
needs to change unless it wants to read that field.

**Adding a new stage requires:**
1. Write one new `Handler` subclass.
2. Call `set_next()` in the right position in `SubmissionPipeline.__init__`.
3. That is all. No existing handler is modified.

The last test — `test_adding_new_handler_without_touching_existing_ones` —
verifies this claim. An `AuditLogHandler` is inserted mid-chain with zero
changes to any existing class. This is the same Open/Closed Principle
demonstrated in Task 01, applied at the pipeline level.

---

### Library Trade-offs

**Chain of Responsibility vs Middleware Stack**

Both model the same concept. The choice is about context:

| | CoR (object-based) | Middleware (function-based) |
|---|---|---|
| Handler state | Natural — instance attributes | Requires closure variables |
| Testability | Instantiate handler directly | Must wrap in the full stack |
| Framework integration | Manual | Built-in for Django, FastAPI, Express |
| Readability | Explicit `set_next` wiring | Implicit stack order (decorator order) |

If this pipeline were a FastAPI application, the idempotency and validation
stages would be framework middleware and the business stages would be route
dependencies. The conceptual structure is identical; only the syntax differs.

**`SubmissionContext` as a dataclass vs a dict**

The context could be a plain `dict`. Using a `@dataclass` instead provides:
- Type hints on every field — IDE auto-complete and static analysis work
- Default values declared in one place, not scattered across handlers
- Named fields that document intent (`was_replay: bool` vs `ctx["was_replay"]`)
- `add_warning()` and `add_error()` methods that centralise logging

The trade-off: `@dataclass` fields must be declared in advance. If a handler
needs to store something genuinely ad-hoc, it must add a field or use the
`enrichment_data: dict` escape hatch.

**Why `SubmissionContext` is mutable**

Immutable context objects (returning a new object from each handler) are safer
under concurrency — no shared mutable state. The trade-off is memory allocation
and the need to copy every field on each handler call.

For this pipeline, each `SubmissionContext` is owned by exactly one request
for its entire lifetime — no sharing between threads — so mutability is safe
and cheaper.

---

### System Resiliency

**What happens when a handler raises an unexpected exception?**

Currently: the exception propagates to `SubmissionPipeline.run()` and then to
the caller. The context is not cached (since `run()` only caches on clean return).
The message queue redelivers the submission — at-least-once semantics mean it
will be retried.

In production, you would wrap `self._head.handle(ctx)` in a try/except that
catches unexpected errors, marks the context with `status=ERROR`, and returns it
rather than raising. This gives the caller a structured error response instead of
an exception.

**Graduated failure — warnings vs errors:**

The pipeline distinguishes two levels of problems:

- `ctx.errors`: fatal — the chain stopped because of this. The caller must not
  proceed. Examples: missing required fields, unrecoverable system error.
- `ctx.warnings`: advisory — the chain completed but something needs attention.
  Examples: potential duplicate, enrichment data missing, manual review required.

Callers check `ctx.errors` to decide whether the result is usable.
Callers check `ctx.warnings` to decide what to surface to the underwriter.
This graduated model prevents callers from treating every imperfect result as a
crash — which is the wrong behaviour in a system that processes thousands of
submissions per day.

**Idempotency across the full pipeline:**

The `IdempotencyHandler` stores the complete `SubmissionContext` after the chain
finishes. A replayed request receives the exact same context object — including
its `enrichment_data`, `warnings`, and final `status`. The caller cannot tell
the difference between a first run and a replay by looking at the result alone;
only `was_replay: bool` distinguishes them. This is the correct behaviour: the
idempotency contract is that the outcome is identical, not just similar.

---

## How the Tasks Connect

```
task_04  Free text → ExtractedSubmission (LLM + Pydantic schema enforcement)
                              ↓
                         raw payload dict
                              ↓
task_06  IdempotencyHandler   (task_05 pattern — SHA-256 hash, cache-aside)
              ↓
         ValidationHandler   (task_01 models — Pydantic field validation)
              ↓
         TriageHandler        (task_01 rules — Strategy pattern inside CoR)
              ↓
         DeduplicationHandler (task_03 pattern — fuzzy match pre-filter)
              ↓
         EnrichmentHandler    (task_02 pattern — retry, ThreadPoolExecutor)
              ↓
         SubmissionContext    (final result — status, enrichment, warnings)
```

Each task is a standalone module. This pipeline is the composition layer.
Replacing any one task (e.g., switching the enrichment API) requires changing
only the corresponding handler's `__init__` — not the pipeline structure.

---

## Pipeline Diagram

The full end-to-end flow, including all short-circuit exits and the circuit
breaker state machine embedded inside `EnrichmentHandler`.

See [pipeline_diagram.mmd](pipeline_diagram.mmd).

To render: paste the file contents into [mermaid.live](https://mermaid.live), or open the README preview with the `bierner.markdown-mermaid` VSCode extension.

**Hard short-circuits** (chain stops, no further handlers run):

| Handler | Condition |
|---|---|
| `IdempotencyHandler` | Payload hash already in cache |
| `ValidationHandler` | Required field is missing |
| `TriageHandler` | Underwriting rule returns DECLINED |

**Soft pass-throughs** (chain continues with a warning on the context):

| Handler | Condition |
|---|---|
| `TriageHandler` | Rule returns MANUAL_REVIEW |
| `DeduplicationHandler` | company_id matches a known submission |
| `EnrichmentHandler` | API fails (after retries, or circuit OPEN) |

**Retry vs Circuit Breaker** — two complementary resilience mechanisms at different scopes:

- **tenacity retry** — handles one request hiccuping (millisecond scale).
  Retries the same call up to 3 times with exponential back-off.
- **Circuit breaker** — handles the API being down (second/minute scale).
  After `failure_threshold` consecutive failures, stops all calls for
  `recovery_timeout` seconds so the API can recover without being hammered.

---

## File Structure

```
task_06_pipeline/
├── pipeline.py          # Handler base, all concrete handlers, SubmissionPipeline
├── circuit_breaker.py   # Three-state circuit breaker (CLOSED / OPEN / HALF_OPEN)
├── pipeline_diagram.mmd # Mermaid source for the pipeline flowchart
└── tests/
    └── test_pipeline.py
```

## Running the Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
```
