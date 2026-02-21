# Task 03 — Submission Triage & Validation

## The Problem

Insurance brokers submit applications (called "submissions") via API in JSON format.
Before any pricing or underwriting work can begin, each submission must pass through
a **triage** step that answers two questions:

1. Is this submission worth looking at at all? (appetite check)
2. If yes, does it need special handling before an underwriter touches it?

The complication is that the incoming data is **dirty**. Brokers integrate with us in
different ways, and the data they send is inconsistent:

- Revenue might arrive as `5000000`, `"5000000"`, `"$5,000,000"`, or `"5,000,000.00"`.
- State might be `"ny"`, `"NY"`, `"New York"`, or `"new york"`.
- Required fields might be missing entirely if the broker's system has a bug.

A single bad record in a batch **must not crash the entire pipeline**. The business
loses money when submissions are dropped silently, and loses trust when pipelines
crash noisily.

---

## Solutions Considered

### Option A — One Big Function with `if/elif`

```python
def triage(raw: dict) -> str:
    revenue = int(raw["revenue"].replace("$", "").replace(",", ""))
    if revenue > 500_000_000:
        return "declined"
    elif raw["industry"] == "Construction" and raw["state"] == "NY":
        return "manual_review"
    ...
```

**Why we rejected this:**
- The `if/elif` chain grows without bound as underwriting appetite changes.
- Every new rule requires editing the same function, creating merge conflicts and
  regression risk.
- There is no validation layer — a missing `revenue` key raises a `KeyError` and
  crashes the process.
- Testing requires reading the entire function to understand what cases exist.

---

### Option B — Pydantic for Validation + Strategy Pattern for Rules ✓

Split the problem into two distinct concerns:

1. **Validation** (Pydantic): Transform untrusted input into a trusted, typed object.
2. **Rule evaluation** (Strategy): Apply business logic to the clean object.

This is the approach implemented here.

---

## Patterns Used

### Pydantic v2 — Schema Validation at the Boundary

Pydantic is used to define `Submission` as a typed model. Before any business logic
runs, the raw dict is passed through `Submission.model_validate()`. If validation
fails, a `ValidationError` is raised with precise field-level error messages.

Two types of validators are used:

**`@field_validator(mode="before")`** — runs *before* Pydantic casts the value to
the declared type. This is where dirty-data cleaning happens:

```python
@field_validator("revenue", mode="before")
@classmethod
def parse_revenue(cls, v):
    if isinstance(v, str):
        cleaned = re.sub(r"[$,\s]", "", v)
        return Decimal(cleaned)
    return v
```

The `mode="before"` is critical. Without it, Pydantic would try to cast the string
`"$5,000,000"` directly to `Decimal` and raise a validation error before our cleaner
even runs.

**`@model_validator(mode="after")`** — runs *after* all fields are validated and
cast. Used for cross-field checks that need access to the complete model:

```python
@model_validator(mode="after")
def company_id_not_blank(self) -> "Submission":
    if not self.company_id.strip():
        raise ValueError("company_id must not be blank")
    return self
```

**Why `Decimal` and not `float` for revenue?**
`float` uses binary floating-point, which cannot represent most decimal fractions
exactly. `float("499999999.99") > 500_000_000` can produce surprising results.
`Decimal` uses decimal arithmetic and is the correct type for any financial value.

---

### Strategy Pattern — Rules as First-Class Objects

The Strategy pattern defines a family of algorithms (rules), encapsulates each one,
and makes them interchangeable. Here, each underwriting rule is its own class that
implements a single `evaluate()` method.

```
Rule (ABC)
├── RevenueOutOfAppetiteRule
├── SanctionedIndustryRule
└── ConstructionNewYorkRule
```

The `evaluate()` method returns a `RuleResult` if the rule fires, or `None` if it
does not. This is the **Null Object** variation of the pattern — returning `None`
instead of a sentinel value means the engine's loop needs no special-case handling:

```python
for rule in self._rules:
    result = rule.evaluate(submission)
    if result is not None:   # clean — no isinstance checks, no sentinel strings
        return TriagedSubmission(...)
```

**Why this scales:**
Adding a new rule for, say, "decline Florida flood risk" means writing one new class
and adding it to the engine's rule list. No existing code changes. This satisfies the
**Open/Closed Principle**: the system is open for extension but closed for
modification.

**Configurable thresholds via constructor injection:**
Rules accept their parameters in `__init__`, not as hardcoded constants:

```python
RevenueOutOfAppetiteRule(
    min_revenue=Decimal("100_000"),
    max_revenue=Decimal("1_000_000"),
)
```

This allows tests to use custom thresholds without subclassing, and makes rules
configurable from an external config file without changing code.

---

### Dependency Injection — Rules into the Engine

The engine receives its rule chain via the constructor:

```python
class SubmissionTriage:
    def __init__(self, rules: list[Rule] | None = None):
        self._rules = rules or self._DEFAULT_RULES
```

In production: `SubmissionTriage()` — uses the default chain.
In tests: `SubmissionTriage(rules=[RevenueOutOfAppetiteRule()])` — isolated to one rule.
In A/B experiments: pass a different chain from a feature flag.

---

### Generator for Batch Processing

`process_batch()` is a generator (uses `yield`, not `return`):

```python
def process_batch(self, raws: Iterable[dict]) -> Iterator[TriagedSubmission]:
    for raw in raws:
        result = self.process(raw)
        if result is not None:
            yield result
```

This means the caller gets results one at a time as they are produced. The entire
batch is never held in memory simultaneously. If the submission stream comes from
a database cursor or a message queue, this processes it in O(1) memory regardless
of batch size.

---

## Engineering Deep Dive

### Orthogonality & Separation of Concerns

The three files in this task represent three completely independent concerns. A
change in one must never require a change in the others.

| Layer | File | Knows about | Does NOT know about |
|---|---|---|---|
| Validation boundary | `models.py` | Data shapes, cleaning, type coercion | Rules, decisions, the engine |
| Business logic | `rules.py` | Clean `Submission` objects, appetite thresholds | HTTP, databases, how data arrived |
| Orchestration | `engine.py` | How to iterate the rule chain | What any specific rule does |

**Practical test of orthogonality — ask: "what file do I touch if...?"**

- Revenue format changes from a string to a nested `{"amount": 5000000, "currency": "USD"}` object?
  → Only `models.py` (the validator). The rules and engine are unchanged.
- A new rule is added: "decline submissions from Alaska during hurricane season"?
  → Add one new class to `rules.py`, add it to the engine's list. `models.py` untouched.
- The engine needs to run rules concurrently instead of sequentially?
  → Only `engine.py`. The rules and models don't care.

This is the **Single Responsibility Principle** applied at the module level. Each
file has exactly one reason to change.

---

### Library Trade-offs: Pydantic vs Alternatives

**Pydantic vs marshmallow**

`marshmallow` (2013) is the older, battle-tested alternative. Both validate and
deserialise data but differ in style:

| | Pydantic v2 | marshmallow |
|---|---|---|
| Schema definition | Python type hints — no separate schema class | Explicit `Schema` class with `fields.X` |
| IDE support | Full auto-complete from type hints | Limited — fields are runtime objects |
| Speed | Rust core (v2) — very fast | Pure Python |
| JSON Schema export | Built-in | Via plugin |
| API stability | v1 → v2 had breaking changes | Very stable API |

For a new project: Pydantic. For a codebase already on marshmallow: stay, the
migration cost is not worth it.

**Pydantic vs manual `data.get()` checks**

```python
# Manual
revenue_raw = data.get("revenue")
if revenue_raw is None:
    raise ValueError("revenue is required")
try:
    revenue = Decimal(str(revenue_raw).replace("$", "").replace(",", ""))
except:
    raise ValueError(f"Invalid revenue: {revenue_raw!r}")
```

This is 8 lines of code that Pydantic expresses in a 3-line validator. The real
cost of manual validation is not the lines written — it is the error messages that
are never written, the edge cases never tested, and the fields that silently default
to `None` when a broker sends a typo.

**Trade-off to acknowledge in an interview:**
Pydantic adds a compilation step at import time. Very large models (50+ fields) can
slow application startup. For a Lambda function that cold-starts frequently, this
matters. For a long-lived service, it doesn't.

---

### System Resiliency

**How failures are handled:**

| Failure scenario | What happens | Is data lost? | Is the service crashed? |
|---|---|---|---|
| Missing required field (e.g., no `company_name`) | `ValidationError` caught in `_parse()`, `None` returned | No — logged with field-level detail | No |
| Unparseable revenue string (e.g., `"not-a-number"`) | Same as above | No | No |
| Blank `company_id` | `model_validator` raises, caught in `_parse()` | No | No |
| A rule throws an unexpected exception | Propagates — this is a code bug, not bad data | Depends on caller | No — caller's batch loop should guard |

**Why validation failures return `None` instead of raising:**

The engine's contract is: *"give me a stream of raw dicts, I'll give you back a
stream of triage decisions."* A broker submitting bad data is not an exceptional
event in a system that processes thousands of submissions from dozens of integrations
— it is a normal operating condition. Raising would force every caller to wrap every
call in a try/except, which they will forget to do.

**What gets logged and why:**

```python
logger.error(
    "Validation failed for submission id=%r: %s",
    raw.get("company_id", "UNKNOWN"),
    exc.errors(include_url=False),
)
```

- `%r` (repr) for `company_id`: renders `None` as `None` and non-printable
  characters visibly, rather than silently corrupting the log line.
- `include_url=False`: Pydantic v2 adds a documentation URL to each error by
  default. In production logs this adds noise and leaks internal library details.
- `exc.errors()` returns a list of structured dicts — field name, error type, value
  — which a log aggregator (Datadog, CloudWatch) can parse and alert on.
- The rule engine logs at `INFO` when a rule fires, so an operator can trace exactly
  why a submission was declined, without reading code.

---

## File Structure

```
task_03_triage/
├── models.py       # Pydantic models — the validation boundary
├── rules.py        # Strategy pattern — one class per business rule
├── engine.py       # Orchestrator — wires validation + rule chain together
└── tests/
    └── test_triage.py
```

## Running the Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
```
