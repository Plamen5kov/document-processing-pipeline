# Task 04 — LLM Data Extraction with Schema Enforcement

## The Problem

Brokers don't always use structured forms. A common real-world scenario is a
broker sending a free-text email like:

> "Hi, attaching a submission for Acme Corp out of Dallas, Texas. They're a
> manufacturing business doing around $4.5M in annual revenue."

The system needs to turn that prose into a structured `Submission` object that can
be triaged by the rules engine from Task 01. To do that, it must call an LLM.

The challenge: **LLM outputs are unstructured and probabilistic.** The same prompt
can produce:

- Clean JSON on the first try
- JSON wrapped in a markdown code block (```json ... ```)
- A sentence explaining why it cannot extract the data
- Valid JSON but with hallucinated fields (`"revenue": "approximately four million"`)
- Valid JSON with a missing field because the source text didn't mention it

None of these can be handled by a simple `json.loads()`. The system needs a full
extraction pipeline that validates, retries, and classifies partial results.

---

## The "Extraction + Judging Agent" Pattern

Kalepa's engineering blog describes coupling two agents:

- **Extraction Agent**: calls the LLM and retrieves raw output
- **Judging Agent**: validates whether the output meets the required schema

In this implementation, **Pydantic IS the judging agent**. It enforces the output
contract with the same mechanism used at the broker input boundary (Task 01) —
the difference is that here all fields are `Optional` because a partial extraction
is more useful than a failed one.

```
Free text
   ↓
SubmissionExtractor._call_llm()     ← Extraction agent
   ↓
SubmissionExtractor._parse_and_validate()
   ↓
ExtractedSubmission (Pydantic model) ← Judging agent
   ↓
ExtractionResult (SUCCESS / PARTIAL / FAILED)
```

---

## Solutions Considered

### Option A — Prompt Engineering Only (hope it returns valid JSON)

```python
response = llm.complete("Extract fields, return JSON")
data = json.loads(response)
```

**Why we rejected this:**
- `json.loads` crashes on markdown-wrapped output (```` ```json ... ``` ````).
- No schema validation — a hallucinated `"revenue": "approximately four million"`
  passes silently.
- No retry — one bad LLM response loses the submission.
- No distinction between "field not found" (PARTIAL) and "cannot parse" (FAILED).

---

### Option B — `instructor` Library

`instructor` is a popular library that patches any LLM SDK to enforce structured
output natively. With OpenAI:

```python
import instructor
client = instructor.from_openai(openai.OpenAI())
result = client.chat.completions.create(
    model="gpt-4o",
    response_model=ExtractedSubmission,
    messages=[{"role": "user", "content": prompt}],
)
```

`instructor` internally uses function calling / JSON mode and retries on validation
failure. It is the right choice when you control the full LLM stack and want minimal
boilerplate.

**Why we did not use it here:**
- Hides all the retry, parsing, and validation logic — the point of this task is
  to understand those mechanics.
- Couples the code to a specific SDK (OpenAI, Anthropic). The `LLMClient` protocol
  here is provider-agnostic.
- Harder to test without mocking at the `instructor` layer specifically.

---

### Option C — `LLMClient` Protocol + Pydantic + Retry Loop ✓

Manual pipeline with full visibility into each step. This is the approach
implemented. Understanding it makes Option B easier to reason about.

---

## Patterns Used

### The `LLMClient` Protocol (Dependency Inversion)

```python
class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...
```

The extractor depends on this abstract interface, not on any specific LLM SDK.
In production, swap in an `AnthropicClient` or `OpenAIClient`. In tests, use
`MockLLMClient` with pre-programmed responses. No API key required to run the
test suite.

This is the same Dependency Inversion principle from Task 01 (rules injected into
the engine) and Task 02 (HTTP client injected into the enricher) — applied here
to the LLM layer.

---

### `Optional` Fields on LLM Output Models

Task 01's `Submission` model has all required fields — missing a field is a bug.

Task 04's `ExtractedSubmission` has all `Optional` fields — missing a field is
expected. The source text may not contain every piece of information. A partial
extraction that has `company_name` and `revenue` is enough to run triage; we do
not need to discard it because `zip_code` was not mentioned.

```python
REQUIRED_FOR_TRIAGE = frozenset({"company_name", "revenue", "industry", "state"})
```

`_missing_fields()` computes which required fields were not extracted, and the
result is classified as `SUCCESS`, `PARTIAL`, or `FAILED` accordingly.

---

### Revenue Coercion for LLM Output

LLMs produce revenue values in many formats. The validator handles them all by
doing numeric arithmetic, not string manipulation:

```python
multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
for suffix, factor in multipliers.items():
    if cleaned.lower().endswith(suffix):
        return Decimal(cleaned[:-1]) * factor  # "4.5M" → Decimal("4.5") × 1_000_000
```

The critical detail: `Decimal("4.5") * 1_000_000 = Decimal("4500000")`, not
`"4.5" + "000000" = "4.5000000"`. Always multiply numerically for currency.

Unrecognisable values (`"approximately four million"`, `None`) return `None`
rather than raising — the validator treats them as "not found."

---

### Retry with Prompt Strengthening

On the first attempt, the prompt is clear but concise. If the LLM returns
unparseable output, the second attempt appends a stronger reminder:

```
REMINDER: Your previous response could not be parsed. Return ONLY a raw
JSON object. Example: {"company_name": "Acme", "revenue": 5000000, ...}
```

This mirrors how a human would respond to a colleague who misunderstood the
first instruction: you don't repeat the exact same words, you clarify.

Unlike Task 02's tenacity retries (infrastructure failures), these retries are
about **prompt iteration** — each attempt has a better chance of success because
the prompt carries forward what went wrong.

---

## Engineering Deep Dive

### Orthogonality & Separation of Concerns

| Layer | File | Responsibility | Does NOT know about |
|---|---|---|---|
| Data contract | `models.py` | Field types, coercion, partial/required | LLM, prompts, retry |
| Extraction logic | `extractor.py` | Prompt construction, retry, parsing | What fields mean downstream |

A change to the revenue coercion logic (e.g., handle `"€4.5M"`) → only `models.py`.
A change to the retry strategy (e.g., 3 retries instead of 2) → only `extractor.py`.
A change to the LLM provider → swap the `LLMClient` implementation.

The downstream triage engine (Task 01) receives an `ExtractedSubmission` and can
apply its rules without knowing the data came from an LLM. The output contract is
the same `Submission`-compatible shape.

---

### Library Trade-offs

**`instructor` vs manual pipeline**

| | `instructor` | Manual (this task) |
|---|---|---|
| Boilerplate | Minimal | More verbose |
| Transparency | Hidden retry/validation | Explicit, auditable |
| Provider coupling | Per-SDK patch | Provider-agnostic protocol |
| Testability | Requires mocking at SDK level | Simple `MockLLMClient` |
| Production speed | Faster to ship | Better for understanding |

For a production service already on a specific LLM provider: use `instructor`.
For an interview or a multi-provider system: the manual pattern shows the
mechanics that `instructor` abstracts away.

**Pydantic validators `mode="before"` — why not raise on bad input?**

In Task 01, validators raise on invalid revenue (it is a bug in the broker's system).
Here, validators return `None` on invalid revenue (it is an expected LLM limitation).
The same mechanism produces opposite behaviour because the *contract* is different.
This distinction — "invalid input" vs "uncertain output" — is the core conceptual
difference between the two tasks.

---

### System Resiliency

| Failure scenario | What happens | Is data lost? |
|---|---|---|
| LLM returns prose, not JSON | `JSONDecodeError` caught, retry with stronger prompt | No |
| LLM returns wrong JSON schema | `ValidationError` caught, retry | No |
| All retries exhausted | `ExtractionResult(status=FAILED)` returned | No — escalate to human queue |
| LLM returns partial extraction | `ExtractionResult(status=PARTIAL)` returned | No — caller decides |
| LLM client raises an exception | Caught in retry loop, returned as FAILED | No |

**The extractor never raises.** The contract is: give me text, I give you an
`ExtractionResult`. What the caller does with `status=FAILED` is a business
decision — send to human review, alert on high failure rates, or route to a
secondary extraction strategy.

**Observability:**
- `INFO` log on success/partial with which fields were missing
- `WARNING` log on each failed attempt with the error type
- `raw_llm_output` preserved in the result for debugging
- `error` field on FAILED contains the last exception message

---

## File Structure

```
task_04_llm_extraction/
├── models.py       # ExtractedSubmission (Optional fields) + ExtractionResult
├── extractor.py    # LLMClient protocol + SubmissionExtractor + retry
└── tests/
    └── test_extraction.py
```

## Running the Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
```
