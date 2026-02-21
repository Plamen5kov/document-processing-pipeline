# Task 04 — Conflict Detection (Deduplication)

## The Problem

Multiple brokers often submit the same company. If "Goldman Sachs Group LLC" is
already in our system and a new broker submits "Goldman Sachs", we must detect this
before an underwriter opens a second file and duplicates work — or worse, prices
the same risk twice.

The challenge is that names are never exact. Real-world variations include:

| In database | Incoming submission |
|---|---|
| `Kalepa Inc.` | `Kalepa, Inc` |
| `International Business Machines Corp.` | `IBM` |
| `Kalepa Inc.` | `KALEPA INC` |

A simple string equality check catches none of these. We need **fuzzy matching**.

The second challenge is **scale**. The database might have 10 000 existing records.
Running a fuzzy comparison between the incoming submission and every single record is
O(n) per submission. At 500 new submissions per day that is 5 million fuzzy string
comparisons — expensive, and unnecessary.

---

## Solutions Considered

### Option A — Exact String Match

```python
if new_name.lower() == existing_name.lower():
    return "duplicate"
```

**Why we rejected this:**
Catches only identical strings after lowercasing. `"Kalepa Inc."` vs `"Kalepa, Inc"`
would not match. This misses the vast majority of real duplicates.

---

### Option B — Levenshtein Distance on All Records

```python
for record in all_10_000_records:
    score = levenshtein_distance(new_name, record.name)
    if score < threshold:
        return "duplicate"
```

**Why we rejected this:**
- O(n) fuzzy comparisons per submission. At n = 10 000 and 500 submissions/day this
  is 5 million comparisons per day, and Levenshtein is not cheap.
- Levenshtein also has a known weakness for company names: it is character-level and
  sensitive to word order. `"IBM Corp"` vs `"Corp IBM"` scores badly despite being
  the same entity.

---

### Option C — Inverted Index Pre-filter + `token_sort_ratio` ✓

Break the problem into two phases:

**Phase 1 (Pre-filter, O(1) per lookup):** Use an inverted index keyed by the first
5 digits of the ZIP code. Most real duplicates come from the same geographic area —
the same physical company has the same address. This collapses the 10 000 records
into ~50–200 per ZIP bucket.

**Phase 2 (Fuzzy match, O(k) where k << n):** Run `rapidfuzz` only against the
pre-filtered bucket. With k ≈ 100, each submission requires ~100 comparisons instead
of 10 000. A 100× speedup from one O(n) index build at startup.

This two-phase approach is the same concept behind:
- Database indexes (exact key → row pointer, avoiding full table scans)
- Elasticsearch shards (route by a field value to narrow the search space)
- Locality-Sensitive Hashing (hash similar items into the same bucket)

---

## Patterns Used

### Inverted Index (Pre-filter)

An inverted index maps a **derived property** (ZIP prefix) to the set of records
that have that property. It is built once at `__init__` time in O(n), then queried
in O(1) per lookup.

```python
# Build — O(n)
self._index: dict[str, list[SubmissionRecord]] = {}
for record in existing:
    bucket_key = record.zip_code[:5]
    self._index.setdefault(bucket_key, []).append(record)

# Query — O(1)
bucket = self._index.get(candidate.zip_code[:5], [])
```

`setdefault(key, [])` is the idiomatic Python way to build a "list of items per
key" dict. It returns the existing list if the key exists, or inserts and returns a
new empty list if it does not — in one operation, no `if key not in d:` needed.

**Trade-off:** This uses more memory (the index sits in RAM). In production, this
index would live in Redis or a database, not in-process. The algorithmic idea is
the same.

---

### String Normalisation Pipeline

Before any fuzzy comparison, both strings are normalised through the same pipeline:

```
"International Business Machines Corp."
  → lowercase:       "international business machines corp."
  → strip punctuation: "international business machines corp "
  → drop suffixes + single chars: "business machines"
```

```python
def normalise(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    tokens = [
        tok for tok in name.split()
        if tok not in _ENTITY_SUFFIXES and len(tok) > 1
    ]
    return " ".join(tokens)
```

Three decisions worth explaining:

**`frozenset` for suffix lookup:** Checking `tok in _ENTITY_SUFFIXES` is O(1)
with a set (hash lookup), vs O(n) with a list (linear scan). With a list of 20
suffixes and 1 000 tokens per second, this does not matter. But using the right
data structure by default is a habit worth forming.

**Dropping single-character tokens:** After punctuation is replaced with spaces,
`"S.A."` becomes `"s a"` — two single-character tokens that carry no signal.
Keeping them would cause `"Café S.A."` and `"Café Ltd"` to score lower than they
should. Single-char tokens are dropped.

**Dropping entity suffixes:** `"Inc"`, `"LLC"`, `"Corp"` describe legal structure,
not identity. Two companies named `"Kalepa Inc"` and `"Kalepa LLC"` are very likely
the same company — the suffix should not prevent a match.

---

### `rapidfuzz` — `token_sort_ratio` Scorer

`rapidfuzz` is a fast implementation of fuzzy string matching algorithms (written
in C++ with Python bindings, ~10–100× faster than `fuzzywuzzy` / `python-Levenshtein`).

We use `token_sort_ratio` rather than the plain `ratio()`:

- `ratio()`: character-level comparison. `"IBM Corp"` vs `"Corp IBM"` → ~57%.
- `token_sort_ratio()`: sorts tokens alphabetically before comparing.
  `"IBM Corp"` → `"corp ibm"`, `"Corp IBM"` → `"corp ibm"` → 100%.

For company names, where word order varies ("Goldman Sachs Group" vs "Group Goldman
Sachs"), `token_sort_ratio` is the correct scorer.

`process.extractOne()` with `score_cutoff` returns the single best match above the
threshold, or `None` if nothing clears the bar:

```python
match = process.extractOne(
    normalised_candidate,
    choices.keys(),
    scorer=fuzz.token_sort_ratio,
    score_cutoff=90.0,
)
```

Setting `score_cutoff` is important: `rapidfuzz` can skip candidates early once it
determines they cannot reach the cutoff, making the search faster than evaluating
every candidate fully.

---

### `dataclass` for Internal DTOs

`SubmissionRecord` and `DuplicateMatch` are plain `@dataclass` objects, not Pydantic
models. The distinction matters:

- **Pydantic models** are for the validation boundary — untrusted external input
  that must be cleaned and type-checked.
- **Dataclasses** are for internal data transfer between functions that already trust
  their inputs. They are lighter (no validation overhead) and clearer in intent.

`frozen=True` on `SubmissionRecord` makes it immutable and hashable (can be used
as a dict key or set member):

```python
@dataclass(frozen=True)
class SubmissionRecord:
    company_id: str
    company_name: str
    zip_code: str
```

---

### Testing Philosophy

**Normaliser tested independently from the detector.** If the normaliser has a bug,
it appears in normaliser tests with a clear error message. If only the detector were
tested, a normaliser bug would surface as a confusing false negative in a fuzzy
match test.

**`parametrize` for the decision matrix.** The combination of (company name, ZIP,
expected outcome) forms a matrix. `pytest.mark.parametrize` expresses this as a
table rather than five nearly-identical test methods:

```python
@pytest.mark.parametrize(
    "company_name, zip_code, expected_duplicate, expected_matched_id",
    [
        ("Kalepa Inc",         "10001", True,  "DB-001"),
        ("Random Company LLC", "10001", False, None),
        ("Kalepa Inc",         "99999", False, None),   # wrong ZIP — pre-filter blocks it
    ],
)
```

**Testing the pre-filter directly.** One test verifies that a candidate with a ZIP
that has no bucket immediately returns `is_potential_duplicate=False`. This proves
the efficiency claim — the fuzzy matcher is never called — and guards against a
future refactor accidentally removing the pre-filter.

---

## Engineering Deep Dive

### Orthogonality & Separation of Concerns

This task has two files with a clean, one-directional dependency:
`normalizer.py` ← `detector.py`. The normaliser does not know it is used inside a
detector. The detector does not know how normalisation works internally.

| Layer | File | Responsibility | Knows about | Does NOT know about |
|---|---|---|---|---|
| String cleaning | `normalizer.py` | Canonical form of a company name | Regex, token lists | Fuzzy matching, indexes, records |
| Detection engine | `detector.py` | Pre-filter + fuzzy comparison | The index, rapidfuzz, `normalizer.normalise` | How normalisation works internally |

**Practical test — "what changes when...?"**

- The normalisation rules change (e.g., also strip city names like "New York" from
  company names)?
  → Only `normalizer.py`. The detector calls `normalise()` — it doesn't care what
  happens inside.
- The pre-filter strategy changes from ZIP prefix to phone area code?
  → Only the `_zip_prefix()` method and `_index` construction in `detector.py`.
  The normaliser is untouched.
- We switch from `rapidfuzz` to a phonetic algorithm (Soundex) for healthcare
  submissions?
  → Only `_fuzzy_match()` in `detector.py`. Everything else is unchanged.
- The caller wants to know the score but not the binary yes/no decision?
  → `DuplicateMatch` already carries `similarity_score`. The caller can apply a
  different threshold to the raw score. The detector itself does not need to change.

**The principle at work here is information hiding.** The `check()` method's
signature — `check(candidate) -> DuplicateMatch` — is stable. Whether the
implementation uses a ZIP index, a trigram index, or a database query is an
implementation detail the caller never sees and never depends on. You can change the
entire internal algorithm without touching a single caller.

A mid-level implementation would have the index construction, normalisation, and
fuzzy matching in a single function. It works until the day you need to add a second
index strategy (say, phone number prefix as a fallback when ZIP is missing) — at
which point the single function becomes an unreadable tangle.

---

### Library Trade-offs

**`rapidfuzz` vs `fuzzywuzzy` / `thefuzz`**

`fuzzywuzzy` is the original library and is still widely referenced in documentation
and Stack Overflow answers. `rapidfuzz` is a direct replacement with the same API.

| | `fuzzywuzzy` | `rapidfuzz` |
|---|---|---|
| Implementation | Pure Python | C++ extension (10–100× faster) |
| License | GPL-2 (commercial use restrictions) | MIT |
| API | `from fuzzywuzzy import fuzz` | `from rapidfuzz import fuzz` |
| Maintenance | Maintenance mode | Actively developed |

There is no reason to use `fuzzywuzzy` in new code. If you see it in an interview
context, naming `rapidfuzz` as the modern replacement and explaining why (speed,
license) signals that you know the ecosystem.

**`rapidfuzz` vs `difflib.SequenceMatcher` (Python stdlib)**

`difflib` is built into Python — zero additional dependencies. For a small script
comparing a handful of strings, it is the right choice. At production volume
(millions of comparisons per day), `rapidfuzz` is 10–50× faster and supports
token-level scorers that `difflib` lacks. The dependency cost is justified.

**`token_sort_ratio` vs other scorers**

`rapidfuzz` offers several scorers, each suited to different text patterns:

| Scorer | Best for | Weakness |
|---|---|---|
| `ratio` | Short, fixed-format strings | Word-order sensitive: "IBM Corp" ≠ "Corp IBM" |
| `token_sort_ratio` | Company names, names with varying word order | Does not handle partial containment ("IBM" vs "International Business Machines") |
| `partial_ratio` | Short abbreviations vs long full names | Can over-match: "ACE" matches many things |
| `token_set_ratio` | Strings with repeated or extra tokens | Less predictable for short company names |

For company names where the same entity can appear in different word orders
("Goldman Sachs Group" vs "Group Goldman Sachs") but we do not expect abbreviations
to match full names, `token_sort_ratio` is the correct choice. Acknowledging that
abbreviation matching (IBM vs International Business Machines) requires a separate
lookup table or an acronym-expansion pass — not a fuzzy scorer — shows depth of
understanding.

**Score threshold — why 90%?**

90% is a balance between false positives (flagging different companies as duplicates)
and false negatives (missing real duplicates). The right threshold is
business-specific:
- In underwriting, a false positive (flagging two different companies as the same)
  wastes an underwriter's time but is survivable.
- A false negative (pricing the same risk twice) is a financial exposure.
- Therefore, err toward a lower threshold (flag more, miss fewer). 85% would be
  defensible; 95% would miss too many real duplicates.

Making `SIMILARITY_THRESHOLD` a module-level constant (not hardcoded in the
function) means it can be changed in one place and is visible to code reviewers.

---

### System Resiliency

**The detector is read-only by design.**

`DuplicateDetector` never writes to the index, the database, or any external
system. It reads from an index built at construction time and returns a result.
This means:
- It is safe to call from multiple threads simultaneously (no shared mutable state).
- It cannot corrupt data by being called with bad input.
- Failures can only be of two kinds: a wrong answer (a match scored incorrectly) or
  an unhandled exception in a library call. Neither will destroy data.

**How `check()` handles edge cases without raising:**

| Input scenario | What happens |
|---|---|
| ZIP code not in any existing record | `_get_bucket()` returns `[]`, method returns `is_potential_duplicate=False` immediately. No fuzzy match runs. |
| ZIP code is `None` or empty string | `_zip_prefix()` returns `""`, bucket lookup returns `[]`, same safe path. |
| All records in the bucket normalise to empty string | `choices` dict has one key `""`. Fuzzy match against `""` will never clear `score_cutoff=90`. Returns `False`. |
| `check_batch([])` called with empty list | `check_batch` returns `[]`. No errors. |

None of these scenarios require special-case handling in the caller. The method
returns the same type — `DuplicateMatch` — regardless of what happened internally.
This is the **contract** of the public API: always returns a result, never raises.

**The index failure mode:**

The one real failure scenario is building the index from a database that returns
corrupt data (e.g., a `None` zip_code for a record). `_zip_prefix()` guards this:

```python
return (zip_code or "").strip()[:ZIP_PREFIX_LENGTH]
```

A `None` zip_code becomes `""` — the record goes into the empty-string bucket
rather than crashing the build. A submission with a `None` ZIP will only be
compared against other `None`-ZIP records. This is the correct behaviour: we do
not know its location, so we cannot pre-filter it against geographically similar
records, but we also do not lose it.

---

## File Structure

```
task_04_deduplication/
├── normalizer.py   # String cleaning pipeline
├── detector.py     # Inverted index + fuzzy match
└── tests/
    └── test_deduplication.py
```

## Running the Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -v
```
