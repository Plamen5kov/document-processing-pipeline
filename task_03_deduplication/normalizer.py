"""
Company name normalisation for fuzzy comparison.

Why normalise before comparing?
  "IBM" vs "International Business Machines" → 0% similarity before normalise,
  still low after, but that is intentional — abbreviations need a separate
  lookup strategy (acronym expansion table).

  "Kalepa Inc." vs "kalepa, inc" → should score 100% — normalisation makes
  this trivial for the fuzzy matcher.

Concept: functional, composable string transformations chained together.
"""

import re

# Entity suffixes to strip — stored as a frozenset for O(1) membership tests.
_ENTITY_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "llc",
        "ltd",
        "limited",
        "co",
        "company",
        "lp",
        "llp",
        "plc",
        "gmbh",
        "sa",
        "ag",
        "group",
        "holdings",
        "international",
    }
)


def normalise(name: str) -> str:
    """
    Produce a canonical form of a company name.

    Pipeline:
      1. Lowercase
      2. Replace punctuation with spaces
      3. Tokenise and strip entity suffixes
      4. Collapse whitespace

    Examples:
      "Kalepa Inc."              → "kalepa"
      "Kalepa, Inc"              → "kalepa"
      "International Business Machines Corp." → "business machines"
      "ACME Holdings Group LLC"  → "acme"
    """
    name = name.lower()

    # Replace any non-alphanumeric character (commas, dots, hyphens) with a space
    name = re.sub(r"[^\w\s]", " ", name)

    # Remove tokens that are entity suffixes OR single characters.
    # Single chars appear from abbreviated forms: "S.A." → "s a" after
    # punctuation stripping — they carry no discriminatory signal.
    tokens = [
        tok for tok in name.split()
        if tok not in _ENTITY_SUFFIXES and len(tok) > 1
    ]

    return " ".join(tokens).strip()
