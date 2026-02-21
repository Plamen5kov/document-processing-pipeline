"""
Pydantic models for LLM-extracted submission data.

The critical difference from task_01's Submission model:

  task_01 (broker input):  all required fields, hard fail on missing.
  task_04 (LLM output):    all Optional fields, soft fail on missing.

Why? Broker data is a formal API contract — missing fields ARE bugs.
LLM output is probabilistic — a field genuinely might not appear in
the source text. We want to use what we got, not discard it.

The Pydantic model here plays the role of the "Judging Agent" in the
Extraction+Judging pipeline: it enforces the output contract and
classifies confidence, without throwing away partial work.
"""

import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


class ExtractionStatus(str, Enum):
    SUCCESS = "success"   # All required fields extracted
    PARTIAL = "partial"   # Some required fields missing — downstream decides if usable
    FAILED  = "failed"    # Could not parse LLM output at all (JSON error, schema mismatch)


# Fields that MUST be present for a downstream triage to run.
# zip_code is useful but not blocking.
REQUIRED_FOR_TRIAGE: frozenset[str] = frozenset(
    {"company_name", "revenue", "industry", "state"}
)


class ExtractedSubmission(BaseModel):
    """
    LLM-extracted submission fields. Every field is Optional because
    the source text may not contain all information.

    Validator behaviour on bad LLM output:
      - Revenue like "~$4.5M" or "four million" → coerced or set to None
        (do NOT raise — a partial extraction is more useful than a failed one)
      - State like "New York" → uppercased → "NEW YORK"
        (same normalisation as task_01 so downstream rules work unchanged)
    """

    company_name: Optional[str] = None
    revenue: Optional[Decimal] = None
    industry: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    @field_validator("revenue", mode="before")
    @classmethod
    def coerce_revenue(cls, v: object) -> Optional[Decimal]:
        if v is None:
            return None
        # Decimal passed directly (e.g. from tests or internal code)
        if isinstance(v, Decimal):
            return v
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            # Handle "$4,500,000", "4.5M", "500k", "4500000.00", etc.
            cleaned = re.sub(r"[$,\s]", "", v)
            # Suffix multipliers: multiply numerically, not by string concatenation.
            # "4.5M" → Decimal("4.5") × 1_000_000 = Decimal("4500000")
            multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
            for suffix, factor in multipliers.items():
                if cleaned.lower().endswith(suffix):
                    try:
                        return Decimal(cleaned[:-1]) * factor
                    except InvalidOperation:
                        return None
            try:
                return Decimal(cleaned)
            except InvalidOperation:
                # LLM returned something un-parseable — treat as not found
                return None
        return None

    @field_validator("state", mode="before")
    @classmethod
    def normalise_state(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        return v.strip().upper()

    @field_validator("company_name", "industry", mode="before")
    @classmethod
    def clean_string(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        stripped = v.strip()
        return stripped if stripped else None


class ExtractionResult(BaseModel):
    """
    Full record of one extraction attempt: what went in, what came out,
    what was missing, and whether it succeeded.

    This is the object the caller receives — it contains everything
    needed to decide whether to proceed, retry, or escalate.
    """

    source_text: str
    status: ExtractionStatus
    extracted: Optional[ExtractedSubmission] = None
    missing_fields: list[str] = []
    raw_llm_output: Optional[str] = None
    error: Optional[str] = None
