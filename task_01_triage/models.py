"""
Pydantic v2 models for incoming insurance submissions.

Key concepts:
  - field_validator: cleans a single field before the model is built
  - model_validator: cross-field validation (runs after all fields are set)
  - Enum: restricts triage status to a fixed set of values
  - Decimal: exact arithmetic for money (never use float for currency)
"""

import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class TriageStatus(str, Enum):
    APPROVED = "approved"
    DECLINED = "declined"
    MANUAL_REVIEW = "manual_review"


class Submission(BaseModel):
    """
    Represents a broker-submitted insurance application.

    Accepts 'dirty' revenue strings like "$1,000,000" and normalises them
    to a Decimal for safe arithmetic comparisons downstream.
    """

    company_id: str
    company_name: str
    revenue: Decimal
    industry: str
    state: str
    zip_code: Optional[str] = None
    broker_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Field validators (mode="before" = run BEFORE Pydantic type casting) #
    # ------------------------------------------------------------------ #

    @field_validator("revenue", mode="before")
    @classmethod
    def parse_revenue(cls, v: object) -> Decimal:
        """Accept strings like '$1,000,000' or '1000000' or a plain number."""
        if isinstance(v, str):
            cleaned = re.sub(r"[$,\s]", "", v)
            try:
                return Decimal(cleaned)
            except InvalidOperation:
                raise ValueError(f"Cannot parse revenue from string: {v!r}")
        return v  # Pydantic will coerce int/float → Decimal automatically

    @field_validator("state", mode="before")
    @classmethod
    def normalise_state(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("state must be a string")
        return v.strip().upper()

    @field_validator("industry", mode="before")
    @classmethod
    def normalise_industry(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("industry must be a string")
        return v.strip().title()

    # ------------------------------------------------------------------ #
    # Cross-field validator                                                #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def company_id_not_blank(self) -> "Submission":
        if not self.company_id.strip():
            raise ValueError("company_id must not be blank")
        return self


class TriagedSubmission(BaseModel):
    """Output of the triage engine — wraps the validated submission + decision."""

    submission: Submission
    status: TriageStatus
    reason: Optional[str] = None
