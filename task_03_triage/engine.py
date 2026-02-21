"""
SubmissionTriage — the engine that orchestrates rule evaluation.

Key design decisions:
  1. Rules are injected via the constructor (Dependency Injection), making
     the engine trivially testable with custom rule sets.
  2. Pydantic ValidationError is caught at the boundary so a single bad
     submission never poisons the batch.
  3. Python's logging module (not print) is used throughout.  Callers can
     configure handlers/formatters however they like.
"""

import logging
from typing import Iterable, Iterator, Optional

from pydantic import ValidationError

from models import Submission, TriagedSubmission, TriageStatus
from rules import (
    ConstructionNewYorkRule,
    RevenueOutOfAppetiteRule,
    Rule,
    SanctionedIndustryRule,
)

logger = logging.getLogger(__name__)


class SubmissionTriage:
    """
    Applies an ordered list of Rules to each incoming submission.

    The first rule that fires determines the outcome (short-circuit).
    If no rule fires, the submission is APPROVED.

    Usage:
        engine = SubmissionTriage()
        result = engine.process(raw_dict)        # single submission
        results = list(engine.process_batch(list_of_dicts))  # stream
    """

    # Default production rule chain — order matters.
    # Revenue check runs first because it's the fastest/cheapest filter.
    _DEFAULT_RULES: list[Rule] = [
        RevenueOutOfAppetiteRule(),
        SanctionedIndustryRule(),
        ConstructionNewYorkRule(),
    ]

    def __init__(self, rules: Optional[list[Rule]] = None) -> None:
        # Allow callers to override the chain (e.g., in tests or A/B experiments)
        self._rules = rules if rules is not None else self._DEFAULT_RULES

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def process(self, raw: dict) -> Optional[TriagedSubmission]:
        """
        Validate and triage a single raw submission dict.

        Returns None (and logs) if validation fails — caller's batch loop
        can safely continue to the next item.
        """
        submission = self._parse(raw)
        if submission is None:
            return None
        return self._apply_rules(submission)

    def process_batch(self, raws: Iterable[dict]) -> Iterator[TriagedSubmission]:
        """
        Generator: yields TriagedSubmission for each valid item.

        Skips (and logs) invalid submissions rather than raising — this is
        the correct behaviour for a stream processor where partial progress
        is better than a full stop.

        Example:
            results = list(engine.process_batch(submissions))
            approved = [r for r in results if r.status == TriageStatus.APPROVED]
        """
        for raw in raws:
            result = self.process(raw)
            if result is not None:
                yield result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _parse(self, raw: dict) -> Optional[Submission]:
        try:
            return Submission.model_validate(raw)
        except ValidationError as exc:
            # Log a structured error but never re-raise — keep the batch alive.
            logger.error(
                "Validation failed for submission id=%r: %s",
                raw.get("company_id", "UNKNOWN"),
                exc.errors(include_url=False),  # include_url=False keeps logs clean
            )
            return None

    def _apply_rules(self, submission: Submission) -> TriagedSubmission:
        for rule in self._rules:
            result = rule.evaluate(submission)
            if result is not None:
                logger.info(
                    "Rule %s fired for company_id=%r → %s",
                    type(rule).__name__,
                    submission.company_id,
                    result.status.value,
                )
                return TriagedSubmission(
                    submission=submission,
                    status=result.status,
                    reason=result.reason,
                )

        logger.debug("No rules fired for company_id=%r → APPROVED", submission.company_id)
        return TriagedSubmission(submission=submission, status=TriageStatus.APPROVED)
