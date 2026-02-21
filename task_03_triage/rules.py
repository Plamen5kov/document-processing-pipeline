"""
Strategy Pattern for triage rules.

Why Strategy?
  - Adding a new underwriting rule means adding ONE new class, never editing
    existing code (Open/Closed Principle).
  - Rules can be composed, ordered, and even loaded from config at runtime.
  - Each rule is independently unit-testable.

Pattern shape:
  Rule (ABC)            ← abstract interface
    └── evaluate(sub)   ← returns RuleResult | None
          None  = rule did not fire; engine tries the next rule.
          RuleResult = rule fired; engine returns this decision immediately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from models import Submission, TriageStatus


@dataclass(frozen=True)
class RuleResult:
    status: TriageStatus
    reason: str


class Rule(ABC):
    @abstractmethod
    def evaluate(self, submission: Submission) -> Optional[RuleResult]:
        """Return a RuleResult if this rule fires, None otherwise."""
        ...


# --------------------------------------------------------------------------- #
# Concrete rules                                                               #
# --------------------------------------------------------------------------- #


class RevenueOutOfAppetiteRule(Rule):
    """
    Decline submissions whose revenue falls outside the company's appetite.

    Thresholds are instance attributes so tests (and future config) can
    override them without subclassing.
    """

    def __init__(
        self,
        min_revenue: Decimal = Decimal("10_000"),
        max_revenue: Decimal = Decimal("500_000_000"),
    ) -> None:
        self.min_revenue = min_revenue
        self.max_revenue = max_revenue

    def evaluate(self, submission: Submission) -> Optional[RuleResult]:
        if submission.revenue > self.max_revenue:
            return RuleResult(
                status=TriageStatus.DECLINED,
                reason=(
                    f"Revenue ${submission.revenue:,} exceeds the maximum "
                    f"appetite of ${self.max_revenue:,}."
                ),
            )
        if submission.revenue < self.min_revenue:
            return RuleResult(
                status=TriageStatus.DECLINED,
                reason=(
                    f"Revenue ${submission.revenue:,} is below the minimum "
                    f"appetite of ${self.min_revenue:,}."
                ),
            )
        return None


class ConstructionNewYorkRule(Rule):
    """
    Flag Construction submissions in NY for manual review.

    Both 'NY' and 'NEW YORK' are accepted because the field validator
    on Submission.state already uppercases the value; the normalise_state
    validator only produces 2-letter codes when the broker sends them, so
    we accept the written-out form as well for robustness.
    """

    _NY_VARIANTS = frozenset({"NY", "NEW YORK", "NEW YORK STATE"})

    def evaluate(self, submission: Submission) -> Optional[RuleResult]:
        if (
            submission.industry == "Construction"
            and submission.state in self._NY_VARIANTS
        ):
            return RuleResult(
                status=TriageStatus.MANUAL_REVIEW,
                reason=(
                    "Construction submissions in New York require manual review "
                    "due to complex local regulations (e.g., Labor Law 240/241)."
                ),
            )
        return None


class SanctionedIndustryRule(Rule):
    """
    Example of a data-driven rule: decline submissions from industries
    listed in a configurable sanctions list.  This shows how the Strategy
    pattern scales — the rule logic is generic; the data drives behaviour.
    """

    DEFAULT_SANCTIONED = frozenset({"Gambling", "Tobacco", "Weapons Manufacturing"})

    def __init__(self, sanctioned_industries: Optional[frozenset] = None) -> None:
        self.sanctioned = sanctioned_industries or self.DEFAULT_SANCTIONED

    def evaluate(self, submission: Submission) -> Optional[RuleResult]:
        if submission.industry in self.sanctioned:
            return RuleResult(
                status=TriageStatus.DECLINED,
                reason=f"Industry '{submission.industry}' is outside our appetite.",
            )
        return None
