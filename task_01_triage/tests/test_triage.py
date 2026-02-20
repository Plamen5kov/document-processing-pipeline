"""
Tests for the Submission Triage engine.

Senior testing philosophy demonstrated here:
  - Each class groups tests for one concern (Arrange-Act-Assert per method).
  - Fixtures provide reusable, labelled test data — no copy-paste.
  - Edge cases and error paths are tested, not just the happy path.
  - Rules are tested in isolation AND through the full engine.
  - pytest.mark.parametrize replaces repetitive test methods.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# Make the parent package importable when running pytest from this directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import SubmissionTriage
from models import Submission, TriageStatus
from rules import (
    ConstructionNewYorkRule,
    RevenueOutOfAppetiteRule,
    Rule,
    RuleResult,
    SanctionedIndustryRule,
)

# --------------------------------------------------------------------------- #
# Shared fixtures                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def engine() -> SubmissionTriage:
    return SubmissionTriage()


@pytest.fixture()
def valid_raw() -> dict:
    """A clean, within-appetite submission that should be APPROVED."""
    return {
        "company_id": "ACME-001",
        "company_name": "Acme Corp",
        "revenue": 5_000_000,
        "industry": "Retail",
        "state": "CA",
        "zip_code": "90210",
    }


# --------------------------------------------------------------------------- #
# 1. Input validation / dirty-data handling                                    #
# --------------------------------------------------------------------------- #


class TestInputValidation:
    def test_dirty_revenue_string_dollar_sign_and_commas(self, engine, valid_raw):
        valid_raw["revenue"] = "$5,000,000"
        result = engine.process(valid_raw)
        assert result is not None
        assert result.submission.revenue == Decimal("5000000")

    def test_dirty_revenue_plain_integer_string(self, engine, valid_raw):
        valid_raw["revenue"] = "5000000"
        result = engine.process(valid_raw)
        assert result is not None

    def test_state_is_uppercased(self, engine, valid_raw):
        valid_raw["state"] = "ca"
        result = engine.process(valid_raw)
        assert result.submission.state == "CA"

    def test_industry_is_title_cased(self, engine, valid_raw):
        valid_raw["industry"] = "retail"
        result = engine.process(valid_raw)
        assert result.submission.industry == "Retail"

    def test_missing_required_field_returns_none(self, engine):
        """A submission missing 'company_name' must be skipped, not crash."""
        result = engine.process({"company_id": "X1", "revenue": 100_000})
        assert result is None

    def test_invalid_revenue_string_returns_none(self, engine, valid_raw):
        valid_raw["revenue"] = "not-a-number"
        result = engine.process(valid_raw)
        assert result is None

    def test_blank_company_id_returns_none(self, engine, valid_raw):
        valid_raw["company_id"] = "   "
        result = engine.process(valid_raw)
        assert result is None

    def test_batch_skips_invalid_keeps_valid(self, engine, valid_raw):
        """Invalid submissions must not poison the rest of the batch."""
        bad = {"company_id": "BAD", "revenue": "not-a-number"}
        results = list(engine.process_batch([bad, valid_raw, bad]))
        assert len(results) == 1
        assert results[0].submission.company_id == "ACME-001"


# --------------------------------------------------------------------------- #
# 2. Revenue rule                                                              #
# --------------------------------------------------------------------------- #


class TestRevenueRule:
    @pytest.mark.parametrize(
        "revenue, expected_status",
        [
            (500_000_001, TriageStatus.DECLINED),   # over max
            (500_000_000, TriageStatus.APPROVED),   # exactly at max boundary
            (5_000_000,   TriageStatus.APPROVED),   # comfortably within
            (10_000,      TriageStatus.APPROVED),   # exactly at min boundary
            (9_999,       TriageStatus.DECLINED),   # under min
        ],
    )
    def test_revenue_boundaries(self, engine, valid_raw, revenue, expected_status):
        valid_raw["revenue"] = revenue
        result = engine.process(valid_raw)
        assert result.status == expected_status

    def test_custom_thresholds_respected(self, valid_raw):
        """Engine accepts rules with custom thresholds — DI in action."""
        strict_rule = RevenueOutOfAppetiteRule(
            min_revenue=Decimal("100_000"),
            max_revenue=Decimal("1_000_000"),
        )
        engine = SubmissionTriage(rules=[strict_rule])
        valid_raw["revenue"] = 50_000  # below custom min
        result = engine.process(valid_raw)
        assert result.status == TriageStatus.DECLINED


# --------------------------------------------------------------------------- #
# 3. Construction / New York rule                                              #
# --------------------------------------------------------------------------- #


class TestConstructionNewYorkRule:
    @pytest.mark.parametrize(
        "industry, state, expected_status",
        [
            ("Construction", "NY",        TriageStatus.MANUAL_REVIEW),
            ("Construction", "new york",  TriageStatus.MANUAL_REVIEW),  # normalised
            ("Construction", "CA",        TriageStatus.APPROVED),        # wrong state
            ("Retail",       "NY",        TriageStatus.APPROVED),        # wrong industry
            ("Retail",       "CA",        TriageStatus.APPROVED),        # neither
        ],
    )
    def test_rule_combinations(self, engine, valid_raw, industry, state, expected_status):
        valid_raw["industry"] = industry
        valid_raw["state"] = state
        result = engine.process(valid_raw)
        assert result.status == expected_status

    def test_manual_review_reason_is_descriptive(self, engine, valid_raw):
        valid_raw["industry"] = "Construction"
        valid_raw["state"] = "NY"
        result = engine.process(valid_raw)
        assert result.reason is not None
        assert len(result.reason) > 20  # not a cryptic code


# --------------------------------------------------------------------------- #
# 4. Sanctioned industry rule                                                  #
# --------------------------------------------------------------------------- #


class TestSanctionedIndustryRule:
    def test_gambling_is_declined(self, engine, valid_raw):
        valid_raw["industry"] = "Gambling"
        result = engine.process(valid_raw)
        assert result.status == TriageStatus.DECLINED

    def test_custom_sanctions_list(self, valid_raw):
        rule = SanctionedIndustryRule(sanctioned_industries=frozenset({"Crypto"}))
        engine = SubmissionTriage(rules=[rule])
        valid_raw["industry"] = "Crypto"
        result = engine.process(valid_raw)
        assert result.status == TriageStatus.DECLINED

    def test_non_sanctioned_industry_passes(self, valid_raw):
        rule = SanctionedIndustryRule(sanctioned_industries=frozenset({"Crypto"}))
        engine = SubmissionTriage(rules=[rule])
        valid_raw["industry"] = "Retail"
        result = engine.process(valid_raw)
        assert result.status == TriageStatus.APPROVED


# --------------------------------------------------------------------------- #
# 5. Rule ordering — first-match wins                                          #
# --------------------------------------------------------------------------- #


class TestRuleOrdering:
    def test_first_matching_rule_wins(self, valid_raw):
        """
        A Construction/NY submission that is ALSO over-revenue should be
        DECLINED (revenue rule fires first in the default chain), not
        MANUAL_REVIEW.
        """
        engine = SubmissionTriage()
        valid_raw["revenue"] = 600_000_000
        valid_raw["industry"] = "Construction"
        valid_raw["state"] = "NY"
        result = engine.process(valid_raw)
        assert result.status == TriageStatus.DECLINED


# --------------------------------------------------------------------------- #
# 6. Plugging in a brand-new rule — the Strategy pattern in action             #
# --------------------------------------------------------------------------- #


class TestPluggingInANewRule:
    """
    This test class is the most important one for the interview.

    It proves the Open/Closed Principle claim: a new underwriting requirement
    is implemented by writing ONE new class and injecting it — zero lines in
    the existing engine, models, or rules files are touched.

    Scenario: the underwriting team adds a new appetite restriction —
    "Flag for manual review any submission from Florida with revenue over $10M,
    because of elevated hurricane exposure."

    Without the Strategy pattern: find the right elif in a 500-line function,
    hope you don't break the other 30 conditions, and update every test that
    touches that function.

    With the Strategy pattern: write HighValueFloridaRule, inject it, done.
    """

    def test_new_rule_implemented_without_touching_engine_or_existing_rules(
        self, valid_raw
    ):
        # ── Step 1: Write the new rule ────────────────────────────────────────
        # It lives in the TEST FILE — not in rules.py.
        # This proves the rule system is open for extension without any
        # changes to the production codebase.

        class HighValueFloridaRule(Rule):
            """
            Flag submissions from Florida with revenue over $10M for manual
            review due to elevated hurricane exposure.
            """
            THRESHOLD = 10_000_000

            def evaluate(self, submission: Submission) -> RuleResult | None:
                if (
                    submission.state == "FL"
                    and submission.revenue > self.THRESHOLD
                ):
                    return RuleResult(
                        status=TriageStatus.MANUAL_REVIEW,
                        reason=(
                            f"Florida submissions over ${self.THRESHOLD:,} "
                            "require manual review due to hurricane exposure."
                        ),
                    )
                return None

        # ── Step 2: Inject the new rule into the engine ───────────────────────
        # The engine receives the rule chain via its constructor.
        # No monkey-patching, no subclassing the engine, no module reloading.
        engine = SubmissionTriage(rules=[HighValueFloridaRule()])

        # ── Step 3: Verify the new rule fires correctly ───────────────────────
        valid_raw["state"] = "FL"
        valid_raw["revenue"] = 15_000_000
        result = engine.process(valid_raw)

        assert result.status == TriageStatus.MANUAL_REVIEW
        assert "hurricane" in result.reason.lower()

    def test_new_rule_does_not_fire_below_threshold(self, valid_raw):
        class HighValueFloridaRule(Rule):
            THRESHOLD = 10_000_000

            def evaluate(self, submission: Submission) -> RuleResult | None:
                if submission.state == "FL" and submission.revenue > self.THRESHOLD:
                    return RuleResult(status=TriageStatus.MANUAL_REVIEW, reason="FL high value")
                return None

        engine = SubmissionTriage(rules=[HighValueFloridaRule()])
        valid_raw["state"] = "FL"
        valid_raw["revenue"] = 5_000_000  # under the threshold

        result = engine.process(valid_raw)
        assert result.status == TriageStatus.APPROVED

    def test_new_rule_composed_with_existing_rules(self, valid_raw):
        """
        Show that a custom rule can be prepended to the default chain.
        The new rule fires first; existing rules are unchanged.
        """
        class PriorityClientRule(Rule):
            """Immediately approve submissions from a known priority broker."""
            PRIORITY_BROKER_IDS = frozenset({"BROKER-GOLD-001", "BROKER-GOLD-002"})

            def evaluate(self, submission: Submission) -> RuleResult | None:
                if submission.broker_id in self.PRIORITY_BROKER_IDS:
                    return RuleResult(
                        status=TriageStatus.APPROVED,
                        reason="Priority broker — fast-tracked.",
                    )
                return None

        # Prepend new rule before the standard chain
        engine = SubmissionTriage(rules=[
            PriorityClientRule(),
            *SubmissionTriage._DEFAULT_RULES,   # keep the existing rules after
        ])

        # A submission that would normally be declined (gambling) from a
        # priority broker should be approved — priority rule fires first.
        valid_raw["industry"] = "Gambling"
        valid_raw["broker_id"] = "BROKER-GOLD-001"
        result = engine.process(valid_raw)

        assert result.status == TriageStatus.APPROVED
        assert "Priority broker" in result.reason

    def test_existing_rules_are_not_affected_by_new_rule(self, valid_raw):
        """
        The engine with the new rule still correctly declines over-revenue
        submissions — existing behaviour is preserved.
        """
        class NoOpRule(Rule):
            """A rule that never fires — used to verify existing chain still works."""
            def evaluate(self, submission: Submission) -> RuleResult | None:
                return None

        engine = SubmissionTriage(rules=[
            NoOpRule(),
            *SubmissionTriage._DEFAULT_RULES,
        ])

        valid_raw["revenue"] = 999_000_000  # over max
        result = engine.process(valid_raw)

        assert result.status == TriageStatus.DECLINED
