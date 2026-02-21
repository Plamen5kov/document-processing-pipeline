"""
Tests for the Chain of Responsibility pipeline.

What these tests demonstrate:
  - Each handler can be tested in a minimal chain (just that handler + no next)
  - Short-circuit behaviour: declined submissions don't reach enrichment
  - Pass-through behaviour: duplicate warnings don't stop the chain
  - The full pipeline integrates all stages in the correct order
  - Adding a new handler requires zero changes to existing handlers
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import (
    DeduplicationHandler,
    EnrichmentHandler,
    Handler,
    IdempotencyHandler,
    PipelineStatus,
    SubmissionContext,
    SubmissionPipeline,
    ValidationHandler,
)

# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

VALID_PAYLOAD = {
    "company_id": "ACME-001",
    "company_name": "Acme Corp",
    "revenue": 5_000_000,
    "industry": "Retail",
    "state": "CA",
}


def _ctx(payload: dict | None = None) -> SubmissionContext:
    return SubmissionContext(payload=payload or VALID_PAYLOAD)


# --------------------------------------------------------------------------- #
# 1. Handler base — set_next / _next wiring                                   #
# --------------------------------------------------------------------------- #


class TestHandlerWiring:
    def test_set_next_returns_the_next_handler_for_fluent_chaining(self):
        a, b, c = Handler(), Handler(), Handler()
        result = a.set_next(b)
        assert result is b   # fluent: a.set_next(b).set_next(c) works

    def test_handler_with_no_next_returns_context_unchanged(self):
        h = ValidationHandler()          # next is None
        ctx = _ctx()
        result = h.handle(ctx)
        assert result is ctx             # same object returned

    def test_chain_calls_next_handler(self):
        call_log = []

        class RecordingHandler(Handler):
            def handle(self, ctx):
                call_log.append(type(self).__name__)
                return self._next(ctx)

        a = RecordingHandler()
        b = RecordingHandler()
        a.set_next(b)
        a.handle(_ctx())

        assert call_log == ["RecordingHandler", "RecordingHandler"]


# --------------------------------------------------------------------------- #
# 2. IdempotencyHandler                                                        #
# --------------------------------------------------------------------------- #


class TestIdempotencyHandler:
    def test_miss_populates_key_and_continues(self):
        reached_next = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached_next.append(True)
                return ctx

        cache = {}
        h = IdempotencyHandler(cache)
        h.set_next(Sentinel())
        h.handle(_ctx())

        assert reached_next == [True]

    def test_hit_returns_cached_context_and_short_circuits(self):
        reached_next = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached_next.append(True)
                return ctx

        # Pre-populate cache with a known context
        cached_ctx = _ctx()
        cached_ctx.status = PipelineStatus.APPROVED
        cache = {}

        # First call — populates the key
        h = IdempotencyHandler(cache)
        h.set_next(Sentinel())
        first = h.handle(_ctx())
        key = first.idempotency_key
        cache[key] = cached_ctx   # simulate the pipeline storing it

        # Second call — should hit the cache and NOT call Sentinel again
        reached_next.clear()
        second = h.handle(_ctx())

        assert second.was_replay is True
        assert second.status == PipelineStatus.APPROVED
        assert reached_next == []   # chain was short-circuited


# --------------------------------------------------------------------------- #
# 3. ValidationHandler                                                         #
# --------------------------------------------------------------------------- #


class TestValidationHandler:
    def test_valid_payload_populates_company_id_and_continues(self):
        reached = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached.append(ctx.company_id)
                return ctx

        h = ValidationHandler()
        h.set_next(Sentinel())
        h.handle(_ctx())

        assert reached == ["ACME-001"]

    def test_missing_field_sets_error_and_short_circuits(self):
        reached = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached.append(True)
                return ctx

        h = ValidationHandler()
        h.set_next(Sentinel())
        ctx = SubmissionContext(payload={"company_id": "X"})   # missing all other fields
        result = h.handle(ctx)

        assert result.status == PipelineStatus.ERROR
        assert result.errors                    # error message recorded
        assert reached == []                    # chain stopped


# --------------------------------------------------------------------------- #
# 4. TriageHandler                                                             #
# --------------------------------------------------------------------------- #


class TestTriageHandler:
    def _make_rule(self, *, fires: bool, status: PipelineStatus, reason: str = "test"):
        """Build a minimal mock rule."""
        result = MagicMock()
        result.status = status
        result.reason = reason
        mock_rule = MagicMock()
        mock_rule.evaluate_raw.return_value = result if fires else None
        return mock_rule

    def test_declined_rule_short_circuits_before_next_handler(self):
        reached = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached.append(True)
                return ctx

        rule = self._make_rule(fires=True, status=PipelineStatus.DECLINED)
        h = ValidationHandler()
        h.set_next(TriageHandler := __import__("pipeline").TriageHandler)

        from pipeline import TriageHandler
        triage = TriageHandler(rules=[rule])
        triage.set_next(type("S", (Handler,), {"handle": lambda self, c: (reached.append(True), c)[1]})())

        ctx = _ctx()
        ctx.company_id = "ACME-001"
        result = triage.handle(ctx)

        assert result.status == PipelineStatus.DECLINED
        assert reached == []   # enrichment never called

    def test_no_matching_rule_approves_and_continues(self):
        reached = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached.append(True)
                return ctx

        from pipeline import TriageHandler
        rule = self._make_rule(fires=False, status=PipelineStatus.PENDING)
        h = TriageHandler(rules=[rule])
        h.set_next(Sentinel())

        ctx = _ctx()
        ctx.company_id = "ACME-001"
        result = h.handle(ctx)

        assert result.status == PipelineStatus.APPROVED
        assert reached == [True]   # chain continued


# --------------------------------------------------------------------------- #
# 5. DeduplicationHandler — soft failure, chain continues                      #
# --------------------------------------------------------------------------- #


class TestDeduplicationHandler:
    def test_known_id_adds_warning_but_continues(self):
        reached = []

        class Sentinel(Handler):
            def handle(self, ctx):
                reached.append(True)
                return ctx

        h = DeduplicationHandler(existing_ids={"ACME-001"})
        h.set_next(Sentinel())
        ctx = _ctx()
        ctx.company_id = "ACME-001"
        result = h.handle(ctx)

        assert result.status == PipelineStatus.DUPLICATE
        assert result.warnings              # warning was added
        assert reached == [True]            # chain still continued — NOT short-circuited

    def test_unknown_id_passes_through_unchanged(self):
        h = DeduplicationHandler(existing_ids={"OTHER"})
        ctx = _ctx()
        ctx.company_id = "ACME-001"
        result = h.handle(ctx)

        assert result.status == PipelineStatus.PENDING   # unchanged by this handler
        assert result.warnings == []


# --------------------------------------------------------------------------- #
# 6. Full pipeline integration                                                 #
# --------------------------------------------------------------------------- #


class TestFullPipeline:
    def _pipeline(
        self,
        *,
        existing_ids: set[str] = frozenset(),
        risk_db: dict | None = None,
        rules: list | None = None,
    ) -> SubmissionPipeline:
        return SubmissionPipeline(
            cache={},
            rules=rules or [],
            existing_ids=existing_ids,
            risk_db=risk_db or {"ACME-001": {"risk_score": 72}},
        )

    def test_happy_path_reaches_enrichment(self):
        pipeline = self._pipeline()
        result = pipeline.run(VALID_PAYLOAD)

        assert result.status == PipelineStatus.APPROVED
        assert result.enrichment_data == {"risk_score": 72}
        assert result.errors == []

    def test_validation_failure_stops_before_enrichment(self):
        pipeline = self._pipeline(risk_db={"ACME-001": {"risk_score": 99}})
        result = pipeline.run({"company_id": "X"})   # missing required fields

        assert result.status == PipelineStatus.ERROR
        assert result.enrichment_data == {}    # enrichment never ran

    def test_declined_submission_is_not_enriched(self):
        enrichment_called = []

        class SpyEnrichmentHandler(EnrichmentHandler):
            def handle(self, ctx):
                enrichment_called.append(True)
                return super().handle(ctx)

        # Manually build a minimal pipeline to inject the spy
        from pipeline import IdempotencyHandler, ValidationHandler, TriageHandler

        decline_rule = MagicMock()
        decline_result = MagicMock()
        decline_result.status = PipelineStatus.DECLINED
        decline_result.reason = "Over revenue"
        decline_rule.evaluate_raw.return_value = decline_result

        cache = {}
        head = IdempotencyHandler(cache)
        head.set_next(ValidationHandler()) \
            .set_next(TriageHandler([decline_rule])) \
            .set_next(SpyEnrichmentHandler(risk_db={}))

        ctx = head.handle(SubmissionContext(payload=VALID_PAYLOAD))

        assert ctx.status == PipelineStatus.DECLINED
        assert enrichment_called == []    # the spy was never reached

    def test_duplicate_flag_does_not_stop_enrichment(self):
        pipeline = self._pipeline(
            existing_ids={"ACME-001"},
            risk_db={"ACME-001": {"risk_score": 55}},
        )
        result = pipeline.run(VALID_PAYLOAD)

        # Duplicate warning was set...
        assert result.status == PipelineStatus.DUPLICATE
        assert result.warnings

        # ...but enrichment still ran — soft failure
        assert result.enrichment_data == {"risk_score": 55}

    def test_second_run_with_same_payload_is_a_replay(self):
        pipeline = self._pipeline()
        pipeline.run(VALID_PAYLOAD)         # first run — processes and caches
        result = pipeline.run(VALID_PAYLOAD)  # second run — cache hit

        assert result.was_replay is True

    def test_adding_new_handler_without_touching_existing_ones(self):
        """
        This is the Chain of Responsibility equivalent of the Strategy test
        in task_01: a new stage is added by writing one new class and
        inserting it into the chain — no existing handler changes.
        """

        class AuditLogHandler(Handler):
            """New requirement: log every submission to an audit trail."""

            def __init__(self) -> None:
                super().__init__()
                self.logged: list[str] = []

            def handle(self, ctx: SubmissionContext) -> SubmissionContext:
                self.logged.append(ctx.company_id or "UNKNOWN")
                return self._next(ctx)   # always continue — pure side-effect handler

        audit = AuditLogHandler()

        # Insert between validation and triage — no existing class modified
        cache = {}
        from pipeline import IdempotencyHandler, ValidationHandler, TriageHandler, DeduplicationHandler, EnrichmentHandler

        head = IdempotencyHandler(cache)
        head.set_next(ValidationHandler()) \
            .set_next(audit) \
            .set_next(TriageHandler([])) \
            .set_next(DeduplicationHandler(set())) \
            .set_next(EnrichmentHandler({"ACME-001": {}}))

        head.handle(SubmissionContext(payload=VALID_PAYLOAD))

        assert audit.logged == ["ACME-001"]   # new handler ran
