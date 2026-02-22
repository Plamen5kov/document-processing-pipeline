"""
Chain of Responsibility — Submission Processing Pipeline.

How this differs from the Strategy pattern in task_01:

  Strategy (task_01):
    - Rules are ALTERNATIVES for the same decision ("should this be declined?")
    - The ENGINE iterates the list — rules don't know about each other
    - All rules answer the same question; first match wins

  Chain of Responsibility (this task):
    - Handlers are SEQUENTIAL STAGES — each does a DIFFERENT job
    - Each HANDLER decides whether to pass to the next or short-circuit
    - The client sends the request only to the first handler; the rest
      are wired together inside the chain

The analogy: Strategy is a committee voting on one question. Chain of
Responsibility is a conveyor belt where each station either rejects the
part or passes it to the next station with something added.

Real-world fit:
  This pattern matches how submission processing actually works in production:
  - A duplicate idempotency key → stop immediately, no enrichment wasted
  - A validation failure → stop immediately, no API calls wasted
  - A triage decline → stop immediately, no enrichment wasted on a rejected sub
  - A fuzzy duplicate flag → add a warning, but keep going (not a hard stop)
  - Enrichment failure → mark the field, keep going (soft failure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

from circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Context — the object that flows through the chain                            #
# --------------------------------------------------------------------------- #


class PipelineStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    DECLINED  = "declined"
    DUPLICATE = "potential_duplicate"
    ERROR     = "error"


@dataclass
class SubmissionContext:
    """
    Mutable carrier object passed from handler to handler.

    Each handler reads what it needs and writes its results back.
    This is the key difference from a simple function call chain:
    the context accumulates information as it travels — later handlers
    can see what earlier handlers found.
    """

    payload: dict

    # Filled in as the chain progresses
    status: PipelineStatus = PipelineStatus.PENDING
    company_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    triage_reason: Optional[str] = None
    enrichment_data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Set to True by IdempotencyHandler if this is a replayed request
    was_replay: bool = False

    def add_warning(self, msg: str) -> None:
        logger.warning("Pipeline warning for %r: %s", self.company_id, msg)
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        logger.error("Pipeline error for %r: %s", self.company_id, msg)
        self.errors.append(msg)


# --------------------------------------------------------------------------- #
# Handler base class                                                            #
# --------------------------------------------------------------------------- #


class Handler:
    """
    Abstract base for all pipeline stages.

    Each handler:
      1. Does its work on the context.
      2. Either returns (short-circuit) or calls self._next(context) to continue.

    The decision to call _next is made BY THE HANDLER, not by an external engine.
    This is the defining characteristic of Chain of Responsibility vs Strategy.
    """

    def __init__(self) -> None:
        self._next_handler: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        """
        Wire the next handler and return it so calls can be chained fluently:

            a.set_next(b).set_next(c).set_next(d)
        """
        self._next_handler = handler
        return handler

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        """Override in subclasses. Call self._next(ctx) to continue the chain."""
        return self._next(ctx)

    def _next(self, ctx: SubmissionContext) -> SubmissionContext:
        if self._next_handler:
            return self._next_handler.handle(ctx)
        return ctx  # end of chain — return context as-is


# --------------------------------------------------------------------------- #
# Concrete handlers                                                            #
# --------------------------------------------------------------------------- #


class IdempotencyHandler(Handler):
    """
    Stage 1: Check if we have already processed this exact payload.

    Short-circuits on cache hit — no downstream work is done.
    On miss: records the key in the context and passes to the next handler.
    The result will be stored in the cache AFTER the chain completes
    (the caller's responsibility — the handler only checks, not stores,
    to avoid storing partial results if a later handler fails).
    """

    def __init__(self, cache: dict) -> None:
        super().__init__()
        self._cache = cache

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        import hashlib, json
        key = hashlib.sha256(
            json.dumps(ctx.payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        ctx.idempotency_key = key

        if key in self._cache:
            logger.info("Idempotency hit — returning cached result for key=%s", key[:16])
            return replace(self._cache[key], was_replay=True)  # ← short-circuit: copy, don't mutate the cache

        return self._next(ctx)  # ← pass to next handler


class ValidationHandler(Handler):
    """
    Stage 2: Extract and validate required fields from the raw payload.

    Short-circuits on validation failure — no point triaging invalid data.
    On success: populates ctx.company_id and other fields, then continues.
    """

    _REQUIRED = ("company_id", "company_name", "revenue", "industry", "state")

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        missing = [f for f in self._REQUIRED if ctx.payload.get(f) in (None, "")]
        if missing:
            ctx.status = PipelineStatus.ERROR
            ctx.add_error(f"Missing required fields: {missing}")
            return ctx  # ← short-circuit

        ctx.company_id = ctx.payload["company_id"]
        logger.info("Validation passed for company_id=%r", ctx.company_id)
        return self._next(ctx)


class TriageHandler(Handler):
    """
    Stage 3: Apply underwriting appetite rules.

    Short-circuits on DECLINED — no enrichment wasted on rejected submissions.
    On APPROVED or MANUAL_REVIEW: continues to enrich and deduplicate.

    The rules themselves use the Strategy pattern from task_01 — both patterns
    coexist. CoR structures the pipeline; Strategy structures the rule set
    inside one stage of that pipeline.
    """

    def __init__(self, rules: list) -> None:
        super().__init__()
        self._rules = rules

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        for rule in self._rules:
            result = rule.evaluate_raw(ctx)
            if result is not None:
                ctx.status = result.status
                ctx.triage_reason = result.reason
                if ctx.status == PipelineStatus.DECLINED:
                    logger.info("Triage DECLINED %r — stopping chain", ctx.company_id)
                    return ctx  # ← short-circuit: don't enrich a declined submission
                # MANUAL_REVIEW: flag it, but keep going
                ctx.add_warning(f"Manual review required: {result.reason}")
                break

        if ctx.status == PipelineStatus.PENDING:
            ctx.status = PipelineStatus.APPROVED

        return self._next(ctx)


class DeduplicationHandler(Handler):
    """
    Stage 4: Check for fuzzy duplicates.

    Soft failure — a potential duplicate gets a warning but the chain
    continues. The submission is flagged, not dropped. An underwriter
    will review the warning.

    Contrast with TriageHandler which HARD-stops on DECLINED.
    This shows that not all handlers short-circuit — some just annotate
    and pass through.
    """

    def __init__(self, existing_ids: set[str]) -> None:
        super().__init__()
        self._existing = existing_ids

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        if ctx.company_id in self._existing:
            ctx.status = PipelineStatus.DUPLICATE
            ctx.add_warning(
                f"company_id={ctx.company_id!r} already exists — potential duplicate."
            )
            # ← does NOT short-circuit: pass through so enrichment still runs
        return self._next(ctx)


class EnrichmentHandler(Handler):
    """
    Stage 5: Enrich with external risk data.

    Soft failure — enrichment errors are recorded as warnings, not fatal errors.
    The submission proceeds with whatever data was retrieved.
    An 'enrichment_failed' flag lets downstream consumers handle this gracefully.

    Circuit breaker integration:
      If a CircuitBreaker is provided and its state is OPEN, the handler
      fast-fails without touching the API — no timeout, no retry wait.
      On success/failure it informs the breaker so it can track the rate.
    """

    def __init__(self, risk_db: dict, breaker: Optional[CircuitBreaker] = None) -> None:
        super().__init__()
        self._risk_db = risk_db
        self._breaker = breaker

    def handle(self, ctx: SubmissionContext) -> SubmissionContext:
        # Circuit breaker fast-fail: skip the API entirely when OPEN
        if self._breaker and not self._breaker.allow_request():
            logger.warning(
                "Circuit OPEN — skipping enrichment for %r", ctx.company_id
            )
            ctx.add_warning(
                "Enrichment skipped: Risk API circuit breaker is OPEN. "
                "The API has been failing — enrichment will resume automatically."
            )
            ctx.enrichment_data = {"enrichment_failed": True, "reason": "circuit_open"}
            return self._next(ctx)

        # Normal path: call the API (in production this would be requests.get(...))
        data = self._risk_db.get(ctx.company_id)
        if data is None:
            ctx.add_warning(f"No enrichment data found for {ctx.company_id!r}")
            ctx.enrichment_data = {"enrichment_failed": True}
            if self._breaker:
                self._breaker.record_failure()
        else:
            ctx.enrichment_data = data
            logger.info("Enrichment succeeded for %r", ctx.company_id)
            if self._breaker:
                self._breaker.record_success()

        return self._next(ctx)


# --------------------------------------------------------------------------- #
# Pipeline builder                                                             #
# --------------------------------------------------------------------------- #


class SubmissionPipeline:
    """
    Assembles the handler chain and provides a single entry point.

    The builder separates chain construction from chain execution.
    Tests can build minimal chains (e.g., just Validation → Triage) to
    test one stage without the overhead of the full pipeline.
    """

    def __init__(
        self,
        cache: dict,
        rules: list,
        existing_ids: set[str],
        risk_db: dict,
        breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        # Build the chain: each set_next() returns the next handler
        # so the calls can be written as a readable left-to-right sequence.
        head = IdempotencyHandler(cache)
        head \
            .set_next(ValidationHandler()) \
            .set_next(TriageHandler(rules)) \
            .set_next(DeduplicationHandler(existing_ids)) \
            .set_next(EnrichmentHandler(risk_db, breaker))

        self._head = head
        self._cache = cache

    def run(self, payload: dict) -> SubmissionContext:
        ctx = SubmissionContext(payload=payload)
        result = self._head.handle(ctx)

        # Store result in cache AFTER the full chain completes.
        # Storing before would cache a partial result if a later handler fails.
        if result.idempotency_key and not result.was_replay:
            self._cache[result.idempotency_key] = result

        return result
