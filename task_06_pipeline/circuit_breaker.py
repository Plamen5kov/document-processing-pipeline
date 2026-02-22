"""
Circuit Breaker — resilience pattern for external dependencies.

The problem it solves:
  Retry (task_02, tenacity) handles a SINGLE transient failure:
    "the API hiccuped — wait 1s and try again."

  Circuit breaker handles SUSTAINED failure:
    "the API has been returning errors for 30 seconds — stop trying,
    fail fast, and give it time to recover."

  Without a circuit breaker, a downed external API causes every request
  in the queue to: call → wait for timeout (5s) → retry × 3 → fail.
  At 1000 pending submissions, that is 1000 × 15s = 4+ hours of wasted
  waiting.  With a circuit breaker, after the first ~5 failures the
  remaining 995 calls fail in microseconds.

The three states:

  ┌─────────┐   failure_threshold reached   ┌──────┐
  │ CLOSED  │ ─────────────────────────────▶ │ OPEN │
  │(normal) │                                │(fast │
  └─────────┘                                │ fail)│
       ▲                                     └──────┘
       │                                         │
  success (probe)                   recovery_timeout elapsed
       │                                         │
  ┌───────────┐                                  │
  │ HALF_OPEN │ ◀────────────────────────────────┘
  │  (probe)  │
  └───────────┘
       │ failure (probe)
       ▼
    OPEN  (timer resets)

  CLOSED    → normal operation; all calls pass through.
  OPEN      → all calls fail immediately; no external call is made.
  HALF_OPEN → one probe call is allowed through to test recovery.
              Success → CLOSED.  Failure → OPEN (timer resets).

Relationship to retry:
  Retry and circuit breaker operate at different scopes:
  - Retry:           retries ONE request a few times (milliseconds)
  - Circuit breaker: tracks failure rate ACROSS requests (seconds/minutes)

  Typical production composition:
    request → circuit_breaker.allow_request()?
                yes → tenacity @retry → external_call()
                        success → circuit_breaker.record_success()
                        all retries fail → circuit_breaker.record_failure()
                no  → fast-fail immediately
"""

import threading
import time
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    CLOSED    = "closed"      # Normal — calls pass through
    OPEN      = "open"        # Failing — calls are rejected immediately
    HALF_OPEN = "half_open"   # Probing — one call allowed to test recovery


class CircuitBreaker:
    """
    Thread-safe three-state circuit breaker.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        if breaker.allow_request():
            try:
                result = call_external_api()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
        else:
            # Fast-fail — don't call the API at all
            raise CircuitOpenError("API circuit is OPEN")
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        # One lock guards all state mutations — prevents torn reads
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Read-only configuration properties                                   #
    # ------------------------------------------------------------------ #

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def recovery_timeout(self) -> float:
        return self._recovery_timeout

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self.__current_state()

    def allow_request(self) -> bool:
        """
        Return True if this request should be attempted.
        Must be paired with record_success() or record_failure() after the call.
        """
        with self._lock:
            state = self.__current_state()
            # CLOSED: always allow. HALF_OPEN: allow the probe. OPEN: deny.
            return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Call after a successful external call."""
        with self._lock:
            self._failure_count = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Call after a failed external call (after retries are exhausted)."""
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()  # timer starts only when circuit opens

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def __current_state(self) -> CircuitState:
        """
        Evaluate the current state, applying the OPEN → HALF_OPEN
        time-based transition.  Must be called with self._lock held.
        Name-mangled (__) to prevent accidental unlocked calls from subclasses.
        """
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
        return self._state
