"""
Risk API client with retry, timeout, and camelCase normalisation.

Key concepts demonstrated:
  - tenacity: declarative retry strategy (exponential backoff, jitter)
  - requests: HTTP with explicit timeout (NEVER call requests without one)
  - Custom exception hierarchy: lets callers catch precisely what they need
  - Separation of concerns: _fetch() handles HTTP; enrich_submission() handles
    business logic (mapping, error classification)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mapper import normalise_keys

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (would live in settings / env vars in production)              #
# --------------------------------------------------------------------------- #

RISK_API_BASE = "https://api.risk-provider.example.com/v1"
REQUEST_TIMEOUT = 5.0   # seconds — always set a timeout on outbound HTTP calls
MAX_RETRIES = 3
BACKOFF_MIN = 1         # seconds
BACKOFF_MAX = 10        # seconds


# --------------------------------------------------------------------------- #
# Exception hierarchy                                                          #
# --------------------------------------------------------------------------- #


class RiskApiError(Exception):
    """Base for all Risk API errors."""


class RiskApiServerError(RiskApiError):
    """5xx — transient; safe to retry."""


class RiskApiClientError(RiskApiError):
    """4xx — our fault; retrying won't help."""


# --------------------------------------------------------------------------- #
# Output model (dataclass is lighter than Pydantic for internal DTOs)          #
# --------------------------------------------------------------------------- #


@dataclass
class EnrichmentResult:
    company_id: str
    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Low-level HTTP call — wrapped by tenacity                                    #
# --------------------------------------------------------------------------- #


@retry(
    # Retry up to MAX_RETRIES times after the first attempt
    stop=stop_after_attempt(MAX_RETRIES),
    # Wait 1s, 2s, 4s … capped at 10s between attempts (exponential backoff)
    wait=wait_exponential(multiplier=1, min=BACKOFF_MIN, max=BACKOFF_MAX),
    # Only retry on transient errors — never on 4xx (our bug) or parse errors
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, RiskApiServerError)),
    # Log before sleeping so operators can see what's happening
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # Raise the *original* exception, not tenacity's RetryError wrapper
    reraise=True,
)
def _fetch_risk_data(company_id: str) -> dict:
    """
    Single HTTP attempt.  tenacity will call this repeatedly on failure.

    Raises:
        requests.Timeout        — if the server is slow (retriable)
        requests.ConnectionError — network issue (retriable)
        RiskApiServerError      — 5xx response (retriable)
        RiskApiClientError      — 4xx response (not retriable)
    """
    url = f"{RISK_API_BASE}/companies/{company_id}/risk"

    # Always specify timeout= — a hanging socket blocks the thread forever.
    response = requests.get(url, timeout=REQUEST_TIMEOUT)

    if response.status_code == 429:
        # Rate-limited: treat as transient so tenacity retries
        raise RiskApiServerError(f"Rate-limited by Risk API (429) for {company_id}")

    if response.status_code >= 500:
        raise RiskApiServerError(
            f"Risk API server error {response.status_code} for {company_id}"
        )

    if response.status_code >= 400:
        raise RiskApiClientError(
            f"Risk API client error {response.status_code} for {company_id}"
        )

    return response.json()


# --------------------------------------------------------------------------- #
# Public function                                                              #
# --------------------------------------------------------------------------- #


def enrich_submission(company_id: str) -> EnrichmentResult:
    """
    Fetch external risk signals for a company, returning an EnrichmentResult.

    Never raises — a failed enrichment is a business outcome, not an
    unhandled exception.  The caller decides what to do with success=False.
    """
    try:
        raw = _fetch_risk_data(company_id)
        snake_data = normalise_keys(raw)
        logger.info("Enrichment succeeded for company_id=%r", company_id)
        return EnrichmentResult(company_id=company_id, success=True, data=snake_data)

    except (requests.Timeout, requests.ConnectionError, RiskApiServerError) as exc:
        # All retries exhausted — log and return a failure result
        logger.error(
            "Enrichment failed after %d retries for company_id=%r: %s",
            MAX_RETRIES,
            company_id,
            exc,
        )
        return EnrichmentResult(
            company_id=company_id,
            success=False,
            error=f"Transient API failure after {MAX_RETRIES} retries: {exc}",
        )

    except RiskApiClientError as exc:
        # Client errors won't benefit from retry — report immediately
        logger.error(
            "Enrichment client error for company_id=%r: %s",
            company_id,
            exc,
        )
        return EnrichmentResult(
            company_id=company_id,
            success=False,
            error=f"Non-retriable API error: {exc}",
        )
