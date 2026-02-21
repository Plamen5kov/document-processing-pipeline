"""
Tests for the enrichment client and batch enricher.

Key testing concepts:
  - responses library: declarative HTTP mocking — no monkey-patching needed.
  - unittest.mock.patch: swap out the retry-wrapped function to test retry
    behaviour without actually sleeping (speeds up tests enormously).
  - Side-effect lists: simulate "fail twice, then succeed" sequences.
  - Testing concurrency: verify that results are present, not order-dependent
    (as_completed returns in arbitrary order).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests
import responses as responses_lib  # aliased to avoid name clash with pytest fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import (
    RISK_API_BASE,
    EnrichmentResult,
    RiskApiClientError,
    RiskApiServerError,
    _fetch_risk_data,
    enrich_submission,
)
from enricher import BatchEnrichmentReport, enrich_batch
from mapper import normalise_keys


# --------------------------------------------------------------------------- #
# 1. camelCase → snake_case mapper                                             #
# --------------------------------------------------------------------------- #


class TestMapper:
    def test_simple_camel_case(self):
        assert normalise_keys({"riskScore": 42}) == {"risk_score": 42}

    def test_nested_dict(self):
        result = normalise_keys({"companyId": "X", "riskData": {"lossRatio": 0.5}})
        assert result == {"company_id": "X", "risk_data": {"loss_ratio": 0.5}}

    def test_list_of_dicts(self):
        result = normalise_keys([{"riskScore": 1}, {"riskScore": 2}])
        assert result == [{"risk_score": 1}, {"risk_score": 2}]

    def test_scalar_passthrough(self):
        assert normalise_keys("hello") == "hello"  # type: ignore[arg-type]

    def test_already_snake_case_unchanged(self):
        assert normalise_keys({"risk_score": 1}) == {"risk_score": 1}

    def test_https_abbreviation(self):
        assert normalise_keys({"HTTPSEnabled": True}) == {"https_enabled": True}


# --------------------------------------------------------------------------- #
# 2. HTTP client — happy path (using `responses` library)                      #
# --------------------------------------------------------------------------- #


class TestFetchRiskDataHappyPath:
    @responses_lib.activate
    def test_successful_response_returns_parsed_json(self):
        """
        `responses.activate` intercepts outgoing requests.get() calls and
        returns the registered mock — no real network traffic.
        """
        company_id = "ACME-001"
        responses_lib.add(
            responses_lib.GET,
            f"{RISK_API_BASE}/companies/{company_id}/risk",
            json={"riskScore": 72, "industryCode": "RET"},
            status=200,
        )

        result = _fetch_risk_data.__wrapped__(company_id)  # __wrapped__ bypasses tenacity

        assert result == {"riskScore": 72, "industryCode": "RET"}

    @responses_lib.activate
    def test_enrich_submission_normalises_keys(self):
        company_id = "ACME-002"
        responses_lib.add(
            responses_lib.GET,
            f"{RISK_API_BASE}/companies/{company_id}/risk",
            json={"riskScore": 55, "claimsHistory": []},
            status=200,
        )

        result = enrich_submission(company_id)

        assert result.success is True
        assert result.data == {"risk_score": 55, "claims_history": []}


# --------------------------------------------------------------------------- #
# 3. Retry behaviour — simulated failures                                      #
# --------------------------------------------------------------------------- #


class TestRetryBehaviour:
    def test_retries_on_server_error_then_succeeds(self):
        """
        Simulate: 500 → 500 → 200.  The function should succeed on the third
        attempt.  We patch `requests.get` so no real HTTP happens and no
        real sleeping occurs.
        """
        good_response = MagicMock()
        good_response.status_code = 200
        good_response.json.return_value = {"riskScore": 80}

        server_error_response = MagicMock()
        server_error_response.status_code = 503

        with patch("client.requests.get") as mock_get:
            # side_effect list: each call pops the next value
            mock_get.side_effect = [
                server_error_response,
                server_error_response,
                good_response,
            ]
            # Patch tenacity's sleep so the test doesn't actually wait
            with patch("tenacity.nap.time.sleep"):
                result = _fetch_risk_data("ACME-003")

        assert result == {"riskScore": 80}
        assert mock_get.call_count == 3

    def test_all_retries_exhausted_raises(self):
        """After MAX_RETRIES failures, the last exception must propagate."""
        server_error_response = MagicMock()
        server_error_response.status_code = 500

        with patch("client.requests.get", return_value=server_error_response):
            with patch("tenacity.nap.time.sleep"):
                with pytest.raises(RiskApiServerError):
                    _fetch_risk_data("ACME-FAIL")

    def test_timeout_is_retriable(self):
        with patch("client.requests.get", side_effect=requests.Timeout):
            with patch("tenacity.nap.time.sleep"):
                result = enrich_submission("ACME-TIMEOUT")

        assert result.success is False
        assert "retries" in result.error

    def test_client_error_is_not_retried(self):
        """404 = our bug; retrying won't fix it.  Must fail immediately."""
        bad_response = MagicMock()
        bad_response.status_code = 404

        with patch("client.requests.get", return_value=bad_response) as mock_get:
            result = enrich_submission("ACME-404")

        assert result.success is False
        assert mock_get.call_count == 1  # no retry


# --------------------------------------------------------------------------- #
# 4. Batch enrichment with ThreadPoolExecutor                                  #
# --------------------------------------------------------------------------- #


class TestBatchEnrichment:
    def test_all_succeed(self):
        """All IDs succeed → report.failed is empty."""

        def fake_enrich(company_id: str) -> EnrichmentResult:
            return EnrichmentResult(company_id=company_id, success=True, data={})

        with patch("enricher.enrich_submission", side_effect=fake_enrich):
            report = enrich_batch(["A", "B", "C"], max_workers=2)

        assert len(report.succeeded) == 3
        assert len(report.failed) == 0
        assert report.success_rate == 1.0

    def test_partial_failures_are_captured(self):
        """One failure must not prevent other IDs from being processed."""

        def fake_enrich(company_id: str) -> EnrichmentResult:
            if company_id == "BAD":
                return EnrichmentResult(company_id=company_id, success=False, error="x")
            return EnrichmentResult(company_id=company_id, success=True, data={})

        with patch("enricher.enrich_submission", side_effect=fake_enrich):
            report = enrich_batch(["A", "BAD", "C"], max_workers=2)

        assert len(report.succeeded) == 2
        assert len(report.failed) == 1
        assert report.failed[0].company_id == "BAD"

    def test_success_rate_calculation(self):
        succeeded = [EnrichmentResult(company_id="A", success=True)]
        failed = [EnrichmentResult(company_id="B", success=False)]
        report = BatchEnrichmentReport(succeeded=succeeded, failed=failed)
        assert report.success_rate == 0.5

    def test_empty_batch_returns_empty_report(self):
        report = enrich_batch([], max_workers=2)
        assert report.succeeded == []
        assert report.failed == []
