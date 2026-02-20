"""
Tests for LLM extraction.

Key technique: MockLLMClient with a side_effect list.
  Each call to complete() pops the next response from the list.
  This lets us simulate: clean success, partial extraction, parse failure
  + recovery, and complete failure — all without an API key.

What we verify:
  - The extractor never raises (always returns ExtractionResult)
  - status=SUCCESS when all required fields are present
  - status=PARTIAL when some fields are missing
  - status=FAILED after all retries are exhausted
  - The LLM is retried with a stronger prompt on parse failure
  - Dirty revenue values ("$4.5M", "$1,000,000") are coerced correctly
  - The mock is only called as many times as needed (no wasted API calls)
"""

import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor import LLMClient, SubmissionExtractor, _missing_fields
from models import ExtractedSubmission, ExtractionStatus


# --------------------------------------------------------------------------- #
# Test double                                                                  #
# --------------------------------------------------------------------------- #


class MockLLMClient(LLMClient):
    """
    Controllable LLM stub.
    Pass responses as a list — each call() pops the next one.
    Uses a MagicMock internally so we can assert call counts.
    """

    def __init__(self, responses: list[str]) -> None:
        self._mock = MagicMock(side_effect=responses)

    def complete(self, prompt: str) -> str:
        return self._mock(prompt)

    @property
    def call_count(self) -> int:
        return self._mock.call_count


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _json_response(**kwargs) -> str:
    """Build a clean JSON string the extractor would receive from an LLM."""
    return json.dumps(kwargs)


FULL_RESPONSE = _json_response(
    company_name="Acme Corp",
    revenue=5_000_000,
    industry="Retail",
    state="NY",
    zip_code="10001",
)

PARTIAL_RESPONSE = _json_response(
    company_name="Acme Corp",
    revenue=5_000_000,
    industry=None,   # LLM could not determine industry
    state="NY",
    zip_code=None,
)


# --------------------------------------------------------------------------- #
# 1. Happy path                                                                #
# --------------------------------------------------------------------------- #


class TestSuccessfulExtraction:
    def test_all_fields_extracted_returns_success(self):
        client = MockLLMClient([FULL_RESPONSE])
        result = SubmissionExtractor(client).extract("some text")

        assert result.status == ExtractionStatus.SUCCESS
        assert result.extracted is not None
        assert result.extracted.company_name == "Acme Corp"
        assert result.extracted.revenue == Decimal("5000000")
        assert result.missing_fields == []

    def test_llm_called_exactly_once_on_success(self):
        client = MockLLMClient([FULL_RESPONSE])
        SubmissionExtractor(client).extract("some text")
        assert client.call_count == 1

    def test_state_is_uppercased(self):
        response = _json_response(
            company_name="Acme", revenue=100_000, industry="Retail", state="ny"
        )
        client = MockLLMClient([response])
        result = SubmissionExtractor(client).extract("text")
        assert result.extracted.state == "NY"


# --------------------------------------------------------------------------- #
# 2. Partial extraction                                                        #
# --------------------------------------------------------------------------- #


class TestPartialExtraction:
    def test_missing_optional_field_is_partial_not_failed(self):
        client = MockLLMClient([PARTIAL_RESPONSE])
        result = SubmissionExtractor(client).extract("text")

        assert result.status == ExtractionStatus.PARTIAL
        assert "industry" in result.missing_fields
        assert result.extracted is not None           # we still have what was found
        assert result.extracted.company_name == "Acme Corp"

    def test_zip_code_missing_does_not_affect_status(self):
        """zip_code is not in REQUIRED_FOR_TRIAGE — its absence is not PARTIAL."""
        response = _json_response(
            company_name="Acme", revenue=100_000, industry="Retail",
            state="CA", zip_code=None
        )
        client = MockLLMClient([response])
        result = SubmissionExtractor(client).extract("text")
        assert result.status == ExtractionStatus.SUCCESS
        assert result.missing_fields == []


# --------------------------------------------------------------------------- #
# 3. Dirty revenue values                                                      #
# --------------------------------------------------------------------------- #


class TestRevenueCoercion:
    @pytest.mark.parametrize(
        "raw_revenue, expected",
        [
            (5_000_000,    Decimal("5000000")),      # plain int
            ("$5,000,000", Decimal("5000000")),      # dollar + commas
            ("4.5M",       Decimal("4500000")),      # shorthand
            ("500k",       Decimal("500000")),       # k-suffix
            ("not-a-number", None),                  # hallucination → None
            (None,          None),                   # LLM returned null
        ],
    )
    def test_revenue_coercion(self, raw_revenue, expected):
        response = _json_response(
            company_name="Acme", revenue=raw_revenue, industry="Retail", state="CA"
        )
        client = MockLLMClient([response])
        result = SubmissionExtractor(client).extract("text")

        if expected is None:
            assert result.extracted.revenue is None
        else:
            assert result.extracted.revenue == expected


# --------------------------------------------------------------------------- #
# 4. Retry behaviour                                                           #
# --------------------------------------------------------------------------- #


class TestRetryBehaviour:
    def test_malformed_json_triggers_retry_then_succeeds(self):
        """
        First call returns prose (not JSON).
        Second call returns valid JSON.
        Result should be SUCCESS and LLM called twice.
        """
        client = MockLLMClient(["I cannot extract that.", FULL_RESPONSE])
        result = SubmissionExtractor(client, max_retries=2).extract("text")

        assert result.status == ExtractionStatus.SUCCESS
        assert client.call_count == 2

    def test_retry_prompt_contains_stronger_formatting_instruction(self):
        """On attempt 2+, the prompt should include the REMINDER suffix."""
        client = MockLLMClient(["bad json", FULL_RESPONSE])
        SubmissionExtractor(client, max_retries=2).extract("text")

        first_prompt  = client._mock.call_args_list[0][0][0]
        second_prompt = client._mock.call_args_list[1][0][0]

        assert "REMINDER" not in first_prompt
        assert "REMINDER" in second_prompt

    def test_markdown_wrapped_json_is_parsed_correctly(self):
        """LLMs often wrap JSON in ```json ... ``` — the extractor must strip it."""
        wrapped = f"```json\n{FULL_RESPONSE}\n```"
        client = MockLLMClient([wrapped])
        result = SubmissionExtractor(client).extract("text")
        assert result.status == ExtractionStatus.SUCCESS

    def test_all_retries_exhausted_returns_failed_status(self):
        """Three bad responses → status=FAILED, never raises."""
        client = MockLLMClient(["bad", "also bad", "still bad"])
        result = SubmissionExtractor(client, max_retries=3).extract("text")

        assert result.status == ExtractionStatus.FAILED
        assert result.error is not None
        assert "3" in result.error          # should mention the attempt count
        assert client.call_count == 3

    def test_extractor_never_raises(self):
        """The extractor contract: it always returns ExtractionResult, never raises."""
        client = MockLLMClient(["not json at all"])
        result = SubmissionExtractor(client, max_retries=1).extract("text")
        assert isinstance(result, __import__("models").ExtractionResult)


# --------------------------------------------------------------------------- #
# 5. _missing_fields helper                                                    #
# --------------------------------------------------------------------------- #


class TestMissingFields:
    def test_returns_empty_when_all_present(self):
        sub = ExtractedSubmission(
            company_name="X", revenue=Decimal("1"), industry="Y", state="CA"
        )
        assert _missing_fields(sub) == []

    def test_returns_missing_field_names(self):
        sub = ExtractedSubmission(company_name="X", revenue=None, industry=None, state="CA")
        missing = _missing_fields(sub)
        assert set(missing) == {"revenue", "industry"}
