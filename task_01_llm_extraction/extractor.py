"""
SubmissionExtractor — the Extraction Agent.

Architecture: Extraction Agent → Pydantic (Judging) → ExtractionResult

The LLM is called via a LLMClient protocol (ABC). This means:
  - In production: swap in an Anthropic/OpenAI client.
  - In tests: use a MockLLMClient with pre-programmed responses.
  - No API key required to run the tests.

Retry strategy for LLM calls:
  Unlike HTTP retries (task_02), LLM retries serve a different purpose.
  We are not retrying because of transient infrastructure failures —
  we are retrying because LLMs are probabilistic and a different prompt
  or temperature may produce valid JSON where the first attempt did not.
  On retry, the prompt is strengthened with explicit formatting instructions.

Partial extraction handling:
  If the LLM extracts 3 of 4 required fields, the result is PARTIAL,
  not FAILED. The caller (e.g., a human review queue) can decide what
  to do with it. Discarding partial work silently is wrong.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import ValidationError

from models import (
    REQUIRED_FOR_TRIAGE,
    ExtractedSubmission,
    ExtractionResult,
    ExtractionStatus,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                              #
# --------------------------------------------------------------------------- #

_BASE_PROMPT = """\
You are an insurance data extraction assistant.

Extract the following fields from the text. If a field is not present, \
set its value to null. Return ONLY a raw JSON object — no markdown, \
no explanation, no code blocks.

Fields:
  company_name  - Name of the company being insured
  revenue       - Annual revenue as a plain number (e.g. 5000000)
  industry      - Business industry or sector
  state         - US state, preferably 2-letter code (e.g. "NY")
  zip_code      - 5-digit ZIP code if mentioned

Text:
{text}"""

# On retry: append a stronger formatting reminder
_RETRY_SUFFIX = """

REMINDER: Your previous response could not be parsed. Return ONLY a \
raw JSON object. Example: {{"company_name": "Acme", "revenue": 5000000, \
"industry": "Retail", "state": "NY", "zip_code": null}}"""


# --------------------------------------------------------------------------- #
# LLM client protocol                                                          #
# --------------------------------------------------------------------------- #


class LLMClient(ABC):
    """
    Minimal interface for an LLM text completion.

    Production implementations would wrap anthropic.Anthropic,
    openai.OpenAI, or any other provider. The extractor only depends
    on this contract, not on any specific SDK.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt and return the model's text completion."""
        ...


# --------------------------------------------------------------------------- #
# Extractor                                                                    #
# --------------------------------------------------------------------------- #


class SubmissionExtractor:
    """
    Extracts structured submission data from unstructured broker text.

    Usage:
        extractor = SubmissionExtractor(client=MyLLMClient())
        result = extractor.extract("Acme Corp, Texas, $5M revenue, Construction")

        if result.status == ExtractionStatus.SUCCESS:
            submission = result.extracted   # ExtractedSubmission — ready for triage
        elif result.status == ExtractionStatus.PARTIAL:
            # Some fields missing — send to human review queue
            print(f"Missing: {result.missing_fields}")
        else:
            # All attempts failed — log and escalate
            logger.error(result.error)
    """

    def __init__(self, client: LLMClient, max_retries: int = 2) -> None:
        self._client = client
        self._max_retries = max_retries

    def extract(self, text: str) -> ExtractionResult:
        """
        Attempt to extract submission fields from text, with retries.

        Never raises — returns ExtractionResult(status=FAILED) on all errors.
        """
        last_error: Optional[str] = None
        raw_output: Optional[str] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                raw_output = self._call_llm(text, attempt)
                extracted = self._parse_and_validate(raw_output)
                missing = _missing_fields(extracted)
                status = (
                    ExtractionStatus.PARTIAL if missing else ExtractionStatus.SUCCESS
                )
                logger.info(
                    "Extraction %s on attempt %d (missing=%s)",
                    status.value,
                    attempt,
                    missing or "none",
                )
                return ExtractionResult(
                    source_text=text,
                    status=status,
                    extracted=extracted,
                    missing_fields=missing,
                    raw_llm_output=raw_output,
                )

            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Extraction attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    last_error,
                )

        return ExtractionResult(
            source_text=text,
            status=ExtractionStatus.FAILED,
            raw_llm_output=raw_output,
            error=(
                f"All {self._max_retries} extraction attempts failed. "
                f"Last error: {last_error}"
            ),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _call_llm(self, text: str, attempt: int) -> str:
        prompt = _BASE_PROMPT.format(text=text)
        if attempt > 1:
            prompt += _RETRY_SUFFIX
        return self._client.complete(prompt)

    @staticmethod
    def _parse_and_validate(raw: str) -> ExtractedSubmission:
        """
        Parse the raw LLM string into a validated ExtractedSubmission.

        Two failure modes, both raised so the retry loop can catch them:
          json.JSONDecodeError  — LLM wrapped output in markdown or prose
          ValidationError       — Valid JSON but wrong schema
        """
        # Strip common LLM formatting artifacts: ```json...``` or ```...```
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)  # raises JSONDecodeError if not valid JSON
        return ExtractedSubmission.model_validate(data)  # raises ValidationError if wrong shape


# --------------------------------------------------------------------------- #
# Pure helper                                                                  #
# --------------------------------------------------------------------------- #


def _missing_fields(extracted: ExtractedSubmission) -> list[str]:
    """Return which required-for-triage fields were not extracted."""
    return [f for f in REQUIRED_FOR_TRIAGE if getattr(extracted, f) is None]
