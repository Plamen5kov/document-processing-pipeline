"""
Duplicate submission detector.

The problem at scale:
  10 000 existing records × fuzzy_compare per candidate = 10 000 Levenshtein
  computations per new submission.  At a few thousand new submissions per day
  that's tens of millions of string comparisons.  Even at microseconds each,
  that's too slow.

Solution — two-phase approach:
  Phase 1 (Pre-filter):  Exact match on ZIP prefix using an inverted index
                         (a dict keyed by the first N digits of the ZIP code).
                         Reduces the candidate pool from ~10 000 to ~50–200.
  Phase 2 (Fuzzy match): Run rapidfuzz only against the pre-filtered pool.

Concept: inverted index — trading memory for lookup speed.
  Build time: O(n)    — one pass over existing records
  Query time:  O(k)   — k = candidates in the ZIP bucket, not n

This is the same idea behind database indexes, Elasticsearch shards, and
locality-sensitive hashing (LSH).

RapidFuzz scorer — token_sort_ratio:
  Sorts tokens alphabetically before comparing, making "Machines IBM" score
  the same as "IBM Machines".  Better than simple ratio() for company names
  where word order varies.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

from normalizer import normalise

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 90.0   # score in [0, 100]
ZIP_PREFIX_LENGTH = 5          # first 5 digits of the ZIP code


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubmissionRecord:
    """Lightweight record representing a submission already in the database."""
    company_id: str
    company_name: str
    zip_code: str


@dataclass
class DuplicateMatch:
    """The result of a duplicate check for one candidate."""
    candidate_id: str
    is_potential_duplicate: bool
    matched_id: Optional[str] = None
    similarity_score: float = 0.0
    matched_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Detector                                                                     #
# --------------------------------------------------------------------------- #


class DuplicateDetector:
    """
    Detects whether an incoming submission is a likely duplicate of an existing
    one, using a ZIP-prefix inverted index as a pre-filter.

    Usage:
        detector = DuplicateDetector(existing_records)
        match = detector.check(candidate)
        if match.is_potential_duplicate:
            print(f"Duplicate of {match.matched_id} (score={match.similarity_score:.1f})")
    """

    def __init__(self, existing: list[SubmissionRecord]) -> None:
        # Build the inverted index once at construction time — O(n)
        self._index: dict[str, list[SubmissionRecord]] = {}
        for record in existing:
            bucket_key = self._zip_prefix(record.zip_code)
            self._index.setdefault(bucket_key, []).append(record)

        logger.info(
            "DuplicateDetector built index: %d records across %d ZIP buckets",
            len(existing),
            len(self._index),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check(self, candidate: SubmissionRecord) -> DuplicateMatch:
        """
        Check a single candidate against the index.

        Args:
            candidate: The incoming submission to check.

        Returns:
            DuplicateMatch — always returns an object, never raises.
        """
        bucket = self._get_bucket(candidate.zip_code)
        if not bucket:
            logger.debug(
                "No records in ZIP bucket '%s' for company_id=%r",
                self._zip_prefix(candidate.zip_code),
                candidate.company_id,
            )
            return DuplicateMatch(
                candidate_id=candidate.company_id,
                is_potential_duplicate=False,
            )

        return self._fuzzy_match(candidate, bucket)

    def check_batch(self, candidates: list[SubmissionRecord]) -> list[DuplicateMatch]:
        """Check multiple candidates.  Each is independent — no short-circuits."""
        return [self.check(c) for c in candidates]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_bucket(self, zip_code: str) -> list[SubmissionRecord]:
        return self._index.get(self._zip_prefix(zip_code), [])

    @staticmethod
    def _zip_prefix(zip_code: str) -> str:
        """Return the first ZIP_PREFIX_LENGTH digits, lower-cased, stripped."""
        return (zip_code or "").strip()[:ZIP_PREFIX_LENGTH]

    def _fuzzy_match(
        self,
        candidate: SubmissionRecord,
        pool: list[SubmissionRecord],
    ) -> DuplicateMatch:
        """
        Run rapidfuzz against a pre-filtered pool.

        rapidfuzz.process.extractOne:
          - Iterates over `choices` and returns the single best match.
          - score_cutoff means it returns None if no match clears the bar.
          - We use token_sort_ratio: sorts tokens before comparing, so
            "IBM Corp" ~ "Corp IBM" scores 100.
        """
        normalised_candidate = normalise(candidate.company_name)

        # Build a mapping: normalised_name → original record
        # (a dict comprehension — if two records normalise to the same string,
        #  the last one wins, which is acceptable for deduplication purposes)
        choices: dict[str, SubmissionRecord] = {
            normalise(r.company_name): r for r in pool
        }

        match = process.extractOne(
            normalised_candidate,
            choices.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=SIMILARITY_THRESHOLD,
        )

        if match is None:
            return DuplicateMatch(
                candidate_id=candidate.company_id,
                is_potential_duplicate=False,
            )

        matched_normalised_name, score, _ = match
        matched_record = choices[matched_normalised_name]

        logger.info(
            "Potential duplicate: '%s' → '%s' (score=%.1f)",
            candidate.company_name,
            matched_record.company_name,
            score,
        )

        return DuplicateMatch(
            candidate_id=candidate.company_id,
            is_potential_duplicate=True,
            matched_id=matched_record.company_id,
            similarity_score=score,
            matched_name=matched_record.company_name,
        )
