"""
Tests for normalisation and duplicate detection.

Notable patterns:
  - Testing the normaliser independently from the detector (unit isolation).
  - Parametrize covers the full decision matrix for the detector.
  - One test verifies the pre-filter *is actually working* (proves efficiency
    claim, not just correctness).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import DuplicateDetector, DuplicateMatch, SubmissionRecord
from normalizer import normalise


# --------------------------------------------------------------------------- #
# 1. Normaliser unit tests                                                     #
# --------------------------------------------------------------------------- #


class TestNormalise:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Kalepa Inc.",                "kalepa"),
            ("Kalepa, Inc",               "kalepa"),
            ("KALEPA INC",                "kalepa"),
            ("Kalepa Holdings LLC",        "kalepa"),
            ("International Business Machines Corp.", "business machines"),
            ("ACME Group",                 "acme"),
            ("  Extra  Spaces  ",          "extra spaces"),
            ("Café & Boulangerie S.A.",    "café boulangerie"),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert normalise(raw) == expected

    def test_empty_string(self):
        assert normalise("") == ""

    def test_only_suffixes(self):
        # "LLC Corp Inc" — all tokens stripped — result is empty
        assert normalise("LLC Corp Inc") == ""


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def existing_records() -> list[SubmissionRecord]:
    return [
        SubmissionRecord("DB-001", "Kalepa Inc.",                  "10001"),
        SubmissionRecord("DB-002", "Acme Corporation",             "10001"),
        SubmissionRecord("DB-003", "International Business Machines Corp.", "10002"),
        SubmissionRecord("DB-004", "Goldman Sachs Group",          "10003"),
        SubmissionRecord("DB-005", "Some Unrelated Company LLC",   "90210"),
    ]


@pytest.fixture()
def detector(existing_records) -> DuplicateDetector:
    return DuplicateDetector(existing_records)


# --------------------------------------------------------------------------- #
# 2. Happy-path duplicates                                                     #
# --------------------------------------------------------------------------- #


class TestDuplicateDetection:
    @pytest.mark.parametrize(
        "company_name, zip_code, expected_duplicate, expected_matched_id",
        [
            # Exact match after normalisation
            ("Kalepa Inc",         "10001", True,  "DB-001"),
            # Punctuation variant
            ("Kalepa, Inc.",       "10001", True,  "DB-001"),
            # Case variant
            ("KALEPA INC",         "10001", True,  "DB-001"),
            # Completely different company, same ZIP — should NOT match
            ("Random Company LLC", "10001", False, None),
            # Different ZIP — pre-filter blocks it even if name matches
            ("Kalepa Inc",         "99999", False, None),
        ],
    )
    def test_detection_matrix(
        self, detector, company_name, zip_code, expected_duplicate, expected_matched_id
    ):
        candidate = SubmissionRecord("NEW-001", company_name, zip_code)
        result = detector.check(candidate)
        assert result.is_potential_duplicate is expected_duplicate
        assert result.matched_id == expected_matched_id

    def test_high_similarity_score_on_near_match(self, detector):
        candidate = SubmissionRecord("NEW-002", "Kalepa Inc", "10001")
        result = detector.check(candidate)
        assert result.similarity_score >= 90.0

    def test_match_is_returned_for_ibm_variant(self, detector):
        """Shows token_sort_ratio handles word-order differences."""
        candidate = SubmissionRecord("NEW-003", "Machines Business International", "10002")
        result = detector.check(candidate)
        assert result.is_potential_duplicate is True
        assert result.matched_id == "DB-003"


# --------------------------------------------------------------------------- #
# 3. Pre-filter effectiveness                                                  #
# --------------------------------------------------------------------------- #


class TestPrefilter:
    def test_no_candidates_in_zip_bucket_returns_no_duplicate(self):
        """
        A candidate whose ZIP prefix has no existing records must immediately
        return is_potential_duplicate=False — the fuzzy matcher never runs.
        """
        existing = [SubmissionRecord("DB-001", "Kalepa Inc", "10001")]
        detector = DuplicateDetector(existing)
        candidate = SubmissionRecord("NEW-001", "Kalepa Inc", "99999")  # different ZIP
        result = detector.check(candidate)
        assert result.is_potential_duplicate is False

    def test_index_groups_records_by_zip_prefix(self):
        """Verify the internal index only compares within the same ZIP bucket."""
        zip_a = [SubmissionRecord(f"A-{i}", f"Company {i}", "10001") for i in range(100)]
        zip_b = [SubmissionRecord(f"B-{i}", f"Company {i}", "90210") for i in range(100)]
        detector = DuplicateDetector(zip_a + zip_b)

        # Two buckets should exist in the index
        assert len(detector._index) == 2
        assert len(detector._index["10001"]) == 100
        assert len(detector._index["90210"]) == 100


# --------------------------------------------------------------------------- #
# 4. Batch check                                                               #
# --------------------------------------------------------------------------- #


class TestBatchCheck:
    def test_returns_one_result_per_candidate(self, detector):
        candidates = [
            SubmissionRecord("C1", "Kalepa Inc",  "10001"),
            SubmissionRecord("C2", "Unknown Corp", "10001"),
            SubmissionRecord("C3", "Kalepa Inc",  "99999"),
        ]
        results = detector.check_batch(candidates)
        assert len(results) == 3
        assert results[0].is_potential_duplicate is True
        assert results[1].is_potential_duplicate is False
        assert results[2].is_potential_duplicate is False

    def test_empty_candidates_returns_empty_list(self, detector):
        assert detector.check_batch([]) == []
