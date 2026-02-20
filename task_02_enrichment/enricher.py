"""
Batch enrichment using ThreadPoolExecutor.

The concept: why threads for I/O-bound work?
───────────────────────────────────────────
When a thread calls requests.get() it blocks waiting for bytes from the
network.  The GIL (Global Interpreter Lock) is RELEASED during that I/O
wait, so other threads can run Python code.  This means threading gives
real concurrency for network calls — unlike CPU-bound tasks where threads
fight over the GIL.

Compare:
  Sequential (10 submissions × 2s each)  = ~20s
  ThreadPoolExecutor(max_workers=5)       = ~4s   (5 in-flight at once)
  asyncio + aiohttp                       = ~2s   (all in-flight at once, no threads)

Rule of thumb:
  - I/O-bound + existing sync library (requests)  → ThreadPoolExecutor ✓
  - I/O-bound + greenfield                        → asyncio + aiohttp ✓
  - CPU-bound (parsing, ML inference)             → ProcessPoolExecutor ✓
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from client import EnrichmentResult, enrich_submission

logger = logging.getLogger(__name__)

# Number of concurrent outbound HTTP connections.
# Keep this below the Risk API's documented rate limit.
DEFAULT_MAX_WORKERS = 5


@dataclass
class BatchEnrichmentReport:
    succeeded: list[EnrichmentResult] = field(default_factory=list)
    failed: list[EnrichmentResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = len(self.succeeded) + len(self.failed)
        return len(self.succeeded) / total if total else 0.0


def enrich_batch(
    company_ids: Sequence[str],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> BatchEnrichmentReport:
    """
    Enrich multiple submissions concurrently using a thread pool.

    Args:
        company_ids:  IDs to enrich.
        max_workers:  Thread pool size.  Tune based on API rate limits.

    Returns:
        A BatchEnrichmentReport with succeeded / failed split.

    Design notes:
      - as_completed() processes results as they arrive, not in submission
        order — faster than waiting for the slowest call in each "round".
      - The thread pool is used as a context manager — threads are cleaned
        up even if an exception escapes (defensive resource management).
      - Individual failures are recorded in the report; they never abort the
        whole batch.
    """
    report = BatchEnrichmentReport()

    # ThreadPoolExecutor submits each call to a thread from the pool.
    # max_workers caps how many threads run concurrently.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Build a future → company_id mapping so we can log which call finished
        future_to_id = {
            pool.submit(enrich_submission, company_id): company_id
            for company_id in company_ids
        }

        # as_completed yields each future as soon as its thread finishes —
        # no wasted time waiting for slower calls to catch up.
        for future in as_completed(future_to_id):
            company_id = future_to_id[future]
            try:
                result: EnrichmentResult = future.result()
            except Exception as exc:
                # enrich_submission() is designed not to raise, but we guard
                # here anyway — defensive programming for production.
                logger.exception(
                    "Unexpected error enriching company_id=%r: %s", company_id, exc
                )
                result = EnrichmentResult(
                    company_id=company_id,
                    success=False,
                    error=f"Unexpected: {exc}",
                )

            if result.success:
                report.succeeded.append(result)
            else:
                report.failed.append(result)

    logger.info(
        "Batch enrichment complete: %d succeeded, %d failed (%.0f%% success rate)",
        len(report.succeeded),
        len(report.failed),
        report.success_rate * 100,
    )
    return report


# --------------------------------------------------------------------------- #
# Generator variant — memory-efficient for very large batches                  #
# --------------------------------------------------------------------------- #


def enrich_stream(
    company_ids: Sequence[str],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Iterator[EnrichmentResult]:
    """
    Same as enrich_batch but yields results one by one.

    Use this when the downstream consumer can start processing before the
    entire batch is done (pipeline pattern), or when the batch is too large
    to hold all results in memory at once.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(enrich_submission, cid): cid for cid in company_ids
        }
        for future in as_completed(future_to_id):
            company_id = future_to_id[future]
            try:
                yield future.result()
            except Exception as exc:
                logger.exception("Unexpected error for %r: %s", company_id, exc)
                yield EnrichmentResult(
                    company_id=company_id, success=False, error=str(exc)
                )
