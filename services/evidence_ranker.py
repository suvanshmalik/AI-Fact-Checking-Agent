"""
services/evidence_ranker.py

Responsibilities:
- Score each SearchResult for a claim using a weighted composite:

    score = 0.50 × authority_score
          + 0.30 × keyword_overlap_score
          + 0.20 × recency_score

- Mutate SearchResult.relevance_score in-place (it lives on the model
  so the ranker never needs a parallel data structure).
- Return the top-K results sorted by descending relevance_score.

Scoring details
---------------
authority_score (0.0–1.0):
  1.0 if SearchResult.is_authoritative (domain in trusted_domains list)
  0.0 otherwise

keyword_overlap_score (0.0–1.0):
  Jaccard similarity between keyword sets of (claim.text) and (result.snippet + title).
  Jaccard = |intersection| / |union|

recency_score (0.0–1.0):
  1.0  if published_date is within config.ranker.recency_full_score_days (default 30 days)
  0.0  if published_date is None or older than config.ranker.recency_horizon_days (default 365 days)
  Linear interpolation between the two thresholds for dates in between.

All weights live in config.ranker and must sum to 1.0.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import config
from models.schemas import Claim, SearchResult
from utils.helpers import extract_keywords

logger = logging.getLogger(__name__)

# Safety check: assert weights sum to 1.0 at module load time
_w = config.ranker
_WEIGHT_SUM = _w.weight_authority + _w.weight_keyword_overlap + _w.weight_recency
assert abs(_WEIGHT_SUM - 1.0) < 1e-6, (
    f"Ranker weights must sum to 1.0; got {_WEIGHT_SUM:.6f}"
)


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def _authority_score(result: SearchResult) -> float:
    """1.0 if the result is from a trusted domain, 0.0 otherwise."""
    return 1.0 if result.is_authoritative else 0.0


def _keyword_overlap_score(claim: Claim, result: SearchResult) -> float:
    """
    Jaccard similarity between claim keywords and result (title + snippet) keywords.
    Returns 0.0 if either set is empty.
    """
    claim_kws  = extract_keywords(claim.text)
    result_kws = extract_keywords(f"{result.title} {result.snippet}")

    if not claim_kws or not result_kws:
        return 0.0

    intersection = claim_kws & result_kws
    union        = claim_kws | result_kws
    return len(intersection) / len(union)


def _recency_score(result: SearchResult) -> float:
    """
    Linear recency decay.

    full_score_days=30  → 1.0 score
    horizon_days=365    → 0.0 score
    None published_date → 0.0 (penalise undated content)
    """
    if result.published_date is None:
        return 0.0

    now = datetime.now(tz=timezone.utc)
    # Ensure published_date is timezone-aware
    pub = result.published_date
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)

    age_days = (now - pub).days
    full  = config.ranker.recency_full_score_days
    limit = config.ranker.recency_horizon_days

    if age_days <= full:
        return 1.0
    if age_days >= limit:
        return 0.0
    # Linear interpolation: from 1.0 at full_score_days to 0.0 at horizon_days
    return 1.0 - (age_days - full) / (limit - full)


# ---------------------------------------------------------------------------
# EvidenceRanker
# ---------------------------------------------------------------------------

class EvidenceRanker:
    """
    Ranks a list of SearchResult objects for a given Claim.

    Usage:
        ranker = EvidenceRanker()
        top_evidence = ranker.rank(claim, search_results)
    """

    def __init__(self) -> None:
        self._wa = config.ranker.weight_authority
        self._wk = config.ranker.weight_keyword_overlap
        self._wr = config.ranker.weight_recency
        self._top_k = config.pipeline.top_k_evidence
        logger.info(
            "EvidenceRanker: weights authority=%.2f keyword=%.2f recency=%.2f  top_k=%d",
            self._wa, self._wk, self._wr, self._top_k,
        )

    def rank(self, claim: Claim, results: list[SearchResult]) -> list[SearchResult]:
        """
        Score and sort `results` by composite relevance score for `claim`.

        Mutates SearchResult.relevance_score in-place.
        Returns the top-K results in descending score order.

        Args:
            claim:   The Claim being verified.
            results: Raw SearchResult list from WebSearcher.

        Returns:
            List[SearchResult] of length ≤ config.pipeline.top_k_evidence.
        """
        if not results:
            logger.warning("Claim %s: no search results to rank", claim.claim_id)
            return []

        for result in results:
            score = self._composite_score(claim, result)
            result.relevance_score = round(score, 4)
            logger.debug(
                "Claim %s | %s → score=%.4f (auth=%.2f kw=%.2f rec=%.2f)",
                claim.claim_id,
                result.url[:60],
                score,
                _authority_score(result),
                _keyword_overlap_score(claim, result),
                _recency_score(result),
            )

        ranked = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        top_k  = ranked[: self._top_k]

        logger.info(
            "Claim %s: ranked %d result(s), keeping top %d (best score=%.4f)",
            claim.claim_id, len(results), len(top_k),
            top_k[0].relevance_score if top_k else 0.0,
        )
        return top_k

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _composite_score(self, claim: Claim, result: SearchResult) -> float:
        """
        Weighted composite:
            score = wa × authority + wk × keyword_overlap + wr × recency
        Clipped to [0.0, 1.0] to guard against floating point drift.
        """
        score = (
            self._wa * _authority_score(result)
            + self._wk * _keyword_overlap_score(claim, result)
            + self._wr * _recency_score(result)
        )
        return max(0.0, min(1.0, score))
