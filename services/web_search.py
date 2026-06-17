"""
services/web_search.py

Responsibilities:
- Run a Tavily web search for each Claim.
- Parse raw Tavily results into SearchResult objects.
- Pre-populate `is_authoritative` based on trusted_domains from config.
- Pass `published_date` as Optional[datetime] (None when Tavily omits it).
- Does NOT rank results — that is entirely evidence_ranker.py's job.

Error handling:
- Retries via @retry_with_backoff.
- Returns an empty list on unrecoverable failure so verification can still run
  (verifier.py will mark such claims UNVERIFIABLE due to no evidence).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from tavily import TavilyClient

from config import config
from models.schemas import Claim, SearchResult
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_query(claim: Claim) -> str:
    """
    Construct a focused Tavily search query from a Claim.
    Prefixes the claim type to guide results (e.g. "statistic: global GDP grew 3.2% in 2023").
    """
    prefix_map = {
        "statistic": "verify statistic",
        "date":      "verify date fact",
        "financial": "verify financial figure",
        "metric":    "verify technical metric",
        "other":     "fact check",
    }
    prefix = prefix_map.get(claim.claim_type.value, "fact check")
    return f"{prefix}: {claim.text}"


# ---------------------------------------------------------------------------
# Domain authority checker
# ---------------------------------------------------------------------------

def _is_authoritative(url: str, trusted_domains: tuple[str, ...]) -> bool:
    """
    Return True if the URL's registered domain matches any entry in
    trusted_domains (exact suffix match, so 'reuters.com' catches
    'www.reuters.com' and 'mobile.reuters.com').
    """
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().lstrip("www.")
        return any(hostname == td or hostname.endswith(f".{td}") for td in trusted_domains)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------

def _parse_date(raw: str | None) -> datetime | None:
    """
    Attempt to parse Tavily's published_date string into a UTC datetime.
    Returns None on any parse failure — never raises.
    """
    if not raw:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.debug("Could not parse date string: '%s'", raw)
    return None


# ---------------------------------------------------------------------------
# Web searcher
# ---------------------------------------------------------------------------

class WebSearcher:
    """
    Runs Tavily searches for individual claims.

    Usage:
        searcher = WebSearcher()
        results = searcher.search(claim)
    """

    def __init__(self) -> None:
        if not config.tavily.api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Add it to .env (local) or Streamlit secrets (cloud)."
            )
        self._client = TavilyClient(api_key=config.tavily.api_key)
        self._trusted = config.tavily.trusted_domains
        logger.info(
            "WebSearcher initialised. Trusted domains: %d", len(self._trusted)
        )

    def search(self, claim: Claim) -> list[SearchResult]:
        """
        Search for evidence for a single Claim.

        Returns:
            List of SearchResult objects (unranked, up to config.tavily.max_results).
            Returns [] on failure (logged as error).
        """
        query = _build_query(claim)
        logger.info("Tavily search [claim %s]: %s", claim.claim_id, query[:120])

        try:
            raw_results = self._call_tavily(query)
        except Exception as exc:
            logger.error(
                "Tavily search failed for claim %s after retries: %s",
                claim.claim_id, exc,
            )
            return []

        results = [self._parse_result(r) for r in raw_results]
        logger.info(
            "Claim %s: %d raw result(s) returned",
            claim.claim_id, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry_with_backoff(delays=config.retry.delays)
    def _call_tavily(self, query: str) -> list[dict]:
        """
        Make the Tavily API call. Decorated with retry_with_backoff.
        max_results fetches more than needed so evidence_ranker has
        a larger pool to select from.
        """
        response = self._client.search(
            query        = query,
            max_results  = config.tavily.max_results,
            search_depth = config.tavily.search_depth,
            include_answer = False,
            include_raw_content = False,
        )
        return response.get("results", [])

    def _parse_result(self, raw: dict) -> SearchResult:
        """Convert a raw Tavily result dict into a SearchResult."""
        url   = str(raw.get("url",     ""))
        title = str(raw.get("title",   ""))
        # Tavily returns 'content' as the snippet field
        snippet = str(raw.get("content", raw.get("snippet", "")))

        return SearchResult(
            url              = url,
            title            = title,
            snippet          = snippet,
            relevance_score  = 0.0,    # populated by evidence_ranker
            is_authoritative = _is_authoritative(url, self._trusted),
            published_date   = _parse_date(raw.get("published_date")),
        )
