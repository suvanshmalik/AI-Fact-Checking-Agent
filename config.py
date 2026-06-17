"""
config.py — Centralised configuration for the AI Fact Checker pipeline.

Resolution order for secrets: st.secrets (Streamlit Cloud) → os.environ (local .env).
All pipeline constants live here; no magic strings in service modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Secret resolver
# ---------------------------------------------------------------------------

def _resolve_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Try st.secrets first (works on Streamlit Cloud), fall back to os.environ.
    Catches all exceptions so this is safe to call during pytest/import time
    when Streamlit runtime is not active.
    """
    try:
        import streamlit as st
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GeminiConfig:
    api_key:              Optional[str] = field(default_factory=lambda: _resolve_secret("GEMINI_API_KEY"))
    # gemini-2.5-flash for both extraction and verification (Step 2 decision)
    model:                str           = "gemini-2.5-flash"
    temperature_extract:  float         = 0.1   # low temp → deterministic JSON
    temperature_verify:   float         = 0.2   # slight headroom for reasoning
    max_output_tokens:    int           = 4096


@dataclass
class TavilyConfig:
    api_key:         Optional[str] = field(default_factory=lambda: _resolve_secret("TAVILY_API_KEY"))
    max_results:     int           = 8    # fetch 8, ranker keeps top 5
    search_depth:    str           = "advanced"
    # Domains given authority boost in evidence_ranker.py
    trusted_domains: tuple[str, ...] = (
        # General reference
        "wikipedia.org",
        "investopedia.com",
        # AI / tech orgs
        "openai.com",
        "anthropic.com",
        "google.com",
        "microsoft.com",
        # Academic / government
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
        "nature.com",
        "sciencedirect.com",
        # Financial
        "bloomberg.com",
        "reuters.com",
        "sec.gov",
        "ft.com",
        # Statistics
        "statista.com",
        "ourworldindata.org",
        "data.worldbank.org",
    )


@dataclass
class RetryConfig:
    # Step 2 decision: 1s → 2s → 4s exponential backoff
    delays:      tuple[float, ...] = (1.0, 2.0, 4.0)
    max_attempts: int              = 4   # 1 initial + 3 retries


@dataclass
class PipelineConfig:
    max_claims: int = 50   # Step 2 decision (increased from 20)
    top_k_evidence: int = 5  # evidence_ranker keeps this many per claim
    # Claim types to extract, in descending priority.
    # "other" is included so the extractor can tag borderline claims
    # rather than silently dropping them.
    claim_priority: tuple[str, ...] = (
        "statistic",
        "date",
        "financial",
        "metric",
        "other",
    )
    # Claim types to SKIP — opinions and marketing language
    ignored_types: tuple[str, ...] = (
        "opinion",
        "marketing",
        "subjective",
    )


# ---------------------------------------------------------------------------
# Evidence ranker weights (must sum to 1.0)
# ---------------------------------------------------------------------------

@dataclass
class RankerConfig:
    weight_authority:        float = 0.50
    weight_keyword_overlap:  float = 0.30
    weight_recency:          float = 0.20
    # Recency decay: full score within this many days, zero beyond horizon
    recency_full_score_days: int   = 30
    recency_horizon_days:    int   = 365


# ---------------------------------------------------------------------------
# Top-level config object — import this everywhere
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    gemini:   GeminiConfig   = field(default_factory=GeminiConfig)
    tavily:   TavilyConfig   = field(default_factory=TavilyConfig)
    retry:    RetryConfig    = field(default_factory=RetryConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ranker:   RankerConfig   = field(default_factory=RankerConfig)


# Module-level singleton — import and use directly:
#   from config import config
config = AppConfig()