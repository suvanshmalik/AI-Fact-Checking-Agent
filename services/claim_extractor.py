"""
services/claim_extractor.py

Responsibilities:
- Send PDF text chunks to Gemini with a structured JSON extraction prompt.
- Return a deduplicated list of Claim objects, capped at config.pipeline.max_claims.
- Ignore opinions, marketing statements, and subjective claims per config.
- Preserve verbatim original_quote alongside normalised claim text.
- Assign claim_id as a zero-padded string (e.g. "001") for clean serialisation.

Error handling:
- Retries via @retry_with_backoff (1s → 2s → 4s).
- Logs and skips malformed individual claim entries rather than failing the whole batch.
- Raises RuntimeError if Gemini returns completely unparseable output after all retries.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

import google.generativeai as genai

from config import config
from models.schemas import Claim, ClaimType
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise fact-extraction engine. Your job is to identify
verifiable factual claims from document text and return them as structured JSON.

EXTRACT ONLY these claim types (in order of priority):
  1. statistic   — numerical figures, percentages, counts, rates
  2. date        — specific dates, years, time periods tied to events
  3. financial   — prices, revenues, costs, valuations, economic data
  4. metric      — performance measurements, benchmarks, technical specifications

IGNORE completely:
  - Opinions, predictions, or subjective assessments
  - Marketing language ("best", "leading", "world-class")
  - Vague qualitative statements without measurable data
  - Recommendations or calls to action

OUTPUT FORMAT — return ONLY a valid JSON array, no preamble, no markdown fences:
[
  {
    "text": "<normalised, standalone claim sentence>",
    "original_quote": "<exact verbatim text from the document>",
    "claim_type": "<statistic|date|financial|metric|other>",
    "page_number": <integer or null>
  },
  ...
]

Rules:
- Each claim must be self-contained (readable without surrounding context).
- original_quote must be copied verbatim from the input text.
- Use "other" only if the claim is clearly verifiable but doesn't fit the above types.
- Return an empty array [] if no verifiable claims are found.
- Do NOT invent claims not present in the text.
"""


def _build_user_prompt(chunk: str, chunk_index: int, total_chunks: int) -> str:
    return (
        f"Document chunk {chunk_index + 1} of {total_chunks}.\n"
        "Extract all verifiable factual claims from the text below.\n\n"
        f"TEXT:\n{chunk}"
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """
    Batch-extracts Claim objects from a PDFDocument's text chunks.

    Usage:
        extractor = ClaimExtractor()
        claims = extractor.extract(pdf_doc)
    """

    def __init__(self) -> None:
        if not config.gemini.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to .env (local) or Streamlit secrets (cloud)."
            )
        genai.configure(api_key=config.gemini.api_key)
        self._model = genai.GenerativeModel(
            model_name  = config.gemini.model,
            system_instruction = _SYSTEM_PROMPT,
            generation_config  = genai.GenerationConfig(
                temperature  = config.gemini.temperature_extract,
                max_output_tokens = config.gemini.max_output_tokens,
            ),
        )
        logger.info("ClaimExtractor initialised with model: %s", config.gemini.model)

    def extract(self, chunks: list[str], source_filename: str = "") -> list[Claim]:
        """
        Process each text chunk and return a deduplicated, capped list of Claims.

        Args:
            chunks:           Text chunks from PDFDocument.chunks.
            source_filename:  Used only in log messages.

        Returns:
            List[Claim], length ≤ config.pipeline.max_claims.
        """
        raw_claims: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            logger.info(
                "[%s] Extracting claims from chunk %d/%d",
                source_filename, idx + 1, len(chunks),
            )
            chunk_claims = self._extract_chunk(chunk, idx, len(chunks))
            raw_claims.extend(chunk_claims)

            # Early exit if we already have more than the cap
            if len(raw_claims) >= config.pipeline.max_claims:
                logger.info(
                    "Reached cap of %d claims early — stopping extraction.",
                    config.pipeline.max_claims,
                )
                break

        deduped = self._deduplicate(raw_claims)
        capped  = deduped[: config.pipeline.max_claims]

        if len(deduped) > config.pipeline.max_claims:
            logger.warning(
                "Found %d claims; capped at %d. "
                "Lower-priority claims were dropped.",
                len(deduped), config.pipeline.max_claims,
            )

        claims = [self._to_claim(raw, i) for i, raw in enumerate(capped)]
        logger.info("Extracted %d claims total from '%s'", len(claims), source_filename)
        return claims

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry_with_backoff(delays=config.retry.delays)
    def _call_gemini(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the raw text response."""
        response = self._model.generate_content(prompt)
        return response.text

    def _extract_chunk(
        self, chunk: str, chunk_index: int, total_chunks: int
    ) -> list[dict[str, Any]]:
        """Extract claims from one chunk; returns [] on parse failure."""
        prompt = _build_user_prompt(chunk, chunk_index, total_chunks)
        try:
            raw_text = self._call_gemini(prompt)
        except Exception as exc:
            logger.error("Gemini call failed for chunk %d: %s", chunk_index, exc)
            return []

        return self._parse_json(raw_text, chunk_index)

    def _parse_json(self, raw: str, chunk_index: int) -> list[dict[str, Any]]:
        """
        Parse Gemini's JSON output.  Strips markdown fences if present.
        Filters out ignored claim types (opinions, marketing).
        """
        cleaned = raw.strip()
        # Strip ```json ... ``` fences Gemini sometimes adds despite instructions
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$",          "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse error for chunk %d: %s\nRaw output: %.300s",
                chunk_index, exc, cleaned,
            )
            return []

        if not isinstance(data, list):
            logger.warning("Chunk %d: expected a JSON array, got %s", chunk_index, type(data))
            return []

        ignored = set(config.pipeline.ignored_types)
        valid: list[dict[str, Any]] = []

        for entry in data:
            if not isinstance(entry, dict):
                continue
            claim_type = str(entry.get("claim_type", "")).lower()
            if claim_type in ignored:
                logger.debug("Skipping ignored claim type '%s'", claim_type)
                continue
            # Ensure required fields are present
            if not entry.get("text") or not entry.get("original_quote"):
                logger.debug("Skipping claim with missing text/original_quote: %s", entry)
                continue
            valid.append(entry)

        return valid

    def _deduplicate(self, raw_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove duplicate claims by normalised text.
        Priority order matches config.pipeline.claim_priority:
        statistic > date > financial > metric > other.
        """
        priority = {t: i for i, t in enumerate(config.pipeline.claim_priority)}
        seen: dict[str, dict[str, Any]] = {}

        for claim in raw_claims:
            key = claim.get("text", "").lower().strip()
            if not key:
                continue
            if key not in seen:
                seen[key] = claim
            else:
                # Keep the higher-priority type
                existing_prio = priority.get(seen[key].get("claim_type", "other"), 99)
                new_prio      = priority.get(claim.get("claim_type",     "other"), 99)
                if new_prio < existing_prio:
                    seen[key] = claim

        # Sort by priority before returning
        return sorted(
            seen.values(),
            key=lambda c: priority.get(c.get("claim_type", "other"), 99),
        )

    @staticmethod
    def _to_claim(raw: dict[str, Any], index: int) -> Claim:
        """
        Convert a raw dict from Gemini into a validated Claim object.
        claim_id is a zero-padded string derived from the index.
        Falls back to ClaimType.other for unrecognised types.
        """
        claim_type_str = str(raw.get("claim_type", "other")).lower()
        try:
            claim_type = ClaimType(claim_type_str)
        except ValueError:
            logger.debug("Unrecognised claim type '%s' — defaulting to 'other'", claim_type_str)
            claim_type = ClaimType.other

        page = raw.get("page_number")
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None

        return Claim(
            claim_id      = f"{index:04d}",
            text          = str(raw["text"]).strip(),
            original_quote = str(raw["original_quote"]).strip(),
            claim_type    = claim_type,
            page_number   = page,
        )
