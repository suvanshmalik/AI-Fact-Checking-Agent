"""
services/verifier.py

Responsibilities:
- Accept a Claim and its top-K ranked SearchResult list from evidence_ranker.
- Build a structured prompt containing the claim and evidence snippets.
- Call Gemini (gemini-1.5-flash) and parse its JSON verdict response.
- Return a fully validated VerifiedClaim object.
- Mark claims as UNVERIFIABLE when no evidence is available.
- Mark claims as VerificationStatus.failed and return a stub on unrecoverable errors.

Prompt design decisions:
- Evidence is injected in descending relevance_score order (already sorted by ranker).
- Gemini is asked to return ONLY a JSON object — no markdown, no preamble.
- `corrected_fact` is explicitly excluded when verdict is VERIFIED to satisfy
  the VerifiedClaim model validator.
- Confidence is requested as a float in [0.0, 1.0]; if Gemini returns an int
  (0–10 or 0–100 scale), it is normalised to [0.0, 1.0] before validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import google.generativeai as genai

from config import config
from models.schemas import Claim, SearchResult, VerificationStatus, VerifiedClaim, Verdict
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert fact-checking assistant.

Given a claim and supporting evidence, classify the claim into EXACTLY ONE of:

1. Verified
   - Evidence clearly supports the claim.

2. Inaccurate
   - Claim contains outdated, incomplete, or partially incorrect information.

3. False
   - Evidence directly contradicts the claim or no credible evidence supports it.

Respond ONLY with valid JSON:

{
  "verdict": "Verified | Inaccurate | False",
  "confidence": 0.95,
  "explanation": "Short evidence-based explanation",
  "corrected_fact": "Correct information if needed"
}

Rules:
- confidence must be between 0 and 1.
- corrected_fact should be omitted or empty for Verified claims.
- Base decisions ONLY on supplied evidence.
- Do not output markdown.
"""


def _build_evidence_block(results: list[SearchResult]) -> str:
    """
    Format the top-K evidence items into a numbered block for the prompt.
    Evidence is already sorted descending by relevance_score (from ranker).
    """
    if not results:
        return "No web evidence available."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        auth_tag = "[AUTHORITATIVE SOURCE] " if r.is_authoritative else ""
        date_str = (
            r.published_date.strftime("%Y-%m-%d") if r.published_date else "date unknown"
        )
        lines.append(
            f"[{i}] {auth_tag}{r.title}\n"
            f"    Source: {r.url}\n"
            f"    Date:   {date_str}\n"
            f"    Score:  {r.relevance_score:.4f}\n"
            f"    Snippet: {r.snippet[:500]}"
        )
    return "\n\n".join(lines)


def _build_user_prompt(claim: Claim, results: list[SearchResult]) -> str:
    evidence_block = _build_evidence_block(results)
    return (
        f"CLAIM TO VERIFY:\n{claim.text}\n\n"
        f"CLAIM TYPE: {claim.claim_type.value}\n\n"
        f"EVIDENCE ({len(results)} source(s)):\n{evidence_block}"
    )


# ---------------------------------------------------------------------------
# Confidence normaliser
# ---------------------------------------------------------------------------

def _normalise_confidence(raw: Any) -> float:
    """
    Gemini sometimes returns confidence on a 0–10 or 0–100 scale despite
    instructions.  Detect and normalise to [0.0, 1.0].
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Could not parse confidence '%s'; defaulting to 0.5", raw)
        return 0.5

    if value > 1.0:
        if value <= 10.0:
            value = value / 10.0
        elif value <= 100.0:
            value = value / 100.0
        else:
            value = 1.0

    return round(max(0.0, min(1.0, value)), 4)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """
    Verifies a single Claim against ranked evidence using Gemini.

    Usage:
        verifier = Verifier()
        verified = verifier.verify(claim, ranked_evidence)
    """

    def __init__(self) -> None:
        if not config.gemini.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to .env (local) or Streamlit secrets (cloud)."
            )
        genai.configure(api_key=config.gemini.api_key)
        self._model = genai.GenerativeModel(
            model_name         = config.gemini.model,
            system_instruction = _SYSTEM_PROMPT,
            generation_config  = genai.GenerationConfig(
                temperature       = config.gemini.temperature_verify,
                max_output_tokens = config.gemini.max_output_tokens,
            ),
        )
        logger.info("Verifier initialised with model: %s", config.gemini.model)

    def verify(self, claim: Claim, evidence: list[SearchResult]) -> VerifiedClaim:
        """
        Verify `claim` against `evidence`.

        - Returns an UNVERIFIABLE stub when evidence is empty.
        - Returns a VerificationStatus.failed stub on unrecoverable error.
        - Never raises — always returns a VerifiedClaim so the pipeline continues.

        Args:
            claim:    The Claim to verify.
            evidence: Top-K ranked SearchResult list from EvidenceRanker.

        Returns:
            VerifiedClaim
        """
        if not evidence:
            logger.warning("Claim %s: no evidence — marking UNVERIFIABLE", claim.claim_id)
            return self._unverifiable_stub(claim, reason="No web evidence found.")

        logger.info(
            "Verifying claim %s with %d evidence item(s)", claim.claim_id, len(evidence)
        )
        prompt = _build_user_prompt(claim, evidence)

        try:
            raw_text = self._call_gemini(prompt)
        except Exception as exc:
            logger.error(
                "Gemini verify call failed for claim %s: %s", claim.claim_id, exc
            )
            return self._failed_stub(claim, str(exc))

        return self._parse_and_build(claim, evidence, raw_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry_with_backoff(delays=config.retry.delays)
    def _call_gemini(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    def _parse_and_build(
        self,
        claim: Claim,
        evidence: list[SearchResult],
        raw: str,
    ) -> VerifiedClaim:
        """Parse Gemini's JSON verdict and construct a VerifiedClaim."""
        cleaned = raw.strip()
        # Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$",          "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse error for claim %s: %s\nRaw: %.300s",
                claim.claim_id, exc, cleaned,
            )
            return self._failed_stub(claim, f"JSON parse error: {exc}")

        # Parse verdict
        verdict_str = str(
            data.get("verdict", "Inaccurate")
        ).strip().lower()
        
        if verdict_str == "verified":
            verdict = Verdict.VERIFIED
        elif verdict_str == "false":
            verdict = Verdict.FALSE
        else:
            verdict = Verdict.INACCURATE

        confidence    = _normalise_confidence(data.get("confidence", 0.5))
        explanation   = str(data.get("explanation", "")).strip() or "No explanation provided."
        corrected_raw = data.get("corrected_fact")   # may be absent (VERIFIED case)

        # Normalise corrected_fact: strip whitespace; set to None if empty or VERIFIED
        corrected: str | None = None
        if corrected_raw is not None:
            stripped = str(corrected_raw).strip()
            corrected = stripped if stripped else None

        # Model validator enforces corrected_fact is None for VERIFIED
        if verdict == Verdict.VERIFIED and corrected:
            logger.debug(
                "Gemini returned corrected_fact for a VERIFIED claim %s — clearing it.",
                claim.claim_id,
            )
            corrected = None

        try:
            result = VerifiedClaim(
                claim         = claim,
                verdict       = verdict,
                confidence    = confidence,
                evidence      = evidence,
                explanation   = explanation,
                corrected_fact = corrected,
                status        = VerificationStatus.complete,
            )
        except Exception as exc:
            logger.error(
                "VerifiedClaim validation failed for claim %s: %s",
                claim.claim_id, exc,
            )
            return self._failed_stub(claim, str(exc))

        logger.info(
            "Claim %s → %s (conf=%.2f)",
            claim.claim_id, result.verdict.value, result.confidence,
        )
        return result

        # ------------------------------------------------------------------
    # Stub factories
    # ------------------------------------------------------------------

    @staticmethod
    def _unverifiable_stub(claim: Claim, reason: str) -> VerifiedClaim:
        return VerifiedClaim(
            claim=claim,
            verdict=Verdict.INACCURATE,
            confidence=0.0,
            evidence=[],
            explanation=reason,
            corrected_fact=None,
            status=VerificationStatus.complete,
        )

    @staticmethod
    def _failed_stub(claim: Claim, reason: str) -> VerifiedClaim:
        return VerifiedClaim(
            claim=claim,
            verdict=Verdict.INACCURATE,
            confidence=0.0,
            evidence=[],
            explanation=f"Verification failed: {reason}",
            corrected_fact=None,
            status=VerificationStatus.failed,
        )
