"""
Pydantic v2 data models for the AI claim verification pipeline.
All inter-module data is typed — no raw dicts passed between services.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Claim categories in descending verification priority."""

    statistic = "statistic"
    date = "date"
    financial = "financial"
    metric = "metric"
    other = "other"


class Verdict(str, Enum):
    """Final verdict shown to the user."""

    VERIFIED = "Verified"
    INACCURATE = "Inaccurate"
    FALSE = "False"


class VerificationStatus(str, Enum):
    """Processing state of a claim."""

    pending = "Pending"
    processing = "Processing"
    complete = "Complete"
    failed = "Failed"


# ---------------------------------------------------------------------------
# Claim Model
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    """A single claim extracted from the PDF."""

    claim_id: str

    text: str = Field(
        ...,
        min_length=5
    )

    original_quote: str = Field(
        ...,
        description="Original quote extracted from PDF"
    )

    claim_type: ClaimType

    page_number: Optional[int] = None

    status: VerificationStatus = VerificationStatus.pending


# ---------------------------------------------------------------------------
# Search Result Model
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """A search result returned from Tavily."""

    url: str
    title: str
    snippet: str

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    is_authoritative: bool = False

    published_date: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Verified Claim Model
# ---------------------------------------------------------------------------

class VerifiedClaim(BaseModel):
    """Claim after evidence-based verification."""

    claim: Claim

    verdict: Verdict

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )

    evidence: list[SearchResult] = Field(
        default_factory=list
    )

    explanation: str

    corrected_fact: Optional[str] = None

    status: VerificationStatus = VerificationStatus.complete

    _HIGH_THRESHOLD: ClassVar[float] = 0.85
    _MEDIUM_THRESHOLD: ClassVar[float] = 0.60

    @model_validator(mode="after")
    def validate_corrected_fact(self) -> "VerifiedClaim":
        """
        Verified claims should not contain corrected facts.
        """

        if isinstance(self.corrected_fact, str):
            cleaned = self.corrected_fact.strip()
            self.corrected_fact = cleaned if cleaned else None

        if self.verdict == Verdict.VERIFIED:
            self.corrected_fact = None

        return self

    @property
    def confidence_label(self) -> str:
        if self.confidence >= self._HIGH_THRESHOLD:
            return "High"

        if self.confidence >= self._MEDIUM_THRESHOLD:
            return "Medium"

        return "Low"


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

class SummaryStats(BaseModel):
    """Aggregate statistics for all verified claims."""

    total: int = 0

    verified: int = 0

    inaccurate: int = 0

    false: int = 0

    failed: int = 0

    avg_confidence: float = 0.0

    @classmethod
    def from_verified_claims(
        cls,
        claims: list[VerifiedClaim]
    ) -> "SummaryStats":

        if not claims:
            return cls()

        completed_claims = [
            claim
            for claim in claims
            if claim.status == VerificationStatus.complete
        ]

        confidences = (
            [claim.confidence for claim in completed_claims]
            if completed_claims
            else [0.0]
        )

        return cls(
            total=len(claims),

            verified=sum(
                1
                for claim in completed_claims
                if claim.verdict == Verdict.VERIFIED
            ),

            inaccurate=sum(
                1
                for claim in completed_claims
                if claim.verdict == Verdict.INACCURATE
            ),

            false=sum(
                1
                for claim in completed_claims
                if claim.verdict == Verdict.FALSE
            ),

            failed=sum(
                1
                for claim in claims
                if claim.status == VerificationStatus.failed
            ),

            avg_confidence=round(
                sum(confidences) / len(confidences),
                3
            ),
        )


# ---------------------------------------------------------------------------
# Report Model
# ---------------------------------------------------------------------------

class Report(BaseModel):
    """Top-level report object."""

    source_filename: str

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    claims: list[VerifiedClaim] = Field(
        default_factory=list
    )

    stats: SummaryStats = Field(
        default_factory=SummaryStats
    )

    def refresh_stats(self) -> None:
        self.stats = SummaryStats.from_verified_claims(
            self.claims
        )
