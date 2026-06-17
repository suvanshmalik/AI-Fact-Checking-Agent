"""
utils/helpers.py — Shared utilities: retry decorator, text helpers.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(delays: tuple[float, ...] = (1.0, 2.0, 4.0)) -> Callable[[F], F]:
    """
    Decorator: retry a function up to len(delays) extra times on exception.
    Waits `delays[attempt]` seconds between each retry.
    Logs each failure. Re-raises the last exception if all retries are exhausted.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt, delay in enumerate([0.0, *delays]):
                if attempt > 0:
                    logger.warning(
                        "Retry %d/%d for %s — sleeping %.1fs",
                        attempt, len(delays), func.__qualname__, delay,
                    )
                    time.sleep(delay)
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.error(
                        "%s failed (attempt %d): %s",
                        func.__qualname__, attempt + 1, exc,
                    )
            raise RuntimeError(
                f"{func.__qualname__} failed after {len(delays) + 1} attempts"
            ) from last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


def chunk_text(text: str, max_chars: int = 12_000) -> list[str]:
    """
    Split `text` into chunks of at most `max_chars` characters,
    breaking on paragraph boundaries where possible.
    Used by pdf_processor to stay within Gemini's context window.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def extract_keywords(text: str) -> set[str]:
    """
    Return a set of lowercased alphanumeric tokens (≥3 chars) from `text`.
    Used by evidence_ranker for keyword-overlap scoring.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "day", "get", "has",
        "him", "his", "how", "its", "may", "new", "now", "old", "see",
        "two", "who", "did", "let", "put", "say", "she", "too", "use",
        "that", "this", "with", "from", "have", "been", "were", "will",
        "said", "each", "which", "their", "there", "what", "when", "where",
    }
    return set(tokens) - stopwords
