"""
services/pdf_processor.py

Responsibilities:
- Extract raw text from a PDF using PyMuPDF (fitz).
- Detect scanned/image-only PDFs and warn rather than silently fail.
- Return text chunked to stay within Gemini's context window.
- Expose page-level metadata (page_count, char_count_per_page).

Errors:
- Raises ValueError for password-protected PDFs.
- Raises RuntimeError for scanned PDFs with < MIN_CHARS_THRESHOLD characters
  extracted (signals to the caller that OCR would be needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF

from utils.helpers import chunk_text

logger = logging.getLogger(__name__)

# If total extracted chars is below this, the PDF is likely scanned/image-only.
_MIN_CHARS_THRESHOLD = 200
# Maximum characters per chunk sent to Gemini claim extractor.
_CHUNK_MAX_CHARS = 12_000


@dataclass
class PDFDocument:
    """Structured output of the PDF processing step."""
    filename:      str
    page_count:    int
    total_chars:   int
    full_text:     str                      # complete extracted text
    chunks:        list[str]                # text split for Gemini
    chars_per_page: list[int] = field(default_factory=list)
    is_scanned:    bool = False             # True → warn user in UI
    warnings:      list[str] = field(default_factory=list)


class PDFProcessor:
    """
    Extracts text from a PDF file or file-like object.

    Usage:
        processor = PDFProcessor()
        doc = processor.process(path_or_bytes, filename="paper.pdf")
    """

    def process(
        self,
        source: str | Path | bytes | BinaryIO,
        filename: str = "document.pdf",
    ) -> PDFDocument:
        """
        Extract text from `source`.

        Args:
            source: File path (str/Path), raw bytes, or a file-like object.
            filename: Display name used in PDFDocument and log messages.

        Returns:
            PDFDocument with extracted text, chunks, and metadata.

        Raises:
            ValueError: If the PDF is password-protected.
            RuntimeError: If PyMuPDF cannot open the source.
        """
        logger.info("Opening PDF: %s", filename)
        pdf = self._open(source)

        try:
            return self._extract(pdf, filename)
        finally:
            pdf.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open(self, source: str | Path | bytes | BinaryIO) -> fitz.Document:
        """Open a PDF from various source types."""
        try:
            if isinstance(source, (str, Path)):
                pdf = fitz.open(str(source))
            elif isinstance(source, bytes):
                pdf = fitz.open(stream=source, filetype="pdf")
            else:
                # File-like object (Streamlit UploadedFile)
                raw = source.read()
                pdf = fitz.open(stream=raw, filetype="pdf")
        except Exception as exc:
            raise RuntimeError(f"PyMuPDF could not open the PDF: {exc}") from exc

        if pdf.is_encrypted:
            pdf.close()
            raise ValueError(
                "PDF is password-protected. Please provide an unlocked PDF."
            )
        return pdf

    def _extract(self, pdf: fitz.Document, filename: str) -> PDFDocument:
        """Extract text page by page and build PDFDocument."""
        page_texts: list[str] = []
        chars_per_page: list[int] = []
        warnings: list[str] = []

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")          # type: ignore[attr-defined]
            page_texts.append(text)
            chars_per_page.append(len(text))
            logger.debug("Page %d: %d chars", page_num, len(text))

        full_text = "\n\n".join(page_texts).strip()
        total_chars = len(full_text)
        is_scanned = total_chars < _MIN_CHARS_THRESHOLD

        if is_scanned:
            msg = (
                f"Very little text extracted ({total_chars} chars). "
                "This PDF may be scanned or image-only. "
                "Results may be incomplete without OCR preprocessing."
            )
            warnings.append(msg)
            logger.warning(msg)
        else:
            logger.info(
                "Extracted %d chars across %d pages from '%s'",
                total_chars, pdf.page_count, filename,
            )

        chunks = chunk_text(full_text, max_chars=_CHUNK_MAX_CHARS)
        logger.info("Split into %d chunk(s) for Gemini", len(chunks))

        return PDFDocument(
            filename      = filename,
            page_count    = pdf.page_count,
            total_chars   = total_chars,
            full_text     = full_text,
            chunks        = chunks,
            chars_per_page = chars_per_page,
            is_scanned    = is_scanned,
            warnings      = warnings,
        )
