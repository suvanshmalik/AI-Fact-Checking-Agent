"""
services/report_generator.py

Generates downloadable CSV and PDF reports from a Report object.
Used by Streamlit download buttons in app.py.
"""

from __future__ import annotations

from io import BytesIO
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)

from models.schemas import Report


class ReportGenerator:
    """
    Generate CSV and PDF reports from a Report object.
    """

    @staticmethod
    def generate_csv(report: Report) -> bytes:
        """
        Convert report into CSV bytes.
        """

        rows = []

        for item in report.claims:
            rows.append(
                {
                    "Claim ID": item.claim.claim_id,
                    "Claim": item.claim.text,
                    "Verdict": item.verdict.value,
                    "Confidence": item.confidence,
                    "Explanation": item.explanation,
                    "Corrected Fact": item.corrected_fact or "",
                    "Evidence Sources": "; ".join(
                        e.url for e in item.evidence
                    ),
                }
            )

        df = pd.DataFrame(rows)

        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def generate_pdf(report: Report) -> bytes:
        """
        Generate PDF report and return bytes.
        """

        buffer = BytesIO()

        doc = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        story = []

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------

        story.append(
            Paragraph(
                "AI Fact Checking Report",
                styles["Title"],
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"Source File: {report.source_filename}",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )

        story.append(Spacer(1, 16))

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------

        story.append(
            Paragraph(
                "Summary",
                styles["Heading2"],
            )
        )

        summary_text = f"""
        Total Claims: {report.stats.total}<br/>
        Verified: {report.stats.verified}<br/>
        Inaccurate: {report.stats.inaccurate}<br/>
        False: {report.stats.false}<br/>
        Failed: {report.stats.failed}<br/>
        Average Confidence: {report.stats.avg_confidence}
        """

        story.append(
            Paragraph(
                summary_text,
                styles["BodyText"],
            )
        )

        story.append(Spacer(1, 20))

        # ---------------------------------------------------------
        # Claim Details
        # ---------------------------------------------------------

        story.append(
            Paragraph(
                "Detailed Results",
                styles["Heading2"],
            )
        )

        story.append(Spacer(1, 12))

        for idx, item in enumerate(report.claims, start=1):

            verdict_color = colors.green

            if item.verdict.value == "Inaccurate":
                verdict_color = colors.orange

            if item.verdict.value == "False":
                verdict_color = colors.red

            story.append(
                Paragraph(
                    f"<b>Claim {idx}</b>",
                    styles["Heading3"],
                )
            )

            story.append(
                Paragraph(
                    f"<b>Claim:</b> {item.claim.text}",
                    styles["BodyText"],
                )
            )

            story.append(
                Paragraph(
                    f"<b>Verdict:</b> "
                    f"<font color='{verdict_color.hexval()}'>{item.verdict.value}</font>",
                    styles["BodyText"],
                )
            )

            story.append(
                Paragraph(
                    f"<b>Confidence:</b> {item.confidence:.2f}",
                    styles["BodyText"],
                )
            )

            story.append(
                Paragraph(
                    f"<b>Explanation:</b> {item.explanation}",
                    styles["BodyText"],
                )
            )

            if item.corrected_fact:
                story.append(
                    Paragraph(
                        f"<b>Correct Fact:</b> {item.corrected_fact}",
                        styles["BodyText"],
                    )
                )

            if item.evidence:

                evidence_html = "<br/>".join(
                    f"• {e.title} ({e.url})"
                    for e in item.evidence
                )

                story.append(
                    Paragraph(
                        f"<b>Sources:</b><br/>{evidence_html}",
                        styles["BodyText"],
                    )
                )

            story.append(Spacer(1, 12))

        doc.build(story)

        pdf_bytes = buffer.getvalue()

        buffer.close()

        return pdf_bytes
