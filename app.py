import streamlit as st

from models.schemas import Report

from services.pdf_processor import PDFProcessor
from services.claim_extractor import ClaimExtractor
from services.web_search import WebSearcher
from services.evidence_ranker import EvidenceRanker
from services.verifier import Verifier
from services.report_generator import ReportGenerator


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="AI Fact Checker",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AI Fact-Checking Agent")
st.write(
    "Upload a PDF and automatically verify factual claims using live web data."
)


# --------------------------------------------------
# Service Initialization
# --------------------------------------------------

@st.cache_resource
def get_services():
    return {
        "processor": PDFProcessor(),
        "extractor": ClaimExtractor(),
        "searcher": WebSearcher(),
        "ranker": EvidenceRanker(),
        "verifier": Verifier(),
    }


try:
    services = get_services()

except Exception as exc:
    st.error(f"Failed to initialize services: {exc}")
    st.stop()


# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Please upload a PDF file.")
    st.stop()


# --------------------------------------------------
# Run Pipeline
# --------------------------------------------------

if st.button("Run Fact Check"):

    try:

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ------------------------------------------
        # PDF Processing
        # ------------------------------------------

        status_text.info("Processing PDF...")

        pdf_doc = services["processor"].process(
            uploaded_file,
            filename=uploaded_file.name
        )

        progress_bar.progress(10)

        if pdf_doc.is_scanned:
            st.warning(
                "This PDF appears to be scanned/image-based. "
                "Extraction quality may be reduced."
            )

        # ------------------------------------------
        # Claim Extraction
        # ------------------------------------------

        status_text.info("Extracting claims...")

        claims = services["extractor"].extract(
            pdf_doc.chunks,
            source_filename=pdf_doc.filename
        )

        progress_bar.progress(25)

        if not claims:
            st.error("No factual claims detected.")
            st.stop()

        # ------------------------------------------
        # Create Report
        # ------------------------------------------

        report = Report(
            source_filename=pdf_doc.filename
        )

        total_claims = len(claims)

        # ------------------------------------------
        # Verification Loop
        # ------------------------------------------

        for idx, claim in enumerate(claims):

            status_text.info(
                f"Verifying claim {idx + 1} of {total_claims}"
            )

            search_results = services["searcher"].search(
                claim
            )

            ranked_results = services["ranker"].rank(
                claim,
                search_results
            )

            verified_claim = services["verifier"].verify(
                claim,
                ranked_results
            )

            report.claims.append(
                verified_claim
            )

            report.refresh_stats()

            progress_value = 25 + int(
                ((idx + 1) / total_claims) * 75
            )

            progress_bar.progress(progress_value)

        # ------------------------------------------
        # Finished
        # ------------------------------------------

        status_text.success("Fact-check completed.")

        st.divider()

        # ------------------------------------------
        # Summary Metrics
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Claims",
            report.stats.total
        )

        col2.metric(
            "Verified",
            report.stats.verified
        )

        col3.metric(
            "Inaccurate",
            report.stats.inaccurate
        )

        col4.metric(
            "False",
            report.stats.false
        )

        st.divider()

        # ------------------------------------------
        # Results
        # ------------------------------------------

        for item in report.claims:

            verdict = item.verdict.value

            if verdict == "Verified":
                st.success(item.claim.text)

            elif verdict == "False":
                st.error(item.claim.text)

            else:
                st.warning(item.claim.text)

            with st.expander(
                f"Details - Claim {item.claim.claim_id}"
            ):

                st.write(
                    f"**Verdict:** {item.verdict.value}"
                )

                st.write(
                    f"**Confidence:** {item.confidence:.2f}"
                )

                st.write(
                    f"**Explanation:** {item.explanation}"
                )

                if item.corrected_fact:
                    st.write(
                        f"**Correct Fact:** {item.corrected_fact}"
                    )

                if item.evidence:

                    st.subheader("Evidence")

                    for evidence in item.evidence:

                        st.markdown(
                            f"""
**{evidence.title}**

{evidence.url}

{evidence.snippet}
"""
                        )

        # ------------------------------------------
        # Downloads
        # ------------------------------------------

        csv_data = ReportGenerator.generate_csv(
            report
        )

        pdf_data = ReportGenerator.generate_pdf(
            report
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download CSV Report",
                data=csv_data,
                file_name="fact_check_report.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name="fact_check_report.pdf",
                mime="application/pdf",
            )

    except Exception as exc:

        st.exception(exc)