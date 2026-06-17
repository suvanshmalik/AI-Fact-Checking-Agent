# 🔍 AI Fact-Checking Agent

## Overview

For this project, I built an AI-powered fact-checking tool that helps users verify information contained in PDF documents.

The idea came from a simple problem: verifying claims in reports, articles, research papers, or documents is often a manual and time-consuming process. Instead of checking every statement individually, users can upload a PDF and let the system automatically identify factual claims, search for supporting evidence online, and provide a verdict along with explanations.

The application is built using Streamlit and combines Google's Gemini model with live web search through Tavily.

---

## Live Application

**Deployed App:**
(https://ai-fact-checking-agent-nhzsxaanwsvcadeo4larl8.streamlit.app/)

**GitHub Repository:**
(https://github.com/suvanshmalik/AI-Fact-Checking-Agent)

---

## What the Application Does

1. Upload a PDF document.
2. Extract textual content from the PDF.
3. Identify factual claims that can be verified.
4. Search the web for relevant evidence.
5. Evaluate the claim against the retrieved evidence.
6. Display a verdict, confidence score, and explanation.
7. Generate downloadable CSV and PDF reports.

---

## Key Features

### PDF Upload and Processing

* Supports PDF document uploads
* Extracts text using PyMuPDF
* Handles multi-page documents

### Automated Claim Extraction

The system identifies factual statements that are suitable for verification, such as:

* Statistics
* Dates
* Numerical metrics
* Financial claims
* General factual statements

### Evidence Retrieval

Relevant information is gathered from live web sources using Tavily Search. The retrieved results are ranked before being passed to the verification stage.

### AI-Based Verification

Each claim is evaluated using Gemini and classified as:

* Verified
* Inaccurate
* False

The system also provides:

* Confidence scores
* Explanations
* Supporting evidence links

### Report Generation

Users can download:

* CSV reports
* PDF reports

for further review or sharing.

---

## Tech Stack

| Component       | Technology        |
| --------------- | ----------------- |
| Frontend        | Streamlit         |
| AI Model        | Google Gemini     |
| Search Engine   | Tavily            |
| PDF Processing  | PyMuPDF           |
| Data Validation | Pydantic          |
| Reporting       | Pandas, ReportLab |
| Language        | Python            |

---

## Project Structure

```text
AI-Fact-Checking-Agent/
│
├── app.py
├── config.py
├── requirements.txt
│
├── models/
│   └── schemas.py
│
├── services/
│   ├── pdf_processor.py
│   ├── claim_extractor.py
│   ├── web_search.py
│   ├── evidence_ranker.py
│   ├── verifier.py
│   └── report_generator.py
│
└── utils/
    └── helpers.py
```

---

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/suvanshmalik/AI-Fact-Checking-Agent.git
cd AI-Fact-Checking-Agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key
```

Run the application:

```bash
python -m streamlit run app.py
```

---

## Challenges Faced

A few interesting challenges came up while building the project:

* Handling malformed JSON responses from the LLM
* Managing API rate limits and retries
* Ranking web search results based on source quality and relevance
* Creating a workflow that remained responsive while processing multiple claims
* Generating clean downloadable reports from verification results

Addressing these challenges helped make the application more reliable and practical for real-world usage.

---

## Possible Future Improvements

If I were to continue developing this project, I would explore:

* OCR support for scanned PDFs
* Highlighting claims directly inside PDFs
* Multi-language verification
* Source credibility scoring
* Batch document verification
* Browser extension integration

---

## Author

**Suvansh Malik**

Built as part of a Product Management Trainee assessment.
