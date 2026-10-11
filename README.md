# 🏦 GenAI Financial Research Assistant

An end-to-end Generative AI RAG system that reads 5 years of annual financial reports (10-K filings) and answers natural language questions about them — with cited sources, measurable accuracy scores, and a full evaluation suite.

---

## What This Project Does

Financial analysts spend hours digging through dense SEC filings to find specific information. This project automates that process using **Retrieval-Augmented Generation (RAG)** — a technique where an AI retrieves only the most relevant passages from a large document set before generating a precise, cited answer.

Ask a question like *"What major acquisition did JPMorgan complete in 2023?"* and the system:
1. Searches across 5 years of filings to find the most relevant passages
2. Feeds those passages to a large language model (LLM)
3. Returns a grounded answer with inline citations and source references

---

## Key Features

- **Automated data pipeline** — downloads 10-K filings directly from SEC.gov
- **Semantic search** — finds relevant passages by meaning, not just keywords
- **Cited answers** — every fact in the response is traced back to a specific filing and year
- **Two-layer evaluation** — measures answer quality using both a custom LLM evaluator and the industry-standard DeepEval framework
- **Interactive web app** — clean Streamlit UI with adjustable settings and sample questions

---

## How It Works

```
SEC.gov filings  →  Text extraction  →  Vector database (ChromaDB)
                                               ↓
User question  →  Semantic search  →  Top relevant chunks  →  LLM (LLaMA 3.3 70B via Groq)  →  Cited answer
```

The system breaks each 10-K filing into small, overlapping chunks of text, converts them into numerical vectors (embeddings), and stores them in ChromaDB. When a question comes in, the system finds the most semantically similar chunks and passes them to the LLM as context — keeping the answer grounded in real data.

---

## Evaluation Results

The system was evaluated on 5 financial questions with known ground-truth answers, using two independent evaluation methods.

### Custom LLM Evaluator (via Groq + LLaMA 3.3 70B)

| Metric | Score |
|---|---|
| Faithfulness (answer stays grounded in retrieved context) | **0.87** |
| Answer Relevancy (answer addresses the question) | **0.90** |
| Correctness (answer matches ground truth) | **0.86** |

### DeepEval Framework (industry-standard)

| Metric | Score |
|---|---|
| Faithfulness | **0.91** |
| Answer Relevancy | **1.00** |
| Contextual Precision (retrieved chunks well-ranked) | **0.77** |
| Contextual Recall (context covers the ground truth) | **1.00** |
| Hallucination Rate | **0.00** ✅ |

**Zero hallucinations across all test cases.** The system never fabricated information — it either answered correctly from the documents or declined to answer.

---

## Tech Stack

| Area | Tools |
|---|---|
| Data ingestion | SEC EDGAR API, PyMuPDF, Python |
| Embeddings & vector search | `sentence-transformers` (all-MiniLM-L6-v2), ChromaDB |
| LLM inference | LLaMA 3.3 70B via Groq API |
| Orchestration | LangChain |
| Evaluation | Custom LLM-as-Judge, DeepEval |
| Web UI | Streamlit |
| Language | Python |

---

## Project Structure

```
├── 1_download_filings.py   # Pull 10-K filings from SEC.gov
├── 2_vector_db.py          # Chunk, embed, and index into ChromaDB
├── 3_interface.py          # Streamlit web app
├── 4_llm_eval.py           # Custom LLM-as-Judge evaluation
├── 5_deepeval.py           # DeepEval framework evaluation
├── rag_pipeline.py         # Core RAG pipeline (retriever + generator)
├── retriever.py            # Semantic search over ChromaDB
├── generator.py            # LLM answer generation with LangChain
├── requirements.txt
└── data/
    ├── raw/                # Downloaded 10-K filings
    └── chroma_db/          # Persistent vector database
```

---

## Skills Demonstrated

- Building production-style **RAG pipelines** from scratch — no off-the-shelf shortcuts
- Working with **vector databases** and semantic embeddings for document retrieval
- Integrating and prompting **large language models** via API (Groq / LLaMA)
- Designing and running **LLM evaluation frameworks** (both custom and third-party)
- Pulling structured data from **public APIs** (SEC EDGAR)
- Building **AI-powered web interfaces** with Streamlit
- Applying AI techniques to a **regulated financial domain** (SEC filings, compliance-adjacent)

---

## Sample Questions

- *What major acquisition did JPMorgan complete in 2023?*
- *How does JPMorgan describe its liquidity risk management approach?*
- *What are the main business segments JPMorgan operates in?*
- *How does JPMorgan manage credit risk?*
- *How does JPMorgan describe the competitive environment it operates in?*
- *What is JPMorgan's dividend policy?*

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Download filings from SEC.gov
python 1_download_filings.py

# Step 2: Build the vector database
python 2_vector_db.py

# Step 3: Launch the web app
streamlit run 3_interface.py

# Optional: Run evaluations
python 4_llm_eval.py
python 5_deepeval.py
```

> **Note:** Requires a [Groq API key](https://console.groq.com/) (free tier available). Set your key in `generator.py` and the evaluation scripts.

---

*Built as part of an AI engineering portfolio focused on production-ready LLM systems in regulated environments.*
