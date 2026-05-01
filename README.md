# WattBot: Retrieval-Augmented Generation for AI Sustainability Q&A

WattBot is a retrieval-augmented question-answering system for extracting precise answers from academic papers about AI energy consumption and carbon footprint research. The system combines layout-aware PDF parsing, hybrid retrieval, reranking, local LLM inference, and type-specific answer extraction to answer numerical, boolean, range, count, and entity-comparison questions with supporting evidence.

## Resume Summary

Built an end-to-end RAG pipeline over a corpus of ~25 AI sustainability papers, improving answer accuracy from a BM25/FAISS baseline of ~54% to 62.4% through OCR-based document parsing, hybrid dense/sparse retrieval, cross-encoder reranking, and specialized prompts for different question types. The final system runs locally with Qwen2.5-7B via Ollama, avoiding external API keys and data egress.

## Demo

[Watch the demo on YouTube](https://youtu.be/9_xwr4Wil40)

## Problem Statement

Given questions about AI sustainability papers, extract exact answers with document attribution and supporting evidence.

- Corpus: ~25 PDF papers on AI sustainability topics
- Training set: 41 human-labeled questions
- Answer types: boolean (0/1), numerical, unit-qualified values, ranges, and counts
- Output fields: answer, answer value, answer unit, reference ID, reference URL, supporting materials, and explanation

## Key Contributions

- Designed a two-stage RAG system from baseline retrieval to an advanced local inference pipeline.
- Replaced simple PDF text extraction with layout-aware parsing to preserve tables and scientific document structure.
- Combined dense retrieval and BM25 to handle both semantic questions and exact numeric/entity matching.
- Added cross-encoder reranking to improve evidence precision before LLM generation.
- Implemented question-type routing for boolean, difference, factor, count, percentage, range, cost, and default extraction cases.
- Measured iterative accuracy improvements and documented the design decisions behind each version.

## System Architecture

```text
PDF Papers
  -> Marker OCR / layout-aware parsing
  -> LlamaIndex MarkdownNodeParser
  -> BAAI/bge-base-en-v1.5 embeddings
  -> Persisted vector index (~20K nodes)

Question
  -> Hybrid retrieval (60% vector + 40% BM25)
  -> Cross-encoder reranking
  -> Question-type classification
  -> Specialized extraction prompt
  -> Qwen2.5-7B via Ollama
  -> Structured answer parsing
  -> Evidence-backed CSV output
```

## Results

| Version | Accuracy | Key Improvement |
| --- | --- | --- |
| v1 | ~54% | BM25 + FAISS baseline |
| v3 | 57.9% | LlamaIndex integration |
| v4 | ~60.0% | Marker OCR and wider retrieval context |
| v5 | 62.4% | Evidence-first boolean prompts and type-specialized extraction |

## Technology Stack

| Component | Technology |
| --- | --- |
| LLM | Qwen2.5-7B through Ollama local inference |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF Parsing | Marker OCR / layout-aware parsing |
| Vector Index | LlamaIndex, FAISS |
| Sparse Retrieval | BM25Okapi, rank-bm25 |
| Orchestration | Python 3.11, Jupyter, pandas |

## Repository Structure

```text
NLP_teamproject/
├── README.md
├── pipeline/                        # Advanced RAG pipeline (v3-v5)
│   ├── README.md                    # Detailed pipeline setup and design notes
│   ├── rag_inference.ipynb          # Main inference and submission notebook
│   ├── requirements.txt
│   ├── wattbot_data/                # Input data, not tracked in git
│   └── agent_storage_v4/            # Pre-built vector index, not tracked in git
└── baseline/                        # Colab prototype (v1)
    ├── retrieval_pipeline.py
    └── requirements.txt
```

## Quick Start

See [pipeline/README.md](pipeline/README.md) for full setup instructions.

```bash
# 1. Start Ollama
ollama serve
ollama pull qwen2.5:7b

# 2. Create environment
python3.11 -m venv .venv_311
source .venv_311/bin/activate
pip install -r pipeline/requirements.txt

# 3. Open pipeline/rag_inference.ipynb and run cells in order
```

## Data

Input files are not committed due to size. Required files:

- `pipeline/wattbot_data/pdfs/` -- source PDF papers
- `pipeline/wattbot_data/train_QA.csv` -- training questions with ground-truth answers
- `pipeline/wattbot_data/test_Q.csv` -- test questions
- `pipeline/wattbot_data/metadata.csv` -- paper ID to URL mapping
- `pipeline/agent_storage_v4/` -- pre-built LlamaIndex vector index (~20K nodes), downloadable from [Google Drive](https://drive.google.com/file/d/1HVwAt6VQSqHKeRovd2lw0-4LRFWO0x0k/view?usp=drive_link)

## Why This Project Matters

This project demonstrates practical RAG engineering beyond a basic chatbot: document ingestion, retrieval quality, reranking, prompt design, local model serving, structured output parsing, and measurable iteration against labeled evaluation data.

## References

- [Ollama](https://ollama.ai) -- Local LLM runtime
- [LlamaIndex](https://www.llamaindex.ai) -- RAG orchestration framework
- [Marker](https://github.com/VikParuchuri/marker) -- Layout-aware PDF parsing
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) -- Text embeddings
- [MS-MARCO MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
