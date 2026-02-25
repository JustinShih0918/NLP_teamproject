# WattBot: Retrieval-Augmented Generation for AI Sustainability Q&A

A question-answering system for extracting precise answers from academic papers on AI
sustainability. Built for a structured NLP competition, the system handles numerical
extraction, boolean reasoning, range queries, and entity comparisons over a corpus of
~25 research papers, evaluated against human-annotated ground truth.

---

## Problem Statement

Given questions about AI energy consumption and carbon footprint research, extract exact
answers -- numerical values, TRUE/FALSE labels, ranges, or counts -- from the source
papers with supporting evidence and document attribution.

- Corpus: ~25 PDF papers on AI sustainability topics
- Training set: 41 questions with ground-truth answers
- Answer types: boolean (0/1), numerical, unit-qualified values, ranges, counts

---

## System Overview

The project evolved through two stages, each in its own directory.

### Stage 1: Baseline -- `baseline/`

Initial prototype developed in Google Colab:

- PDF text chunked manually with source tracking
- BM25 (rank-bm25) and FAISS (IndexFlatIP, all-MiniLM-L6-v2 384-dim) indexes built at runtime
- Hybrid score fusion: alpha-weighted sum of normalised BM25 and dense scores
- Qwen2.5-3B-Instruct in 4-bit NF4 quantization via bitsandbytes
- Structured JSON prompt for `answer_value` extraction

### Stage 2: Advanced Pipeline -- `pipeline/`

Complete rebuild replacing every component:

- Marker OCR for layout-aware PDF parsing with proper table extraction
- LlamaIndex for node management and persistent vector index (~20K nodes)
- BAAI/bge-base-en-v1.5 embeddings (768-dim) replacing MiniLM-L6
- BM25Okapi with improved number-preserving tokenization
- Hybrid retrieval: 60% vector + 40% BM25, top-30 candidates per query
- Cross-encoder reranking over 50 candidates: MS-MARCO MiniLM-L-6-v2
- Qwen2.5-7B served locally via Ollama (no API keys, no data egress)
- Question-type classification dispatching specialised extraction prompts

---

## Results

| Version | Accuracy | Key Improvement |
|---------|----------|-----------------|
| v1 | ~54% | BM25 + FAISS baseline |
| v3 | 57.9% | LlamaIndex integration |
| v4 | ~60.0% | Marker OCR, wider context window (top_k=12) |
| v5 | 62.4% | Evidence-first boolean prompts, type-specialised extraction |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | Qwen2.5-7B (Ollama, local inference) |
| Embeddings | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF Parsing | Marker OCR (layout detection + table extraction) |
| Vector Index | LlamaIndex / FAISS (IndexFlatIP) |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Orchestration | Python 3.11, LlamaIndex Core |

---

## Repository Structure

```
NLP_teamproject/
├── README.md
├── pipeline/                        # Advanced RAG pipeline (v3-v5)
│   ├── rag_inference.ipynb          # Main inference and submission notebook
│   ├── requirements.txt
│   ├── wattbot_data/                # Input data (not tracked in git)
│   └── agent_storage_v4/           # Pre-built vector index (~20K nodes)
└── baseline/                        # Colab prototype (v1)
    ├── retrieval_pipeline.py        # Full Colab pipeline as a Python script
    └── requirements.txt
```

---

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

---

## Data

Input files are not committed due to size. Required files:

- `pipeline/wattbot_data/pdfs/` -- source PDF papers
- `pipeline/wattbot_data/train_QA.csv` -- training questions with ground-truth answers
- `pipeline/wattbot_data/test_Q.csv` -- test questions
- `pipeline/wattbot_data/metadata.csv` -- paper ID to URL mapping
- `pipeline/agent_storage_v4/` -- pre-built LlamaIndex vector index, available via
  Google Drive (link in pipeline/README.md)

---

## References

- [Ollama](https://ollama.ai) -- Local LLM runtime
- [LlamaIndex](https://www.llamaindex.ai) -- RAG orchestration framework
- [Marker](https://github.com/VikParuchuri/marker) -- Layout-aware PDF parsing
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) -- Text embeddings
- [MS-MARCO MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
