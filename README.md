# WattBot: Retrieval-Augmented Generation for AI Sustainability Q&A

WattBot is a retrieval-augmented question-answering system for extracting precise answers from academic papers about AI energy consumption and carbon footprint research. The system combines layout-aware PDF parsing, hybrid retrieval, reranking, local LLM inference, and type-specific answer extraction to answer numerical, boolean, range, count, and entity-comparison questions with supporting evidence.

## Demo

[Watch the demo on YouTube](https://youtu.be/9_xwr4Wil40)

## Problem Statement

Given questions about AI sustainability papers, extract exact answers with document attribution and supporting evidence.

- Corpus: ~25 PDF papers on AI sustainability topics
- Training set: 41 human-labeled questions
- Answer types: boolean (0/1), numerical, unit-qualified values, ranges, and counts
- Output fields: answer, answer value, answer unit, reference ID, reference URL, supporting materials, and explanation

## System Overview

The project contains two stages: an initial retrieval baseline and an advanced RAG pipeline.

### Stage 1: Baseline (`baseline/`)

- Manual PDF text chunking with source tracking
- BM25 sparse retrieval with `rank-bm25`
- FAISS dense retrieval with `all-MiniLM-L6-v2` embeddings
- Hybrid score fusion over normalized BM25 and dense retrieval scores
- Qwen2.5-3B-Instruct with 4-bit NF4 quantization
- JSON-style answer extraction prompt

### Stage 2: Advanced Pipeline (`pipeline/`)

- Layout-aware PDF parsing with Marker OCR
- LlamaIndex node management and persisted vector storage
- BAAI/bge-base-en-v1.5 embeddings
- BM25Okapi with number-preserving tokenization
- Hybrid retrieval with dense and sparse candidates
- Cross-encoder reranking with MS-MARCO MiniLM
- Qwen2.5-7B served locally through Ollama
- Question-type classification and specialized extraction prompts

## Architecture

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

| Version | Accuracy | Key Change |
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
│   ├── README.md                    # Detailed setup and pipeline notes
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

## References

- [Ollama](https://ollama.ai) -- Local LLM runtime
- [LlamaIndex](https://www.llamaindex.ai) -- RAG orchestration framework
- [Marker](https://github.com/VikParuchuri/marker) -- Layout-aware PDF parsing
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) -- Text embeddings
- [MS-MARCO MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
