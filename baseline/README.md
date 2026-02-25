# Baseline Retrieval Pipeline (v1)

Hybrid BM25 + FAISS retrieval with a quantized Qwen2.5-3B-Instruct LLM, built to
answer questions about AI sustainability research papers. Originally developed and
run on Google Colab.

---

## Overview

- PDF text chunked with source attribution stored in `chunks.json`
- BM25 sparse index (rank-bm25) and FAISS inner-product dense index
  (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
- Hybrid score fusion: normalised BM25 + normalised dense weighted by alpha
- Qwen2.5-3B-Instruct in 4-bit NF4 quantization via bitsandbytes
- Structured JSON prompt for `answer_value` and `explanation` extraction
- Outputs `submission.csv` in the required competition format

---

## Main Script

`retrieval_pipeline.py` -- the full Colab notebook exported as a runnable Python
script. Contains all stages in sequence:

1. BM25 and FAISS index construction from `chunks.json`
2. LLM loading (Qwen2.5-3B, 4-bit quantized)
3. Hybrid retrieval and answer generation
4. Submission CSV creation

---

## Required Inputs

| Path | Description |
|------|-------------|
| `outputs/chunks.json` | List of `{"source", "text"}` objects |
| `outputs/retrieval_config.json` | Embedding model name and dimensions |
| `outputs/bm25_index.pkl` | Serialised BM25Okapi index |
| `outputs/faiss_index.idx` | FAISS flat inner-product index |
| `outputs/chunk_embeddings.npy` | Dense embedding matrix |
| `metadata.csv` | Paper ID to URL mapping |
| `test_Q.csv` | Test questions with `id`, `question`, `answer_unit` columns |

---

## Quick Start (Google Colab)

1. Upload this directory to your Google Drive under `MyDrive/NLP_teamproject/`.
2. Open `retrieval_pipeline.py` in Colab via File > Upload notebook.
3. Verify `PROJECT_ROOT` points to the correct Drive folder.
4. Install dependencies: `pip install -r requirements.txt`
5. Run all cells.

## Quick Start (Local)

1. Remove the `from google.colab import drive` import and `drive.mount(...)` call.
2. Set `PROJECT_ROOT` to the local path containing your input files.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python retrieval_pipeline.py`

---

## Dependencies

See `requirements.txt`. Key packages:

- `sentence-transformers` -- dense embeddings (all-MiniLM-L6-v2)
- `rank-bm25` -- BM25 sparse retrieval
- `faiss-cpu` -- approximate nearest-neighbour search
- `transformers` + `bitsandbytes` -- 4-bit quantized LLM inference
- `torch` -- PyTorch backend