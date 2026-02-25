# Advanced RAG Pipeline (v3-v5)

Retrieval-Augmented Generation system for extracting precise answers from AI
sustainability research papers. Runs entirely locally via Ollama -- no external
API keys or data egress required.

---

## Architecture

```
PDF Papers
  -> Marker OCR  (layout-aware parsing, table extraction)
  -> MarkdownNodeParser  (LlamaIndex chunking)
  -> BAAI/bge-base-en-v1.5 embeddings
  -> Persisted vector index  (agent_storage_v4/)

At query time:
  Question
  -> Hybrid Retrieval  (60% vector + 40% BM25, top-30 candidates)
  -> Cross-encoder reranking  (top-50 -> top-12)
  -> Question-type classification
  -> Specialised extraction prompt
  -> Qwen2.5-7B via Ollama
  -> Structured answer parsing  (Value / Unit / Quote)
```

---

## Directory Structure

```
pipeline/
├── rag_inference.ipynb       # Main inference and submission notebook
├── requirements.txt
├── wattbot_data/             # Input data (not tracked in git)
│   ├── pdfs/                 # Source PDF papers
│   ├── train_QA.csv          # Training questions with ground-truth answers
│   ├── test_Q.csv            # Test questions
│   └── metadata.csv          # Paper ID to URL mapping
└── agent_storage_v4/         # Pre-built LlamaIndex vector index (~20K nodes)
```

---

## Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally
- 16 GB RAM recommended (8 GB minimum with the 3B model)

---

## Setup

**1. Install Ollama and pull the model**

```bash
brew install ollama      # macOS
ollama serve
# In a separate terminal:
ollama pull qwen2.5:7b
```

**2. Create a Python virtual environment**

```bash
python3.11 -m venv .venv_311
source .venv_311/bin/activate
pip install -r pipeline/requirements.txt
```

**3. Obtain input data**

Place the following files under `pipeline/wattbot_data/`:

| File | Description |
|------|-------------|
| `pdfs/` | Source PDF papers |
| `train_QA.csv` | Training questions and ground-truth answers |
| `test_Q.csv` | Test questions |
| `metadata.csv` | Paper ID to URL mapping |

**4. Obtain the pre-built index**

Download `agent_storage_v4/` from the project Google Drive share and place the
contents into `pipeline/agent_storage_v4/`. This index stores ~20,000 nodes parsed
by Marker OCR with full layout and table metadata retained.

If the index is unavailable, rebuild it by running the Marker OCR parsing notebook
(takes approximately 30 minutes).

---

## Running Inference

Open `rag_inference.ipynb` in VS Code or Jupyter and select the `.venv_311` kernel.
Run cells in order:

| Cell | Description | Frequency |
|------|-------------|-----------|
| 1 | Install pip packages | First time only |
| 2 | Import libraries | Every session |
| 3 | Configure models (Ollama, embeddings, reranker) | Every session |
| 4 | Load pre-built index from agent_storage_v4/ | Every session |
| 5 | Build HybridRetriever | Every session |
| 6-8 | Query functions and prompt templates | Every session |
| 9 | Validate on training set | Optional |
| 10 | Generate submission CSV | When ready |

**Generating a submission file**

```python
submission_df = generate_submission_v4(
    hybrid_retriever, reranker, llm,
    TEST_CSV_PATH, "submission_v5.csv"
)
```

Output columns: `id, question, answer, answer_value, answer_unit, ref_id,
ref_url, supporting_materials, explanation`

---

## Key Design Decisions

### Hybrid Retrieval

BM25 alone misses semantic matches; dense retrieval alone struggles with exact
numbers and model names. Combining both at a 60/40 ratio improves recall across
both question types.

### Cross-Encoder Reranking

A bi-encoder retrieves 50 candidates efficiently; the cross-encoder then scores
each candidate against the query jointly, significantly improving precision before
the LLM sees the context.

### Question-Type Classification

Different question types require fundamentally different extraction strategies.
The system classifies each question and routes it to a specialised prompt:

| Type | Trigger | Strategy |
|------|---------|----------|
| Boolean | "true or false" | Evidence-first chain, explicit T/F conclusion |
| Difference | "difference between" | Extract two values for same entity, subtract |
| Factor | "by what factor" | Technique-specific extraction, ignore other factors |
| Count | "how many" (no "percent") | Explicit stated count only, no calculation |
| Percentage | "percent", "proportion" | Number without % sign |
| Range | "range" | [low,high] bracket format |
| Cost | "cost", "price" | Million/billion unit conversion |
| Default | -- | Condition-matching extraction with upper-bound for ranges |

---

## Results

| Version | Accuracy | Notes |
|---------|----------|-------|
| v3 | 57.9% | LlamaIndex baseline |
| v4 | ~60.0% | Marker OCR, wider retrieval context (top_k=12) |
| v5 | 62.4% | Evidence-first boolean prompts, type-specialised prompts |

---

## Troubleshooting

**Ollama connection error**

```bash
ollama serve
```

**Out of memory**

Switch to the smaller model in the configuration cell:

```python
llm = Ollama(model="qwen2.5:3b", ...)
```

**Index not found**

Download from Google Drive (see Setup step 4), or re-run the Marker OCR parsing
notebook to rebuild from the source PDFs.
