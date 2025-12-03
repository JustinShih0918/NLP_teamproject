# WattBot RAG System - Document Q&A for AI Sustainability Papers

A Retrieval-Augmented Generation (RAG) system for answering questions about AI sustainability research papers. Uses local LLMs via Ollama with no API keys required.

## 📁 Project Structure

```
Final/
├── test_ollama.ipynb          # Main notebook - run this for inference
├── agent_ollama.ipynb         # PDF parsing with Marker OCR (run once)
├── test_ollama_v2.ipynb       # Experimental version with improvements
├── requirements.txt           # Python dependencies
├── wattbot_data/              # Data directory
│   ├── pdfs/                  # Source PDF papers
│   ├── train_QA.csv           # Training questions with answers
│   ├── test_Q.csv             # Test questions
│   └── metadata.csv           # Paper metadata
├── agent_storage_v4/          # Pre-built Marker-parsed index (20K+ nodes)
└── submission_*.csv           # Generated submission files
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** (required for Marker OCR)
2. **Ollama** installed and running locally
3. **~8GB RAM** minimum (16GB recommended)

### Step 1: Install Ollama & Download Model

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Download the Qwen 2.5 7B model (in a new terminal)
ollama pull qwen2.5:7b
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv_311
source .venv_311/bin/activate
```

### Step 3: Run the Notebook

1. **Open `test_ollama.ipynb`** in VS Code or Jupyter
2. **Select the `.venv_311` kernel**
3. **Run cells in order:**

| Cell | Description | Run Once? |
|------|-------------|-----------|
| 1 | Install pip packages | ✅ First time only |
| 2 | Import libraries | Every session |
| 3 | Configure models (Ollama, embeddings, reranker) | Every session |
| 4 | Load Marker-parsed index from `agent_storage_v4/` | Every session |
| 5 | Build hybrid retriever (BM25 + Vector) | Every session |
| 6-8 | Query functions and prompts | Every session |
| 9 | Validation on training set | Optional |
| 10 | Generate submission CSV | When ready |

## 📊 Pipeline Overview

```
PDF Papers → Marker OCR → Chunking → Embeddings → Vector Index
                                          ↓
Question → Query Decomposition → Hybrid Retrieval → Reranking → LLM → Answer
                                  (BM25 + Vector)
```

### Key Components

| Component | Model/Tool | Purpose |
|-----------|------------|---------|
| **LLM** | `qwen2.5:7b` (Ollama) | Answer generation |
| **Embeddings** | `BAAI/bge-base-en-v1.5` | Semantic search |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Result quality |
| **PDF Parser** | Marker OCR | Table extraction |
| **Retrieval** | Hybrid (BM25 + Vector) | Best of both worlds |

## 🔧 Troubleshooting

### Ollama Connection Error
```
Error: Could not connect to Ollama
```
**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

### Out of Memory
```
Error: CUDA out of memory / MPS out of memory
```
**Solution:** Use a smaller model:
```python
llm = Ollama(model="qwen2.5:3b", ...)  # 3B instead of 7B
```

### Marker Index Not Found
```
FileNotFoundError: agent_storage_v4 not found
```
**Solution:** Run `agent_ollama.ipynb` first to create the Marker-parsed index, or check that the `agent_storage_v4/` directory exists.

**actual solution:** download from [agent_storage_v4](https://drive.google.com/file/d/1HVwAt6VQSqHKeRovd2lw0-4LRFWO0x0k/view?usp=drive_link), put the data inside into `shih/agent_storage_v4`

### Slow Performance
- First run downloads models (~4GB)
- Use CPU if no GPU: embeddings will be slower
- Marker parsing is slow (~30min for all PDFs) - use pre-built index

## 📈 Performance

| Version | Accuracy | Notes |
|---------|----------|-------|
| v3 | 57.9% | Basic RAG |
| v4 | ~60% | Marker OCR + wider context |
| v5 | 62.4% | JSON output + evidence-first T/F |

## 🗂️ Data

- **41 training questions** with ground truth answers
- **~25 PDF papers** on AI sustainability topics
- Question types: TRUE/FALSE, numerical extraction, differences, ranges

## 📝 Generating Submissions

After validation, generate the test submission:

```python
# In the last cell of test_ollama.ipynb
submission_df = generate_submission_v4(
    hybrid_retriever, reranker, llm,
    TEST_CSV_PATH, "submission_v5.csv"
)
```

Output format:
```csv
id,question,answer,answer_value,answer_unit,ref_id,ref_url,supporting_materials,explanation
```

## 🔗 Dependencies

Key packages (see Cell 1 for full list):
- `llama-index-core>=0.14.0`
- `llama-index-llms-ollama>=0.9.0`
- `llama-index-embeddings-huggingface>=0.6.0`
- `sentence-transformers`
- `rank-bm25`
- `torch`

## 📚 References

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Marker](https://github.com/VikParuchuri/marker) - PDF OCR with layout detection
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5) - Text embeddings
