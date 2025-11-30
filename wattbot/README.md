# WattBot — Retrieval + LLM pipeline

Short guide
- Hybrid retrieval (BM25 + FAISS) + LLM answering pipeline (originally from Colab).
- Main script: wattbot_v1_0.537.py

Required inputs (place under PROJECT_ROOT or outputs/ as in script)
- outputs/chunks.json         (list of {"source","text"})
- outputs/retrieval_config.json
- outputs/bm25_index.pkl
- outputs/faiss_index.idx
- outputs/chunk_embeddings.npy
- metadata.csv                (id,url)
- test_Q.csv                  (id,question,answer_unit)

Quick start (Colab)
1. Upload this repo to GitHub and open the notebook/script in Colab.
2. Mount Drive or set PROJECT_ROOT to your Drive folder with the required files.
3. Install dependencies: pip install -r requirements.txt (or use the script's pip cells).
4. Run cells or run the script to build indexes and generate submission.csv.

Quick start (local)
1. Install dependencies: pip install -r requirements.txt
2. Remove or guard google.colab parts (drive.mount) and set PROJECT_ROOT path.
3. Run: python wattbot_v1_0.537.py