"""
Milestone 1 — RAG Core: Ingestion Pipeline
--------------------------------------------
Reads all knowledge base documents (Leave Policy, WFH Policy, Holiday
Calendar, HR Guidelines, Employee Handbook), splits them into ~400
character chunks, embeds each chunk using Ollama's `nomic-embed-text`
model, and stores the vectors in a FAISS index for retrieval.

Run this once (and again any time the knowledge base docs change):
    python ingest.py
"""

import os
import glob
import json
import requests
import numpy as np
import faiss

KB_DIR = "knowledge_base"
INDEX_PATH = "vector_store.index"
META_PATH = "vector_store_meta.json"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_text(text):
    """Get an embedding vector for a piece of text via Ollama."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def build_index():
    doc_paths = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    if not doc_paths:
        raise SystemExit(f"No documents found in {KB_DIR}/")

    all_chunks = []       # text of each chunk
    all_meta = []          # {"source": filename, "chunk_id": i}

    for path in doc_paths:
        source_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = chunk_text(raw_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": source_name, "chunk_id": i})

        print(f"  {source_name}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks with '{EMBED_MODEL}'...")
    vectors = []
    for i, chunk in enumerate(all_chunks):
        vec = embed_text(chunk)
        vectors.append(vec)
        if (i + 1) % 10 == 0 or (i + 1) == len(all_chunks):
            print(f"  embedded {i + 1}/{len(all_chunks)}")

    vectors = np.array(vectors, dtype="float32")

    # L2-normalize so we can use inner product as cosine similarity.
    # (Raw nomic-embed-text vectors aren't unit-length, so raw L2 distance
    # is not a reliable similarity signal — this was the bug that caused
    # every query to be refused.)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors /= norms

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product on unit vectors == cosine similarity
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "meta": all_meta}, f, indent=2)

    print(f"\nDone. Index saved to '{INDEX_PATH}', metadata saved to '{META_PATH}'.")
    print(f"Total vectors: {index.ntotal}, dimension: {dim}")


if __name__ == "__main__":
    print("Scanning knowledge base documents...")
    build_index()
