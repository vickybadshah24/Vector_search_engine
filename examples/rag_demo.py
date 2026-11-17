"""Tiny RAG retrieval demo using the C++ index.

Uses sentence-transformers if installed (pip install sentence-transformers); otherwise falls back
to a hashed bag-of-words embedding so the demo runs offline (weaker semantics).
"""
import re
import zlib

import numpy as np
import vsearch

DOCS = [
    "HNSW builds a multi-layer proximity graph so searches start coarse and refine at the bottom layer.",
    "Retrieval-augmented generation grounds an LLM answer in documents fetched from a vector database.",
    "SIMD instructions such as AVX2 compute eight float operations per instruction for distance kernels.",
    "Redis is an in-memory key-value store often used for caching and rate limiting.",
    "Recall@k measures what fraction of the true k nearest neighbours the index returns.",
    "Dynamic programming solves overlapping subproblems by storing intermediate results.",
    "Chrome extensions built on Manifest V3 use service workers instead of persistent background pages.",
    "Cosine similarity compares the angle between two embedding vectors, ignoring their magnitude.",
]

try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    DIM = 384
    def embed(texts):
        return np.asarray(_model.encode(texts), dtype=np.float32)
except Exception:  # offline fallback
    DIM = 512
    def embed(texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        return out


index = vsearch.Index(DIM, max_elements=1000, space="cosine")
index.add_items(embed(DOCS), ids=list(range(len(DOCS))))

question = "How does an LLM use a vector database to answer questions?"
labels, dists = index.knn_query(embed([question]), k=3, ef=50)
context = [DOCS[i] for i in labels[0] if i >= 0]

print("Retrieved context:")
for i, d in zip(labels[0], dists[0]):
    print(f"  [{d:.3f}] {DOCS[i]}")
print("\nPrompt to send to your LLM:\n")
print("Answer using only this context:\n- " + "\n- ".join(context) + f"\n\nQuestion: {question}")
