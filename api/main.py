"""FastAPI service on top of the C++ HNSW engine.

    VS_DIM=384 VS_MAX_ELEMENTS=100000 uvicorn api.main:app --port 8000
"""
import os
import threading
from typing import Dict, List, Optional

import numpy as np
import vsearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DIM = int(os.getenv("VS_DIM", "384"))
MAX_ELEMENTS = int(os.getenv("VS_MAX_ELEMENTS", "100000"))
SPACE = os.getenv("VS_SPACE", "cosine")  # "cosine" or "l2"

app = FastAPI(title="Vector Search Engine (HNSW, C++)", version="0.1.0")
index = vsearch.Index(DIM, MAX_ELEMENTS, space=SPACE, M=16, ef_construction=200)
payloads: Dict[int, dict] = {}
_next_id = 0
_id_lock = threading.Lock()


class AddRequest(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[int]] = None
    payloads: Optional[List[dict]] = Field(default=None, description="metadata, e.g. the source text")


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = Field(10, ge=1, le=1000)
    ef: Optional[int] = Field(None, ge=1, le=10000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {"dim": DIM, "space": SPACE, "size": len(index), "max_elements": MAX_ELEMENTS,
            "memory_mb": round(index.memory_bytes() / 1e6, 2)}


@app.post("/vectors")
def add_vectors(req: AddRequest):
    global _next_id
    arr = np.asarray(req.vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != DIM:
        raise HTTPException(422, f"vectors must have shape (n, {DIM})")
    n = arr.shape[0]
    if req.ids is not None and len(req.ids) != n:
        raise HTTPException(422, "ids must match the number of vectors")
    if req.payloads is not None and len(req.payloads) != n:
        raise HTTPException(422, "payloads must match the number of vectors")
    if len(index) + n > MAX_ELEMENTS:
        raise HTTPException(507, "index capacity exceeded")
    with _id_lock:
        ids = req.ids or list(range(_next_id, _next_id + n))
        _next_id = max(_next_id, max(ids) + 1)
    index.add_items(arr, ids=ids)
    if req.payloads:
        payloads.update(zip(ids, req.payloads))
    return {"added": n, "size": len(index), "ids": ids}


@app.post("/search")
def search(req: SearchRequest):
    if len(req.vector) != DIM:
        raise HTTPException(422, f"vector must have length {DIM}")
    if len(index) == 0:
        return {"results": []}
    labels, dists = index.knn_query(np.asarray(req.vector, dtype=np.float32), k=req.k, ef=req.ef)
    results = [{"id": int(i), "distance": float(d), "payload": payloads.get(int(i))}
               for i, d in zip(labels[0], dists[0]) if i >= 0]
    return {"results": results}
