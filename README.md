# Vector Search Engine (HNSW) in C++17

A from-scratch implementation of the HNSW approximate nearest-neighbour algorithm
(Malkov & Yashunin, 2016) with AVX2 SIMD distance kernels, multithreaded index building,
Python bindings (pybind11) and a FastAPI REST service. Includes tests, a benchmark harness,
Docker and CI.

## Layout
```
include/hnsw.hpp      the whole engine (header-only)
src/bindings.cpp      pybind11 module  ->  import vsearch
bench/bench.cpp       recall / QPS / p50 / p99 / memory benchmark vs brute force
tests/test_hnsw.cpp   C++ tests (SIMD vs scalar, recall, cosine, parallel build, edge cases)
python/               pytest tests for the bindings
api/main.py           FastAPI service (POST /vectors, POST /search, GET /stats)
examples/rag_demo.py  retrieval step of a RAG pipeline
```

## Build & test (C++)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench --n 100000 --dim 128 --queries 1000 --threads 8
```
Use `-DVS_AVX2=OFF` for CPUs without AVX2 (a scalar fallback is compiled automatically).

Real datasets (SIFT1M, GloVe) in `.fvecs` format:
```bash
./build/bench --base sift_base.fvecs --query sift_query.fvecs
```

## Python
```bash
pip install .                # builds the C++ extension
pytest python
```
```python
import numpy as np, vsearch
index = vsearch.Index(dim=128, max_elements=100_000, space="l2", M=16, ef_construction=200)
index.add_items(np.random.rand(10_000, 128).astype("float32"), num_threads=8)
labels, dists = index.knn_query(np.random.rand(5, 128).astype("float32"), k=10, ef=64)
```
`space="l2"` returns squared L2 distance; `space="cosine"` returns `1 - cosine similarity`.
Missing results (index smaller than k) are padded with label `-1`.

## REST API
```bash
pip install fastapi "uvicorn[standard]"
VS_DIM=384 uvicorn api.main:app --port 8000
curl -X POST localhost:8000/vectors -H 'content-type: application/json' \
     -d '{"vectors": [[...384 floats...]], "payloads": [{"text": "hello"}]}'
curl -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"vector": [...384 floats...], "k": 5}'
```
Docker: `docker build -t vsearch . && docker run -p 8000:8000 vsearch`

## Design notes
- **Graph:** multi-layer proximity graph, exponential level distribution (`mL = 1/ln M`),
  layer 0 degree cap `2M`, upper layers `M`; neighbour selection uses the paper's heuristic
  (Algorithm 4) with kept pruned connections.
- **SIMD:** AVX2 kernels for L2 and dot product (cosine = normalised vectors + dot).
- **Concurrency:** capacity is preallocated so vectors never move; each node has its own
  mutex and a thread only ever holds one node lock at a time (no lock-order deadlocks). A
  global mutex guards the entry point. Searches are safe concurrently with inserts.
- **Search:** per-thread epoch-tagged visited arrays (no per-query clearing).
- **Not implemented (good extensions):** persistence (save/load), deletions, int8 quantisation,
  AVX-512/NEON kernels, SoA/aligned memory layout, prefetching.

## Benchmark results
Fill this in with YOUR machine's numbers (`./build/bench`, ideally on SIFT1M) and compare to
hnswlib/FAISS on the same data. Sample run on a 1-core sandbox, synthetic clustered data
(50k x 128, M=16, efC=200): recall@10 = 0.96 at ef=40 (~16k QPS, p99 0.11 ms),
0.996 at ef=80 (~10k QPS, p99 0.15 ms).
