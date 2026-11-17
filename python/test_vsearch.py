import numpy as np
import pytest
import vsearch


def brute_force(base, q, k):
    d = ((q[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


def test_recall_and_shapes():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3000, 32), dtype=np.float32)
    q = rng.standard_normal((50, 32), dtype=np.float32)
    idx = vsearch.Index(32, 3000, space="l2", M=16, ef_construction=200)
    idx.add_items(base, num_threads=2)
    assert len(idx) == 3000
    labels, dists = idx.knn_query(q, k=10, ef=100)
    assert labels.shape == (50, 10) and dists.shape == (50, 10)
    truth = brute_force(base, q, 10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, truth)])
    assert recall > 0.95
    assert np.all(np.diff(dists, axis=1) >= -1e-5)  # ascending


def test_custom_ids_and_single_vector():
    idx = vsearch.Index(4, 10, space="cosine")
    v = np.eye(4, dtype=np.float32)
    idx.add_items(v, ids=[10, 20, 30, 40])
    labels, dists = idx.knn_query(np.array([0, 0, 1, 0], dtype=np.float32), k=1)
    assert labels[0, 0] == 30 and abs(dists[0, 0]) < 1e-5


def test_errors():
    idx = vsearch.Index(4, 2)
    with pytest.raises(ValueError):
        idx.add_items(np.zeros((1, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        vsearch.Index(4, 2, space="manhattan")
    idx.add_items(np.zeros((2, 4), dtype=np.float32), num_threads=1)
    with pytest.raises(RuntimeError):
        idx.add_items(np.zeros((1, 4), dtype=np.float32), num_threads=1)


def test_missing_results_padded():
    idx = vsearch.Index(2, 5)
    idx.add_items(np.array([[0, 0], [1, 1]], dtype=np.float32))
    labels, dists = idx.knn_query(np.array([[0, 0]], dtype=np.float32), k=4)
    assert list(labels[0][2:]) == [-1, -1] and np.isinf(dists[0][2:]).all()
