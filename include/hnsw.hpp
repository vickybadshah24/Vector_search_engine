// hnsw.hpp - header-only HNSW (Hierarchical Navigable Small World) index.
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
// using HNSW graphs", 2016.  C++17, thread-safe inserts and searches.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace vs {

enum class Metric { L2, Cosine };

// ---------------------------------------------------------------- distances
inline float l2_sqr(const float* a, const float* b, size_t d) {
    size_t i = 0;
    float s = 0.f;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= d; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }
    alignas(32) float t[8];
    _mm256_store_ps(t, acc);
    s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#endif
    for (; i < d; ++i) {
        float x = a[i] - b[i];
        s += x * x;
    }
    return s;
}

inline float dot(const float* a, const float* b, size_t d) {
    size_t i = 0;
    float s = 0.f;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= d; i += 8)
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    alignas(32) float t[8];
    _mm256_store_ps(t, acc);
    s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#endif
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}

// Runs fn(i) for i in [0, n) on `threads` threads (0 = all cores).
template <class F>
void parallel_for(size_t n, size_t threads, F&& fn) {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<size_t>(threads, std::max<size_t>(n, 1));
    if (threads <= 1) {
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr err;
    std::mutex err_mtx;
    auto worker = [&]() {
        try {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= n) break;
                fn(i);
            }
        } catch (...) {
            std::lock_guard<std::mutex> g(err_mtx);
            if (!err) err = std::current_exception();
            next.store(n);
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

// -------------------------------------------------------------------- index
class HNSW {
public:
    using Label = int64_t;
    using Result = std::pair<Label, float>;  // (label, distance)

    // max_elements: fixed capacity (memory is preallocated so inserts can run
    // concurrently).  L2 returns *squared* L2; Cosine returns 1 - cos_sim.
    HNSW(size_t dim, size_t max_elements, Metric metric = Metric::L2, size_t M = 16,
         size_t ef_construction = 200, uint64_t seed = 100)
        : dim_(dim), max_(max_elements), metric_(metric), M_(M), maxM_(M), maxM0_(2 * M),
          efc_(std::max(ef_construction, M)), seed_(seed), mult_(1.0 / std::log(double(M))),
          data_(max_elements * dim), labels_(max_elements), nodes_(max_elements) {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
        if (M < 2) throw std::invalid_argument("M must be >= 2");
    }

    size_t size() const { return count_.load(); }
    size_t dim() const { return dim_; }
    size_t max_elements() const { return max_; }
    void set_ef(size_t ef) { ef_ = ef; }
    size_t ef() const { return ef_; }

    // Approximate memory footprint in bytes (vectors + graph links).
    size_t memory_bytes() const {
        size_t total = size() * dim_ * sizeof(float);
        for (size_t i = 0; i < size(); ++i)
            for (auto& l : nodes_[i].links) total += l.capacity() * sizeof(uint32_t);
        return total;
    }

    // Thread-safe.  Throws std::runtime_error if the index is full.
    void add(const float* vec, Label label) {
        uint32_t id = allocate();
        float* dst = data_ptr(id);
        std::memcpy(dst, vec, dim_ * sizeof(float));
        if (metric_ == Metric::Cosine) normalize(dst);
        labels_[id] = label;

        const int level = random_level(id);
        {
            Node& n = nodes_[id];
            n.level = level;
            n.links.assign(level + 1, {});
            for (int l = 0; l <= level; ++l) n.links[l].reserve(l == 0 ? maxM0_ : maxM_);
        }

        std::unique_lock<std::mutex> glock(global_);
        if (entry_ < 0) {
            entry_ = int(id);
            max_level_ = level;
            return;
        }
        const uint32_t ep = uint32_t(entry_);
        const int top = max_level_;
        if (level <= top) glock.unlock();  // otherwise keep it until we update the entry point

        const float* q = dst;
        uint32_t cur = ep;
        float curd = dist(q, data_ptr(ep));
        for (int l = top; l > level; --l) greedy(q, cur, curd, l);

        std::vector<Cand> eps{{curd, cur}};
        for (int l = std::min(level, top); l >= 0; --l) {
            auto heap = search_layer(q, eps, efc_, l);
            std::vector<Cand> cands = to_sorted(heap);
            std::vector<Cand> sel = select_neighbors(cands, M_);
            {
                std::lock_guard<std::mutex> g(nodes_[id].mtx);
                auto& lst = nodes_[id].links[l];
                lst.clear();
                for (auto& c : sel) lst.push_back(c.second);
            }
            for (auto& c : sel) connect(c.second, id, l);
            eps = std::move(cands);
        }
        if (level > top) {
            entry_ = int(id);
            max_level_ = level;
        }
    }

    // Insert n vectors (row-major) in parallel. labels may be nullptr
    // (labels then default to the insertion position, offset by current size).
    void add_batch(const float* data, const Label* labels, size_t n, size_t threads = 0) {
        const Label base = Label(size());
        parallel_for(n, threads, [&](size_t i) {
            add(data + i * dim_, labels ? labels[i] : base + Label(i));
        });
    }

    // k nearest neighbours of `query`, ascending by distance. Thread-safe.
    std::vector<Result> search(const float* query, size_t k, size_t ef = 0) const {
        int ep, top;
        {
            std::lock_guard<std::mutex> g(global_);
            ep = entry_;
            top = max_level_;
        }
        if (ep < 0 || k == 0) return {};
        if (ef == 0) ef = ef_;
        ef = std::max(ef, k);

        const float* q = query;
        thread_local std::vector<float> qbuf;
        if (metric_ == Metric::Cosine) {
            qbuf.assign(query, query + dim_);
            normalize(qbuf.data());
            q = qbuf.data();
        }
        uint32_t cur = uint32_t(ep);
        float curd = dist(q, data_ptr(cur));
        for (int l = top; l > 0; --l) greedy(q, cur, curd, l);

        auto heap = search_layer(q, {{curd, cur}}, ef, 0);
        while (heap.size() > k) heap.pop();
        std::vector<Result> out(heap.size());
        for (size_t i = out.size(); i-- > 0;) {
            out[i] = {labels_[heap.top().second], heap.top().first};
            heap.pop();
        }
        return out;
    }

    // Batched search; out arrays are nq*k. Missing results: label -1, dist +inf.
    void search_batch(const float* queries, size_t nq, size_t k, size_t ef, Label* out_labels,
                      float* out_dist, size_t threads = 0) const {
        parallel_for(nq, threads, [&](size_t i) {
            auto r = search(queries + i * dim_, k, ef);
            for (size_t j = 0; j < k; ++j) {
                out_labels[i * k + j] = j < r.size() ? r[j].first : -1;
                out_dist[i * k + j] =
                    j < r.size() ? r[j].second : std::numeric_limits<float>::infinity();
            }
        });
    }

private:
    using Cand = std::pair<float, uint32_t>;  // (distance, internal id)
    struct Node {
        std::mutex mtx;
        int level = -1;
        std::vector<std::vector<uint32_t>> links;  // links[layer]
    };

    float dist(const float* a, const float* b) const {
        return metric_ == Metric::L2 ? l2_sqr(a, b, dim_) : 1.0f - dot(a, b, dim_);
    }
    float* data_ptr(size_t id) { return data_.data() + id * dim_; }
    const float* data_ptr(size_t id) const { return data_.data() + id * dim_; }

    void normalize(float* v) const {
        float n = std::sqrt(dot(v, v, dim_));
        if (n > 0.f)
            for (size_t i = 0; i < dim_; ++i) v[i] /= n;
    }

    uint32_t allocate() {
        size_t cur = count_.load();
        do {
            if (cur >= max_) throw std::runtime_error("HNSW index is full");
        } while (!count_.compare_exchange_weak(cur, cur + 1));
        return uint32_t(cur);
    }

    int random_level(uint32_t id) const {  // splitmix64 -> uniform -> exponential
        uint64_t z = seed_ + (uint64_t(id) + 1) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        double u = std::max((z >> 11) * (1.0 / 9007199254740992.0), 1e-12);
        return std::min(int(-std::log(u) * mult_), 32);
    }

    struct Visited {
        std::vector<uint32_t> tag;
        uint32_t epoch = 0;
    };
    Visited& visited() const {
        thread_local Visited v;
        if (v.tag.size() < max_) {
            v.tag.assign(max_, 0);
            v.epoch = 0;
        }
        if (++v.epoch == 0) {
            std::fill(v.tag.begin(), v.tag.end(), 0);
            v.epoch = 1;
        }
        return v;
    }

    // Greedy walk to the closest node on one layer (used on upper layers).
    void greedy(const float* q, uint32_t& cur, float& curd, int layer) const {
        thread_local std::vector<uint32_t> nbrs;
        for (bool changed = true; changed;) {
            changed = false;
            {
                std::lock_guard<std::mutex> g(nodes_[cur].mtx);
                nbrs = nodes_[cur].links[layer];
            }
            for (uint32_t nb : nbrs) {
                float d = dist(q, data_ptr(nb));
                if (d < curd) {
                    curd = d;
                    cur = nb;
                    changed = true;
                }
            }
        }
    }

    // Best-first search on one layer; returns a max-heap of up to ef candidates.
    std::priority_queue<Cand> search_layer(const float* q, const std::vector<Cand>& eps, size_t ef,
                                           int layer) const {
        Visited& vis = visited();
        std::priority_queue<Cand> top;
        std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> cand;
        for (const Cand& e : eps) {
            if (vis.tag[e.second] == vis.epoch) continue;
            vis.tag[e.second] = vis.epoch;
            top.push(e);
            cand.push(e);
        }
        while (top.size() > ef) top.pop();
        thread_local std::vector<uint32_t> nbrs;
        while (!cand.empty()) {
            Cand c = cand.top();
            if (top.size() >= ef && c.first > top.top().first) break;
            cand.pop();
            {
                std::lock_guard<std::mutex> g(nodes_[c.second].mtx);
                nbrs = nodes_[c.second].links[layer];
            }
            for (uint32_t nb : nbrs) {
                if (vis.tag[nb] == vis.epoch) continue;
                vis.tag[nb] = vis.epoch;
                float d = dist(q, data_ptr(nb));
                if (top.size() < ef || d < top.top().first) {
                    cand.push({d, nb});
                    top.push({d, nb});
                    if (top.size() > ef) top.pop();
                }
            }
        }
        return top;
    }

    static std::vector<Cand> to_sorted(std::priority_queue<Cand>& heap) {
        std::vector<Cand> v(heap.size());
        for (size_t i = v.size(); i-- > 0;) {
            v[i] = heap.top();
            heap.pop();
        }
        return v;
    }

    // Neighbour-selection heuristic (Algorithm 4 of the paper, keepPrunedConnections).
    std::vector<Cand> select_neighbors(const std::vector<Cand>& sorted, size_t M) const {
        if (sorted.size() <= M) return sorted;
        std::vector<Cand> out, skipped;
        out.reserve(M);
        for (const Cand& c : sorted) {
            if (out.size() >= M) break;
            bool good = true;
            const float* cv = data_ptr(c.second);
            for (const Cand& s : out)
                if (dist(cv, data_ptr(s.second)) < c.first) {
                    good = false;
                    break;
                }
            (good ? out : skipped).push_back(c);
        }
        for (const Cand& c : skipped) {
            if (out.size() >= M) break;
            out.push_back(c);
        }
        return out;
    }

    // Adds the edge nb -> id on `layer`, pruning nb's list if it overflows.
    // Only one node lock is ever held at a time, so there are no lock-order deadlocks.
    void connect(uint32_t nb, uint32_t id, int layer) {
        const size_t cap = layer == 0 ? maxM0_ : maxM_;
        std::lock_guard<std::mutex> g(nodes_[nb].mtx);
        auto& lst = nodes_[nb].links[layer];
        if (lst.size() < cap) {
            lst.push_back(id);
            return;
        }
        const float* nv = data_ptr(nb);
        std::vector<Cand> c;
        c.reserve(cap + 1);
        c.push_back({dist(nv, data_ptr(id)), id});
        for (uint32_t x : lst) c.push_back({dist(nv, data_ptr(x)), x});
        std::sort(c.begin(), c.end());
        auto sel = select_neighbors(c, cap);
        lst.clear();
        for (auto& s : sel) lst.push_back(s.second);
    }

    size_t dim_, max_;
    Metric metric_;
    size_t M_, maxM_, maxM0_, efc_;
    uint64_t seed_;
    double mult_;
    size_t ef_ = 50;
    std::vector<float> data_;
    std::vector<Label> labels_;
    mutable std::vector<Node> nodes_;
    std::atomic<size_t> count_{0};
    mutable std::mutex global_;  // guards entry_ / max_level_
    int entry_ = -1;
    int max_level_ = -1;
};

}  // namespace vs
