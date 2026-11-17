#include <iostream>
#include <random>
#include <set>
#include "hnsw.hpp"

static int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

static std::vector<float> random_data(size_t n, size_t d, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd;
    std::vector<float> v(n * d);
    for (auto& x : v) x = nd(rng);
    return v;
}

static double recall(const vs::HNSW& idx, const std::vector<float>& base, size_t n, size_t d,
                     const std::vector<float>& queries, size_t nq, size_t k, size_t ef,
                     bool cosine) {
    size_t hit = 0;
    for (size_t qi = 0; qi < nq; ++qi) {
        const float* q = &queries[qi * d];
        std::vector<std::pair<float, size_t>> all(n);
        for (size_t i = 0; i < n; ++i) {
            const float* b = &base[i * d];
            if (cosine) {
                float nb = std::sqrt(vs::dot(b, b, d)), nq2 = std::sqrt(vs::dot(q, q, d));
                all[i] = {1.f - vs::dot(q, b, d) / (nb * nq2), i};
            } else {
                all[i] = {vs::l2_sqr(q, b, d), i};
            }
        }
        std::partial_sort(all.begin(), all.begin() + k, all.end());
        std::set<size_t> truth;
        for (size_t i = 0; i < k; ++i) truth.insert(all[i].second);
        for (auto& r : idx.search(q, k, ef)) hit += truth.count(size_t(r.first));
    }
    return double(hit) / double(nq * k);
}

int main() {
    // SIMD distance equals scalar reference on awkward dimensions
    for (size_t d : {1, 7, 8, 13, 64, 100}) {
        auto a = random_data(1, d, 1), b = random_data(1, d, 2);
        float ref = 0, ref_dot = 0;
        for (size_t i = 0; i < d; ++i) {
            ref += (a[i] - b[i]) * (a[i] - b[i]);
            ref_dot += a[i] * b[i];
        }
        CHECK(std::fabs(vs::l2_sqr(a.data(), b.data(), d) - ref) < 1e-3f);
        CHECK(std::fabs(vs::dot(a.data(), b.data(), d) - ref_dot) < 1e-3f);
    }

    // empty index
    {
        vs::HNSW idx(4, 10);
        float q[4] = {0, 0, 0, 0};
        CHECK(idx.search(q, 5).empty());
    }

    // tiny index: exact answers, k > size, capacity error, labels
    {
        vs::HNSW idx(2, 5);
        float pts[5][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {10, 10}};
        for (int i = 0; i < 5; ++i) idx.add(pts[i], 100 + i);
        float q[2] = {2.2f, 0};
        auto r = idx.search(q, 3);
        CHECK(r.size() == 3);
        CHECK(r[0].first == 102 && r[1].first == 103 && r[2].first == 101);
        CHECK(idx.search(q, 50).size() == 5);
        bool threw = false;
        try {
            idx.add(pts[0], 999);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }

    const size_t n = 5000, d = 32, nq = 100, k = 10;
    auto base = random_data(n, d, 10), queries = random_data(nq, d, 11);

    {  // L2 recall, single thread
        vs::HNSW idx(d, n, vs::Metric::L2, 16, 200);
        for (size_t i = 0; i < n; ++i) idx.add(&base[i * d], vs::HNSW::Label(i));
        double r = recall(idx, base, n, d, queries, nq, k, 100, false);
        std::cout << "L2 recall@10 (ef=100, 1 thread):  " << r << "\n";
        CHECK(r > 0.95);
    }
    {  // L2 recall, parallel build
        vs::HNSW idx(d, n, vs::Metric::L2, 16, 200);
        idx.add_batch(base.data(), nullptr, n, 4);
        CHECK(idx.size() == n);
        double r = recall(idx, base, n, d, queries, nq, k, 100, false);
        std::cout << "L2 recall@10 (ef=100, 4 threads): " << r << "\n";
        CHECK(r > 0.95);
    }
    {  // Cosine
        vs::HNSW idx(d, n, vs::Metric::Cosine, 16, 200);
        idx.add_batch(base.data(), nullptr, n, 4);
        double r = recall(idx, base, n, d, queries, nq, k, 100, true);
        std::cout << "Cosine recall@10 (ef=100):        " << r << "\n";
        CHECK(r > 0.95);
        auto res = idx.search(&base[7 * d], 1);
        CHECK(res.size() == 1 && res[0].first == 7 && std::fabs(res[0].second) < 1e-4f);
    }
    {  // Batched search
        vs::HNSW idx(d, n);
        idx.add_batch(base.data(), nullptr, n, 4);
        std::vector<vs::HNSW::Label> labels(nq * k);
        std::vector<float> dists(nq * k);
        idx.search_batch(queries.data(), nq, k, 64, labels.data(), dists.data(), 4);
        auto single = idx.search(&queries[0], k, 64);
        CHECK(single[0].first == labels[0]);
    }

    if (failures == 0) std::cout << "ALL TESTS PASSED\n";
    return failures == 0 ? 0 : 1;
}
