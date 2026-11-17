// Benchmark: build time, recall@k, QPS and latency percentiles vs. brute force.
//   ./bench                                   synthetic clustered data
//   ./bench --base sift_base.fvecs --query sift_query.fvecs   (SIFT1M / GloVe in fvecs format)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include "hnsw.hpp"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static std::vector<float> read_fvecs(const std::string& path, size_t& dim, size_t& n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t d;
    f.read(reinterpret_cast<char*>(&d), 4);
    dim = size_t(d);
    f.seekg(0, std::ios::end);
    n = size_t(f.tellg()) / (4 + 4 * dim);
    f.seekg(0);
    std::vector<float> out(n * dim);
    for (size_t i = 0; i < n; ++i) {
        f.seekg(4, std::ios::cur);
        f.read(reinterpret_cast<char*>(&out[i * dim]), std::streamsize(4 * dim));
    }
    return out;
}

static std::vector<float> synthetic(size_t n, size_t d, unsigned seed, const std::vector<float>& centers) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd;
    size_t nc = centers.size() / d;
    std::vector<float> v(n * d);
    for (size_t i = 0; i < n; ++i) {
        size_t c = rng() % nc;
        for (size_t j = 0; j < d; ++j) v[i * d + j] = centers[c * d + j] + nd(rng);
    }
    return v;
}

int main(int argc, char** argv) {
    size_t n = 50000, d = 128, nq = 1000, k = 10, M = 16, efc = 200, threads = 0;
    std::string base_path, query_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--n") n = std::stoul(next());
        else if (a == "--dim") d = std::stoul(next());
        else if (a == "--queries") nq = std::stoul(next());
        else if (a == "--k") k = std::stoul(next());
        else if (a == "--M") M = std::stoul(next());
        else if (a == "--efc") efc = std::stoul(next());
        else if (a == "--threads") threads = std::stoul(next());
        else if (a == "--base") base_path = next();
        else if (a == "--query") query_path = next();
    }
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());

    std::vector<float> base, queries;
    if (!base_path.empty()) {
        size_t d2, nq2;
        base = read_fvecs(base_path, d, n);
        queries = read_fvecs(query_path, d2, nq2);
        nq = std::min(nq, nq2);
        queries.resize(nq * d);
    } else {
        std::mt19937 rng(1);
        std::normal_distribution<float> nd;
        std::vector<float> centers(50 * d);
        for (auto& x : centers) x = 4.f * nd(rng);
        base = synthetic(n, d, 2, centers);
        queries = synthetic(nq, d, 3, centers);
    }
    std::printf("dataset: n=%zu dim=%zu queries=%zu k=%zu | M=%zu efC=%zu threads=%zu\n", n, d, nq, k,
                M, efc, threads);

    // exact ground truth
    auto t0 = Clock::now();
    std::vector<std::set<int64_t>> truth(nq);
    vs::parallel_for(nq, threads, [&](size_t qi) {
        std::vector<std::pair<float, int64_t>> all(n);
        for (size_t i = 0; i < n; ++i) all[i] = {vs::l2_sqr(&queries[qi * d], &base[i * d], d), int64_t(i)};
        std::partial_sort(all.begin(), all.begin() + long(k), all.end());
        for (size_t i = 0; i < k; ++i) truth[qi].insert(all[i].second);
    });
    std::printf("brute-force ground truth: %.2fs\n", secs(t0, Clock::now()));

    vs::HNSW idx(d, n, vs::Metric::L2, M, efc);
    t0 = Clock::now();
    idx.add_batch(base.data(), nullptr, n, threads);
    double build = secs(t0, Clock::now());
    std::printf("build: %.2fs (%.0f vectors/s) | memory: %.1f MB\n\n", build, double(n) / build,
                double(idx.memory_bytes()) / 1e6);

    std::printf("%6s %10s %12s %12s %9s %9s\n", "ef", "recall@k", "QPS(1 thr)",
                ("QPS(" + std::to_string(threads) + " thr)").c_str(), "p50 ms", "p99 ms");
    for (size_t ef : {10, 20, 40, 80, 160, 320}) {
        if (ef < k) continue;
        std::vector<double> lat(nq);
        size_t hit = 0;
        auto s = Clock::now();
        for (size_t qi = 0; qi < nq; ++qi) {
            auto a = Clock::now();
            auto r = idx.search(&queries[qi * d], k, ef);
            lat[qi] = secs(a, Clock::now()) * 1e3;
            for (auto& x : r) hit += truth[qi].count(x.first);
        }
        double qps1 = double(nq) / secs(s, Clock::now());
        std::sort(lat.begin(), lat.end());

        std::vector<vs::HNSW::Label> L(nq * k);
        std::vector<float> D(nq * k);
        s = Clock::now();
        idx.search_batch(queries.data(), nq, k, ef, L.data(), D.data(), threads);
        double qpsN = double(nq) / secs(s, Clock::now());

        std::printf("%6zu %10.4f %12.0f %12.0f %9.3f %9.3f\n", ef, double(hit) / double(nq * k), qps1,
                    qpsN, lat[nq / 2], lat[size_t(double(nq) * 0.99)]);
    }
    return 0;
}
