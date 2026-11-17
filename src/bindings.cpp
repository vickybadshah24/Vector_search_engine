// pybind11 bindings: `import vsearch`
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include "hnsw.hpp"

namespace py = pybind11;
using FloatArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

class Index {
public:
    Index(size_t dim, size_t max_elements, const std::string& space, size_t M, size_t ef_construction,
          uint64_t seed)
        : index_(dim, max_elements, parse_space(space), M, ef_construction, seed) {}

    void add_items(FloatArr data, py::object ids, size_t num_threads) {
        auto [ptr, n] = rows(data);
        std::vector<int64_t> labels;
        if (!ids.is_none()) {
            LabelArr a = LabelArr::ensure(ids);
            if (!a || size_t(a.size()) != n) throw std::invalid_argument("ids must have one entry per row");
            labels.assign(a.data(), a.data() + n);
        } else {
            for (size_t i = 0; i < n; ++i) labels.push_back(int64_t(index_.size() + i));
        }
        py::gil_scoped_release nogil;
        index_.add_batch(ptr, labels.data(), n, num_threads);
    }

    py::tuple knn_query(FloatArr data, size_t k, py::object ef, size_t num_threads) {
        auto [ptr, nq] = rows(data);
        size_t ef_v = ef.is_none() ? 0 : ef.cast<size_t>();
        py::array_t<int64_t> labels({nq, k});
        py::array_t<float> dists({nq, k});
        auto* lp = labels.mutable_data();
        auto* dp = dists.mutable_data();
        {
            py::gil_scoped_release nogil;
            index_.search_batch(ptr, nq, k, ef_v, lp, dp, num_threads);
        }
        return py::make_tuple(labels, dists);
    }

    void set_ef(size_t ef) { index_.set_ef(ef); }
    size_t size() const { return index_.size(); }
    size_t dim() const { return index_.dim(); }
    size_t max_elements() const { return index_.max_elements(); }
    size_t memory_bytes() const { return index_.memory_bytes(); }

private:
    static vs::Metric parse_space(const std::string& s) {
        if (s == "l2") return vs::Metric::L2;
        if (s == "cosine") return vs::Metric::Cosine;
        throw std::invalid_argument("space must be 'l2' or 'cosine'");
    }
    std::pair<const float*, size_t> rows(const FloatArr& a) const {
        if (a.ndim() == 1 && size_t(a.shape(0)) == index_.dim()) return {a.data(), 1};
        if (a.ndim() == 2 && size_t(a.shape(1)) == index_.dim()) return {a.data(), size_t(a.shape(0))};
        throw std::invalid_argument("expected array of shape (n, " + std::to_string(index_.dim()) + ")");
    }
    vs::HNSW index_;
};

PYBIND11_MODULE(vsearch, m) {
    m.doc() = "HNSW approximate nearest-neighbour search (C++17, AVX2, multithreaded)";
    py::class_<Index>(m, "Index")
        .def(py::init<size_t, size_t, const std::string&, size_t, size_t, uint64_t>(), py::arg("dim"),
             py::arg("max_elements"), py::arg("space") = "l2", py::arg("M") = 16,
             py::arg("ef_construction") = 200, py::arg("seed") = 100)
        .def("add_items", &Index::add_items, py::arg("data"), py::arg("ids") = py::none(),
             py::arg("num_threads") = 0)
        .def("knn_query", &Index::knn_query, py::arg("data"), py::arg("k") = 10,
             py::arg("ef") = py::none(), py::arg("num_threads") = 0)
        .def("set_ef", &Index::set_ef)
        .def("memory_bytes", &Index::memory_bytes)
        .def("__len__", &Index::size)
        .def_property_readonly("dim", &Index::dim)
        .def_property_readonly("max_elements", &Index::max_elements);
}
