#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/gemm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_gemm(py::module &m) {
    m.def("gemm",
          &op::gemm,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          R"doc(General matrix multiplication: C = alpha * A @ B + beta * C.)doc");

    m.def("gemm_",
          &op::gemm_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha"),
          py::arg("beta"),
          R"doc(In-place general matrix multiplication.)doc");
}

} // namespace infinicore::ops
