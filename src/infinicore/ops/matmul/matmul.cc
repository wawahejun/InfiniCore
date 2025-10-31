#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op {

Tensor matmul(Tensor a, Tensor b) {
    return gemm(a, b, 1.0f, 0.0f);
}

void matmul_(Tensor c, Tensor a, Tensor b) {
    Gemm::execute(c, a, b, 1.0f, 0.0f);
}
} // namespace infinicore::op
