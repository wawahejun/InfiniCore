#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Gemm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, float, float);
    static void execute(Tensor c, Tensor a, Tensor b, float alpha, float beta);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);
void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta);

} // namespace infinicore::op
