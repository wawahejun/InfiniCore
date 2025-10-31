#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

Tensor matmul(Tensor a, Tensor b);
void matmul_(Tensor c, Tensor a, Tensor b);

} // namespace infinicore::op
