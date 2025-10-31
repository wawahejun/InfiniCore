from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gemm(input, other, alpha=1.0, beta=0.0, *, out=None):
    if out is None:
        return Tensor(_infinicore.gemm(input._underlying, other._underlying, alpha, beta))

    _infinicore.gemm_(out._underlying, input._underlying, other._underlying, alpha, beta)
