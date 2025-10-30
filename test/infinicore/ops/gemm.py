import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from infinicore.ops.gemm import gemm as ic_gemm
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (operation_mode, nbatch, m, n, k, a_strides, b_strides, c_strides)
# If nbatch is None: a_shape=(m, k), b_shape=(k, n), c_shape=(m, n)
# If nbatch is provided: a_shape=(nbatch, m, k), b_shape=(nbatch, k, n), c_shape=(nbatch, m, n)
# Aligned with test/infiniop/gemm.py shapes/strides and per-case alpha/beta
# Each item: (alpha, beta, operation_mode, nbatch, m, n, k, a_strides, b_strides, c_strides)
_TEST_CASES_DATA = [
    # (1) alpha=1.0, beta=0.0, a=(1,2048), b=(2048,2048), c=(1,2048)
    (1.0, 0.0, TestCase.BOTH, None, 1, 2048, 2048, None, None, None),
    # (2) alpha=1.0, beta=0.0, a=(2,4,2048), b=(2,2048,2048), c=(2,4,2048)
    (1.0, 0.0, TestCase.BOTH, 2, 4, 2048, 2048, None, None, None),
    # (3) alpha=1.0, beta=0.0, strided (4096,1)
    (1.0, 0.0, TestCase.BOTH, None, 1, 2048, 2048, (4096, 1), (4096, 1), (4096, 1)),
    # (4) alpha=1.0, beta=1.0, only meaningful for IN_PLACE (needs existing C)
    (1.0, 1.0, TestCase.IN_PLACE, None, 6, 2560, 2048, (2048, 1), (1, 2048), (2560, 1)),
    # (5) alpha=1.0/8.0, beta=0.0, a=(4,48,64), b=(4,64,6), c=(4,48,6)
    (1.0 / 8.0, 0.0, TestCase.BOTH, 4, 48, 6, 64, None, None, None),
]


def parse_test_cases(data):
    """
    Parse gemm test case data according to format:
    (operation_mode, nbatch, m, n, k, a_strides, b_strides, c_strides)
    """
    alpha = data[0]
    beta = data[1]
    operation_mode = data[2]
    nbatch = data[3]
    m, n, k = data[4], data[5], data[6]
    a_strides = data[7] if len(data) > 7 else None
    b_strides = data[8] if len(data) > 8 else None
    c_strides = data[9] if len(data) > 9 else None

    # Determine shapes based on batch dimension
    if nbatch is None:
        a_shape = (m, k)
        b_shape = (k, n)
        c_shape = (m, n)
    else:
        a_shape = (nbatch, m, k)
        b_shape = (nbatch, k, n)
        c_shape = (nbatch, m, n)

    # Create input specifications
    inputs = []

    # Tensor a
    if a_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(a_shape, a_strides))
    else:
        inputs.append(TensorSpec.from_tensor(a_shape))

    # Tensor b
    if b_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(b_shape, b_strides))
    else:
        inputs.append(TensorSpec.from_tensor(b_shape))

    # Output tensor
    if c_strides is not None:
        output = TensorSpec.from_strided_tensor(
            c_shape,
            c_strides,
            init_mode=TensorInitializer.ONES if beta != 0.0 else TensorInitializer.RANDOM,
        )
    else:
        output = TensorSpec.from_tensor(
            c_shape,
            init_mode=TensorInitializer.ONES if beta != 0.0 else TensorInitializer.RANDOM,
        )

    return TestCase(operation_mode, inputs, output, alpha=alpha, beta=beta)


# Parse test cases
_TEST_CASES = [parse_test_cases(data) for data in _TEST_CASES_DATA]

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}


class OpTest(BaseOperatorTest):
    """GEMM test with simplified test case parsing

    Note: We test default alpha=1.0 and beta=0.0 so it should match torch.matmul.
    """

    def __init__(self):
        super().__init__("Gemm")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, a, b, out=None, **kwargs):
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 0.0)
        mm = torch.matmul(a, b)
        if out is None:
            return mm.mul(alpha)
        out.mul_(beta)
        out.add_(mm, alpha=alpha)
        return out

    def infinicore_operator(self, a, b, out=None, **kwargs):
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 0.0)
        if out is None:
            return ic_gemm(a, b, alpha=alpha, beta=beta)
        return ic_gemm(a, b, alpha=alpha, beta=beta, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
