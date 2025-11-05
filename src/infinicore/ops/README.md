# infinicore::ops 开发指南

infinicore::ops 模块包含了 InfiniCore 所有 C++ 算子的接口和实现。外部用户可以通过 `include/infinicore/ops/*OPNAME*/*OPNAME*.h` 中定义的 C++ 接口进行算子调用。部分算子会通过 pybind 暴露给 python 前端。

## 开发指南

### 1. 算子定义

创建 `include/infinicore/ops/*OPNAME*/*OPNAME*.h` 头文件，并根据算子名称定义算子的类以及外部计算接口（包括 in-place 和 out-of-place 两种模式），注意算子名称不能重复。

一个算子类主要包含以下部分：

- schema 定义，用于描述算子的输入输出参数形式。
- execute 函数，算子的计算逻辑。
- dispatcher 分发器，用于注册算子在不同设备上的 kernel 实现。一个进程中，一种算子只有一个全局分发器，每种设备上只能同时注册一个 kernel 实现，可以多次注册对之前的实现进行覆盖。详细信息请参考 `include/infinicore/ops/common/dispatcher.hpp`。

示例 `Matmul` 算子的头文件如下：

```c++
#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Matmul {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor matmul(Tensor a, Tensor b);
void matmul_(Tensor c, Tensor a, Tensor b);
}
```

### 2. 算子实现

在 `src/infinicore/ops/*OPNAME*/*OPNAME*.cpp` 文件中实现算子的计算逻辑。

- execute 函数，使用算子的分发器，调用对应硬件上的核函数。
- 计算接口，使用 execute 函数实现算子接口的计算逻辑，包括 in-place 和 out-of-place 两种模式，其中 in-place 模式的接口函数名以 `_` 结尾，将输出接口写入给定的参数中；out-of-place 模式的接口会为输出创建新的 Tensor。

示例 `Matmul` 算子的实现如下：

```c++
#include "infinicore/ops/matmul.hpp"

namespace infinicore::op {

common::OpDispatcher<Matmul::schema> &Matmul::dispatcher() {
    static common::OpDispatcher<Matmul::schema> dispatcher_;
    return dispatcher_;
};

void Matmul::execute(Tensor c, Tensor a, Tensor b) {
    dispatcher().lookup(context::getDevice().getType())(c, a, b);
}

Tensor matmul(Tensor a, Tensor b) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    matmul_(c, a, b);
    return c;
}

void matmul_(Tensor c, Tensor a, Tensor b) {
    Matmul::execute(c, a, b);
}
}
```

### 3. Kernel 注册

在 `src/infinicore/ops/*OPNAME*/` 目录中添加算子和函数实现，并在算子的分发器中进行注册。你可以选择为单个设备、多个设备、或全部平台注册 kernel 实现（函数指针），你还可以通过使用 `override_existing` 模式覆盖之前的实现。具体信息请参考 `include/infinicore/ops/common/dispatcher.hpp`：

```c++
// 为某个设备注册 kernel 实现
void registerDevice(Device::Type device_type, Fn fn, bool override_existing = true);

// 为多个设备注册 kernel 实现
void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true);

// 为全部平台注册 kernel 实现
void registerAll(Fn fn, bool override_existing = true);

// 查找 kernel 实现
Fn lookup(Device::Type device_type) const;
```

如果你为多个（或全部）设备注册了同一个 kernel 实现，那么你需要自行实现不同设备的分发机制。比如本框架中的 InfiniOP 算子库，其算子接口在不同平台都保持了一致，并根据当前设备类型自动分发，因此在注册时会为所有平台注册同一个计算函数。以 Matmul 算子为例：

```c++
namespace infinicore::op::matmul_impl::infiniop {

// InfiniOP 算子缓存（线程级）
thread_local common::OpCache<size_t, infiniopGemmDescriptor_t> caches(
    100,
    [](infiniopGemmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyGemmDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数
void calculate(Tensor c, Tensor a, Tensor b){
    // ...
    INFINICORE_CHECK_ERROR(infiniopGemm(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), 1.f, 0.f, context::getStream()));
}

// 在加载 InfiniCore 时为全平台注册 InfiniOP实现
static bool registered = []() {
    Matmul::dispatcher().registerAll(&calculate, false);
    return true;
}();

}
```

你可以仿照上面的例子单独为不同平台实现核函数并注册。请注意在 `xmake/*lua` 中添加对源文件的编译方式，并做好跨平台隔离工作以保证项目在别的平台上也可以正常编译。你可以选择像上面的例子一样，通过 `static bool registered = []() {...}` 方式在加载时注册核函数，但请注意避免加载时为同一个算子重复注册不同核函数的未定义行为。你也可以在程序运行时显式地注册算子。

如果你想通过 InfiniOP 库来实现算子，请参考 [`InfiniOP 开发者文档`](src/infiniop/README.md) 文件。

### 4. Python 接口

通过 pybind11 将 C++ 算子暴露给 Python 前端，需要在 `src/infinicore/pybind11/ops/*OPNAME*/` 目录中添加相应的头文件，并在 `src/infinicore/pybind11/ops.hpp` 中调用。之后你需要在 `python/infinicore/ops/` 目录中为算子添加一个 Python 文件，通过调用你刚才定义的 pybind 接口实现你的 Python 接口，并将 Python 接口通过 `python/infinicore/__init__.py` 暴露给外部。

### 5. Python 测试

在实现了 Python 接口后，你需要在 `/test/infinicore/ops/` 中添加相应的算子测试脚本，并确保测试通过。该目录下的测试使用了统一的测试框架，大部分测试功能已经实现，比如根据形状构建随机张量、自动测试算子的正确性和性能等。你需要继承 `BaseOperatorTest` 类并实现 `get_test_cases`、`get_tensor_dtypes`、`get_tolerance_map`、`torch_operator`、`infinicore_operator` 等跟算子有关的方法。其中 `torch_operator` 为对比用的 pytorch 版算子实现，而 `infinicore_operator` 为你所实现的 InfiniCore 版算子。以 silu 算子为例：

```python
class OpTest(BaseOperatorTest):
    """SiLU test with simplified test case parsing"""

    def __init__(self):
        super().__init__("SiLU")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, input, out=None, **kwargs):
        # SiLU implementation: input * sigmoid(input)
        sigmoid_input = torch.sigmoid(input)
        result = input * sigmoid_input
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        return infinicore.silu(input, out=out)
```

在测试脚本中你需要为算子测试脚本添加测例。请参考 `TestCase` 类的定义，提供输入输出张量的形状、数据类型、步长，以及其他参数的数值等。你可以指定算子计算是否涉及 in-place 或 out-of-place 模式。你可以像示例一样将测例写的更简洁，并通过 `parse_test_cases` 函数来解析测例数据。

```python
_TEST_CASES_DATA = [
    # Basic 2D SiLU
    (TestCase.BOTH, (2, 4), None, None),
    (TestCase.BOTH, (128, 64), None, None),
    # 3D SiLU
    (TestCase.BOTH, (2, 4, 8), None, None),
    (TestCase.BOTH, (4, 48, 6), None, None),
    # Strided tensors
    (TestCase.BOTH, (1, 2048), (4096, 1), (4096, 1)),
    (TestCase.BOTH, (6, 2560), (2048, 1), (2560, 1)),
    # Mixed cases
    (TestCase.BOTH, (8, 16, 32), None, None),
    # Large tensors
    (TestCase.BOTH, (16, 5632), None, None),
    (TestCase.BOTH, (4, 4, 5632), None, None),
]

def parse_test_cases(data):
    """
    Parse silu test case data according to format:
    (operation_mode, shape, input_strides, output_strides)
    """
    operation_mode = data[0]
    shape = data[1]
    input_strides = data[2] if len(data) > 2 else None
    output_strides = data[3] if len(data) > 3 else None

    # Create input specifications
    inputs = []

    # Tensor input
    if input_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, input_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Output tensor
    if output_strides is not None:
        output = TensorSpec.from_strided_tensor(shape, output_strides)
    else:
        output = TensorSpec.from_tensor(shape)

    return TestCase(operation_mode, inputs, output)


# Parse test cases
_TEST_CASES = [parse_test_cases(data) for data in _TEST_CASES_DATA]
```

对于支持多种精度的算子，你可以指定测试通过的误差范围。

```python
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}
```

运行测试指令检查算子的正确性和性能：

```bash
python test/infinicore/run.py --ops matmul --nvidia --verbose --bench
```
