import torch
import infinicore

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .datatypes import to_torch_dtype, to_infinicore_dtype
from .devices import InfiniDeviceNames, torch_device_map
from .tensor import TensorSpec, TensorInitializer
from .utils import (
    create_test_comparator,
    infinicore_tensor_from_torch,
    profile_operation,
    synchronize_device,
    convert_infinicore_to_torch,
)


class TestCase:
    """Test case with all configuration included"""

    def __init__(
        self,
        inputs,
        kwargs=None,
        output_spec=None,
        comparison_target=None,
        description="",
        tolerance=None,
    ):
        """
        Initialize a test case with complete configuration

        Args:
            inputs: List of TensorSpec objects or scalars
            kwargs: Additional keyword arguments for the operator
            output_spec: TensorSpec for output tensor (for in-place operations)
            comparison_target: Target for comparison ('out', index, or None for return value)
            description: Test case description
            tolerance: Tolerance settings for this test case {'atol': float, 'rtol': float}
        """
        self.inputs = []

        # Process inputs
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                self.inputs.append(TensorSpec.from_tensor(inp))
            elif isinstance(inp, TensorSpec):
                self.inputs.append(inp)
            else:
                self.inputs.append(inp)

        self.kwargs = kwargs or {}
        self.output_spec = output_spec
        self.comparison_target = comparison_target
        self.description = description
        self.tolerance = tolerance or {"atol": 1e-5, "rtol": 1e-3}

    def get_tensor_input_count(self):
        """Count the number of tensor inputs (excluding scalars)"""
        count = 0
        for inp in self.inputs:
            if isinstance(inp, TensorSpec) and not inp.is_scalar:
                count += 1
        return count

    def __str__(self):
        input_strs = []
        for inp in self.inputs:
            if hasattr(inp, "is_scalar") and inp.is_scalar:
                dtype_str = f", dtype={inp.dtype}" if inp.dtype else ""
                input_strs.append(f"scalar({inp.value}{dtype_str})")
            elif hasattr(inp, "shape"):
                dtype_str = f", {inp.dtype}" if inp.dtype else ""
                init_str = (
                    f", init={inp.init_mode}"
                    if inp.init_mode != TensorInitializer.RANDOM
                    else ""
                )
                if hasattr(inp, "strides") and inp.strides:
                    strides_str = f", strides={inp.strides}"
                    input_strs.append(
                        f"tensor{inp.shape}{strides_str}{dtype_str}{init_str}"
                    )
                else:
                    input_strs.append(f"tensor{inp.shape}{dtype_str}{init_str}")
            else:
                input_strs.append(str(inp))

        base_str = f"TestCase("
        if self.description:
            base_str += f"{self.description}"
        base_str += f" - inputs=[{', '.join(input_strs)}]"

        if self.kwargs or self.output_spec:
            kwargs_strs = []
            for key, value in self.kwargs.items():
                if key == "out" and isinstance(value, int):
                    kwargs_strs.append(f"{key}={value}")
                else:
                    kwargs_strs.append(f"{key}={value}")
            output_spec = self.output_spec
            if output_spec and isinstance(output_spec, TensorSpec):
                dtype_str = f", {output_spec.dtype}" if output_spec.dtype else ""
                init_str = (
                    f", init={output_spec.init_mode}"
                    if output_spec.init_mode != TensorInitializer.RANDOM
                    else ""
                )
                if hasattr(output_spec, "strides") and output_spec.strides:
                    strides_str = f", strides={output_spec.strides}"
                    kwargs_strs.append(
                        f"out=tensor{output_spec.shape}{strides_str}{dtype_str}{init_str}"
                    )
                else:
                    kwargs_strs.append(
                        f"out=tensor{output_spec.shape}{dtype_str}{init_str}"
                    )

            base_str += f", kwargs={{{', '.join(kwargs_strs)}}}"

        base_str += ")"
        return base_str


class TestConfig:
    """Test configuration"""

    def __init__(self, debug=False, bench=False, num_prerun=10, num_iterations=1000):
        self.debug = debug
        self.bench = bench
        self.num_prerun = num_prerun
        self.num_iterations = num_iterations


class TestRunner:
    """Test runner"""

    def __init__(self, test_cases, test_config):
        self.test_cases = test_cases
        self.config = test_config
        self.failed_tests = []

    def run_tests(self, devices, test_func, test_type="Test"):
        for device in devices:
            print(f"\n{'='*60}")
            print(f"Testing {test_type} on {InfiniDeviceNames[device]}")
            print(f"{'='*60}")

            for test_case in self.test_cases:
                try:
                    print(f"{test_case}")

                    test_func(device, test_case, self.config)
                    print(f"\033[92m✓\033[0m Passed")
                except Exception as e:
                    error_msg = f"Error: {e}"
                    print(f"\033[91m✗\033[0m {error_msg}")
                    self.failed_tests.append(error_msg)
                    if self.config.debug:
                        raise

        return len(self.failed_tests) == 0

    def print_summary(self):
        if self.failed_tests:
            print(f"\n\033[91m{len(self.failed_tests)} tests failed:\033[0m")
            for failure in self.failed_tests:
                print(f"  - {failure}")
            return False
        else:
            print("\n\033[92mAll tests passed!\033[0m")
            return True


class BaseOperatorTest(ABC):
    """Base operator test"""

    def __init__(self, operator_name):
        self.operator_name = operator_name
        self.test_cases = self.get_test_cases()

    @abstractmethod
    def get_test_cases(self):
        """Return list of TestCase objects with complete configuration"""
        pass

    def torch_operator(self, *args, **kwargs):
        """PyTorch operator function"""
        raise NotImplementedError("torch_operator not implemented")

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore operator function"""
        raise NotImplementedError("infinicore_operator not implemented")

    def prepare_inputs_and_kwargs(self, test_case, device):
        """Prepare inputs and kwargs, replacing TensorSpec objects with actual tensors"""
        inputs = []
        kwargs = test_case.kwargs.copy()

        # Prepare input tensors
        for i, input_spec in enumerate(test_case.inputs):
            if isinstance(input_spec, TensorSpec):
                if input_spec.is_scalar:
                    inputs.append(input_spec.value)
                else:
                    tensor = input_spec.create_torch_tensor(device)
                    inputs.append(tensor)
            else:
                inputs.append(input_spec)

        # Prepare output tensor if specified in output_spec
        if test_case.output_spec is not None:
            output_tensor = test_case.output_spec.create_torch_tensor(device)
            kwargs["out"] = output_tensor

        # Handle integer indices for in-place operations
        if "out" in kwargs and isinstance(kwargs["out"], int):
            input_idx = kwargs["out"]
            if 0 <= input_idx < len(inputs) and isinstance(
                inputs[input_idx], torch.Tensor
            ):
                kwargs["out"] = inputs[input_idx]
            else:
                raise ValueError(
                    f"Invalid input index for in-place operation: {input_idx}"
                )

        return inputs, kwargs

    def run_test(self, device, test_case, config):
        """Unified test execution flow"""
        device_str = torch_device_map[device]

        # Prepare inputs and kwargs with actual tensors
        inputs, kwargs = self.prepare_inputs_and_kwargs(test_case, device)

        # For in-place operations on input tensors, we need to preserve the original state
        original_inputs = []
        if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
            # This is an in-place operation on an input tensor
            # Store original values for comparison
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    original_inputs.append(inp.clone().detach())
                else:
                    original_inputs.append(inp)

        # Create infinicore inputs (cloned to avoid in-place modifications affecting reference)
        infini_inputs = []
        torch_input_clones = []

        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                cloned_inp = inp.clone().detach()
                torch_input_clones.append(cloned_inp)
                infini_tensor = infinicore_tensor_from_torch(cloned_inp)
                infini_inputs.append(infini_tensor)
            else:
                infini_inputs.append(inp)

        # Determine comparison target
        comparison_target = test_case.comparison_target

        # Handle infinicore output
        infini_kwargs = kwargs.copy()
        if "out" in infini_kwargs and isinstance(infini_kwargs["out"], torch.Tensor):
            if isinstance(comparison_target, int):
                infini_kwargs["out"] = infini_inputs[comparison_target]
            else:
                cloned_out = infini_kwargs["out"].clone().detach()
                torch_input_clones.append(cloned_out)
                infini_kwargs["out"] = infinicore_tensor_from_torch(cloned_out)

        # Check operator implementations
        torch_implemented = True
        infini_implemented = True

        try:
            torch_result = self.torch_operator(*inputs, **kwargs)
            if torch_result is None:
                torch_implemented = False
        except NotImplementedError:
            torch_implemented = False
            torch_result = None

        try:
            infini_result = self.infinicore_operator(*infini_inputs, **infini_kwargs)
            if infini_result is None:
                infini_implemented = False
        except NotImplementedError:
            infini_implemented = False
            infini_result = None

        # Skip if neither operator is implemented
        if not torch_implemented and not infini_implemented:
            print(f"⚠ Both operators not implemented - test skipped")
            return

        # Single operator execution without comparison
        if not torch_implemented or not infini_implemented:
            missing_op = (
                "torch_operator" if not torch_implemented else "infinicore_operator"
            )
            print(
                f"⚠ {missing_op} not implemented - running single operator without comparison"
            )

            if config.bench:
                if torch_implemented:

                    def torch_op():
                        return self.torch_operator(*inputs, **kwargs)

                    profile_operation(
                        "PyTorch   ",
                        torch_op,
                        device_str,
                        config.num_prerun,
                        config.num_iterations,
                    )
                if infini_implemented:

                    def infini_op():
                        return self.infinicore_operator(*infini_inputs, **infini_kwargs)

                    profile_operation(
                        "InfiniCore",
                        infini_op,
                        device_str,
                        config.num_prerun,
                        config.num_iterations,
                    )
            return

        if comparison_target is None:
            # Compare return values (out-of-place)
            torch_comparison = torch_result
            infini_comparison = infini_result
        elif comparison_target == "out":
            # Compare output tensor from kwargs (explicit output)
            torch_comparison = kwargs.get("out")
            infini_comparison = infini_kwargs.get("out")
        elif isinstance(comparison_target, int):
            # Compare specific input tensor (in-place operation on input)
            # For in-place operations, we compare the modified input tensor
            if 0 <= comparison_target < len(inputs):
                torch_comparison = inputs[comparison_target]
                infini_comparison = infini_inputs[comparison_target]
            else:
                raise ValueError(
                    f"Invalid comparison target index: {comparison_target}"
                )
        else:
            raise ValueError(f"Invalid comparison target: {comparison_target}")

        # Validate comparison targets
        if torch_comparison is None or infini_comparison is None:
            raise ValueError("Comparison targets cannot be None")

        # Perform comparison
        atol = test_case.tolerance.get("atol", 1e-5)
        rtol = test_case.tolerance.get("rtol", 1e-3)

        compare_fn = create_test_comparator(config, atol, rtol, test_case.description)

        is_valid = compare_fn(infini_comparison, torch_comparison)
        assert is_valid, f"Result comparison failed for {test_case}"

        # Benchmarking
        if config.bench:
            if comparison_target is None:
                # Out-of-place benchmarking
                def torch_op():
                    return self.torch_operator(*inputs, **kwargs)

                def infini_op():
                    return self.infinicore_operator(*infini_inputs, **infini_kwargs)

            else:
                # In-place benchmarking
                def torch_op():
                    self.torch_operator(*inputs, **kwargs)
                    return (
                        kwargs.get("out")
                        if "out" in kwargs
                        else inputs[comparison_target]
                    )

                def infini_op():
                    self.infinicore_operator(*infini_inputs, **infini_kwargs)
                    return (
                        infini_kwargs.get("out")
                        if "out" in infini_kwargs
                        else infini_inputs[comparison_target]
                    )

            profile_operation(
                "PyTorch   ",
                torch_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
            profile_operation(
                "InfiniCore",
                infini_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
