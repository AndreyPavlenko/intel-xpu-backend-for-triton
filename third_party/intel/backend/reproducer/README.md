# Standalone reproducer

This module contains a generator of a standalone reproducer for the Intel XPU backend for Triton. The reproducer is a
simple C++ program that allows to run a single kernel on the Intel XPU device. The reproducer is generated from the
serialized data produced by the Triton backend.

## Creating a reproducer

To create a reproducer, all you need is to just set the `TRITON_XPU_CREATE_REPRODUCER` environment variable to the path
where the reproducer should be created and run a kernel on the Intel XPU device.

#### Here is an example of how to create a reproducer for a simple kernel, that multiplies two tensors:

```python
import os

import numpy as np
import torch
import triton
import triton.language as tl

# Set a destination folder for the reproducer.
os.environ["TRITON_XPU_CREATE_REPRODUCER"] = "reproducer"


# Define the kernel
@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


# Optionally, create a reference data to compare the results with.
shape = (16, 32, 8)
x_cpu = torch.rand(shape, dtype=torch.bfloat16, device="cpu")
y_cpu = torch.rand(shape, dtype=torch.bfloat16, device="cpu")
output_cpu = (x_cpu * y_cpu)

# Create the input and output tensors.
x = x_cpu.to("xpu")
y = y_cpu.to("xpu")
output = torch.empty_like(x)

# Optionally, attach the reference data to the output tensors. If attached, the reproducer
# will have a generated code, that compares the outputs with the reference data.
output.compare_with = output_cpu
# If the tensor has a floating point data, you can also specify the comparison tolerance.
output.compare_with_rtol = 1e-05
output.compare_with_atol = 1e-07
output.compare_with_equal_nan = True

# Run the kernel
n_elements = int(np.prod(shape))
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

After running the script, you will find the reproducer in the specified folder. The reproducer is a cmake project, that
allows to launch the kernel on the Intel XPU device without using python. This is a pure C++ program that can be used to
debug and benchmark the kernel. To launch the reproducer, you may either use the provided `launch.sh` script or import
it as a cmake project in your favourite IDE.
