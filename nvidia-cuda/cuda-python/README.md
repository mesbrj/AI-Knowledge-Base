# [CUDA Python](https://developer.nvidia.com/cuda/python)

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/latest/index.html)
- [NVIDIA CUDA wrappers (NVIDIA Blog)](https://developer.nvidia.com/blog/unifying-the-cuda-python-ecosystem/)

# CUDA Python Ecosystem Overview

- [Nvidia CUDA Toolkit 13.1](https://developer.nvidia.com/cuda/toolkit) - Full SDK and Runtime: compiled libraries, JIT libnvvm (LLVM compiler), NVRTC (C++ string runtime compilation), CUDA NVCC (C/C++ Compiler), headers, profiling and debugging tools

- Python [cuda-toolkit](https://pypi.org/project/cuda-toolkit/) (PyPI) meta-package - Runtime (only): libnvvm, NVRTC and compiled libraries

- Python [cuda-python](https://pypi.org/project/cuda-python/) (PyPI) meta-package - Python bindings for CUDA Driver and Runtime APIs.

- [Numba](https://numba.readthedocs.io/en/stable/index.html) (CPU intesive tasks and paralelism) / [Numba-CUDA](https://nvidia.github.io/numba-cuda/) (GPU computing) - Just-in-time compiler for Python that translates a subset of Python and NumPy code into fast machine code (CPU and GPU) using LLVM. The minimal requirement for Numba-CUDA is the [CUDA runtime](https://numba.readthedocs.io/en/stable/cuda/overview.html#software).


# Python Libraries for GPU Computing

- [PyTorch](https://pytorch.org/get-started/locally/)
- [CuPy (NumPy and SciPy)](https://cupy.dev/)
- [RAPIDS - cuDF (mirrored pandas API)](https://rapids.ai/ecosystem/#featured-software)
- [Numba-CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html) python [bindings](https://numba.readthedocs.io/en/stable/cuda/overview.html#cuda-bindings)
    - Own ctypes-based bindings for default.
    - [cuda-python](https://pypi.org/project/cuda-python/) will be used by default in a future Numba release.

## Installation Steps

### **Pre-requisites for local development**:
- *NVIDIA Driver installed on Windows host machine*
- ***NVIDIA CUDA Toolkit (SDK)*** installed in WSL2 Ubuntu 24.04.3 LTS*

`numba-cuda`, `cuda-python`, `CuPy`, `RAPIDS` and `PyTorch` will be installed via pip to always use the CUDA toolkit SDK installed globally to build and run your libraries.
```bash
# Environment variables for CUDA Toolkit SDK
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### **Pre-requisites for containerized development**:
- *NVIDIA Driver installed on Windows host machine*
- ***NVIDIA Container Toolkit*** installed on WSL2 Ubuntu 24.04.3 LTS

Use the [official NVIDIA CUDA base image](https://hub.docker.com/r/nvidia/cuda) with the required CUDA Toolkit version to build and run the desired libraries. Three options are available: `base`, `runtime` and `devel` (SDK) useful for multi-stage builds.

