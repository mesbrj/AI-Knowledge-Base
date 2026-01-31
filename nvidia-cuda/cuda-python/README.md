[CUDA Python](https://developer.nvidia.com/cuda/python)
- [cuda-python](https://pypi.org/project/cuda-python/)
- [Numba-CUDA](https://nvidia.github.io/numba-cuda/)
- [docs](https://nvidia.github.io/cuda-python/latest/index.html)
- [NVIDIA CUDA wrappers](https://developer.nvidia.com/blog/unifying-the-cuda-python-ecosystem/)

[CuPy](https://cupy.dev/)
- [Installation](https://docs.cupy.dev/en/stable/install.html)

[RAPIDS - cuDF](https://rapids.ai/ecosystem/#featured-software)
- [Quick Start](https://rapids.ai/#quick-start)
- [Installation Guide](https://docs.rapids.ai/install/)


## Installation Steps

**Pre-requisites**:
- *NVIDIA Driver installed on Windows host machine*
- *NVIDIA CUDA Toolkit installed in WSL2 Ubuntu 24.04.3 LTS*

**Steps**:
- Install miniconda
- CUDA Python
- Install CuPy
- Install RAPIDS
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh
conda install numba
conda install -c conda-forge cupy
conda create -n rapids-25.12 -c rapidsai -c conda-forge rapids=25.12 python=3.13 'cuda-version=13.1'
```

