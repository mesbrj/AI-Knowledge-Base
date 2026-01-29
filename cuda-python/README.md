[CUDA Python](https://developer.nvidia.com/cuda/python)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
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
- Install miniconda
- CUDA Python
- Install CuPy
- Install RAPIDS
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh
conda install numba
conda install cudatoolkit
conda install -c conda-forge cupy
conda create -n rapids-25.12 -c rapidsai -c conda-forge rapids=25.12 python=3.13 'cuda-version=13.0'
```

