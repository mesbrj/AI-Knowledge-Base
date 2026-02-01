# GPU Access in WSL2 containers
Pre-requisites:
- ***NVIDIA Driver installed only on Windows host***

## WSL2 - Ubuntu 24.04.3 LTS

[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

- [With apt: Ubuntu, Debian](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian)

- Verify / Generete [(CDI) Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) (Podman)

```bash
nvidia-ctk cdi list 
# If no CDI devices found, generate them
# INFO[0000] Found 0 CDI devices

sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml

nvidia-ctk cdi list
# INFO[0000] Found 1 CDI devices
# nvidia.com/gpu=all
```

```bash
# If necessary check the systemd services for CDI refresh
nvidia-cdi-refresh.path
nvidia-cdi-refresh.service
```

- Run a container with GPU access

```bash
podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi
```
![](/nvidia-cuda/container-toolkit/nvidia-smi.png)

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  docker.io/nvidia/samples:nbody nbody -benchmark -numbodies=512
```
![](/nvidia-cuda/container-toolkit/nbody-sample.png)
```bash
podman run -it --name tfgpu --device nvidia.com/gpu=all -p 8888:8888 -v "${PWD}:/tf/notebooks" docker.io/tensorflow/tensorflow:latest-gpu-jupyter
```
![](/nvidia-cuda/container-toolkit/tensorflow-gpu-jupyter.png)

### **Pre-requisites for containerized development**:
- *NVIDIA Driver installed on Windows host machine*
- ***NVIDIA Container Toolkit*** installed on WSL2 Ubuntu 24.04.3 LTS

**(1)** Use the [official NVIDIA CUDA base image](https://hub.docker.com/r/nvidia/cuda) with the required CUDA Toolkit version to build and run the desired libraries and applications. Three options are available: `base`, `runtime` and `devel` (SDK) useful for multi-stage builds.

**(2)** If only direct driver access is needed, a common base image such as `ubuntu`, `fedora`, etc... may work as well.
![](/nvidia-cuda/container-toolkit/fedora-base-image.png)
