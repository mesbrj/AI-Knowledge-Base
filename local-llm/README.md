# Pre-requisite

- ***NVIDIA Driver installed on Linux Host or only in Windows host for WSL2***
- [***NVIDIA Container Toolkit***](https://github.com/mesbrj/GPU-Computing-Knowledge-Base/blob/main/nvidia-cuda/container-toolkit/README.md) installed on Linux Host or in WSL2 for conteinerized deployment
>
- **OBS: NVIDIA Container Toolkit 1.19.1 (current version on 07/2026) uses CDI spec version 0.7.0**

    Only podman 5.1.0 and above support CDI spec version 0.7.0. The Ubuntu 22.04 LTS last podman version is 4.9.3, which does not support CDI spec version 0.7.0. The above workaround downgrades the NVIDIA CDI spec version to 0.6.0 and strips the additionalGids blocks (spec 0.7.0) from the nvidia.yaml file.

```shell
# 1. Back up the original and downgrade to 0.6.0 and strip the additionalGids blocks
sudo cp /etc/cdi/nvidia.yaml /etc/cdi/nvidia.yaml.bak
awk '
  /^cdiVersion: 0.7.0$/ { print "cdiVersion: 0.6.0"; next }
  /^        additionalGids:$/ { g=1; next }
  g==1 && /^            - [0-9]+$/ { next }
  { g=0 } { print }
' /etc/cdi/nvidia.yaml.bak | sudo tee /etc/cdi/nvidia.yaml >/dev/null

# 2. Verify the changes
sed -n '2p' /etc/cdi/nvidia.yaml            # -> cdiVersion: 0.6.0
grep -c additionalGids /etc/cdi/nvidia.yaml # -> 0

# 3. start and test ollama (Confirm the GPU is visible in the container)
podman run -d -v /home/mesb/.ollama:/root/.ollama -p 11434:11434 --device nvidia.com/gpu=all --name ollama ollama/ollama
podman exec ollama nvidia-smi

# If you ever re-run nvidia-ctk cdi generate (e.g., after a driver update), you'll need to reapply the step 1, since it regenerates a 0.7.0 spec.
```

## Ollama

- [**Hardware support**](https://docs.ollama.com/gpu)
- [**Ollama Container image**](https://hub.docker.com/r/ollama/ollama)
- [**Ollama documentation**](https://docs.ollama.com/)

```shell
# Running ollama container with GPU support
podman run -d -v /home/mesb/.ollama:/root/.ollama -p 11434:11434 --device nvidia.com/gpu=all --name ollama ollama/ollama
```

![ollama container](/local-llm/ollama-container.png)

```shell
# Pulling models from ollama library
podman exec -it ollama ollama pull gemma3:12b
# Inference test
podman exec -it ollama ollama run gemma3:12b "What are the key features of LangChain Framework?"
```

![ollama logs](/local-llm/ollama-logs.png)

### **Tested Models (ollama library)**

- [gemma3:12b Q4_K_M (size ~8.1GB)](https://ollama.com/library/gemma3:12b)
- [llama3.1:8b Q8_0 (size ~8.5GB)](https://ollama.com/library/llama3.1:8b-instruct-q8_0)
- [deepseek-r1:14b Q4_K_M (size ~9.0GB)](https://ollama.com/library/deepseek-r1:14b)
- [nomic-embed-text:v1.5 F16 (size ~274MB)](https://ollama.com/library/nomic-embed-text:v1.5)

## llama.cpp

- [Local gpt-oss models](https://github.com/ggml-org/llama.cpp/discussions/15396)

## NVIDIA NIM [(NVIDIA Inference Microservices)](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)

- [Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html)
- [Get Started with NVIDIA NIM for LLMs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html)
