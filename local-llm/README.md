# Pre-requisite on WSL2
- ***NVIDIA Driver installed only on Windows host***
- [***NVIDIA Container Toolkit***](https://github.com/mesbrj/GPU-Computing-Knowledge-Base/blob/main/nvidia-cuda/container-toolkit/README.md) installed on WSL2 for conteinerized deployment

## Ollama
- [**Hardware support**](https://docs.ollama.com/gpu)
- [**Ollama Container image**](https://hub.docker.com/r/ollama/ollama)
- [**Ollama documentation**](https://docs.ollama.com/)
```shell
# Running ollama container with GPU support
podman run -d -v /home/mesb/.ollama:/root/.ollama -p 11434:11434 --device nvidia.com/gpu=all --name ollama ollama/ollama:0.15.6
```
![](/local-llm/ollama-container.png)

```shell
# Pulling models from ollama library
podman exec -it ollama ollama pull gemma3:12b
# Inference test
podman exec -it ollama ollama run gemma3:12b "What are the key features of LangChain Framework?"
```
![](/local-llm/ollama-logs.png)

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

