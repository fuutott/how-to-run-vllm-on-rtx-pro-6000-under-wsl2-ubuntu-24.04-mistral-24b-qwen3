# How to run VLLM on RTX PRO 6000 (cuda 12.8) under WSL2 Ubuntu 24.04 on windows 11 to play around with mistral 24b 2501, 2503, and qwen 3

Tried and tested on 6th June 2025. 

windows nvidia-smi 
> NVIDIA-SMI 576.52                 Driver Version: 576.52         CUDA Version: 12.9

wsl2 ubuntu nvidia-smi
> NVIDIA-SMI 575.57.04              Driver Version: 576.52         CUDA Version: 12.9

## Guide

*Before we start. WSL2 by default allocates 1TB for vhdx and let's it grow.*

*Consider your system storage layout. I've set mine to 256gb default size in Start > WSL Settings > File sytem > Default VHD Size*

*I've also enabled WSL Settings > Optional Features > Enable sparse VHD by default*  



Install Ubuntu 24.04.1 LTS from microsoft store and start, go through initial setup, once done get in and 


```bash
apt-get update apt-get upgrade
```


(Optional) Once update finishes you can move Ubuntu vhdx to where you want it on your storage, resize if you expect to work with bigger models:


```bash
wsl --shutdown
wsl --manage Ubuntu-24.04 --move D:\wsl\Ubuntu-24.04 
wsl --shutdown && wsl --manage Ubuntu-24.04 --resize 512GB
wsl --set-default Ubuntu-24.04
```


Go back to ubuntu.
Follow steps to install Cuda toolkit 12.8 as per guide in the link below:

Linux > x86_64 > WSL-Ubuntu > 2.0 > deb (network)

https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network


```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```


Once cuda toolkit is installed set paths to nvcc and confirm versions


```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
nvcc --version # verify that nvcc is in your PATH
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME
```


Create project directory


```bash
mkdir vllm-local-build
cd vllm-local-build
```


Get uv to manage venv:


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh 
source $HOME/.local/bin/env
uv venv --python 3.12 --seed
```


(Optional but recommended) Before you activate venv add this at the end of venv/bin/activate script of your choice with right syntax.
I got multiple gpus, nvidia-smi says rtx pro 6000 is id 0 so I'm limiting vllm to just visible device 0


```bash
EXPORT HF_TOKEN=your hf token
EXPORT CUDA_DEVICE_ORDER=PCI_BUS_ID
EXPORT CUDA_VISIBLE_DEVICES=0    
EXPORT TORCH_CUDA_ARCH_LIST="12.0"
```


Activate venv


```bash
source .venv/bin/activate
```


Install torch cu128


```bash
pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
```


Confirm torch version


```bash
python - <<'PY'
import torch, torchvision, torchaudio
print(torch.__version__, "CUDA:", torch.version.cuda)
print(torchvision.__version__)
print(torchaudio.__version__)
PY
```


This will be required by xformers


```bash
sudo apt install python3.12-dev
pip install ninja
```


Install xformers for mistral vision 


```bash
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers --no-build-isolation
```


Install flashinfer as it seems faster


```bash
RUN git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
RUN pip install --no-build-isolation -e . -v
```


Get back to project root


```bash
cd ..
```

Build vllm with existing torch 


```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```


Install bitsandbytes for quantization


```bash
pip install bitsandbytes
```

That's it time to 

## Run some models


**google/gemma-3-1b-it** If all went well this should just work:


```bash
vllm serve "google/gemma-3-1b-it"
```


**mistralai/Mistral-Small-24B-Instruct-2501** this should also be ok


```bash
vllm serve mistralai/Mistral-Small-24B-Instruct-2501 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --dtype bfloat16 \
    --max-model-len 32000 \
    --max-seq-len-to-capture 32000 \
    --quantization bitsandbytes
```


**mistralai/Mistral-Small-3.1-24B-Instruct-2503** this works, bitsandbytes do not support mistral's pixtral in this version so quants dropped


```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-seq-len-to-capture 65536 
```


**Qwen/Qwen3-30B-A3B** this works quite well


```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype bfloat16 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072
```


**OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym** this loads but super slow inference


```bash
vllm serve OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym \
    --load_format auto \
    --dtype auto \
    --max-model-len 65536 \
    --max-seq-len-to-capture 65536 
```


##HERE BE DRAGONS:



> --quantization, -q
> Possible choices: aqlm, awq, deepspeedfp, fp8, fbgemm_fp8, marlin, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, squeezellm, compressed-tensors, bitsandbytes, qqq, None




> --dtype
> Possible choices: auto, half, float16, bfloat16, float, float32
> Data type for model weights and activations.
> “auto” will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
> “half” for FP16. Recommended for AWQ quantization.
> “float16” is the same as “half”.
> “bfloat16” for a balance between precision and range.
> “float” is shorthand for FP32 precision.
> “float32” for FP32 precision.




> --load-format
> Possible choices: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral
> The format of the model weights to load.
> “auto” will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.
> “pt” will load the weights in the pytorch bin format.
> “safetensors” will load the weights in the safetensors format.
> “npcache” will load the weights in pytorch format and store a numpy cache to speed up the loading.
> “dummy” will initialize the weights with random values, which is mainly for profiling.
> “tensorizer” will load the weights using tensorizer from CoreWeave. See the Tensorize vLLM Model script in the Examples section for more information.
> “bitsandbytes” will load the weights using bitsandbytes quantization.

