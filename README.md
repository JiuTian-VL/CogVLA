

## Installation

```bash
# Create and activate conda environment
conda create -n cogvla python=3.10 -y
conda activate cogvla

# Clone CogVLA repo and pip install to download dependencies
git clone git@github.com:JiuTian-VL/CogVLA.git
cd CogVLA
pip install -e .

# Install Flash Attention 2 for training
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Quick Start

Download the checkpoint from [Hugging Face](https://github.com/JiuTian-VL/CogVLA). 

Fill the checkpoint path in `demo.py`. Then run the following command

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py
```

## Training and Evaluation

See [LIBERO.md](docs/LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ALOHA.md](docs/ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.
