## Relevant Files

Evaluation
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: OpenVLA-specific eval utils
  * `robot_utils.py`: Other eval utils

Training
* `vla-scripts/finetune.py`: VLA fine-tuning script


## Setup

Set up basic environment following `README.md`.

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From base dir
```

## Evaluations

1. Download the CogVLA checkpoint from [HF](https://github.com/JiuTian-VL/CogVLA).

2. Run the scripts to start evaluations.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts-sh/eval.sh
```

Please note that:

- **Setting `--center_crop True` is important** because we fine-tuned CogVLA with random crop augmentations.
- Each evaluation trial result will be automatically saved into `rollouts/` directory.
- The results reported in our paper were obtained using **Python 3.10 and PyTorch 2.2.0**
  on an **NVIDIA A800 GPU**. Please stick to these package versions if possible.
  Note that results may vary slightly if you use a different GPU or environment. If the discrepancy is large,
  please post a GitHub issue, and we will look into it.

## Fine-Tuning

Download the RLDS format of [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds).

The converted datasets are contributed by [OpenVLA-OFT](https://github.com/moojink/openvla-oft), thanks.

```bash
git lfs clone git@hf.co:datasets/openvla/modified_libero_rlds
```

Then, launch the fine-tuning by filling out `scripts-sh/finetune.sh` and running following command:

```bash
bash scripts-sh/finetune.sh
```

- We strongly recommend testing your policy with the **same device/GPU** used to train it! Otherwise, performance may drop substantially. 
- To avoid wasting to many space of your hard disk, you can set `--merge_lora_during_training False` and use [scripts-sh/merge.py](scripts-sh/merge.py) for merging the LoRA adapter into the base model offline.
