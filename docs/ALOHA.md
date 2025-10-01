## Relevant Files

Evaluation
* `experiments/robot/aloha/`: ALOHA training and eval files
  * `run_aloha_eval.py`: ALOHA eval script (CLIENT SIDE)
  * `aloha_utils.py`: ALOHA eval utils
  * Other ALOHA robot environment files copied from the original [ALOHA GitHub repo](https://github.com/tonyzhaozh/aloha):
    * `constants.py`
    * `real_env.py`
    * `robot_utils.py`
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: OpenVLA-specific eval utils
  * `robot_utils.py`: Other eval utils
* `vla-scripts/deploy.py`: VLA server deploy script (SERVER SIDE)

Note: Unlike the LIBERO evaluation setup, we use a server-client interface here. This is particularly useful if the user's machine which commands the robot does not have access to a local GPU with sufficient specs to run the fine-tuned VLA policies.

Training
* `experiments/robot/aloha/`: ALOHA training and eval files
  * `preprocess_split_aloha_data.py`: ALOHA data preprocessing script
* `vla-scripts/finetune.py`: VLA fine-tuning script

## Fine-Tuning

We assume that you have collected a set of expert demonstrations on the ALOHA robot already.

1. First, use `preprocess_split_aloha_data.py` to preprocess the raw ALOHA dataset: downsize images from 480x640 to 256x256 and split into training and validation sets. For example:

```bash
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /data/aloha1_raw/put_green_pepper_into_pot/ \
  --out_base_dir /data/aloha1_preprocessed/ \
  --percent_val 0.05
```

2. Then, convert the preprocessed ALOHA datasets into a single RLDS dataset that is compatible with OpenVLA fine-tuning. This process is the same as in the OpenVLA repo. See instructions for converting to RLDS [here](https://github.com/moojink/rlds_dataset_builder) (a sample ALOHA preprocessed-to-RLDS conversion script is available [here](https://github.com/moojink/rlds_dataset_builder/blob/main/aloha1_put_X_into_pot_300_demos/aloha1_put_X_into_pot_300_demos_dataset_builder.py)).

3. After converting to RLDS, register the dataset with dataloader by adding an entry for it in `configs.py` ([here](../prismatic/vla/datasets/rlds/oxe/configs.py#L680)), `transforms.py` ([here](../prismatic/vla/datasets/rlds/oxe/transforms.py#L928)), and `mixtures.py` ([here](../prismatic/vla/datasets/rlds/oxe/mixtures.py#L216)).

4. Before fine-tuning, set the desired ALOHA action chunk size in [`prismatic/vla/constants.py`](../prismatic/vla/constants.py) (see `NUM_ACTIONS_CHUNK` in `ALOHA_CONSTANTS`). 

**Do NOT** modify `ACTION_PROPRIO_NORMALIZATION_TYPE`, Since the ALOHA robot action space is absolute joint angles, we do not want to use a normalization scheme that clips outlier values (like the Q1-Q99 normalization we used with the relative end-effector pose actions for LIBERO), since that would prevent the model from outputting certain robot joint angles that are crucial for solving the task.

5. Then, launch the fine-tuning by filling out `scripts-sh/finetune_aloha.sh` and running following command:

```bash
bash scripts-sh/finetune_aloha.sh
```

- We strongly recommend testing your policy with the **same device/GPU** used to train it! Otherwise, performance may drop substantially. 
- To avoid wasting to many space of your hard disk, you can set `--merge_lora_during_training False` and use [scripts-sh/merge.py](scripts-sh/merge.py) for merging the LoRA adapter into the base model offline.

## Evaluations

On the **server-side**, install a few additional packages for the server-client interface:

```bash
pip install uvicorn fastapi json-numpy
```

Then launch the VLA server by:

```bash
bash scripts-sh/eval_aloha_deploy.sh
```

On the **client-side**, install a few additional packages for ALOHA robot execution:

```bash
pip install -r experiments/robot/aloha/requirements_aloha.txt
```

Then, specify the VLA server URL or IP address in the `vla_server_url` argument and run:

```bash
bash scripts-sh/eval_aloha_run.sh
```
