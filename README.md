# MMedAgent-Plus

MMedAgent-Plus is a multimodal medical tool-use agent. A vision-language planner, fine-tuned from Qwen2.5-VL-7B, decides which specialist models to call, in what order, and how to pass intermediate results between them. It then turns the tool outputs into a final answer.

This repository contains the code to **fine-tune the planner**. The benchmark, the evaluation scripts and the dataset-construction scripts are in the companion repository [MMedAgent-Bench](https://github.com/RoxanneWAAANG/MMedAgent-Bench).

> **Status:** partial release while the paper is under review. The agent deployment code (`agent/`) and the interactive demo will be added later.

## Repository layout

```
agent/                    agent deployment (to be released)
qwenvl/
  train/train_qwen.py     training entry point
  train/trainer.py        attention patch + optimizer groups
  train/argument.py       model / data / training arguments
  data/__init__.py        dataset registry (reads MMEDAGENT_DATA_ROOT)
  data/data_qwen.py       supervised dataset + collator
  data/data_qwen_packed.py
  data/rope2d.py
scripts/
  sft_7b.sh               full SFT, the configuration used in the paper
  deepspeed/              ZeRO-2 / ZeRO-3 / ZeRO-3-offload configs
```

The training code is adapted from the [Qwen2.5-VL fine-tuning framework](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune) (Apache-2.0).

## Installation

```bash
conda create -n mmedagent python=3.11 -y
conda activate mmedagent
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

## Data

Test data is hosted on Hugging Face: ![Roxanne-WANG/MMedAgent-V2](https://huggingface.co/datasets/Roxanne-WANG/MMedAgent-V2).

Images is stored on Google Drive: [instruction_dataset_images](https://drive.google.com/drive/folders/1-AfZ5Ox1nDgs1hNvNyGBWSBX8cwC0xtu?usp=sharing)

```bash
huggingface-cli download Roxanne-WANG/MMedAgent-V2 --repo-type dataset --local-dir ./data
export MMEDAGENT_DATA_ROOT=$(pwd)/data
```

Expected layout (image paths in the annotation files are relative to `MMEDAGENT_DATA_ROOT`):

```
$MMEDAGENT_DATA_ROOT/
  train/
    single_tool_multi_round.jsonl
    multi_tool_multiround.jsonl
    multi_tool_single_round.jsonl
  tool_image/
    train/...
```

Each line is one dialogue in the Qwen-VL conversation format:

```json
{
  "image": "tool_image/train/xxx.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nSegment the lesion in this ultrasound."},
    {"from": "gpt", "value": "...", "actions": [{"API_name": "MedSAM", "API_params": {"...": "..."}}]},
    {"from": "human", "value": "MedSAM output: ..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

The dataset names accepted by `--dataset_use` are defined in [qwenvl/data/__init__.py](qwenvl/data/__init__.py). Append `%N` to subsample, for example `multi_tool_single_round%50`.

Some source datasets have licenses that do not permit redistribution. For those, the Hugging Face release contains only the annotations and sample indices, and you need to download the images from the original source. See the dataset card for the per-source list.

## Training

```bash
export MMEDAGENT_DATA_ROOT=/path/to/data
bash scripts/sft_7b.sh
```

These environment variables can be overridden:

| Variable | Default |
|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `OUTPUT_DIR` | `./checkpoints/mmedagent-plus-7b` |
| `NPROC_PER_NODE` | `4` |
| `DEEPSPEED_CONFIG` | `./scripts/deepspeed/zero3_offload.json` |

Paper configuration: 208,928 training samples; the vision encoder is frozen, and the connector and language model are trained. AdamW, lr 1e-5, weight decay 0.01, cosine schedule with 10% warmup, gradient clipping 1.0, bf16, one epoch, effective batch size 256 (4 × A100-80GB × batch 1 × accumulation 64).

If you train on a different number of GPUs, adjust `grad_accum_steps` in the script to keep the effective batch size at 256.

## Model weights

The fine-tuning weight is hosted on Hugging Face: ![ZihaoLin/mmedagent-nov24](https://huggingface.co/ZihaoLin/mmedagent-nov24).

```bash
huggingface-cli download ZihaoLin/mmedagent-nov24 --local-dir ./checkpoints/mmedagent-plus-7b
```

## Evaluation

See [MMedAgent-Bench](https://github.com/RoxanneWAAANG/MMedAgent-Bench).

<!-- ## Citation

```bibtex
@article{mmedagentplus,
  title  = {MMedAgent-Plus: Scaling Multimodal Medical Agents across Modalities, Tasks and Workflows},
  note   = {Under review},
  year   = {2026}
}
```

## License

Code: Apache-2.0 (see [LICENSE](LICENSE)). The data and model weights are released under separate terms; see the Hugging Face cards. -->
