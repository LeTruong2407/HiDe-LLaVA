# HiDe-LLaVA Training README

This guide covers the full flow:

1. create / activate the environment
2. configure asset paths
3. download model weights
4. download UCIT data and instructions
5. fill missing external datasets
6. launch training

This repo has already been patched so you only need to edit one path file before running the helper scripts.

## 1. Prerequisites

Recommended machine:

- Linux
- NVIDIA GPU
- CUDA-compatible PyTorch environment
- enough disk for:
  - LLaVA base model
  - CLIP model
  - UCIT assets
  - checkpoints

Notes:

- macOS / CPU-only machines are fine for import validation, but not for practical full training.
- Full UCIT training scripts are written for Deepspeed multi-GPU usage.

## 2. Activate the environment

From the repo root:

```bash
conda activate hide-llava
cd path_to_folder
```

If you are on a Linux GPU machine, also install GPU-only extras:

```bash
pip install -r requirements.gpu-linux.txt
```

If you want Flash Attention and your machine supports it:

```bash
pip install flash-attn --no-build-isolation
pip install -U setuptools wheel
```

## 3. Edit one file

Open this file:

`/Research_HiDE/HiDe-LLaVA/scripts/HiDe/paths.sh`

At minimum, check or edit:

- `HIDE_ASSETS_ROOT`
- `LLAVA_BASE_MODEL`
- `CLIP_MODEL`
- `DATA_ROOT`
- `INSTRUCTION_ROOT`
- `UCIT_OUTPUT_ROOT`
- `DEEPSPEED_INCLUDE`
- `MASTER_PORT`

Default layout:

```bash
HiDe-LLaVA/hide-llava-assets/
├── models/
│   ├── llava-v1.5-7b/
│   └── clip-vit-large-patch14-336/
├── datasets/
└── instructions/
```

This matches the current default in `/Users/truongle/Documents/Research_HiDE/HiDe-LLaVA/scripts/HiDe/paths.sh:10`.

## 4. Login to Hugging Face

Some downloads are easier if you log in first:

```bash
huggingface-cli login
```

## 5. Download model weights

Run:

```bash
bash scripts/download/download_models.sh
```

This script:

- downloads `liuhaotian/llava-v1.5-7b`
- downloads `openai/clip-vit-large-patch14-336`
- replaces the downloaded LLaVA `config.json` with the repo-specific one

Expected outputs:

- `$LLAVA_BASE_MODEL`
- `$CLIP_MODEL`

By default:

- `/Users/truongle/Documents/Research_HiDE/HiDe-LLaVA/hide-llava-assets/models/llava-v1.5-7b`
- `/Users/truongle/Documents/Research_HiDE/HiDe-LLaVA/hide-llava-assets/models/clip-vit-large-patch14-336`

## 6. Download UCIT data from Hugging Face

Download everything hosted in the UCIT Hugging Face dataset:

```bash
bash scripts/download/download_ucit_hf.sh
```

If you only want the instruction JSONs first:

```bash
bash scripts/download/download_ucit_hf.sh instructions
```

If you only want the sample subset:

```bash
bash scripts/download/download_ucit_hf.sh sample
```

This script copies:

- instruction JSON files into `$INSTRUCTION_ROOT`
- Hugging Face-hosted images into `$DATA_ROOT` when available

## 7. Download the remaining external image datasets

Not every image source is fully hosted in the UCIT Hugging Face dataset.

Print the external source URLs:

```bash
bash scripts/download/print_external_data_urls.sh
```

You can now use helper scripts for all external datasets:

```bash
bash scripts/download/download_arxivqa.sh
bash scripts/download/download_clevr_math.sh
```

For `IconQA`, provide the current official archive URL from the IconQA site:

```bash
ICONQA_ARCHIVE_URL='https://.../iconqa_data.zip' \
bash scripts/download/download_iconqa.sh
```

For `VizWiz`, provide the current official archive URLs from the VizWiz captioning page:

```bash
VIZWIZ_TRAIN_URL='https://...train.zip' \
VIZWIZ_VAL_URL='https://...val.zip' \
VIZWIZ_TEST_URL='https://...test.zip' \
bash scripts/download/download_vizwiz.sh
```

Or run the wrapper for all external datasets:

```bash
bash scripts/download/download_all_external.sh
```

Why are `IconQA` and `VizWiz` parameterized?

- As of **July 23, 2026**, their official download links may change independently of this repo.
- Their public dataset pages are stable, but their direct archive URLs can be less stable.

You still need current official download URLs for:

- ArxivQA
- CLEVR-Math
- IconQA
- VizWiz

After downloading them, the scripts place them under the directories expected by the repo.

## 8. Required final folder structure

### Models

```bash
$LLAVA_BASE_MODEL
$CLIP_MODEL
```

### Images

The scripts expect a shared image root:

```bash
$DATA_ROOT/
├── ArxivQA/
├── CLEVR/
├── Flickr30k/
├── IconQA/
├── ImageNet-R/
└── VizWiz/
```

### Instruction JSON files

The scripts expect:

```bash
$INSTRUCTION_ROOT/
├── ArxivQA/
│   ├── train_4w.json
│   └── test_3000.json
├── CLEVR/
│   ├── train_4w.json
│   └── test_3000.json
├── Flickr30k/
│   ├── train_brief_4w.json
│   ├── test_3000.json
│   └── val_coco_type_3000.json
├── IconQA/
│   ├── train.json
│   └── test_3000.json
├── ImageNet-R/
│   ├── train.json
│   └── test_3000.json
└── VizWiz/
    ├── train.json
    ├── test_3000.json
    └── val_coco_type_3000.json
```

These defaults are already wired in:

`/Users/truongle/Documents/Research_HiDE/HiDe-LLaVA/scripts/HiDe/paths.sh`

## 9. Quick sanity checks before training

### Check imports

```bash
conda run -n hide-llava python -m llava.train.train_MOE --help
```

### Check key paths

```bash
source scripts/HiDe/paths.sh
echo "$LLAVA_BASE_MODEL"
echo "$CLIP_MODEL"
echo "$DATA_ROOT"
echo "$INSTRUCTION_ROOT"
echo "$UCIT_OUTPUT_ROOT"
```

### Verify the first task JSON exists

```bash
source scripts/HiDe/paths.sh
ls "$UCIT_TASK1_TRAIN_JSON"
```

### Check all required assets quickly

```bash
bash scripts/download/check_assets.sh
```

## 10. Local smoke test first

If you only want to verify that the code path launches on this machine, run the 1-step CPU/local smoke test:

```bash
conda activate hide-llava
cd /Users/truongle/Documents/Research_HiDE/HiDe-LLaVA
bash scripts/HiDe/Train_UCIT/smoke_local.sh
```

This forces:

- `TRAIN_LAUNCHER=python`
- batch size `1`
- `--max_steps 1`
- `--no_cuda True`

It still requires real model files, instruction JSON, and images to exist under the configured paths.

## 11. Start training

### Train only task 1

```bash
conda activate hide-llava
cd /Users/truongle/Documents/Research_HiDE/HiDe-LLaVA
bash scripts/HiDe/Train_UCIT/Task1.sh
```

### Train all UCIT tasks sequentially

```bash
conda activate hide-llava
cd /Users/truongle/Documents/Research_HiDE/HiDe-LLaVA
bash scripts/HiDe/Train_UCIT/train_all.sh
```

## 12. What each training script does

UCIT training order:

1. `Task1.sh` -> ImageNet-R
2. `Task2.sh` -> ArxivQA
3. `Task3.sh` -> VizWiz
4. `Task4.sh` -> IconQA
5. `Task5.sh` -> CLEVR
6. `Task6.sh` -> Flickr30k

Each later task loads:

- the base LLaVA model
- the same CLIP model
- the previous task checkpoint through `--previous_task_model_path`

Outputs are written under:

- `$UCIT_OUTPUT_ROOT/Task1_llava_lora_ours`
- `$UCIT_OUTPUT_ROOT/Task2_llava_lora_ours`
- ...
- `$UCIT_OUTPUT_ROOT/Task6_llava_lora_ours`

## 13. Common commands

### Re-download models

```bash
bash scripts/download/download_models.sh
```

### Re-download UCIT Hugging Face assets

```bash
bash scripts/download/download_ucit_hf.sh
```

### Download a tiny real `ImageNet-R` subset for local tests

```bash
bash scripts/download/download_imagenet_r_tiny.sh 32
bash scripts/download/make_tiny_imagenet_r_split.sh 32
```

This creates:

- `hide-llava-assets/datasets/ImageNet-R/...` with a small set of real images
- `hide-llava-assets/instructions/ImageNet-R/train_tiny_32.json`
- `hide-llava-assets/instructions/ImageNet-R/test_tiny_32.json`

You can then point local smoke or Kaggle quick tests at the tiny JSON.

### Download a tiny real `Flickr30k` subset for local tests

```bash
bash scripts/download/download_flickr30k_tiny.sh 32
bash scripts/download/make_tiny_flickr30k_split.sh 32
```

This creates:

- `hide-llava-assets/datasets/Flickr30k/train/...`
- `hide-llava-assets/datasets/Flickr30k/val/...`
- `hide-llava-assets/instructions/Flickr30k/train_tiny_32.json`
- `hide-llava-assets/instructions/Flickr30k/test_tiny_32.json`
- `hide-llava-assets/instructions/Flickr30k/val_tiny_32.json`

### Train and evaluate tiny `Flickr30k` locally

```bash
bash scripts/download/download_flickr30k_tiny.sh 32
bash scripts/download/make_tiny_flickr30k_split.sh 32
bash scripts/HiDe/Train_UCIT/flickr_tiny_local.sh
bash scripts/HiDe/Eval_UCIT/eval_flickr30k_tiny_local.sh
```

This is the best small real-data train/eval path for a local Mac check.

### Show missing external dataset sources

```bash
bash scripts/download/print_external_data_urls.sh
```

### Run only task 3 later

```bash
bash scripts/HiDe/Train_UCIT/Task3.sh
```

### Force non-Deepspeed training

```bash
export TRAIN_LAUNCHER=python
bash scripts/HiDe/Train_UCIT/Task1.sh
```

## 14. Troubleshooting

### `huggingface-cli: command not found`

Activate the environment first:

```bash
conda activate hide-llava
```

### `deepspeed: command not found`

That is now optional for local testing.

Use the Python fallback:

```bash
export TRAIN_LAUNCHER=python
bash scripts/HiDe/Train_UCIT/Task1.sh
```

Or run the prepared smoke test:

```bash
bash scripts/HiDe/Train_UCIT/smoke_local.sh
```

If you want multi-GPU Deepspeed on Linux later, install:

```bash
pip install -r requirements.gpu-linux.txt
```

### Model path not found

Check:

```bash
source scripts/HiDe/paths.sh
ls "$LLAVA_BASE_MODEL"
ls "$CLIP_MODEL"
```

### Instruction JSON not found

Check:

```bash
source scripts/HiDe/paths.sh
find "$INSTRUCTION_ROOT" -maxdepth 2 -type f | sort
```

### Image folders not found

Check:

```bash
source scripts/HiDe/paths.sh
find "$DATA_ROOT" -maxdepth 2 -type d | sort
```

## 15. Short version

If you just want the minimum command sequence:

```bash
conda activate hide-llava
cd /Users/truongle/Documents/Research_HiDE/HiDe-LLaVA
huggingface-cli login
bash scripts/download/download_models.sh
bash scripts/download/download_ucit_hf.sh
bash scripts/download/print_external_data_urls.sh
# use helper scripts for external datasets
bash scripts/download/download_arxivqa.sh
bash scripts/download/download_clevr_math.sh
# fill current official URLs first for the next two:
# ICONQA_ARCHIVE_URL=...
# VIZWIZ_TRAIN_URL=...
# VIZWIZ_VAL_URL=...
# VIZWIZ_TEST_URL=...
bash scripts/download/check_assets.sh
bash scripts/HiDe/Train_UCIT/smoke_local.sh
```

If task 1 works, run:

```bash
bash scripts/HiDe/Train_UCIT/train_all.sh
```