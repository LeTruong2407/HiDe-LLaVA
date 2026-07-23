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
$HOME/hide-llava-assets/
├── models/
│   ├── llava-v1.5-7b/
│   └── clip-vit-large-patch14-336/
├── datasets/
└── instructions/
```

If you are happy with that layout, you may not need to edit much.

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

- `$HOME/hide-llava-assets/models/llava-v1.5-7b`
- `$HOME/hide-llava-assets/models/clip-vit-large-patch14-336`

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

You still need to manually download some image datasets such as:

- ArxivQA
- CLEVR-Math
- IconQA
- VizWiz

After downloading them, place them under the directories expected by the repo.

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

## 10. Start training

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

## 11. What each training script does

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

## 12. Common commands

### Re-download models

```bash
bash scripts/download/download_models.sh
```

### Re-download UCIT Hugging Face assets

```bash
bash scripts/download/download_ucit_hf.sh
```

### Show missing external dataset sources

```bash
bash scripts/download/print_external_data_urls.sh
```

### Run only task 3 later

```bash
bash scripts/HiDe/Train_UCIT/Task3.sh
```

## 13. Troubleshooting

### `huggingface-cli: command not found`

Activate the environment first:

```bash
conda activate hide-llava
```

### `deepspeed: command not found`

You are probably not on the Linux GPU setup yet. Install:

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

## 14. Short version

If you just want the minimum command sequence:

```bash
conda activate hide-llava
cd /Users/truongle/Documents/Research_HiDE/HiDe-LLaVA
huggingface-cli login
bash scripts/download/download_models.sh
bash scripts/download/download_ucit_hf.sh
bash scripts/download/print_external_data_urls.sh
# manually download any remaining external image datasets into $DATA_ROOT
bash scripts/HiDe/Train_UCIT/Task1.sh
```

If task 1 works, run:

```bash
bash scripts/HiDe/Train_UCIT/train_all.sh
```
