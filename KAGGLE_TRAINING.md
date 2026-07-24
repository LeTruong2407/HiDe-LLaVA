# Kaggle Training Guide

Use this guide when you move from local smoke testing to a real Linux GPU run on Kaggle.

## 1. Environment

Inside Kaggle terminal or notebook shell:

```bash
cd /kaggle/working/HiDe-LLaVA
conda activate hide-llava || true
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements.gpu-linux.txt
python -m pip install -e .
```

Optional:

```bash
python -m pip install flash-attn --no-build-isolation
```

## 2. Configure paths

Edit:

`scripts/HiDe/paths.sh`

Typical Kaggle-style choices:

- `HIDE_ASSETS_ROOT=/kaggle/working/HiDe-LLaVA/hide-llava-assets`
- `UCIT_OUTPUT_ROOT=/kaggle/working/HiDe-LLaVA/outputs/ucit`

## 3. Download assets

```bash
huggingface-cli login
bash scripts/download/download_models.sh
bash scripts/download/download_ucit_hf.sh
```

Then download the external datasets as needed:

```bash
bash scripts/download/download_arxivqa.sh
bash scripts/download/download_clevr_math.sh
```

For datasets with changing official archive links:

```bash
ICONQA_ARCHIVE_URL='https://.../iconqa_data.zip' bash scripts/download/download_iconqa.sh

VIZWIZ_TRAIN_URL='https://...train.zip' \
VIZWIZ_VAL_URL='https://...val.zip' \
VIZWIZ_TEST_URL='https://...test.zip' \
bash scripts/download/download_vizwiz.sh
```

Check assets:

```bash
bash scripts/download/check_assets.sh
```

## 4. Launch options

The UCIT task scripts now auto-detect `deepspeed`.

- If `deepspeed` exists and `TRAIN_LAUNCHER=auto`, they use `deepspeed`
- Otherwise they fall back to `python -m llava.train.train_mem_MOE`

You can force plain Python on Kaggle single-GPU runs:

```bash
export TRAIN_LAUNCHER=python
```

Or force Deepspeed:

```bash
export TRAIN_LAUNCHER=deepspeed
```

## 5. Train

Task 1 only:

```bash
bash scripts/HiDe/Train_UCIT/Task1.sh
```

All UCIT tasks:

```bash
bash scripts/HiDe/Train_UCIT/train_all.sh
```

## 6. Recommended Kaggle fallback

If Deepspeed is unavailable or unstable on Kaggle:

```bash
export TRAIN_LAUNCHER=python
export TRAIN_PER_DEVICE_BATCH=1
export TRAIN_GRAD_ACCUM_STEPS=8
bash scripts/HiDe/Train_UCIT/Task1.sh
```

Then scale up carefully if memory allows.
