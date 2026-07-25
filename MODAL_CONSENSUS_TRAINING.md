# Modal A100 Consensus Training

This runs the consensus-aware HiDe Task 1 training on `ImageNet-R` using one
Modal `A100-40GB` GPU.

## 1. Install Modal locally

```bash
python -m pip install modal
modal setup
```

## 2. Put assets in the Modal volume

The Modal app expects this volume:

```text
hide-llava-assets
```

It must contain the same layout used locally:

```text
models/llava-v1.5-7b/
models/clip-vit-large-patch14-336/
instructions/ImageNet-R/train.json
datasets/ImageNet-R/train/
```

Upload your local assets with Modal CLI, for example:

```bash
modal volume create hide-llava-assets
modal volume put hide-llava-assets hide-llava-assets/models /models
modal volume put hide-llava-assets hide-llava-assets/instructions /instructions
modal volume put hide-llava-assets hide-llava-assets/datasets/ImageNet-R /datasets/ImageNet-R
```

Outputs are written to a second volume:

```bash
modal volume create hide-llava-outputs
```

## 3. Check assets remotely

If you want Modal to download the Task 1 assets into the volume:

```bash
modal run modal_imagenet_r_consensus.py --action prepare-assets
```

If you uploaded assets manually, check them:

```bash
modal run modal_imagenet_r_consensus.py --action check-assets
```

## 4. Run a short pilot

```bash
modal run modal_imagenet_r_consensus.py --action train --max-steps 20
```

## 5. Run Task 1 training

```bash
modal run modal_imagenet_r_consensus.py --action train
```

The checkpoint is stored in the `hide-llava-outputs` volume:

```text
ucit_consensus_modal_a100/Task1_llava_lora_ours/
```

Expected consensus files:

```text
adapter_config.json
adapter_model.bin
non_lora_trainables.bin
consensus_subspaces.pt
consensus_summary.json
```

## 6. Tune A100 settings

Defaults are conservative for a 40GB A100:

```text
train_batch=4
grad_accum=8
model_max_length=1024
sample_limit=128
```

Override them from the Modal command:

```bash
modal run modal_imagenet_r_consensus.py \
  --action train \
  --train-batch 6 \
  --grad-accum 8 \
  --model-max-length 1024 \
  --sample-limit 128
```

If CUDA runs out of memory, lower `--train-batch` first, then
`--model-max-length`.
