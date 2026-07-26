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

The default training profile is `balanced`, tuned to use more of an A100 40GB
than the original conservative run:

```text
train_batch=8
grad_accum=4
model_max_length=1536
sample_limit=256
samples_per_forward=8
```

To push harder:

```bash
modal run modal_imagenet_r_consensus.py --action train --profile aggressive
```

To fall back to the older low-memory settings:

```bash
modal run modal_imagenet_r_consensus.py --action train --profile safe
```

To stop a running Modal app:

```bash
bash scripts/modal/stop_modal_training.sh
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

## 6. Evaluate Task 1

Run a short eval first:

```bash
modal run modal_imagenet_r_consensus.py --action eval --eval-max-samples 100
```

Run the full ImageNet-R test split:

```bash
modal run modal_imagenet_r_consensus.py --action eval
```

The result is saved in the `hide-llava-outputs` volume:

```text
results/UCIT/each_dataset/ImageNet-R/consensus-modal-a100-task1/Result.text
```

Download it locally:

```bash
modal volume get hide-llava-outputs \
  results/UCIT/each_dataset/ImageNet-R/consensus-modal-a100-task1/Result.text \
  Result.text
```

## 7. Tune A100 settings

Profile defaults:

```text
safe:       train_batch=4,  grad_accum=8, model_max_length=1024, sample_limit=128
balanced:   train_batch=8,  grad_accum=4, model_max_length=1536, sample_limit=256
aggressive: train_batch=12, grad_accum=3, model_max_length=2048, sample_limit=256
```

Override them from the Modal command:

```bash
modal run modal_imagenet_r_consensus.py \
  --action train \
  --profile balanced \
  --train-batch 10 \
  --grad-accum 4 \
  --model-max-length 1536 \
  --sample-limit 256 \
  --samples-per-forward 8
```

If CUDA runs out of memory, lower `--train-batch` first, then
`--model-max-length`.
