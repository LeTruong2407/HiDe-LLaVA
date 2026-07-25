# Running the Consensus-aware HiDe Method

This guide explains how to train the consensus-aware method from
`propose_method.tex` on the UCIT benchmark. The provided launchers are configured
for Kaggle with two Tesla T4 GPUs, each with 16GB VRAM.

For implementation details and design adaptations, see
`CONSENSUS_METHOD_IMPLEMENTATION.md`.

## 1. Enter the repository

```bash
cd /root/HiDe-LLaVA
```

Install the project if the current Kaggle session has not done so:

```bash
python3 -m pip install -e .
```

## 2. Check models and datasets

The default asset location is:

```text
hide-llava-assets/
```

Check the required files:

```bash
bash scripts/download/check_assets.sh
```

To download and organize all assets that can be fetched automatically:

```bash
bash scripts/download/setup_all_ucit_assets.sh
```

Task 1 requires:

```text
hide-llava-assets/models/llava-v1.5-7b/
hide-llava-assets/models/clip-vit-large-patch14-336/
hide-llava-assets/instructions/ImageNet-R/train.json
hide-llava-assets/datasets/ImageNet-R/train/
```

Do not start the full six-task schedule until all six datasets pass the asset
check.

## 3. Run Task 1 with the new method

Start consensus-aware training on ImageNet-R:

```bash
bash scripts/HiDe/Train_UCIT/run_task1_consensus_kaggle_2x16gb.sh
```

The launcher uses:

- two T4 GPUs through DeepSpeed;
- micro-batch size `1` per GPU;
- gradient accumulation `16`;
- maximum sequence length `768`;
- 4-bit NF4 base-model loading and FP16 computation;
- gradient checkpointing;
- consensus rank `32`;
- shared consensus rank `32`;
- complementary-subspace weight `eta = 0.5`;
- proposal-faithful uniform `1/T` lower-layer fusion;
- at most 64 sampled activation rows per projection.

The settings are defined once in:

```text
scripts/HiDe/Train_UCIT/kaggle_consensus_2x16gb_env.sh
```

Task 1 is written to:

```text
outputs/ucit_consensus/Task1_llava_lora_ours/
```

## 4. Verify Task 1

After Task 1 finishes, check its files:

```bash
ls -lh outputs/ucit_consensus/Task1_llava_lora_ours/
```

The output should include:

```text
adapter_config.json
adapter_model.bin
non_lora_trainables.bin
consensus_subspaces.pt
consensus_summary.json
```

Confirm that consensus statistics were produced:

```bash
python3 -m json.tool \
  outputs/ucit_consensus/Task1_llava_lora_ours/consensus_summary.json \
  > /tmp/task1_consensus_summary.txt

sed -n '1,80p' /tmp/task1_consensus_summary.txt
```

Task 1 has no previous task, so `previous_task_overlap` is expected to be
`null`.

## 5. Run later tasks sequentially

Every task depends on the preceding task's adapter, anchors, and consensus
subspaces. Run only one task at a time and wait for it to finish successfully.

First load the same safe consensus configuration:

```bash
source scripts/HiDe/Train_UCIT/kaggle_consensus_2x16gb_env.sh
```

Then run the remaining tasks:

```bash
bash scripts/HiDe/Train_UCIT/run_task2_consensus_kaggle_2x16gb.sh
bash scripts/HiDe/Train_UCIT/run_task3_consensus_kaggle_2x16gb.sh
bash scripts/HiDe/Train_UCIT/run_task4_consensus_kaggle_2x16gb.sh
bash scripts/HiDe/Train_UCIT/run_task5_consensus_kaggle_2x16gb.sh
bash scripts/HiDe/Train_UCIT/run_task6_consensus_kaggle_2x16gb.sh
```

The required order is:

1. ImageNet-R
2. ArxivQA
3. VizWiz Caption
4. IconQA
5. CLEVR-Math
6. Flickr30k Caption

Do not skip a task. For example, Task 3 expects the Task 2 checkpoint under the
same `UCIT_OUTPUT_ROOT`.

## 6. Run all tasks automatically

After Task 1 and the asset checks are confirmed, the complete schedule can be
run with:

```bash
bash scripts/HiDe/Train_UCIT/run_all_consensus_kaggle_2x16gb.sh
```

This command starts at Task 1. Do not use it to continue from a completed Task
1 because the current training code does not automatically skip finished
tasks. Continue manually from Task 2 instead.

## 7. Analyze consensus subspaces

After two or more tasks, measure pairwise principal-angle overlap:

```bash
python3 scripts/analysis/analyze_consensus_subspaces.py \
  outputs/ucit_consensus/Task3_llava_lora_ours \
  --output outputs/ucit_consensus/task3_subspace_analysis.json
```

The report contains:

- overlap for every pair of task bases;
- mean overlap per decoder layer;
- mean overlap per projection type;
- mean overlap for every instrumented module.

This validates top-subspace agreement. It does not perform the proposal's
top-versus-bottom comparison because bottom singular vectors are intentionally
not retained in training checkpoints.

## 8. Override method settings

Set overrides before starting a launcher. For example:

```bash
export CONSENSUS_ETA=0.25
export CONSENSUS_RANK=16
export CONSENSUS_RANK_SHARED=16
export CONSENSUS_SAMPLE_LIMIT=64

bash scripts/HiDe/Train_UCIT/run_task1_consensus_kaggle_2x16gb.sh
```

Use a different output directory for every ablation:

```bash
export UCIT_OUTPUT_ROOT="$PWD/outputs/ucit_consensus_eta025"
```

The main suggested eta sweep is:

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

`CONSENSUS_RANK_SHARED` must not exceed `CONSENSUS_RANK`, and
`CONSENSUS_SAMPLE_LIMIT` must be at least `CONSENSUS_RANK`.

## 9. OOM safety

The default launcher is intentionally conservative for 2x16GB T4 GPUs. If CUDA
runs out of memory, stop the failed process before retrying and reduce sequence
length first:

```bash
export TRAIN_MODEL_MAX_LENGTH=512
```

If host RAM becomes too high during consensus estimation, reduce the sampled
rows:

```bash
export CONSENSUS_SAMPLE_LIMIT=32
export CONSENSUS_RANK=16
export CONSENSUS_RANK_SHARED=16
```

Keep these settings unchanged during one complete continual sequence. Changing
rank between Task 1 and Task 2 makes experiment comparisons difficult and can
produce incompatible assumptions about the stored subspace bank.

Check for an existing training process before starting another:

```bash
ps -ef | rg 'deepspeed|train_mem_MOE' | rg -v 'rg '
```

Never run two training jobs on the same two GPUs. That is likely to OOM and may
crash the Kaggle session.

## 10. Expected checkpoint flow

```text
base LLaVA
  -> Task1 + task-1 consensus basis
  -> Task2 + task-1/task-2 consensus basis
  -> Task3 + task-1/task-2/task-3 consensus basis
  -> Task4
  -> Task5
  -> Task6 final continual model
```

The persistent consensus state is stored in `consensus_subspaces.pt` beside
each task adapter and is loaded automatically by the next task. The explicit
consensus launchers verify this file before starting Task 2 through Task 6.

For the proposal-to-code fidelity review and the limitations that remain, read
`CONSENSUS_METHOD_AUDIT.md`.
