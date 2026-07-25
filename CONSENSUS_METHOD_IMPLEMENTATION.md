# Consensus-aware HiDe Implementation

This document records how the method in `propose_method.tex` is mapped onto
the current HiDe-LLaVA code, including choices made where the proposal and
repository behavior do not line up exactly.

## Implemented method

For every LoRA-attached projection below the final decoder layer, training
samples a bounded buffer of that projection's input activations. At the end
of each task, uncentered PCA is computed directly from the activation matrix
with `torch.pca_lowrank(center=False)`. Only the top task basis is retained.

All task bases for a projection are stacked, and another low-rank decomposition
produces the consensus basis `U_c`. The dense projector is never materialized.
Inference computes

```text
x_filtered = eta * x + (1 - eta) * U_c (U_c^T x)
```

and applies the uniform average of all learned lower-layer LoRA updates to
`x_filtered`. The final
decoder layer remains the original HiDe sample-dependent mixture based on CLIP
image/text anchors.

The task bases and current consensus basis are stored in
`consensus_subspaces.pt`. Per-projection sample count, retained sketched energy,
and overlap with the previous task basis are stored in
`consensus_summary.json`.

## Implementation adaptations

### The code does not contain a single persistent lower-layer expert

The proposal describes HiDe's lower layers as one merged General Expert. This
repository actually retains one `A` and `B` slot per task and fuses them in the
forward pass. The implementation therefore filters the activation before
summing the stored task updates. This is functionally equivalent to applying
the proposed projector to a sum of task LoRA updates, while preserving
sequential checkpoints.

### The original lower-layer code introduces cross-expert terms

The baseline implementation separately sums all `A_t` matrices and all `B_t`
matrices, then multiplies the sums. This yields cross terms `B_i A_j` for
`i != j`, which are absent from both conventional LoRA expert fusion and the
proposal. Consensus mode instead computes the mathematically intended sum of
`B_t A_t`. Consequently, `eta=1` is the conventional unfiltered expert-sum
baseline, but it is not bit-for-bit equivalent to the repository's cross-term
behavior.

### Uniform fusion coefficients

The repository has no implementation of HiDe's learned fusion coefficients; its
lower-layer code hardcodes a weight of one. Consensus mode implements the
proposal's explicit uniform rule and divides the fused update by the number of
learned tasks. Set `--consensus_normalize_fusion False` only to reproduce the
old repository's unnormalized scale as an ablation.

### Bounded sampling for Kaggle

Keeping full token activation matrices for roughly 217 projections is not
practical on 2x16GB Kaggle sessions. Each projection uses reservoir sampling to
maintain a bounded CPU sample representative of the full task stream. The
Kaggle launcher defaults to 64 rows and two candidate tokens per forward.
This is deliberately conservative; 128 or 256 rows should improve subspace
estimation when host RAM permits.

### Distributed estimation

Each process observes its own data shard. Only rank 0 finalizes and saves the
subspaces, so the current implementation estimates bases from rank 0's bounded
sample buffer. This avoids a large distributed object gather and is a
reasonable first implementation for shuffled two-GPU training. A later
experiment can all-gather the small sample buffers before PCA if rank-local
estimates prove noisy.

### Quantized training

The original 4-bit factory bypasses `HiDeMOELoraLinear` and creates generic
single-expert PEFT layers. Consensus mode now uses a dedicated bitsandbytes
4-bit HiDe layer so task experts and activation collection remain active.
Consensus mode rejects 8-bit loading because an equivalent 8-bit expert wrapper
has not been implemented. Dense 16-bit and 4-bit paths are supported.

## Hyperparameters

- `--consensus_enable`: enable the method; default is `False`.
- `--consensus_rank`: rank retained for each task activation subspace.
- `--consensus_rank_shared`: rank of the consensus subspace.
- `--consensus_eta`: complementary-subspace weight in `[0, 1]`.
- `--consensus_normalize_fusion`: use the proposal's `1/T` uniform average.
- `--consensus_sample_limit`: maximum sampled activation rows per projection.
- `--consensus_samples_per_forward`: sampled rows per projection and forward.
- `--consensus_oversample`: randomized PCA oversampling rank.

Run only Task 1 with the low-memory consensus configuration:

```bash
bash scripts/HiDe/Train_UCIT/run_task1_consensus_kaggle_2x16gb.sh
```

Start the full low-memory UCIT schedule with:

```bash
bash scripts/HiDe/Train_UCIT/run_all_consensus_kaggle_2x16gb.sh
```

For the proposal's required pilot, use the consensus Task 1 through Task 3 launchers with
the same environment variables and inspect `consensus_summary.json`. The
`previous_task_overlap` values are mean cosines of principal angles between the
new task basis and the immediately preceding task basis. Compute all pairwise
top-subspace overlaps from a task checkpoint with:

```bash
python3 scripts/analysis/analyze_consensus_subspaces.py \
  outputs/ucit_consensus/Task3_llava_lora_ours \
  --output outputs/ucit_consensus/task3_subspace_analysis.json
```

This reports overlap by module, decoder layer, and projection type. The
proposal's top-versus-bottom comparison is not available from the compact
checkpoint because bottom singular vectors are deliberately not retained; run
that comparison as a separate activation-extraction pilot if it is required
for the formal empirical claim.
