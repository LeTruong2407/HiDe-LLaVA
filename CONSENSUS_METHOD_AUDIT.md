# Consensus Method Audit

This document compares `propose_method.tex` with the implementation used by the
UCIT Task 1 through Task 6 continual sequence.

## Implemented mathematical contract

For every LoRA projection below the final decoder layer and every completed task
`t`, the code stores a rank-`k` basis `U_t` estimated from uncentered input
activations. It stacks those bases and computes the leading rank-`k_c` left
singular vectors `U_c`, which is equivalent to extracting the dominant
eigenspace of the average task projectors without materializing a dense
`d x d` matrix.

At inference, it computes

```text
x_eta = eta * x + (1 - eta) * U_c (U_c^T x)
Delta y = (1 / T) * sum_t B_t A_t x_eta
```

By linearity this is exactly

```text
Delta W_G = (1 / T) * sum_t [Delta W_t P_c
                              + eta * Delta W_t (I - P_c)]
```

which is the uniform-coefficient boxed fusion rule in the proposal. The final
decoder layer is excluded and continues to use HiDe image/text-anchor routing.

## Repository-specific adaptation

The proposal describes one materialized remaining-layer General Expert. This
repository preallocates six rank-8 task slots inside a total LoRA rank of 48.
The implementation evaluates the exact fused update lazily as a sum of
`B_t A_t` terms instead of materializing a dense `Delta W_G` or recompressing it
to rank 8. This avoids the incorrect cross terms produced by multiplying a sum
of `A_t` factors by a sum of `B_t` factors.

The functional fusion rule is exact, but its effective rank can grow to 48.
Recompressing the sum back to rank 8 would be a different, lossy method and is
not implemented. Parameter allocation remains fixed for this six-task codebase
because all six slots already exist from Task 1.

## Task 1 through Task 6 state flow

Each consensus launcher injects the same method arguments. Task `t > 1` loads:

- all prior LoRA slots from Task `t - 1`;
- image/text routing anchors and boundaries;
- all per-projection task bases and the current consensus basis.

After training task `t`, its new basis is appended and a new consensus basis is
saved in `consensus_subspaces.pt`. Later-task launchers refuse to run if the
previous consensus artifact is missing.

## Corrections made by this audit

- Lower-layer task updates now use `1 / T` uniform fusion normalization.
- Activation collection now uses reservoir sampling over the whole task rather
  than retaining only rows from the first batches.
- Captured energy is measured against total activation energy, not only the
  singular values returned by the randomized sketch.
- Anchors are non-gradient FP32 running statistics; checkpoints containing
  NaN/Inf are rejected.
- Automatic top-layer routing only includes experts already trained.

## What is not established yet

The implementation makes the proposal executable, but it does not prove an
accuracy enhancement. The proposal explicitly requires a principal-angle pilot
before making that claim. `scripts/analysis/analyze_consensus_subspaces.py`
measures pairwise overlap among retained top subspaces after Task 2 or Task 3.
The proposed top-versus-bottom comparison requires a separate controlled
activation pilot because bottom/null directions are intentionally not retained
in continual checkpoints.

The repository also does not implement learned HiDe fusion coefficients
`epsilon_t`; its baseline hardcodes a weight of one. The current method therefore
uses the proposal's uniform-average rule. Learned coefficients are a separate
extension and should not be claimed as part of these experiments.

## Ablations required for a meaningful result

Use separate output roots and compare at least:

- `eta = 1`: uniform unfiltered fusion control;
- `eta = 0.5`: default soft consensus filtering;
- `eta = 0`: hard consensus filtering;
- automatic routing versus the known task expert at the final layer;
- Task 1, Task 2/3 pilot, and final Task 6 average accuracy.

An enhancement is supported only if soft/hard consensus improves retained-task
or average continual accuracy over `eta = 1` under the same training budget and
seed.
