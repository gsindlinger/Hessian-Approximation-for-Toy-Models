# Proposed staged experiments — awaiting run confirmation

The user has authorized preparing scripts and configurations, but has explicitly
requested confirmation before any experiment, training job, or smoke test runs.
The files in this directory have not been executed. Numerical validation remains
pending. Each numerical entry point defaults to displaying its plan and requires
`--execute` to do work. `plan.py` only prints commands and never launches them.

The priority is **new model/dataset pairs → PBRF on those saved models → unrolling**.
EoS, LLM work, pseudoinverse-mechanism work, and CK-FAC implementation are outside
this batch. The first proposal uses the existing dense model families; a tiny
transformer remains a subsequent architecture extension, not an implemented arm.

## 1. Train and save the base models

The proposed first batch has eight pairs, with training seeds 0, 1, and 2 for
each: **24 base models**. The model seed varies initialization and training order;
dataset split and subset seeds are fixed independently and shared by both
architectures. All settings below are proposals for review.

| GPU | Pair ID | Dataset | Architecture | Initial SGD LR | Epochs |
|---|---|---|---|---:|---:|
| 0 | `mnist_mlp8` | MNIST | tanh MLP, 8 hidden layers of width 16 | 0.1 | 50 |
| 1 | `mnist_resnet3` | MNIST | ReLU residual MLP, 3 blocks of widths [16,16] | 0.02 | 50 |
| 2 | `fashion_mnist_mlp8` | Fashion-MNIST | tanh MLP, 8 hidden layers of width 16 | 0.1 | 50 |
| 3 | `fashion_mnist_resnet3` | Fashion-MNIST | ReLU residual MLP, 3 blocks of widths [16,16] | 0.02 | 50 |
| 4 | `energy_mlp2` | Energy, both regression targets | tanh MLP, widths [32,32] | 0.01 | 500 |
| 5 | `energy_resnet3` | Energy, both regression targets | ReLU residual MLP, 3 blocks of widths [16,16] | 0.01 | 500 |
| 6 | `concrete_mlp2` | Concrete | tanh MLP, widths [32,32] | 0.01 | 500 |
| 7 | `concrete_resnet3` | Concrete | ReLU residual MLP, 3 blocks of widths [16,16] | 0.01 | 500 |

Each GPU processes its pair's three seeds sequentially. No GPU assignment is a
reservation or running job. The launch must check availability again after
confirmation. Do not start separate stage processes on the same GPU concurrently.

- Images: fixed 6,000-example subset of the canonical training split; 1,000
  query examples from the separate canonical test split. Classification loss is
  cross-entropy. Input standardization is fitted on the training subset only.
- Regression: fixed 80/20 split. Standardize inputs and outputs using training
  statistics only. Loss is mean squared error averaged over output dimensions.
- All models use plain SGD, cosine LR decay, batch size 64, and no weight decay.
- Save the actual initial weights, intermediate checkpoints, final weights,
  training/evaluation curves, sample indices, normalization statistics, and full
  provenance. Downstream stages load these artifacts rather than training a new
  base model or reconstructing a different data split.
- Use local cached datasets. Missing data is an explicit error; the scripts do
  not silently download datasets or log to external services.

Proposed output layout:

```text
experiments/results/staged_pairs/<pair_id>/seed_<seed>/
```

Inspect training curves, finite losses/weights, and task performance before
selecting checkpoints for the next stage. A completed training schedule is not
automatically evidence of convergence, and the epoch counts are not claimed to
be tuned for every pair. No runtime estimates have been measured for these new
scripts.

## 2. PBRF on the same base checkpoints

The initial screening proposal uses final checkpoints, 32 candidate deletions,
and tied damping/proximal weights `0.01` and `0.1`, with removal strength `1.0`.
This is 64 nonlinear fits per base model (1,536 if all 24 base models are used).
First validate one small case; inspect the result before expanding to all pairs.
The script also supports smaller removal strengths and other damping values for
linearity diagnostics. The broader defaults of the standalone script are not
automatically the approved screening budget.

The finite-removal objective is the one written in the expanded draft:

```text
mean(all training examples' Bregman divergences from the base model)
  - removal_strength / N * loss(deleted example)
  + lambda / 2 * squared_distance_to_base_weights
```

The deleted example retains its Bregman term. Nonlinear proximal optimization
uses L-BFGS and records termination and gradient diagnostics; its outputs are
finite optimization endpoints unless the convergence evidence supports a
stronger description. This protocol differs from the old 150-epoch SGD sweep,
so results must not be pooled under the same protocol label.

Comparisons include exact GNH, exact block-GNH, K-FAC/EK-FAC, Shampoo/E-Shampoo,
FOOF/EFOOF (`mac`/`emac` in code), and identity/E-Identity. Use exact output-label
expectations for classification and deterministic output-curvature probes for
MSE, avoiding empirical-Fisher or label-sampling confounds in this first test.
Record raw predicted/actual query-loss changes, rank and magnitude agreement,
and linear-system and proximal-optimization diagnostics. Every method for a
given target uses the same absolute lambda as the target objective.

## 3. Gradient-perturbation unrolling

The primary intervention is a one-time weight perturbation along a normalized
per-example training gradient at the saved initialization. It is not persistent
example reweighting. Saved later checkpoints can be selected separately.

Proposed screening settings:

- 16 training-gradient probes and 16 held-out queries;
- horizons of 10, 50, and 100 SGD steps;
- perturbation norms of `1e-4`, `1e-3`, and `1e-2`;
- identical recorded minibatches for the reference, approximations, and actual
  positive/negative perturbation replays;
- constant plain-SGD LR, explicitly recorded, initially using the base run's
  initial LR; batch size 32, no momentum or weight decay.

Compare zero curvature (`H=0`, so `J=I`), isotropic unit curvature (`H=I`), exact
batch Hessian, exact batch GNH, block-GNH, K-FAC, EK-FAC, and FOOF. Additional
methods and random probes are optional. Approximate operators use the same
minibatch as the exact reference at each step.

Measure parameter-displacement agreement, error relative to the curvature
correction, and predicted query-loss changes `epsilon * query_grad.T @ J @ v`
against actual replay outcomes. Query-gradient cosines are supplementary:
dot products retain the effect magnitude. Central finite differences and
one-sided changes are separate readouts. An autodiff comparison inside future
execution checks the exact unrolling recurrence.

This stage is prepared independently; it does not start automatically when
training or PBRF finishes.

## Reviewing commands

These examples only display plans, because `--execute` is omitted:

```bash
.venv/bin/python -m experiments.staged_pairs.plan --phase train
.venv/bin/python -m experiments.staged_pairs.plan --phase pbrf
.venv/bin/python -m experiments.staged_pairs.plan --phase unrolling
```

The planner's `--execution-commands` option prints executable commands for
inspection; the planner itself still never submits them. Actual stage commands
and smoke-test settings remain subject to the user's requested confirmation.
Do not pipe the planner output into a shell.

Prepared code has only received static inspection. No dry runs, imports,
compilation checks, numerical tests, training, or experiments have been executed.
