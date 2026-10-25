Standalone unrolling POC
=======================

Two seed-42 presets are approved below. The Python file is the source; the
notebook is its unexecuted percent-cell conversion, with ``MODEL = "resnet3"``
and ``RUN = False``. Production code stays within this folder: no shared
runner, model-checkpoint replay, collectors, DB, W&B, or core-code changes.

Experiment
----------

* ``resnet3``: the existing Small ResNet MLP3 architecture on Digits, 2,720
  parameters, ReLU, fresh Xavier initialization, SGD learning rate 0.003.
* ``transformer``: modular addition modulo 97, two decoder blocks, four heads,
  width 16, ReLU, all 9,840 parameters included, SGD learning rate 0.001.

Both use seed 42, batch size 32, float64, and local constant-rate SGD without
momentum, weight decay, or dropout. These are fresh initialization experiments,
not replay of saved training; in particular the transformer does NOT replay
its earlier AdamW optimizer. Data are read from the existing local NPZ splits,
without downloads. The query is the sum of ALL test-example losses.

Freeze 16 raw training-example gradients at initialization. Perturb the weights
by plus/minus epsilon times each gradient and train with identical saved
minibatches. Read out at horizons 0, 1, 10, 50, and 100 steps. Optional
``start_step`` performs local SGD warmup before injection.
The baseline, perturbed branches and all curvature methods share one schedule.
No data point is deleted/reweighted; this is not the full SOURCE estimator.

Compare propagated directions with independent whole-replay JVPs (two probes)
and actual symmetric perturbed training at epsilon 1e-7, 1e-6, 1e-5, 1e-4,
1e-3, and 1e-2. Larger epsilons remain deliberate nonlinear-response controls.
Read out both dot products and cosines with the final BASELINE query gradient.
Normalization is off by default: enabling it changes the injection and rankings,
and results must then be called normalized-gradient probes.

Methods: zero curvature, identity curvature, exact minibatch/full-training H,
minibatch/full-training GNH, exact parameter-block H, and full-training
KFAC/EKFAC/FOOF. Batch variants of the factors are also supported. Curvature is
recomputed at each baseline step; factors use deterministic exact output
curvature, not empirical Fisher or Monte Carlo sampling. No inverse or damping.
Full-training curvature is a batch-noise intervention, not the exact SGD oracle.

Zero curvature propagates the original direction unchanged. Identity shrinks it
by (1-lr)^steps and is not a no-curvature baseline. Report magnitude-sensitive
errors as well as cosine/rank agreement. Undefined correction metrics at step 0
and zero-curvature correction cosines remain NaN, with valid counts.

Use
---

From the repository root, generate the notebook WITHOUT execution::

    .venv/bin/python experiments/unrolling/unrolling.py --notebook

In the notebook, select ``MODEL``, inspect ``CONFIG``/``OUTPUT_NAME``, then set
``RUN = True`` only when ready to execute. The corresponding explicit CLI
commands are::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/unrolling/unrolling.py --model resnet3 --seed 42 --run
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/unrolling/unrolling.py --model transformer --seed 42 --run

Choose an available GPU; do not duplicate an already active run. Add
``--check`` to ``--run`` for horizons 0 and 1 only, with the same preset and
a separate ``_check`` output name. A check is not a completed full run.

Help and conversion import only the standard library, even if RUN was edited.
The numerical code enables float64 in its own process/kernel.
All outputs stay under ``outputs/OUTPUT_NAME``. Each run exclusively reserves
one new directory, then saves each configured horizon incrementally: the
configuration/status, initialization/batches/probes, response arrays, and
``metrics.csv``. A summary plot is saved after the full run finishes.
``config.json`` records ``running`` or ``complete``; intermediate files do not
imply completion. Existing output directories are refused on a new invocation;
incremental writes belong only to the run that reserved that directory.

Validation and interpretation
-----------------------------

``check_operators.py`` contains tiny CPU metric, replay, and closed-form CE/MSE
operator checks. ``transformer.py`` is the local all-parameter adapter;
``test_transformer.py`` checks its factors and native-forward parity. Native
model imports are confined to that test, never the production experiment.

The ResNet seed-42 run completed all five horizons through 100 SGD steps;
its exact recurrence passed the independent replay-JVP checks. At step 100,
epsilon 1e-7 gives endpoint relative error 4.67e-9 and query-score relative
error 7.43e-10. A separate exact-only check at 1e-8 and 1e-9 also agrees
(endpoint errors 4.66e-8 and 4.67e-7). Larger perturbations can leave this
local regime: at step 100, epsilon 1e-6 already has query-score error 0.840.
Keep these nonlinear controls; do not treat derivative accuracy as accuracy
for arbitrary finite perturbations. This is a short-horizon pilot, not
training to convergence or a multi-seed conclusion.

Inspect h_batch versus finite differences across the FULL epsilon grid first:
small epsilon can suffer cancellation; large epsilon can leave the linear regime
(especially with ReLU). The plot displays the smallest epsilon only as a quick
view, not as automatic epsilon selection. Inspect saved run status and the
exact-H checks at each horizon before drawing conclusions about approximations.
