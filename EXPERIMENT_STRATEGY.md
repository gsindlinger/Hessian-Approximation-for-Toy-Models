# Experiment strategy for the new story (2026-09-09)

> **§6 is the live checklist (decisions after the user's answers). §3–§4 are the original questions/proposal and are superseded where §6 says so.**

Status: readthrough done (both paper versions, `habo/new_story.md`, the notes folder, the repo, the
PBRF/unrolling/EoS notebooks, `runs.db`, and the reference papers in `habo/resources`). Nothing has
been run. This document collects (1) findings that change verdicts in the new story, (2) open
questions, (3) a proposed experiment plan by track. Once the questions are settled it becomes the
step-by-step checklist.

---

## 1. Findings that change the verdict table

### 1.1 Edge of stability: FOOF does *not* track sharpness (contradicts the EoS row)

`experiments/notebooks/edge_of_stability.ipynb` (small MLP8 on digits, full-batch GD, η = 0.03,
1000 steps, MCMC pseudo-targets) produced these plateau values (read off the saved figure):

| curvature | λ_max at plateau | ratio to exact |
|---|---|---|
| exact H | 66.7 = 2/η (EoS reproduced, hovers from step ≈ 380) | 1.00 |
| GNH, FIM | ≈ 52–54 | ≈ 0.80 |
| K-FAC, EK-FAC, Shampoo, E-Shampoo | ≈ 42 | ≈ 0.63 |
| MAC (= FOOF), EMAC | ≈ 10.5, **flat from step 0** | 6.5 → 0.15 |

The activation covariance barely changes under GD, so FOOF's top eigenvalue is a constant; all
sharpness dynamics live in the pseudo-gradient factor. The Kronecker methods track the *shape*
(rise, then plateau) at ≈ 0.63 of the level; GNH/FIM at ≈ 0.8. None would let you predict 2/η.

Consequence for the table: `W` for EoS should be **No**, `F` should be **partial (shape, not
level)**, and the interpretation flips from "redundancy" to "approximation problem" — unless there
is other EoS evidence I have not seen. This also removes the sentence "the one fidelity test
approximations pass, activations pass too".

### 1.2 Unrolling: the correct no-curvature baseline is H = 0, and every approximation ties with it

Table `tab:unroll-l2` compares against identity (H = I). Under H = I the trajectory Jacobian is
(1−η)^T·I = 0.048·I (η = 0.03, T = 100), i.e. a uniform 95% shrinkage — a scale artifact, not a
"no curvature" baseline. The no-curvature baseline is **H = 0 ⇒ J̃ = I**, whose error is the whole
curvature-driven correction ‖(J − I)w₀‖ = 4.0745 (printed in
`experiments_approximators.ipynb`, cell 16), i.e. **rel. L2 = 0.325**.

| method | ‖(J − J̃)w₀‖ | rel. L2 | ratio to H = 0 | cos(curvature correction) |
|---|---|---|---|---|
| H = 0 (J̃ = I) | 4.07 | 0.325 | 1.000 | 0 by definition |
| K-FAC | 4.04 | 0.322 | 0.992 | 0.13 |
| EK-FAC / Shampoo / E-Sh. | 4.06–4.07 | 0.323–0.324 | ≈ 1.00 | 0.07–0.08 |
| block Hessian (exact per-batch blocks) | 4.19 | 0.334 | 1.028 | −0.06 |
| GNH (exact per-batch) | 4.34 | 0.347 | 1.066 | −0.05 |
| FIM | 4.48 | 0.357 | 1.100 | −0.23 |
| identity (H = I) | 12.01 | 0.96 | — | — |

So no approximation recovers *any* of the curvature-driven part of the training Jacobian, and the
two oracle rows the new story asks for already exist: the exact per-batch **GNH alone** destroys
the signal (the residual term R = H − G matters at init), and the exact **block-diagonal Hessian
alone** destroys it (cross-layer blocks matter). Kronecker structure adds nothing on top.

Consequence: `F` for unrolling is **No**, not "partial"; the "0.3 vs 0.96" framing has to go.
Caveat that also has to go into the redesign: the Kronecker rows are estimated from the 32-sample
minibatch with *empirical-Fisher* pseudo-targets (true labels), i.e. sample-starved and the wrong
Fisher. The oracle rows show it would not have mattered, but the redesign should use population
factors (full train set, MCMC/all-classes) so the verdict is unambiguous.

### 1.3 PBRF: the verdict is architecture- and λ-dependent; the ResNet arms lack seeds

Ground-truth provenance is confirmed: `experiments/pbrf_mnist_sweep/run_one.py` retrains on the
PBRF objective with minibatch SGD (150 epochs, lr/10, warm start, per-point mask, vmapped over 320
deleted points). The a-fortiori defense stands. Bae et al. use λ = 0.001, 20 removed points, 50% of
base epochs, LiSSA; we use λ = 0.01, 320 points, 16 seeds — say so in the caption.

Seeded numbers (MLP [16×8] on MNIST-10%, 16 seeds, Spearman vs PBRF): GNH **0.92 ± 0.03**,
K-FAC 0.57 ± 0.07, EK-FAC 0.57, Shampoo 0.57, E-Shampoo 0.57, identity 0.47 ± 0.06. (Figure 1 in the
paper shows the seed-0 / 40-point numbers 0.98 / ≈0.5 / 0.32; swap in the seeded ones.)

But the ResNet arms (seed 0 only, `PBRF_MNIST_REPRODUCTION.md`) do not show the same gap:

| arch (λ = 0.01) | GNH S | K-FAC S | identity S |
|---|---|---|---|
| ResNet 3×[16,16] | 0.75 (1.00 at tied λ = 0.1) | 0.80 | 0.41 |
| ResNet 8×[8,16] | 0.90 | 0.65 | 0.43 |

At λ = 0.01 K-FAC beats GNH on the 3-block ResNet; at tied λ = 0.1 GNH is perfect. The diagnosis
(rank-113 GNH, exact inverse conditioning at small λ) is in `PBRF_RESNET_DIAGNOSIS.md`. This is
exactly the "regime" point of claim 2a.2, but it must be in the paper before a reviewer finds it.
Missing rows in the PBRF sweep: MAC/EMAC (FOOF), E-Identity, exact H (damped), CK-FAC.

### 1.4 IF / LDS: text bug and the Hong et al. regime

- The paper's LDS protocol paragraph says "leave-one-out perturbations"; `experiments/configs/lds.yaml`
  is α = 0.5 subsets, K = 100, R = 50 (ELSO, 5000 retrainings per model/epoch). Fix the text.
- Hong et al. (2025, the paper our Small MLP8 comes from) get the ordering
  H ≳ GGN > block-GGN > EK-FAC > K-FAC with **no damping and a |λ| > 1e-4 pseudo-inverse**; their
  own tables show the H→GGN step carries most of the LDS loss early in training although it is
  < 4% of the error. Our DB gives the opposite ordering under auto-mean damping. The story should
  state this as the same regime effect, not as a contradiction. (`RESULTS.md` §3.1: auto-mean and
  pseudo-inverse iHVPs stay near-orthogonal to the exact damped inverse across the whole sweep;
  only uniform λ ≳ 1e-3 aligns.)
- All LDS numbers in the paper are single-seed.

### 1.5 Optimization row: Benzing's own stated gap is exactly what we measure

Benzing (App. "KFAC derivation"): "none of our experiments directly checks how accurate this
approximation of the Fisher Information is … it remains plausible that without heuristic damping,
KFAC is very similar to the Fisher." And (App. "Laplace"): he proposes testing whether
beyond-optimization uses of Kronecker curvature "rely on proximity to the curvature matrix" by
replacing the second factor with the identity. That is the FOOF row of our LDS/PBRF/EoS tables.
Worth one sentence in Claim 1: we close the gap he left open, and the answer is "no, K-FAC is not
close to the Fisher/GNH" (matrix cosine ≤ 0.35 to H, `RESULTS.md` §3.3) *and* the identity-second-
factor test comes out as he predicted.

### 1.6 Weight editing: what the Goodfire paper does not have

Merullo et al. compare only against BalancedSubnet and SVD truncation; no magnitude pruning, no
random-direction, no EK-FAC. Our "identity + OBD = magnitude pruning" row is therefore new to that
line of work, and "EK-FAC eigenvalues change the pair ranking Λ_ij without changing the basis" is a
natural, cheap ablation if the weight-editing redo happens. (Out of this repo; see Q9.)

---

## 2. Literature gaps our toy setting can fill (from the reference reads)

- **SOURCE** never validates its curvature approximation inside unrolling: no comparison to exact
  unrolled differentiation, no swap of EK-FAC for exact GNH / H / identity, no damping sweep. Its
  only evidence that "the Hessian matters" is Hydra (H = 0, per-step gradients) at LDS 0.16 vs
  SOURCE 0.46 and IF (λ = 1e-8) 0.30 on FashionMNIST. Confound: SOURCE's ψ(H) implies damping
  λ = 1/(η̄K) ≈ 1e-3–1e-1 while its IF baseline runs at 1e-8. IF at matched damping is never run.
- **ASTRA** shows LDS keeps rising as low-curvature directions of the damped GGN converge, and that
  truncated Neumann = extra implicit damping; it also uses λ = 1/(η̄K) untuned. No exact-H
  reference, no K-FAC-without-correction, no identity row beyond "SNI iterate 0".
- **Wang et al. (taming)** propose a damping surrogate ξ = t₂/√(t₁t₃) with t_k = vᵀ(F+λI)^{-k}Fv,
  derived for the empirical Fisher; spectrum-quantile rules failed in their hands. ξ is cheap to
  evaluate for every approximation in our toy models, so a "regime-selection rule" experiment is
  affordable.
- **Bae 2022** defines the PBRF conditional on λ (ground truth moves with λ), uses 20 points.

---

## 3. Open questions (need answers before launching)

1. **Venue and deadline.** The reviewer tags (nYcm, icRW) in `new_story.md` imply real reviews exist.
   Target now (ICLR 2027?) and hard date? Can you drop the reviews into `habo/`?
2. **`pinv_mechanism_notes.md`** (P1/P2, §7/§8) is not in either repo or their git histories. Where is it?
3. **EoS row** — do you accept the flip in §1.1 (FOOF blind; approximations track shape at 0.6–0.8×)?
   If yes, the EoS experiment becomes another "approximation problem" row and the LLM-scale story
   rests on weight editing alone.
4. **Unrolling baseline** — agree to replace the identity row by H = 0 and report cos(curvature
   correction)? This makes 2b "established" rather than "pending sufficiency", at the price of
   losing the "cheap suffices" column entry (there is no cheap winner; nothing works).
5. **Unrolling reference** — F (exact unrolled derivative) or W (actual retraining)? Proposal: both,
   on the same fixed batch order, so the M verdict is empirical as the new story wants.
6. **PBRF presentation** — report all three MNIST architectures with seeds and the tied-λ sweep
   (recommended), or MLP only?
7. **Scope of "more model/dataset pairs".** Inside the JAX infra we can add cheaply: MNIST-10% small
   nets with the *exact* Hessian (14.5k params; 1.7 GB dense, fine on an A40), Fashion-MNIST small
   nets, CIFAR-10 flattened at medium scale, energy/concrete regression (MSE), Adam-trained variants,
   wider/deeper MLPs. A genuinely different family (tiny transformer on modular addition, or a small
   CNN) needs new model code plus collector support for weight sharing (K-FAC expand/reduce),
   roughly 2–3 agent-days. Which tier?
8. **LDS seeds** — 3–5 training seeds on toy digits MLP8 + ResNet3 at epochs {10, 100, 1000} for
   {auto-mean 0.1, uniform 1e-3, pseudo-inverse 1e-5} plus Fashion-MNIST ResNet MLP3 at epoch 16?
   (≈ 5000 retrainings per model/epoch/seed; ELSO cache is shared across methods.)
9. **LLM scale** (weight-editing redo, EM seeds) is "critical path" in the new story but the
   Goodfire/bergson pipelines are not on this machine (only `/root/bergson`). Out of scope for this
   sprint, or point me at the machine/codebase?
10. **CK-FAC library implementation** (`CKFAC_PLAN.md`, steps 1–7, ~1 agent-day, touches `src/`) —
    in scope now so CK-FAC gets rows in PBRF/LDS/unrolling, or keep the notebook proof of concept?
11. **Damping-rule study** (3a "regime-selection rule") — an experiment (test Wang's ξ, SOURCE's
    1/(η̄T), spectrum quantiles against the existing LDS sweep in `runs.db`) or a paragraph?
12. **Execution model.** Your memory file says you run scripts yourself; you now asked for agents.
    Proposal: agents write and smoke-test scripts in separate directories, full runs go on GPUs you
    assign (all 8 A40s are idle right now; node is shared). Which GPUs and what budget?

---

## 4. Proposed tracks

Each track is one agent, one directory, one CLI script (no notebooks), JSON outputs + a plot
script, following the `experiments/pbrf_mnist_sweep/` pattern (per-job subprocess pinned to one GPU).

### Track A — Unrolling redesign (row 2b; highest value per GPU-hour)
Goal: decide F and M for unrolling with the right baselines, on ≥ 2 architectures and 2 datasets.
- Models: digits MLP8 (2976), digits ResNet3 (2720), MNIST-10% MLP [16×8] (14.5k). Fixed recorded batch order.
- Curvatures per step: H = 0; identity (kept, labelled as scale artifact); exact per-batch H (reference);
  exact full-batch H (batch-noise oracle); exact per-batch GNH; exact block-H; segment-averaged exact H
  (SOURCE stationarity oracle, L = 1, 3); K-FAC/EK-FAC/Shampoo/FOOF/EFOOF/E-Identity with (a) per-batch
  factors, (b) population factors (full train, MCMC and all-classes); CK-FAC if Track E lands.
- Probes: w₀; random directions; per-example training gradients (data-space ∂w_T/∂β_i via the exact
  recursion); query gradients → attribution scores.
- References: exact unrolled derivative (jvp through the training loop) **and** actual retraining with the
  point masked on the same batch order (finite difference), 100–320 points, vmapped.
- SOURCE test: implement the segment estimator (exp and ψ) with each curvature; add IF at λ = 1/(η̄T) to
  separate "unrolling" from "damping rule".
- Regimes: start from init / epoch 10 / epoch 100; horizon 100 steps / 1 epoch / 5 epochs; η ∈ {0.03, 0.1};
  SGD (Adam optional later).
- Metrics: rel. L2, cosine of curvature correction, Spearman of attribution scores vs exact-unrolled and vs
  retraining; group-removal LDS on the trajectory optional.
- Compute: minutes per config at 3k params; ≈ 10 min per config at 14.5k params (f64 dense chain) — use
  jvp-based J·V products instead of materialising J where possible.

### Track B — PBRF completion (row 2a.1)
- Extend `run_one.py`: methods + MAC/EMAC/E-Identity/exact-H (damped)/CK-FAC; tied-λ grid
  {1e-3, 1e-2, 1e-1, 1} recomputing the ground truth per λ; partial correlation controlling for identity
  (`pbrf.md` protocol); Pearson and Spearman; noise ceiling per arm.
- Arms: MNIST MLP [16×8], ResNet 3×[16,16], ResNet 8×[8,16] × 16 seeds; add Fashion-MNIST small MLP and
  energy regression (port `pbrf_reproduction.ipynb`).
- Compute: ≈ 4 min per (arch, seed, λ) job on one A40 → ≈ 200 jobs ≈ 2 h on 8 GPUs.

### Track C — Second exact-Hessian dataset + LDS seeds (rows 2a.2–3, reviewer nYcm)
- Add MNIST-10% small MLP/ResNet to the main pipeline (`train_models` → `analyze_hessians` →
  `lds_analysis`) with exact H (switch the exact builder to one `jax.hessian` of the mean loss for 14.5k
  params), full metric suite, LDS.
- Seeds: retrain toy digits models with 3–5 seeds; rerun analysis + LDS on the reduced grid of Q8.
- Deliverable: seeded versions of Figs. 2–4 and the cross-method Spearman on a second dataset.
- Compute: the largest item; ≈ 10–20 GPU-hours total, parallel across GPUs.

### Track D — Edge of stability (row 2c, EoS)
- Convert the notebook to a script: models {digits MLP8, digits ResNet3, MNIST MLP}, η grid, 3 seeds,
  approximators incl. identity/E-identity/block-H/block-FIM/FOOF/EFOOF; CE and MSE; full-batch GD and
  minibatch SGD; Adam (preconditioned sharpness λ_max(P^{-1/2}HP^{-1/2})) optional.
- Metrics: plateau level ratio; shape tracking (Spearman of λ_max(t) during the rise); **critical-LR
  prediction**: η_crit predicted as 2/λ_max(approx) at a trained point vs the empirical η at which GD
  diverges — the level test that matters.
- Compute: ≈ 10 min per run.

### Track E — CK-FAC as a library mode (`CKFAC_PLAN.md` steps 1–6, then pytest once)
- Touches `src/config.py`, `src/hessians/computer/{computer,registry}.py`, `experiments/analyze_hessians.py`,
  plots. Run it before Tracks A/B add CK-FAC rows; keep it sequential with Track C's pipeline edits.

### Track F — Damping-rule study (corollary 3a; analysis-only + a few HVPs)
- For every (model, epoch, method) in `runs.db`: compute Wang's ξ(λ), SOURCE's 1/(η̄T), spectrum
  quantiles, auto-mean; score each rule by the LDS it lands on vs the best LDS in the existing 22-λ sweep.
- Deliverable: one figure "rule vs oracle LDS" per method family; the sentence a practitioner needs.

Not in this sprint unless Q9 says otherwise: weight-editing redo, EM fine-tuning seeds.

### Concurrency and etiquette
- A, B, D, F live in separate directories and can run in parallel; C and E edit shared code and run
  sequentially (E first). No branches; direct edits on main (your convention).
- Agents only smoke-test on tiny configs until you assign GPUs and OK full runs; one full run per config,
  no polling loops (shared node).
- Suggested order: E → A + B + D in parallel → C (long) → F (cheap, can go anytime).

---

## 5. Literature notes from the web scan (2026-09-09) that bear on the plan

Full list (≈ 40 entries with arXiv IDs) is in the agent transcript; only what changes decisions is here.

### 5.1 Edge of stability — pick the right target before judging the approximation
- **Cohen et al. 2022 (arXiv:2207.14484)** define *preconditioned sharpness* λ_max(P⁻¹H) with optimizer-specific
  thresholds (heavy ball (2+2β)/η; Adam 38/η at β₁ = 0.9). This gives a fidelity test nobody has run: if
  P_KFAC ≈ H then λ_max(P_KFAC⁻¹H) ≈ 1 and a K-FAC-preconditioned run could never reach the threshold.
  **Add to Track D:** train with K-FAC/FOOF/Shampoo as preconditioners on the toy model and measure
  λ_max(P⁻¹H) against the threshold. Directly tests "K-FAC is not curvature" on the optimization row too.
- **Andreyev & Beneventano 2024/25 (arXiv:2412.20553)**: for minibatch SGD the quantity that saturates at 2/η
  is *batch sharpness* (gradient-weighted directional curvature gᵀHg/‖g‖² on minibatches), not λ_max(H).
  That is a gradient-weighted statistic, i.e. exactly the object the CK-FAC / gradient-weighted-Frobenius
  part of the paper is about. **Add to Track D:** the SGD variant should report batch sharpness for each
  approximation, not λ_max. Hypothesis worth stating in advance: Fisher/Shampoo-type statistics track the
  gradient-weighted top of the spectrum better than λ_max.
- **Lowell & Kastner 2024 (arXiv:2406.00127)**: progressive sharpening is driven by *cross-layer* Jacobian
  alignment. Block-diagonal approximations discard that mechanism by construction, which is a ready-made
  explanation for the 0.63× plateau in §1.1. Cite in the caveat "how block spectra map to full λ_max".
- Nobody uses Kronecker/layerwise approximations as a sharpness proxy; practitioners at LLM scale use
  exact-HVP power iteration (Ramasinghe et al. 2026, arXiv:2606.16768). The EoS row is a genuine gap.

### 5.2 Novelty check
- Direct competitors for the **optimization** half: Benzing 2022 and Zhang et al. 2025 (arXiv:2502.01763,
  K-FAC as a feature-learning fix "not relying on how well KFAC approximates" second order); Frans et al.
  2025 (arXiv:2510.25000, variance adaptation, not spectral accuracy, explains matrix-whitening optimizers).
  None touches iHVPs, unrolling, weight editing or EoS. The novelty is "FOOF beyond optimization" plus the
  fidelity measurements; cite all three in Claim 1.
- Papers that will be cited **against** us: Hong et al. 2025 (arXiv:2509.23437), ASTRA (arXiv:2507.14740),
  Mlodozeniec et al. 2025 (arXiv:2410.13850, K-FAC on the right GGN beats TRAK/D-TRAK and is damping-
  insensitive), SOURCE, DataInf, Laplace-LoRA. The regime argument (§1.4) has to address the first three.
- Nothing found that (a) measures whether K-FAC/EK-FAC/Shampoo eigenvalues track λ_max(H) during training,
  (b) tests preconditioned-EoS thresholds for Kronecker preconditioners, (c) runs a curvature-free control
  for the K-FAC memorization edit on OLMo beyond Merullo's own SVD row.

### 5.3 Attribution ceiling and cheap baselines (supports Claim 2a.3 and 3b)
- **MAGIC** (Ilyas & Engstrom 2025, arXiv:2504.16430): exact metagradients give Spearman 0.96 on
  CIFAR-10/ResNet-9 vs 0.25 for TRAK and EK-FAC; Gemma-2B 0.97 vs "no better than random". The ceiling is far
  above IF and the gap is trajectory dependence, not Kronecker error — the same conclusion as §1.2 at scale.
  **Bergson** (Quirke et al., EleutherAI, Jun 2026, arXiv:2606.11660) implements MAGIC/SOURCE/TrackStar; that
  is the natural LLM-scale home for Track A if Q9 is ever in scope.
- **TrackStar** (Chang et al. 2025, arXiv:2410.17413) ablation at 8B: gradient dot product MRR 0.064 →
  +Adam second-moment normalisation 0.266 → +unit norm 0.290 → +Fisher 0.300. The Hessian term adds 0.01.
  **AirRep** (arXiv:2505.18513): gradient-free encoder beats LoGra/TRAK on LDS. Both are one-line support
  for Corollary 3b.
- dattri benchmark (arXiv:2410.04555): IF variants win only for logistic regression; ~random LOO
  correlation for MLP/MNIST for every method; only ensembled TRAK non-trivial on ResNet-9/NanoGPT.

### 5.4 Damping rules (Track F inputs)
- Auto-mean (0.1 × mean eigenvalue) is Kronfluence folklore attributed to Grosse et al. 2023; the docs also
  mention 10 × lr × num_iterations for SGD-momentum models (≈ SOURCE's 1/(η̄K) inverted — check the sign
  convention before citing). Klochkov & Liu 2024 (arXiv:2409.17357) set LiSSA parameters from trace and
  λ_max (solver convergence, not attribution quality). Wang et al.'s ξ is the only rule tied to LDS.

### 5.5 Weight editing (if Q9 is in scope)
- Merullo's plain SVD row is within 1.4 points of K-FAC on quotes (17.5 vs 16.1), i.e. a curvature-free control
  is already close in their own table. Curvature-free OLMo-7B editors to cite: OVE (arXiv:2606.18767), GSS
  (arXiv:2602.08901); Hubble (arXiv:2510.19811) gives OLMo-family models with inserted memorisation as ground
  truth. Cheapest curvature baseline: diagonal-Fisher dampening (SSD, arXiv:2308.07707).

---

## 6. Decisions and live checklist (after user answers, 2026-09-09)

Target: ICLR resubmission, deadline **2026-09-25**. Reviews are in `habo/NeurIPS reviews HABO 2026(1).md`.
What the reviews ask for, mapped to tracks: (nYcm/AC) decoupling on another exact-Hessian architecture →
Tracks C + T; (icRW W1/W2, AC) *why* identity works and actionable rules → Track A (unrolling), the
analysis-seed noise result in C, EoS per-layer in D; LLM seeds → out of scope (handled elsewhere);
(ByQ3) clarity/notation → writing phase.

Decisions:
- EoS: keep the row, but evaluate **per layer and excluding the input layer** (layer-0 activation
  covariance is the constant input covariance and dominates FOOF's global λ_max). Track D.
- Unrolling: redo. Primary design = the user's: perturb the init along per-example gradients, train,
  and test whether the (approximate) training Jacobian predicts the outcome, read out as the projection
  on / cosine with the query gradient at the final checkpoint. Both references (exact unrolled derivative
  and actual retraining). Track A.
- PBRF: keep all architectures; architecture dependence is a feature (regime dependence). Add rows and
  arms, then decide presentation. Track B.
- Model families: MLP, ResNet-MLP, SwiGLU, plus a **tiny transformer on modular addition**. No CNNs. Track T.
- CK-FAC stays a notebook proof of concept. Damping-rule study is a paragraph unless time remains.
- LLM-scale work is out of scope here. `pinv_mechanism_notes.md` is irrelevant.
- All 8 A40s may be used; agents launch runs themselves; one run per config, no polling loops.

Facts established while checking "seeds":
- Every LDS number is one base model (seed 42), one analysis seed (42), R = 50 retraining reps, CI over
  1797 queries. Base-model seeds are not there and are standard to omit; **analysis-side noise is not
  quantified** and is large: the same (model, epoch, λ) gives K-FAC LDS 0.190 with all-classes
  pseudo-targets vs 0.139 with MCMC, Shampoo 0.202 vs 0.124 (result_ids 2590/2591). That gap exceeds
  every method-vs-method gap in the paper. Quantify it with analysis seeds (ELSO is cached, so it costs
  only attribution recomputation). Track C.

GPU assignment: A → 0,1 · B → 2,3,4 · D → 5 · T → 6 · C → 7 (+6 once T is done).

### Checklist
- [ ] **Original-paper seed replication (requested 2026-09-10; TODO, not launched):**
      add two base-training seeds, 43 and 44, to the original seed-42 experiments,
      bringing each original setup to three seeds. Interpret "experiments we already
      have" as the three named Digits models (Small MLP8, Small ResNet MLP3,
      Small ResNet MLP8) and two Fashion-MNIST models (ResNet MLP3, ResNet SwiGLU3).
      Reuse their original training recipes, saved data splits, checkpoint schedules,
      and per-model analysis/LDS scope; audit compatibility with the updated native
      training code before launching. These are base-model replicas, not additional
      pseudo-target/analysis seeds or LDS retraining repetitions. The new Concrete,
      Energy, MNIST and modular-addition pairs already use seeds 42/43/44 and are
      not being expanded to five seeds by this TODO. Pending scheduling/confirmation;
      do not interrupt the active intermediate-checkpoint sweep.
- [ ] **A. Unrolling redesign** (`experiments/unrolling/`): weight-space gradient perturbations at init
      and at checkpoints, H = 0 and identity baselines, exact/GNH/block/full-batch/segment-averaged
      oracles, Kronecker family with per-batch and population factors, SOURCE estimator with each
      curvature, data-space (masked point) variant second; digits MLP8, digits ResNet3, MNIST MLP [16×8];
      metrics: cos/rel-L2 of Δw, Spearman/Pearson of query-gradient projections vs exact-linear and vs
      actual retrain; linearity check over ε.
- [ ] **B. PBRF completion** (`experiments/pbrf_mnist_sweep/`): add MAC/EMAC/E-Identity/exact-H rows,
      tied-λ grid, Fashion-MNIST and energy arms, 16 seeds on the MNIST ResNets, partial correlation vs
      identity, seed-averaged figure replacing Fig. 1.
- [ ] **D. EoS per layer** (`experiments/eos/`): per-layer block λ_max (exact block vs each approximation),
      global λ_max with layer 0 removed, 3 models × LR grid × seeds, all approximators; minibatch variant
      with batch sharpness; preconditioned-sharpness test with K-FAC/FOOF preconditioning if time.
- [ ] **T. Tiny transformer + modular addition** in the library (model, dataset, collector expand
      handling, tests, yaml), then the standard pipeline (metrics with exact H, influence, LDS) on it;
      afterwards add it as an arm to A, B, D.
- [ ] **C. Pipeline arms + analysis-seed noise** (`experiments/configs/`, `experiments/outputs/`):
      analysis-seed replication on the three digits models; MNIST-10% and Fashion-MNIST-10% small MLP and
      ResNet through train → analyze (exact H) → LDS.
- [ ] Writing: verdict table with regimes stated; fix LDS protocol text; notation; Hong/ASTRA/Mlodozeniec
      reconciliation paragraph; Benzing gap sentence.
