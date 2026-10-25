# Mining `runs_claude.db` — findings beyond the paper

**Status:** autonomous analysis in progress (LDS deliberately excluded; influence `.npy`
not used per instruction). Findings are tiered by my confidence and every claim links to a
reproducible script under `experiments/analysis_claude/`. Numbers are stamped from the runs
in `experiments/analysis_claude/out/*.json`.

## Scope & caveats (read first)

- **This is the refactored re-run DB.** My notes indicate the *published* figures came from a
  pre-refactor `runs.db` on `/mnt`. So "reproduce" below means *the paper's qualitative claims
  re-validated on `runs_claude.db`*, not bit-matching published numbers.
- **The `metrics` table is toy-scale only** — 3 `digits` models: `mlp_08580ee2573a` (2976 params,
  the 8-layer MLP), `resnet_mlp_0a69ab6297da` (2720), `resnet_mlp_adc3030f5daa` (3232). The
  ~500k `fashion_mnist` MLP-ResNets appear **only** in the `influence`/LDS path → their results
  (paper Fig 2/5) are **out of scope** here.
- **`matrix` & `hvp` metrics are λ-independent** (no damping applied in `compare_matrices`/
  `compare_hvps`): byte-identical across all 6 `damping_value`s and 3 `damping_strategy`s
  (verified: matrix max-spread 2e-16, hvp 3e-4 float jitter). They are **deduped** before
  aggregation; only `ihvp`/`round_trip` move with λ.
- **`reference` = the matrix compared against:** `exact` = true Hessian **H**, `gnh` = Gauss-Newton
  **G**. So `approximator=gnh, reference=exact` is the direct **G-vs-H gap**. Relative metrics
  divide by the reference.
- Within the metrics table only `pseudo_target=mcmc`, `vector_num_samples=500`, `seed=42` are
  present — so `all_classes`/sample-size knobs **cannot** be studied here.
- The **`exact` Hessian is indefinite**, so quantities computed *against* `exact` under inversion
  (iHVP, round-trip, inner-product ratios) are partly contaminated by the instability of H⁻¹
  itself — flagged throughout. `gnh` is the better-behaved reference (this is exactly the
  `if_bregman` point: influence functions answer the GNH/PBRF question, not the H question).

---

## TL;DR — the strongest new findings

1. **Inversion damping, not curvature quality, creates GNH≈H.** G and H agree in *forward* space
   (matrix cosine 0.98, HVP rel-err 0.10) but their *inverses* are **orthogonal at small damping**
   (cosine≈0, rel-err 7.25 at λ=1e-9), converging only as λ grows (cosine 0.996 at λ=1).
2. **The inversion *strategy* decides whether the inverse is even directionally right.** Only
   `uniform` (absolute λ≳1e-3) aligns approximate iHVPs with the exact damped inverse
   (cosine 0.84→0.99); `auto_mean` and `pseudo_inverse` stay near-**orthogonal** (cosine ≤0.09)
   across the *entire* λ sweep.
3. **Kronecker structure barely matters; the eigenvalue correction does the work.** After
   EV-correction + uniform damping, `eidentity` ≈ `ekfac` ≈ `emac` ≈ `eshampoo` to within 0.003
   cosine at every λ — the *trivial* (identity) eigenbasis matches K-FAC.
4. **Block-diagonalization is ~5× more lossy than Gauss-Newton.** `block_hessian`-vs-`exact`
   Frobenius ≈ 0.80 vs `gnh`-vs-`exact` ≈ 0.16; the GN residual R is also nearly **traceless**
   (trace-distance ≈ 1e-4).
5. **Architecture-dependent GN limit + post-convergence curvature drift.** The two ResNets'
   residual R→0 with training (0.11→0.005) but the plain MLP **plateaus at ~0.39** — while
   val-accuracy is flat from epoch 10. Curvature keeps drifting for hundreds of epochs after
   accuracy converges.
6. **The metric you report changes the verdict.** `cosine_similarity` of the iHVP is **negatively**
   correlated with influence-score (inner-product) fidelity — choosing the best-cosine method can
   make the actual influence scores *worse*; `identity` jumps from rank ~11 (norm) to ~5 (direction).

---

## 1. Reproductions (paper claims re-validated on this DB)  ·  `02_reproduce.py`, `05_robust.py`

| Paper claim | Verdict here | Evidence |
|---|---|---|
| Matrix-distance, HVP-error, iHVP-error are **decoupled** (Fig 1b) | ✅ **partially, sharpened** | matrix↔HVP rank ρ≈**+0.99** (forward metrics agree); HVP↔iHVP ρ is **weak & sign-unstable** (−0.60 to +0.16 across subsets) → forward fidelity does **not** predict inverse fidelity |
| All Kronecker methods cluster; **Identity is the outlier**; none recovers the exact action (Table 1) | ⚠️ **holds only after EV-correction** | raw `mac`/`identity` are 1000× outliers in matrix/HVP error; their EV-corrected forms (`emac`/`eidentity`) rejoin the cluster (HVP rel-err ≈0.80 like `ekfac` 0.79). *(Note: the DB's HVP rel-err ‖(H−Ĥ)g‖/‖Hg‖ is a different normalization than the paper's unrolling rel-L² ≈0.32, so absolute values differ.)* |
| **Damping dominates** method choice | ⚠️ **weak at the iHVP-fidelity level** | for iHVP rel-err (uniform), η²(damping)=0.055 ≳ η²(method)=0.042 ≳ η²(epoch)=0.011 — damping edges out method, but both explain little variance (the paper's claim is about LDS, not iHVP error) |

**Caveat on the decoupling:** the forward↔forward correlation (matrix↔HVP ≈+0.99) is genuine but
partly inflated by the scale-degenerate `mac`/`identity` (huge in both). The robust statement is
**forward fidelity (matrix or HVP) carries essentially no information about inverse (iHVP) fidelity.**

---

## 2. GNH red-team (the focus of `if_bregman`)  ·  `03_gnh_redteam.py`

### 2.1 G≈H forward, but inversion amplifies the gap → damping is the real actor  **[high confidence]**

The G-vs-H gap by level (`gnh` vs `reference=exact`, mean over 3 models × 11 epochs):

| level | metric | G-vs-H |
|---|---|---|
| matrix | relative_frobenius | 0.16 |
| matrix | cosine_similarity | **0.98** |
| matrix | trace_distance | **1e-4** (R ~traceless) |
| hvp | relative_error | **0.10** |
| ihvp | relative_error | 4.3 (mean — λ-dominated, see below) |

iHVP `gnh`-vs-`exact` as a function of **uniform** damping λ:

| λ | iHVP rel-err | iHVP cosine |
|---|---|---|
| 1e-9 | **7.25** | **−0.000** (orthogonal) |
| 1e-6 | 1.40 | 0.017 |
| 1e-3 | 0.21 | 0.861 |
| 1e-2 | 0.11 | 0.911 |
| 1e-1 | 0.09 | 0.931 |
| 1e-0 | 0.018 | **0.996** |

> **Finding G-1.** The Gauss-Newton residual R = H − G is small and nearly traceless in *forward*
> space (matrix cosine 0.98, HVP rel-err 0.10), but **under inversion the small residual is
> amplified into orthogonality at small damping** (cosine ≈ 0 at λ=1e-9). The apparent "GNH is a
> good Hessian for influence functions" holds *only because of damping*, exactly as the
> perturbation identity (paper eq. 5) predicts. This is the cleanest single confirmation of the
> paper's HVP/iHVP decoupling, localized to G-vs-H.

### 2.2 Is the *exact* Hessian a treacherous inversion target?  **[refined — partly refuted]**

iHVP rel-err for each method, `reference=exact` vs `reference=gnh` (uniform, λ=1e-3):

- For most methods, error-vs-`exact` (0.41–0.46) is **slightly lower** than error-vs-`gnh`
  (0.48–0.58) → at *moderate* damping the exact target is **not** pathological (my prior hypothesis
  was **refuted** — good).
- But **directionally** (cosine) every method aligns better with `gnh` (≈0.91) than with `exact`
  (≈0.83) — the approximators are Fisher/GN-type, so their inverses point the GNH way.
- At **small** λ the exact target *is* treacherous: **54 (method, λ) configs have negative iHVP
  cosine vs exact** (`gnh`, `kfac`, `ekfac`, `shampoo`, `eshampoo`, `block_hessian`) — the indefinite
  H⁻¹ flips sign along negative-curvature directions.

> **Finding G-2.** Whether "compare against the true H" is sensible is **damping-dependent**: fine
> at λ≳1e-3, actively misleading (anti-aligned inverses) at λ≲1e-6. Practically, `gnh` is the
> stable reference — consistent with `if_bregman`'s argument that influence functions target the
> GNH/PBRF object, not H⁻¹.

### 2.3 Convergence of the G-vs-H gap is architecture-dependent  **[high confidence]**

`relative_frobenius(gnh, exact)` by epoch (val-accuracy is flat at 0.95 from epoch 10 onward):

| epoch | MLP (2976) | ResNet (2720) | ResNet (3232) |
|---|---|---|---|
| 10 | 0.98 | 0.11 | 0.18 |
| 100 | 0.33 | 0.033 | 0.025 |
| 500 | 0.39 | 0.0053 | 0.0099 |
| 1000 | **0.39** | **0.0048** | **0.010** |

> **Finding G-3.** The ResNets approach the Gauss-Newton limit (R→0, to ~0.5–1% Frobenius) as
> training proceeds, but the **plain MLP's residual plateaus at ~0.39 and never vanishes** (and is
> non-monotonic — dips at epoch 100, rises, settles). Since accuracy is flat from epoch 10, **the
> curvature keeps drifting for hundreds of epochs after the model has "converged."** This both
> (a) gives a concrete post-convergence-drift result and (b) shows the validity of the GN
> approximation is **architecture-specific** — a caution for influence-function practice on
> non-residual nets.

### 2.4 The sampled Fisher is *not* the GNH it is supposed to equal  **[high confidence]**

`fim` (mcmc, 1 pseudo-label/sample) vs `gnh` (exact GN): matrix rel-Frobenius **5.36**, cosine **0.44**
(vs `exact`: 4.81 / 0.45). `block_fim` is even further (cosine 0.27).

> **Finding G-4.** For cross-entropy the Fisher equals the GNH *in expectation / with all classes*,
> but the **single-sample mcmc empirical Fisher in this DB is wildly off** (5× in Frobenius, cosine
> 0.44). Concrete evidence for the paper's §7 "are approximators sample-starved?" — and a reason the
> `fim`/`block_fim` rows behave worse than the Kronecker methods. *(The `all_classes` strategy that
> would test the fix is only in the 500k influence data, not the metrics table.)*

---

## 3. New discoveries  ·  `04_discover.py`, `05_robust.py`

### 3.1 Inversion **strategy** dominates inverse fidelity  **[high confidence]**

iHVP cosine vs `exact`, averaged over methods/models/epochs, by strategy × λ:

| λ | `auto_mean` | `pseudo_inverse` | `uniform` |
|---|---|---|---|
| 1e-9 | 0.001 | 0.019 | 0.001 |
| 1e-3 | 0.002 | 0.090 | **0.840** |
| 1e-2 | −0.002 | 0.071 | **0.906** |
| 1e-0 | 0.090 | 0.035 | **0.995** |

> **Finding D-1.** Only **`uniform`** (absolute λ) produces inverses that align with the exact
> damped inverse. **`auto_mean` stays near-orthogonal across the whole sweep** (its λ is a
> *multiplicative factor* on the tiny per-layer mean eigenvalue → always under-damped, even at
> factor=1.0), and **`pseudo_inverse` never exceeds cosine 0.09** (eigenvalue truncation discards
> the directions inversion needs). This quantifies the paper's remark that "LDS becomes competitive
> only after switching the inversion strategy" — at the iHVP-fidelity level the strategy decides
> whether the inverse is even *directionally* correct.

### 3.2 Eigenvalue correction does the work; the Kronecker eigenbasis barely matters  **[high confidence]**

iHVP cosine vs `exact` (uniform) — EV-corrected family:

| λ | `eidentity` | `ekfac` | `emac` | `eshampoo` | `gnh` |
|---|---|---|---|---|---|
| 1e-3 | 0.841 | 0.838 | 0.838 | 0.839 | 0.861 |
| 1e-2 | 0.906 | 0.906 | 0.906 | 0.906 | 0.911 |
| 1e-0 | 0.994 | 0.995 | 0.994 | 0.995 | 0.996 |

> **Finding D-2.** Once you apply the eigenvalue correction + uniform damping, **`eidentity`
> (identity eigenbasis — no Kronecker structure at all) is indistinguishable from `ekfac`/
> `eshampoo`/`emac`** (within 0.003 cosine at every λ; identical iHVP rel-err 0.202 at epoch 1000).
> The empirical-eigenvalue correction Λ + damping carry the signal; the A⊗B eigenbasis choice is
> nearly irrelevant for the resulting inverse direction. Strong metric-level support for "Identity
> is hard to beat" / Benzing's FOOF observation that the gains aren't from curvature structure.

### 3.3 The eigenvalue correction fixes **scale**, not **direction**  **[high confidence]**

matrix vs `exact` (mean):

| method | cosine | rel-Frobenius |
|---|---|---|
| `mac` | 0.14 | **6402** |
| `emac` | 0.27 | **1.13** |
| `identity` | 0.085 | **2453** |
| `eidentity` | 0.090 | **1.04** |
| `kfac` | 0.24 | 2.01 |
| `ekfac` | 0.35 | 1.08 |
| `gnh` | **0.97** | 0.20 |

> **Finding D-3.** The EV correction collapses the raw `mac`/`identity` Frobenius error by ~2000×
> (a pure **scale** fix) — but the **direction stays wrong**: every Kronecker matrix (corrected or
> not) has cosine ≤0.35 to the true Hessian; only the full `gnh` is directionally faithful (0.97).
> So at the *matrix* level the entire Kronecker family is a poor model of H's structure; their
> usefulness (§3.2) lives entirely in the damped-inverse top-subspace, not in matching H.

### 3.4 Block-diagonalization costs more than Gauss-Newton  **[medium-high confidence]**

`relative_frobenius` vs `exact` (mean): `block_hessian` **0.80** vs `gnh` **0.16** (≈5×). Trace
gap `gnh`-vs-`exact` ≈ 1e-4.

> **Finding D-4.** Throwing away the **cross-layer** Hessian blocks (block-diagonal H) loses ~5×
> more relative Frobenius mass than the entire Gauss-Newton residual R. i.e., **off-block-diagonal
> curvature is bigger than the GN residual**, and the GN residual is ~traceless. *(Aggregate over
> epochs/models; both vary by architecture — to be refined.)*

### 3.5 The metric decides the winner — and cosine *misleads* for influence scores  **[medium confidence]**

- **Ranking inversion:** mean rank (1=best, vs exact): `gnh`≈1, `block_hessian`≈2 everywhere; but
  raw **`identity` jumps from rank ~11** (matrix/HVP/iHVP-rel-err) **to ~5** on iHVP **cosine** — its
  inverse points roughly right even though its magnitude is hopeless.
- **Influence-score fidelity is its own axis.** Across well-scaled methods, the iHVP
  `cosine_similarity` is **negatively** correlated with influence-score error
  (`RELATIVE_INNER_PRODUCT_DIFF`): ρ≈**−0.43** (vs exact), **−0.28** (vs gnh); `relative_error` is a
  weak-*positive* proxy (ρ≈+0.41 / +0.75). Kronecker methods overshoot influence scores 5–15× vs the
  (unstable) exact reference; `gnh` ≈1.5×.

> **Finding D-5.** **Picking the approximation with the best iHVP cosine can make the actual
> influence scores (inner products) *worse*** — cosine is anti-correlated with influence-score
> fidelity. This extends the paper's "metrics are decoupled" thesis down to the influence-score
> level and is a direct caution about how influence approximations are evaluated. *(Small n=9–10
> methods; influence-score ratios are vs the indefinite `exact` — treat magnitudes as indicative,
> the rank/relationship as the finding.)*

---

## 4. Speculative leads & anomalies (need raw vectors, the `.npy`s, or LDS to settle)

- **HVP cosine collapses at late epochs while norm-error stays small.** For `gnh` vs `exact`, by
  epoch (MLP): cosine 0.89→**0.006** while rel-err stays ~0.23 and energy-diff →0.004. Likely
  mechanism: as the toy models converge **H becomes near-singular**, so HVPs in most probe
  directions are numerically ~0 and their *direction* is noise → batch-mean cosine → 0 even though
  the vectors are close in norm. Consistent with "gradients live in a low-dim subspace" (§7). Needs
  the raw HVP vectors to confirm — flagged so we don't trust HVP-cosine on near-singular operators.
- **`block_hessian` round-trip is non-monotonic in damping** (error 28 → 139 → 79 as λ goes
  1e-3→1e-2→1e-1): adding damping makes the round-trip *worse* in the mid-range — an indefiniteness
  artifact (damping an indefinite block doesn't monotonically stabilize it).
- **`trace_distance` ranks methods differently** from the norm metrics (ρ≈0.22 vs frobenius/spectral
  which agree at ρ≈1.0) — trace is a near-useless discriminator here (it sees only Σλ, and R is
  traceless).
- **Influence-score magnitudes vs `gnh`** (the stable reference) and a small toy-scale `.npy`
  cross-method ranking check would convert §3.5 from "medium" to "high" — deferred per the
  "stay away from `.npy`" instruction.

---

## 5. Methods / reproducibility

- Scripts: `experiments/analysis_claude/{lib,01_overview,02_reproduce,03_gnh_redteam,04_discover,05_robust}.py`;
  JSON dumps in `experiments/analysis_claude/out/`. Run with `.venv/bin/python`.
- All numbers are from `runs_claude.db` as of this run; analysis-only (no training / Hessian recompute).
- Confidence tags: **high** = direct, stable read across models/epochs; **medium** = small-n or
  subset-sensitive; leads in §4 are explicitly unconfirmed.

_Open next steps (when LDS / `.npy` are back in scope): tie §3.5 to actual LDS; test `all_classes`
vs `mcmc` (Finding G-4) on the 500k influence arrays; extend the GN-residual convergence story to
the MLP-ResNet scale._
