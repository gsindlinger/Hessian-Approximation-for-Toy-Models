# CKFAC — gradient-aware ("C-weighted") eigenvalue correction as a first-class library mode

## Context

`experiments/notebooks/new_grad_vs_frob.ipynb` (cell 6) prototypes a **C-weighted optimal eigenvalue correction**: in the fixed K-FAC eigenbasis, instead of the Frobenius-optimal diagonal, fit Λ under the risk inner product ⟨X,Y⟩_C = tr(XCYᵀ) with C = GᵀG/N the **uncentered** second moment of per-example true-target training gradients (theory: `experiments/notebooks/grad_vs_frob.txt`, Remark 2.2c & §4). Because the basis elements q qᵀ have a diagonal C-Gram (orthonormal, disjoint layer supports), the projection is coordinate-wise:

    λ_q = qᵀ sym(CH) q / qᵀ C q        (Frobenius/EKFAC = the C = I special case)

with the Frobenius-optimal λ_q = qᵀHq kept as tie-break on dead directions (qᵀCq ≈ 0, C rank-deficient).

The notebook does this densely (np.kron, full C, full H, float64). Goal: implement it as a streaming library path in `src/hessians/computer/computer.py`, structured **exactly like the existing eigenvalue correction** (method + static jitted chunk kernel next to `_compute_layer_lambda_chunk_sum`, computer.py:420), with `ckfac` as a registry binding.

**Key identity (verified numerically at machine precision against the dense notebook formula, incl. C's cross-layer blocks):** for a basis vector q of layer l,

- **den**: qᵀCq = meanₙ (qᵀuₙ)² → per layer `mean ã² ⊗ g̃²` — the existing kernel's quantity, fed with grad-cov data (uₙ's layer block = aₙgₙᵀ is rank-1; ã = Q_Aᵀa, g̃ = Q_Gᵀg).
- **num**: qᵀ sym(CH) q = qᵀCHq = meanₙ (qᵀuₙ)(qᵀHuₙ) → one full-network HVP per sample; per layer `mean ãᵢ g̃ₒ h̃[i,o]` with h̃ = Q_Aᵀ (Huₙ)ₗ Q_G.

So C is never materialized, and H enters only through HVPs. Cross-layer blocks of both C and H enter exactly, through the full-network uₙ and Huₙ.

## Step checklist (execute one step at a time; check off as steps land)

- [x] **Step 0** — Write this tracking doc
- [ ] **Step 1** — Enums in `src/config.py` (§1)
- [ ] **Step 2** — Core mode + kernels in `src/hessians/computer/computer.py` (§2)
- [ ] **Step 3** — Registry wiring (§3)
- [ ] **Step 4** — Tests: mechanical enum rename + new `tests/test_ckfac.py` with the dense-reference equivalence test — validates Steps 1–3 before any pipeline wiring (§6)
- [ ] **Step 5** — `experiments/analyze_hessians.py` wiring: EF grad-cov collection + `HessianCtx` (§4)
- [ ] **Step 6** — Plots wiring (§5)
- [ ] **Step 7** — Verification runs (pytest once; optional notebook regression)

## Settled design decisions

1. **C source**: EF collector context over the **full train set** — new field `grad_cov_context: Optional[DataActivationsGradients]`. Reuse the primary collector data when the run's pseudo-target strategy is already EMPIRICAL_FISHER; otherwise one extra EF collection (deterministic → cached/shared across strategies and seeds). Kernel stays general in `(g (N,O,k), probs (N,k))` so other contexts (ALL_CLASSES → Fisher-C) work for free. Uncentered.
2. **H side**: new field `target_estimator: Optional[HessianEstimator]` — any **built** estimator (EXACT in the pipeline; GNH also valid). HU via its `estimate_hvp`; its dense layer-diagonal blocks give the qᵀHq fallback. (The exact-H LayerMatrix is stored undamped — hessian.py:62.)
3. **Dead directions**: notebook-faithful — per layer, `den ≤ float32-eps · max(den)` → keep the **Frobenius optimum** λ = qᵀHq, computed exactly and cheaply from the target's dense diagonal block (feasible precisely because the target is a built estimator; a matrix-free target would have needed I·O HVPs per layer).
4. **Packaging**: mode on `KroneckerEstimator` mirroring the existing correction: replace `apply_eigenvalue_correction: bool` with enum `EigenvalueCorrection = NONE | FISHER | GRAD_WEIGHTED` (default FISHER ≙ current True); `ckfac` = registry binding of `EKFACComputer` with GRAD_WEIGHTED (composes with the Shampoo/MAC/Identity bases for free, though only `ckfac` is registered for now); full pipeline + plots wiring.

## Verified conventions the implementation relies on

- Flat layout (`LayerVector.from_flat/to_flat`, layer_matrix.py:596-622): layer order = `param_groups`; within layer row-major `(I,O)`, flat index `i*O+o` — matches `kron(Q_A,Q_G)` and `Lambda.flatten()`. Eigenbasis projection of a layer slice = `Q_A.T @ V @ Q_G` (as in `KroneckerFactors.matvec`, layer_matrix.py:176).
- No model uses biases → per-sample layer grad is exactly `outer(a, g)`; EF collection: k=1, probs=ones (float32), N = full train set; a/g are float64 (`jax_enable_x64` global, src/__init__.py:11).
- `KroneckerFactors` stores an arbitrary Λ (layer_matrix.py:119-124); `damped`/`inverse` add damping with no abs — negative ckfac Λ entries store fine, but inverse/AUTO_MEAN semantics are caveats (below).
- Scale: digits D=2976, N_train=3823; MNIST sweep D≈14.5k — all matmuls/HVPs trivial as dense matvecs.
- Notebook reference numbers (digits mlp_08580ee2573a epoch 10, N_grads=1000 subset): ckfac `‖D‖_F = 20.47`, `‖DC^½‖_F = 5.958` vs block-diag(H) floor `3.872 / 6.48`; 153/2976 dead directions.

## Implementation detail per step

### §1. `src/config.py`
- Add `class EigenvalueCorrection(str, Enum): NONE, FISHER, GRAD_WEIGHTED` near `DampingStrategy` (config.py:98).
- Add `HessianApproximationMethod.CKFAC = "ckfac"` (enum at config.py:59-87).

### §2. `src/hessians/computer/computer.py` (core)
- `KroneckerEstimator` fields: replace `apply_eigenvalue_correction: bool = True` with `eigenvalue_correction: EigenvalueCorrection = FISHER`; add `grad_cov_context: Optional[DataActivationsGradients] = None` and `target_estimator: Optional[HessianEstimator] = None`.
- `__post_init__` (computer.py:225): FISHER → require `corr_context` (current message); GRAD_WEIGHTED → require `grad_cov_context` and a **built** `target_estimator`, and validate `target_estimator.layer_matrix.param_groups == compute_context.layer_names` + matching `layer_shapes`.
- `_compute_lambdas` (computer.py:378): dispatch — NONE → `outer(λ_A, λ_G)` (unchanged); FISHER → existing path (unchanged); GRAD_WEIGHTED → new `_compute_lambdas_grad_weighted(activation_eigvecs, gradient_eigvecs)`.
- New driver `_compute_lambdas_grad_weighted`, chunked via `_accumulate_chunks` over `grad_cov_context` (N rows, `self.batch_size`):
  1. per chunk & layer: `u = einsum("ni,nok->nkio", a, g)` → reshape `(N,k,I*O)`, concat layers in `layer_names` order → `(N,k,D)` → `(N*k,D)`;
  2. `hu_flat = self.target_estimator.estimate_hvp(u_flat)` (damping None → 0);
  3. slice `hu_flat` back per layer via offsets from layer shapes → `(N,k,I,O)`;
  4. accumulate `(num_sum, den_sum)` from the new kernel.
  Finally per layer: `lam_frob = _compute_layer_frob_lambda(H_ll, Q_A, Q_G)` with `H_ll` = dense of `target_estimator.layer_matrix.blocks[(l,l)]`; `dead = den_sum ≤ jnp.finfo(jnp.float32).eps * den_sum.max()`; `Λ = where(dead, lam_frob, num_sum / where(dead, 1, den_sum))` (total_prob cancels in the ratio; the threshold is relative so unnormalized sums are fine).
- New static jitted kernel (sibling of computer.py:420), signature symmetric with the existing one:
  `_compute_layer_grad_lambda_chunk_sums(a (N,I), g (N,O,k), probs (N,k), hu (N,k,I,O), Q_A, Q_G) -> (num_sum (I,O), den_sum (I,O))`
  with `ã = a @ Q_A`, `g̃ = einsum("op,npk->nok", Q_G.T, g)`, `h̃ = einsum("pi,nkpo,oq->nkiq", Q_A, hu, Q_G)`;
  `num_sum = einsum("ni,nok,nk,nkio->io", ã, g̃, probs, h̃)`; `den_sum = einsum("ni,nok,nk->io", ã**2, g̃**2, probs)` (== existing kernel's value).
- New static jitted helper `_compute_layer_frob_lambda(H_ll (IO,IO), Q_A, Q_G) -> (I,O)`: reshape `(I,O,I,O)`, contract `einsum("pqrs,pi,qo,ri,so->io", ...)` staged in two steps for efficiency.
- `_build` (computer.py:246) unchanged — blocks remain ordinary `KroneckerFactors(Q_A, Q_G, Λ, λ_A, λ_G)`, so save/load/damping round-trip.

### §3. `src/hessians/computer/registry.py`
- Update the 4 partials `apply_eigenvalue_correction=False` → `eigenvalue_correction=EigenvalueCorrection.NONE` (registry.py:40-59).
- Add `CKFAC: partial(EKFACComputer, eigenvalue_correction=EigenvalueCorrection.GRAD_WEIGHTED)` and `LAYER_MATRIX_DIRECTORY[CKFAC] = "ckfac_layer_matrix"`.
- New `_GRAD_WEIGHTED_METHODS = frozenset({CKFAC})`; `get_computer` (registry.py:82) gains optional `grad_cov_context=None, target_estimator=None`, injected only for those methods (mirror the corr_context guard). `get_compute_context`: no change (CKFAC falls through to `collector_data`).
- `ekfac.py`/`emac.py`/`eshampoo.py`/`eidentity.py`: no logic changes (hooks untouched; docstring touch-ups where they mention the bool).

### §4. `experiments/analyze_hessians.py`
- Grad-cov collection: if `strategy == EMPIRICAL_FISHER` → reuse `primary`; else run one extra EF collection (repetitions=1) into its own cached dir, following the existing `_collect`/`collector_dir_corr` pattern (analyze_hessians.py:188-240, dir naming at :748-757). EF is deterministic — cacheable across strategies.
- `HessianCtx` (:244-263): add `collector_data_grad_cov` field; in `get()`, special-case CKFAC → `target = self.get(HessianApproximationMethod.EXACT)` (already built for comparison metrics; cached), pass `grad_cov_context` + `target_estimator` through `get_computer`.
- `resolve_damping` (:315): no code change — the `KroneckerEstimator` branch picks ckfac up; see caveats.

### §5. Plots
- `plots/hessian_data.py`: `APPROX_ORDER` (:40, insert after "ekfac"); `METHOD_FAMILY` (:215, `"ckfac": "kfac"`); do **NOT** add to `EIGENVALUE_CORRECTED` (:207 — that set means "2× collector cost"; ckfac's extra cost is the exact-H build — add a comment).
- `plots/hessian_plots.py`: `COLORS["ckfac"] = "#000000"`, `LABELS["ckfac"] = "CK-FAC"` (:24-34), matching the notebook.
- `plots/code_vocab.json`: regenerate via `python plots/build_code_vocab.py` after first data appears.

### §6. Tests
- Mechanical: update `apply_eigenvalue_correction` usages to the enum (`tests/test_ekfac.py` ~8 sites, `tests/test_shampoo.py`, any others surfaced by grep).
- New `tests/test_ckfac.py` (pattern fixtures from test_ekfac.py — an EF collector fixture already exists there):
  1. **Dense-reference equivalence** (load-bearing): tiny trained model; exact `HessianComputer`; EF collector data; build ckfac via `EKFACComputer(eigenvalue_correction=GRAD_WEIGHTED, grad_cov_context=ef_data, target_estimator=exact)`. Reference: reconstruct flat U from the same (a,g), dense `C=UᵀU/N`, `sym(CH)`, per-layer `np.kron` num/den + `lam_frob` fallback with the same threshold (notebook cell-6 formula verbatim) → assert Λ per block (and `estimate_hessian(0.0)`) allclose. Covers ordering, projections, chunking, fallback end-to-end.
  2. **Batched vs single** build consistency (`batch_size=` chunking), as in test_ekfac.py:340.
  3. **Risk optimality property**: `‖(H−A)C^{1/2}‖_F` of ckfac ≤ that of EKFAC and of KFAC built on the same basis (guaranteed by projection optimality; dead directions are risk-neutral, so the inequality is exact).
  4. Finiteness of `estimate_hessian`.

### §7. Verification (run manually)
- `pytest tests/test_ckfac.py tests/test_ekfac.py` once.
- Optional end-to-end regression: rerun notebook cells 3–7 with `"ckfac"` loaded from a fresh `analyze_hessians` factors dir instead of computed inline. Note: library C uses the **full train set** (3823) vs the notebook's 1000-subset, so expect small drift from `‖D‖_F = 20.47 / ‖DC^½‖_F = 5.958`; the dense-reference unit test is the exact-equivalence check.

## Caveats (document in docstrings, don't "fix")

- ckfac Λ is **indefinite and overshoots by design** (mimics off-block curvature the block-diagonal family cannot represent): `‖D‖_F` gets much worse while risk improves. `inverse`/`damped` add damping with no abs → inverses can blow up near Λ+d = 0; `AUTO_MEAN` damping (mean Λ) is semantically dubious here — recommend UNIFORM damping for ckfac runs; primarily a forward-error/risk object initially.
- Fitting C on the full train set while analysis probes are a train subsample is the method's semantics (train-distribution risk), not leakage; held-out risk would be an eval-side change (test-gradient probes).
