# Transformer curvature conventions

The toy decoder exposes **every trainable parameter leaf** in the order used by
`jax.flatten_util.ravel_pytree`. Embeddings, LayerNorm scale and bias, all dense
biases, QKV, attention output, MLP matrices, and the output head are included.
Matrices keep their native `(input, output)` shape; vectors form `(1,D)` blocks.
The exact Hessian and GNH include **all cross-block entries**, not a last-layer
or frozen-backbone subspace. Block approximations deliberately discard those
cross-block entries. This grouping differs from an affine kernel-plus-bias block
and is recorded in the collector manifest.

## Cache and Fisher

For a shared matrix, activations have shape `(N,T,I)` and true preactivation
cotangents have shape `(N,T,O,k)`. Here `k` indexes empirical labels, sampled
pseudo-targets, or all classes. The per-example parameter gradient is

`D[n,:,:,k] = sum_t outer(a[n,t,:], g[n,t,:,k])`.

FIM, block FIM, Shampoo moments, and eigenvalue corrections sum over tokens
**before** taking squares or outer products. Thus these quantities retain
cross-token terms. Final-token loss cotangents propagate through causal
attention and residual connections using autodiff; they are not treated as
independent token losses. Embeddings use one-hot activations. LayerNorm bias
uses activation one and its output cotangent; scale uses activation one and
the product of normalized input with its output cotangent. Their upstream
normalization Jacobian is differentiated normally.

## Kronecker approximations

For shared matrices, KFAC uses the **KFAC-expand** approximation of
[Eschenhagen et al. (NeurIPS 2023), Section 3.2 and Appendix B](https://arxiv.org/html/2311.00636v2).
The scalar normalization is assigned as

`A = (1/N) sum_n sum_t a[n,t] a[n,t]^T`

`G = (1/(T sum_nk p[n,k])) sum_nkt p[n,k] g[n,t,k] g[n,t,k]^T`.

This gives the same Kronecker product as assigning the reciprocal `T` to A.
The task has one final-token label per example (a *reduce setting* in that
paper's terminology), but expand is an explicit approximation choice that can
be used in either setting. It is **not** claimed to be exact for attention.
MAC retains this sum-token A and replaces G by identity. For an independent
identity-curvature loss over layer outputs, sum-token A is the corresponding
shared-linear-map GNH. With T=1 these formulas reduce to the existing dense
formulas. No extra trainable matrix is introduced for vector parameters.

Shampoo uses left/right moments of the complete D matrices and the existing
trace-restricted normalization. A truly zero-gradient block yields zero rather
than a `0/0` NaN. The E variants project complete D into the chosen Kronecker
eigenbasis before squaring, retaining cross-token terms in the correction.
The identity variant remains identity over all parameters.

## Exact references and tests

The transformer exact Hessian streams examples and chunks differentiated
parameter directions (128 by default). It materializes the complete P-by-P
matrix; chunking only limits autodiff workspace. GNH streams full C-by-P output
Jacobians, with all classes and parameters present. The existing dense-model
reference paths remain unchanged.

New analysis recipes set `computation_config.parameter_dtype: float64`. The
analysis driver promotes **every floating parameter leaf before flattening**,
and promotes continuous inputs/targets while retaining integer token IDs and
labels. This avoids the mixed-dtype `ravel_pytree` pitfall where a float64 flat
vector is silently unpacked into original float32 leaves. Promotion is
analysis-only: checkpoint values and the training recipe are not rewritten.
Precision-specific collector/factor directories and manifest provenance prevent
reuse of a float32 cache in a float64 analysis. `null` preserves legacy behavior.

`tests/test_shared_parameter_curvature.py` checks token sums and normalization
against explicit independent NumPy loops. `tests/test_transformer_curvature.py`
checks all-parameter cache reconstruction, parameter order, cache persistence,
full Hessian against direct autodiff, all-classes Fisher against full GNH, and
all default block/Kronecker estimators on an untrained tiny decoder. These are
correctness checks, not training-performance or large-run feasibility claims.
