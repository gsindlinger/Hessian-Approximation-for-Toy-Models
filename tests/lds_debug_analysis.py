# %% [markdown]
# # Part 5 — LDS / ELSO Attribution Quality Analysis
#
# Evaluates how well each Hessian approximation predicts leave-some-out retraining outcomes
# (Linear Datamodelling Score, LDS), following the ELSO protocol of Hong et al. (2025).
#
# **Protocol**
# - Draw $K$ random subsets $S_j \subseteq D$ of fraction $\alpha$.
# - Retrain $R$ models on $D \setminus S_j$ with different seeds; average test losses.
# - Ground truth: $\Delta m_j(z_q) = \mathbb{E}_\xi[L(z_q,\theta(D\setminus S_j))] - L(z_q,\theta(D))$
# - Predicted:    $\hat g(z_q,S_j) = \sum_{z_i\in S_j}\nabla m(z_q)^\top \hat H^{-1}\nabla L(z_i,\theta)$
# - LDS = Spearman$(\{\Delta m_j\}_j,\{\hat g_j\}_j)$ averaged over test queries
#
# **Methods**: Exact, GNH, Block Hessian, Block GNH, KFAC, KFAC (wrong), EKFAC, EKFAC (wrong)
# **Setting**: single MLP, width=16, depth=4, evaluated at epochs {10, 100}

# %%
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr as _spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 11})
torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}  |  device: {DEVICE}")

# %%
# Dataset
DIM_IN = 64
DIM_OUT = 10

digits = load_digits()
X_np, y_np = digits.data.astype(np.float64), digits.target
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.1, random_state=45, stratify=y_np
)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)
X_train = torch.tensor(X_train_np, dtype=torch.float64, device=DEVICE)
X_test = torch.tensor(X_test_np, dtype=torch.float64, device=DEVICE)
y_train = torch.tensor(y_train_np, dtype=torch.long, device=DEVICE)
y_test = torch.tensor(y_test_np, dtype=torch.long, device=DEVICE)
print(
    f"Train: {X_train.shape}  Test: {X_test.shape}  Classes: {len(torch.unique(y_train))}"
)


# %%
# Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


# %%
# Hessian utilities


def extract_block_diagonal(H, model):
    H_bd, offset = torch.zeros_like(H), 0
    for p in model.parameters():
        n = p.numel()
        H_bd[offset : offset + n, offset : offset + n] = H[
            offset : offset + n, offset : offset + n
        ]
        offset += n
    return H_bd


def compute_exact_hessian(model, X, y):
    dev = next(model.parameters()).device
    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    out = model(X)
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])
    H = torch.zeros(P, P, device=dev)
    for i in range(P):
        g2 = torch.autograd.grad(
            grad_flat[i], params, retain_graph=(i < P - 1), allow_unused=True
        )
        H[i] = torch.cat(
            [
                g.flatten() if g is not None else torch.zeros(p.numel(), device=dev)
                for g, p in zip(g2, params)
            ]
        ).detach()
    return H


def compute_gnh(model, X, y):
    dev = next(model.parameters()).device
    n, C = X.shape[0], 10
    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    GNH = torch.zeros(P, P, device=dev)
    for i in range(n):
        xi = X[i : i + 1]
        with torch.no_grad():
            p_i = torch.softmax(model(xi), dim=1).squeeze()
        H_z = torch.diag(p_i) - torch.outer(p_i, p_i)
        logits_i = model(xi)
        J = torch.zeros(C, P, device=dev)
        for c in range(C):
            g2 = torch.autograd.grad(
                logits_i[0, c], params, retain_graph=(c < C - 1), allow_unused=True
            )
            J[c] = torch.cat(
                [
                    g.flatten() if g is not None else torch.zeros(p.numel(), device=dev)
                    for g, p in zip(g2, params)
                ]
            ).detach()
        GNH += J.T @ H_z @ J
    return GNH / n


class KFACCollector:
    def __init__(self, model):
        self.activations, self.gradients, self._hooks = {}, {}, []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                self.activations[name] = []
                self.gradients[name] = []
                self._hooks.append(mod.register_forward_hook(self._fwd(name)))
                self._hooks.append(mod.register_full_backward_hook(self._bwd(name)))

    def _fwd(self, name):
        def hook(mod, inp, out):
            self.activations[name].append(inp[0].detach())

        return hook

    def _bwd(self, name):
        def hook(mod, gi, go):
            self.gradients[name].append(go[0].detach())

        return hook

    @property
    def layer_names(self):
        return list(self.activations.keys())

    def clear(self):
        for k in self.activations:
            self.activations[k] = []
            self.gradients[k] = []

    def remove(self):
        for h in self._hooks:
            h.remove()


def compute_kfac_factors(model, X, mcmc_repetitions=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    dev = X.device
    N, C = X.shape[0], 10
    coll = KFACCollector(model)
    A_sum = {k: None for k in coll.layer_names}
    G_sum = {k: None for k in coll.layer_names}
    for i in range(N):
        xi = X[i : i + 1]
        with torch.no_grad():
            probs = torch.softmax(model(xi), dim=1).squeeze()
        cw = (
            [(c, probs[c].item()) for c in range(C)]
            if mcmc_repetitions is None
            else [
                (c.item(), 1.0)
                for c in torch.multinomial(probs, mcmc_repetitions, replacement=True)
            ]
        )
        for c, w in cw:
            model.zero_grad()
            coll.clear()
            F.cross_entropy(model(xi), torch.tensor([c], device=dev)).backward()
            for name in coll.layer_names:
                a = coll.activations[name][0].squeeze(0)
                g = coll.gradients[name][0].squeeze(0)
                A_sum[name] = (
                    w * torch.outer(a, a)
                    if A_sum[name] is None
                    else A_sum[name] + w * torch.outer(a, a)
                )
                G_sum[name] = (
                    w * torch.outer(g, g)
                    if G_sum[name] is None
                    else G_sum[name] + w * torch.outer(g, g)
                )
    coll.remove()
    model.zero_grad()
    norm = N if mcmc_repetitions is None else N * mcmc_repetitions
    A_covs = {k: (v + v.T) / (2 * norm) for k, v in A_sum.items()}
    G_covs = {k: (v + v.T) / (2 * norm) for k, v in G_sum.items()}
    return A_covs, G_covs


def kfac_full_matrix(model, A_covs, G_covs):
    dev = next(model.parameters()).device
    P = sum(p.numel() for p in model.parameters())
    H, offset = torch.zeros(P, P, device=dev), 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            sz = mod.weight.numel()
            H[offset : offset + sz, offset : offset + sz] = torch.kron(
                G_covs[name], A_covs[name]
            )
            offset += sz
    return H


def kfac_full_matrix_wrong(model, A_covs, G_covs):
    dev = next(model.parameters()).device
    P = sum(p.numel() for p in model.parameters())
    H, offset = torch.zeros(P, P, device=dev), 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            sz = mod.weight.numel()
            H[offset : offset + sz, offset : offset + sz] = torch.kron(
                A_covs[name], G_covs[name]
            )  # wrong
            offset += sz
    return H


def compute_kfac_eigenvectors(A_covs, G_covs):
    Q_A, Q_G, lam_A, lam_G = {}, {}, {}, {}
    for name in A_covs:
        va, qa = torch.linalg.eigh(A_covs[name])
        vg, qg = torch.linalg.eigh(G_covs[name])
        Q_A[name], lam_A[name] = qa, va
        Q_G[name], lam_G[name] = qg, vg
    return Q_A, Q_G, lam_A, lam_G


def compute_ekfac_corrections(model, X, Q_A, Q_G, mcmc_repetitions=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    dev = X.device
    N, C = X.shape[0], 10
    coll = KFACCollector(model)
    corr_sum = {
        name: torch.zeros(mod.weight.shape, dtype=X.dtype, device=dev)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name in Q_A
    }
    for i in range(N):
        xi = X[i : i + 1]
        with torch.no_grad():
            probs = torch.softmax(model(xi), dim=1).squeeze()
        cw = (
            [(c, probs[c].item()) for c in range(C)]
            if mcmc_repetitions is None
            else [
                (c.item(), 1.0)
                for c in torch.multinomial(probs, mcmc_repetitions, replacement=True)
            ]
        )
        for c, w in cw:
            model.zero_grad()
            coll.clear()
            F.cross_entropy(model(xi), torch.tensor([c], device=dev)).backward()
            for name in corr_sum:
                a = coll.activations[name][0].squeeze(0)
                g = coll.gradients[name][0].squeeze(0)
                a_tilde = a @ Q_A[name]  # (I,) = Q_A^T a
                g_tilde = g @ Q_G[name]  # (O,) = Q_G^T g
                corr_sum[name] += w * torch.outer(g_tilde, a_tilde) ** 2
    coll.remove()
    model.zero_grad()
    norm = N if mcmc_repetitions is None else N * mcmc_repetitions
    return {k: v / norm for k, v in corr_sum.items()}


def ekfac_full_matrix(model, Q_A, Q_G, corrections):
    dev = next(model.parameters()).device
    P = sum(p.numel() for p in model.parameters())
    H, offset = torch.zeros(P, P, device=dev), 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in Q_A:
            sz = mod.weight.numel()
            Q = torch.kron(Q_G[name], Q_A[name])  # G outer, A inner (correct)
            lam = corrections[name].flatten()
            H[offset : offset + sz, offset : offset + sz] = Q @ torch.diag(lam) @ Q.T
            offset += sz
    return H


def compute_ekfac_corrections_wrong(
    model, X, Q_A, Q_G, mcmc_repetitions=None, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)
    dev = X.device
    N, C = X.shape[0], 10
    coll = KFACCollector(model)
    corr_sum = {
        name: torch.zeros(
            mod.weight.shape[1], mod.weight.shape[0], dtype=X.dtype, device=dev
        )
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name in Q_A
    }
    for i in range(N):
        xi = X[i : i + 1]
        with torch.no_grad():
            probs = torch.softmax(model(xi), dim=1).squeeze()
        cw = (
            [(c, probs[c].item()) for c in range(C)]
            if mcmc_repetitions is None
            else [
                (c.item(), 1.0)
                for c in torch.multinomial(probs, mcmc_repetitions, replacement=True)
            ]
        )
        for c, w in cw:
            model.zero_grad()
            coll.clear()
            F.cross_entropy(model(xi), torch.tensor([c], device=dev)).backward()
            for name in corr_sum:
                a = coll.activations[name][0].squeeze(0)
                g = coll.gradients[name][0].squeeze(0)
                a_tilde = Q_A[name] @ a  # wrong: Q@v instead of v@Q
                g_tilde = Q_G[name] @ g
                corr_sum[name] += w * torch.outer(a_tilde, g_tilde) ** 2
    coll.remove()
    model.zero_grad()
    norm = N if mcmc_repetitions is None else N * mcmc_repetitions
    return {k: v / norm for k, v in corr_sum.items()}


def ekfac_full_matrix_wrong(model, Q_A, Q_G, corrections):
    dev = next(model.parameters()).device
    P = sum(p.numel() for p in model.parameters())
    H, offset = torch.zeros(P, P, device=dev), 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in Q_A:
            sz = mod.weight.numel()
            Q = torch.kron(Q_A[name], Q_G[name])  # A outer, G inner (wrong)
            lam = corrections[name].flatten()
            H[offset : offset + sz, offset : offset + sz] = Q @ torch.diag(lam) @ Q.T
            offset += sz
    return H


# %%
## 5.1  Hyperparameters
LDS_HIDDEN_DIMS = [16] * 4  # depth-4 MLP (4 hidden layers, width 16)
LDS_EPOCHS_LIST = [10, 100, 1000]
LDS_NUM_SUBSETS = 100  # K
LDS_SEEDS_PER_SUBSET = 50  # R
LDS_SUBSET_FRAC = 0.5  # α
LDS_EPS = 1e-4  # pseudo-inverse eigenvalue threshold
LDS_BASE_SEED = 105  # seed for the reference model
LDS_SUBSET_SEED = 205  # seed for subset generation
LDS_LR = 0.03
LDS_BATCH_SIZE = 32

_N_train = len(X_train)
_sub_size = int(LDS_SUBSET_FRAC * _N_train)
_total = LDS_NUM_SUBSETS * LDS_SEEDS_PER_SUBSET * len(LDS_EPOCHS_LIST)
print(f"Model : MLP(64 → {' → '.join(str(h) for h in LDS_HIDDEN_DIMS)} → 10)")
print(f"N_train={_N_train}, subset_size={_sub_size}")
print(
    f"Total ELSO retrains: {LDS_NUM_SUBSETS} × {LDS_SEEDS_PER_SUBSET} × {len(LDS_EPOCHS_LIST)} = {_total}"
)


# %%
## 5.2  Utility functions


def train_model_on(model, X, y, seed, epochs, lr=LDS_LR, batch_size=LDS_BATCH_SIZE):
    """Train `model` on (X, y) for `epochs` steps from the current parameter state."""
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(Xb), yb).backward()
            opt.step()
    model.eval()


def per_sample_grads(model, X, y):
    """Returns (N, P) tensor: ∇_θ L(z_i, θ) for each training point."""
    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    G = torch.zeros(len(X), P, dtype=X.dtype, device=X.device)
    for i in range(len(X)):
        model.zero_grad()
        F.cross_entropy(model(X[i : i + 1]), y[i : i + 1]).backward()
        G[i] = torch.cat([p.grad.flatten() for p in params]).detach()
    model.zero_grad()
    return G


def pinv_ihvp(H, V, eps=LDS_EPS):
    """Eigen-decomposition pseudo-inverse IHVP.
    V : (n, P) row vectors  →  returns (n, P): Ĥ^{-1} v for each row."""
    lm, Q = torch.linalg.eigh(H)
    inv_lm = torch.where(lm.abs() > eps, 1.0 / lm, torch.zeros_like(lm))
    return (V @ Q) * inv_lm @ Q.T


def approx_error_eq9(H_exact, H_approx, V, eps=LDS_EPS):
    """Eq. 9 of Hong et al.:  (1/N) Σ ‖H Ĥ^{-1}v_i − v_i‖² / ‖v_i‖²"""
    Hinv_V = pinv_ihvp(H_approx, V, eps)  # (N, P)
    HHinv_V = Hinv_V @ H_exact.T  # (N, P)
    num = ((HHinv_V - V) ** 2).sum(-1)
    denom = (V**2).sum(-1).clamp(min=1e-20)
    return (num / denom).mean().item()


def build_hessians_lds(model, X, y):
    """Compute all 8 Hessian variants used in the LDS experiment."""
    print("    H_exact", end="", flush=True)
    H_exact = compute_exact_hessian(model, X, y)

    print(" · GNH", end="", flush=True)
    H_gnh = compute_gnh(model, X, y)

    H_bd = extract_block_diagonal(H_exact, model)
    H_gnh_bd = extract_block_diagonal(H_gnh, model)

    print(" · KFAC", end="", flush=True)
    A, G = compute_kfac_factors(model, X)
    H_kfac = kfac_full_matrix(model, A, G)
    H_kfac_wrong = kfac_full_matrix_wrong(model, A, G)

    print(" · EKFAC", end="", flush=True)
    QA, QG, _, _ = compute_kfac_eigenvectors(A, G)
    corr = compute_ekfac_corrections(model, X, QA, QG)
    corr_w = compute_ekfac_corrections_wrong(model, X, QA, QG)
    H_ekfac = ekfac_full_matrix(model, QA, QG, corr)
    H_ekfac_wrong = ekfac_full_matrix_wrong(model, QA, QG, corr_w)

    print(" ✓", flush=True)
    return {
        "Exact": H_exact,
        "GNH": H_gnh,
        "Block Hessian": H_bd,
        "Block GNH": H_gnh_bd,
        "KFAC": H_kfac,
        "KFAC (wrong)": H_kfac_wrong,
        "EKFAC": H_ekfac,
        "EKFAC (wrong)": H_ekfac_wrong,
    }


# %%
## 5.3  ELSO ground-truth retraining


def run_elso(subset_indices, epochs):
    """
    For each subset S_j retrain LDS_SEEDS_PER_SUBSET models on D\\S_j.
    Returns (N_test, K) tensor of per-query mean test losses.
    Seeds are deterministic in (j, r) and epoch-independent.
    torch.manual_seed is set BEFORE MLP construction so that both
    weight initialisation and minibatch ordering are controlled.
    """
    dev = X_test.device
    K = len(subset_indices)
    N_test = len(X_test)
    losses = torch.zeros(N_test, K, dtype=torch.float64, device=dev)

    for j, idx in enumerate(subset_indices):
        mask = torch.ones(len(X_train), dtype=torch.bool, device=dev)
        mask[torch.tensor(idx, device=dev)] = False
        X_sub, y_sub = X_train[mask], y_train[mask]

        seed_sum = torch.zeros(N_test, dtype=torch.float64, device=dev)
        for r in range(LDS_SEEDS_PER_SUBSET):
            seed = j * LDS_SEEDS_PER_SUBSET + r  # deterministic, epoch-independent
            torch.manual_seed(seed)  # controls both init AND batch order
            m = MLP(DIM_IN, LDS_HIDDEN_DIMS, DIM_OUT).to(dev)
            train_model_on(m, X_sub, y_sub, seed=seed, epochs=epochs)
            with torch.no_grad():
                seed_sum += F.cross_entropy(
                    m(X_test), y_test, reduction="none"
                ).double()

        losses[:, j] = seed_sum / LDS_SEEDS_PER_SUBSET
        if (j + 1) % 6 == 0 or (j + 1) == K:
            print(f"    [{j + 1:2d}/{K}]", end=" ", flush=True)

    print()
    return losses  # (N_test, K)


# %%
## 5.4  LDS scoring


def compute_lds(ihvp_q, train_grads, subset_indices, delta_m, n_bootstrap=2000):
    """
    ihvp_q         : (N_q, P)   Ĥ^{-1} ∇m(z_q) for each test query
    train_grads    : (N_train, P)
    subset_indices : list of K index-lists
    delta_m        : (N_q, K)   ground-truth Δm_j(z_q)  [already averaged over R seeds]

    Protocol (Hong et al. 2025, App. A):
      Step 3 — for each query z_q, compute Spearman r over the K subsets between
               {Δm_j(z_q)} and {g_τ(z_q, S_j)}.
      Step 4 — average those per-query r values → scalar LDS.
      CI     — 95 % bootstrap CI by resampling the K subsets with replacement,
               accounting for "randomness in subset selection".

    Returns: (mean_LDS, ci_low, ci_high, per_query_lds_array)
    """
    N_train = train_grads.shape[0]
    K = len(subset_indices)
    N_q = ihvp_q.shape[0]

    # Build (N_train, K) binary membership matrix
    group_mat = torch.zeros(
        N_train, K, dtype=train_grads.dtype, device=train_grads.device
    )
    for j, idx in enumerate(subset_indices):
        group_mat[idx, j] = 1.0

    # Predicted group effects: (N_q, K)
    predicted_np = ((ihvp_q @ train_grads.T) @ group_mat).detach().cpu().numpy()
    delta_m_np = delta_m.cpu().numpy()  # (N_q, K)

    def _mean_lds_over_j(j_idx):
        """Mean Spearman r across queries, evaluated on subset-index array j_idx."""
        rs = []
        for q in range(N_q):
            r, _ = _spearmanr(delta_m_np[q, j_idx], predicted_np[q, j_idx])
            rs.append(0.0 if np.isnan(r) else float(r))
        return np.mean(rs)

    all_j = np.arange(K)
    mean_lds = _mean_lds_over_j(all_j)

    # Per-query r on full K subsets (for diagnostics / table output)
    lds_vals = np.array(
        [
            (lambda r: 0.0 if np.isnan(r) else float(r))(
                _spearmanr(delta_m_np[q], predicted_np[q])[0]
            )
            for q in range(N_q)
        ]
    )

    # 95 % bootstrap CI — resample K subsets with replacement
    rng_boot = np.random.default_rng(0)
    boot_means = np.array(
        [
            _mean_lds_over_j(rng_boot.choice(K, size=K, replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return mean_lds, ci_low, ci_high, lds_vals


# %%
## 5.5  Main LDS experiment loop

# Shared subset indices — identical across all epoch settings
_rng = np.random.default_rng(LDS_SUBSET_SEED)
_sub_size = int(LDS_SUBSET_FRAC * len(X_train))
_subset_indices = [
    _rng.choice(len(X_train), size=_sub_size, replace=False).tolist()
    for _ in range(LDS_NUM_SUBSETS)
]

lds_all = {}  # epoch → {method → {lds_mean, ci_low, ci_high, approx_error}}

for ep in LDS_EPOCHS_LIST:
    t0 = time.time()
    print(f"\n{'=' * 60}\nEpoch = {ep}\n{'=' * 60}")

    # ── reference model ───────────────────────────────────────────────
    print("  Training reference model...", flush=True)
    torch.manual_seed(LDS_BASE_SEED)  # controls both init AND batch order
    _model_ref = MLP(DIM_IN, LDS_HIDDEN_DIMS, DIM_OUT).to(DEVICE)
    train_model_on(_model_ref, X_train, y_train, seed=LDS_BASE_SEED, epochs=ep)
    with torch.no_grad():
        _ref_losses = F.cross_entropy(
            _model_ref(X_test), y_test, reduction="none"
        ).double()  # (N_test,)

    # ── Hessians ──────────────────────────────────────────────────────
    print("  Hessians:", flush=True)
    _hessians = build_hessians_lds(_model_ref, X_train, y_train)

    # ── per-sample gradients ──────────────────────────────────────────
    print("  Per-sample gradients (train)...", flush=True)
    _train_grads = per_sample_grads(_model_ref, X_train, y_train)
    print("  Per-sample gradients (test)...", flush=True)
    _query_grads = per_sample_grads(_model_ref, X_test, y_test)

    # ── ELSO ground truth ─────────────────────────────────────────────
    print(
        f"  ELSO retraining (K={LDS_NUM_SUBSETS}, R={LDS_SEEDS_PER_SUBSET}, ep={ep})...",
        flush=True,
    )
    _retrained = run_elso(_subset_indices, ep)  # (N_test, K)
    _delta_m = _retrained - _ref_losses.unsqueeze(1)  # (N_test, K)

    # ── LDS per method ────────────────────────────────────────────────
    print("  LDS scores:", flush=True)
    _epoch_res = {}
    for name, H in _hessians.items():
        _ihvp_q = pinv_ihvp(H, _query_grads)
        lds_mean, ci_low, ci_high, _ = compute_lds(
            _ihvp_q, _train_grads, _subset_indices, _delta_m
        )
        app_err = (
            0.0
            if name == "Exact"
            else approx_error_eq9(_hessians["Exact"], H, _train_grads)
        )
        _epoch_res[name] = {
            "lds_mean": lds_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "approx_error": app_err,
        }
        print(
            f"    {name:<20s}  LDS={lds_mean:+.4f} "
            f"[{ci_low:+.4f}, {ci_high:+.4f}]  err={app_err:.3e}"
        )

    lds_all[ep] = _epoch_res
    print(f"  elapsed: {time.time() - t0:.0f}s")

print("\n✓ All epochs done.")


# %%
## 5.6  Results table

_methods = list(lds_all[LDS_EPOCHS_LIST[0]].keys())

print(f"{'Method':<22}" + "".join(f"  {'ep=' + str(ep):>22}" for ep in LDS_EPOCHS_LIST))
print("-" * (22 + 24 * len(LDS_EPOCHS_LIST)))
for nm in _methods:
    row = f"{nm:<22}"
    for ep in LDS_EPOCHS_LIST:
        r = lds_all[ep][nm]
        row += f"  {r['lds_mean']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
    print(row)

print()
print("Approximation Error (Eq. 9)")
print("-" * (22 + 18 * len(LDS_EPOCHS_LIST)))
for nm in _methods:
    row = f"{nm:<22}"
    for ep in LDS_EPOCHS_LIST:
        row += f"  {lds_all[ep][nm]['approx_error']:>16.4e}"
    print(row)


# %%
## 5.7  Visualization

_methods = list(lds_all[LDS_EPOCHS_LIST[0]].keys())
_palette = plt.cm.tab10(np.linspace(0, 0.9, len(_methods)))
_COLS = dict(zip(_methods, _palette))
_MARKS = dict(zip(_methods, ["o", "s", "D", "^", "v", "<", ">", "P"]))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ── left: grouped bar chart of LDS by epoch ───────────────────────────
ax = axes[0]
n_m = len(_methods)
n_ep = len(LDS_EPOCHS_LIST)
x = np.arange(n_ep)
width = 0.8 / n_m

for i, nm in enumerate(_methods):
    means = [lds_all[ep][nm]["lds_mean"] for ep in LDS_EPOCHS_LIST]
    ci_lows = [lds_all[ep][nm]["ci_low"] for ep in LDS_EPOCHS_LIST]
    ci_highs = [lds_all[ep][nm]["ci_high"] for ep in LDS_EPOCHS_LIST]
    # asymmetric error bars from bootstrap CI
    err_lo = [m - lo for m, lo in zip(means, ci_lows)]
    err_hi = [hi - m for m, hi in zip(means, ci_highs)]
    off = (i - n_m / 2 + 0.5) * width
    ax.bar(
        x + off,
        means,
        width,
        yerr=[err_lo, err_hi],
        label=nm,
        color=_COLS[nm],
        alpha=0.85,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

ax.set_xticks(x)
ax.set_xticklabels([f"ep={ep}" for ep in LDS_EPOCHS_LIST])
ax.set_ylabel("LDS (Spearman ρ, mean with 95 % bootstrap CI)")
ax.set_title("Attribution Quality (LDS) — width=16, depth=4")
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.axhline(0, color="black", lw=0.7, ls="--")

# ── right: approx error vs LDS scatter (one point per method × epoch) ─
ax = axes[1]
for ep in LDS_EPOCHS_LIST:
    for nm in _methods:
        if nm == "Exact":
            continue
        r = lds_all[ep][nm]
        ax.errorbar(
            r["approx_error"],
            r["lds_mean"],
            yerr=[[r["lds_mean"] - r["ci_low"]], [r["ci_high"] - r["lds_mean"]]],
            fmt=_MARKS[nm],
            color=_COLS[nm],
            markersize=8,
            capsize=3,
            elinewidth=0.8,
            zorder=3,
            label=nm if ep == LDS_EPOCHS_LIST[0] else None,
        )
        ax.annotate(
            str(ep),
            xy=(r["approx_error"], r["lds_mean"]),
            fontsize=6,
            xytext=(4, 1),
            textcoords="offset points",
            color="dimgray",
        )

ax.set_xscale("log")
ax.set_xlabel("Approximation Error  (Eq. 9 of Hong et al.)")
ax.set_ylabel("LDS (Spearman ρ, mean with 95 % bootstrap CI)")
ax.set_title("Approx. Error vs Attribution Quality")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3, which="both")
ax.axhline(0, color="black", lw=0.7, ls="--")

plt.tight_layout()
plt.savefig("lds_analysis_width16_depth4.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: lds_analysis_width16_depth4.png")
