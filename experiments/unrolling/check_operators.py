"""Explicit tiny CPU checks for the standalone POC; no result/data/DB writes.

Run directly after authorizing the short numerical checks. The nonlinear smoke
test reads the existing Digits split; closed-form checks use in-memory data.
"""

from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


def main():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np

    source = Path(__file__).with_name("unrolling.py")
    print("Source SHA256:", hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
    spec = importlib.util.spec_from_file_location("unrolling_operator_subject", source)
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)

    methods = ["zero", "identity", "h_batch", "gnh_batch", "block_h_batch",
               "kfac_batch", "ekfac_batch", "foof_batch"]
    base = {**subject.CONFIG, "architecture": "mlp", "widths": [3], "horizons": [0, 1], "start_step": 0,
            "batch_size": 4, "num_probes": 3, "epsilons": [1e-4, 1e-3, 1e-2],
            "methods": methods, "normalize_probes": False, "learning_rate": .03,
            "activation": "tanh", "loss": "cross_entropy"}

    def responses(result):
        return dict(zip(methods, result["snapshots"][1]["responses"]))

    # Pure metric check: positive uniform shrinkage preserves directions/ranks
    # but must not be mistaken for exact magnitude agreement.
    probes = np.eye(3)
    q = np.array([1., 2., 4.])
    epsilon = np.array(base["epsilons"])
    metric_methods = ["zero", "identity", "h_batch"]
    snap = {"responses": np.stack([probes, .97 * probes, probes]), "query_gradient": q,
            "plus_delta": epsilon[:, None, None] * probes,
            "minus_delta": -epsilon[:, None, None] * probes,
            "query_plus": 10 + epsilon[:, None] * (probes @ q),
            "query_minus": 10 - epsilon[:, None] * (probes @ q)}
    rows = subject.measure(snap, probes, metric_methods, epsilon, 0)
    for row in rows:
        assert row["valid_correction_norms"] == 0 and row["valid_correction_cosines"] == 0
        assert np.isnan(row["correction_cosine"])
        np.testing.assert_allclose(row["score_spearman_exact"], 1., atol=1e-12)
        np.testing.assert_allclose(row["query_cosine_mae_fd"], 0., atol=1e-12)
        expected = .03 if row["method"] == "identity" else 0.
        np.testing.assert_allclose(row["score_rel_l2_exact"], expected, atol=1e-12)
    exact = .8 * probes + .1 * np.roll(probes, 1, axis=1)
    snap.update(responses=np.stack([probes, .97 * probes, exact]),
                plus_delta=epsilon[:, None, None] * exact,
                minus_delta=-epsilon[:, None, None] * exact,
                query_plus=10 + epsilon[:, None] * (exact @ q),
                query_minus=10 - epsilon[:, None] * (exact @ q))
    for row in subject.measure(snap, probes, metric_methods, epsilon, 1):
        if row["method"] == "zero":
            np.testing.assert_allclose(row["correction_rel_error"], 1., atol=1e-12)
            assert row["valid_correction_cosines"] == 0
        if row["method"] == "h_batch":
            np.testing.assert_allclose(row["score_rel_l2_exact"], 0., atol=1e-12)
    print("PASS metric zero/identity/correction definitions", flush=True)

    smoke = subject.run_experiment(base)
    u = responses(smoke)
    np.testing.assert_array_equal(u["zero"], smoke["protocol"]["probes"])
    np.testing.assert_allclose(u["identity"], .97 * u["zero"], rtol=1e-12, atol=1e-12)
    assert all(np.isfinite(value).all() for value in u.values())
    smoke_rows = [r for r in smoke["rows"] if r["horizon"] == 1 and r["method"] == "h_batch"]
    smallest = min(smoke_rows, key=lambda r: r["epsilon"])
    assert smallest["response_rel_l2_fd"] < 1e-5, smallest
    print("PASS width[3] Digits CE smoke; exact endpoint FD relative errors:",
          [(r["epsilon"], r["response_rel_l2_fd"]) for r in smoke_rows], flush=True)

    rng = np.random.default_rng(123)
    train_x, test_x = rng.normal(size=(8, 3)), rng.normal(size=(5, 3))

    @contextmanager
    def memory_split(train_y, test_y):
        original_load = np.load
        @contextmanager
        def dataset(part):
            yield {"inputs": (train_x if part == "train" else test_x).copy(),
                   "targets": (train_y if part == "train" else test_y).copy()}
        def load(path, *args, **kwargs):
            path = Path(path)
            if "__operator_memory_split__" in path.parts:
                return dataset(path.parent.name)
            return original_load(path, *args, **kwargs)
        with patch.object(np, "load", side_effect=load):
            yield str(source.parent / "__operator_memory_split__")

    # For linear logits H = GNH. Independently form CE output curvature and
    # factor products in NumPy to catch class-count / batch-count mistakes.
    with memory_split(np.arange(8) % 3, np.arange(5) % 3) as split:
        ce = subject.run_experiment({**base, "widths": [], "split_dir": split})
    u = responses(ce)
    bx = train_x[ce["protocol"]["batch_indices"][0]]
    weight = ce["protocol"]["theta_start"].reshape(3, 3)
    logits = bx @ weight
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    covariance = np.stack([np.diag(p) - np.outer(p, p) for p in probabilities])
    a, g = bx.T @ bx / len(bx), covariance.mean(axis=0)
    v = ce["protocol"]["probes"].reshape(3, 3, 3)
    logits_jvp = np.einsum("bi,nic->nbc", bx, v)
    curved_logits = np.einsum("nbc,bce->nbe", logits_jvp, covariance)
    exact_product = np.einsum("bi,nbe->nie", bx, curved_logits) / len(bx)
    expected_h = v - .03 * exact_product
    for method in ("h_batch", "gnh_batch", "block_h_batch"):
        np.testing.assert_allclose(u[method], expected_h.reshape(3, -1), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(u["kfac_batch"], (v - .03 * (a @ v @ g)).reshape(3, -1), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(u["foof_batch"], (v - .03 * (a @ v)).reshape(3, -1), rtol=1e-10, atol=1e-11)
    _, qa = np.linalg.eigh(a)
    _, qg = np.linalg.eigh(g)
    diagonals = np.diagonal(qg.T @ covariance @ qg, axis1=-2, axis2=-1)
    corrected = ((bx @ qa) ** 2).T @ diagonals / len(bx)
    ekfac_product = qa @ (corrected * (qa.T @ v @ qg)) @ qg.T
    np.testing.assert_allclose(u["ekfac_batch"], (v - .03 * ekfac_product).reshape(3, -1), rtol=1e-10, atol=1e-11)
    print("PASS linear CE H/GNH/block-H and independently formed KFAC/EKFAC/FOOF", flush=True)

    for outputs in (1, 3):
        with memory_split(rng.normal(size=(8, outputs)), rng.normal(size=(5, outputs))) as split:
            mse = subject.run_experiment({**base, "widths": [], "loss": "mse", "split_dir": split})
        u = responses(mse)
        normalization = mse["normalization"]["x"]
        normalized_x = (train_x - normalization["mean"]) / normalization["std"]
        bx = normalized_x[mse["protocol"]["batch_indices"][0]]
        a = bx.T @ bx / len(bx)
        v = mse["protocol"]["probes"].reshape(3, 3, outputs)
        expected_h = (v - .03 * (2 / outputs) * (a @ v)).reshape(3, -1)
        for method in ("h_batch", "gnh_batch", "block_h_batch", "kfac_batch", "ekfac_batch"):
            np.testing.assert_allclose(u[method], expected_h, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(u["foof_batch"], (v - .03 * (a @ v)).reshape(3, -1), rtol=1e-10, atol=1e-11)
        exact_rows = [r for r in mse["rows"] if r["horizon"] == 1 and r["method"] == "h_batch"]
        assert all(r["response_rel_l2_fd"] < 1e-8 and r["score_rel_l2_fd"] < 1e-8 for r in exact_rows)
        print(f"PASS linear MSE output_dim={outputs}: 2/output_dim scaling and finite differences", flush=True)
    print("All tiny CPU checks passed. No result, dataset or database files written.", flush=True)


if __name__ == "__main__":
    main()
