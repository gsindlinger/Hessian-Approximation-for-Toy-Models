"""Generate train/val loss and accuracy curve PDFs from offline wandb runs.

Reads per-epoch history straight out of each run's binary `.wandb` datastore
(no wandb server / sync required) and writes two PDFs per run into
`figures/loss_and_acc/`:
    <label_slug>_loss.pdf   train vs val loss (log-y)
    <label_slug>_acc.pdf    train vs val accuracy

Run:
    python plots/loss_acc_curves.py
"""

import json, os, glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "loss_and_acc")
os.makedirs(OUT, exist_ok=True)

# Human-readable labels, keyed by the model config hash (= model dir name).
MODEL_LABELS = {
    "mlp_08580ee2573a": "Small MLP8",
    "resnet_mlp_0a69ab6297da": "Small ResNet MLP3",
    "resnet_mlp_adc3030f5daa": "Small ResNet MLP8",
    "resnet_mlp_fbc1db7ec868": "ResNet MLP3",
    "resnet_mlp_swiglu_d4186381c706": "ResNet SwiGLU3",
}

# (dataset, model_name, wandb_run_dir) -- run dir matched via config-name == model_name.
RUNS = [
    ("digits", "mlp_08580ee2573a", "run-20260525_113150-tdiyji6k"),
    ("digits", "resnet_mlp_0a69ab6297da", "run-20260525_113344-idev8mzf"),
    ("digits", "resnet_mlp_adc3030f5daa", "run-20260525_113543-9tbz810d"),
    ("fashion_mnist", "resnet_mlp_fbc1db7ec868", "run-20260525_131609-efms7fp3"),
    ("fashion_mnist", "resnet_mlp_swiglu_d4186381c706", "run-20260525_131930-q6cvxu5e"),
]


def read_history(run_dir):
    """Scan the offline .wandb datastore and return per-step history rows."""
    wf = glob.glob(os.path.join(ROOT, "wandb", run_dir, "*.wandb"))[0]
    ds = DataStore()
    ds.open_for_scan(wf)
    rows = []
    while True:
        rec = ds.scan_data()
        if rec is None:
            break
        r = pb.Record()
        r.ParseFromString(rec)
        if r.WhichOneof("record_type") == "history":
            row = {}
            for it in r.history.item:
                k = it.key or ".".join(it.nested_key)
                try:
                    row[k] = json.loads(it.value_json)
                except Exception:
                    row[k] = it.value_json
            rows.append(row)
    rows.sort(key=lambda d: d.get("_step", 0))
    return rows


def col(rows, key):
    xs, ys = [], []
    for d in rows:
        if key in d and d[key] is not None:
            xs.append(d["_step"])
            ys.append(d[key])
    return xs, ys


def main():
    for dataset, name, run_dir in RUNS:
        rows = read_history(run_dir)
        label = MODEL_LABELS[name]
        tag = label.lower().replace(" ", "_")

        # ---- loss ----
        fig, ax = plt.subplots(figsize=(6, 4))
        x, y = col(rows, "train/loss")
        ax.plot(x, y, label="train loss", lw=1.5)
        x, y = col(rows, "val/loss")
        ax.plot(x, y, label="val loss", lw=1.5)
        # ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{label} ({dataset})\ntrain vs val loss", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OUT, f"{tag}_loss.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)

        # ---- accuracy ----
        fig, ax = plt.subplots(figsize=(6, 4))
        x, y = col(rows, "train/accuracy")
        ax.plot(x, y, label="train acc", lw=1.5)
        x, y = col(rows, "val/accuracy")
        ax.plot(x, y, label="val acc", lw=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{label} ({dataset})\ntrain vs val accuracy", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OUT, f"{tag}_acc.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)


if __name__ == "__main__":
    main()
