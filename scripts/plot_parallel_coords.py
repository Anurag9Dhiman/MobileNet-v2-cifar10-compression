"""Local parallel-coordinates plot from results/sweep_results.csv, rendered via
matplotlib — used in the report as a supplement/fallback to (not a replacement
for) the Wandb-hosted Parallel Coordinates chart."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="./results/sweep_results.csv")
    ap.add_argument("--out_png", type=str, default="./results/parallel_coordinates.png")
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = [
            {k: float(v) for k, v in row.items() if k != "run_name"}
            for row in csv.DictReader(f)
        ]

    dims = ["weight_quant_bits", "activation_quant_bits", "prune_sparsity",
            "compression_ratio", "model_size_mb", "quantized_acc"]
    data = np.array([[r[d] for d in dims] for r in rows])
    mins, maxs = data.min(axis=0), data.max(axis=0)
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    normed = (data - mins) / ranges

    acc = data[:, dims.index("quantized_acc")]
    acc_norm = (acc - acc.min()) / (acc.max() - acc.min() if acc.max() > acc.min() else 1.0)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(dims))
    for i in range(normed.shape[0]):
        ax.plot(xs, normed[i], color=cmap(acc_norm[i]), alpha=0.6, linewidth=1.2)

    for x in xs:
        ax.axvline(x, color="gray", linewidth=0.8, zorder=0)

    ax.set_ylim(-0.12, 1.18)
    ax.set_xticks(xs)
    ax.set_xticklabels(dims, rotation=15, ha="right")
    ax.set_yticks([])
    for i, d in enumerate(dims):
        ax.text(i, 1.10, f"{maxs[i]:.3g}", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
        ax.text(i, -0.10, f"{mins[i]:.3g}", ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_title("MobileNetV2/CIFAR-10 quantization sweep (color = quantized_acc)", pad=16)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(acc.min(), acc.max()))
    fig.colorbar(sm, ax=ax, label="quantized_acc (%)")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png} from {len(rows)} runs")


if __name__ == "__main__":
    main()
