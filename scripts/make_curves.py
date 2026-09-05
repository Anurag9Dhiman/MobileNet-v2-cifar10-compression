"""Generates loss/accuracy curve figures from the training CSV log for the report."""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", type=str, default="./results/train_log.csv")
    ap.add_argument("--out_dir", type=str, default="./results")
    args = ap.parse_args()

    rows = load_log(args.log_csv)
    epochs = [r["epoch"] + 1 for r in rows]
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train loss")
    ax.plot(epochs, [r["test_loss"] for r in rows], label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("MobileNetV2 / CIFAR-10 loss curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [r["train_acc"] for r in rows], label="train top-1 acc")
    ax.plot(epochs, [r["test_acc"] for r in rows], label="test top-1 acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("MobileNetV2 / CIFAR-10 accuracy curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "accuracy_curve.png"), dpi=150)

    best = max(rows, key=lambda r: r["test_acc"])
    print(f"best test_acc={best['test_acc']:.2f} at epoch={int(best['epoch'])+1}")
    print(f"final test_acc={rows[-1]['test_acc']:.2f}")
    print(f"figures written to {args.out_dir}/loss_curve.png and accuracy_curve.png")


if __name__ == "__main__":
    main()
