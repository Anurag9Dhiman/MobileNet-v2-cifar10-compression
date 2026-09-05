"""Pulls every run's config + summary metrics from the Wandb project into a
local CSV (used for the report table and as a local fallback parallel-coordinates
plot if the Wandb UI chart isn't screenshotted)."""
import argparse
import csv

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default="cs6886-a2-mobilenetv2-cifar10")
    ap.add_argument("--out_csv", type=str, default="./results/sweep_results.csv")
    args = ap.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{api.default_entity}/{args.project}")

    fieldnames = [
        "run_name", "weight_quant_bits", "activation_quant_bits", "prune_sparsity",
        "quantized_acc", "weight_compression_ratio", "activation_compression_ratio",
        "compression_ratio", "model_size_mb", "fp32_model_size_mb",
    ]
    rows = []
    for run in runs:
        cfg, summ = run.config, run.summary
        if "quantized_acc" not in summ:
            continue
        rows.append({
            "run_name": run.name,
            "weight_quant_bits": cfg.get("weight_quant_bits"),
            "activation_quant_bits": cfg.get("activation_quant_bits"),
            "prune_sparsity": cfg.get("prune_sparsity", 0.0),
            "quantized_acc": summ.get("quantized_acc"),
            "weight_compression_ratio": summ.get("weight_compression_ratio"),
            "activation_compression_ratio": summ.get("activation_compression_ratio"),
            "compression_ratio": summ.get("compression_ratio"),
            "model_size_mb": summ.get("model_size_mb"),
            "fp32_model_size_mb": summ.get("fp32_model_size_mb"),
        })

    rows.sort(key=lambda r: (r["weight_quant_bits"] or 0, r["activation_quant_bits"] or 0, r["prune_sparsity"] or 0))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs to {args.out_csv}")


if __name__ == "__main__":
    main()
