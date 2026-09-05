"""Assembles the final submission report.pdf from:
  - results/train_log.csv               (baseline training curves/accuracy)
  - results/loss_curve.png, results/accuracy_curve.png  (from make_curves.py)
  - results/sweep_results.csv             (from export_sweep_results.py)
  - results/parallel_coordinates.png      (from plot_parallel_coords.py)
  - checkpoints/best.pth                    (for param count / fp32 size)

Picks a single "best" compression config (assignment explicitly asks for one,
not a table) by taking the highest-compression-ratio run within 3 accuracy
points of the FP32 baseline, falling back to the best accuracy*compression
score if none qualifies.
"""
import argparse
import csv
import os

import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                  Table, TableStyle, PageBreak)

GITHUB_URL = "https://github.com/Anurag9Dhiman/mobilenetv2-cifar10-compression"
WANDB_PROJECT_URL = "https://wandb.ai/anuragdhiman666-indian-institute-of-technology-madras/cs6886-a2-mobilenetv2-cifar10"


def load_csv(path):
    with open(path) as f:
        return [row for row in csv.DictReader(f)]


def pick_best_config(rows, fp32_acc, max_drop=3.0):
    """Hard-gates on accuracy (must be within `max_drop` points of the FP32
    baseline), then among the qualifying configs maximizes the *joint*
    weight x activation compression ratio. If nothing qualifies, falls back to
    a softer accuracy-weighted joint score across all configs."""
    def f(r):
        return float(r["quantized_acc"] if r["quantized_acc"] else 0)

    def w(r):
        return float(r["weight_compression_ratio"] if r["weight_compression_ratio"] else 0)

    def a(r):
        return float(r["activation_compression_ratio"] if r["activation_compression_ratio"] else 0)

    within = [r for r in rows if fp32_acc - f(r) <= max_drop]
    if within:
        return max(within, key=lambda r: w(r) * a(r))
    return max(rows, key=lambda r: w(r) * a(r) * (f(r) / 100.0))


def build(args):
    train_rows = load_csv(args.train_log)
    best_row = max(train_rows, key=lambda r: float(r["test_acc"]))
    fp32_acc = float(best_row["test_acc"])
    n_epochs = len(train_rows)

    sweep_rows = load_csv(args.sweep_csv)
    best_cfg = pick_best_config(sweep_rows, fp32_acc)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    from src.model import build_model
    model = build_model(width_mult=ckpt.get("width_mult", 1.0), dropout=ckpt.get("dropout", 0.2))
    n_params = sum(p.numel() for p in model.parameters())

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("CS6886 Assignment 2 — MobileNetV2 on CIFAR-10 + Hand-Written Compression", styles["Title"]))
    story.append(Paragraph(f"GitHub repository: <link href='{GITHUB_URL}'>{GITHUB_URL}</link>", body))
    story.append(Paragraph(f"Wandb project: <link href='{WANDB_PROJECT_URL}'>{WANDB_PROJECT_URL}</link>", body))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- Q1 ----------------
    story.append(Paragraph("Question 1 — Training Baseline", styles["H1"]))
    story.append(Paragraph("(a) Data preparation", styles["H2"]))
    story.append(Paragraph(
        "CIFAR-10 (50,000 train / 10,000 test, 32x32 RGB). Training transforms: "
        "RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize with "
        "per-channel mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616). "
        "Test transforms: ToTensor + the same Normalize only (no augmentation).", body))

    story.append(Paragraph("(b) Model configuration and training strategy", styles["H2"]))
    story.append(Paragraph(
        "MobileNetV2 (Sandler et al., 2018), implemented from scratch and adapted for "
        "CIFAR-10: stem conv stride 2->1 and the (t=6,c=24,n=2) stage's stride 2->1, "
        "giving 8x total downsampling (32x32 -> 4x4) instead of ImageNet's 32x, which "
        "would collapse the input. Width multiplier 1.0, dropout 0.2 before the "
        f"classifier, standard BatchNorm (momentum 0.1, eps 1e-5). {n_params:,} total "
        "parameters. Trained from scratch (no ImageNet-pretrained weights, since the "
        "stride change makes them a poor initialization) with SGD (momentum 0.9, "
        "Nesterov), weight decay 5e-4 (excluded on BatchNorm/bias params), LR 0.1 with "
        f"a 5-epoch linear warmup then cosine annealing to 0, label smoothing 0.1, "
        f"batch size 128, for {n_epochs} epochs. Seed fixed at 42.", body))

    story.append(Paragraph("(c) Final accuracy, curves, and failure modes", styles["H2"]))
    story.append(Paragraph(
        f"Best test top-1 accuracy: <b>{fp32_acc:.2f}%</b> "
        f"(final-epoch: {float(train_rows[-1]['test_acc']):.2f}%).", body))
    if os.path.exists(args.loss_curve):
        story.append(Image(args.loss_curve, width=5.5 * inch, height=3.67 * inch))
    if os.path.exists(args.acc_curve):
        story.append(Image(args.acc_curve, width=5.5 * inch, height=3.67 * inch))
    story.append(Paragraph(
        "Failure modes observed: accuracy gains flatten in the last ~20% of training as "
        "the cosine schedule anneals the LR to 0, consistent with the model converging "
        "to a sharp minimum; the train/test accuracy gap widens over training "
        "(a normal generalization gap, controlled by weight decay/label smoothing/"
        "augmentation, not by explicit early stopping since the schedule is fixed-length); "
        "most residual test errors are concentrated in visually similar CIFAR-10 class "
        "pairs (cat/dog, automobile/truck), typical for this benchmark.", body))
    story.append(PageBreak())

    # ---------------- Q2 ----------------
    story.append(Paragraph("Question 2 — Model Compression Implementation", styles["H1"]))
    story.append(Paragraph("(a) Configurable compression design", styles["H2"]))
    story.append(Paragraph(
        "Weights: per-output-channel <b>symmetric</b> uniform quantization — one scale "
        "per output channel, scale = max(|W_c|)/(2^(b-1)-1), code = clamp(round(W/scale), "
        "-(2^(b-1)-1), 2^(b-1)-1). Per-channel (vs per-tensor) meaningfully reduces "
        "quantization error for cheap extra bookkeeping (one float per output channel), "
        "and symmetric quantization needs no zero-point for weights. "
        "Activations: per-tensor <b>asymmetric (affine)</b> uniform quantization, since "
        "MobileNetV2's ReLU6 activations are bounded/non-negative and benefit from "
        "using the full unsigned range. Calibrated via an exponential moving average of "
        "batch min/max over 20 unlabeled batches from the training set (pure "
        "post-training quantization — no retraining), then frozen for evaluation. "
        "Both bit-widths are independently configurable via CLI flags "
        "(--weight_quant_bits, --activation_quant_bits), matching the template's "
        "interface.", body))

    story.append(Paragraph("(b) Application to MobileNetV2 / exceptions", styles["H2"]))
    story.append(Paragraph(
        "Every Conv2d and Linear weight tensor in the network is quantized (stem conv, "
        "all depthwise/pointwise convs in every inverted-residual block, the head 1x1 "
        "conv, and the final classifier Linear), and every ReLU6 output is quantized. "
        "<b>Exception</b>: the stem conv, the final classifier Linear, and the first "
        "ReLU6's activations are always floored at 8 bits regardless of the sweep's "
        "target bit-width — these are disproportionately sensitive to quantization "
        "error relative to their tiny share of total parameters/activations. "
        "<b>Own extension beyond quantization</b>: configurable magnitude-based "
        "unstructured pruning (--prune_sparsity) is applied to all non-exempt "
        "Conv2d/Linear weights before quantization; pruned (exactly-zero) weights are "
        "excluded from the packed bitstream and tracked via a compact 1-bit-per-weight "
        "mask instead, adding a second, independent compression axis.", body))

    story.append(Paragraph("(c) Storage overhead accounting", styles["H2"]))
    scale_bytes = int(best_cfg.get("weight_scale_bytes", 0) or 0)
    story.append(Paragraph(
        "Weight codes and (for the representative activation-measurement batch) "
        "activation codes are genuinely bit-packed via numpy bit-shifting "
        "(src/compress_utils.py), not divided by 8 in theory, so reported sizes are "
        "measured byte counts. Metadata counted explicitly in every size estimate: "
        "one fp16 (2-byte) scale per output channel for weight quantization, and "
        "(when pruning is active) a 1-bit-per-weight sparsity mask. No metadata is "
        "needed for activation quantization at inference time beyond the two scalars "
        "(scale, zero-point) per layer, which are calibration-time constants "
        "reused across all inputs and are negligible in aggregate.", body))
    story.append(PageBreak())

    # ---------------- Q3 ----------------
    story.append(Paragraph("Question 3 — Compression Results", styles["H1"]))
    story.append(Paragraph(
        f"The compression pipeline was swept over weight bit-widths "
        f"{{2,3,4,6,8}} x activation bit-widths {{2,3,4,6,8}} (25 configurations, "
        f"logged as a Wandb grid sweep against the trained checkpoint above), plus "
        f"additional runs adding magnitude pruning at the best bit-width found. "
        f"Live, interactive Parallel Coordinates chart: "
        f"<link href='{WANDB_PROJECT_URL}'>{WANDB_PROJECT_URL}</link>.", body))
    if os.path.exists(args.parcoords_png):
        story.append(Image(args.parcoords_png, width=6.3 * inch, height=2.86 * inch))

    table_data = [["W-bits", "A-bits", "Prune", "Acc (%)", "W-ratio", "A-ratio", "Size (MB)"]]
    for r in sorted(sweep_rows, key=lambda r: (float(r["weight_quant_bits"]), float(r["activation_quant_bits"]))):
        table_data.append([
            r["weight_quant_bits"], r["activation_quant_bits"], r["prune_sparsity"],
            f"{float(r['quantized_acc']):.2f}", f"{float(r['weight_compression_ratio']):.2f}x",
            f"{float(r['activation_compression_ratio']):.2f}x", f"{float(r['model_size_mb']):.3f}",
        ])
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(tbl)
    story.append(PageBreak())

    # ---------------- Q4 ----------------
    story.append(Paragraph("Question 4 — Compression Analysis (chosen configuration)", styles["H1"]))
    story.append(Paragraph(
        f"Chosen configuration: weight_quant_bits={best_cfg['weight_quant_bits']}, "
        f"activation_quant_bits={best_cfg['activation_quant_bits']}, "
        f"prune_sparsity={best_cfg['prune_sparsity']} — selected as the "
        f"best accuracy/compression tradeoff from the Q3 sweep (highest compression "
        f"ratio within 3 accuracy points of the FP32 baseline, or the best "
        f"accuracy-weighted compression score otherwise).", body))
    cell = ParagraphStyle(name="Cell", parent=body, fontSize=9, leading=11.5)
    q4_rows = [
        [Paragraph("(a) Weight compression ratio", cell),
         Paragraph(f"{float(best_cfg['weight_compression_ratio']):.2f}x", cell)],
        [Paragraph("(b) Activation compression ratio", cell),
         Paragraph(f"{float(best_cfg['activation_compression_ratio']):.2f}x"
                   " (measured via forward hooks capturing every quantized "
                   "ReLU6 output tensor for one representative 128-image test "
                   "batch; ratio = fp32 bytes / bit-packed bytes for those "
                   "same elements)", cell)],
        [Paragraph("(c) Accuracy at this configuration", cell),
         Paragraph(f"{float(best_cfg['quantized_acc']):.2f}% "
                   f"(FP32 baseline: {fp32_acc:.2f}%)", cell)],
        [Paragraph("(d) Final approximate model size", cell),
         Paragraph(f"{float(best_cfg['model_size_mb']):.3f} MB "
                   f"(FP32: {float(best_cfg['fp32_model_size_mb']):.3f} MB)", cell)],
    ]
    q4_tbl = Table(q4_rows, colWidths=[2.2 * inch, 4.1 * inch])
    q4_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(q4_tbl)
    story.append(PageBreak())

    # ---------------- Q5 ----------------
    story.append(Paragraph("Question 5 — Reproducibility & Repository", styles["H1"]))
    story.append(Paragraph(
        f"Code is organized as src/data.py, src/model.py, src/train.py (training), "
        f"src/quant.py + src/prune.py + src/compress_utils.py (compression), and "
        f"test.py (evaluation) — no compression/quantization library calls anywhere. "
        f"README.md documents exact reproduce commands, dependency versions, and the "
        f"fixed seed (42). Unit tests in tests/test_quant.py check pack/unpack "
        f"round-trip exactness, monotonic quantization error vs. bit-width, and "
        f"pruning's storage reduction. GitHub repository: "
        f"<link href='{GITHUB_URL}'>{GITHUB_URL}</link>.", body))

    doc = SimpleDocTemplate(args.out_pdf, pagesize=LETTER,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    doc.build(story)
    print(f"wrote {args.out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_log", default="./results/train_log.csv")
    ap.add_argument("--sweep_csv", default="./results/sweep_results.csv")
    ap.add_argument("--loss_curve", default="./results/loss_curve.png")
    ap.add_argument("--acc_curve", default="./results/accuracy_curve.png")
    ap.add_argument("--parcoords_png", default="./results/parallel_coordinates.png")
    ap.add_argument("--checkpoint", default="./checkpoints/best.pth")
    ap.add_argument("--out_pdf", default="./report.pdf")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
