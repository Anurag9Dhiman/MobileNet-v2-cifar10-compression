"""Post-training quantization (+ optional pruning) evaluation CLI.

Loads a trained FP32 checkpoint, applies the hand-written compression pipeline
(magnitude pruning -> per-channel weight quantization -> calibrated per-tensor
activation quantization), evaluates CIFAR-10 test accuracy, and reports/logs
compression ratios and the resulting approximate model size.

Example (matches the template's interface):
    python test.py --weight_quant_bits 8 --activation_quant_bits 8
    python test.py --weight_quant_bits 4 --activation_quant_bits 4 --prune_sparsity 0.3
"""
import argparse
import time

import torch

from src.compress_utils import (estimate_activation_storage,
                                 estimate_weight_storage, fp32_model_size_mb,
                                 model_size_mb)
from src.data import get_calibration_loader, get_test_loader
from src.model import build_model
from src.prune import magnitude_prune_
from src.quant import (apply_weight_quantization, calibrate_activations,
                       insert_activation_quantizers)
from src.utils import get_device, seed_everything, top1_accuracy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight_quant_bits", type=int, default=8)
    ap.add_argument("--activation_quant_bits", type=int, default=8)
    ap.add_argument("--prune_sparsity", type=float, default=0.0)
    ap.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--calib_batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", type=str, default="cs6886-a2-mobilenetv2-cifar10")
    return ap.parse_args()


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        acc = top1_accuracy(logits, y)
        correct += acc * x.size(0) / 100.0
        total += x.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        num_classes=10,
        width_mult=ckpt.get("width_mult", 1.0),
        dropout=ckpt.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    fp32_acc_at_train = ckpt.get("test_acc")

    first_conv = model.features[0][0]
    final_linear = model.classifier[1]
    exempt = (first_conv, final_linear)

    # 1) magnitude pruning (own extension), applied before quantization
    masks = magnitude_prune_(model, args.prune_sparsity, exempt_modules=exempt)

    # 2) per-channel symmetric weight quantization
    weight_info = apply_weight_quantization(model, args.weight_quant_bits, exempt_modules=exempt)

    # 3) calibrated per-tensor asymmetric activation quantization
    quantizers = insert_activation_quantizers(model, args.activation_quant_bits, exempt_first=True)
    calib_loader = get_calibration_loader(data_dir=args.data_dir, seed=args.seed)
    calibrate_activations(model, quantizers, calib_loader, device, num_batches=args.calib_batches)

    test_loader = get_test_loader(data_dir=args.data_dir, batch_size=args.batch_size)

    # 4) capture activation codes from one representative batch for storage accounting
    for q in quantizers:
        q.recording = True
    with torch.no_grad():
        sample_x, _ = next(iter(test_loader))
        model(sample_x.to(device))
    for q in quantizers:
        q.recording = False

    t0 = time.time()
    quantized_acc = evaluate_accuracy(model, test_loader, device)
    eval_time = time.time() - t0

    weight_storage = estimate_weight_storage(weight_info, masks=masks if args.prune_sparsity > 0 else None)
    act_storage = estimate_activation_storage(quantizers)
    size_mb = model_size_mb(weight_storage)
    fp32_mb = fp32_model_size_mb(weight_storage)

    result = {
        "weight_quant_bits": args.weight_quant_bits,
        "activation_quant_bits": args.activation_quant_bits,
        "prune_sparsity": args.prune_sparsity,
        "quantized_acc": quantized_acc,
        "weight_compression_ratio": weight_storage["compression_ratio"],
        "activation_compression_ratio": act_storage["compression_ratio"],
        "compression_ratio": weight_storage["compression_ratio"],
        "model_size_mb": size_mb,
        "fp32_model_size_mb": fp32_mb,
        "weight_scale_bytes": weight_storage["scale_bytes"],
        "weight_mask_bytes": weight_storage["mask_bytes"],
        "weight_packed_bytes": weight_storage["packed_weight_bytes"],
    }

    print("=" * 70)
    for k, v in result.items():
        print(f"{k:>28s}: {v}")
    if fp32_acc_at_train is not None:
        print(f"{'fp32_checkpoint_test_acc':>28s}: {fp32_acc_at_train}")
    print(f"{'eval_time_sec':>28s}: {eval_time:.1f}")
    print("=" * 70)

    if args.wandb_mode != "disabled":
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config={
                "weight_quant_bits": args.weight_quant_bits,
                "activation_quant_bits": args.activation_quant_bits,
                "prune_sparsity": args.prune_sparsity,
            }, reinit=True)
            wandb.log(result)
            wandb.finish()
        except Exception as e:
            print(f"[warn] wandb logging failed ({e}); continuing without it")


if __name__ == "__main__":
    main()
