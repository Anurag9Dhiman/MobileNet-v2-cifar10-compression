"""Baseline FP32 training of MobileNetV2 on CIFAR-10.

Example:
    python -m src.train --epochs 80 --batch_size 128 --lr 0.1
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import build_model
from src.utils import AverageMeter, CSVLogger, get_device, seed_everything, top1_accuracy


def split_decay_params(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_lr_lambda(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(top1_accuracy(logits, y), x.size(0))
    return loss_meter.avg, acc_meter.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--width_mult", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./checkpoints")
    ap.add_argument("--log_csv", type=str, default="./results/train_log.csv")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", type=str, default="cs6886-a2-mobilenetv2-cifar10")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )

    model = build_model(num_classes=10, width_mult=args.width_mult, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params} (~{n_params * 4 / 1024 / 1024:.2f} MB fp32)")

    decay, no_decay = split_decay_params(model)
    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, momentum=0.9, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_epochs, args.epochs)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=vars(args))
        except Exception as e:
            print(f"[warn] wandb init failed ({e}); continuing without wandb")
            use_wandb = False

    logger = CSVLogger(args.log_csv, fieldnames=[
        "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "epoch_time_sec",
    ])

    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(top1_accuracy(logits, y), x.size(0))

        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "train_loss": loss_meter.avg, "train_acc": acc_meter.avg,
            "test_loss": test_loss, "test_acc": test_acc, "lr": cur_lr,
            "epoch_time_sec": epoch_time,
        }
        logger.log(row)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {loss_meter.avg:.4f} "
              f"train_acc {acc_meter.avg:.2f} | test_loss {test_loss:.4f} "
              f"test_acc {test_acc:.2f} | lr {cur_lr:.5f} | {epoch_time:.1f}s")

        if use_wandb:
            wandb.log(row)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "test_acc": test_acc,
                "width_mult": args.width_mult, "dropout": args.dropout,
            }, best_path)

    print(f"best test_acc: {best_acc:.2f} | checkpoint: {best_path}")
    if use_wandb:
        wandb.summary["best_test_acc"] = best_acc
        wandb.finish()


if __name__ == "__main__":
    main()
