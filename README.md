# MobileNetV2 on CIFAR-10 + Hand-Written Compression

CS6886 (System Engineering for Deep Learning) — Assignment 2. Trains MobileNetV2
(adapted for CIFAR-10's 32x32 inputs) from scratch, then applies a **hand-written**
post-training compression pipeline — per-channel weight quantization, calibrated
per-tensor activation quantization, and magnitude pruning — with no compression
library/API calls anywhere in `src/`.

## Environment

- Python 3.13.7, macOS (Apple Silicon, MPS backend; no CUDA used)
- Exact dependency versions in [requirements.txt](requirements.txt):
  `torch==2.8.0`, `torchvision==0.23.0`, `numpy==2.4.4`, `matplotlib==3.10.9`,
  `wandb==0.24.0`, `reportlab==4.4.4`
- `pip install -r requirements.txt`

If you hit an SSL certificate error downloading CIFAR-10 on macOS's python.org
build, point Python at the `certifi` bundle before running anything:
```bash
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```

## Repository layout

```
src/
  data.py             CIFAR-10 loaders + transforms
  model.py            MobileNetV2, adapted for CIFAR-10 (from scratch)
  train.py            baseline FP32 training loop
  quant.py            per-channel weight quant + calibrated activation quant
  prune.py            magnitude pruning (the "own extension")
  compress_utils.py   real bit-packing + storage/compression-ratio accounting
  utils.py            seeding, meters, CSV logger
test.py               PTQ (+pruning) evaluation CLI
sweep.yaml            wandb grid sweep over bit-widths
scripts/
  make_curves.py      loss/accuracy curve figures from the training CSV log
  build_report.py     assembles report.pdf
tests/test_quant.py   unit checks for quantization/pruning/packing correctness
checkpoints/best.pth  best FP32 checkpoint (committed)
results/              training log CSV, figures, sweep exports
```

## Reproduce: baseline training

Seed is fixed to `42` (torch/numpy/random) for reproducibility.

```bash
python -m src.train \
  --epochs 120 --batch_size 128 --lr 0.1 --wd 5e-4 \
  --warmup_epochs 5 --label_smoothing 0.1 --seed 42 \
  --wandb_mode online   # or "disabled" to skip wandb entirely
```

Saves the best (by test top-1) checkpoint to `checkpoints/best.pth` and a
per-epoch CSV to `results/train_log.csv`. Then:

```bash
python scripts/make_curves.py   # writes results/loss_curve.png, results/accuracy_curve.png
```

## Reproduce: compression evaluation (single config)

Matches the template's interface, extended with a pruning flag:

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
python test.py --weight_quant_bits 4 --activation_quant_bits 4 --prune_sparsity 0.3
```

Each invocation: loads `checkpoints/best.pth`, applies magnitude pruning (if
`--prune_sparsity > 0`) then per-channel symmetric weight quantization, calibrates
per-tensor activation quantizers on 20 unlabeled batches from the training set,
evaluates CIFAR-10 test accuracy, and prints/logs weight/activation/model
compression ratios and the final approximate model size (MB).

## Reproduce: bit-width sweep (Wandb Parallel Coordinates)

```bash
wandb login                 # one-time; needs a wandb.ai API key
wandb sweep sweep.yaml       # prints a sweep ID
wandb agent <sweep_id>        # runs the grid (25 configs: 5 weight bits x 5 activation bits)
```

View the Parallel Coordinates chart under the sweep's page in the Wandb project
`cs6886-a2-mobilenetv2-cifar10`.

## Design choices (summary — full writeup in the report PDF)

- **Model**: MobileNetV2, stem conv stride 2->1 and the `(t=6,c=24,n=2)` stage's
  stride 2->1 (8x total downsample instead of 32x, since ImageNet's 32x would
  collapse a 32x32 input). Width multiplier 1.0, dropout 0.2, trained from scratch.
- **Weights**: per-output-channel symmetric uniform quantization (own code,
  `src/quant.py::quantize_weight_per_channel`).
- **Activations**: per-tensor asymmetric (affine) uniform quantization, fit via
  an EMA calibration pass over 20 unlabeled training batches — no retraining
  (`src/quant.py::ActivationQuantizer`).
- **Exceptions**: the stem conv, the final classifier `Linear`, and the first
  ReLU6's activations are floored at 8 bits regardless of the sweep's target
  bit-width (most error-sensitive, smallest share of params/activations).
- **Own extension**: configurable magnitude-based unstructured pruning
  (`src/prune.py`) applied before quantization; pruned zeros are excluded from
  the packed bitstream and tracked via a 1-bit-per-weight mask instead.
- **Storage accounting**: weight/activation integer codes are genuinely
  bit-packed via numpy bit-shifting (`src/compress_utils.py`), not divided by 8
  in theory — reported sizes are measured byte counts.

## Results

_Filled in after the baseline training run and the bit-width sweep complete —
see `report.pdf` for the full writeup (Q1-Q5) with figures and tables._

- FP32 baseline test top-1 accuracy: TBD
- Chosen best compression config: TBD
- Final approximate model size after compression: TBD

## Unit tests

```bash
python -m tests.test_quant
```
Checks: pack/unpack bit round-trip exactness (bits in {2,3,4,6,8}), weight
quantization error shrinking monotonically with bit-width, activation
calibration correctness, and that pruning strictly reduces packed storage size.
