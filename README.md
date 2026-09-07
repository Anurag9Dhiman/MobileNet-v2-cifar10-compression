# MobileNetV2 on CIFAR-10 + Hand-Written Compression

**Repository**: https://github.com/Anurag9Dhiman/MobileNet-v2-cifar10-compression  
**Course**: CS6886 (System Engineering for Deep Learning) — Assignment 2

Trains MobileNetV2 (adapted for CIFAR-10's 32x32 inputs) from scratch, then applies a **hand-written** post-training compression pipeline — per-channel weight quantization, calibrated per-tensor activation quantization, and magnitude pruning — with no compression library/API calls anywhere in `src/`.

---

## Question 5: Reproducibility & Repository Overview

### (a) Codebase Architecture & Separation of Concerns

The codebase is strictly modularized with clear separation across training, evaluation, and compression:

- **Training**:
  - `src/data.py`: CIFAR-10 dataset downloading, data augmentations (RandomCrop, RandomHorizontalFlip, normalization), DataLoader instantiation, and unaugmented calibration split.
  - `src/model.py`: From-scratch MobileNetV2 architecture adapted for 32x32 inputs (8x total spatial downsampling to prevent collapse).
  - `src/train.py`: Full FP32 baseline training pipeline (SGD with momentum, linear warmup + cosine annealing scheduler, weight decay excluded on 1D/BatchNorm params, checkpointing).
- **Evaluation**:
  - `test.py`: Standalone CLI to load checkpoints, run PTQ calibration, evaluate quantized test top-1 accuracy on CIFAR-10, measure execution time, and bit-pack activations for accurate storage accounting.
- **Compression (Zero External Libraries)**:
  - `src/quant.py`: Hand-written per-output-channel symmetric weight quantization and calibrated per-tensor asymmetric activation quantization with EMA min/max calibration.
  - `src/prune.py`: Configurable magnitude-based unstructured weight pruning applied before quantization.
  - `src/compress_utils.py`: Genuine numpy bit-level packing (`pack_bits`, `unpack_bits`), 1-bit pruning mask packing, and exact storage/overhead accounting (including fp16 scales and masks).
- **Testing & Verification**:
  - `tests/test_quant.py`: Unit tests for bit-packing round-trip exactness ({2,3,4,6,8} bits), monotonic quantization error decay, and pruning storage savings.
- **Utilities & Scripts**:
  - `src/utils.py`: Deterministic seed configuration, running average meters, top-1 accuracy, CSV logger.
  - `scripts/`: Scripts for plotting loss/accuracy curves, exporting Wandb sweep results, plotting parallel coordinates, and building `report.pdf`.

```
src/
  data.py             CIFAR-10 loaders + transforms (Training)
  model.py            MobileNetV2, adapted for CIFAR-10 (Training)
  train.py            Baseline FP32 training loop (Training)
  quant.py            Per-channel weight quant + calibrated activation quant (Compression)
  prune.py            Magnitude pruning extension (Compression)
  compress_utils.py   Real bit-packing + storage accounting (Compression)
  utils.py            Seeding, meters, CSV logger
test.py               PTQ (+pruning) evaluation CLI (Evaluation)
sweep.yaml            Wandb grid sweep over bit-widths
scripts/              Plotting curves, export sweeps, and building report.pdf
tests/test_quant.py   Unit tests for quantization, pruning, and bit-packing
checkpoints/best.pth  Trained FP32 baseline checkpoint
results/              Training CSV logs, figures, sweep CSV export
```

### (b) Environment, Dependencies & Seed Configuration

- **Environment**: Python 3.13.7, macOS (Apple Silicon, MPS backend; also compatible with CUDA and CPU).
- **Dependencies**: Pinned in `requirements.txt`:
  - `torch==2.8.0`
  - `torchvision==0.23.0`
  - `numpy==2.4.4`
  - `matplotlib==3.10.9`
  - `wandb==0.24.0`
  - `reportlab==4.4.4`
- **Installation**:
  ```bash
  pip install -r requirements.txt
  ```
- **Seed Configuration**: Globally fixed seed `42` configured via `src/utils.py::seed_everything(seed=42)` across Python's `random`, `numpy`, `torch.manual_seed`, `torch.mps.manual_seed`, and PyTorch DataLoader `torch.Generator` instances for exact determinism.

If you encounter an SSL certificate error downloading CIFAR-10 on macOS python.org builds:
```bash
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```

## Reproduce: baseline training

Seed is fixed to `42` (torch/numpy/random) for reproducibility.

```bash
python -m src.train \
  --epochs 80 --batch_size 128 --lr 0.1 --wd 5e-4 \
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
`cs6886-a2-mobilenetv2-cifar10`. Optional follow-up showing the pruning extension's
effect at the sweep's best bit-width:

```bash
python test.py --weight_quant_bits 4 --activation_quant_bits 4 --prune_sparsity 0.3
python test.py --weight_quant_bits 4 --activation_quant_bits 4 --prune_sparsity 0.5
```

Then export results and build the final report:

```bash
python scripts/export_sweep_results.py    # -> results/sweep_results.csv
python scripts/plot_parallel_coords.py     # -> results/parallel_coordinates.png
PYTHONPATH=. python scripts/build_report.py  # -> report.pdf
```

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

Full writeup (Q1-Q5) with figures and tables: [report.pdf](report.pdf).

- FP32 baseline test top-1 accuracy: **94.07%** (80 epochs, seed 42)
- Chosen best compression config: **weight_quant_bits=4, activation_quant_bits=4,
  prune_sparsity=0** — 92.42% accuracy (-1.65pp vs FP32), 7.71x weight compression,
  7.83x activation compression
- Final approximate model size after compression: **1.089 MB** (FP32: 8.402 MB)
- Full 25-point bit-width sweep + pruning follow-up: `results/sweep_results.csv`,
  `results/parallel_coordinates.png`; live chart at the Wandb project above.

## Unit tests

```bash
python -m tests.test_quant
```
Checks: pack/unpack bit round-trip exactness (bits in {2,3,4,6,8}), weight
quantization error shrinking monotonically with bit-width, activation
calibration correctness, and that pruning strictly reduces packed storage size.
