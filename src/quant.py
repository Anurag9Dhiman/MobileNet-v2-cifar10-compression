"""Hand-written post-training quantization (PTQ) primitives.

Two schemes, both implemented from scratch (no torch.quantization / bitsandbytes /
any compression library calls):
  - Weights:     per-output-channel SYMMETRIC uniform quantization.
  - Activations: per-tensor ASYMMETRIC (affine) uniform quantization, fitted via
                 an EMA-calibration pass over unlabeled batches (no retraining).
"""
import torch
import torch.nn as nn


def quantize_weight_per_channel(weight, bits):
    """Per-output-channel symmetric uniform quantization of a Conv2d/Linear
    weight tensor (dim 0 = output channels).

    Returns (dequantized_weight, int_codes in [-qmax, qmax], per-channel scale).
    """
    qmax = 2 ** (bits - 1) - 1
    out_c = weight.shape[0]
    flat = weight.reshape(out_c, -1)
    amax = flat.abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / qmax
    scale_b = scale.view(out_c, *([1] * (weight.dim() - 1)))
    codes = torch.clamp(torch.round(weight / scale_b), -qmax, qmax)
    dequant = codes * scale_b
    return dequant, codes, scale


def apply_weight_quantization(model, bits, exempt_modules=()):
    """Overwrites each Conv2d/Linear's weight in place with its fake-quantized
    (quantize-then-dequantize) value, for evaluation. `exempt_modules` (e.g. the
    stem conv / final classifier) are floored at 8 bits regardless of `bits`.

    Returns {module: {"codes": np.ndarray, "scale": np.ndarray, "bits": int}}
    for downstream storage-size accounting (see compress_utils.py).
    """
    info = {}
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            b = max(bits, 8) if module in exempt_modules else bits
            dequant, codes, scale = quantize_weight_per_channel(module.weight.data, b)
            module.weight.data.copy_(dequant)
            info[module] = {
                "codes": codes.detach().cpu().numpy().astype(int),
                "scale": scale.detach().cpu().numpy(),
                "bits": b,
            }
    return info


class ActivationQuantizer(nn.Module):
    """Per-tensor asymmetric quantizer for (non-negative, ReLU6-bounded)
    activations. Two-phase PTQ usage:
      1. mode="calibrate": forward passes over unlabeled batches update an EMA
         of the batch min/max (activations pass through unchanged).
      2. mode="quantize": the frozen min/max define an affine [0, 2^bits-1]
         quantizer applied on every forward call.
    """

    def __init__(self, bits=8, ema_momentum=0.9):
        super().__init__()
        self.bits = bits
        self.ema_momentum = ema_momentum
        self.mode = "off"  # "off" | "calibrate" | "quantize"
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self._initialized = False
        # Set True for exactly one representative forward pass to capture
        # integer codes for activation storage accounting (see compress_utils.py).
        self.recording = False
        self.last_codes = None
        self.last_numel = 0

    def forward(self, x):
        if self.mode == "off":
            return x
        if self.mode == "calibrate":
            with torch.no_grad():
                batch_min, batch_max = x.min(), x.max()
                if not self._initialized:
                    self.running_min.copy_(batch_min)
                    self.running_max.copy_(batch_max)
                    self._initialized = True
                else:
                    m = self.ema_momentum
                    self.running_min.copy_(m * self.running_min + (1 - m) * batch_min)
                    self.running_max.copy_(m * self.running_max + (1 - m) * batch_max)
            return x

        # mode == "quantize"
        qmax = 2 ** self.bits - 1
        rng = (self.running_max - self.running_min).clamp(min=1e-8)
        scale = rng / qmax
        zero_point = torch.clamp(torch.round(-self.running_min / scale), 0, qmax)
        codes = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)
        if self.recording:
            self.last_codes = codes.detach().cpu().numpy().astype(int)
            self.last_numel = x.numel()
        return (codes - zero_point) * scale


def insert_activation_quantizers(model, bits, exempt_first=True):
    """Attaches an ActivationQuantizer (via forward hook) after every ReLU6 in
    the model. The first ReLU6 encountered (the stem's) is floored at 8 bits,
    mirroring the weight-side first-layer exception."""
    quantizers = []
    idx = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU6):
            q_bits = max(bits, 8) if (exempt_first and idx == 0) else bits
            q = ActivationQuantizer(bits=q_bits)
            module.register_forward_hook(lambda m, inp, out, q=q: q(out))
            quantizers.append(q)
            idx += 1
    return quantizers


def set_quantizer_mode(quantizers, mode):
    for q in quantizers:
        q.mode = mode


@torch.no_grad()
def calibrate_activations(model, quantizers, calib_loader, device, num_batches=20):
    """Runs `num_batches` unlabeled forward passes to fit activation min/max
    EMA statistics, then freezes the quantizers into "quantize" mode."""
    set_quantizer_mode(quantizers, "calibrate")
    model.eval()
    for i, (x, _) in enumerate(calib_loader):
        if i >= num_batches:
            break
        model(x.to(device))
    set_quantizer_mode(quantizers, "quantize")
