"""Sanity checks for the hand-written quantization/pruning/packing primitives.
Run with: python -m tests.test_quant
"""
import numpy as np
import torch
import torch.nn as nn

from src.compress_utils import (estimate_activation_storage,
                                 estimate_weight_storage, pack_bits,
                                 unpack_bits)
from src.prune import magnitude_prune_
from src.quant import (ActivationQuantizer, apply_weight_quantization,
                       calibrate_activations, insert_activation_quantizers,
                       quantize_weight_per_channel)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    for bits in [2, 3, 4, 6, 8]:
        n = 1000
        codes = rng.integers(0, 2 ** bits, size=n)
        packed = pack_bits(codes, bits)
        expected_bytes = int(np.ceil(n * bits / 8))
        assert len(packed) == expected_bytes, (bits, len(packed), expected_bytes)
        recovered = unpack_bits(packed, bits, n)
        assert np.array_equal(codes, recovered), f"round-trip mismatch at bits={bits}"
    print("[ok] pack/unpack round-trip exact for bits in {2,3,4,6,8}")


def test_weight_quant_error_monotonic():
    torch.manual_seed(0)
    w = torch.randn(16, 8, 3, 3)
    errs = {}
    for bits in [2, 3, 4, 6, 8]:
        dequant, codes, scale = quantize_weight_per_channel(w, bits)
        err = (dequant - w).abs().mean().item()
        errs[bits] = err
        qmax = 2 ** (bits - 1) - 1
        assert codes.min() >= -qmax and codes.max() <= qmax
    bits_sorted = sorted(errs)
    for a, b in zip(bits_sorted, bits_sorted[1:]):
        assert errs[b] <= errs[a] + 1e-6, f"error should shrink with more bits: {errs}"
    print(f"[ok] weight quant error shrinks monotonically with bits: {errs}")


def test_activation_quantizer_calibration():
    torch.manual_seed(0)
    q = ActivationQuantizer(bits=8, ema_momentum=0.0)  # momentum 0 -> track last batch exactly
    q.mode = "calibrate"
    x = torch.rand(4, 3, 8, 8) * 6.0  # ReLU6-like range
    q(x)
    assert abs(q.running_min.item() - x.min().item()) < 1e-5
    assert abs(q.running_max.item() - x.max().item()) < 1e-5
    q.mode = "quantize"
    q.recording = True
    out = q(x)
    assert out.shape == x.shape
    assert q.last_codes is not None and q.last_numel == x.numel()
    assert (out - x).abs().max().item() < (x.max() - x.min()).item() / (2 ** 8 - 1) + 1e-4
    print("[ok] ActivationQuantizer calibrates and quantizes within expected error bound")


def test_apply_weight_quantization_and_storage():
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU6(), nn.Flatten(), nn.Linear(8 * 30 * 30, 10))
    first_conv, final_linear = model[0], model[3]
    info = apply_weight_quantization(model, bits=4, exempt_modules=(first_conv, final_linear))
    assert info[first_conv]["bits"] == 8  # floored despite bits=4
    assert info[final_linear]["bits"] == 8  # floored despite bits=4
    storage = estimate_weight_storage(info)
    assert storage["compression_ratio"] > 1.0
    print(f"[ok] weight quantization + storage accounting: ratio={storage['compression_ratio']:.2f}x, "
          f"compressed={storage['total_compressed_bytes']}B")


def test_pruning_reduces_nonzero_and_storage():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3))
    conv = model[0]
    masks = magnitude_prune_(model, sparsity=0.5)
    nz_frac = (conv.weight.data != 0).float().mean().item()
    assert 0.35 < nz_frac < 0.65, nz_frac
    info = apply_weight_quantization(model, bits=4)
    storage_pruned = estimate_weight_storage(info, masks=masks)
    storage_dense = estimate_weight_storage(info)
    assert storage_pruned["total_compressed_bytes"] < storage_dense["total_compressed_bytes"]
    print(f"[ok] pruning reduces storage: dense={storage_dense['total_compressed_bytes']}B "
          f"pruned={storage_pruned['total_compressed_bytes']}B")


def test_activation_storage_end_to_end():
    torch.manual_seed(0)
    from src.model import build_model
    model = build_model()
    model.eval()
    quantizers = insert_activation_quantizers(model, bits=4)

    class FakeLoader:
        def __iter__(self):
            for _ in range(3):
                yield torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)

    calibrate_activations(model, quantizers, FakeLoader(), device=torch.device("cpu"), num_batches=3)
    for q in quantizers:
        q.recording = True
    with torch.no_grad():
        model(torch.randn(2, 3, 32, 32))
    act_storage = estimate_activation_storage(quantizers)
    assert act_storage["compression_ratio"] > 1.0
    print(f"[ok] activation storage accounting: ratio={act_storage['compression_ratio']:.2f}x over "
          f"{len(quantizers)} ReLU6 sites")


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_weight_quant_error_monotonic()
    test_activation_quantizer_calibration()
    test_apply_weight_quantization_and_storage()
    test_pruning_reduces_nonzero_and_storage()
    test_activation_storage_end_to_end()
    print("\nAll quant/prune/compress_utils sanity checks passed.")
