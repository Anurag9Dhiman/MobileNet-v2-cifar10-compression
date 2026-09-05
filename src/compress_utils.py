"""Storage / compression-ratio accounting for the quantization + pruning
pipeline. Weight and activation codes are genuinely bit-packed via numpy
bit-shifting (not divided by 8 in theory), so reported sizes are measured byte
counts, not theoretical ones."""
import numpy as np

SCALE_BYTES = 2  # per-channel weight scales stored as fp16


def pack_bits(codes, bits):
    """Pack an array of non-negative integers (each < 2**bits) into a tight
    bitstream, MSB-first per value."""
    codes = np.asarray(codes).astype(np.uint32).flatten()
    n = codes.shape[0]
    if n == 0:
        return b""
    bitplane = np.zeros((n, bits), dtype=np.uint8)
    for i in range(bits):
        bitplane[:, i] = (codes >> (bits - 1 - i)) & 1
    return np.packbits(bitplane.flatten()).tobytes()


def unpack_bits(data, bits, n):
    """Inverse of pack_bits: recovers the original n non-negative int codes."""
    bitstring = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bitstring = bitstring[: n * bits].reshape(n, bits)
    out = np.zeros(n, dtype=np.uint32)
    for i in range(bits):
        out |= bitstring[:, i].astype(np.uint32) << (bits - 1 - i)
    return out


def pack_mask(mask):
    return np.packbits(np.asarray(mask).astype(np.uint8).flatten()).tobytes()


def estimate_weight_storage(weight_info, masks=None):
    """weight_info: {module: {"codes": signed int ndarray, "scale": ndarray, "bits": int}}
    (from quant.apply_weight_quantization). masks: optional {module: bool ndarray}
    of surviving positions (from prune.magnitude_prune_).

    Returns byte-level breakdown + overall weight compression ratio vs fp32.
    """
    fp32_bytes = 0
    packed_bytes = 0
    scale_bytes = 0
    mask_bytes = 0

    for module, info in weight_info.items():
        codes, bits, scale = info["codes"], info["bits"], info["scale"]
        fp32_bytes += codes.size * 4
        scale_bytes += scale.size * SCALE_BYTES

        qmax = 2 ** (bits - 1) - 1
        unsigned_codes = (codes + qmax).astype(np.uint32)

        if masks is not None and module in masks:
            mask = masks[module].detach().cpu().numpy().astype(bool)
            mask_bytes += len(pack_mask(mask))
            surviving = unsigned_codes.flatten()[mask.flatten()]
            packed_bytes += len(pack_bits(surviving, bits))
        else:
            packed_bytes += len(pack_bits(unsigned_codes, bits))

    total_compressed = packed_bytes + scale_bytes + mask_bytes
    return {
        "fp32_bytes": fp32_bytes,
        "packed_weight_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "mask_bytes": mask_bytes,
        "total_compressed_bytes": total_compressed,
        "compression_ratio": fp32_bytes / max(1, total_compressed),
    }


def estimate_activation_storage(quantizers):
    """quantizers: list of quant.ActivationQuantizer, each with `.recording`
    enabled during exactly one representative forward pass over a fixed batch."""
    fp32_bytes = 0
    packed_bytes = 0
    for q in quantizers:
        if q.last_codes is None:
            continue
        fp32_bytes += q.last_numel * 4
        packed_bytes += len(pack_bits(q.last_codes, q.bits))
    return {
        "fp32_bytes": fp32_bytes,
        "packed_bytes": packed_bytes,
        "compression_ratio": fp32_bytes / max(1, packed_bytes),
    }


def model_size_mb(weight_storage):
    return weight_storage["total_compressed_bytes"] / (1024 * 1024)


def fp32_model_size_mb(weight_storage):
    return weight_storage["fp32_bytes"] / (1024 * 1024)
