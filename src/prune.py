"""Magnitude-based unstructured pruning — the 'own extension' beyond bit-width
quantization alone. Applied before weight quantization; pruned (exactly-zero)
weights are excluded from the packed bitstream and represented instead by a
compact 1-bit-per-weight mask (see compress_utils.py)."""
import torch
import torch.nn as nn


def magnitude_prune_(model, sparsity, exempt_modules=()):
    """In place: zeroes the smallest-magnitude `sparsity` fraction of weights in
    every Conv2d/Linear layer (except `exempt_modules`).

    Returns {module: bool_mask} marking surviving (nonzero) positions.
    """
    masks = {}
    if sparsity <= 0:
        return masks
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module not in exempt_modules:
            w = module.weight.data
            k = int(sparsity * w.numel())
            if k == 0:
                masks[module] = torch.ones_like(w, dtype=torch.bool)
                continue
            threshold = w.abs().flatten().kthvalue(k).values
            mask = w.abs() > threshold
            w.mul_(mask)
            masks[module] = mask
    return masks
