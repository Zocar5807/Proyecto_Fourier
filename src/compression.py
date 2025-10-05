"""Coefficient pruning strategies for 2D FFT compression.

Includes global top-|F| pruning and adaptive radial pruning, utilities to
reconstruct from masks, estimate retained energy, and approximate compressed size.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_alpha(alpha: float) -> float:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    return float(alpha)


def _keep_topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    """Return boolean mask keeping the top-k absolute values along the flattened array.

    If k == 0, returns all False; if k >= size, returns all True.
    """
    flat = values.ravel()
    size = flat.size
    if k <= 0:
        return np.zeros(values.shape, dtype=bool)
    if k >= size:
        return np.ones(values.shape, dtype=bool)
    # Partition for kth largest (np.argpartition gives indices of kth position)
    idx = np.argpartition(flat, size - k)[size - k :]
    threshold = flat[idx].min()
    return values >= threshold


def global_prune(F: np.ndarray, alpha: float) -> Tuple[np.ndarray, int]:
    """Global top-|F| pruning.

    Parameters
    ----------
    F : np.ndarray
        Complex 2D spectrum (fftshifted or not – mask shape only depends on F.shape).
    alpha : float
        Fraction of coefficients to keep (0, 1].

    Returns
    -------
    mask : np.ndarray (bool)
        Mask of coefficients to keep.
    k_kept : int
        Number of kept coefficients (exactly floor(alpha * N)).
    """
    _validate_alpha(alpha)
    magnitudes = np.abs(F)
    total = magnitudes.size
    k_kept = int(np.floor(alpha * total))
    mask = _keep_topk_mask(magnitudes, k_kept)
    return mask, k_kept


def adaptive_radial_prune(
    F: np.ndarray, alpha: float, num_rings: int = 64
) -> Tuple[np.ndarray, int]:
    """Adaptive radial pruning that preserves a fixed global budget.

    Strategy:
    - Compute radial index for each frequency bin relative to the center.
    - Aggregate magnitude energy per ring.
    - Allocate a number of coefficients to keep per ring proportional to ring energy.
    - Within each ring, keep the top-k magnitudes.

    This guarantees the total kept coefficients equals floor(alpha * N).
    """
    _validate_alpha(alpha)
    height, width = F.shape
    yy = np.arange(height) - height / 2.0
    xx = np.arange(width) - width / 2.0
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    r = np.hypot(Y, X)
    # Discretize radius into rings
    max_r = r.max() + 1e-9
    ring_idx = np.floor((r / max_r) * num_rings).astype(int)
    ring_idx = np.clip(ring_idx, 0, num_rings - 1)

    magnitudes = np.abs(F)
    total_bins = magnitudes.size
    k_total = int(np.floor(alpha * total_bins))

    # Compute energy per ring
    energies = np.bincount(ring_idx.ravel(), weights=(magnitudes ** 2).ravel(), minlength=num_rings)
    counts = np.bincount(ring_idx.ravel(), minlength=num_rings).astype(int)
    # Avoid zero-energy rings causing zero allocation if there are bins
    energy_sum = energies.sum()
    if energy_sum <= 0:
        # Fall back to uniform allocation by population
        energies = counts.astype(float)
        energy_sum = energies.sum()

    # Initial allocation proportional to energy
    alloc = np.floor(k_total * (energies / (energy_sum + 1e-12))).astype(int)
    # Ensure we do not exceed bins in any ring
    alloc = np.minimum(alloc, counts)

    # Distribute remaining budget greedily by fractional parts of ideal allocation
    ideal = k_total * (energies / (energy_sum + 1e-12))
    fractional = ideal - np.floor(ideal)
    remaining = k_total - int(alloc.sum())
    if remaining > 0:
        take_order = np.argsort(-fractional)
        # Keep allocating in cycles until budget is exhausted or all rings saturate
        while remaining > 0:
            progressed = False
            for idx in take_order:
                if remaining == 0:
                    break
                if alloc[idx] < counts[idx]:
                    alloc[idx] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                # All rings saturated; cannot allocate more
                break

    # Build mask by selecting per-ring top-k
    mask = np.zeros_like(magnitudes, dtype=bool)
    for ring in range(num_rings):
        k = int(alloc[ring])
        if k <= 0:
            continue
        ring_positions = (ring_idx == ring)
        mags_ring = magnitudes[ring_positions]
        # Select top-k within this ring
        if k >= mags_ring.size:
            mask[ring_positions] = True
        else:
            kth_mask = _keep_topk_mask(mags_ring, k)
            # kth_mask aligns with mags_ring order; map back
            mask_ring = np.zeros_like(mags_ring, dtype=bool)
            mask_ring[kth_mask] = True
            # Assign back preserving the original order
            mask_indices = np.where(ring_positions)
            mask[mask_indices] = mask_ring

    # Due to ties, we may exceed k_total by a few. Trim if necessary.
    kept = int(mask.sum())
    if kept > k_total:
        # Turn off the smallest of the kept magnitudes globally to match budget
        kept_values = magnitudes[mask]
        extra = kept - k_total
        # Identify threshold within kept values
        idx = np.argpartition(kept_values, extra - 1)[:extra]
        # Map to positions and unset
        positions = np.vstack(np.where(mask)).T
        to_unset = tuple(positions[idx].T)
        mask[to_unset] = False

    return mask, int(mask.sum())


def reconstruct_from_mask(F: np.ndarray, mask: np.ndarray, shifted: bool = True) -> np.ndarray:
    """Reconstruct spatial image from spectrum F using a boolean mask.

    If shifted is True, F is assumed to be fftshifted.
    Returns a real-valued image.
    """
    from numpy.fft import ifft2, ifftshift

    F_pruned = np.where(mask, F, 0)
    if shifted:
        img = np.real(ifft2(ifftshift(F_pruned)))
    else:
        img = np.real(ifft2(F_pruned))
    return img


def energy_retained(F: np.ndarray, mask: np.ndarray) -> float:
    """Return fraction of spectral energy retained by mask."""
    total = float(np.sum(np.abs(F) ** 2)) + 1e-12
    kept = float(np.sum((np.abs(F) ** 2)[mask]))
    return kept / total


def estimate_compressed_bytes(
    mask: np.ndarray,
    F: np.ndarray | None = None,
    bits_per_value: int = 16,
    include_indices: bool = True,
) -> int:
    """Roughly estimate compressed size in bytes.

    - Each kept complex coefficient stores two values at bits_per_value each.
    - If include_indices, also account for storing (row, col) as 32-bit ints.
    """
    kept = int(np.count_nonzero(mask))
    coeff_bits = kept * 2 * int(bits_per_value)
    index_bits = kept * 2 * 32 if include_indices else 0
    return int(np.ceil((coeff_bits + index_bits) / 8.0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.normal(size=(128, 128))
    from numpy.fft import fft2, fftshift

    F = fftshift(fft2(img))
    mask_g, k_g = global_prune(F, alpha=0.2)
    mask_a, k_a = adaptive_radial_prune(F, alpha=0.2, num_rings=32)
    print("global kept:", k_g, "adaptive kept:", k_a, "bytes est:", estimate_compressed_bytes(mask_a))


