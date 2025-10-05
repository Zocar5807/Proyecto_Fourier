"""Frequency-domain filters for 2D images.

Implements ideal, Gaussian, and Butterworth low/high-pass filters.
All functions return float masks of shape (H, W) with values in [0, 1].
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

PassType = Literal["low", "high"]


def _radial_distance(shape: Tuple[int, int]) -> np.ndarray:
    """Return radial distance grid (in pixels) from the frequency center.

    Center is defined at (H/2, W/2) matching fftshift convention.
    """
    height, width = shape
    y = np.arange(height) - height / 2.0
    x = np.arange(width) - width / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.hypot(yy, xx)
    return r


def ideal_filter(shape: Tuple[int, int], cutoff: float, pass_type: PassType = "low") -> np.ndarray:
    """Ideal low/high-pass filter mask.

    - cutoff: radius in pixels.
    - pass_type: 'low' keeps r <= cutoff, 'high' keeps r > cutoff.
    """
    r = _radial_distance(shape)
    if pass_type == "low":
        mask = (r <= float(cutoff)).astype(float)
    else:
        mask = (r > float(cutoff)).astype(float)
    return mask


def gaussian_filter(shape: Tuple[int, int], sigma: float, pass_type: PassType = "low") -> np.ndarray:
    """Gaussian low/high-pass filter mask.

    - sigma: standard deviation of Gaussian in pixels.
    """
    r = _radial_distance(shape)
    # Low-pass Gaussian
    lp = np.exp(-(r ** 2) / (2.0 * (float(sigma) ** 2) + 1e-12))
    if pass_type == "low":
        return lp
    # High-pass is complementary
    return 1.0 - lp


def butterworth_filter(
    shape: Tuple[int, int], cutoff: float, order: int = 2, pass_type: PassType = "low"
) -> np.ndarray:
    """Butterworth low/high-pass filter mask.

    - cutoff: radius in pixels.
    - order: Butterworth order (>=1).
    """
    r = _radial_distance(shape)
    epsilon = 1e-12
    # Avoid division by zero at the center
    r_norm = np.maximum(r / (float(cutoff) + epsilon), epsilon)
    lp = 1.0 / (1.0 + r_norm ** (2 * int(order)))
    if pass_type == "low":
        return lp
    return 1.0 - lp


if __name__ == "__main__":
    # Minimal self-test
    shape = (256, 256)
    ideal_lp = ideal_filter(shape, cutoff=30, pass_type="low")
    gauss_hp = gaussian_filter(shape, sigma=20.0, pass_type="high")
    bw_lp = butterworth_filter(shape, cutoff=40, order=2, pass_type="low")
    print(
        "Filters created:",
        ideal_lp.shape,
        float(ideal_lp.mean()),
        float(gauss_hp.mean()),
        float(bw_lp.mean()),
    )


