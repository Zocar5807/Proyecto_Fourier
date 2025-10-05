"""Image quality metrics: MSE, PSNR, SSIM with safe data_range handling.

Provides a single evaluate() entrypoint returning a dict of metrics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def _as_float(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(float, copy=False)
    return image.astype(float, copy=True)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a_f = _as_float(a)
    b_f = _as_float(b)
    diff = a_f - b_f
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None) -> float:
    a_f = _as_float(a)
    b_f = _as_float(b)
    if data_range is None:
        # Infer from input dynamic range combined
        amin = min(float(a_f.min()), float(b_f.min()))
        amax = max(float(a_f.max()), float(b_f.max()))
        data_range = amax - amin if amax > amin else 1.0
    return float(_psnr(a_f, b_f, data_range=float(data_range)))


def ssim(a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None) -> float:
    a_f = _as_float(a)
    b_f = _as_float(b)
    if data_range is None:
        amin = min(float(a_f.min()), float(b_f.min()))
        amax = max(float(a_f.max()), float(b_f.max()))
        data_range = amax - amin if amax > amin else 1.0
    # If images are 2D; set channel_axis None; for multi-channel, set channel_axis=-1
    return float(_ssim(a_f, b_f, data_range=float(data_range), channel_axis=None))


def evaluate(original: np.ndarray, reconstructed: np.ndarray, data_range: Optional[float] = None) -> Dict[str, float]:
    """Compute MSE, PSNR, SSIM between two images.

    If data_range is None, it is inferred from both images.
    """
    return {
        "mse": mse(original, reconstructed),
        "psnr": psnr(original, reconstructed, data_range=data_range),
        "ssim": ssim(original, reconstructed, data_range=data_range),
    }


if __name__ == "__main__":
    # Minimal self-test
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 64))
    b = a + 0.01 * rng.normal(size=(64, 64))
    print(evaluate(a, b))


