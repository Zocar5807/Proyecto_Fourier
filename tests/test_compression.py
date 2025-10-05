import numpy as np

from numpy.fft import fft2, fftshift

from src.compression import adaptive_radial_prune, global_prune, reconstruct_from_mask
from src.metrics import evaluate


def _synthetic_image(shape=(128, 128)):
    y = np.linspace(0, 1, shape[0])[:, None]
    x = np.linspace(0, 1, shape[1])[None, :]
    img = (
        0.6 * np.sin(2 * np.pi * (5 * x + 3 * y))
        + 0.3 * np.cos(2 * np.pi * (2 * x - 4 * y))
    )
    return img.astype(float)


def test_adaptive_prune_keeps_budget():
    img = _synthetic_image((96, 96))
    F = fftshift(fft2(img))
    alpha = 0.25
    mask, kept = adaptive_radial_prune(F, alpha=alpha, num_rings=32)
    total = F.size
    assert kept == int(np.floor(alpha * total))


def test_psnr_over_20db_after_reconstruction():
    img = _synthetic_image((128, 128))
    F = fftshift(fft2(img))
    alpha = 0.7
    mask, _ = global_prune(F, alpha=alpha)
    recon = reconstruct_from_mask(F, mask, shifted=True)
    metrics = evaluate(img, recon)
    assert metrics["psnr"] > 20.0


