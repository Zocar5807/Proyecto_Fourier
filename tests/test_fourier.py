import numpy as np

from src.fourier_tools import fft2d, ifft2d, parseval_error


def test_parseval_identity_tight():
    rng = np.random.default_rng(123)
    img = rng.normal(size=(64, 96))
    F = fft2d(img, shift=True)
    err = parseval_error(img, F)
    assert err < 1e-5


