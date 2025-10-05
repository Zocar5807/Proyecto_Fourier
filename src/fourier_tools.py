import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def fft2d(img: np.ndarray, shift: bool = True) -> np.ndarray:
    """Compute 2D FFT; optionally return shift-centered spectrum."""
    F = fft2(img)
    return fftshift(F) if shift else F


def ifft2d(F: np.ndarray, shifted: bool = True) -> np.ndarray:
    """Inverse 2D FFT from shifted or unshifted spectrum; returns real image."""
    if shifted:
        return np.real(ifft2(ifftshift(F)))
    return np.real(ifft2(F))


def magnitude_spectrum(F: np.ndarray) -> np.ndarray:
    return np.abs(F)


def phase_spectrum(F: np.ndarray) -> np.ndarray:
    return np.angle(F)


def power_spectrum(F: np.ndarray) -> np.ndarray:
    mag = np.abs(F)
    return mag ** 2


def energy_from_spectrum(F: np.ndarray) -> float:
    """Total spectral energy sum(|F|^2)."""
    return float(np.sum(np.abs(F) ** 2))


def parseval_error(img: np.ndarray, F: np.ndarray | None = None) -> float:
    """Return |sum(x^2) - (1/N) sum(|F|^2)| for unitary consistency (fft as numpy)."""
    if F is None:
        F = fftshift(fft2(img))
    space_energy = float(np.sum(np.asarray(img, dtype=float) ** 2))
    spec_energy = float(np.sum(np.abs(F) ** 2)) / float(img.size)
    return abs(space_energy - spec_energy)


def radial_frequency_grid(shape: tuple[int, int]) -> np.ndarray:
    """Return radial distance grid from center (fftshift convention)."""
    h, w = shape
    y = np.arange(h) - h / 2.0
    x = np.arange(w) - w / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.hypot(yy, xx)


def radial_mask(shape: tuple[int, int], cutoff: float, pass_type: str = "low") -> np.ndarray:
    """Ideal radial mask; pass_type in {"low", "high"}."""
    r = radial_frequency_grid(shape)
    if pass_type == "low":
        return (r <= float(cutoff)).astype(float)
    return (r > float(cutoff)).astype(float)


if __name__ == "__main__":
    # Minimal checks
    rng = np.random.default_rng(42)
    img = rng.normal(size=(128, 128))
    F = fft2d(img, shift=True)
    err = parseval_error(img, F)
    print("parseval error:", err)
    m = radial_mask(img.shape, cutoff=20)
    print("mask mean:", float(m.mean()))
