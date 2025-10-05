"""Top-level package for reusable Fourier project utilities.

This package exposes common submodules and selected functions for convenient imports
from notebooks, e.g.:

    from src.compression import global_prune, adaptive_radial_prune
    from src.fourier_tools import energy_from_spectrum
    from src.metrics import evaluate
"""

from . import fourier_tools as fourier_tools
from . import filters as filters
from . import compression as compression
from . import metrics as metrics

# Re-export common functions for ergonomics
from .fourier_tools import (
    fft2d,
    ifft2d,
    magnitude_spectrum,
    phase_spectrum,
    power_spectrum,
    energy_from_spectrum,
    parseval_error,
    radial_frequency_grid,
    radial_mask,
)
from .compression import (
    global_prune,
    adaptive_radial_prune,
    reconstruct_from_mask,
    energy_retained,
    estimate_compressed_bytes,
)
from .metrics import evaluate

__all__ = [
    # submodules
    "fourier_tools",
    "filters",
    "compression",
    "metrics",
    # selected functions
    "fft2d",
    "ifft2d",
    "magnitude_spectrum",
    "phase_spectrum",
    "power_spectrum",
    "energy_from_spectrum",
    "parseval_error",
    "radial_frequency_grid",
    "radial_mask",
    "global_prune",
    "adaptive_radial_prune",
    "reconstruct_from_mask",
    "energy_retained",
    "estimate_compressed_bytes",
    "evaluate",
]

__version__ = "0.1.0"


