import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

def compute_fft(img):
    F = fftshift(fft2(img))
    return F

def magnitude_spectrum(F):
    return np.abs(F)

def ifft_from_shifted(F_shifted):
    return np.real(ifft2(ifftshift(F_shifted)))
