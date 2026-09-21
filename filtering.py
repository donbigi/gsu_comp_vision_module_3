"""
Image blurring by filtering — spatial convolution vs. frequency-domain
multiplication.

The convolution theorem is the single fact this module demonstrates:

        g(x, y) = f(x, y) * h(x, y)          (spatial: convolution)
        G(u, v) = F(u, v) · H(u, v)          (frequency: pointwise product)
        g       = IDFT(G)                    (same image, up to round-off)

`h` is the blur kernel (Gaussian or box / mean), `F = DFT(f)` and
`H = DFT(h)` are the 2-D discrete Fourier transforms, and `*` denotes
linear convolution.  Blurring is a *low-pass* filter: it suppresses high
spatial frequencies (edges, noise) and keeps low ones (smooth regions).

Two independent implementations are provided and compared:

    convolve2d_spatial   -> convolution in the *spatial* domain (OpenCV,
                           with a pure-NumPy reference for small images)
    convolve2d_frequency -> pointwise multiplication in the *frequency*
                           domain (NumPy FFT)

They must agree to machine precision; `compare()` reports the residual.
"""

from __future__ import annotations

import numpy as np
import cv2

# ======================================================================
# KERNELS (the spatial filter h)
# ======================================================================


def gaussian_kernel(sigma: float, ksize: int | None = None) -> np.ndarray:
    """Return a normalized 2-D Gaussian kernel of odd size.

    The continuous Gaussian is

        h(x, y) = (1 / (2 pi sigma^2)) * exp(-(x^2 + y^2) / (2 sigma^2))

    sampled on a square grid.  The discrete kernel is re-normalized so its
    entries sum to 1 (so it preserves the image mean / DC component).

    If `ksize` is not given it defaults to 2 * ceil(3*sigma) + 1, which
    captures ~99.7% of the Gaussian mass.
    """
    if ksize is None:
        radius = int(np.ceil(3.0 * sigma))
        ksize = 2 * radius + 1
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    axis = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


def box_kernel(ksize: int) -> np.ndarray:
    """Return a uniform (box / moving-average) kernel of odd size.

    Every entry is 1 / ksize^2, so the filter is the arithmetic mean of the
    neighbourhood — the simplest possible low-pass filter.
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    return np.full((ksize, ksize), 1.0 / (ksize * ksize), dtype=np.float64)


# ======================================================================
# SPATIAL FILTERING  (convolution in space)
# ======================================================================


def convolve2d_spatial(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve `image` with `kernel` in the spatial domain.

    Uses OpenCV's `filter2D`, which computes the standard sliding-window
    "same"-size convolution with zero border (pixels outside the image are
    treated as 0):

        g[i, j] = sum_{p,q}  f[i + p - r, j + q - r] * kernel[p, q]

    (`r` is the kernel radius; the kernel is symmetric for Gaussian and box,
    so the correlation `filter2D` performs is identical to convolution.)
    Returns float64.
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    return cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)


def convolve2d_spatial_numpy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Pure-NumPy spatial convolution (the literal "sum of products").

    This is the textbook definition written directly: slide the (flipped)
    kernel over the zero-padded image and accumulate products.  It is
    O(H * W * ksize^2) and materializes every window, so use it only on
    small images / crops.  It exists so the "convolution in space" in the
    theory has a one-to-one code counterpart, independent of OpenCV.

        g[i, j] = sum_{p,q} f[i - p, j - q] * kernel[p, q]
    """
    from numpy.lib.stride_tricks import sliding_window_view

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2

    padded = np.pad(image, ((rh, rh), (rw, rw)), mode="constant")
    windows = sliding_window_view(padded, (kh, kw))
    flipped = np.flip(kernel)  # true convolution (no-op for symmetric kernels)
    return np.einsum("ijkl,kl->ij", windows, flipped)


# ======================================================================
# FREQUENCY-DOMAIN FILTERING  (multiplication in the Fourier domain)
# ======================================================================


def convolve2d_frequency(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve `image` with `kernel` by multiplying DFTs.

    Implements  G = F . H  then  g = IDFT(G), where F = DFT(image),
    H = DFT(kernel).

    The DFT computes *circular* convolution, but we want *linear*
    convolution (the kind spatial filtering does with a zero border).  We
    therefore zero-pad both arrays to the full linear-convolution size
    (M + Kh - 1) x (N + Kw - 1) first; over a padded array the circular
    convolution coincides with the linear one.  The central M x N block is
    then the "same"-size output.

    The kernel origin is placed at (0, 0) of the padded array; for the
    symmetric kernels used here that is equivalent to the centred kernel,
    so the crop offset is exactly the kernel radius.
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    m, n = image.shape
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2

    out_h = m + kh - 1
    out_w = n + kw - 1

    f_pad = np.zeros((out_h, out_w), dtype=np.float64)
    f_pad[:m, :n] = image

    k_pad = np.zeros((out_h, out_w), dtype=np.float64)
    k_pad[:kh, :kw] = kernel  # origin at (0, 0)

    F = np.fft.fft2(f_pad)
    H = np.fft.fft2(k_pad)   # the filter's transfer function
    G = F * H                # pointwise multiplication in frequency
    g_full = np.fft.ifft2(G).real

    return g_full[rh:rh + m, rw:rw + n]


# ======================================================================
# COMPARISON / METRICS
# ======================================================================


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Numerically compare two images and report their discrepancy.

    For two implementations of the *same* operation the residual is pure
    floating-point round-off, so the metrics collapse to ~1e-12.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = float("inf") if rmse == 0.0 else 20.0 * np.log10(255.0 / rmse)
    return {
        "max_abs_diff": max_abs,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
    }


def amplified_difference(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the |difference| scaled to fill [0, 255], plus the scale used.

    Because the two results are near-identical, the difference is tiny
    (float noise).  Scaling it to full range lets the viewer see that there
    is *no structure* left — only numerical noise.
    """
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    peak = float(diff.max()) if diff.size else 0.0
    if peak == 0.0:
        return np.zeros_like(diff, dtype=np.uint8), 0.0
    scale = 255.0 / peak
    return np.clip(diff * scale, 0.0, 255.0).astype(np.uint8), scale


# ======================================================================
# SPECTRA (for visualising the frequency domain)
# ======================================================================


def centered_log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Log-magnitude of a spectrum, shifted to centre and scaled to uint8.

    The DC component (zero frequency) is centred so the classic low-pass /
    high-pass picture appears: bright centre = low frequencies.
    """
    mag = np.abs(np.fft.fftshift(spectrum))
    mag = np.log1p(mag)
    mag -= mag.min()
    mag /= (mag.max() + 1e-12)
    return (mag * 255.0).astype(np.uint8)


def filter_transfer(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return H(u, v) = DFT of the (centred) kernel, zero-padded to `shape`.

    The kernel's centre is moved to (0, 0) with `ifftshift` (as the DFT
    convention expects), then the kernel is zero-padded and transformed.
    `centered_log_magnitude()` of the result shows the filter's frequency
    response.
    """
    kh, kw = kernel.shape
    m, n = shape
    centred = np.fft.ifftshift(kernel)
    k_pad = np.pad(centred, ((0, m - kh), (0, n - kw)), mode="constant")
    return np.fft.fft2(k_pad)


# ======================================================================
# HELPERS
# ======================================================================


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip a float image to [0, 255] and convert to uint8."""
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def upscale_to_uint8(small: np.ndarray, target: int = 160) -> np.ndarray:
    """Nearest-neighbour upscale a small array to a square for display.

    Used to render a small kernel (e.g. 15x15) at a visible size.
    """
    small = small.astype(np.float64)
    small -= small.min()
    rng = small.max() - small.min()
    if rng > 0:
        small = small / rng
    small = (small * 255.0).astype(np.uint8)
    return cv2.resize(small, (target, target), interpolation=cv2.INTER_NEAREST)


def synthetic_test_image(size: int = 512) -> np.ndarray:
    """Build a grayscale test image with energy at *all* frequencies.

    It mixes a smooth gradient (low frequency), a sharp square (mid),
    a checkerboard and a sinusoid (high), a single bright impulse (a delta
    whose blur is exactly the kernel — the point-spread function), and
    white noise.  Blurring it makes every effect visible at once.
    """
    size = int(size)
    img = np.zeros((size, size), dtype=np.float64)

    # Low frequency: smooth gradients.
    ramp = np.linspace(0.0, 255.0, size, dtype=np.float64)
    img += 0.20 * ramp[None, :]
    img += 0.05 * ramp[:, None]

    # Mid frequency: a sharp square (strong edges), ~31% of the image.
    w = int(round(0.31 * size))
    x0 = y0 = int(round(0.08 * size))
    img[y0:y0 + w, x0:x0 + w] = 230.0

    # High frequency: checkerboard patch.
    cw = int(round(0.31 * size))
    c0x = int(round(0.58 * size))
    c0y = int(round(0.12 * size))
    block = max(2, cw // 8)
    yy, xx = np.mgrid[0:cw, 0:cw]
    img[c0y:c0y + cw, c0x:c0x + cw] = ((xx // block + yy // block) % 2) * 255.0

    # A single specific high frequency: vertical sinusoid bars.
    period = max(6, int(round(0.023 * size)))
    xx, yy = np.mgrid[0:size, 0:size]
    img += 60.0 * np.sin(2.0 * np.pi * xx / period) * ((xx > 0.62 * size) & (yy < 0.39 * size))

    # A bright impulse (delta) — its blur IS the kernel (the PSF).
    img[size // 2, size // 2] = 255.0

    # All frequencies: white noise.
    rng = np.random.default_rng(42)
    img += rng.normal(0.0, 18.0, (size, size))

    return np.clip(img, 0.0, 255.0)
