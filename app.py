"""
Web demonstration — image blurring by filtering.

Serves an interactive page where the user picks a kernel (Gaussian or box),
tunes its parameters, optionally uploads their own image, and sees — side by
side — the result of:

  - spatial filtering   (convolution in space:   g = f * h)
  - frequency filtering (multiplication in freq: G = F * H, g = IDFT(G))

together with the image/filter spectra and the numerical residual that shows
the two results are identical (the convolution theorem).

Run directly (python app.py) or via Docker (see README / docker-compose.yml).
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import filtering

app = Flask(__name__)

MAX_SIDE = 640  # cap the working resolution so FFT stays fast / memory-safe


# ======================================================================
# Helpers
# ======================================================================


def ndarray_to_data_url(arr_u8: np.ndarray) -> str:
    """Encode a uint8 image as a base64 PNG data URL."""
    ok, buf = cv2.imencode(".png", arr_u8)
    if not ok:
        raise ValueError("could not encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64


def decode_upload(data_url: str) -> np.ndarray:
    """Decode a client-uploaded data URL to a float64 grayscale image."""
    header, _, b64 = data_url.partition(",")
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode uploaded image")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64)


def clamp_odd(value: int, lo: int = 1, hi: int = 201) -> int:
    """Clamp a kernel size to an odd integer within [lo, hi]."""
    value = int(value)
    value = max(lo, min(hi, value))
    if value % 2 == 0:
        value += 1
    return value


def build_kernel(kernel_type: str, sigma: float, ksize: int) -> np.ndarray:
    """Build a Gaussian or box kernel from the request parameters."""
    sigma = max(0.2, float(sigma))
    if kernel_type == "box":
        return filtering.box_kernel(clamp_odd(ksize, 3, 201))
    # Gaussian: ksize=0 means "auto" (2*ceil(3*sigma)+1).
    if ksize and int(ksize) > 0:
        return filtering.gaussian_kernel(sigma, clamp_odd(ksize, 3, 201))
    return filtering.gaussian_kernel(sigma)


def run_pipeline(image: np.ndarray, kernel: np.ndarray) -> dict:
    """Blur `image` both ways and package every result for the client."""
    spatial = filtering.convolve2d_spatial(image, kernel)
    frequency = filtering.convolve2d_frequency(image, kernel)
    metrics = filtering.compare(spatial, frequency)

    diff_u8, scale = filtering.amplified_difference(spatial, frequency)

    # Frequency-domain pictures (image-size, centred).  These literally show
    # F(u,v), H(u,v) and the pointwise product F·H.
    f_img = np.fft.fft2(image)
    h_img = filtering.filter_transfer(kernel, image.shape)
    g_img = f_img * h_img

    return {
        "original": ndarray_to_data_url(filtering.to_uint8(image)),
        "spatial": ndarray_to_data_url(filtering.to_uint8(spatial)),
        "frequency": ndarray_to_data_url(filtering.to_uint8(frequency)),
        "difference": ndarray_to_data_url(diff_u8),
        "diff_scale": scale,
        "kernel": ndarray_to_data_url(filtering.upscale_to_uint8(kernel)),
        "spectrum_image": ndarray_to_data_url(filtering.centered_log_magnitude(f_img)),
        "spectrum_filter": ndarray_to_data_url(filtering.centered_log_magnitude(h_img)),
        "spectrum_result": ndarray_to_data_url(filtering.centered_log_magnitude(g_img)),
        "metrics": metrics,
        "equal": metrics["max_abs_diff"] < 1e-9,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "kernel_shape": [int(kernel.shape[0]), int(kernel.shape[1])],
    }


# ======================================================================
# Routes
# ======================================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/filter", methods=["POST"])
def api_filter():
    data = request.get_json(force=True) or {}

    kernel_type = data.get("kernel_type", "gaussian")
    sigma = float(data.get("sigma", 2.0))
    ksize = int(data.get("ksize", 0))

    image_b64 = data.get("image")
    if image_b64 and image_b64.startswith("data:image"):
        image = decode_upload(image_b64)
    else:
        size = int(data.get("size", 512))
        image = filtering.synthetic_test_image(max(128, min(1024, size)))

    # Cap the working resolution for speed / memory.
    h, w = image.shape
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        image = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    kernel = build_kernel(kernel_type, sigma, ksize)

    try:
        result = run_pipeline(image, kernel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result["kernel_type"] = kernel_type
    result["sigma"] = sigma
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
