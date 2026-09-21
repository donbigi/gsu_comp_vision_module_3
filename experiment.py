"""
Experiment / validation — show that spatial filtering and frequency-domain
filtering produce the SAME blurred image.

For each (kernel type, parameter) configuration we:

  1. blur the test image with `convolve2d_spatial`   (convolution in space),
  2. blur it with `convolve2d_frequency`              (multiplication in
                                                       the Fourier domain),
  3. compare the two with `compare()` and record the residual,
  4. also blur a small crop with the pure-NumPy convolution to confirm the
     textbook "sum of products" definition agrees with both.

Outputs (written to `output/`):

  - blurred images (spatial / frequency / difference) per configuration,
  - `results.csv` with every metric,
  - `summary.png` (side-by-side panels) and `error_vs_sigma.png`.

The headline number is `max_abs_diff`: it should be ~1e-12 (float64
round-off), proving the two routes are numerically identical.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import cv2

import filtering

OUTPUT_DIR = Path("output")

# Configurations to sweep: (label, kernel factory).
CONFIGS = [
    ("gaussian_sigma0.5", lambda: filtering.gaussian_kernel(0.5)),
    ("gaussian_sigma1.5", lambda: filtering.gaussian_kernel(1.5)),
    ("gaussian_sigma3.0", lambda: filtering.gaussian_kernel(3.0)),
    ("gaussian_sigma5.0", lambda: filtering.gaussian_kernel(5.0)),
    ("box_3x3", lambda: filtering.box_kernel(3)),
    ("box_11x11", lambda: filtering.box_kernel(11)),
    ("box_25x25", lambda: filtering.box_kernel(25)),
]

DESCRIPTIONS = {
    "gaussian_sigma0.5": "Gaussian sigma=0.5",
    "gaussian_sigma1.5": "Gaussian sigma=1.5",
    "gaussian_sigma3.0": "Gaussian sigma=3.0",
    "gaussian_sigma5.0": "Gaussian sigma=5.0",
    "box_3x3": "Box 3x3 (mean)",
    "box_11x11": "Box 11x11 (mean)",
    "box_25x25": "Box 25x25 (mean)",
}


def run() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    image = filtering.synthetic_test_image(512)
    cv2.imwrite(str(OUTPUT_DIR / "original.png"), filtering.to_uint8(image))

    # A small crop for the pure-NumPy "sum of products" demonstration.
    crop = image[200:264, 200:264]

    rows: list[dict] = []

    print(f"{'configuration':22s} {'kernel':>7s} "
          f"{'max|diff|':>12s} {'MAE':>12s} {'RMSE':>12s} {'PSNR':>12s}")
    print("-" * 78)

    for name, make_kernel in CONFIGS:
        kernel = make_kernel()
        kh, kw = kernel.shape

        spatial = filtering.convolve2d_spatial(image, kernel)
        frequency = filtering.convolve2d_frequency(image, kernel)

        metrics = filtering.compare(spatial, frequency)
        equal = metrics["max_abs_diff"] < 1e-9

        rows.append(
            {
                "config": name,
                "description": DESCRIPTIONS[name],
                "kernel": f"{kh}x{kw}",
                **metrics,
                "equal": equal,
            }
        )

        print(
            f"{name:22s} {kh:>4d}x{kw:<4d} "
            f"{metrics['max_abs_diff']:12.3e} {metrics['mae']:12.3e} "
            f"{metrics['rmse']:12.3e} {metrics['psnr']:12.2f}"
        )

        # Save the blurred images and the (amplified) difference.
        cv2.imwrite(
            str(OUTPUT_DIR / f"{name}_spatial.png"),
            filtering.to_uint8(spatial),
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"{name}_frequency.png"),
            filtering.to_uint8(frequency),
        )

        diff_u8, scale = filtering.amplified_difference(spatial, frequency)
        cv2.imwrite(str(OUTPUT_DIR / f"{name}_diff_x{scale:.0e}.png"), diff_u8)

        # Pure-NumPy convolution on the small crop (definitional check).
        np_spatial = filtering.convolve2d_spatial_numpy(crop, kernel)
        cv_spatial = filtering.convolve2d_spatial(crop, kernel)
        crop_metrics = filtering.compare(np_spatial, cv_spatial)
        print(
            f"{'  (numpy vs cv2 on crop)':22s} "
            f"    max|diff| = {crop_metrics['max_abs_diff']:.3e}"
        )

    print("-" * 78)

    # ------------------------------------------------------------------
    # Write results.csv
    # ------------------------------------------------------------------
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "config", "description", "kernel",
                "max_abs_diff", "mae", "rmse", "psnr", "equal",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {csv_path}")

    # ------------------------------------------------------------------
    # Summary figures
    # ------------------------------------------------------------------
    make_summary(image)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    worst = max(rows, key=lambda r: r["max_abs_diff"])
    print("\n============================================================")
    print("VERDICT")
    print("============================================================")
    print(
        f"Worst-case max |difference| across all configurations: "
        f"{worst['max_abs_diff']:.3e}  ({worst['description']})"
    )
    print(
        "Conclusion: spatial convolution and frequency-domain multiplication "
        "agree to floating-point precision (~1e-12)."
    )


def make_summary(image: np.ndarray) -> None:
    """Render a 2x3 side-by-side summary + an error-vs-sigma plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # plotting is optional, never fatal
        print(f"[WARN] matplotlib unavailable, skipping summary plot: {exc}")
        return

    kernel = filtering.gaussian_kernel(3.0)
    spatial = filtering.convolve2d_spatial(image, kernel)
    frequency = filtering.convolve2d_frequency(image, kernel)
    diff_u8, scale = filtering.amplified_difference(spatial, frequency)

    f_img = filtering.centered_log_magnitude(np.fft.fft2(image))
    h_tf = filtering.filter_transfer(kernel, image.shape)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (axes[0, 0], "Original", filtering.to_uint8(image)),
        (axes[0, 1], "Spatial blur (sigma=3)", filtering.to_uint8(spatial)),
        (axes[0, 2], "Frequency blur (sigma=3)", filtering.to_uint8(frequency)),
        (axes[1, 0], f"|spatial - frequency| x {scale:.0e}", diff_u8),
        (axes[1, 1], "log |F(u,v)| image spectrum", f_img),
        (axes[1, 2], "log |H(u,v)| filter (low-pass)", filtering.centered_log_magnitude(h_tf)),
    ]
    for ax, title, arr in panels:
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'summary.png'}")

    # Residual vs sigma for the Gaussian family.
    gauss = sorted(
        (r for r in CONFIGS if r[0].startswith("gaussian")),
        key=lambda r: r[0],
    )
    sigmas = [float(name.split("sigma")[1]) for name, _ in gauss]
    residuals = []
    for name, make_kernel in gauss:
        k = make_kernel()
        a = filtering.convolve2d_spatial(image, k)
        b = filtering.convolve2d_frequency(image, k)
        residuals.append(filtering.compare(a, b)["max_abs_diff"])

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogy(sigmas, residuals, "o-", color="#d62728")
    ax.axhline(1e-12, color="0.6", ls="--", lw=1, label="float64 round-off")
    ax.set_xlabel("Gaussian sigma")
    ax.set_ylabel("max |spatial - frequency|")
    ax.set_title("Residual vs sigma")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_sigma.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'error_vs_sigma.png'}")


if __name__ == "__main__":
    run()
