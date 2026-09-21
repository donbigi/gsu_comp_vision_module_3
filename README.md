# GSU Computer Vision — Module 3

**Image blurring with a filtering approach, demonstrated in both the spatial and the
Fourier domain, shipped as a Dockerised web application.**

The single idea this module demonstrates is the **convolution theorem**:

> Convolution in the spatial domain is equivalent to pointwise multiplication in the
> frequency (Fourier) domain:
>
> $$\mathcal{F}\{f * h\} \;=\; \mathcal{F}\{f\}\cdot\mathcal{F}\{h\} \;=\; F\cdot H$$

Blurring is a low-pass filter `h`. Applying it two different ways — (1) sliding-window
convolution in space, and (2) multiplying `F(u,v)` by `H(u,v)` and inverting — gives the
*same* image, to floating-point precision.

---

## Contents

| File | What it is |
|---|---|
| `filtering.py` | Core implementation: Gaussian/box kernels, spatial convolution (OpenCV + pure-NumPy reference), FFT-multiply convolution, comparison metrics, spectra, synthetic test image |
| `experiment.py` | Reproducible validation: blurs a test image both ways across 7 kernel configurations, writes images + `results.csv` + plots, prints the equivalence table |
| `app.py` | Flask web application (interactive demonstration) |
| `templates/`, `static/` | Front end (single-page UI) |
| `theory.md` / `theory.pdf` / `theory.docx` | Theory: DFT, convolution theorem derivation, Gaussian-in-frequency, evidence (plain language, ASCII math) |
| `make_pdf.py`, `make_docx.py` | Regenerate `theory.pdf` / `theory.docx` from the same write-up |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Container packaging |

---

## Quick start — Docker (the web app)

```bash
cd gsu_comp_vision_module_3
docker compose up --build
```

Then open **http://localhost:8000** . The page runs the blur **both ways** on a
synthetic test image (or your own upload), shows the two results side by side, the
image/filter spectra, and the numerical residual (~10⁻¹²) proving they are identical.

> `docker compose down` to stop it. The container is also reachable via
> `docker run --rm -p 8000:8000 gsu-cv-module3-blur`.

### What the web app shows

- **Controls** — kernel type (Gaussian / box), Gaussian σ, kernel size, image source
  (synthetic or upload), and a *Run* button.
- **Metrics** — `max |spatial − frequency|`, RMSE, PSNR, and an *identical* verdict.
- **Images** — original, spatial blur (`g = f ∗ h`), frequency blur (`g = IDFT(F·H)`),
  and the |difference| (amplified ~10¹³×, showing only floating-point noise).
- **Spectra** — `log|F(u,v)|` (image), `log|H(u,v)|` (the low-pass filter), and
  `log|F·H|` (the product) — the frequency-domain picture of the same operation.

---

## Run the experiment (evidence / validation) without Docker

Requires Python 3.9+ with `numpy`, `opencv-python`, and (for the plots) `matplotlib`:

```bash
pip install numpy opencv-python matplotlib
python3 experiment.py
```

It prints a table like this and writes `output/results.csv`, `summary.png`,
`error_vs_sigma.png`, plus per-configuration blurred/difference images:

```
configuration           kernel    max|diff|          MAE         RMSE         PSNR
------------------------------------------------------------------------------
gaussian_sigma0.5         5x5       2.842e-13    3.209e-14    4.411e-14       315.24
gaussian_sigma1.5        11x11      3.638e-12    4.439e-13    7.018e-13       291.21
...
box_25x25                25x25      2.210e-12    3.107e-13    4.897e-13       294.33
```

The maximum absolute difference between the two routes is **~10⁻¹²** (float64
round-off) and the PSNR is ~290–315 dB — the two images are indistinguishable.

---

## How it works (implementation notes)

**Spatial route** — `filtering.convolve2d_spatial` convolves the image with the kernel
via `cv2.filter2D` (zero border). A pure-NumPy `convolve2d_spatial_numpy` implements the
literal "sum of products" definition and agrees with it to ~10⁻¹³ on a test crop.

**Frequency route** — `filtering.convolve2d_frequency` computes
`G = FFT(image) · FFT(kernel)` and inverts. The one subtlety: the DFT performs
*circular* convolution, but spatial filtering with a zero border is *linear*
convolution. The code therefore zero-pads both arrays to the full linear-convolution
size `(M + k_h − 1) × (N + k_w − 1)` before transforming, then crops the central
`M × N` block. Over the padded arrays, circular ≡ linear convolution.

See `theory.md` for the full derivation and the Gaussian-is-its-own-transform result.

---

## Project structure

```
gsu_comp_vision_module_3/
├── filtering.py            # spatial + frequency filtering + metrics + test image
├── experiment.py           # validation: equivalence evidence (CSV + plots)
├── app.py                  # Flask web app
├── templates/index.html    # UI
├── static/style.css        # styling
├── static/app.js           # front-end logic
├── theory.md               # theory + evidence write-up (plain language, ASCII math)
├── theory.pdf              # PDF export of the theory
├── theory.docx             # Word export of the theory
├── make_pdf.py             # regenerates theory.pdf (needs reportlab)
├── make_docx.py            # regenerates theory.docx (needs python-docx)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── output/                 # generated evidence (created on run)
```
# gsu_comp_vision_module_3
