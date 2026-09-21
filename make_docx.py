"""
Generate theory.docx from the module's theory write-up.

The text is written in plain, simple language with plain-ASCII math only
(no Greek letters or special symbols), so it is easy to read and type.
Requires: pip install python-docx
Run:      python3 make_docx.py
"""

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = "theory.docx"

doc = Document()

# Base font.
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)


def add_runs(paragraph, text):
    """Add runs to a paragraph, honouring **bold** markers."""
    for i, part in enumerate(text.split("**")):
        if part == "":
            continue
        run = paragraph.add_run(part)
        if i % 2 == 1:
            run.bold = True


def shade(paragraph, fill):
    """Light grey background for a paragraph (formula block)."""
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)


def title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4A)
    p.paragraph_format.space_after = Pt(2)
    return p


def subtitle(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p.paragraph_format.space_after = Pt(12)
    return p


def heading(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4A)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    return p


def body(text):
    p = doc.add_paragraph()
    add_runs(p, text)
    p.paragraph_format.space_after = Pt(6)
    return p


def bullet(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(3)
    add_runs(p, "-  " + text)
    return p


def formula(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    lines = text.split("\n")
    for j, line in enumerate(lines):
        run = p.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(9.5)
        if j < len(lines) - 1:
            run.add_break()
    shade(p, "F2F3F7")
    return p


# ----------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------

title("Image Blurring: Same Result in Space and in Frequency")
subtitle(
    "Blurring by convolution (a spatial filter) and blurring by multiplying the "
    "Fourier transform by the filter's transform (a frequency-domain filter) give the "
    "same image. This follows from the convolution theorem."
)

heading("1. What blurring is")
body(
    "A digital image is a grid of numbers, f[x, y]. Blurring replaces each pixel with a "
    "weighted average of the pixels around it:"
)
formula("g[x, y] = sum over all p, q of  h[p, q] * f[x - p, y - q]")
body(
    "This is called **convolution**, written f * h. The small grid of weights, h, is "
    "the **kernel** (also called a filter or mask). Two kernels are used here:"
)
bullet(
    "**Gaussian** (smooth and round). Sampled on a grid and rescaled so the weights "
    "add up to 1:"
)
formula("h[x, y] = (1 / (2 * pi * sigma^2)) * exp( -(x^2 + y^2) / (2 * sigma^2) )")
bullet("**Box / mean** (uniform): every weight is 1 / k^2 over a k-by-k window.")
body(
    "A blur keeps smooth, slowly-changing parts of the picture and removes sharp edges "
    "and noise. Smooth parts are **low frequencies**; edges and noise are **high "
    "frequencies**. That is why a blur is called a low-pass filter, and why the picture "
    "looks soft."
)

heading("2. The frequency domain")
body(
    "The 2-D discrete Fourier transform (DFT) rewrites an image as a sum of wave "
    "patterns. Each wave has a frequency (u, v), and F[u, v] is a complex number that "
    "says how much of that wave is in the image:"
)
formula("F[u, v] = sum over x, y of  f[x, y] * exp( -j * 2 * pi * (u*x/M + v*y/N) )")
body(
    "Here j is the imaginary unit (j^2 = -1), and M, N are the image width and height. "
    "Small u and v (near the origin) are low frequencies: smooth areas. Large u and v "
    "are high frequencies: edges and noise. The inverse DFT turns F back into f exactly."
)

heading("3. The convolution theorem")
body(
    "This is the main result of the module. Let F = DFT of f and H = DFT of h. Then the "
    "DFT of a convolution equals the product of the two DFTs:"
)
formula("DFT of (f * h)  =  F * H      (that is,  F{ f * h } = F{f} * F{h} )")
body(
    "**Proof** (shown in 1-D; the 2-D case works the same, applied to both directions). "
    "Write the DFT of the convolution and rearrange:"
)
formula(
    "F{f * h}[u]\n"
    "  = sum over n of ( sum over m of  f[m] * h[n - m] ) * exp( -j * 2 * pi * u*n/N )\n"
    "  = sum over m of  f[m] * sum over n of  h[n - m] * exp( -j * 2 * pi * u*n/N )\n"
    "  ( put k = n - m, so n = k + m )\n"
    "  = sum over m of  f[m] * sum over k of  h[k] * exp( -j * 2 * pi * u*(k + m)/N )\n"
    "  = ( sum over m of  f[m] * exp( -j * 2 * pi * u*m/N ) )\n"
    "      * ( sum over k of  h[k] * exp( -j * 2 * pi * u*k/N ) )\n"
    "  = F[u] * H[u]"
)
body("That proves it. What it means for blurring:")
bullet("Blur in frequency: G[u, v] = F[u, v] * H[u, v], then g = inverse DFT of G.")
bullet(
    "H[u, v] (the DFT of the kernel) is the **transfer function**: it scales each "
    "frequency up or down. For a blur it is big at low frequencies and small at high "
    "frequencies (a low-pass shape)."
)
bullet(
    "One filter, two views: convolution in space and multiplication in frequency give "
    "the same image."
)

heading("4. A Gaussian's transform is a Gaussian")
body("The Gaussian kernel is special: its Fourier transform is another Gaussian.")
formula(
    "exp( -x^2 / (2 * sigma^2) )  gives  "
    "sqrt(2 * pi) * sigma * exp( -2 * pi^2 * sigma^2 * u^2 )"
)
body(
    "So \"convolve with a Gaussian of width sigma\" is the same as \"multiply by a "
    "Gaussian of width 1 / sigma\" in frequency. A wide Gaussian in space (heavy blur) "
    "gives a narrow low-pass in frequency, and the other way around. A box kernel "
    "instead has a sinc-shaped transform, sinc(u) * sinc(v), which is why box blurs "
    "show faint ringing near edges."
)

heading("5. Implementation and validation (evidence)")
body("filtering.py computes the same blur two independent ways:")

t1 = doc.add_table(rows=3, cols=3)
t1.style = "Table Grid"
for row in t1.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            p.paragraph_format.space_after = Pt(2)
t1_data = [
    ["Route", "Operation", "Code"],
    [
        "Spatial",
        "g = f * h (slide the kernel over the image)",
        "convolve2d_spatial (OpenCV),\nconvolve2d_spatial_numpy (pure NumPy)",
    ],
    [
        "Frequency",
        "G = F * H, then g = inverse DFT of G",
        "convolve2d_frequency (NumPy FFT)",
    ],
]
for r, row_data in enumerate(t1_data):
    for c, val in enumerate(row_data):
        cell = t1.cell(r, c)
        lines = val.split("\n")
        first = cell.paragraphs[0]
        run = first.add_run(lines[0])
        if r == 0:
            run.bold = True
        for extra in lines[1:]:
            p2 = cell.add_paragraph()
            p2.add_run(extra)

body(
    "One important detail: the DFT does **circular** convolution, but blurring with a "
    "zero border is **linear** convolution. So the frequency path zero-pads the image "
    "and kernel to (M + kh - 1) by (N + kw - 1) before the transform, then crops the "
    "middle M-by-N block. On the padded arrays, circular convolution equals linear "
    "convolution."
)
body(
    "The experiment (experiment.py) blurs a test image both ways. The test image "
    "contains a gradient, a sharp square, a checkerboard, a sinusoid, a single bright "
    "dot, and random noise, so it has energy at every frequency. It uses Gaussian "
    "kernels (sigma = 0.5, 1.5, 3, 5) and box kernels (3x3, 11x11, 25x25)."
)
body("Results (float64, saved in output/results.csv):")

t2 = doc.add_table(rows=8, cols=5)
t2.style = "Table Grid"
t2_data = [
    ["configuration", "kernel", "max abs difference", "RMSE", "PSNR"],
    ["Gaussian sigma=0.5", "5x5", "2.8e-13", "4.4e-14", "315 dB"],
    ["Gaussian sigma=1.5", "11x11", "3.6e-12", "7.0e-13", "291 dB"],
    ["Gaussian sigma=3.0", "19x19", "3.5e-12", "6.9e-13", "291 dB"],
    ["Gaussian sigma=5.0", "31x31", "2.9e-12", "6.0e-13", "293 dB"],
    ["Box 3x3", "3x3", "4.0e-13", "6.0e-14", "313 dB"],
    ["Box 11x11", "11x11", "3.3e-12", "6.2e-13", "292 dB"],
    ["Box 25x25", "25x25", "2.2e-12", "4.9e-13", "294 dB"],
]
for r, row_data in enumerate(t2_data):
    for c, val in enumerate(row_data):
        cell = t2.cell(r, c)
        run = cell.paragraphs[0].add_run(val)
        if r == 0:
            run.bold = True

body(
    "The largest difference anywhere is about 1e-12, which is just floating-point "
    "round-off, and the PSNR is about 290 to 315 dB. When the results are rounded to "
    "8-bit images, the saved files are byte-for-byte the same in six of seven cases; "
    "the one exception (box 3x3) differs by at most 1 grey level in 19 of 262144 "
    "pixels, at pixels that land exactly on a rounding boundary. The pure-NumPy "
    "convolution agrees with OpenCV to about 1e-13 on a test crop."
)
body(
    "**Conclusion.** Convolution in space and multiplication in frequency produce the "
    "same blurred image, up to round-off, exactly as the convolution theorem predicts."
)

heading("6. Running it")
bullet("Web app: docker compose up, then open http://localhost:8000.")
bullet("Evidence: python3 experiment.py (writes output/results.csv and plots).")

doc.save(OUT)
print("Wrote", OUT)
