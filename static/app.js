"use strict";

const $ = (id) => document.getElementById(id);

const kernelType = $("kernel-type");
const sigma = $("sigma");
const ksize = $("ksize");
const source = $("source");
const fileInput = $("file");
const runBtn = $("run");

let uploadedDataUrl = null;

// ---- formatting ----------------------------------------------------------

function fmtSci(x) {
  if (x === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(x)));
  const m = x / Math.pow(10, e);
  return m.toFixed(2) + "×10" + sup(e);
}

function sup(n) {
  const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³",
    4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return String(n).split("").map((c) => map[c] || c).join("");
}

// ---- control visibility --------------------------------------------------

function syncControls() {
  const gaussian = kernelType.value === "gaussian";
  $("sigma-control").hidden = !gaussian;
  $("ksize-control").hidden = false;
  $("upload-control").hidden = source.value !== "upload";
}

kernelType.addEventListener("change", syncControls);
source.addEventListener("change", syncControls);
sigma.addEventListener("input", () => { $("sigma-val").textContent = (+sigma.value).toFixed(1); });

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => { uploadedDataUrl = reader.result; };
  reader.readAsDataURL(f);
});

// ---- run -----------------------------------------------------------------

async function run() {
  runBtn.disabled = true;
  runBtn.textContent = "Filtering…";

  const payload = {
    kernel_type: kernelType.value,
    sigma: +sigma.value,
    ksize: ksize.value === "" ? 0 : +ksize.value,
    size: 512,
  };
  if (source.value === "upload") {
    if (!uploadedDataUrl) {
      alert("Choose an image file first (Image source = Upload).");
      runBtn.disabled = false;
      runBtn.textContent = "Run filter";
      return;
    }
    payload.image = uploadedDataUrl;
  }

  try {
    const res = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    render(data);
  } catch (err) {
    console.error(err);
    alert("Filtering failed: " + err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run filter";
  }
}

function render(d) {
  $("img-original").src = d.original;
  $("img-spatial").src = d.spatial;
  $("img-frequency").src = d.frequency;
  $("img-diff").src = d.difference;
  $("img-kernel").src = d.kernel;
  $("spec-image").src = d.spectrum_image;
  $("spec-filter").src = d.spectrum_filter;
  $("spec-result").src = d.spectrum_result;

  const m = d.metrics;
  $("m-max").textContent = fmtSci(m.max_abs_diff);
  $("m-rmse").textContent = fmtSci(m.rmse);
  $("m-psnr").textContent = m.psnr === Infinity ? "∞ dB" : m.psnr.toFixed(1) + " dB";

  const verdict = $("m-verdict");
  verdict.textContent = d.equal ? "IDENTICAL ✓" : "DIFFERS";
  verdict.classList.toggle("good", d.equal);

  const scaleLabel = d.diff_scale === 0
    ? "|difference| = 0"
    : "|difference| ×" + fmtSci(d.diff_scale) + " (only float noise)";
  $("cap-diff").textContent = scaleLabel;

  $("cap-kernel").textContent =
    `Kernel (spatial filter h) — ${d.kernel_shape[0]}×${d.kernel_shape[1]}`;
}

runBtn.addEventListener("click", run);
syncControls();
run(); // load the demonstration immediately
