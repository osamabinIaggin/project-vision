#!/usr/bin/env python3
"""
Build the technical report: page-proportioned figures, then the PDF.

The analysis figures are generated for screen inspection and several are
row-stacked composites with extreme aspect ratios — the transfer panel is 4:1
tall. Placed full-width in an A4 document these span two or three pages, orphan
their captions and leave half-empty leaves. They are also full-resolution PNGs,
which inflate the rendered PDF to several times the size a reader needs.

This selects a representative subset of rows from each composite, reproportions
it toward the page, and re-encodes photographic content as JPEG. Nothing is
altered other than which rows are shown and at what resolution: no crop within a
panel, no rescaling of one panel relative to another.

The document itself is authored as a single self-contained HTML file with print
stylesheets (A4 page box, page-break control on figures and tables) and rendered
by headless Chrome. No LaTeX toolchain is assumed, which keeps the report
buildable on the same machine as the rest of the pipeline.

Run via:
    .venv/bin/python scripts/29_paper_figures.py
"""
import os, shutil, subprocess
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "docs", "figures")
DST = os.path.join(ROOT, "docs", "paper", "fig")

# name: (row height incl. gap, rows to keep, long-edge cap, format)
PLAN = {
    # 6 rows of (image | verified label | prediction); three suffice to show
    # the region-level character of the output.
    "unet_predictions_consensus": (260, [0, 1, 2], 1200, "JPEG"),
    # 12 rows: three samples for each of four communities, in name order.
    # One row per community keeps the site-to-site comparison legible.
    "transfer_middleodaw": (260, [0, 3, 6, 9], 1200, "JPEG"),
    # three stacked (OSM | Open Buildings) panels; two carry the argument.
    "label_sources_oldfadama": (806, [1, 2], 1500, "JPEG"),
    # charts and maps: already page-proportioned, keep lossless
    "flood_risk_oldfadama": (None, None, 1500, "PNG"),
    "builtup_timeseries_oldfadama": (None, None, 1400, "PNG"),
    "drain_capacity_accra": (None, None, 1600, "PNG"),
}


def main():
    os.makedirs(DST, exist_ok=True)
    for name, (rowh, rows, cap, fmt) in PLAN.items():
        src = os.path.join(SRC, f"{name}.png")
        if not os.path.exists(src):
            print(f"  {name}: source missing, skipped")
            continue
        im = Image.open(src).convert("RGB")
        w, h = im.size

        if rowh and rows:
            keep = [r for r in rows if (r + 1) * rowh <= h + rowh]
            tiles = [im.crop((0, r * rowh, w, min((r + 1) * rowh, h))) for r in keep]
            out = Image.new("RGB", (w, sum(t.size[1] for t in tiles)), (255, 255, 255))
            y = 0
            for t in tiles:
                out.paste(t, (0, y)); y += t.size[1]
            im = out
            w, h = im.size

        scale = min(1.0, cap / max(w, h))
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        ext = "jpg" if fmt == "JPEG" else "png"
        dst = os.path.join(DST, f"{name}.{ext}")
        im.save(dst, fmt, quality=88, optimize=True) if fmt == "JPEG" \
            else im.save(dst, fmt, optimize=True)
        kb = os.path.getsize(dst) / 1024
        print(f"  {name:34s} {im.size[0]:5d}x{im.size[1]:5d} "
              f"(h/w {im.size[1]/im.size[0]:.2f})  {kb:7.0f} KB  {ext}")
    render()


CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def render():
    """HTML -> PDF via headless Chrome, the only print engine assumed present."""
    src = os.path.join(ROOT, "docs", "paper", "vision_report.html")
    pdf = os.path.join(ROOT, "docs", "paper", "vision_report.pdf")
    if not os.path.exists(src):
        print("  vision_report.html missing; figures written, PDF skipped")
        return
    exe = next((c for c in CHROME_CANDIDATES if os.path.exists(c)), None) \
        or shutil.which("chromium") or shutil.which("google-chrome")
    if not exe:
        print("  no Chrome/Chromium found; open vision_report.html and print to PDF")
        return
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "--virtual-time-budget=15000", "file://" + src],
                   check=True, capture_output=True)
    print(f"\n  wrote {os.path.relpath(pdf, ROOT)} "
          f"({os.path.getsize(pdf)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
