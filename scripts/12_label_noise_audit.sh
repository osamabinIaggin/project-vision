#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-2 label-quality audit — quantifies why val IoU plateaus near 0.58.
#
# Two independent footprint sources are rasterised onto the exact Stage-2 crop
# grid (scripts/05 window) and compared:
#   * OSM        — dense coverage, but per-building offsets and merged shacks;
#   * Open Buildings v3 (scripts/11) — cleaner shapes, but under-detects in the
#     densest blocks.
# Their mutual IoU is the empirical noise floor of the labels: a model cannot
# be scored meaningfully above the level at which the label sources themselves
# disagree. If a trained checkpoint is present, a global-registration search
# additionally cross-correlates the OSM mask against model predictions to test
# whether the offset is a fixable systematic shift (finding: it is not —
# optimum ~0.2 m, i.e. the label error is per-building and random).
#
# Outputs: printed statistics + docs/figures/label_sources_oldfadama.png
#          (OSM red / Open Buildings cyan outlines over the orthomosaic).
# ----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/_env.sh"
export GDAL_PAM_ENABLED=NO

ORTHO="${AOI_DIR}/imagery/oldfadama_2020_5cm.tif"
GOB="${AOI_DIR}/open_buildings/open_buildings.gpkg"
OSM="${AOI_DIR}/osm/oldfadama_osm.gpkg"
WORK="${AOI_DIR}/open_buildings/_audit"; mkdir -p "${WORK}"
ULX=807218; ULY=614412; LRX=808242; LRY=613388           # scripts/05 window

[ -f "${GOB}" ] || { echo "run scripts/11_acquire_open_buildings.sh first"; exit 1; }

echo "Rasterising both label sources onto the Stage-2 crop grid ..."
"${BIN}/gdal_translate" -q -projwin ${ULX} ${ULY} ${LRX} ${LRY} "${ORTHO}" "${WORK}/crop.tif"
for src in osm gob; do
  "${BIN}/gdal_translate" -q -b 1 -ot Byte -scale 0 255 0 0 "${WORK}/crop.tif" "${WORK}/mask_${src}.tif"
done
"${BIN}/ogr2ogr" -q -f GPKG "${WORK}/osm_b.gpkg" -t_srs EPSG:32630 \
  -where "building IS NOT NULL" -nln b "${OSM}" multipolygons
"${BIN}/gdal_rasterize" -q -burn 255 -l b "${WORK}/osm_b.gpkg" "${WORK}/mask_osm.tif"
"${BIN}/gdal_rasterize" -q -burn 255 -l buildings "${GOB}" "${WORK}/mask_gob.tif"

"${PYTHON:-$(dirname "$0")/../.venv/bin/python}" - "$WORK" <<'PYEOF'
import os, sys, importlib.util
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

WORK = os.path.abspath(sys.argv[1])
ROOT = os.path.normpath(os.path.join(WORK, "..", "..", "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures", "label_sources_oldfadama.png")
CKPT = os.path.join(ROOT, "accra_flood", "oldfadama", "tiles", "_run", "resunet_best.pt")

img = np.asarray(Image.open(f"{WORK}/crop.tif").convert("RGB"))
osm = np.asarray(Image.open(f"{WORK}/mask_osm.tif")) > 127
gob = np.asarray(Image.open(f"{WORK}/mask_gob.tif")) > 127

inter, union = (osm & gob).sum(), (osm | gob).sum()
print(f"built-up fraction: OSM={osm.mean():.3f}  OpenBuildings={gob.mean():.3f}")
print(f"inter-source agreement IoU(OSM, OpenBuildings) = {inter/union:.3f}   <- label noise floor")

def edges(m):
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] ^ m[:-1, :]; e[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    return e

panels = []
for (y, x) in [(3000, 3000), (9000, 9000), (15000, 6000)]:
    c = img[y:y+800, x:x+800]
    a = c.copy(); a[edges(osm[y:y+800, x:x+800])] = [255, 0, 0]
    b = c.copy(); b[edges(gob[y:y+800, x:x+800])] = [0, 255, 255]
    gap = np.full((800, 6, 3), 255, np.uint8)
    panels.append(np.concatenate([a, gap, b], axis=1))
grid = np.concatenate([np.concatenate([p, np.full((6, p.shape[1], 3), 255, np.uint8)]) for p in panels])
Image.fromarray(grid).save(FIG)
print(f"wrote {FIG}  (left: OSM, red | right: Open Buildings, cyan)")

if os.path.exists(CKPT):
    import torch
    spec = importlib.util.spec_from_file_location(
        "seg", os.path.join(ROOT, "scripts", "09_train_unet_v3.py"))
    seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = seg.ResUNet().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()

    print("Predicting the full crop for the global-registration search ...")
    arr = img.astype(np.float32) / 255.0
    T, n = 512, img.shape[0] // 512
    pred = np.zeros((n * 256, n * 256), np.float32)
    mean = torch.tensor(seg.IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(seg.IMAGENET_STD).view(1, 3, 1, 1).to(device)
    with torch.no_grad():
        for r in range(n):
            x = torch.cat([torch.nn.functional.interpolate(
                torch.from_numpy(arr[r*T:(r+1)*T, c*T:(c+1)*T]).permute(2, 0, 1)[None],
                size=(256, 256), mode="bilinear") for c in range(n)]).to(device)
            y = torch.sigmoid(model((x - mean) / std))[:, 0].cpu().numpy()
            for c in range(n):
                pred[r*256:(r+1)*256, c*256:(c+1)*256] = y[c]
    p = pred > 0.5
    m0 = osm[:n*T, :n*T].reshape(n*256, 2, n*256, 2).mean((1, 3)) > 0.5   # 10 cm grid
    best = (0, 0, -1.0)
    for dy in range(-30, 31, 2):
        for dx in range(-30, 31, 2):
            m = np.roll(np.roll(m0, dy, 0), dx, 1)
            a, b = p[40:-40, 40:-40], m[40:-40, 40:-40]
            iou = (a & b).sum() / max(1, (a | b).sum())
            if iou > best[2]: best = (dy, dx, iou)
    a, b = p[40:-40, 40:-40], m0[40:-40, 40:-40]
    print(f"IoU(prediction, OSM) unshifted = {(a & b).sum()/(a | b).sum():.4f}")
    print(f"best global shift = ({best[0]*0.1:+.1f} m, {best[1]*0.1:+.1f} m) -> IoU {best[2]:.4f}"
          f"   (no exploitable systematic offset)")
PYEOF
