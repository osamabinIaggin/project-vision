#!/usr/bin/env python3
"""
Stage-4 (integrated) — composite flood-exposure surface for the Old Fadama AOI.

Combines the two independent products of the pipeline:
  * HAZARD   — height above nearest drainage (HAND) from the Stage-3 FABDEM
               hydrology (scripts/15), bilinearly resampled onto the AOI grid.
               Classes follow common HAND practice: <2 m severe, 2-5 m
               moderate, >5 m marginal (for the fluvial mechanism; pluvial
               ponding is not modelled).
  * EXPOSURE — built-up extent predicted by the consensus-trained U-Net
               (scripts/13) on the 2020 orthomosaic, at 10 cm.

Outputs:
  * accra_flood/output/hydro/aoi_hand.tif      — HAND on the AOI grid;
  * accra_flood/output/flood_exposure.csv      — built-up area (ha) per class;
  * docs/figures/flood_risk_oldfadama.png      — ortho + tinted risk overlay.

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/16_flood_risk_surface.py
"""
import os, subprocess, importlib.util
import numpy as np
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
AOI = os.path.join(ROOT, "accra_flood", "oldfadama")
HYDRO = os.path.join(ROOT, "accra_flood", "output", "hydro")
OUTCSV = os.path.join(ROOT, "accra_flood", "output", "flood_exposure.csv")
FIG = os.path.join(ROOT, "docs", "figures", "flood_risk_oldfadama.png")
BIN = os.environ.get("BIN", "")

# scripts/05 crop window (EPSG:32630) — the canonical AOI grid at 5 cm.
ULX, ULY, LRX, LRY = 807218, 614412, 808242, 613388

spec = importlib.util.spec_from_file_location("seg", os.path.join(HERE, "09_train_unet_v3.py"))
seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)


def build_aoi_hand():
    """Warp basin HAND onto the AOI window at 2 m (bilinear)."""
    dst = os.path.join(HYDRO, "aoi_hand.tif")
    subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                    "-t_srs", "EPSG:32630", "-tr", "2", "2", "-r", "bilinear",
                    "-ot", "Float32",
                    "-te", str(ULX), str(LRY), str(LRX), str(ULY),
                    # baseline profile so downstream PIL readers can open it
                    "-co", "PROFILE=BASELINE", "-co", "COMPRESS=NONE",
                    os.path.join(HYDRO, "hand.tif"), dst], check=True)
    return dst


def predict_builtup(device="mps"):
    """Consensus-model built-up probability over the AOI crop (10 cm grid)."""
    ortho = os.path.join(AOI, "imagery", "oldfadama_2020_5cm.tif")
    crop = os.path.join(HYDRO, "_aoi_crop.tif")
    subprocess.run([os.path.join(BIN, "gdal_translate"), "-q",
                    "-projwin", str(ULX), str(ULY), str(LRX), str(LRY),
                    ortho, crop], check=True)
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(crop).convert("RGB"), dtype=np.float32) / 255.0
    os.remove(crop)
    n = img.shape[0] // 512
    model = seg.ResUNet().to(device)
    model.load_state_dict(torch.load(
        os.path.join(AOI, "tiles", "_run", "resunet_consensus_best.pt"),
        map_location=device))
    model.eval()
    mean = torch.tensor(seg.IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(seg.IMAGENET_STD).view(1, 3, 1, 1).to(device)
    pred = np.zeros((n * 256, n * 256), dtype=np.float32)
    with torch.no_grad():
        for r in range(n):
            x = torch.stack([
                torch.nn.functional.interpolate(
                    torch.from_numpy(img[r*512:(r+1)*512, c*512:(c+1)*512]
                                     ).permute(2, 0, 1)[None],
                    size=(256, 256), mode="bilinear")[0]
                for c in range(n)]).to(device)
            y = torch.sigmoid(model((x - mean) / std))[:, 0].cpu().numpy()
            for c in range(n):
                pred[r*256:(r+1)*256, c*256:(c+1)*256] = y[c]
    return img, pred


def main():
    os.makedirs(HYDRO, exist_ok=True)
    hand_path = build_aoi_hand()
    Image.MAX_IMAGE_PIXELS = None
    img, prob = predict_builtup()
    built = prob > 0.5
    side = built.shape[0]                                     # 10 cm grid
    hand = np.asarray(Image.open(hand_path).resize((side, side), Image.BILINEAR))

    px_ha = 0.1 * 0.1 / 1e4                                   # ha per 10 cm px
    classes = [("severe (HAND < 2 m)", hand < 2),
               ("moderate (2-5 m)", (hand >= 2) & (hand < 5)),
               ("marginal (>= 5 m)", hand >= 5)]
    lines = ["class,builtup_ha,share_of_builtup"]
    tot = built.sum()
    print(f"AOI built-up total: {tot*px_ha:.1f} ha")
    for name, m in classes:
        a = (built & m).sum()
        print(f"  {name:22s} {a*px_ha:6.1f} ha  ({a/tot:5.1%} of built-up)")
        lines.append(f"{name},{a*px_ha:.2f},{a/tot:.4f}")
    open(OUTCSV, "w").write("\n".join(lines) + "\n")

    corridor_figure()
    print(f"wrote {FIG} and {OUTCSV}")


# 2015-06-03 flooded sites (ReliefWeb FL-2015-000065-GHA) + high-ground controls.
SITES = [("Old Fadama", 5.5475, -0.2226, True),
         ("Circle", 5.5715, -0.2087, True),
         ("Alajo", 5.5845, -0.2145, True),
         ("Kaneshie", 5.5651, -0.2286, True),
         ("Airport res.", 5.6050, -0.1780, False),
         ("Legon", 5.6510, -0.1870, False)]


def corridor_figure():
    """Basin-corridor HAND-class map: within the AOI itself HAND is uniformly
    ~0 m (the whole settlement is floodplain), so the informative rendering is
    the lower-Odaw corridor with the AOI and the 2015 flood sites in context."""
    from PIL import ImageDraw
    W_, S_, E_, N_ = -0.30, 5.52, -0.15, 5.68           # corridor window (WGS84)
    grids = {}
    for name in ("hand", "streams"):
        dst = os.path.join(HYDRO, f"_cor_{name}.tif")
        subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                        "-ot", "Float32", "-co", "PROFILE=BASELINE",
                        "-te", str(W_), str(S_), str(E_), str(N_),
                        "-te_srs", "EPSG:4326", "-t_srs", "EPSG:4326",
                        "-tr", "0.00027", "0.00027",
                        "-r", "near" if name == "streams" else "bilinear",
                        os.path.join(HYDRO, f"{name}.tif"), dst], check=True)
        grids[name] = np.asarray(Image.open(dst), dtype=np.float32)
        os.remove(dst)
    h, s = grids["hand"], grids["streams"]
    rgb = np.zeros((*h.shape, 3), np.uint8)
    rgb[h >= 5] = (235, 235, 228)                        # marginal
    rgb[(h >= 2) & (h < 5)] = (250, 200, 120)            # moderate
    rgb[h < 2] = (228, 110, 90)                          # severe
    rgb[s > 0] = (40, 70, 200)                           # drainage network
    scale = 3
    im = Image.fromarray(rgb).resize((rgb.shape[1]*scale, rgb.shape[0]*scale),
                                     Image.NEAREST)
    d = ImageDraw.Draw(im)

    def px(lat, lon):
        return ((lon - W_) / 0.00027 * scale / 1, (N_ - lat) / 0.00027 * scale)

    # AOI rectangle (scripts/05 window, WGS84 approx)
    x0, y0 = px(5.5520, -0.2273); x1, y1 = px(5.5428, -0.2180)
    d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=3)
    for name, lat, lon, flooded in SITES:
        x, y = px(lat, lon)
        col = (140, 0, 0) if flooded else (0, 90, 0)
        d.ellipse([x-7, y-7, x+7, y+7], fill=col, outline=(255, 255, 255), width=2)
        d.text((x+10, y-6), name, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    im.save(FIG)


if __name__ == "__main__":
    main()
