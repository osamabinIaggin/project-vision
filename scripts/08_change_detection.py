#!/usr/bin/env python3
"""
Path-A application — apply the trained segmentation model to the 2020 and 2024
orthomosaics and detect NEW built-up encroachment between the two epochs.

The model yields coarse built-up masks (its honest capability). We do not need
crisp footprints here: the question is "where did built-up area appear, on or
beside the drainage network, between 2020 and 2024?" — which a region-level mask
answers. Inputs are the common-grid PNGs produced by scripts/08_change_detection.sh.

Usage:
    .venv/bin/python scripts/08_change_detection.py
"""
import os, importlib.util
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "accra_flood", "oldfadama", "tiles", "_run", "change")
FIG = os.path.join(HERE, "..", "docs", "figures", "change_detection_oldfadama.png")
CKPT = os.path.join(HERE, "..", "accra_flood", "oldfadama", "tiles", "_run", "resunet_best.pt")

# import the model definition + normalisation constants from scripts/07
spec = importlib.util.spec_from_file_location("seg07", os.path.join(HERE, "07_train_unet_resnet.py"))
seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)
MEAN, STD = seg.IMAGENET_MEAN, seg.IMAGENET_STD

PATCH_M = 512.0          # patch side length in metres (matches the .sh extent)
ANALYSIS = 2048          # downsampled resolution for analysis + figure
M2_PER_PX = (PATCH_M / ANALYSIS) ** 2


def predict_builtup(img_pil, model, device, win=512, size=256):
    """Sliding-window inference -> full-resolution built-up probability map."""
    arr = np.asarray(img_pil.convert("RGB"), np.float32) / 255.0
    H, Wd = arr.shape[:2]
    out = np.zeros((H, Wd), np.float32)
    for y in range(0, H, win):
        for x in range(0, Wd, win):
            tile = arr[y:y + win, x:x + win]
            th, tw = tile.shape[:2]
            t = Image.fromarray((tile * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
            tn = (np.asarray(t, np.float32) / 255.0 - MEAN) / STD
            tt = torch.from_numpy(tn).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(tt))[0, 0].cpu().numpy()
            p = np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR),
                           np.float32) / 255.0
            out[y:y + win, x:x + win] = p
    return out


def down(a, n=ANALYSIS, nearest=True):
    im = Image.fromarray(a)
    return np.asarray(im.resize((n, n), Image.NEAREST if nearest else Image.BILINEAR))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = seg.ResUNet().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()
    print(f"device={device}  loaded {os.path.basename(CKPT)}")

    masks = {}
    for y in ("2020", "2024"):
        img = Image.open(os.path.join(W, f"patch_{y}.png"))
        prob = predict_builtup(img, model, device)
        masks[y] = down((prob > 0.5).astype(np.uint8))            # {0,1} at ANALYSIS res
        print(f"  predicted built-up for {y}")

    drains = down((np.asarray(Image.open(os.path.join(W, "drains.png")).convert("L")) > 127).astype(np.uint8))
    # dilate the drainage network into a ~6 m "near-drain" corridor
    r = int(round(6.0 / (PATCH_M / ANALYSIS)))
    dr_t = torch.from_numpy(drains.astype(np.float32))[None, None]
    nearzone = (F.max_pool2d(dr_t, 2 * r + 1, 1, r)[0, 0].numpy() > 0).astype(np.uint8)

    m20, m24 = masks["2020"], masks["2024"]
    new_builtup = ((m24 == 1) & (m20 == 0)).astype(np.uint8)      # appeared by 2024
    new_on_drain = (new_builtup & nearzone).astype(np.uint8)

    a20 = m20.sum() * M2_PER_PX
    a24 = m24.sum() * M2_PER_PX
    anew = new_builtup.sum() * M2_PER_PX
    adr = new_on_drain.sum() * M2_PER_PX
    print("\n=== Built-up change, Old Fadama patch (model-derived) ===")
    print(f"  built-up 2020:                {a20/1e4:6.2f} ha")
    print(f"  built-up 2024:                {a24/1e4:6.2f} ha")
    print(f"  NEW built-up (2024 not 2020): {anew/1e4:6.2f} ha")
    print(f"  ...of which in drain corridor:{adr/1e4:6.2f} ha  ({100*adr/max(anew,1):.0f}% of new)")

    # ---- figure: 2024 image | 2020 built-up | 2024 built-up | new+drains on 2024 ----
    def panel(a):
        return down(a, 512, nearest=False) if a.ndim == 3 else np.repeat(down(a * 255, 512)[..., None], 3, 2)
    img24 = np.asarray(Image.open(os.path.join(W, "patch_2024.png")).convert("RGB"))
    p_img = down(img24, 512, nearest=False)
    p_20 = np.repeat(down(m20 * 255, 512)[..., None], 3, 2)
    p_24 = np.repeat(down(m24 * 255, 512)[..., None], 3, 2)
    over = p_img.copy()
    nd = down(new_builtup * 255, 512); dz = down(drains * 255, 512)
    over[dz > 127] = [40, 200, 255]                               # drains cyan
    over[nd > 127] = [255, 40, 40]                                # new built-up red
    gap = np.full((512, 6, 3), 64, np.uint8)
    grid = np.concatenate([p_img, gap, p_20, gap, p_24, gap, over], axis=1)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    Image.fromarray(grid.astype(np.uint8)).save(FIG)
    print(f"\nwrote {FIG}\n  panels: 2024 image | 2020 built-up | 2024 built-up | new(red)+drains(cyan)")


if __name__ == "__main__":
    main()
