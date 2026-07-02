#!/usr/bin/env python3
"""
Stage-2 inference-time improvements — flip TTA + decision-threshold calibration.

Retraining with stronger recipes does not beat the v2 model (see scripts/09),
because accuracy is label-noise-bounded. What DOES help, at zero training cost:

  * 4-way flip test-time augmentation — logits are averaged over the dihedral
    flip group (identity, horizontal, vertical, both), cancelling orientation-
    dependent errors;
  * decision-threshold calibration — the default 0.5 is not optimal under
    noisy labels; a sweep locates the IoU-maximising operating point.

On the canonical 217-tile validation split this lifts the v2 checkpoint from
IoU 0.579 (raw, thr 0.5) to 0.593 (TTA, thr 0.45).

Usage:
    .venv/bin/python scripts/10_eval_tta.py [--ckpt resunet_best.pt]
"""
import os, glob, random, argparse, importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(HERE, "..", "accra_flood", "oldfadama", "tiles")

# model definition, dataset, and TTA helper are shared with scripts/09
spec = importlib.util.spec_from_file_location("seg09", os.path.join(HERE, "09_train_unet_v3.py"))
seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="resunet_best.pt", help="checkpoint file in tiles/_run/")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # Reconstruct the canonical split (identical logic/seed to scripts/06-09).
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TILES, "images", "*.png"))
                 if os.path.exists(os.path.join(TILES, "masks", os.path.basename(p))))
    random.shuffle(ids)
    val_ids = ids[int(len(ids) * 0.8):]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ds = seg.TileDataset(val_ids, a.size, augment=False)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False)

    model = seg.ResUNet().to(device)
    model.load_state_dict(torch.load(os.path.join(TILES, "_run", a.ckpt), map_location=device))
    model.eval()
    print(f"ckpt={a.ckpt}  val_tiles={len(ds)}  device={device}")

    with torch.no_grad():
        for tta in (False, True):
            P, M = [], []
            for img, msk in dl:
                img = img.to(device)
                logits = seg.tta_logits(model, img) if tta else model(img)
                P.append(torch.sigmoid(logits).cpu()); M.append(msk)
            P, M = torch.cat(P), torch.cat(M)
            print("4-way flip TTA:" if tta else "raw:")
            for thr in (0.35, 0.40, 0.45, 0.50, 0.55):
                p = (P > thr).float()
                inter = (p * M).sum((1, 2, 3))
                union = ((p + M) > 0).float().sum((1, 2, 3))
                dice = ((2 * inter + 1) / (p.sum((1, 2, 3)) + M.sum((1, 2, 3)) + 1)).mean()
                print(f"  thr={thr:.2f}  IoU={((inter + 1) / (union + 1)).mean():.4f}  Dice={dice:.4f}")


if __name__ == "__main__":
    main()
