#!/usr/bin/env python3
"""
Stage-2 (consensus) — ResNet-34 U-Net trained on cross-source VERIFIED labels.

The label-noise audit (scripts/12) showed the OSM supervision is the accuracy
bottleneck. Rather than hand-annotating, labels are verified by consensus of
two independent sources: pixels where OSM and Google Open Buildings v3 agree
(both building, or both background) are trusted; pixels where they disagree
(~23% — per-building offsets, merged shacks, satellite under-detection) are
marked IGNORE (value 127 in masks_consensus/) and excluded from both the loss
and the metric. Consensus quality was verified visually against the imagery
on sampled tiles before adoption.

Training recipe otherwise mirrors the v2 baseline (scripts/07): plain
resize-256 inputs, flip augmentation, Adam + cosine. Two metrics are reported:
  * verified IoU — on consensus pixels only (the honest benchmark; note it
    excludes the hardest boundary pixels by construction);
  * legacy IoU  — against the raw OSM masks at thr 0.5, for continuity with
    the 0.579 (scripts/07) and 0.593 (scripts/10) numbers.

Prerequisites: the full-crop label rasters from scripts/11+12 (tiles/
masks_consensus/ is built automatically from them on first run).

Usage:
    .venv/bin/python scripts/13_train_unet_consensus.py --epochs 40
"""
import os, glob, random, argparse, importlib.util
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(HERE, "..", "accra_flood", "oldfadama", "tiles")
OUT = os.path.join(TILES, "_run")
FIGDIR = os.path.join(HERE, "..", "docs", "figures")

# model definition + normalisation constants shared with scripts/09
spec = importlib.util.spec_from_file_location("seg09", os.path.join(HERE, "09_train_unet_v3.py"))
seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)
IMAGENET_MEAN, IMAGENET_STD = seg.IMAGENET_MEAN, seg.IMAGENET_STD

AUDIT = os.path.join(HERE, "..", "accra_flood", "oldfadama", "open_buildings", "_audit")
CONSENSUS = os.path.join(TILES, "masks_consensus")


def build_consensus_masks():
    """Slice per-tile consensus masks (0=bg, 255=building, 127=ignore) from the
    full-crop OSM and Open Buildings rasters produced by scripts/12. Tile ids
    encode their (x, y) pixel offset within the scripts/05 crop window."""
    Image.MAX_IMAGE_PIXELS = None
    osm = np.asarray(Image.open(os.path.join(AUDIT, "mask_osm.tif"))) > 127
    gob = np.asarray(Image.open(os.path.join(AUDIT, "mask_gob.tif"))) > 127
    os.makedirs(CONSENSUS, exist_ok=True)
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TILES, "images", "*.png")))
    ig = []
    for name in ids:
        _, xs, ys = name[:-4].split("_")
        x, y = int(xs), int(ys)
        o, g = osm[y:y + 512, x:x + 512], gob[y:y + 512, x:x + 512]
        m = np.where(o == g, np.where(o, 255, 0), 127).astype(np.uint8)
        ig.append((m == 127).mean())
        Image.fromarray(m).save(os.path.join(CONSENSUS, name))
    print(f"built {len(ids)} consensus masks  (mean ignore fraction {np.mean(ig):.3f})")


class ConsensusDataset(Dataset):
    """Returns (image, target, valid) where valid=0 marks ignore pixels."""

    def __init__(self, ids, size, augment=False):
        self.ids, self.size, self.augment = ids, size, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = Image.open(os.path.join(TILES, "images", name)).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        msk = Image.open(os.path.join(TILES, "masks_consensus", name)).resize(
            (self.size, self.size), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        m = np.asarray(msk)
        tgt = (m == 255).astype(np.float32)
        valid = (m != 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img, tgt, valid = img[:, ::-1].copy(), tgt[:, ::-1].copy(), valid[:, ::-1].copy()
            if random.random() < 0.5:
                img, tgt, valid = img[::-1].copy(), tgt[::-1].copy(), valid[::-1].copy()
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(tgt).unsqueeze(0),
                torch.from_numpy(valid).unsqueeze(0))


bce = torch.nn.BCEWithLogitsLoss(reduction="none")

def criterion(logits, target, valid, eps=1.0):
    """BCE + Dice, both restricted to verified pixels."""
    l_bce = (bce(logits, target) * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    num = 2 * (p * t).sum((1, 2, 3)) + eps
    den = (p + t).sum((1, 2, 3)) + eps
    return l_bce + (1 - num / den).mean()


@torch.no_grad()
def masked_iou(logits, target, valid, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float() * valid
    t = target * valid
    inter = (p * t).sum((1, 2, 3))
    union = ((p + t) > 0).float().sum((1, 2, 3))
    return ((inter + 1) / (union + 1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not os.path.isdir(CONSENSUS):
        build_consensus_masks()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    # Canonical split — identical logic/seed to scripts/06-10.
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TILES, "images", "*.png"))
                 if os.path.exists(os.path.join(TILES, "masks", os.path.basename(p))))
    random.shuffle(ids)
    k = int(len(ids) * 0.8)
    train_ids, val_ids = ids[:k], ids[k:]
    print(f"device={device}  tiles={len(ids)}  train={len(train_ids)}  val={len(val_ids)}")

    train_dl = DataLoader(ConsensusDataset(train_ids, a.size, augment=True),
                          batch_size=a.batch, shuffle=True, num_workers=0)
    val_ds = ConsensusDataset(val_ids, a.size, augment=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    model = seg.ResUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for img, tgt, valid in train_dl:
            img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
            opt.zero_grad()
            loss = criterion(model(img), tgt, valid)
            loss.backward(); opt.step()
            tot += loss.item() * img.size(0)
        sched.step()
        model.eval(); vi = 0.0
        with torch.no_grad():
            for img, tgt, valid in val_dl:
                img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
                vi += masked_iou(model(img), tgt, valid) * img.size(0)
        vi /= len(val_ds)
        print(f"epoch {ep:2d}/{a.epochs}  train_loss={tot/len(train_ids):.3f}  verified_val_IoU={vi:.3f}", flush=True)
        if vi > best:
            best = vi
            torch.save(model.state_dict(), os.path.join(OUT, "resunet_consensus_best.pt"))

    print(f"best verified val IoU = {best:.3f}")
    model.load_state_dict(torch.load(os.path.join(OUT, "resunet_consensus_best.pt"), map_location=device))
    model.eval()

    # Final report: verified IoU with TTA, plus the legacy OSM-mask protocol.
    with torch.no_grad():
        vi = 0.0
        for img, tgt, valid in val_dl:
            img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
            vi += masked_iou(seg.tta_logits(model, img), tgt, valid) * img.size(0)
        print(f"verified val IoU with 4-way flip TTA = {vi/len(val_ds):.4f}")

        legacy = seg.TileDataset(val_ids, a.size, augment=False)
        li = 0.0
        for i in range(len(legacy)):
            img, msk = legacy[i]
            logits = seg.tta_logits(model, img[None].to(device))
            li += seg.iou_dice(logits, msk[None].to(device))[0]
        print(f"legacy OSM-mask val IoU (TTA, thr 0.5) = {li/len(legacy):.4f}")
    seg.save_predictions(model, legacy, device, os.path.join(FIGDIR, "unet_predictions_consensus.png"))


if __name__ == "__main__":
    main()
