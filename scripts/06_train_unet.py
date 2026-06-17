#!/usr/bin/env python3
"""
Stage-2 — train a U-Net to segment building footprints from 5 cm orthoimagery.

Reads the co-registered (image, mask) tiles produced by scripts/05 and learns the
mapping image -> building mask. Written to be read top-to-bottom as the six pieces
of a supervised-segmentation pipeline: data, model, loss, metric, training loop,
and prediction visualisation.

Usage:
    .venv/bin/python scripts/06_train_unet.py --epochs 25
"""
import os, glob, random, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(HERE, "..", "accra_flood", "oldfadama", "tiles")
OUT = os.path.join(TILES, "_run")           # model + scratch (gitignored under tiles/)
FIGDIR = os.path.join(HERE, "..", "docs", "figures")


# ---------------------------------------------------------------------------
# 1. DATA — pair each image tile with its mask, split into train / validation.
# ---------------------------------------------------------------------------
class TileDataset(Dataset):
    def __init__(self, ids, size, augment=False):
        self.ids, self.size, self.augment = ids, size, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = Image.open(os.path.join(TILES, "images", name)).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        msk = Image.open(os.path.join(TILES, "masks", name)).convert("L").resize(
            (self.size, self.size), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0          # [H,W,3] in 0..1
        msk = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)  # [H,W] binary
        if self.augment:                                          # cheap flips = free data
            if random.random() < 0.5: img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            if random.random() < 0.5: img, msk = img[::-1].copy(), msk[::-1].copy()
        img = torch.from_numpy(img).permute(2, 0, 1)              # -> [3,H,W]
        msk = torch.from_numpy(msk).unsqueeze(0)                  # -> [1,H,W]
        return img, msk


# ---------------------------------------------------------------------------
# 2. MODEL — the U-Net: encoder (shrink) -> bottleneck -> decoder (expand),
#    with skip connections carrying fine detail across for sharp edges.
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1, self.d2 = DoubleConv(3, base), DoubleConv(base, base * 2)
        self.d3, self.d4 = DoubleConv(base * 2, base * 4), DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2); self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2);  self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2);  self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2);      self.u1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)                        # 1 logit per pixel

    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        b = self.bott(self.pool(c4))                             # bottleneck
        x = self.u4(torch.cat([self.up4(b), c4], 1))            # skip-connect c4
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)


# ---------------------------------------------------------------------------
# 3. LOSS — how wrong a prediction is. BCE (per-pixel) + Dice (overlap-aware,
#    robust to the building/background imbalance).
# ---------------------------------------------------------------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = (p + target).sum((1, 2, 3)) + eps
    return (1 - num / den).mean()

def criterion(logits, target):
    return bce(logits, target) + dice_loss(logits, target)


# ---------------------------------------------------------------------------
# 4. METRIC — Intersection-over-Union and Dice on a held-out tile (0..1, higher better).
# ---------------------------------------------------------------------------
@torch.no_grad()
def iou_dice(logits, target, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((1, 2, 3))
    union = ((p + target) > 0).float().sum((1, 2, 3))
    iou = ((inter + 1) / (union + 1)).mean().item()
    dice = ((2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean().item()
    return iou, dice


# ---------------------------------------------------------------------------
# 6. VISUALISE — save a grid of [image | truth | prediction] for eyeball checks.
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_predictions(model, ds, device, path, n=6):
    model.eval()
    rows = []
    for i in range(min(n, len(ds))):
        img, msk = ds[i]
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        img_u = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        truth = np.repeat((msk[0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
        predu = np.repeat(((pred > 0.5) * 255).astype(np.uint8)[..., None], 3, axis=2)
        gap = np.full((img_u.shape[0], 4, 3), 64, np.uint8)
        rows.append(np.concatenate([img_u, gap, truth, gap, predu], axis=1))
    grid = np.concatenate([np.concatenate([r, np.full((4, r.shape[1], 3), 64, np.uint8)]) for r in rows])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)
    print(f"  wrote {path}  (columns: image | truth | prediction)")


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP — show batches, measure loss, nudge weights, repeat (epochs);
#    validate on unseen tiles each epoch and keep the best checkpoint.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TILES, "images", "*.png"))
                 if os.path.exists(os.path.join(TILES, "masks", os.path.basename(p))))
    random.shuffle(ids)
    k = int(len(ids) * 0.8)
    train_ids, val_ids = ids[:k], ids[k:]
    print(f"device={device}  tiles={len(ids)}  train={len(train_ids)}  val={len(val_ids)}")

    train_dl = DataLoader(TileDataset(train_ids, a.size, augment=True), batch_size=a.batch,
                          shuffle=True, num_workers=0)
    val_ds = TileDataset(val_ids, a.size, augment=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    model = UNet(base=a.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params/1e6:.2f}M")

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for img, msk in train_dl:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            loss = criterion(model(img), msk)
            loss.backward(); opt.step()
            tot += loss.item() * img.size(0)
        model.eval(); vi = vd = 0.0
        with torch.no_grad():
            for img, msk in val_dl:
                img, msk = img.to(device), msk.to(device)
                i_, d_ = iou_dice(model(img), msk)
                vi += i_ * img.size(0); vd += d_ * img.size(0)
        vi /= len(val_ds); vd /= len(val_ds)
        print(f"epoch {ep:2d}/{a.epochs}  train_loss={tot/len(train_ids):.3f}  val_IoU={vi:.3f}  val_Dice={vd:.3f}")
        if vi > best:
            best = vi
            torch.save(model.state_dict(), os.path.join(OUT, "unet_best.pt"))

    print(f"best val IoU = {best:.3f}")
    # reload best and dump a prediction grid for inspection
    model.load_state_dict(torch.load(os.path.join(OUT, "unet_best.pt"), map_location=device))
    save_predictions(model, val_ds, device, os.path.join(FIGDIR, "unet_predictions.png"))


if __name__ == "__main__":
    main()
