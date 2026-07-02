#!/usr/bin/env python3
"""
Stage-2 recipe ablations (v3/v4) — a documented NEGATIVE result.

The v2 model (scripts/07) reached val IoU 0.579. This script tests whether a
stronger optimisation recipe moves that number, holding the validation split
and protocol byte-identical so IoU values are directly comparable:
  * multi-scale random crops at native 5 cm resolution (v2 downsampled every
    512-px tile to 256 px), plus 90-degree rotations and photometric jitter
    (--light-aug disables the crops/jitter, keeping geometric augmentation);
  * differential learning rates — ImageNet encoder at lr/10, decoder at lr;
  * AdamW + linear-warmup cosine schedule over a longer budget;
  * 4-way flip test-time augmentation reported alongside the raw metric.

Outcome (2026-07): the full recipe ("v3") scores 0.561 and the conservative
variant ("v4", --light-aug --lr 2e-4) scores 0.566 — both BELOW the 0.579
baseline. Train loss falls monotonically in every run while validation stays
in a flat 0.54-0.57 band: extra capacity utilisation only memorises label
noise. Together with the label-source audit (scripts/12) this establishes
that Stage-2 accuracy is bounded by OSM label quality, not by the training
recipe; the productive inference-time gain is quantified in scripts/10.

Usage:
    .venv/bin/python scripts/09_train_unet_v3.py --epochs 60                      # v3
    .venv/bin/python scripts/09_train_unet_v3.py --epochs 60 --lr 2e-4 \
        --light-aug --tag v4                                                      # v4
"""
import os, glob, math, random, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(HERE, "..", "accra_flood", "oldfadama", "tiles")
OUT = os.path.join(TILES, "_run")
FIGDIR = os.path.join(HERE, "..", "docs", "figures")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. DATA — train tiles are sampled at native resolution; val is unchanged.
# ---------------------------------------------------------------------------
class TileDataset(Dataset):
    def __init__(self, ids, size, augment=False, light=False):
        self.ids, self.size, self.augment, self.light = ids, size, augment, light

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = Image.open(os.path.join(TILES, "images", name)).convert("RGB")
        msk = Image.open(os.path.join(TILES, "masks", name)).convert("L")

        if self.augment and not self.light:
            # Multi-scale crop: a random square window between the eval scale
            # (full tile -> 256, i.e. 10 cm) and native resolution (256-px
            # window, 5 cm), then resized to the training size.
            w, h = img.size
            c = random.randint(self.size, min(w, h))
            x, y = random.randint(0, w - c), random.randint(0, h - c)
            img = img.crop((x, y, x + c, y + c)).resize((self.size, self.size), Image.BILINEAR)
            msk = msk.crop((x, y, x + c, y + c)).resize((self.size, self.size), Image.NEAREST)
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)
            msk = msk.resize((self.size, self.size), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            k = random.randint(0, 3)                        # 90-degree rotations
            if k: img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()
            if random.random() < 0.5: img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            if random.random() < 0.5: img, msk = img[::-1].copy(), msk[::-1].copy()
            if not self.light:
                # Photometric jitter: global brightness/contrast + per-channel gain.
                img = img * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06)
                img = (img - img.mean()) * random.uniform(0.9, 1.1) + img.mean()
                img = img * np.random.uniform(0.95, 1.05, size=3).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img).permute(2, 0, 1)
        msk = torch.from_numpy(msk).unsqueeze(0)
        return img, msk


def denorm(img_chw):
    """Undo ImageNet normalisation for display."""
    x = img_chw.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. MODEL — identical ResNet-34 U-Net to scripts/07.
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = DoubleConv(cout + skip, cout)

    def forward(self, x, s=None):
        x = self.up(x)
        if s is not None:
            x = torch.cat([x, s], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        enc = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 64, /2
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 64)
        self.up0 = Up(64, 0, 32)                                  # final /2 to full res
        self.head = nn.Conv2d(32, 1, 1)

    def encoder_parameters(self):
        for m in (self.stem, self.l1, self.l2, self.l3, self.l4):
            yield from m.parameters()

    def decoder_parameters(self):
        for m in (self.up4, self.up3, self.up2, self.up1, self.up0, self.head):
            yield from m.parameters()

    def forward(self, x):
        x0 = self.stem(x)                  # 64,  /2
        x1 = self.l1(self.pool(x0))        # 64,  /4
        x2 = self.l2(x1)                   # 128, /8
        x3 = self.l3(x2)                   # 256, /16
        x4 = self.l4(x3)                   # 512, /32  (bottleneck)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.up0(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 3-4. LOSS + METRIC (identical to scripts/06-07, for a fair comparison).
# ---------------------------------------------------------------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = (p + target).sum((1, 2, 3)) + eps
    return (1 - num / den).mean()

def criterion(logits, target):
    return bce(logits, target) + dice_loss(logits, target)

@torch.no_grad()
def iou_dice(logits, target, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((1, 2, 3))
    union = ((p + target) > 0).float().sum((1, 2, 3))
    iou = ((inter + 1) / (union + 1)).mean().item()
    dice = ((2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean().item()
    return iou, dice


@torch.no_grad()
def tta_logits(model, img):
    """Average logits over the 4-way flip group (identity, h, v, hv)."""
    outs = []
    for fh in (False, True):
        for fv in (False, True):
            x = img
            if fh: x = torch.flip(x, dims=[3])
            if fv: x = torch.flip(x, dims=[2])
            y = model(x)
            if fh: y = torch.flip(y, dims=[3])
            if fv: y = torch.flip(y, dims=[2])
            outs.append(y)
    return torch.stack(outs).mean(0)


@torch.no_grad()
def evaluate(model, dl, device, n_items, tta=False):
    model.eval(); vi = vd = 0.0
    for img, msk in dl:
        img, msk = img.to(device), msk.to(device)
        logits = tta_logits(model, img) if tta else model(img)
        i_, d_ = iou_dice(logits, msk)
        vi += i_ * img.size(0); vd += d_ * img.size(0)
    return vi / n_items, vd / n_items


@torch.no_grad()
def save_predictions(model, ds, device, path, n=6):
    model.eval()
    rows = []
    for i in range(min(n, len(ds))):
        img, msk = ds[i]
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        img_u = denorm(img)
        truth = np.repeat((msk[0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
        predu = np.repeat(((pred > 0.5) * 255).astype(np.uint8)[..., None], 3, axis=2)
        gap = np.full((img_u.shape[0], 4, 3), 64, np.uint8)
        rows.append(np.concatenate([img_u, gap, truth, gap, predu], axis=1))
    grid = np.concatenate([np.concatenate([r, np.full((4, r.shape[1], 3), 64, np.uint8)]) for r in rows])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)
    print(f"  wrote {path}  (columns: image | truth | prediction)")


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP — differential LRs, warmup + cosine, best-IoU checkpoint.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)          # decoder rate; encoder = lr/10
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--light-aug", action="store_true",
                    help="geometric augmentation only (rot90+flips at the eval scale); "
                         "disables multi-scale crops and photometric jitter")
    ap.add_argument("--tag", default="v3", help="checkpoint/figure suffix")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    # Split logic is byte-identical to scripts/06-07 so val tiles match exactly.
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TILES, "images", "*.png"))
                 if os.path.exists(os.path.join(TILES, "masks", os.path.basename(p))))
    random.shuffle(ids)
    k = int(len(ids) * 0.8)
    train_ids, val_ids = ids[:k], ids[k:]
    print(f"device={device}  tiles={len(ids)}  train={len(train_ids)}  val={len(val_ids)}")

    train_dl = DataLoader(TileDataset(train_ids, a.size, augment=True, light=a.light_aug),
                          batch_size=a.batch, shuffle=True, num_workers=0)
    val_ds = TileDataset(val_ids, a.size, augment=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    model = ResUNet().to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": a.lr / 10},
        {"params": model.decoder_parameters(), "lr": a.lr},
    ], weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < a.warmup:
            return (ep + 1) / a.warmup
        t = (ep - a.warmup) / max(1, a.epochs - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"ResNet34-U-Net params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for img, msk in train_dl:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            loss = criterion(model(img), msk)
            loss.backward(); opt.step()
            tot += loss.item() * img.size(0)
        sched.step()
        vi, vd = evaluate(model, val_dl, device, len(val_ds))
        print(f"epoch {ep:2d}/{a.epochs}  train_loss={tot/len(train_ids):.3f}  val_IoU={vi:.3f}  val_Dice={vd:.3f}", flush=True)
        if vi > best:
            best = vi
            torch.save(model.state_dict(), os.path.join(OUT, f"resunet_{a.tag}_best.pt"))

    print(f"best val IoU = {best:.3f}")
    model.load_state_dict(torch.load(os.path.join(OUT, f"resunet_{a.tag}_best.pt"), map_location=device))
    ti, td = evaluate(model, val_dl, device, len(val_ds), tta=True)
    print(f"with 4-way flip TTA: val_IoU={ti:.3f}  val_Dice={td:.3f}")
    save_predictions(model, val_ds, device, os.path.join(FIGDIR, f"unet_predictions_{a.tag}.png"))


if __name__ == "__main__":
    main()
