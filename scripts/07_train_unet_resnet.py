#!/usr/bin/env python3
"""
Stage-2 (improved) — U-Net with a PRETRAINED ResNet-34 encoder (transfer learning).

The from-scratch baseline (scripts/06) underfit: with few, label-noisy tiles it
produced coarse blobs. Here the encoder is initialised from ImageNet weights, so it
already encodes edges, corners, and textures; the decoder learns to turn those into
building masks. On small datasets this is typically the single largest quality gain.

Differences vs scripts/06: ResNet-34 encoder, ImageNet input normalisation, a cosine
learning-rate schedule, and a lower fine-tuning learning rate.

Usage:
    .venv/bin/python scripts/07_train_unet_resnet.py --epochs 35
"""
import os, glob, random, argparse
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
# 1. DATA — pretrained encoders expect ImageNet-normalised inputs.
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
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5: img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            if random.random() < 0.5: img, msk = img[::-1].copy(), msk[::-1].copy()
        img = (img - IMAGENET_MEAN) / IMAGENET_STD                # normalise for the encoder
        img = torch.from_numpy(img).permute(2, 0, 1)
        msk = torch.from_numpy(msk).unsqueeze(0)
        return img, msk


def denorm(img_chw):
    """Undo ImageNet normalisation for display."""
    x = img_chw.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. MODEL — ResNet-34 encoder (pretrained) + a U-Net decoder with skip links.
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
# 3-4. LOSS + METRIC (identical to the baseline, for a fair comparison).
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
# 5. TRAINING LOOP — cosine LR schedule, best-checkpoint on val IoU.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
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

    model = ResUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
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
            torch.save(model.state_dict(), os.path.join(OUT, "resunet_best.pt"))

    print(f"best val IoU = {best:.3f}")
    model.load_state_dict(torch.load(os.path.join(OUT, "resunet_best.pt"), map_location=device))
    save_predictions(model, val_ds, device, os.path.join(FIGDIR, "unet_predictions_resnet.png"))


if __name__ == "__main__":
    main()
