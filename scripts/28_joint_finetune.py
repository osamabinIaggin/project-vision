#!/usr/bin/env python3
"""
Stage-2 (joint) — leave-one-site-out fine-tuning across the five AOIs.

scripts/27 established that the Old Fadama checkpoint degrades on unseen
communities (0.798 -> 0.523 pooled), and that Alogboshie's outlier behaviour is a
real domain difference rather than a preprocessing artefact. The open question is
whether that gap is LEARNABLE — closed by showing the model other settlements —
or whether it reflects the same label-granularity ceiling that bounded Stage 2 at
Old Fadama.

Protocol: hold one site out entirely, fine-tune the existing consensus checkpoint
on the remaining four, and evaluate on the held-out site under the identical
verified-pixel metric. This measures generalisation to an unseen settlement
directly rather than inferring it, and holding the site out whole (not a random
tile split) is what makes it a test of transfer instead of of interpolation —
tiles from one 512 m window are far too correlated for a random split to mean
anything across sites.

Every run reports three numbers for the held-out site: the zero-shot score from
scripts/27, the fine-tuned score, and the label-noise floor. The floor is what
tells you whether a given result is good; the zero-shot score is what tells you
whether fine-tuning did anything.

Usage:
    .venv/bin/python scripts/28_joint_finetune.py --holdout alogboshie --epochs 12
"""
import os, glob, csv, json, random, argparse, importlib.util, collections
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CORPORA = {
    "oldfadama": os.path.join(ROOT, "accra_flood", "oldfadama", "tiles"),
    "middleodaw": os.path.join(ROOT, "accra_flood", "middleodaw", "tiles"),
}
CKPT_IN = os.path.join(CORPORA["oldfadama"], "_run", "resunet_consensus_best.pt")
OUTDIR = os.path.join(CORPORA["middleodaw"], "_run")
RESULTS = os.path.join(ROOT, "accra_flood", "output", "joint_finetune.csv")

spec = importlib.util.spec_from_file_location("seg", os.path.join(HERE, "09_train_unet_v3.py"))
seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)

# Zero-shot references from scripts/27, for the gain column.
ZERO_SHOT = {"oldfadama": 0.798, "akweteman": 0.641, "alajo": 0.603,
             "nima": 0.507, "alogboshie": 0.298}


def catalogue():
    """Every consensus-labelled tile in the project, tagged by site."""
    items = []
    for corpus, base in CORPORA.items():
        for p in sorted(glob.glob(os.path.join(base, "images", "*.png"))):
            name = os.path.basename(p)
            con = os.path.join(base, "masks_consensus", name)
            if not os.path.exists(con):
                continue
            head = name.split("_")[0]
            site = "oldfadama" if corpus == "oldfadama" else head
            items.append((site, base, name))
    return items


class Tiles(Dataset):
    def __init__(self, items, size, augment=False):
        self.items, self.size, self.augment = items, size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _site, base, name = self.items[i]
        img = Image.open(os.path.join(base, "images", name)).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        msk = Image.open(os.path.join(base, "masks_consensus", name)).resize(
            (self.size, self.size), Image.NEAREST)
        a = np.asarray(img, dtype=np.float32) / 255.0
        m = np.asarray(msk)
        tgt = (m == 255).astype(np.float32)
        valid = (m != 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                a, tgt, valid = a[:, ::-1].copy(), tgt[:, ::-1].copy(), valid[:, ::-1].copy()
            if random.random() < 0.5:
                a, tgt, valid = a[::-1].copy(), tgt[::-1].copy(), valid[::-1].copy()
        a = (a - seg.IMAGENET_MEAN) / seg.IMAGENET_STD
        return (torch.from_numpy(a).permute(2, 0, 1),
                torch.from_numpy(tgt).unsqueeze(0),
                torch.from_numpy(valid).unsqueeze(0))


bce = torch.nn.BCEWithLogitsLoss(reduction="none")


def criterion(logits, target, valid, eps=1.0):
    l = (bce(logits, target) * valid).sum() / valid.sum().clamp(min=1)
    p, t = torch.sigmoid(logits) * valid, target * valid
    num = 2 * (p * t).sum((1, 2, 3)) + eps
    den = (p + t).sum((1, 2, 3)) + eps
    return l + (1 - num / den).mean()


@torch.no_grad()
def evaluate(model, items, device, size, batch=8, tta=True):
    """Pooled and per-tile verified IoU over a set of tiles."""
    ds = Tiles(items, size, augment=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    inter = union = 0
    ious, pred_px, tgt_px, npx = [], 0, 0, 0
    model.eval()
    for img, tgt, valid in dl:
        img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
        logits = seg.tta_logits(model, img) if tta else model(img)
        p = (torch.sigmoid(logits) > 0.5).float() * valid
        t = tgt * valid
        i = (p * t).sum((1, 2, 3))
        u = ((p + t) > 0).float().sum((1, 2, 3))
        inter += i.sum().item(); union += u.sum().item()
        ious += ((i + 1) / (u + 1)).cpu().tolist()
        pred_px += (torch.sigmoid(logits) > 0.5).float().sum().item()
        tgt_px += tgt.sum().item(); npx += tgt.numel()
    return (inter / max(union, 1), float(np.mean(ious)),
            pred_px / max(npx, 1), tgt_px / max(npx, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True,
                    help="site held out entirely: oldfadama|alogboshie|akweteman|alajo|nima")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)   # fine-tune, not restart
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)

    items = catalogue()
    sites = sorted({s for s, _, _ in items})
    if a.holdout not in sites:
        raise SystemExit(f"unknown site {a.holdout}; have {sites}")
    train_items = [it for it in items if it[0] != a.holdout]
    held_items = [it for it in items if it[0] == a.holdout]
    counts = collections.Counter(s for s, _, _ in items)
    print(f"device={device}  corpus={len(items)} tiles over {len(sites)} sites")
    for s in sites:
        print(f"    {s:12s} {counts[s]:5d}" + ("   <- HELD OUT" if s == a.holdout else ""))

    # A held-in validation slice, to distinguish "learned the new sites" from
    # "forgot the old ones" if the held-out number moves.
    random.shuffle(train_items)
    k = int(len(train_items) * 0.9)
    fit_items, heldin_items = train_items[:k], train_items[k:]
    print(f"  fit={len(fit_items)}  held-in val={len(heldin_items)}  "
          f"held-out={len(held_items)}")

    model = seg.ResUNet().to(device)
    model.load_state_dict(torch.load(CKPT_IN, map_location=device))
    print(f"  initialised from {os.path.basename(CKPT_IN)}")

    base_pooled, base_tile, base_pred, base_tgt = evaluate(
        model, held_items, device, a.size)
    print(f"\n  zero-shot on {a.holdout}: pooled {base_pooled:.3f}  "
          f"per-tile {base_tile:.3f}  pred {base_pred:.1%} vs actual {base_tgt:.1%}")

    dl = DataLoader(Tiles(fit_items, a.size, augment=True), batch_size=a.batch,
                    shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, best_ep = -1.0, 0
    ckpt = os.path.join(OUTDIR, f"resunet_joint_holdout_{a.holdout}.pt")
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for img, tgt, valid in dl:
            img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
            opt.zero_grad()
            loss = criterion(model(img), tgt, valid)
            loss.backward(); opt.step()
            tot += loss.item() * img.size(0)
        sched.step()
        hi, _, _, _ = evaluate(model, heldin_items, device, a.size, tta=False)
        print(f"  epoch {ep:2d}/{a.epochs}  loss={tot/len(fit_items):.4f}  "
              f"held-in pooled IoU={hi:.3f}", flush=True)
        if hi > best:
            best, best_ep = hi, ep
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    ft_pooled, ft_tile, ft_pred, ft_tgt = evaluate(model, held_items, device, a.size)
    hi_pooled, hi_tile, _, _ = evaluate(model, heldin_items, device, a.size)

    print(f"\n--- leave-one-site-out: {a.holdout} ---")
    print(f"  selected epoch {best_ep} on held-in IoU {best:.3f}")
    print(f"  {'':22s} {'pooled':>8s} {'per-tile':>9s}")
    print(f"  {'zero-shot (scripts/27)':22s} {base_pooled:>8.3f} {base_tile:>9.3f}")
    print(f"  {'after fine-tuning':22s} {ft_pooled:>8.3f} {ft_tile:>9.3f}")
    print(f"  {'gain':22s} {ft_pooled-base_pooled:>+8.3f} {ft_tile-base_tile:>+9.3f}")
    print(f"  built-up predicted {ft_pred:.1%} vs actual {ft_tgt:.1%} "
          f"(was {base_pred:.1%})")
    print(f"  held-in sites after fine-tuning: pooled {hi_pooled:.3f} "
          f"(guards against catastrophic forgetting)")

    new = not os.path.exists(RESULTS)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["holdout", "n_held", "n_fit", "epochs", "lr",
                        "zeroshot_pooled", "finetuned_pooled", "gain_pooled",
                        "zeroshot_pertile", "finetuned_pertile",
                        "pred_frac", "actual_frac", "heldin_pooled"])
        w.writerow([a.holdout, len(held_items), len(fit_items), a.epochs, a.lr,
                    f"{base_pooled:.4f}", f"{ft_pooled:.4f}",
                    f"{ft_pooled-base_pooled:+.4f}",
                    f"{base_tile:.4f}", f"{ft_tile:.4f}",
                    f"{ft_pred:.4f}", f"{ft_tgt:.4f}", f"{hi_pooled:.4f}"])
    print(f"\nwrote {ckpt}\n      {RESULTS}")


if __name__ == "__main__":
    main()
