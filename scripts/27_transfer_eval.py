#!/usr/bin/env python3
"""
Stage-2 transfer test — does the Old Fadama representation generalise upstream?

The consensus model (scripts/13) was fitted entirely at Old Fadama, a
lagoon-mouth informal settlement at the catchment outlet. Whether it learned
"dense Accra built fabric" or merely "Old Fadama" has been an open limitation
since Stage 2. This evaluates the unmodified checkpoint, zero-shot, on four
upstream middle-Odaw communities it has never seen (scripts/24-26).

The protocol is the verified one: pixels where OSM and Open Buildings agree are
scored, disagreements are ignored, exactly as at Old Fadama, so the numbers are
directly comparable to the 0.80 pooled / 0.70 per-tile reference.

Critically, the LABEL NOISE FLOOR is reported beside every score. A drop in IoU
on a new AOI is uninterpretable on its own: it may mean the model transferred
badly, or that the labels there are worse. Old Fadama's OSM/Open Buildings
agreement was 0.611, and the model's verified score has to be read against
whatever the corresponding figure is here. Reporting one without the other is
the mistake this project already made once, in the opposite direction, when a
0.55 legacy score was briefly taken for a regression.

Run via:
    .venv/bin/python scripts/27_transfer_eval.py
"""
import os, glob, csv, random, argparse, importlib.util, collections
import numpy as np
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
TILES = os.path.join(ROOT, "accra_flood", "middleodaw", "tiles")
OLDFADAMA = os.path.join(ROOT, "accra_flood", "oldfadama", "tiles")
CKPT = os.path.join(ROOT, "accra_flood", "oldfadama", "tiles", "_run",
                    "resunet_consensus_best.pt")
CSV = os.path.join(ROOT, "accra_flood", "output", "transfer_eval.csv")
FIG = os.path.join(ROOT, "docs", "figures", "transfer_middleodaw.png")
SIZE = 256

spec = importlib.util.spec_from_file_location("seg", os.path.join(HERE, "09_train_unet_v3.py"))
seg = importlib.util.module_from_spec(spec); spec.loader.exec_module(seg)

# Old Fadama reference (scripts/13), for direct comparison.
REF_POOLED, REF_PERTILE, REF_LABEL_FLOOR = 0.80, 0.70, 0.611


def load_tile(name):
    img = Image.open(os.path.join(TILES, "images", name)).convert("RGB").resize(
        (SIZE, SIZE), Image.BILINEAR)
    con = Image.open(os.path.join(TILES, "masks_consensus", name)).resize(
        (SIZE, SIZE), Image.NEAREST)
    osm = Image.open(os.path.join(TILES, "masks", name)).resize(
        (SIZE, SIZE), Image.NEAREST)
    a = (np.asarray(img, dtype=np.float32) / 255.0 - seg.IMAGENET_MEAN) / seg.IMAGENET_STD
    m = np.asarray(con)
    return (torch.from_numpy(a).permute(2, 0, 1),
            (m == 255), (m != 127), np.asarray(osm) > 127)


def main():
    global TILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="score Old Fadama's own validation split through this "
                         "identical code path — the check that a low transfer "
                         "number is a transfer effect and not a pipeline bug")
    a = ap.parse_args()

    if a.control:
        TILES = OLDFADAMA
        names = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(TILES, "images", "*.png"))
                       if os.path.exists(os.path.join(TILES, "masks",
                                                      os.path.basename(p))))
        random.seed(42); random.shuffle(names)      # identical split to scripts/13
        names = sorted(names[int(len(names) * 0.8):])
        print(f"CONTROL: Old Fadama validation split, {len(names)} tiles")
    else:
        if not os.path.isdir(os.path.join(TILES, "images")):
            raise SystemExit("no middle-Odaw corpus; run scripts/24-26 first")
        names = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(TILES, "images", "*.png")))
    if not names:
        raise SystemExit("corpus is empty")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = seg.ResUNet().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print(f"device={device}  tiles={len(names)}  checkpoint={os.path.basename(CKPT)}")

    per = collections.defaultdict(lambda: dict(
        inter=0, union=0, ious=[], li=0, lu=0, pred=0, osm=0, gob=0, npx=0, ign=0))
    rows = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            batch = names[i:i + 8]
            xs, tg, vl, om = zip(*[load_tile(n) for n in batch])
            x = torch.stack(xs).to(device)
            p = (torch.sigmoid(seg.tta_logits(model, x))[:, 0] > 0.5).cpu().numpy()
            for n, pr, t, v, o in zip(batch, p, tg, vl, om):
                comm = n.split("_")[0]
                d = per[comm]
                pv, tv = pr & v, t & v
                inter, union = (pv & tv).sum(), (pv | tv).sum()
                d["inter"] += int(inter); d["union"] += int(union)
                d["ious"].append((inter + 1) / (union + 1))
                # label floor on this tile: OSM vs Open Buildings.
                # consensus encodes it: agree-building = t, disagree = ~v
                gob = (t & v) | ((~o) & (~v))      # reconstruct GOB from consensus+OSM
                d["li"] += int((o & gob).sum()); d["lu"] += int((o | gob).sum())
                d["pred"] += int(pr.sum()); d["osm"] += int(o.sum())
                d["gob"] += int(gob.sum()); d["npx"] += pr.size
                d["ign"] += int((~v).sum())
                rows.append([n, comm, f"{(inter+1)/(union+1):.4f}",
                             f"{pr.mean():.4f}", f"{o.mean():.4f}", f"{(~v).mean():.4f}"])
            if (i // 8) % 20 == 0:
                print(f"  {i+len(batch)}/{len(names)}", flush=True)

    print(f"\n{'community':12s} {'tiles':>6s} {'pooled':>7s} {'per-tile':>9s} "
          f"{'label floor':>12s} {'ignore':>7s} {'pred%':>6s} {'osm%':>6s}")
    tot = dict(inter=0, union=0, ious=[], li=0, lu=0, ign=0, npx=0)
    for comm in sorted(per):
        d = per[comm]
        pooled = d["inter"] / max(d["union"], 1)
        floor = d["li"] / max(d["lu"], 1)
        print(f"{comm:12s} {len(d['ious']):>6d} {pooled:>7.3f} "
              f"{np.mean(d['ious']):>9.3f} {floor:>12.3f} "
              f"{d['ign']/d['npx']:>7.1%} {d['pred']/d['npx']:>6.1%} {d['osm']/d['npx']:>6.1%}")
        for k in ("inter", "union", "li", "lu", "ign", "npx"):
            tot[k] += d[k]
        tot["ious"] += d["ious"]
    pooled = tot["inter"] / max(tot["union"], 1)
    floor = tot["li"] / max(tot["lu"], 1)
    print(f"{'ALL':12s} {len(tot['ious']):>6d} {pooled:>7.3f} "
          f"{np.mean(tot['ious']):>9.3f} {floor:>12.3f} {tot['ign']/tot['npx']:>7.1%}")

    print(f"\nOld Fadama reference (scripts/13): pooled {REF_POOLED:.2f}, "
          f"per-tile {REF_PERTILE:.2f}, label floor {REF_LABEL_FLOOR:.3f}")
    print(f"Transfer, unmodified checkpoint:   pooled {pooled:.2f}, "
          f"per-tile {np.mean(tot['ious']):.2f}, label floor {floor:.3f}")
    dp = pooled - REF_POOLED
    df = floor - REF_LABEL_FLOOR
    print(f"\n  change in verified score  {dp:+.3f}")
    print(f"  change in label floor     {df:+.3f}")
    print("  -> " + (
        "the score moves with the label floor; the representation itself transfers."
        if abs(dp - df) < 0.06 else
        "the score moves independently of the label floor; this is a genuine "
        "transfer effect, not a labelling artefact."))

    if a.control:                      # never overwrite the transfer outputs
        print("\n(control run — comparison and artefacts suppressed)")
        return
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "community", "verified_iou", "pred_fraction",
                    "osm_fraction", "ignore_fraction"])
        w.writerows(rows)
    qualitative(model, names, device)
    print(f"\nwrote {CSV}\n      {FIG}")


def qualitative(model, names, device, per_comm=3):
    """One row per community: image, verified label, prediction."""
    by = collections.defaultdict(list)
    for n in names:
        by[n.split("_")[0]].append(n)
    picks = [n for c in sorted(by) for n in by[c][::max(1, len(by[c]) // per_comm)][:per_comm]]
    if not picks:
        return
    cell, panels = 256, []
    with torch.no_grad():
        for n in picks:
            x, t, v, _o = load_tile(n)
            pr = (torch.sigmoid(seg.tta_logits(model, x[None].to(device)))[0, 0]
                  > 0.5).cpu().numpy()
            img = np.asarray(Image.open(os.path.join(TILES, "images", n)
                                        ).convert("RGB").resize((cell, cell)))
            lab = np.zeros((cell, cell, 3), np.uint8)
            lab[t] = (70, 130, 180); lab[~v] = (200, 200, 200)
            prd = np.zeros((cell, cell, 3), np.uint8)
            prd[pr] = (200, 90, 60)
            gap = np.full((cell, 4, 3), 255, np.uint8)
            panels.append(np.concatenate([img, gap, lab, gap, prd], axis=1))
    grid = np.concatenate([np.concatenate([p, np.full((4, p.shape[1], 3), 255, np.uint8)])
                           for p in panels])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    Image.fromarray(grid).save(FIG)


if __name__ == "__main__":
    main()
