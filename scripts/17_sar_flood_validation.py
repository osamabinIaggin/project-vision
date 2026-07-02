#!/usr/bin/env python3
"""
Stage-3 validation (areal) — Sentinel-1 SAR mapping of the 18 June 2018 Accra
flood, tested against the FABDEM HAND surface.

No agency published a flood-extent product for any Odaw event, so one is
derived here from primary data. The 18 June 2018 flood (FloodList; 169 mm in a
day, Odawna/Circle corridor inundated, streets still ~1 m deep on 20 June) has
a Sentinel-1A radiometrically-terrain-corrected (RTC) acquisition on
2018-06-20 18:18 UTC — two days after onset, same track as a stack of
dry-season references. Data: Microsoft Planetary Computer, anonymous SAS
access, gamma0 VV COGs read remotely via windowed /vsicurl requests.

Method (standard SAR change detection for open-water flooding):
  * reference = per-pixel median of three dry-season scenes (Jan-Feb 2018,
    same platform/track/geometry);
  * flood water = backscatter below the open-water ceiling (-15 dB) AND a
    >= 3 dB drop from the reference (rules out permanently dark surfaces);
  * permanent water = reference itself below -15 dB (lagoon, sea).
Caveat stated up front: VV change detection sees open-water flooding; flooded
pixels under dense roofing (double-bounce) are not detectable, so the derived
extent is a lower bound concentrated in open floodplain — precisely the areas
where HAND should be low.

Validation: the flood mask is compared against HAND (scripts/15) on the same
20 m grid — reporting median HAND of flooded vs non-flooded land, the share of
flood pixels in the severe class, and rank separation (AUC).

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/17_sar_flood_validation.py
"""
import os, json, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
HYDRO = os.path.join(ROOT, "accra_flood", "output", "hydro")
SAR = os.path.join(ROOT, "accra_flood", "output", "sar")
FIG = os.path.join(ROOT, "docs", "figures", "sar_flood_{event}_vs_hand.png")
BIN = os.environ.get("BIN", "")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
W_, S_, E_, N_ = -0.30, 5.52, -0.15, 5.68        # lower-Odaw corridor (WGS84)

# Each event: flood scene + same-platform/track dry-season references.
# 2018: acquisition D+2 after onset (18 June) — post-recession control case.
# 2020: rain from late 8 June, deaths on the 9th; acquisition the SAME evening
#       (storms peak late afternoon; the 18:18 GMT pass samples near peak water).
EVENTS = {
    "2018-06-20": {
        "flood": "S1A_IW_GRDH_1SDV_20180620T181753_20180620T181822_022444_026E40_rtc",
        "dry": ["S1A_IW_GRDH_1SDV_20180115T181749_20180115T181818_020169_022677_rtc",
                "S1A_IW_GRDH_1SDV_20180127T181749_20180127T181818_020344_022C02_rtc",
                "S1A_IW_GRDH_1SDV_20180208T181749_20180208T181818_020519_02319B_rtc"],
    },
    "2020-06-09": {
        "flood": "S1A_IW_GRDH_1SDV_20200609T181805_20200609T181834_032944_03D0DF_rtc",
        "dry": ["S1A_IW_GRDH_1SDV_20200117T181802_20200117T181831_030844_0389FA_rtc",
                "S1A_IW_GRDH_1SDV_20200129T181801_20200129T181831_031019_039022_rtc",
                "S1A_IW_GRDH_1SDV_20200210T181801_20200210T181830_031194_03963A_rtc"],
    },
}
WATER_DB, DROP_DB = -15.0, 3.0


def http_json(url, payload=None, attempts=4):
    cmd = ["curl", "-s", "--max-time", "60", "-H", "Content-Type: application/json"]
    if payload is not None:
        cmd += ["-d", json.dumps(payload)]
    for i in range(attempts):
        r = subprocess.run(cmd + [url], capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            if i == attempts - 1:
                raise
            import time; time.sleep(5 * (i + 1))


BLOB = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc"


def resolve_vv_href(scene_id, token):
    """Resolve the VV asset URL by listing the blob container directly (the
    SAS token grants list; avoids the rate-limited STAC search endpoint)."""
    base = scene_id[:-4] if scene_id.endswith("_rtc") else scene_id
    y, m, d = int(base[17:21]), int(base[21:23]), int(base[23:25])
    # Prefix with the full scene name: the daily folder holds every scene
    # worldwide and paginates long before an unprefixed scan would find ours.
    url = (f"{BLOB}?restype=container&comp=list"
           f"&prefix=GRD/{y}/{m}/{d}/IW/DV/{base}&{token}")
    r = subprocess.run(["curl", "-s", "--max-time", "60", url],
                       capture_output=True, text=True, check=True)
    import re
    names = re.findall(r"<Name>([^<]+)</Name>", r.stdout)
    hits = [n for n in names if n.endswith("iw-vv.rtc.tiff")]
    if not hits:
        raise RuntimeError(f"no VV blob found for {scene_id}")
    return f"{BLOB}/{hits[0]}"


def fetch_vv(scene_id, token):
    """Warp the scene's VV gamma0 onto the corridor grid (EPSG:32630, 20 m)."""
    out = os.path.join(SAR, f"{scene_id[17:25]}_vv.tif")
    if not os.path.exists(out):
        href = resolve_vv_href(scene_id, token)
        subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                        "-t_srs", "EPSG:32630", "-tr", "20", "20",
                        "-te", str(W_), str(S_), str(E_), str(N_),
                        "-te_srs", "EPSG:4326", "-r", "average",
                        "-ot", "Float32", "-co", "PROFILE=BASELINE",
                        f"/vsicurl/{href}?{token}", out], check=True)
    a = np.array(Image.open(out), dtype=np.float32)
    a[a <= 0] = np.nan
    return 10.0 * np.log10(a)                     # gamma0 -> dB


def odaw_basin_mask():
    """Delineate the Odaw watershed (WhiteboxTools) from the Stage-3 D8 pointer,
    pour point snapped to the main stem at the Korle Lagoon inlet, and warp it
    onto the corridor grid."""
    import whitebox
    mask20 = os.path.join(SAR, "_basin20.tif")
    if not os.path.exists(mask20):
        r = subprocess.run([os.path.join(BIN, "gdaltransform"),
                            "-s_srs", "EPSG:4326", "-t_srs", "EPSG:32630",
                            "-output_xy"], input="-0.2230 5.5560\n",
                           capture_output=True, text=True, check=True)
        x, y = r.stdout.split()
        pour_csv = os.path.join(SAR, "_pour.csv")
        open(pour_csv, "w").write(f"WKT,id\n\"POINT ({x} {y})\",1\n")
        pour_shp = os.path.join(SAR, "_pour.shp")
        subprocess.run([os.path.join(BIN, "ogr2ogr"), "-f", "ESRI Shapefile",
                        "-a_srs", "EPSG:32630", pour_shp, pour_csv], check=True)
        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(HYDRO)
        wbt.set_verbose_mode(False)
        wbt.snap_pour_points(pour_shp, "flow_accum.tif",
                             os.path.join(SAR, "_pour_snap.shp"), snap_dist=1000)
        wbt.watershed("d8_pointer.tif", os.path.join(SAR, "_pour_snap.shp"),
                      os.path.join(SAR, "_odaw_basin.tif"))
        subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                        "-t_srs", "EPSG:32630", "-tr", "20", "20",
                        "-te", str(W_), str(S_), str(E_), str(N_),
                        "-te_srs", "EPSG:4326", "-r", "near",
                        "-ot", "Float32", "-co", "PROFILE=BASELINE",
                        os.path.join(SAR, "_odaw_basin.tif"), mask20], check=True)
    return np.array(Image.open(mask20)) > 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="2020-06-09", choices=sorted(EVENTS))
    a = ap.parse_args()
    ev = EVENTS[a.event]

    os.makedirs(SAR, exist_ok=True)
    token = http_json(SAS)["token"]

    print(f"Event {a.event}: fetching dry-season reference stack ...")
    ref = np.nanmedian(np.stack([fetch_vv(s, token) for s in ev["dry"]]), axis=0)
    print(f"Fetching flood scene ({a.event}) ...")
    flood_db = fetch_vv(ev["flood"], token)

    permanent = ref < WATER_DB
    flooded = (flood_db < WATER_DB) & (ref - flood_db >= DROP_DB) & ~permanent
    dry_land = ~flooded & ~permanent & np.isfinite(flood_db)

    # HAND on the identical grid.
    hand_p = os.path.join(SAR, "_hand20.tif")
    subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                    "-t_srs", "EPSG:32630", "-tr", "20", "20",
                    "-te", str(W_), str(S_), str(E_), str(N_), "-te_srs", "EPSG:4326",
                    "-r", "bilinear", "-ot", "Float32", "-co", "PROFILE=BASELINE",
                    os.path.join(HYDRO, "hand.tif"), hand_p], check=True)
    hand = np.asarray(Image.open(hand_p), dtype=np.float32)

    # Restrict validation to the Odaw watershed: the corridor window also
    # contains the Panbros salt pans (Densu side), whose seasonal pond cycle
    # reads as water change but is not storm flooding.
    basin = odaw_basin_mask()

    ha = 20 * 20 / 1e4
    for label, sel in (("corridor (all)", np.ones_like(flooded)),
                       ("Odaw watershed only", basin)):
        fl, dl = flooded & sel, dry_land & sel
        hf, hd = hand[fl], hand[dl]
        if not len(hf):
            continue
        rng = np.random.default_rng(42)
        fs = rng.choice(hf, 20000); ds = rng.choice(hd, 20000)
        auc = ((fs[:, None] < ds[None, ::40]).mean()
               + 0.5 * (fs[:, None] == ds[None, ::40]).mean())
        print(f"\n[{label}] flood water: {fl.sum()*ha:.0f} ha ({fl.sum()} px)")
        print(f"  median HAND — flooded: {np.median(hf):.2f} m   dry land: {np.median(hd):.2f} m")
        for t in (2.0, 5.0):
            print(f"  flood pixels with HAND < {t:.0f} m: {(hf < t).mean():5.1%}   "
                  f"(dry-land base rate {(hd < t).mean():5.1%})")
        print(f"  rank separation AUC: {auc:.3f}")

    # Figure: HAND classes pale, permanent water dark blue, flood detections red.
    rgb = np.zeros((*hand.shape, 3), np.uint8)
    rgb[hand >= 5] = (238, 238, 232)
    rgb[(hand >= 2) & (hand < 5)] = (250, 214, 150)
    rgb[hand < 2] = (240, 170, 150)
    rgb[permanent] = (35, 60, 140)
    rgb[flooded] = (200, 20, 20)
    im = Image.fromarray(rgb).resize((rgb.shape[1]*2, rgb.shape[0]*2), Image.NEAREST)
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    sx = rgb.shape[1]*2 / (E_ - W_); sy = rgb.shape[0]*2 / (N_ - S_)
    d.rectangle([(-0.2273-W_)*sx, (N_-5.5520)*sy, (-0.2180-W_)*sx, (N_-5.5428)*sy],
                outline=(0, 0, 0), width=3)
    fig = FIG.format(event=a.event.replace("-", ""))
    os.makedirs(os.path.dirname(fig), exist_ok=True)
    im.save(fig)
    print(f"wrote {fig}")


if __name__ == "__main__":
    main()
