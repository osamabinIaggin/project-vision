#!/usr/bin/env python3
"""
Stage-4 (temporal) — annual built-up and riparian-encroachment time series for
Old Fadama from Google Open Buildings 2.5D Temporal (2016-2023).

The 2020/2024 drone-pair change detection (scripts/08) could not localise
individual new structures because inter-flight prediction flicker dominates a
two-epoch difference. The Open Buildings 2.5D Temporal dataset (CC-BY 4.0,
anonymous GCS access) replaces the two-epoch problem with an eight-epoch one:
building presence, fractional count, and HEIGHT at 0.5 m grid (4 m effective
resolution), annually 2016-2023, produced by a single consistent model — so
year-on-year differences are meaningful in aggregate.

Per year, over the canonical scripts/05 AOI window:
  * built-up area (presence > 0.5), in ha;
  * mean building height over built pixels (vertical exposure);
  * built-up area within the 15 m riparian buffer of the OSM waterways
    (the Odaw channel and drains crossing the AOI) — the encroachment trend.

Outputs accra_flood/output/builtup_timeseries.csv and a README-ready figure.

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/20_builtup_timeseries.py
"""
import os, json, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
AOI = os.path.join(ROOT, "accra_flood", "oldfadama")
OUTDIR = os.path.join(ROOT, "accra_flood", "output", "temporal")
CSV = os.path.join(ROOT, "accra_flood", "output", "builtup_timeseries.csv")
FIG = os.path.join(ROOT, "docs", "figures", "builtup_timeseries_oldfadama.png")
BIN = os.environ.get("BIN", "")

BUCKET = "open-buildings-temporal-data"
CELL = "0fdfc"                                    # S2 level-7 cell over central Accra
YEARS = list(range(2016, 2024))
ULX, ULY, LRX, LRY = 807218, 614412, 808242, 613388   # scripts/05 window
GRID = 2.0                                        # analysis grid (m)


def gcs_list(prefix):
    names, tok = [], ""
    while True:
        url = (f"https://www.googleapis.com/storage/v1/b/{BUCKET}/o"
               f"?prefix={prefix}&maxResults=1000&pageToken={tok}")
        r = subprocess.run(["curl", "-s", "--max-time", "60", url],
                           capture_output=True, text=True, check=True)
        d = json.loads(r.stdout)
        names += [i["name"] for i in d.get("items", [])]
        tok = d.get("nextPageToken")
        if not tok:
            return names


def fetch_year(year):
    """AOI window of (presence, height) for one epoch, via VRT over the cell."""
    dst = os.path.join(OUTDIR, f"builtup_{year}.tif")
    if not os.path.exists(dst):
        tiles = gcs_list(f"v1/geotiffs/{CELL}_{year}_06_30/")
        vrt = os.path.join(OUTDIR, f"_{year}.vrt")
        srcs = [f"/vsicurl/https://storage.googleapis.com/{BUCKET}/{n}" for n in tiles]
        subprocess.run([os.path.join(BIN, "gdalbuildvrt"), "-q", vrt, *srcs], check=True)
        subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                        "-t_srs", "EPSG:32630", "-tr", str(GRID), str(GRID),
                        "-te", str(ULX), str(LRY), str(LRX), str(ULY),
                        "-r", "average", "-ot", "Float32",
                        "-co", "PROFILE=BASELINE", vrt, dst], check=True)
        os.remove(vrt)
    bands = []
    for b in (2, 3):                      # height, presence (PIL reads
        bp = dst[:-4] + f"_b{b}.tif"      # single-band Float32 only)
        if not os.path.exists(bp):
            subprocess.run([os.path.join(BIN, "gdal_translate"), "-q",
                            "-b", str(b), "-co", "PROFILE=BASELINE", dst, bp],
                           check=True)
        bands.append(np.array(Image.open(bp), dtype=np.float32))
    height, presence = bands
    presence[presence < 0] = 0
    height[height < 0] = 0
    return presence, height


def riparian_mask():
    """15 m buffer of OSM waterways rasterised onto the analysis grid."""
    dst = os.path.join(OUTDIR, "_riparian.tif")
    if not os.path.exists(dst):
        buf = os.path.join(OUTDIR, "_riparian.gpkg")
        subprocess.run([os.path.join(BIN, "ogr2ogr"), "-q", "-f", "GPKG", buf,
                        "-a_srs", "EPSG:32630", "-nln", "rip",
                        "-dialect", "sqlite",
                        # transform to metres BEFORE buffering: the source
                        # layer is EPSG:4326, where 15 means 15 degrees
                        "-sql", "SELECT ST_Buffer(ST_Transform(geom, 32630), 15) g "
                                "FROM lines WHERE waterway IS NOT NULL",
                        os.path.join(AOI, "osm", "oldfadama_osm.gpkg")], check=True)
        n = int(round((LRX - ULX) / GRID))
        m = int(round((ULY - LRY) / GRID))
        subprocess.run([os.path.join(BIN, "gdal_rasterize"), "-q", "-burn", "1",
                        "-ot", "Byte", "-co", "PROFILE=BASELINE",
                        "-te", str(ULX), str(LRY), str(LRX), str(ULY),
                        "-ts", str(n), str(m), "-l", "rip", buf, dst], check=True)
    return np.array(Image.open(dst)) > 0


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rip = riparian_mask()
    px_ha = GRID * GRID / 1e4
    rows = []
    for y in YEARS:
        presence, height = fetch_year(y)
        built = presence > 0.5
        area = built.sum() * px_ha
        h = float(height[built].mean()) if built.any() else 0.0
        rip_area = (built & rip).sum() * px_ha
        rows.append((y, area, h, rip_area))
        print(f"{y}: built-up {area:6.1f} ha   mean height {h:4.2f} m   "
              f"riparian(15 m) {rip_area:5.2f} ha")

    with open(CSV, "w") as f:
        f.write("year,builtup_ha,mean_height_m,riparian_builtup_ha\n")
        for y, a, h, r in rows:
            f.write(f"{y},{a:.2f},{h:.3f},{r:.3f}\n")
    figure(rows)
    print(f"wrote {CSV} and {FIG}")


def figure(rows):
    """Two-panel PIL bar chart: total built-up ha and riparian built-up ha."""
    Wp, Hp, pad = 1200, 420, 60
    im = Image.new("RGB", (Wp, Hp * 2 + 30), (252, 252, 250))
    d = ImageDraw.Draw(im)
    panels = [("Built-up area in AOI (ha), Open Buildings 2.5D Temporal", 1, (70, 110, 170)),
              ("Built-up within 15 m of drains/Odaw channel (ha)", 3, (190, 70, 50))]
    for pi, (title, ci, col) in enumerate(panels):
        oy = pi * (Hp + 30)
        vals = [r[ci] for r in rows]
        vmax = max(vals) * 1.15
        bw = (Wp - 2 * pad) / len(vals)
        d.text((pad, oy + 12), title, fill=(20, 20, 20))
        for i, (r, v) in enumerate(zip(rows, vals)):
            x0 = pad + i * bw + 8
            hpx = (Hp - 2 * pad) * v / vmax
            d.rectangle([x0, oy + Hp - pad - hpx, x0 + bw - 16, oy + Hp - pad], fill=col)
            d.text((x0 + bw / 2 - 22, oy + Hp - pad + 8), str(r[0]), fill=(20, 20, 20))
            d.text((x0 + bw / 2 - 26, oy + Hp - pad - hpx - 18), f"{v:.1f}", fill=(60, 60, 60))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    im.save(FIG)


if __name__ == "__main__":
    main()
