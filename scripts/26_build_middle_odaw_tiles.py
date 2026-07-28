#!/usr/bin/env python3
"""
Stage-2 corpus extension — consensus-verified tile corpus for the middle Odaw.

Reproduces, on four upstream communities, the supervision construction that
Old Fadama's model was trained on (scripts/05, 11-13): the orthomosaic is
resampled to the training GSD, both footprint sources are rasterised onto that
exact grid, and pixels on which they agree become the verified label while
disagreements are marked ignore (127).

Two details carry the scientific weight:

  * SCALE. The Stage-2 model consumes 512 px tiles cut at 5 cm and resized to
    256, i.e. a fixed 25.6 m ground footprint. The Open Cities scenes are
    2.0-5.2 cm native, so each is resampled to 5 cm before tiling. scripts/09
    established that train/eval scale mismatch costs real accuracy; a transfer
    test conducted at the wrong scale would measure the mismatch, not the
    transfer.

  * IDENTICAL LABEL CONSTRUCTION. The comparison is only meaningful if the new
    AOI's labels are as noisy as the old one's in the same way. Both sources
    and the consensus rule are therefore reproduced exactly rather than
    substituted.

One 1024 m window is taken per community, matching the Old Fadama window, and
centred on its scene. Tiles without building signal are dropped, as before.

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/26_build_middle_odaw_tiles.py
"""
import os, json, subprocess, csv
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
AOI = os.path.join(ROOT, "accra_flood", "middleodaw")
TILES = os.path.join(AOI, "tiles")
SCENES = os.path.join(AOI, "scenes.json")
BIN = os.environ.get("BIN", "")

GSD = 0.05          # m, the Stage-2 training resolution
# Window per community. Old Fadama used 1024 m, but that is a TRAINING corpus;
# this one exists to measure transfer, and 512 m over four communities already
# yields several hundred tiles per site — far past the point where the estimate
# is sample-limited. It also matters practically: the scenes are read remotely
# and a 1024 m window costs ~10 minutes of range requests each. Override with
# MIDODAW_WINDOW to widen for training use.
WINDOW = float(os.environ.get("MIDODAW_WINDOW", 512.0))
TILE = 512          # px
MIN_BLDG = 0.005    # retained-tile floor on building fraction (as scripts/05)

# Remote-read tuning: without a block cache GDAL refetches neighbouring blocks
# repeatedly while warping, which dominates the wall clock over a slow link.
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "200000000")
os.environ.setdefault("GDAL_CACHEMAX", "1024")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def gdal(tool, *args, quiet=True, retries=4):
    """Run a GDAL utility, retrying on failure.

    Remote reads intermittently return `response_code=0` on a single range
    request and abort the whole warp; the operation succeeds unchanged on a
    retry. Failures are re-raised with stderr attached rather than swallowed.
    """
    cmd = [os.path.join(BIN, tool)] + (["-q"] if quiet else []) + [str(a) for a in args]
    for attempt in range(1, retries + 1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
        if attempt < retries:
            back = 15 * attempt          # the failures are throttling, not bugs;
            print(f"    {tool} failed (attempt {attempt}/{retries}), "  # back off hard
                  f"waiting {back}s: "
                  f"{r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else 'no stderr'}",
                  flush=True)
            subprocess.run(["sleep", str(back)])
    raise RuntimeError(f"{tool} failed after {retries} attempts:\n{r.stderr[-2000:]}")


def to_utm(lon, lat):
    r = subprocess.run([os.path.join(BIN, "gdaltransform"), "-s_srs", "EPSG:4326",
                        "-t_srs", "EPSG:32630"],
                       input=f"{lon} {lat}\n", capture_output=True, text=True, check=True)
    x, y = r.stdout.split()[:2]
    return float(x), float(y)


def read_gray(path):
    return np.asarray(Image.open(path))


def build(name, url, lon, lat, writer):
    src = "/vsicurl/" + url           # windowed remote read; see scripts/24
    work = os.path.join(TILES, f"_work_{name}")
    os.makedirs(work, exist_ok=True)
    cx, cy = to_utm(lon, lat)
    h = WINDOW / 2
    te = (cx - h, cy - h, cx + h, cy + h)

    crop = os.path.join(work, "crop.tif")
    if os.path.exists(crop):
        print(f"  {name}: reusing cached {WINDOW:.0f} m crop")
    else:
        print(f"  {name}: fetching {WINDOW:.0f} m window remotely at {GSD*100:.0f} cm ...",
              flush=True)
        # -of GTiff is REQUIRED, not decorative: gdalwarp selects its driver from
        # the output extension, and the ".part" staging suffix is unknown to it,
        # so it aborts at driver selection before touching the network.
        gdal("gdalwarp", "-overwrite", "-of", "GTiff",
             "-t_srs", "EPSG:32630", "-tr", GSD, GSD,
             "-te", *te, "-r", "bilinear", "-ot", "Byte",
             "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", src, crop + ".part")
        if os.path.exists(crop + ".part"):
            os.replace(crop + ".part", crop)
        elif not os.path.exists(crop):        # tolerate a concurrent run finishing first
            raise RuntimeError(f"{name}: warp produced no output")
        # A silently truncated remote read yields an all-zero raster that still
        # opens cleanly; check before it can poison the corpus.
        st = subprocess.run([os.path.join(BIN, "gdalinfo"), "-stats", crop],
                            capture_output=True, text=True).stdout
        if "StdDev=0" in st:
            raise RuntimeError(f"{name}: crop is uniform — the remote read failed")

    # both label sources rasterised onto the identical grid
    for tag, gpkg, layer, where in (
            ("osm", os.path.join(AOI, "osm", "middleodaw_osm.gpkg"), "b", "building IS NOT NULL"),
            ("gob", os.path.join(AOI, "open_buildings", "open_buildings.gpkg"), "buildings", None)):
        m = os.path.join(work, f"mask_{tag}.tif")
        gdal("gdal_translate", "-b", 1, "-ot", "Byte", "-scale", 0, 255, 0, 0,
             "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", crop, m)
        if tag == "osm":
            sub = os.path.join(work, "osm_b.gpkg")
            if os.path.exists(sub):
                os.remove(sub)
            gdal("ogr2ogr", "-f", "GPKG", sub, "-t_srs", "EPSG:32630",
                 "-where", where, "-nln", "b", gpkg, "multipolygons")
            gdal("gdal_rasterize", "-burn", 255, "-l", "b", sub, m)
        else:
            gdal("gdal_rasterize", "-burn", 255, "-l", layer, gpkg, m)

    info = subprocess.run([os.path.join(BIN, "gdalinfo"), crop],
                          capture_output=True, text=True, check=True).stdout
    size = [l for l in info.splitlines() if l.startswith("Size is")][0]
    W, H = [int(v) for v in size.replace("Size is", "").replace(",", " ").split()]

    imgdir = os.path.join(TILES, "images"); mskdir = os.path.join(TILES, "masks")
    condir = os.path.join(TILES, "masks_consensus")
    for d in (imgdir, mskdir, condir):
        os.makedirs(d, exist_ok=True)

    kept = 0
    # stripe-wise so a 400-megapixel mask is never held whole
    for y in range(0, H - TILE + 1, TILE):
        strips = {}
        for tag, srcpath, bands in (("img", crop, ("-b", 1, "-b", 2, "-b", 3)),
                                    ("osm", os.path.join(work, "mask_osm.tif"), ("-b", 1)),
                                    ("gob", os.path.join(work, "mask_gob.tif"), ("-b", 1))):
            dst = os.path.join(work, f"_strip_{tag}.tif")
            gdal("gdal_translate", *bands, "-srcwin", 0, y, W, TILE,
                 "-co", "PROFILE=BASELINE", "-co", "COMPRESS=NONE", srcpath, dst)
            strips[tag] = np.asarray(Image.open(dst))
        img_s, osm_s, gob_s = strips["img"], strips["osm"], strips["gob"]
        for x in range(0, W - TILE + 1, TILE):
            o = osm_s[:, x:x + TILE] > 127
            g = gob_s[:, x:x + TILE] > 127
            if o.mean() <= MIN_BLDG and g.mean() <= MIN_BLDG:
                continue
            tid = f"{name}_{x:05d}_{y:05d}.png"
            cons = np.where(o == g, np.where(o, 255, 0), 127).astype(np.uint8)
            Image.fromarray(img_s[:, x:x + TILE]).save(os.path.join(imgdir, tid))
            Image.fromarray((o * 255).astype(np.uint8)).save(os.path.join(mskdir, tid))
            Image.fromarray(cons).save(os.path.join(condir, tid))
            writer.writerow([tid, name, x, y, f"{o.mean():.4f}",
                             f"{g.mean():.4f}", f"{(cons == 127).mean():.4f}"])
            kept += 1
    for f in os.listdir(work):                  # keep crop/masks for cheap reruns
        if f.startswith("_strip_"):
            os.remove(os.path.join(work, f))
    print(f"  {name}: {kept} tiles retained")
    return kept


def main():
    os.makedirs(TILES, exist_ok=True)
    scenes = json.load(open(SCENES))["scenes"]
    man = os.path.join(TILES, "manifest.csv")
    total = 0
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "community", "col", "row",
                    "osm_fraction", "gob_fraction", "ignore_fraction"])
        for name, s in scenes.items():
            bb = s["bbox"]                       # window centred on the scene
            total += build(name, s["url"], (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, w)
    print(f"\ncorpus: {total} tiles -> {TILES}/{{images,masks,masks_consensus}}")
    print(f"manifest: {man}")


if __name__ == "__main__":
    main()
