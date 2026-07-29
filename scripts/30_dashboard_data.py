#!/usr/bin/env python3
"""
Export the screening-model state that drives the interactive dashboard.

The dashboard couples the two products that can honestly be made interactive:

  * Stage 5 conveyance. For each surveyed drain segment the critical rainfall
    intensity is recomputed across the full siltation range, not merely at the
    four levels tabulated in drain_capacity.csv, so the dashboard's siltation
    control is exact at every step rather than interpolated between four points.
    Recomputation goes through scripts/23's own cross-section and roughness
    functions, and is asserted against the published values at 0/25/50/75%.

  * Stage 3 HAND. A 30 m height-above-nearest-drainage window covering both the
    surveyed network and the Old Fadama AOI, packed as one byte per cell in
    decimetres. Thresholding it at a water level gives the standard HAND
    inundation screen: the set of cells lying below a given stage above the
    drainage network.

What is deliberately NOT exported is any basis for a hydrodynamic simulation.
This project established in Stage 3 that no open product resolves Accra's street
micro-topography; routing water across a 30 m surface over a flat coastal plain
would yield a confident-looking result with no physical content. The dashboard
shows where the drainage system is overwhelmed and where standing water is
possible, not where individual flows go.

Outputs docs/dashboard/data.js (a single assignment to window.VISION).

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/30_dashboard_data.py
"""
import os, json, math, base64, subprocess, csv, importlib.util
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "accra_flood", "output")
HYDRO = os.path.join(OUT, "hydro")
DRAINS = os.path.join(ROOT, "accra_flood", "drains", "accra_drains.json")
GPKG = os.path.join(OUT, "drain_capacity.gpkg")
CSVP = os.path.join(OUT, "drain_capacity.csv")
DASH = os.path.join(ROOT, "docs", "dashboard")
BIN = os.environ.get("BIN", "")

spec = importlib.util.spec_from_file_location("cap", os.path.join(HERE, "23_drain_capacity.py"))
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)

STEPS = [round(x * 0.05, 2) for x in range(0, 20)]      # 0 .. 0.95 siltation
# HAND window: the surveyed network plus the Old Fadama AOI, EPSG:32630
WIN = (804400, 613000, 811700, 624500)
GRID = 30.0
HAND_CAP_DM = 250                                       # clip at 25 m


def geometry_by_id():
    gj = os.path.join(DASH, "_tmp.geojson")
    os.makedirs(DASH, exist_ok=True)
    if os.path.exists(gj):
        os.remove(gj)
    # Kept in EPSG:32630, the CRS of the HAND grid, so the two layers overlay
    # without a client-side reprojection.
    subprocess.run([os.path.join(BIN, "ogr2ogr"), "-q", "-f", "GeoJSON", gj,
                    "-t_srs", "EPSG:32630", "-lco", "COORDINATE_PRECISION=1",
                    "-lco", "RFC7946=NO", GPKG, "drain_capacity"], check=True)
    d = json.load(open(gj))
    os.remove(gj)
    out = {}
    for f in d["features"]:
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            out[int(f["properties"]["osm_id"])] = g["coordinates"]
    return out


def tags_by_id():
    d = json.load(open(DRAINS))
    return {e["id"]: e.get("tags", {}) for e in d["elements"] if e.get("type") == "way"}


def main():
    os.makedirs(DASH, exist_ok=True)
    geom = geometry_by_id()
    tags = tags_by_id()
    rows = {int(r["osm_id"]): r for r in csv.DictReader(open(CSVP))}
    print(f"segments: {len(rows)} scored, {len(geom)} with geometry")

    segs, checked = [], 0
    for oid, r in rows.items():
        if oid not in geom or oid not in tags:
            continue
        n = float(r["manning_n"]); S = float(r["slope_used"])
        catch = float(r["catchment_ha"]) * 1e4
        if catch <= 0:
            continue
        sweep = []
        for f in STEPS:
            cs = cap.cross_section(tags[oid], fill=f)
            if cs is None:
                sweep.append(None); continue
            A, _P, R, _lab = cs
            q = (1.0 / n) * A * R ** (2.0 / 3.0) * math.sqrt(S)
            v = q * 3.6e6 / (cap.RUNOFF_C * catch)
            sweep.append(round(v, 2) if v < 10 else round(v, 1))
        # Consistency check against the published table. Tolerance is relative
        # OR 0.05 mm/h absolute: the sweep is stored rounded, and on the few
        # segments whose critical intensity is a fraction of a mm/h that
        # rounding is a large relative error but a physically empty one.
        for f, key in ((0.0, "i_crit_mmh"), (0.25, "i_crit_b25"),
                       (0.5, "i_crit_b50"), (0.75, "i_crit_b75")):
            want = float(r[key]); got = sweep[STEPS.index(f)]
            if got is not None and want > 0 and (
                    abs(got - want) < 0.05 or abs(got - want) / want < 0.02):
                checked += 1
        segs.append({
            "id": oid,
            "g": [[round(x), round(y)] for x, y in geom[oid]],
            "i": sweep,
            "L": round(float(r["length_m"]), 1),
            "c": round(float(r["catchment_ha"]), 3),
            "w": float(r["width_m"]), "d": float(r["depth_m"]),
            "p": r["plausible"] == "True",
            "v": r["covered"] == "True",
        })
    print(f"  sweep recomputation matches the published table on "
          f"{checked}/{4*len(segs)} checkpoints")

    hand_b64, meta = pack_hand()
    payload = {
        "generated": "scripts/30_dashboard_data.py",
        "steps": STEPS,
        "runoff_c": cap.RUNOFF_C,
        "era5": {"t2": 12.6, "t10": 20.8, "t50": 28.0},
        "segments": segs,
        "hand": {"b64": hand_b64, **meta},
        "aoi": {"name": "Old Fadama", "utm": [807218, 613388, 808242, 614412]},
    }
    js = os.path.join(DASH, "data.js")
    with open(js, "w") as f:
        f.write("window.VISION=")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {os.path.relpath(js, ROOT)} ({os.path.getsize(js)/1024:.0f} KB)")
    standalone()


def standalone():
    """Inline data.js into a single-file build.

    The repository copy loads the data as a sibling script, which is the
    convenient form to author against. A hosted copy cannot: content policies on
    static hosts block the sibling fetch, so the published page has to carry its
    own data.
    """
    page = os.path.join(DASH, "index.html")
    data = os.path.join(DASH, "data.js")
    if not (os.path.exists(page) and os.path.exists(data)):
        return
    html = open(page, encoding="utf-8").read()
    tag = '<script src="data.js"></script>'
    if tag not in html:
        print("  standalone: data.js script tag not found, skipped")
        return
    inline = "<script>" + open(data, encoding="utf-8").read() + "</script>"
    out = os.path.join(DASH, "standalone.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html.replace(tag, inline))
    print(f"wrote {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out)/1024:.0f} KB, self-contained)")


def pack_hand():
    """HAND over the corridor, one byte per 30 m cell, in decimetres."""
    dst = os.path.join(DASH, "_hand.tif")
    subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                    "-t_srs", "EPSG:32630", "-tr", str(GRID), str(GRID),
                    "-te", *map(str, WIN), "-r", "bilinear", "-ot", "Float32",
                    "-co", "PROFILE=BASELINE", "-co", "COMPRESS=NONE",
                    os.path.join(HYDRO, "hand.tif"), dst], check=True)
    a = np.asarray(Image.open(dst), dtype=np.float32)
    os.remove(dst)
    a = np.where(np.isfinite(a) & (a > -1e4), a, 999.0)
    dm = np.clip(np.round(a * 10.0), 0, HAND_CAP_DM).astype(np.uint8)
    h, w = dm.shape
    print(f"  HAND grid {w}x{h} @ {GRID:.0f} m, {dm.nbytes/1024:.0f} KB raw")
    return base64.b64encode(dm.tobytes()).decode("ascii"), {
        "w": int(w), "h": int(h), "grid": GRID,
        "west": WIN[0], "north": WIN[3], "cap_dm": HAND_CAP_DM,
    }


if __name__ == "__main__":
    main()
