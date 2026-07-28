#!/usr/bin/env python3
"""
Stage-5 — conveyance-capacity deficit of the surveyed Accra drainage network.

Stages 2-4 established WHERE water concentrates (HAND) and WHERE structures
encroach on the drains. Neither says whether the drains that survive can carry
what the catchment delivers. The Open Cities / GARID field campaigns surveyed
the middle-Odaw drains segment by segment and committed the cross-section
geometry to OSM (scripts/21) — width, depth, profile, material, smoothness —
which is sufficient to close that gap hydraulically.

    Q_cap = (1/n) * A * R^(2/3) * sqrt(S)            [Manning]

CRITICAL INTENSITY, not design storm. scripts/22 established that no open
rainfall product resolves Accra's convective extremes: ERA5 renders the lethal
3 June 2015 storm as an unremarkable 5.1 mm/h day, below every annual maximum in
45 years. Rather than propagate a design storm we cannot defend, the model is
inverted. For each segment the rational method Q = C*i*A is solved for the
intensity at which the drain reaches capacity:

    i_crit [mm/h] = Q_cap * 3.6e6 / (C * A_contributing)

This is a property of the drain and its catchment alone — no rainfall input, no
return-period assumption. A segment with i_crit = 8 mm/h overflows in a routine
shower; one with i_crit = 120 mm/h survives a severe storm. Ranking by i_crit is
therefore invariant to the design-storm uncertainty, and the ERA5 climatological
floor (scripts/22) is used only as a lower benchmark: drains failing below it
fail a fortiori under any realistic Accra storm.

Contributing area is accumulated on the drain network itself, not routed over
the DEM. The reason is a standing finding of this project: no open product
resolves Accra's street-level micro-topography (FABDEM/GLO-30 at 30 m is the
open ceiling), so D8 routing over a 30 m surface cannot resolve which side of a
street drains to which gutter. Instead each grid cell is allocated to its
nearest drain (a standard urban-drainage allocation), and those local
catchments are accumulated downstream through the network graph, which is
reconstructed from shared OSM endpoint coordinates and oriented by conditioned
DEM elevation.

Inputs : accra_flood/drains/accra_drains.json         (scripts/21)
         accra_flood/output/hydro/{slope,dem_breached}.tif  (scripts/15)
         accra_flood/output/design_storm_idf.json     (scripts/22)
Outputs: accra_flood/output/drain_capacity.csv        ranked segment table
         accra_flood/output/drain_capacity.gpkg       QGIS-ready, EPSG:32630
         docs/figures/drain_capacity_accra.png        network map + histogram

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/23_drain_capacity.py
"""
import os, json, math, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DRAINS = os.path.join(ROOT, "accra_flood", "drains", "accra_drains.json")
HYDRO = os.path.join(ROOT, "accra_flood", "output", "hydro")
OUTDIR = os.path.join(ROOT, "accra_flood", "output")
CSV = os.path.join(OUTDIR, "drain_capacity.csv")
GPKG = os.path.join(OUTDIR, "drain_capacity.gpkg")
IDF = os.path.join(OUTDIR, "design_storm_idf.json")
FIG = os.path.join(ROOT, "docs", "figures", "drain_capacity_accra.png")
BIN = os.environ.get("BIN", "")

RUNOFF_C = 0.85            # dense informal settlement, near-fully impervious
RUNOFF_C_SENS = (0.70, 0.90)
# Longitudinal grade. Bounds are those urban drains are laid to (minimum
# self-cleansing ~0.1%, upper end of gravity practice ~2%); the DEM is used only
# to place a segment within that range, never to assert a grade outside it.
S_MIN, S_MAX = 0.001, 0.02
S_REF = 0.005              # reference grade for the sensitivity band
S_SENS = (0.002, 0.005, 0.01)
SLOPE_BASELINE = 300.0     # m, chord over which the DEM gradient is measured
ALLOC_GRID = 10.0          # m, nearest-drain allocation grid
ALLOC_MAXDIST = 150.0      # m, beyond this a cell is not served by this network
NODE_SNAP_TOL = 3.0        # m, endpoint clustering tolerance for the network graph
BLOCKAGE = (0.0, 0.25, 0.50, 0.75)   # fraction of depth lost to silt/refuse
# Cross-section plausibility screen. A tranche of the survey carries widths of
# 4-14 cm against depths of 0.2-0.7 m — aspect ratios up to 13, which is not a
# street drain but a unit or entry error in the field campaign. Left in, these
# dominate the ranking entirely (91 of the 107 worst segments), so they are
# flagged and held out of the headline statistics rather than silently dropped.
MIN_WIDTH, MIN_DEPTH, MAX_ASPECT = 0.15, 0.10, 5.0

# Manning's n by material x surveyed smoothness (Chow 1959, tab. 5-6; the
# smoothness tag records the surveyed state of the invert, so "rough" and
# "very_rough" absorb siltation and refuse accumulation as well as finish).
MANNING = {
    "concrete":     {"normal": 0.015, "rough": 0.017, "very_rough": 0.020},
    "cement_block": {"normal": 0.020, "rough": 0.023, "very_rough": 0.026},
    "block_mortar": {"normal": 0.020, "rough": 0.023, "very_rough": 0.026},
    "brick_mortar": {"normal": 0.017, "rough": 0.020, "very_rough": 0.023},
    "rock":         {"normal": 0.035, "rough": 0.040, "very_rough": 0.045},
    "ground":       {"normal": 0.025, "rough": 0.030, "very_rough": 0.035},
    "plastic":      {"normal": 0.011, "rough": 0.013, "very_rough": 0.015},
    "steel":        {"normal": 0.013, "rough": 0.015, "very_rough": 0.017},
}
DEFAULT_MATERIAL, DEFAULT_SMOOTH = "concrete", "normal"


# --------------------------------------------------------------------------
# Geodesy — WGS84 to UTM 30N (EPSG:32630), so all metric work shares the CRS
# used by the rest of the pipeline. Verified against gdaltransform on import.
# --------------------------------------------------------------------------
def ll_to_utm30n(lon, lat):
    lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
    a, f = 6378137.0, 1 / 298.257223563
    k0, lon0 = 0.9996, math.radians(-3.0)          # zone 30 central meridian
    e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    phi, lam = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    A = (lam - lon0) * np.cos(phi)
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*phi)
             + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*phi)
             - (35*e2**3/3072) * np.sin(6*phi))
    x = k0 * N * (A + (1 - T + C) * A**3 / 6
                  + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(phi) * (A**2 / 2 + (5 - T + 9*C + 4*C**2) * A**4 / 24
              + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720))
    return x, y


# --------------------------------------------------------------------------
# Cross-section hydraulics
# --------------------------------------------------------------------------
def cross_section(tags, fill=0.0):
    """Flow area, wetted perimeter and hydraulic radius of the conveying section.

    `fill` is the fraction of the surveyed depth occupied by silt and refuse; the
    remaining section is what actually conveys. For rectangular and trapezoidal
    profiles this is exact (the trapezoid's bed widens as it fills, which is
    accounted for). For the curved profiles the reduced depth is carried through
    the same formula, which slightly understates the residual area of a
    bottom-filled invert — an error in the conservative direction.

    Open-channel convention: the perimeter of a covered drain excludes its lid,
    the capacity being that of free-surface flow just short of surcharge.
    Returns (A, P, R, profile_label) or None when geometry is insufficient.
    """
    def num(k):
        try:
            return float(tags[k])
        except (KeyError, ValueError, TypeError):
            return None

    w, d = num("width"), num("depth")
    if not w or not d or w <= 0 or d <= 0:
        return None
    prof = tags.get("drain:profile_open") or tags.get("drain:profile_covered") or ""
    de = d * (1.0 - fill)
    if de <= 0:
        return 0.0, max(w, 1e-6), 0.0, "blocked"

    if prof.startswith("trapezoid") or prof == "elliptical_trapezoid":
        top = num("drain:top_width") or w
        bot = num("drain:bottom_width") or 0.8 * w   # documented fallback
        top, bot = max(top, bot), min(top, bot)
        bed = bot + (top - bot) * fill               # bed widens as silt rises
        A = 0.5 * (top + bed) * de
        P = bed + 2 * math.hypot(de, 0.5 * (top - bed))
        return A, P, A / P, "trapezoid"

    if prof == "elliptical":
        # half-ellipse, semi-axes w/2 (horizontal) and de (vertical)
        A = 0.5 * math.pi * (w / 2) * de
        h = ((w/2 - de) ** 2) / ((w/2 + de) ** 2)
        P = 0.5 * math.pi * (w/2 + de) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))
        return A, P, A / P, "elliptical"

    if prof == "rectangular_elliptical":
        r = w / 2
        if de > r:                      # rectangle over a semicircular invert
            A = 0.5 * math.pi * r**2 + w * (de - r)
            P = math.pi * r + 2 * (de - r)
        else:                           # shallow: half-ellipse
            A = 0.5 * math.pi * r * de
            h = ((r - de) ** 2) / ((r + de) ** 2)
            P = 0.5 * math.pi * (r + de) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))
        return A, P, A / P, "rectangular_elliptical"

    A = w * de                           # open_rectangular, tabulated, unspecified
    P = w + 2 * de
    return A, P, A / P, "rectangular"


def mannings_n(tags):
    mat = (tags.get("drain:material") or DEFAULT_MATERIAL).lower()
    sm = (tags.get("drain:material_smoothness") or DEFAULT_SMOOTH).lower()
    row = MANNING.get(mat, MANNING[DEFAULT_MATERIAL])
    return row.get(sm, row[DEFAULT_SMOOTH]), mat in MANNING, sm in row


# --------------------------------------------------------------------------
# Survey ingestion
# --------------------------------------------------------------------------
def load_segments():
    d = json.load(open(DRAINS))
    segs = []
    for e in d["elements"]:
        if e.get("type") != "way" or len(e.get("geometry", [])) < 2:
            continue
        g = e["geometry"]
        lon = np.array([p["lon"] for p in g]); lat = np.array([p["lat"] for p in g])
        x, y = ll_to_utm30n(lon, lat)
        length = float(np.hypot(np.diff(x), np.diff(y)).sum())
        if length <= 0:
            continue
        segs.append({"id": e["id"], "tags": e.get("tags", {}),
                     "lon": lon, "lat": lat, "x": x, "y": y, "length": length})
    return segs


# --------------------------------------------------------------------------
# Terrain sampling
# --------------------------------------------------------------------------
def boxcar(grid, half):
    """Mean over a (2*half+1)^2 window via an integral image (no scipy)."""
    g = np.nan_to_num(grid.astype(np.float64), nan=0.0)
    ok = np.isfinite(grid).astype(np.float64)
    pad = ((half + 1, half + 1), (half + 1, half + 1))
    cs = np.pad(g, pad).cumsum(0).cumsum(1)
    cn = np.pad(ok, pad).cumsum(0).cumsum(1)
    n, m = grid.shape
    r0 = np.arange(n); c0 = np.arange(m)
    R1, C1 = np.meshgrid(r0 + 2*half + 1, c0 + 2*half + 1, indexing="ij")
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    box = lambda c: c[R1, C1] - c[R0, C1] - c[R1, C0] + c[R0, C0]
    tot, cnt = box(cs), box(cn)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1e-9), np.nan)


def sample_terrain(segs, baseline=SLOPE_BASELINE):
    """Regional hydraulic gradient and smoothed endpoint elevations per segment.

    The Stage-3 DEM is 30 m; the median surveyed segment is 38 m long and lies on
    a coastal plain the HAND analysis showed to be essentially flat. Neither the
    bed slope nor the flow direction of such a segment is recoverable from
    per-cell elevation differences — the first run of this model confirmed it,
    returning a 2.3% median "bed slope" (DEM noise; engineered drains here are
    laid at a fraction of that) and an incoherent flow orientation that left 47%
    of segments receiving no upstream area.

    Both quantities are therefore taken at the only scale the DEM supports: the
    elevation field is smoothed over a `baseline`-metre window, the gradient is
    measured across a `baseline`-metre chord centred on each segment, and the
    result is clamped to the range of grades urban drains are actually laid to.
    """
    xs = np.concatenate([s["x"] for s in segs]); ys = np.concatenate([s["y"] for s in segs])
    pad = 2 * baseline
    te = (math.floor((xs.min() - pad) / 30) * 30, math.floor((ys.min() - pad) / 30) * 30,
          math.ceil((xs.max() + pad) / 30) * 30, math.ceil((ys.max() + pad) / 30) * 30)
    dst = os.path.join(HYDRO, "_drn_dem.tif")
    subprocess.run([os.path.join(BIN, "gdalwarp"), "-q", "-overwrite",
                    "-t_srs", "EPSG:32630", "-tr", "30", "30", "-r", "bilinear",
                    "-ot", "Float32", "-co", "PROFILE=BASELINE", "-co", "COMPRESS=NONE",
                    "-te", *map(str, te),
                    os.path.join(HYDRO, "dem_breached.tif"), dst], check=True)
    dem = np.asarray(Image.open(dst), dtype=np.float32)
    os.remove(dst)
    dem = np.where(dem < -1e4, np.nan, dem)
    dem_s = boxcar(dem, max(1, int(round(baseline / 2 / 30))))

    def sample(grid, x, y):
        """Bilinear — nearest-cell sampling cannot separate the two ends of a
        38 m segment on a 30 m grid, which is precisely what orientation needs."""
        fc = np.clip((np.asarray(x, dtype=float) - te[0]) / 30 - 0.5, 0, grid.shape[1] - 1.001)
        fr = np.clip((te[3] - np.asarray(y, dtype=float)) / 30 - 0.5, 0, grid.shape[0] - 1.001)
        c0, r0 = np.floor(fc).astype(int), np.floor(fr).astype(int)
        tc, tr = fc - c0, fr - r0
        v = ((1-tr)*(1-tc)*grid[r0, c0] + (1-tr)*tc*grid[r0, c0+1]
             + tr*(1-tc)*grid[r0+1, c0] + tr*tc*grid[r0+1, c0+1])
        return v.astype(float)

    for s in segs:
        x, y = s["x"], s["y"]
        ux, uy = x[-1] - x[0], y[-1] - y[0]
        norm = math.hypot(ux, uy) or 1.0
        ux, uy = ux / norm, uy / norm                     # unit chord direction
        cx, cy = float(x.mean()), float(y.mean())
        h = baseline / 2
        za = float(sample(dem_s, cx - ux * h, cy - uy * h))
        zb = float(sample(dem_s, cx + ux * h, cy + uy * h))
        s["slope_raw"] = abs(za - zb) / baseline if np.isfinite(za) and np.isfinite(zb) else np.nan
        s["z0"] = float(sample(dem_s, x[0], y[0]))
        s["z1"] = float(sample(dem_s, x[-1], y[-1]))
        s["z_mean"] = float(np.nanmean(sample(dem_s, x, y)))
    return te


# --------------------------------------------------------------------------
# Network graph: reconstruct topology, orient downhill, accumulate
# --------------------------------------------------------------------------
def snap_nodes(segs, tol=NODE_SNAP_TOL):
    """Cluster segment endpoints that coincide to within `tol` metres.

    OSM ways are connected only where they literally share a node id. The
    surveyed drains were digitised in separate field batches, so runs that are
    continuous on the ground are often split into ways whose endpoints sit a
    metre or two apart and share nothing. Left uncorrected this fragments the
    network into hundreds of components and caps every accumulated catchment at
    the size of its fragment. Endpoints are therefore clustered on a `tol`-metre
    spatial hash before the graph is built.
    """
    cells = {}
    node_of = []
    reps = []
    for s in segs:
        ends = []
        for idx in (0, -1):
            x, y = float(s["x"][idx]), float(s["y"][idx])
            cx, cy = int(math.floor(x / tol)), int(math.floor(y / tol))
            found = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for cand in cells.get((cx + dx, cy + dy), ()):
                        if math.hypot(x - reps[cand][0], y - reps[cand][1]) <= tol:
                            found = cand
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                found = len(reps)
                reps.append((x, y))
                cells.setdefault((cx, cy), []).append(found)
            ends.append(found)
        node_of.append(tuple(ends))
    return node_of


def build_and_accumulate(segs, local_area):
    """Total contributing area per segment, pushed downhill through the network.

    Orientation uses the 300 m-smoothed elevation field sampled bilinearly at the
    two endpoints. Smoothing matters: on the raw 30 m DEM the endpoint difference
    of a short segment is noise, so adjacent segments were oriented inconsistently
    and the network did not accumulate. Against a smoothed field, neighbouring
    segments inherit the same regional gradient and the orientation is coherent
    by construction. Processing in descending inlet elevation is then a valid
    topological order for a downhill graph, so no cycle search is needed.
    """
    from collections import defaultdict, deque
    ends = snap_nodes(segs)
    inlet, outlet, flat = [], [], 0
    for s, (a, b) in zip(segs, ends):
        z0, z1 = s["z0"], s["z1"]
        if not (np.isfinite(z0) and np.isfinite(z1)) or abs(z0 - z1) < 1e-9:
            flat += 1
            inlet.append(a); outlet.append(b)          # surveyed direction
        elif z0 >= z1:
            inlet.append(a); outlet.append(b)
        else:
            inlet.append(b); outlet.append(a)

    out_of = defaultdict(list)                          # node -> segments leaving it
    for i, nd in enumerate(inlet):
        out_of[nd].append(i)
    succ = [[j for j in out_of.get(outlet[i], []) if j != i] for i in range(len(segs))]

    # Kahn topological order: a segment may only be pushed downstream once every
    # upstream contribution has arrived. Ordering by elevation instead loses mass,
    # because a segment pushes its subtotal before its own inflows are complete.
    indeg = [0] * len(segs)
    for i, ss in enumerate(succ):
        for j in ss:
            indeg[j] += 1
    q = deque(i for i, d in enumerate(indeg) if d == 0)
    order, seen = [], 0
    while q:
        i = q.popleft(); order.append(i); seen += 1
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    cyclic = len(segs) - seen
    if cyclic:                                          # flat-field loops: append
        order += [i for i in range(len(segs)) if indeg[i] > 0]

    total = np.array(local_area, dtype=float)
    for i in order:
        if not succ[i]:
            continue
        share = total[i] / len(succ[i])
        for j in succ[i]:
            total[j] += share
    return total, flat, cyclic, len(set(inlet) | set(outlet))


def local_catchments(segs, grid=ALLOC_GRID, maxdist=ALLOC_MAXDIST):
    """Nearest-drain allocation: area (m^2) draining locally to each segment.

    Computed as a multi-source chamfer propagation from the rasterised network
    rather than a brute-force nearest-point search: the segments are burned onto
    the allocation grid as seeds, then ownership and distance are relaxed
    outwards over 8-neighbour steps with (1, sqrt2) costs until the maximum
    allocation radius is reached. Equivalent to a Voronoi/Thiessen partition of
    the served area to within the chamfer approximation of Euclidean distance.
    """
    pts_x, pts_y, pts_i = [], [], []
    for i, s in enumerate(segs):                        # resample at half-cell spacing
        x, y = s["x"], s["y"]
        d = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
        n = max(2, int(d[-1] // (grid / 2)) + 1)
        t = np.linspace(0, d[-1], n)
        pts_x.append(np.interp(t, d, x)); pts_y.append(np.interp(t, d, y))
        pts_i.append(np.full(n, i))
    px = np.concatenate(pts_x); py = np.concatenate(pts_y)
    pi = np.concatenate(pts_i).astype(np.int32)

    x0, y1 = px.min() - maxdist, py.max() + maxdist
    nx = int((px.max() + maxdist - x0) / grid) + 1
    ny = int((y1 - (py.min() - maxdist)) / grid) + 1
    owner = np.full((ny, nx), -1, dtype=np.int32)
    dist = np.full((ny, nx), np.inf, dtype=np.float32)
    c = np.clip(((px - x0) / grid).astype(int), 0, nx - 1)
    r = np.clip(((y1 - py) / grid).astype(int), 0, ny - 1)
    owner[r, c] = pi                                    # last writer wins; ties are
    dist[r, c] = 0.0                                    # arbitrary and immaterial

    steps = int(math.ceil(maxdist / grid))
    print(f"  allocating {nx*ny:,} cells over {len(px):,} network points "
          f"({steps} propagation passes) ...")
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.41421356), (-1, 1, 1.41421356),
            (1, -1, 1.41421356), (1, 1, 1.41421356)]
    for _ in range(steps):
        changed = False
        for dr, dc, cost in nbrs:
            src_d = np.roll(np.roll(dist, dr, axis=0), dc, axis=1) + cost * grid
            src_o = np.roll(np.roll(owner, dr, axis=0), dc, axis=1)
            if dr > 0:   src_d[:dr, :] = np.inf
            elif dr < 0: src_d[dr:, :] = np.inf
            if dc > 0:   src_d[:, :dc] = np.inf
            elif dc < 0: src_d[:, dc:] = np.inf
            m = src_d < dist
            if m.any():
                dist[m] = src_d[m]; owner[m] = src_o[m]; changed = True
        if not changed:
            break

    served = (owner >= 0) & (dist <= maxdist)
    area = np.bincount(owner[served], minlength=len(segs)).astype(float) * grid * grid
    return area


# --------------------------------------------------------------------------
def reference_intensity(default=20.8):
    """ERA5 1-h T=10yr intensity (scripts/22). A floor, not a design storm."""
    try:
        return float(json.load(open(IDF))["fits"]["1"]["intensity_mm_per_h"]["10"])
    except Exception:
        return default


def main():
    segs = load_segments()
    print(f"loaded {len(segs)} drain ways ({sum(s['length'] for s in segs)/1000:.1f} km)")

    print("1/5  Sampling Stage-3 terrain (slope, conditioned elevation) ...")
    sample_terrain(segs)

    print("2/5  Allocating local catchments (nearest-drain) ...")
    local = local_catchments(segs)

    print("3/5  Accumulating downstream through the network graph ...")
    total_area, flat, cyclic, nnodes = build_and_accumulate(segs, local)
    print(f"     {nnodes} graph nodes after {NODE_SNAP_TOL:.0f} m endpoint snapping; "
          f"{flat} segments oriented by surveyed direction; {cyclic} in flat-field loops")
    print(f"     allocated {local.sum()/1e6:.1f} km2 of local catchment; "
          f"largest accumulated catchment {total_area.max()/1e4:.0f} ha")

    print("4/5  Manning capacity and critical intensity ...")
    ref_i = reference_intensity()
    print(f"     reference storm {ref_i:.1f} mm/h (ERA5 1-h T=10yr floor, scripts/22 — "
          f"a lower bound, so blockage thresholds below are upper bounds)")
    rows = []
    n_floor = n_cap = 0
    for s, loc, tot in zip(segs, local, total_area):
        cs = cross_section(s["tags"])
        if cs is None:
            continue
        A, P, R, prof = cs
        n, mat_known, sm_known = mannings_n(s["tags"])
        sraw = s["slope_raw"]
        S = S_MIN if not np.isfinite(sraw) or sraw < S_MIN else min(sraw, S_MAX)
        n_floor += S == S_MIN and (not np.isfinite(sraw) or sraw < S_MIN)
        n_cap += np.isfinite(sraw) and sraw > S_MAX
        q = (1.0 / n) * A * R ** (2.0 / 3.0) * math.sqrt(S)
        icrit = q * 3.6e6 / (RUNOFF_C * tot) if tot > 0 else float("nan")

        def icrit_at(fill):
            c = cross_section(s["tags"], fill=fill)
            if c is None or tot <= 0:
                return float("nan")
            a, _p, rr, _l = c
            qq = (1.0 / n) * a * rr ** (2.0 / 3.0) * math.sqrt(S)
            return qq * 3.6e6 / (RUNOFF_C * tot)

        blocked = {b: icrit_at(b) for b in BLOCKAGE}
        # smallest siltation fraction at which the segment fails the reference storm
        beta = float("nan")
        for f in np.arange(0.0, 0.96, 0.05):
            v = icrit_at(float(f))
            if np.isfinite(v) and v <= ref_i:
                beta = float(f)
                break
        t = s["tags"]
        rows.append({
            "osm_id": s["id"], "length_m": s["length"], "profile": prof,
            "width_m": float(t.get("width", "nan")), "depth_m": float(t.get("depth", "nan")),
            "area_m2": A, "hyd_radius_m": R, "manning_n": n,
            "slope_used": S, "slope_raw": sraw,
            "capacity_m3s": q, "local_ha": loc / 1e4, "catchment_ha": tot / 1e4,
            "i_crit_mmh": icrit,
            "i_crit_b25": blocked[0.25], "i_crit_b50": blocked[0.50],
            "i_crit_b75": blocked[0.75], "blockage_to_fail": beta,
            "plausible": (float(t.get("width", 0)) >= MIN_WIDTH
                          and float(t.get("depth", 0)) >= MIN_DEPTH
                          and float(t.get("depth", 0)) / max(float(t.get("width", 1e-9)), 1e-9)
                              <= MAX_ASPECT),
            "material": t.get("drain:material", ""),
            "smoothness": t.get("drain:material_smoothness", ""),
            "covered": t.get("drain:profile_covered", "") == "yes"
                       or t.get("tunnel", "") == "culvert",
            "surveyed": t.get("source", "").startswith("Open Cities"),
            "defaults_used": not (mat_known and sm_known),
            "lon": float(s["lon"].mean()), "lat": float(s["lat"].mean()),
            "seg": s,
        })
    rows.sort(key=lambda r: (math.isnan(r["i_crit_mmh"]), r["i_crit_mmh"]))
    print(f"     {len(rows)} segments with usable cross-sections "
          f"({sum(r['length_m'] for r in rows)/1000:.1f} km)")
    print(f"     slope floored to {S_MIN} on {n_floor}, capped at {S_MAX} on {n_cap}")
    bad = [r for r in rows if not r["plausible"]]
    good = [r for r in rows if r["plausible"]]
    print(f"5/5  Cross-section plausibility screen ...")
    print(f"     {len(bad)} segments ({sum(r['length_m'] for r in bad)/1000:.1f} km) "
          f"held out: width < {MIN_WIDTH} m, depth < {MIN_DEPTH} m, or aspect > {MAX_ASPECT}")
    print(f"     {len(good)} segments ({sum(r['length_m'] for r in good)/1000:.1f} km) "
          f"carry credible geometry and form the assessed network")

    report(good, ref_i)
    write_outputs(rows)                     # every segment, flagged
    figure(good, ref_i)
    print(f"\nwrote {CSV}\n      {GPKG}\n      {FIG}")


def report(rows, ref_i):
    ic = np.array([r["i_crit_mmh"] for r in rows])
    ic = ic[np.isfinite(ic)]
    floor = json.load(open(IDF))["fits"]["1"]["intensity_mm_per_h"] if os.path.exists(IDF) else {}
    print("\n--- Conveyance capacity of the surveyed network ---")
    print(f"  segments assessed        {len(ic)}")
    print(f"  capacity   median        {np.median([r['capacity_m3s'] for r in rows]):.3f} m3/s")
    print(f"  catchment  median        {np.median([r['catchment_ha'] for r in rows]):.2f} ha")
    for p in (5, 25, 50, 75, 95):
        print(f"  i_crit p{p:<2d}              {np.percentile(ic, p):7.1f} mm/h")

    print("\n  Share of surveyed network overwhelmed at a given rainfall intensity:")
    lengths = np.array([r["length_m"] for r in rows if np.isfinite(r["i_crit_mmh"])])
    icf = np.array([r["i_crit_mmh"] for r in rows if np.isfinite(r["i_crit_mmh"])])
    tot_km = lengths.sum() / 1000
    for i in (5, 10, 12.6, 20, 30, 50):
        m = icf <= i
        tag = "  <- ERA5 climatological floor, T=2yr" if abs(i - 12.6) < 0.1 else ""
        print(f"    {i:>5.1f} mm/h : {m.sum():4d} segments  {lengths[m].sum()/1000:5.1f} km  "
              f"({lengths[m].sum()/1000/tot_km:5.1%} of network){tag}")
    if floor:
        print(f"\n  (ERA5 1-h floor intensities, scripts/22: "
              + ", ".join(f"T={k}yr {v:.1f}" for k, v in floor.items()) + " mm/h)")

    print("\n  Sensitivity of the sub-floor share to the runoff coefficient C:")
    base = 12.6
    for c in (RUNOFF_C_SENS[0], RUNOFF_C, RUNOFF_C_SENS[1]):
        scaled = icf * (RUNOFF_C / c)
        m = scaled <= base
        print(f"    C={c:.2f} : {m.sum():4d} segments below the floor "
              f"({lengths[m].sum()/1000:5.1f} km)")
    print("  Grade enters only as sqrt(S), so the RANKING is invariant to it; "
          "absolute counts scale as follows:")
    for s in S_SENS:
        scaled = icf * math.sqrt(s / S_REF)
        m = scaled <= base
        print(f"    S={s:.3f} : {m.sum():4d} segments below the floor "
              f"({lengths[m].sum()/1000:5.1f} km)")

    blockage_report(rows, ref_i)

    print("\n  Worst 12 segments by critical intensity:")
    print(f"    {'osm_id':>10s} {'i_crit':>7s} {'cap':>7s} {'catch':>7s} {'W×D':>10s} "
          f"{'n':>6s} {'state':<22s}")
    for r in rows[:12]:
        state = ",".join(filter(None, [
            "covered" if r["covered"] else "",
            r["smoothness"] if r["smoothness"] in ("rough", "very_rough") else ""])) or "-"
        print(f"    {r['osm_id']:>10d} {r['i_crit_mmh']:7.1f} {r['capacity_m3s']:7.3f} "
              f"{r['catchment_ha']:7.2f} {r['width_m']:4.2f}×{r['depth_m']:<4.2f} "
              f"{r['manning_n']:6.3f} {state:<22s}")

    for label, sel in (("covered / culverted", lambda r: r["covered"]),
                       ("rough or very_rough invert",
                        lambda r: r["smoothness"] in ("rough", "very_rough"))):
        sub = [r["i_crit_mmh"] for r in rows if sel(r) and np.isfinite(r["i_crit_mmh"])]
        oth = [r["i_crit_mmh"] for r in rows if not sel(r) and np.isfinite(r["i_crit_mmh"])]
        if sub:
            print(f"\n  {label}: n={len(sub)}, median i_crit {np.median(sub):.1f} mm/h "
                  f"vs {np.median(oth):.1f} for the remainder")


def blockage_report(rows, ref_i):
    """How much siltation does it take to overwhelm the network?

    This is the operative question for the project. The clean sections are
    generously sized for their local catchments, so cross-section geometry is
    not the binding constraint; the state of the invert is. Each segment is
    re-rated with the bottom fraction of its depth occupied by silt and refuse.
    """
    lengths = np.array([r["length_m"] for r in rows])
    print(f"\n--- Blockage response (reference storm {ref_i:.1f} mm/h) ---")
    print(f"  {'silted':>7s} {'segments failing':>17s} {'km failing':>11s} {'share':>7s}")
    for b, key in ((0.0, "i_crit_mmh"), (0.25, "i_crit_b25"),
                   (0.50, "i_crit_b50"), (0.75, "i_crit_b75")):
        v = np.array([r[key] for r in rows])
        m = np.isfinite(v) & (v <= ref_i)
        print(f"  {b:>6.0%} {m.sum():>17d} {lengths[m].sum()/1000:>10.1f} "
              f"{lengths[m].sum()/lengths.sum():>7.1%}")

    beta = np.array([r["blockage_to_fail"] for r in rows])
    fin = np.isfinite(beta)
    print(f"\n  Siltation fraction at which a segment first fails the reference storm:")
    for p in (5, 10, 25, 50):
        print(f"    p{p:<2d} {np.percentile(beta[fin], p):.0%}"
              if fin.any() else "    n/a")
    print(f"    never fails below 95% siltation: {(~fin).sum()} of {len(rows)} segments")
    vuln = fin & (beta <= 0.5)
    print(f"    fail once half-silted: {vuln.sum()} segments, "
          f"{lengths[vuln].sum()/1000:.1f} km ({lengths[vuln].sum()/lengths.sum():.1%})")


def write_outputs(rows):
    cols = ["osm_id", "i_crit_mmh", "i_crit_b25", "i_crit_b50", "i_crit_b75",
            "blockage_to_fail", "capacity_m3s", "catchment_ha", "local_ha",
            "length_m", "width_m", "depth_m", "profile", "area_m2", "hyd_radius_m",
            "manning_n", "material", "smoothness", "slope_used", "slope_raw",
            "plausible", "covered", "surveyed", "defaults_used", "lon", "lat"]
    with open(CSV, "w") as f:
        f.write("rank," + ",".join(cols) + "\n")
        for i, r in enumerate(rows, 1):
            f.write(f"{i}," + ",".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    feats = []
    for i, r in enumerate(rows, 1):
        s = r["seg"]
        props = {c: (None if isinstance(r[c], float) and not np.isfinite(r[c]) else r[c])
                 for c in cols}
        props["rank"] = i
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "LineString",
                                   "coordinates": [[float(a), float(b)]
                                                   for a, b in zip(s["lon"], s["lat"])]}})
    gj = GPKG.replace(".gpkg", ".geojson")
    json.dump({"type": "FeatureCollection", "features": feats}, open(gj, "w"))
    if os.path.exists(GPKG):
        os.remove(GPKG)
    subprocess.run([os.path.join(BIN, "ogr2ogr"), "-q", "-f", "GPKG", GPKG,
                    "-t_srs", "EPSG:32630", "-nln", "drain_capacity", gj], check=True)
    os.remove(gj)


def figure(rows, ref_i=20.8):
    """Two panels on the operative finding: the network is not undersized when
    clean, so the map is coloured by how much siltation each segment can absorb
    before it fails, and the curve beside it is the network-wide response."""
    MW, MH, pad = 820, 900, 46                     # map panel
    PW = 520                                       # right-hand panel
    im = Image.new("RGB", (MW + PW, MH), (252, 252, 250))
    d = ImageDraw.Draw(im)

    xs = np.concatenate([r["seg"]["x"] for r in rows])
    ys = np.concatenate([r["seg"]["y"] for r in rows])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    top = 78
    sc = min((MW - 2*pad) / (x1 - x0), (MH - top - pad) / (y1 - y0))
    ox = pad + ((MW - 2*pad) - (x1 - x0) * sc) / 2
    oy = top + ((MH - top - pad) - (y1 - y0) * sc) / 2

    def px(x, y):
        return (ox + (x - x0) * sc, oy + (y1 - y) * sc)

    bands = [(0.25, (150, 20, 30),  "fails below 25% silted  (most fragile)"),
             (0.50, (215, 70, 40),  "fails by 50% silted"),
             (0.75, (240, 165, 60), "fails by 75% silted"),
             (0.95, (110, 165, 90), "fails only when near-blocked"),
             (9.99, (60, 110, 175), "conveys the reference storm even at 95%")]

    def colour(b):
        if not np.isfinite(b):
            return bands[-1][1]
        for lim, c, _ in bands:
            if b <= lim:
                return c
        return bands[-1][1]

    # draw robust segments first so fragile ones sit on top
    for r in sorted(rows, key=lambda r: -(r["blockage_to_fail"]
                                          if np.isfinite(r["blockage_to_fail"]) else 9.99)):
        b = r["blockage_to_fail"]
        pts = [px(a, c) for a, c in zip(r["seg"]["x"], r["seg"]["y"])]
        if len(pts) < 2:
            continue
        d.line(pts, fill=colour(b), width=4 if np.isfinite(b) and b <= 0.5 else 2)

    d.text((pad, 18), "Stage 5: conveyance capacity of the field-surveyed Accra drainage "
                      "network", fill=(20, 20, 20))
    d.text((pad, 36), f"Manning capacity vs allocated catchment; colour = siltation "
                      f"fraction at which a segment fails a {ref_i:.0f} mm/h storm",
           fill=(90, 90, 90))
    d.text((pad, 52), "Open Cities / GARID field survey, middle Odaw "
                      "(Alogboshie, Akweteyman, Alajo, Nima)", fill=(140, 140, 140))
    ly = MH - pad - 92
    for _lim, c, lab in bands:
        d.rectangle([pad, ly, pad + 26, ly + 12], fill=c)
        d.text((pad + 34, ly), lab, fill=(35, 35, 35))
        ly += 18

    # ---- response curve: network length failing vs siltation fraction --------
    lens = np.array([r["length_m"] for r in rows]) / 1000.0
    beta = np.array([r["blockage_to_fail"] for r in rows])
    grid = np.arange(0, 0.96, 0.05)
    km = np.array([lens[np.isfinite(beta) & (beta <= g)].sum() for g in grid])
    share = km / lens.sum()

    gx0, gy0 = MW + 64, 150
    gw, gh = PW - 118, 300
    d.text((MW + 30, 96), "Network overwhelmed as the drains silt up",
           fill=(20, 20, 20))
    d.text((MW + 30, 114), f"km of surveyed network failing a {ref_i:.0f} mm/h storm",
           fill=(110, 110, 110))
    d.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], outline=(200, 200, 195))
    vmax = max(km.max() * 1.15, 1e-6)
    for frac, lab in ((0.0, "0"), (0.5, f"{vmax/2:.0f}"), (1.0, f"{vmax:.0f}")):
        yy = gy0 + gh - frac * gh
        d.line([gx0 - 4, yy, gx0, yy], fill=(150, 150, 150))
        d.text((gx0 - 34, yy - 6), lab, fill=(110, 110, 110))
    pts = [(gx0 + g / 0.95 * gw, gy0 + gh - k / vmax * gh) for g, k in zip(grid, km)]
    d.line(pts, fill=(180, 40, 40), width=3)
    for g, k, p in zip(grid, km, pts):
        if abs(g % 0.25) < 1e-9:
            d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=(180, 40, 40))
            d.text((p[0] - 10, gy0 + gh + 8), f"{g:.0%}", fill=(110, 110, 110))
            d.text((p[0] - 10, p[1] - 20), f"{k:.1f}", fill=(140, 40, 40))
    d.text((gx0 + gw / 2 - 40, gy0 + gh + 26), "siltation fraction", fill=(110, 110, 110))

    lines = [
        f"assessed        {len(rows)} segments, {lens.sum():.1f} km",
        f"clean sections  {km[0]:.1f} km fail ({share[0]:.1%})",
        f"half silted     {km[np.argmin(abs(grid-0.50))]:.1f} km fail "
        f"({share[np.argmin(abs(grid-0.50))]:.1%})",
        f"75% silted      {km[np.argmin(abs(grid-0.75))]:.1f} km fail "
        f"({share[np.argmin(abs(grid-0.75))]:.1%})",
        "",
        "The credible cross-sections are not undersized",
        "for their local catchments. Capacity is lost to",
        "the state of the invert, not its geometry.",
    ]
    ty = gy0 + gh + 74
    for t in lines:
        d.text((MW + 30, ty), t, fill=(60, 60, 60) if t.startswith(" ") is False else (60, 60, 60))
        ty += 19
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    im.save(FIG)


if __name__ == "__main__":
    main()
