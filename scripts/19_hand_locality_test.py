#!/usr/bin/env python3
"""
Stage-3 validation (formal) — HAND separation of documented flood localities
from control neighbourhoods, with an exact permutation test.

The 6-site probe (scripts/15) separated 2015 flood sites from high ground but
is too small to state significance. Here the flooded set is every Odaw-basin
locality named in multi-year flood reporting (FloodList; MyJoyOnline's 2026
retrospective "the water still comes to the same places": Circle, Kaneshie,
Odawna, Adabraka, Nima, Alajo, Lapaz, Aboabo — plus Old Fadama, ReliefWeb
FL-2015-000065-GHA), and the control set is elevated residential districts
that do not appear in flood reporting. Localities are geocoded independently
via OSM Nominatim (no manual placement), HAND is sampled at each centroid,
and a one-sided exact Mann-Whitney test is computed by full permutation.

Centroid geocoding is conservative: a locality's centroid may fall on its
slopes rather than its flooded frontage, which adds noise *against* the
hypothesis, not in its favour.

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/19_hand_locality_test.py
"""
import os, json, subprocess, time
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
HYDRO = os.path.join(ROOT, "accra_flood", "output", "hydro")
OUT = os.path.join(ROOT, "accra_flood", "output", "hand_locality_test.csv")
BIN = os.environ.get("BIN", "")

FLOODED = ["Old Fadama", "Kwame Nkrumah Circle", "Odawna", "Adabraka",
           "Kaneshie", "Nima", "Alajo", "Abeka Lapaz", "Aboabo, Accra New Town"]
CONTROL = ["Airport Residential Area", "East Legon", "University of Ghana",
           "Cantonments", "Roman Ridge", "Labone", "Ridge, Accra", "McCarthy Hill"]


def geocode(name):
    url = ("https://nominatim.openstreetmap.org/search?format=json&limit=1"
           f"&q={name.replace(' ', '+')},+Accra,+Ghana")
    r = subprocess.run(["curl", "-s", "--max-time", "30",
                        "-A", "project-vision-research/1.0", url],
                       capture_output=True, text=True, check=True)
    time.sleep(1.2)                                  # Nominatim usage policy
    hits = json.loads(r.stdout)
    if not hits:
        return None
    return float(hits[0]["lat"]), float(hits[0]["lon"])


def hand_at(lat, lon):
    r = subprocess.run([os.path.join(BIN, "gdallocationinfo"), "-valonly",
                        "-wgs84", os.path.join(HYDRO, "hand.tif"),
                        str(lon), str(lat)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def mannwhitney_exact(x, y):
    """One-sided exact MW test: H1 = values in x (flooded) are lower."""
    pooled = np.array(x + y)
    n, m = len(x), len(y)
    def ustat(idx):
        xs = pooled[list(idx)]
        ys = np.delete(pooled, list(idx))
        return sum((xi < yj) + 0.5 * (xi == yj) for xi in xs for yj in ys)
    u_obs = sum((xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    count = total = 0
    for idx in combinations(range(n + m), n):
        total += 1
        if ustat(idx) >= u_obs:
            count += 1
    return u_obs / (n * m), count / total


def main():
    rows = []
    for name, flooded in [(n, True) for n in FLOODED] + [(n, False) for n in CONTROL]:
        loc = geocode(name)
        if loc is None:
            print(f"  ! geocode failed: {name}")
            continue
        h = hand_at(*loc)
        if h is None:
            print(f"  ! outside HAND raster: {name}")
            continue
        rows.append((name, flooded, loc[0], loc[1], h))
        print(f"  {name:28s} {'flood' if flooded else 'ctrl ':5s} "
              f"({loc[0]:.4f},{loc[1]:.4f})  HAND={h:6.2f} m")

    fl = [h for _, f, _, _, h in rows if f]
    ct = [h for _, f, _, _, h in rows if not f]
    auc, p = mannwhitney_exact(fl, ct)
    print(f"\nflooded localities: n={len(fl)}, median HAND {np.median(fl):.2f} m")
    print(f"control localities: n={len(ct)}, median HAND {np.median(ct):.2f} m")
    print(f"AUC (flooded lower) = {auc:.3f}   exact one-sided Mann-Whitney p = {p:.5f}")

    with open(OUT, "w") as f:
        f.write("name,documented_flood_locality,lat,lon,hand_m\n")
        for name, fl_, la, lo, h in rows:
            f.write(f"\"{name}\",{fl_},{la:.5f},{lo:.5f},{h:.2f}\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
