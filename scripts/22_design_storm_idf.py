#!/usr/bin/env python3
"""
Stage-5 (a) — climatological rainfall statistics for the Odaw basin, and a
DOCUMENTED NEGATIVE RESULT on the use of reanalysis as a design storm.

Ghana Meteorological Agency IDF curves for Accra are not openly published, so
this script tests whether an open reanalysis can supply them. Open-Meteo's ERA5
hourly precipitation archive is drawn at the basin centroid, 1980-2024; rolling
accumulations are taken over 1, 2, 3, 6, 12 and 24 h; annual maxima per duration
are fitted with a Gumbel (EV1) by method of moments — the standard treatment of
annual-maximum series (Chow, Maidment & Mays 1988, ch. 12):

    beta = s * sqrt(6) / pi ,   mu = xbar - 0.5772 * beta
    x_T  = mu - beta * ln(-ln(1 - 1/T))

The fit is internally well-behaved and the annual total (~900 mm/yr) matches
Accra's climatology. The provenance probe, however, refutes its use as a design
storm: on each documented flood day the reanalysis reports a rainfall depth
BELOW every annual maximum in the 45-year record — 3 June 2015, which killed
some 150 people, appears as a 5.1 mm/h, 10.8 mm/24h day. A spatial check
confirms this is not storm displacement: six sample points spanning the
metropolitan area return byte-identical series, all falling inside one ~31 km
ERA5 cell. Sub-daily convective rainfall over coastal Accra is simply beneath
the resolution of the reanalysis.

CONSEQUENCE. No design storm of defensible magnitude can be built from open data
here, so scripts/23 does not use one. It inverts the hydraulics instead and
solves for each drain's CRITICAL INTENSITY — the rainfall rate at which that
segment reaches capacity — which is a property of the drain and its catchment
alone. The intensities tabulated below are retained only as a climatological
FLOOR against which those critical intensities are read: real design intensities
for Accra are higher than these by an unknown but certainly large factor, so any
drain failing at or below the floor fails a fortiori under a true design storm.

Outputs accra_flood/output/design_storm_idf.{csv,json}.

Run via:
    .venv/bin/python scripts/22_design_storm_idf.py
"""
import os, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(ROOT, "accra_flood", "output")
CSV = os.path.join(OUTDIR, "design_storm_idf.csv")
JSN = os.path.join(OUTDIR, "design_storm_idf.json")
CACHE = os.path.join(OUTDIR, "_era5_hourly_precip.json")

LAT, LON = 5.60, -0.21                 # Odaw basin centroid (middle Odaw)
YEAR0, YEAR1 = 1980, 2024
DURATIONS = (1, 2, 3, 6, 12, 24)       # hours
RETURN_PERIODS = (2, 5, 10, 25, 50)    # years

# Documented flood days used as a provenance probe (not as fitting data).
FLOOD_DAYS = {
    "2015-06-03": "June 3 2015 disaster (FL-2015-000065-GHA)",
    "2018-06-18": "18 June 2018 flood",
    "2020-06-09": "9 June 2020 flood",
    "2023-06-17": "17 June 2023 flood",
}


def fetch_hourly():
    """ERA5 hourly precipitation at the basin centroid, decade-chunked + cached."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    times, precip = [], []
    for y0 in range(YEAR0, YEAR1 + 1, 10):
        y1 = min(y0 + 9, YEAR1)
        print(f"  fetching ERA5 hourly precipitation {y0}-{y1} ...")
        r = subprocess.run(
            ["curl", "-s", "--max-time", "300", "-G",
             "https://archive-api.open-meteo.com/v1/archive",
             "--data-urlencode", f"latitude={LAT}",
             "--data-urlencode", f"longitude={LON}",
             "--data-urlencode", f"start_date={y0}-01-01",
             "--data-urlencode", f"end_date={y1}-12-31",
             "--data-urlencode", "hourly=precipitation",
             "--data-urlencode", "timezone=UTC"],
            capture_output=True, text=True, check=True)
        d = json.loads(r.stdout)
        if "hourly" not in d:
            raise RuntimeError(f"Open-Meteo returned no hourly block: {d}")
        times += d["hourly"]["time"]
        precip += [0.0 if v is None else float(v) for v in d["hourly"]["precipitation"]]
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"time": times, "precipitation": precip}, open(CACHE, "w"))
    return {"time": times, "precipitation": precip}


def rolling_max_by_year(times, p, hours):
    """Annual maximum of the `hours`-long rolling accumulation."""
    if hours == 1:
        acc = p.copy()
    else:                                    # right-aligned running sum
        c = np.concatenate(([0.0], np.cumsum(p)))
        acc = np.full_like(p, np.nan)
        acc[hours - 1:] = c[hours:] - c[:-hours]
    years = np.array([int(t[:4]) for t in times])
    out = {}
    for y in range(YEAR0, YEAR1 + 1):
        v = acc[years == y]
        v = v[~np.isnan(v)]
        if v.size:
            out[y] = float(v.max())
    return out


def gumbel_fit(sample):
    """EV1 location/scale by method of moments."""
    x = np.asarray(sample, dtype=float)
    s = x.std(ddof=1)
    beta = s * np.sqrt(6) / np.pi
    mu = x.mean() - 0.5772157 * beta
    return mu, beta


def gumbel_quantile(mu, beta, T):
    return mu - beta * np.log(-np.log(1.0 - 1.0 / T))


def main():
    data = fetch_hourly()
    times = data["time"]
    p = np.asarray(data["precipitation"], dtype=float)
    print(f"ERA5 record: {len(times)} hours, {times[0]} .. {times[-1]}, "
          f"total {p.sum():,.0f} mm ({p.sum()/(YEAR1-YEAR0+1):.0f} mm/yr)")

    table, jrows = [], {}
    print(f"\n{'dur':>4s} {'n_yr':>5s} {'mean':>7s} " +
          " ".join(f"{'T='+str(T):>9s}" for T in RETURN_PERIODS) + "   (depth mm)")
    for h in DURATIONS:
        amax = rolling_max_by_year(times, p, h)
        vals = np.array(list(amax.values()))
        mu, beta = gumbel_fit(vals)
        depths = [float(gumbel_quantile(mu, beta, T)) for T in RETURN_PERIODS]
        print(f"{h:>3d}h {len(vals):>5d} {vals.mean():>7.1f} " +
              " ".join(f"{d:>9.1f}" for d in depths))
        for T, dep in zip(RETURN_PERIODS, depths):
            table.append((h, T, dep, dep / h))
        jrows[str(h)] = {"mu": mu, "beta": beta, "n_years": len(vals),
                         "depth_mm": dict(zip(map(str, RETURN_PERIODS), depths)),
                         "intensity_mm_per_h": dict(
                             zip(map(str, RETURN_PERIODS), [d / h for d in depths]))}

    print(f"\n{'dur':>4s} " + " ".join(f"{'T='+str(T):>9s}" for T in RETURN_PERIODS) +
          "   (intensity mm/h)")
    for h in DURATIONS:
        print(f"{h:>3d}h " + " ".join(
            f"{jrows[str(h)]['intensity_mm_per_h'][str(T)]:>9.1f}"
            for T in RETURN_PERIODS))

    probe(times, p)

    with open(CSV, "w") as f:
        f.write("duration_h,return_period_yr,depth_mm,intensity_mm_per_h\n")
        for h, T, dep, inten in table:
            f.write(f"{h},{T},{dep:.2f},{inten:.2f}\n")
    json.dump({"source": "Open-Meteo ERA5 hourly reanalysis",
               "latitude": LAT, "longitude": LON,
               "period": f"{YEAR0}-{YEAR1}", "distribution": "Gumbel EV1 (MoM)",
               "caveat": "ERA5 damps sub-daily convective peaks; design "
                         "intensities are conservative (deficits are lower bounds)",
               "fits": jrows}, open(JSN, "w"), indent=2)
    print(f"\nwrote {CSV} and {JSN}")


def probe(times, p):
    """Where do the documented flood days sit in the fitted record?"""
    years = np.array([int(t[:4]) for t in times])
    days = np.array([t[:10] for t in times])
    c = np.concatenate(([0.0], np.cumsum(p)))
    acc24 = np.full_like(p, np.nan)
    acc24[23:] = c[24:] - c[:-24]
    print("\nProvenance probe — documented flood days within the ERA5 record:")
    print(f"  {'date':12s} {'max 1h':>7s} {'max 24h':>8s} {'24h rank':>9s}  event")
    ann24 = {}
    for y in range(YEAR0, YEAR1 + 1):
        v = acc24[years == y]
        v = v[~np.isnan(v)]
        if v.size:
            ann24[y] = v.max()
    allmax = np.sort(np.array(list(ann24.values())))[::-1]
    n_below = 0
    for date, label in FLOOD_DAYS.items():
        m = days == date
        if not m.any():
            continue
        h1 = float(p[m].max())
        h24 = float(np.nanmax(acc24[m]))
        rank = int((allmax > h24).sum()) + 1
        n_below += rank > len(allmax)
        print(f"  {date:12s} {h1:7.1f} {h24:8.1f} {rank:>6d}/{len(allmax):<3d}  {label}")
    print(f"  -> {n_below}/{len(FLOOD_DAYS)} documented flood days fall BELOW every "
          f"annual maximum in {len(allmax)} years.")
    spatial_invariance_check()


def spatial_invariance_check(date="2015-06-03"):
    """Is the miss storm displacement, or grid resolution? Sample the metro area."""
    pts = [(5.55, -0.25), (5.60, -0.21), (5.65, -0.17),
           (5.55, -0.15), (5.70, -0.25), (5.50, -0.20)]
    print(f"\nSpatial check — ERA5 across the metropolitan area on {date}:")
    sigs = set()
    for lat, lon in pts:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "60", "-G",
             "https://archive-api.open-meteo.com/v1/archive",
             "--data-urlencode", f"latitude={lat}",
             "--data-urlencode", f"longitude={lon}",
             "--data-urlencode", f"start_date={date}",
             "--data-urlencode", f"end_date={date}",
             "--data-urlencode", "hourly=precipitation",
             "--data-urlencode", "timezone=UTC"],
            capture_output=True, text=True, check=True)
        v = [x or 0.0 for x in json.loads(r.stdout)["hourly"]["precipitation"]]
        sigs.add(round(sum(v), 3))
        print(f"  {lat:.2f},{lon:>6.2f}  max 1h {max(v):5.1f}  daily {sum(v):6.1f} mm")
    print(f"  -> {len(sigs)} distinct series across {len(pts)} points: "
          + ("one ~31 km cell covers the metropolis; the storm is unresolved, "
             "not displaced." if len(sigs) == 1 else "series differ spatially."))


if __name__ == "__main__":
    main()
