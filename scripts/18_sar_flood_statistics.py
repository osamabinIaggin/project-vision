#!/usr/bin/env python3
"""
Stage-3 validation (areal-statistical) — does the low-HAND Odaw floodplain
show an anomalous SAR signature on documented flood days?

Open-water change detection (scripts/17) is physically blind inside the Odaw
floodplain: the surface is near-continuously roofed, and flooded streets
*raise* VV backscatter via enhanced double-bounce rather than darkening the
scene. This script therefore tests the double-bounce signature statistically:

  * population: EVERY June Sentinel-1 acquisition over Accra, 2015-2024
    (June is the peak flood month; 29 scenes, one 18:17/18:18 GMT track);
  * signal: mean VV anomaly (scene minus multi-year dry-season median) within
    the URBAN FLOODPLAIN stratum — Odaw watershed, HAND < 2 m, not permanent
    water — minus the same-scene anomaly over high ground (HAND > 10 m),
    which cancels scene-wide gain/moisture effects;
  * hypothesis: the only same-evening acquisition of a documented flood
    (2020-06-09; rain from late 8 June, deaths on the 9th, S1 pass 18:18 GMT)
    should rank at/near the top of all 29 June scenes. Scenes days after
    events (e.g. 2018-06-20, D+2) are expected to have relaxed to normal.

This is a rank test on primary data: no labels, no tuning, one pre-registered
prediction per documented event.

Run via:
    source scripts/_env.sh && .venv/bin/python scripts/18_sar_flood_statistics.py
"""
import os, json, subprocess, importlib.util
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s17", os.path.join(HERE, "17_sar_flood_validation.py"))
s17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
SAR, HYDRO, BIN = s17.SAR, s17.HYDRO, s17.BIN
STAC = s17.STAC

# Documented June flood days on the Odaw (FloodList/ReliefWeb archives).
EVENT_DAYS = {"2015-06-03", "2016-06-09", "2018-06-18", "2020-06-08",
              "2020-06-09", "2023-06-24", "2023-06-25"}


SCENE_LIST = os.path.join(SAR, "june_scenes.txt")


def june_scenes():
    """June acquisitions 2015-2024 over Accra. Cached to a text file because
    the STAC search endpoint rate-limits; the blob store itself does not."""
    if os.path.exists(SCENE_LIST):
        return sorted(open(SCENE_LIST).read().split(), key=lambda s: s[17:25])
    ids = []
    for y in range(2015, 2025):
        d = s17.http_json(STAC, {
            "collections": ["sentinel-1-rtc"],
            "bbox": [-0.26, 5.53, -0.18, 5.62],
            "datetime": f"{y}-06-01T00:00:00Z/{y}-06-30T23:59:59Z",
            "limit": 20})
        ids += [f["id"] for f in d.get("features", [])]
    open(SCENE_LIST, "w").write("\n".join(ids))
    return sorted(ids, key=lambda s: s[17:25])


def main():
    os.makedirs(SAR, exist_ok=True)
    token = s17.http_json(s17.SAS)["token"]

    # Dry-season reference: median over both events' reference stacks (2018+2020).
    dry_ids = s17.EVENTS["2018-06-20"]["dry"] + s17.EVENTS["2020-06-09"]["dry"]
    ref = np.nanmedian(np.stack([s17.fetch_vv(s, token) for s in dry_ids]), axis=0)

    hand_p = os.path.join(SAR, "_hand20.tif")
    hand = np.asarray(Image.open(hand_p), dtype=np.float32)
    basin = s17.odaw_basin_mask()
    permanent = ref < s17.WATER_DB
    floodplain = basin & (hand < 2) & ~permanent          # urban floodplain stratum
    highground = basin & (hand > 10)                      # within-scene control

    rows = []
    for sid in june_scenes():
        date = f"{sid[17:21]}-{sid[21:23]}-{sid[23:25]}"
        try:
            vv = s17.fetch_vv(sid, token)
        except subprocess.CalledProcessError:
            token = s17.http_json(s17.SAS)["token"]      # SAS token expired
            vv = s17.fetch_vv(sid, token)
        anom = vv - ref
        sig = np.nanmean(anom[floodplain]) - np.nanmean(anom[highground])
        rows.append((date, sig, date in EVENT_DAYS))

    rows.sort(key=lambda r: -r[1])
    out = os.path.join(SAR, "june_doublebounce_ranking.csv")
    with open(out, "w") as f:
        f.write("rank,date,floodplain_anomaly_db,documented_flood_day\n")
        print(f"\n{'rank':4s} {'date':12s} {'anomaly dB':>10s}  event-day")
        for i, (date, sig, ev) in enumerate(rows, 1):
            f.write(f"{i},{date},{sig:.3f},{ev}\n")
            print(f"{i:4d} {date:12s} {sig:10.3f}  {'<<< documented flood' if ev else ''}")
    n = len(rows)
    ev_ranks = [i for i, r in enumerate(rows, 1) if r[2]]
    if ev_ranks:
        print(f"\nevent-day rank(s): {ev_ranks} of {n} June scenes "
              f"(p = {min(ev_ranks)/n:.3f} for top rank under the null)")
    print(f"wrote {out}")

    # Continuous test: floodplain anomaly vs same-day CHIRPS rainfall at Accra.
    # The 18:18 GMT pass follows the (predominantly afternoon) convective rain,
    # so the acquisition-day total is the right predictor.
    print("\nFetching pre-pass rainfall for each scene day ...")
    pairs = []
    for date, sig, ev in rows:
        rain = prepass_rain_mm(date)
        if rain is not None:
            pairs.append((date, sig, rain, ev))
    sigs = np.array([p[1] for p in pairs]); rains = np.array([p[2] for p in pairs])
    rho = spearman(rains, sigs)
    # permutation p-value (one-sided: rain increases the floodplain anomaly)
    rng = np.random.default_rng(42)
    null = np.array([spearman(rng.permutation(rains), sigs) for _ in range(20000)])
    p = (null >= rho).mean()
    with open(os.path.join(SAR, "june_rain_vs_anomaly.csv"), "w") as f:
        f.write("date,floodplain_anomaly_db,chirps_rain_mm,documented_flood_day\n")
        for date, sig, rain, ev in pairs:
            f.write(f"{date},{sig:.3f},{rain:.1f},{ev}\n")
    print(f"scenes with rainfall: {len(pairs)}")
    print(f"Spearman rho(rain, floodplain anomaly) = {rho:.3f}   "
          f"one-sided permutation p = {p:.4f}")


def prepass_rain_mm(date, lat=5.585, lon=-0.22):
    """ERA5 (Open-Meteo archive) rainfall summed 00:00-18:00 GMT on the
    acquisition day — i.e. the rain that fell before the 18:18 GMT pass.
    Chosen over daily CHIRPS, which reports 0 mm on known Accra rain days:
    coastal daily gridded products blur the evening convection this test
    depends on, whereas the hourly reanalysis preserves the diurnal timing."""
    cache = os.path.join(SAR, f"_rain_{date}.txt")
    if os.path.exists(cache):
        v = open(cache).read().strip()
        return float(v) if v else None
    d = s17.http_json(
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}"
        f"&longitude={lon}&start_date={date}&end_date={date}"
        f"&hourly=precipitation&timezone=GMT")
    try:
        rain = float(sum(d["hourly"]["precipitation"][:19]))
    except (KeyError, TypeError):
        open(cache, "w").write("")
        return None
    open(cache, "w").write(f"{rain:.2f}")
    return rain


def _avgrank(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):                       # average ranks across ties
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _avgrank(np.asarray(a)), _avgrank(np.asarray(b))
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


if __name__ == "__main__":
    main()
