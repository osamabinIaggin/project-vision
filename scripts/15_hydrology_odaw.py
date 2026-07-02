#!/usr/bin/env python3
"""
Stage-3 — hydrological terrain analysis of the Odaw basin from FABDEM.

Derives the layers that turn "where is the city built" (Stage 2) into "where
does water concentrate" over the catchment that floods central Accra:

  * conditioned DEM        — least-cost depression breaching (WhiteboxTools);
  * D8 flow accumulation   — upstream contributing area per cell;
  * stream network         — accumulation threshold, vectorised;
  * HAND                   — height above nearest drainage, the primary
                             flood-susceptibility covariate (low HAND floods
                             first; cf. Nobre et al. 2011);
  * TWI                    — topographic wetness index ln(SCA/tan(slope)).

The run window covers the full Odaw catchment from the Akwapim foothills to
the Korle Lagoon outfall, so accumulation at the Old Fadama AOI integrates
the true upstream area rather than a cropped fragment.

As an immediate sanity/validation probe, HAND and accumulation are sampled at
sites documented as flooded on 3 June 2015 (ReliefWeb FL-2015-000065-GHA) and
at high-ground control sites; the flooded sites should separate cleanly.

Requires: FABDEM tiles (scripts/14), GDAL binaries (scripts/_env.sh), and the
`whitebox` Python package. Run via:
    source scripts/_env.sh && .venv/bin/python scripts/15_hydrology_odaw.py
"""
import os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "accra_flood", "data")
OUT = os.path.join(ROOT, "accra_flood", "output", "hydro")
BIN = os.environ.get("BIN", "")
TILES = [os.path.join(DATA, t) for t in
         ("N05W001_FABDEM_V1-2.tif", "N06W001_FABDEM_V1-2.tif")]

# Odaw catchment window (WGS84): Akwapim foothills to the Korle outfall.
WEST, SOUTH, EAST, NORTH = -0.35, 5.50, -0.10, 5.95
STREAM_THRESHOLD_CELLS = 3000            # 3000 x 900 m^2 = 2.7 km^2 support

SITES = {                                # (lat, lon, flooded on 2015-06-03)
    "Old Fadama (AOI centre)":     (5.5475, -0.2226, True),
    "Kwame Nkrumah Circle":        (5.5715, -0.2087, True),
    "Alajo":                       (5.5845, -0.2145, True),
    "Kaneshie":                    (5.5651, -0.2286, True),
    "Airport residential (ctrl)":  (5.6050, -0.1780, False),
    "Legon hill (ctrl)":           (5.6510, -0.1870, False),
}


def gdal(tool, *args):
    subprocess.run([os.path.join(BIN, tool), *map(str, args)], check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    dem = os.path.join(OUT, "odaw_fabdem_utm.tif")

    print("1/6  Mosaic + clip + reproject FABDEM to EPSG:32630 @ 30 m ...")
    vrt = os.path.join(OUT, "_fabdem.vrt")
    gdal("gdalbuildvrt", "-q", vrt, *TILES)
    gdal("gdalwarp", "-q", "-overwrite", "-t_srs", "EPSG:32630", "-tr", 30, 30,
         "-r", "bilinear", "-te", WEST, SOUTH, EAST, NORTH, "-te_srs", "EPSG:4326",
         "-co", "COMPRESS=DEFLATE", vrt, dem)

    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(OUT)
    wbt.set_verbose_mode(False)

    print("2/6  Breach depressions (least-cost) ...")
    wbt.breach_depressions_least_cost("odaw_fabdem_utm.tif", "dem_breached.tif",
                                      dist=100, fill=True)
    print("3/6  D8 pointer + flow accumulation ...")
    wbt.d8_pointer("dem_breached.tif", "d8_pointer.tif")
    wbt.d8_flow_accumulation("dem_breached.tif", "flow_accum.tif", out_type="cells")
    print("4/6  Stream network + HAND ...")
    wbt.extract_streams("flow_accum.tif", "streams.tif",
                        threshold=STREAM_THRESHOLD_CELLS)
    wbt.raster_streams_to_vector("streams.tif", "d8_pointer.tif", "streams.shp")
    wbt.elevation_above_stream("dem_breached.tif", "streams.tif", "hand.tif")
    print("5/6  Wetness index ...")
    wbt.slope("dem_breached.tif", "slope.tif")
    wbt.fd8_flow_accumulation("dem_breached.tif", "sca.tif",
                              out_type="specific contributing area")
    wbt.wetness_index("sca.tif", "slope.tif", "twi.tif")

    print("6/6  Probe 2015 flood sites vs high-ground controls ...")
    probe()


def probe():
    """Sample HAND / accumulation at the documented sites via gdallocationinfo."""
    print(f"{'site':30s} {'flooded':7s} {'HAND m':>7s} {'accum cells':>12s} {'TWI':>6s}")
    for name, (lat, lon, flooded) in SITES.items():
        vals = {}
        for key, ras in (("hand", "hand.tif"), ("acc", "flow_accum.tif"), ("twi", "twi.tif")):
            r = subprocess.run([os.path.join(BIN, "gdallocationinfo"),
                                "-valonly", "-wgs84", os.path.join(OUT, ras),
                                str(lon), str(lat)],
                               capture_output=True, text=True, check=True)
            vals[key] = float(r.stdout.strip() or "nan")
        print(f"{name:30s} {str(flooded):7s} {vals['hand']:7.2f} {vals['acc']:12.0f} {vals['twi']:6.2f}")


if __name__ == "__main__":
    main()
