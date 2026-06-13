# Accra Flood-Susceptibility Exercise (Stage 3)

Goal: from elevation data alone, find where water naturally collects in Accra — the
Odaw catchment should "light up" — with **no machine learning and no training data**.
This teaches the core terrain concepts that the later ML stage will lean on.

## What you have
- `data/cop30_accra_W001.tif` — Copernicus GLO-30 DEM, 30 m resolution, covers
  lat 5–6 N, lon 1 W–0 (all of Accra metro incl. the Odaw/Korle catchment).
- `data/cop30_accra_E000.tif` — eastern tile (Tema side, mostly ocean). Usually not needed.

A **DEM (Digital Elevation Model)** is just a grayscale raster where each pixel's
value = ground elevation in metres. Everything below is derived from it.

## The idea (why this works without ML)
Water flows downhill. A place floods when lots of upstream land drains *into* it
(high flow accumulation) AND it's flat (water can't drain away fast). The
**Topographic Wetness Index (TWI)** captures exactly this:

    TWI = ln( upslope_area / tan(slope) )

- Big upslope area  -> lots of water arriving -> high TWI
- Small slope (flat) -> tan(slope) tiny -> high TWI
High TWI = wet, convergent, flood-prone. That's the whole trick.

---

## Recipe (QGIS)

### 0. Open QGIS and load the DEM
- Drag `data/cop30_accra_W001.tif` into the map canvas.
- It looks like a grey square. Right-click the layer -> Properties -> Symbology ->
  render type "Singleband pseudocolor", pick a colour ramp -> you can now see hills
  (high) vs the coastal plain / lagoon (low).

### 1. Reproject to metres (EPSG:32630, UTM zone 30N)
The DEM is in degrees (EPSG:4326). Slope and area only make sense in metres.
- Processing Toolbox (Ctrl+Alt+T) -> search **"Warp (reproject)"** (GDAL).
- Input: the DEM. Target CRS: **EPSG:32630**. Resampling: **Bilinear**.
- Save output as `output/dem_utm.tif`. Use this for everything below.
- Check Layer Properties -> Information: pixel size should be ~30 (metres).

### 2. Slope (GDAL — always available)
- Processing Toolbox -> **"Slope"** (GDAL).
- Input: `dem_utm.tif`. Leave units = degrees. Output: `output/slope_deg.tif`.

### 3. Flow accumulation (GRASS r.watershed — bundled with QGIS)
- Processing Toolbox -> **"r.watershed"** (GRASS).
- Elevation: `dem_utm.tif`.
- The output you want is **"Number of cells that drain through each cell"**
  (the accumulation raster). Save as `output/accum.tif`.
- Other outputs can be left as temporary/skipped.
- Note: r.watershed handles depressions internally, so no separate "fill sinks"
  step is needed here.

### 4. Combine into TWI (Raster Calculator)
- Raster -> Raster Calculator.
- Pixel size in metres = 30 (cell area handled below). Expression:

    ln( ( abs("accum@1") * 30 ) / tan( ("slope_deg@1" + 0.5) * 3.14159265 / 180 ) )

  Why each piece:
  - `abs("accum@1") * 30`  -> upslope contributing area (cells x cell width).
  - `"slope_deg@1" + 0.5`  -> adds a 0.5 deg floor so perfectly flat pixels don't
    divide by zero (tan(0)=0).
  - `* 3.14159 / 180`      -> degrees to radians (tan expects radians).
- Output: `output/twi.tif`.

### 5. Style and read the result
- twi.tif -> Properties -> Symbology -> Singleband pseudocolor -> ramp e.g. "Spectral"
  (reverse so high = blue). Classify.
- High-TWI pixels trace the **drainage network**: the Odaw channel through Circle,
  Alajo, Kaneshie, into the Korle Lagoon should stand out as bright lines/zones.
- That bright network = where water concentrates = your first flood-proneness map.

---

## Faster shortcut (optional)
Install the **"SAGA Next Gen"** plugin (Plugins -> Manage and Install Plugins),
then run **"SAGA Wetness Index"** with just `dem_utm.tif` as input — it does
fill + flow + slope + TWI in one tool. Good for a quick result; the step-by-step
version above is better for *understanding* the pieces.

## What to notice (the learning payoff)
- TWI lights up valleys/channels before you've told it anything about rivers.
- Compare against where Accra actually floods (Circle, Kaneshie, Alajo, Odaw) — the
  overlap is the proof terrain alone carries most of the signal.
- Later (Stage 3 ML) you feed TWI, slope, elevation, distance-to-stream, land cover,
  etc. as *features* into Random Forest, with past flood locations as labels. This
  exercise is you computing those features by hand so you understand what the model sees.
