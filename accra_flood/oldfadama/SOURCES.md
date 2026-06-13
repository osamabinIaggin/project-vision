# Old Fadama / Agbogbloshie — Pilot AOI data sources

Pilot area: Old Fadama / Agbogbloshie, mouth of the Odaw River on the Korle Lagoon, Accra.
The downstream sink of the ~400 km2 Odaw–Korle catchment; epicenter of the June 3 2015 floods.
Bounding box used: W=-0.232, S=5.540, E=-0.214, N=5.556 (lon/lat).

## Imagery — OpenAerialMap (CC-BY, attribution required)

Change-detection pair (what we actually use):
- 2020-05-26, 5 cm, "Old Fadama - Agbogbloshie" (201 MB)
  https://oin-hotosm-temp.s3.amazonaws.com/5ed24f7157676e00057f5843/0/5ed24f7157676e00057f5844.tif
- 2024-08-26, 5 cm, "Oldfadama Agbogloshie_combined_high_compression" (145 MB)
  https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/66e402dfcd0baa0001b62128/0/66e402dfcd0baa0001b62129.tif

Other available (not downloaded):
- 2024-08-27, 5 cm, uncompressed (2.9 GB — skip, identical scene)
- 2024-08-26, 5 cm, low_compression
- 2019-02-11, 30 cm, Maxar Accra Mosaic (city-wide context)
- 2006-06-30, 50 cm, Ghana coastline plate 1 (long-baseline coastal reference)

Local copies: imagery/oldfadama_2020_5cm.tif, imagery/oldfadama_2024_5cm.tif
Full metadata: ../oldfadama/oam_meta.json

## Labels — OpenStreetMap (ODbL)
Buildings, waterways, highways over the AOI, fetched via Overpass API.
Local copy: osm/oldfadama.osm
Likely enriched by the Open Cities Accra / HOTOSM / OSM Ghana mapping effort.

## Why this AOI
Concentrates all flood drivers in one place: wetland encroachment, blocked drains +
solid waste (the choked Odaw sea-outfall culverts), and active GARID dredging.
Two epochs at 5 cm = a 4-year encroachment change-detection window, Stage-1
stitching already done (static orthomosaics, not raw video).

## Caveat
As the basin sink, flooding here is partly driven by upstream + the blocked sea
outfall, not purely local drainage. Extend later to upstream Alogboshie / Alajo /
Akweteyman to avoid a model that only learns "lagoon-mouth slum".
