#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-5 substrate: acquisition of the field-surveyed drainage network of
# central Accra from OpenStreetMap.
#
# The Open Cities Accra / GARID field campaigns (2018-2020) surveyed the
# drains of the middle-Odaw communities (Alogboshie, Akweteyman, Alajo, Nima)
# segment by segment and committed ENGINEERING ATTRIBUTES to OSM: width,
# depth, top/bottom width, cross-section profile, material, smoothness,
# cover/culvert status (source tag: "Open Cities Accra / GARID- Field
# Survey"). 708 of 1,877 drain ways in the window below carry width+depth —
# sufficient cross-section geometry for Manning-equation conveyance capacity,
# i.e. the hazard-amplifier data (undersized/blocked drains) that no agency
# publishes as a standalone dataset.
#
# Window: central Accra / Odaw basin (5.53..5.65 N, -0.28..-0.15 E).
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
OUT_DIR="${AOI_DIR}/../drains"; mkdir -p "${OUT_DIR}"
OSM_JSON="${OUT_DIR}/accra_drains.json"

Q='[out:json][timeout:180];(way["waterway"="drain"](5.53,-0.28,5.65,-0.15);way["waterway"="ditch"](5.53,-0.28,5.65,-0.15););out tags geom;'

echo "Querying Overpass for the surveyed drainage network ..."
curl -s -G --max-time 200 "https://overpass-api.de/api/interpreter" \
     --data-urlencode "data=${Q}" -o "${OSM_JSON}"
python3 - "$OSM_JSON" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
els = [e for e in d.get("elements", []) if e.get("type") == "way"]
n_geom = sum(1 for e in els if all(k in e.get("tags", {}) for k in ("width", "depth")))
print(f"retrieved {len(els)} drain/ditch ways; {n_geom} with width+depth cross-sections")
EOF

echo "Converting to GeoPackage (EPSG:32630) ..."
python3 - "$OSM_JSON" "${OUT_DIR}/accra_drains.geojson" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
feats = []
for e in d.get("elements", []):
    if e.get("type") != "way" or "geometry" not in e:
        continue
    coords = [[p["lon"], p["lat"]] for p in e["geometry"]]
    feats.append({"type": "Feature",
                  "properties": {"osm_id": e["id"], **e.get("tags", {})},
                  "geometry": {"type": "LineString", "coordinates": coords}})
json.dump({"type": "FeatureCollection", "features": feats}, open(sys.argv[2], "w"))
print(f"wrote {len(feats)} features")
EOF
"${BIN}/ogr2ogr" -q -f GPKG "${OUT_DIR}/accra_drains.gpkg" -t_srs EPSG:32630 \
                 -nln drains "${OUT_DIR}/accra_drains.geojson"
rm -f "${OUT_DIR}/accra_drains.geojson"
echo "Done: ${OUT_DIR}/accra_drains.gpkg"
