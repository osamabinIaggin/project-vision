#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-2 corpus extension: the two independent footprint sources over the
# middle-Odaw AOI, mirroring the pair that Old Fadama's consensus supervision
# was built from (scripts/03 and scripts/11).
#
#   * OpenStreetMap (ODbL) — dense, but with per-building offsets and merged
#     shacks in the dense blocks; retrieved over the union of the four Open
#     Cities scene footprints;
#   * Google Open Buildings v3 (CC-BY 4.0 / ODbL) — cleaner per-building
#     geometry, but under-detects in the densest fabric. Accra falls in S2
#     level-6 cell 0fdf, the same cell already used for Old Fadama.
#
# Neither source is trusted alone. Pixels on which the two agree constitute the
# verified supervision (scripts/13); pixels on which they disagree are marked
# ignore. Reproducing that construction on a second, morphologically distinct
# AOI is what makes the transfer test meaningful rather than a comparison
# against a different grade of label noise.
#
# Union of the four scene footprints (WGS84): -0.2500,5.5721 .. -0.1863,5.6479
# ----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/_env.sh"

AOI="${REPO_ROOT}/accra_flood/middleodaw"
OSM_DIR="${AOI}/osm"; GOB_DIR="${AOI}/open_buildings"
mkdir -p "${OSM_DIR}" "${GOB_DIR}"

W=-0.2500; S=5.5721; E=-0.1863; N=5.6479
BB_SWNE="${S},${W},${N},${E}"

# ---- OpenStreetMap ---------------------------------------------------------
Q="[out:xml][timeout:300];(way[\"building\"](${BB_SWNE});relation[\"building\"](${BB_SWNE});way[\"waterway\"](${BB_SWNE});way[\"highway\"](${BB_SWNE}););(._;>;);out body;"

if [[ ! -s "${OSM_DIR}/middleodaw.osm" ]]; then
  for EP in \
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter" \
    "https://overpass-api.de/api/interpreter" \
    "https://overpass.kumi.systems/api/interpreter"; do
    echo "Querying ${EP} ..."
    if curl -fsSL --max-time 400 -H "User-Agent: project-vision/1.0" \
         "${EP}" --data-urlencode "data=${Q}" -o "${OSM_DIR}/middleodaw.osm" \
       && [[ $(wc -c < "${OSM_DIR}/middleodaw.osm") -gt 100000 ]]; then
      echo "  retrieved $(( $(wc -c < "${OSM_DIR}/middleodaw.osm") / 1048576 )) MB"; break
    fi
  done
fi

echo "Transcribing OSM XML -> GeoPackage ..."
rm -f "${OSM_DIR}/middleodaw_osm.gpkg"
"${BIN}/ogr2ogr" -f GPKG "${OSM_DIR}/middleodaw_osm.gpkg" "${OSM_DIR}/middleodaw.osm"
echo "Buildings: $("${BIN}/ogrinfo" -q -where "building IS NOT NULL" \
  "${OSM_DIR}/middleodaw_osm.gpkg" multipolygons | grep -c OGRFeature)"

# ---- Google Open Buildings v3 ---------------------------------------------
URL="https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_6_gzip_no_header/0fdf_buildings.csv.gz"
# Streamed, not staged. The cell is ~400 MB of which the AOI needs ~6%; piping
# curl through gunzip into the bbox filter avoids the intermediate entirely.
# It also sidesteps a real failure mode met here: a `curl -C -` resume onto a
# partially-written file overshot the true length by one buffer and produced a
# CRC-invalid archive that still looked plausible by size alone.
if [[ ! -s "${GOB_DIR}/aoi_buildings.csv" ]]; then
  echo "Streaming Open Buildings cell 0fdf and filtering to the AOI ..."
  tmp="${GOB_DIR}/.aoi_buildings.csv.part"
  set -o pipefail
  curl -sS --retry 5 --retry-delay 3 "${URL}" \
    | gunzip -c \
    | awk -F, -v s="${S}" -v n="${N}" -v w="${W}" -v e="${E}" '
        BEGIN{print "latitude,longitude,area_in_meters,confidence,geometry,full_plus_code"}
        ($1+0)>=s && ($1+0)<=n && ($2+0)>=w && ($2+0)<=e' > "${tmp}"
  mv "${tmp}" "${GOB_DIR}/aoi_buildings.csv"
  echo "  retained $(( $(wc -l < "${GOB_DIR}/aoi_buildings.csv") - 1 )) footprints"
fi

cat > "${GOB_DIR}/aoi_buildings.vrt" <<'EOF'
<OGRVRTDataSource>
  <OGRVRTLayer name="aoi_buildings">
    <SrcDataSource relativeToVRT="1">aoi_buildings.csv</SrcDataSource>
    <GeometryType>wkbPolygon</GeometryType>
    <LayerSRS>EPSG:4326</LayerSRS>
    <GeometryField encoding="WKT" field="geometry"/>
    <Field name="area_in_meters" type="Real"/>
    <Field name="confidence" type="Real"/>
  </OGRVRTLayer>
</OGRVRTDataSource>
EOF

rm -f "${GOB_DIR}/open_buildings.gpkg"
"${BIN}/ogr2ogr" -q -f GPKG "${GOB_DIR}/open_buildings.gpkg" -t_srs EPSG:32630 \
  -nln buildings "${GOB_DIR}/aoi_buildings.vrt"
"${BIN}/ogrinfo" -q -sql \
  "SELECT COUNT(*) n, AVG(confidence) mean_conf, AVG(area_in_meters) mean_m2 FROM buildings" \
  "${GOB_DIR}/open_buildings.gpkg" | tail -4
echo "Done: ${OSM_DIR}/middleodaw_osm.gpkg and ${GOB_DIR}/open_buildings.gpkg"
