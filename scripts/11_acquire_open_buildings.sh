#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Acquire Google Open Buildings v3 footprints for the Old Fadama AOI.
#
# An independent, satellite-derived (~50 cm) building-footprint dataset used to
# audit the OSM training labels (scripts/12). The v3 release is partitioned by
# S2 cell; level-6 token 0fdf covers Accra. The full cell (~420 MB gzip) is
# streamed and filtered to a buffered WGS84 bounding box of the Stage-2 tiling
# window (scripts/05: EPSG:32630 807218,613388 .. 808242,614412), then converted
# to a GeoPackage in EPSG:32630 via a WKT-geometry VRT.
#
# Attribution: Google Open Buildings v3, CC BY 4.0 / ODbL.
#   https://sites.research.google/open-buildings/
# ----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${AOI_DIR}/open_buildings"
mkdir -p "${OUT}"
URL="https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_6_gzip_no_header/0fdf_buildings.csv.gz"
RAW="${OUT}/0fdf_buildings.csv.gz"

# AOI window in WGS84 (+~100 m buffer): lat 5.5418..5.5531, lon -0.2283..-0.2170
if [ ! -s "${OUT}/aoi_buildings.csv" ]; then
  echo "Downloading Open Buildings cell 0fdf (~420 MB) ..."
  curl -sS --retry 3 -C - -o "${RAW}" "${URL}"
  echo "Filtering to the AOI bounding box ..."
  gunzip -c "${RAW}" | awk -F, '
    BEGIN{print "latitude,longitude,area_in_meters,confidence,geometry,full_plus_code"}
    ($1+0)>=5.5418 && ($1+0)<=5.5531 && ($2+0)>=-0.2283 && ($2+0)<=-0.2170' \
    > "${OUT}/aoi_buildings.csv"
  rm -f "${RAW}"
fi

cat > "${OUT}/aoi_buildings.vrt" <<'EOF'
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

rm -f "${OUT}/open_buildings.gpkg"
"${BIN}/ogr2ogr" -q -f GPKG "${OUT}/open_buildings.gpkg" -t_srs EPSG:32630 \
  -nln buildings "${OUT}/aoi_buildings.vrt"
"${BIN}/ogrinfo" -q -sql \
  "SELECT COUNT(*) n, AVG(confidence) mean_conf, AVG(area_in_meters) mean_m2 FROM buildings" \
  "${OUT}/open_buildings.gpkg" | tail -4
echo "Done: ${OUT}/open_buildings.gpkg"
