#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-4 deterministic overlay: enumeration of built structures encroaching
# upon the in-settlement drainage network. A structure is adjudged encroaching
# where its footprint intersects a configurable planar buffer about a mapped
# drain or canal. Operates in the projected CRS (EPSG:32630) so that the buffer
# distance is expressed in metres.
#
# Parameter: BUFFER_M (default 5) — the riparian tolerance in metres.
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
OSM_GPKG="${AOI_DIR}/osm/oldfadama_osm.gpkg"
WORK="${AOI_DIR}/encroachment.gpkg"
BUFFER_M="${BUFFER_M:-5}"

echo "Assembling projected working set (EPSG:32630) ..."
rm -f "${WORK}"
"${BIN}/ogr2ogr" -f GPKG "${WORK}" -t_srs EPSG:32630 -where "building IS NOT NULL" \
  -nln buildings "${OSM_GPKG}" multipolygons
"${BIN}/ogr2ogr" -f GPKG -update -t_srs EPSG:32630 -where "waterway IS NOT NULL" \
  -nln water "${WORK}" "${OSM_GPKG}" lines

echo "Computing encroachment at buffer = ${BUFFER_M} m ..."
"${BIN}/ogrinfo" -q -dialect SQLite -sql "
  SELECT COUNT(*) AS structures_on_drains FROM buildings b
  WHERE EXISTS (SELECT 1 FROM water w
                WHERE w.waterway IN ('drain','canal')
                  AND ST_Intersects(b.geom, ST_Buffer(w.geom, ${BUFFER_M})))
" "${WORK}" | grep -i structures_on_drains

# Persist the flagged subset as a discrete layer for cartographic inspection.
"${BIN}/ogr2ogr" -f GPKG -update -nln flagged_encroachment "${WORK}" "${WORK}" \
  -dialect SQLite -sql "
  SELECT b.geom AS geom FROM buildings b
  WHERE EXISTS (SELECT 1 FROM water w
                WHERE w.waterway IN ('drain','canal')
                  AND ST_Intersects(b.geom, ST_Buffer(w.geom, ${BUFFER_M})))"
echo "Flagged structures persisted to layer 'flagged_encroachment' in ${WORK}"
