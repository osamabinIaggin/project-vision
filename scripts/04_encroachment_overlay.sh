#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-4 deterministic overlay: enumeration of built structures encroaching
# upon the drainage system. Two complementary encroachment modalities are
# adjudicated, both in the projected CRS (EPSG:32630) so that buffer distances
# are metric:
#
#   (i)  ON-DRAIN     — footprint within a buffer of a mapped drain/canal
#                       *centreline* (the in-settlement channels built over);
#   (ii) RIPARIAN     — footprint within a buffer of a water-body *polygon*
#                       (Korle Lagoon and the Odaw channel), which — unlike a
#                       centreline — correctly resolves bank-side structures.
#
# A sensitivity sweep over the riparian tolerance is reported, since the count
# is a monotone function of the buffer and no single distance is canonical.
#
# Parameters: DRAIN_BUFFER_M (default 5), RIPARIAN_BUFFER_M (default 10).
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
OSM_GPKG="${AOI_DIR}/osm/oldfadama_osm.gpkg"
WORK="${AOI_DIR}/encroachment.gpkg"
DRAIN_BUFFER_M="${DRAIN_BUFFER_M:-5}"
RIPARIAN_BUFFER_M="${RIPARIAN_BUFFER_M:-10}"

echo "Assembling projected working set (EPSG:32630) ..."
rm -f "${WORK}"
"${BIN}/ogr2ogr" -f GPKG "${WORK}" -t_srs EPSG:32630 -where "building IS NOT NULL" -nln buildings   "${OSM_GPKG}" multipolygons
"${BIN}/ogr2ogr" -f GPKG -update "${WORK}" -t_srs EPSG:32630 -where "waterway IN ('drain','canal')"  -nln drains    "${OSM_GPKG}" lines
"${BIN}/ogr2ogr" -f GPKG -update "${WORK}" -t_srs EPSG:32630 -where "natural='water'"                -nln waterpoly "${OSM_GPKG}" multipolygons

q() { "${BIN}/ogrinfo" -q -dialect SQLite -sql "$1" "${WORK}" 2>/dev/null | awk -F'= ' '/c \(/{print $2}'; }

echo ""
echo "=== Encroachment, Old Fadama ==="
printf "  on-drain  (<= %s m of drain/canal):      %s\n" "${DRAIN_BUFFER_M}" \
  "$(q "SELECT COUNT(*) c FROM buildings b WHERE EXISTS(SELECT 1 FROM drains w WHERE ST_Intersects(b.geom,ST_Buffer(w.geom,${DRAIN_BUFFER_M})))")"

echo "  riparian sensitivity sweep (<= d m of water body):"
for d in 5 10 20; do
  printf "      d = %2s m : %s\n" "$d" \
    "$(q "SELECT COUNT(*) c FROM buildings b WHERE EXISTS(SELECT 1 FROM waterpoly p WHERE ST_Intersects(b.geom,ST_Buffer(p.geom,$d)))")"
done

printf "  TOTAL distinct (drain %s m UNION water %s m): %s\n" "${DRAIN_BUFFER_M}" "${RIPARIAN_BUFFER_M}" \
  "$(q "SELECT COUNT(*) c FROM buildings b WHERE EXISTS(SELECT 1 FROM drains w WHERE ST_Intersects(b.geom,ST_Buffer(w.geom,${DRAIN_BUFFER_M}))) OR EXISTS(SELECT 1 FROM waterpoly p WHERE ST_Intersects(b.geom,ST_Buffer(p.geom,${RIPARIAN_BUFFER_M})))")"

# Persist the flagged union as a discrete layer for cartographic inspection.
"${BIN}/ogr2ogr" -f GPKG -update -nln flagged_encroachment "${WORK}" "${WORK}" -dialect SQLite -sql "
  SELECT b.geom AS geom FROM buildings b
  WHERE EXISTS(SELECT 1 FROM drains w    WHERE ST_Intersects(b.geom,ST_Buffer(w.geom,${DRAIN_BUFFER_M})))
     OR EXISTS(SELECT 1 FROM waterpoly p WHERE ST_Intersects(b.geom,ST_Buffer(p.geom,${RIPARIAN_BUFFER_M})))"
echo ""
echo "Flagged structures persisted to layer 'flagged_encroachment' in ${WORK}"
