#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Label corpus: extraction of the OpenStreetMap building and drainage ontology
# over the AOI via the Overpass API, and its transcription into a GeoPackage.
#
# Note the recursive node-resolution clause '(._;>;);out body;' — the naive
# 'out geom;' inlines coordinates without emitting standalone <node> elements,
# which the GDAL OSM driver requires to reconstruct way geometry.
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
OSM_DIR="${AOI_DIR}/osm"
mkdir -p "${OSM_DIR}"
B="${AOI_BBOX_SWNE}"

Q="[out:xml][timeout:180];(way[\"building\"](${B});relation[\"building\"](${B});way[\"waterway\"](${B});way[\"natural\"=\"water\"](${B});relation[\"natural\"=\"water\"](${B});way[\"highway\"](${B}););(._;>;);out body;"

# The principal Overpass endpoints intermittently throttle; iterate over mirrors.
for EP in \
  "https://maps.mail.ru/osm/tools/overpass/api/interpreter" \
  "https://overpass-api.de/api/interpreter" \
  "https://overpass.kumi.systems/api/interpreter"; do
  echo "Querying ${EP} ..."
  if curl -fsSL --max-time 240 -H "User-Agent: project-vision/1.0" \
       "${EP}" --data-urlencode "data=${Q}" -o "${OSM_DIR}/oldfadama.osm" \
     && [[ $(wc -c < "${OSM_DIR}/oldfadama.osm") -gt 100000 ]]; then
    echo "  retrieved $(( $(wc -c < "${OSM_DIR}/oldfadama.osm") / 1048576 )) MB"; break
  fi
done

echo "Transcribing OSM XML -> GeoPackage ..."
rm -f "${OSM_DIR}/oldfadama_osm.gpkg"
"${BIN}/ogr2ogr" -f GPKG "${OSM_DIR}/oldfadama_osm.gpkg" "${OSM_DIR}/oldfadama.osm"
echo "Buildings: $("${BIN}/ogrinfo" -q -where "building IS NOT NULL" "${OSM_DIR}/oldfadama_osm.gpkg" multipolygons | grep -c OGRFeature)"
echo "Waterways: $("${BIN}/ogrinfo" -q -where "waterway IS NOT NULL" "${OSM_DIR}/oldfadama_osm.gpkg" lines | grep -c OGRFeature)"
