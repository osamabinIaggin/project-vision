#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Environment resolution shim.
#
# Locates the GDAL/PROJ toolchain bundled within a QGIS distribution and exports
# the ancillary-data paths the command-line utilities require. Sourcing this file
# renders gdalinfo/gdal_translate/gdal_rasterize/ogr2ogr/ogrinfo invocable and
# correctly georeferenced from a non-QGIS shell context.
# ----------------------------------------------------------------------------
set -euo pipefail

QGIS_APP="${QGIS_APP:-$(ls -d /Applications/QGIS*.app 2>/dev/null | head -1)}"
if [[ -z "${QGIS_APP}" || ! -d "${QGIS_APP}" ]]; then
  echo "FATAL: no QGIS application bundle found under /Applications." >&2
  exit 1
fi

export BIN="${QGIS_APP}/Contents/MacOS"
export GDAL_DATA="${QGIS_APP}/Contents/Resources/qgis/gdal"
export PROJ_DATA="${QGIS_APP}/Contents/Resources/qgis/proj"
export PROJ_LIB="${PROJ_DATA}"
export OSM_CONFIG_FILE="${GDAL_DATA}/osmconf.ini"

# Repository-relative anchors
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export AOI_DIR="${REPO_ROOT}/accra_flood/oldfadama"
export DATA_DIR="${REPO_ROOT}/accra_flood/data"

# Canonical Old Fadama bounding box (geographic, WGS84): W S E N
export AOI_BBOX_WSEN="-0.232 5.540 -0.214 5.556"
# Overpass form: S W N E
export AOI_BBOX_SWNE="5.540,-0.232,5.556,-0.214"
