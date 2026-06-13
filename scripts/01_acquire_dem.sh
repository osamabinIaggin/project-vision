#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-3 substrate: acquisition of the Copernicus GLO-30 digital elevation
# model tile subtending metropolitan Accra. Retrieved login-free from the
# public AWS mirror of the Copernicus DEM open archive.
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
mkdir -p "${DATA_DIR}"

BASE="https://copernicus-dem-30m.s3.amazonaws.com"
declare -A TILES=(
  [cop30_accra_W001.tif]="Copernicus_DSM_COG_10_N05_00_W001_00_DEM/Copernicus_DSM_COG_10_N05_00_W001_00_DEM.tif"
  [cop30_accra_E000.tif]="Copernicus_DSM_COG_10_N05_00_E000_00_DEM/Copernicus_DSM_COG_10_N05_00_E000_00_DEM.tif"
)
for out in "${!TILES[@]}"; do
  echo "Retrieving ${out} ..."
  curl -fsSL -o "${DATA_DIR}/${out}" "${BASE}/${TILES[$out]}"
done
echo "DEM acquisition complete; primary tile: ${DATA_DIR}/cop30_accra_W001.tif"
