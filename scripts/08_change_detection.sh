#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Path-A prep: extract a common patch from the 2020 and 2024 orthomosaics onto
# an IDENTICAL grid (same extent + 5 cm resolution, EPSG:32630) so the learned
# building masks are directly comparable between epochs, and rasterise the
# drainage network over the same patch for context/encroachment overlay.
#
# Emits RGB PNGs (model input) + a drainage PNG into tiles/_run/change/.
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
export GDAL_PAM_ENABLED=NO
OSM_GPKG="${AOI_DIR}/osm/oldfadama_osm.gpkg"
W="${AOI_DIR}/tiles/_run/change"; mkdir -p "${W}"

# Common patch inside the 2020 ∩ 2024 overlap (512 m square over settlement + drains)
TE=(807450 613838 807962 614350)            # xmin ymin xmax ymax (EPSG:32630)

for Y in 2020 2024; do
  echo "Warping ${Y} ortho to the common grid ..."
  "${BIN}/gdalwarp" -q -overwrite -t_srs EPSG:32630 -te "${TE[@]}" -tr 0.05 0.05 -r bilinear \
    "${AOI_DIR}/imagery/oldfadama_${Y}_5cm.tif" "${W}/patch_${Y}.tif"
  "${BIN}/gdal_translate" -q -of PNG -b 1 -b 2 -b 3 -mask none "${W}/patch_${Y}.tif" "${W}/patch_${Y}.png"
done

echo "Rasterising drainage network over the patch ..."
"${BIN}/ogr2ogr" -q -f GPKG /tmp/_water.gpkg -t_srs EPSG:32630 -where "waterway IS NOT NULL" -nln wl "${OSM_GPKG}" lines
"${BIN}/ogr2ogr" -q -f GPKG -update /tmp/_water.gpkg -t_srs EPSG:32630 -where "natural='water'" -nln wp "${OSM_GPKG}" multipolygons
"${BIN}/gdal_translate" -q -b 1 -ot Byte -scale 0 255 0 0 "${W}/patch_2020.tif" "${W}/drains.tif"
"${BIN}/gdal_rasterize" -q -burn 255 -l wp /tmp/_water.gpkg "${W}/drains.tif"
"${BIN}/gdal_rasterize" -q -burn 255 -l wl /tmp/_water.gpkg "${W}/drains.tif"
"${BIN}/gdal_translate" -q -of PNG "${W}/drains.tif" "${W}/drains.png"

echo "Prep done -> ${W}/{patch_2020.png, patch_2024.png, drains.png}"
