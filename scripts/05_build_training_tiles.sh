#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-2 data preparation: tessellation of the orthomosaic and its
# OSM-derived building mask into co-registered (image, mask) tile pairs, the
# canonical input format for supervised semantic-segmentation training.
#
# A binary mask is rasterised on a grid identical to a cropped window of the
# orthomosaic, guaranteeing pixel-for-pixel correspondence. The window is then
# partitioned into TILE x TILE chips; only chips whose mask carries building
# signal are retained (a tile is overwhelmingly background pixels regardless,
# so within-tile negatives remain abundant).
#
# Parameters (env-overridable): TILE, ULX/ULY/LRX/LRY (EPSG:32630 window),
# MIN_BLDG_MEAN (mask-mean floor, 0..255).
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
export GDAL_PAM_ENABLED=NO

ORTHO="${AOI_DIR}/imagery/oldfadama_2020_5cm.tif"
OSM_GPKG="${AOI_DIR}/osm/oldfadama_osm.gpkg"
OUT="${AOI_DIR}/tiles"; IMGDIR="${OUT}/images"; MSKDIR="${OUT}/masks"
mkdir -p "${IMGDIR}" "${MSKDIR}"
rm -f "${IMGDIR}"/*.png "${MSKDIR}"/*.png "${OUT}/manifest.csv"

TILE="${TILE:-512}"
ULX="${ULX:-807650}"; ULY="${ULY:-614100}"; LRX="${LRX:-808162}"; LRY="${LRY:-613588}"
MIN_BLDG_MEAN="${MIN_BLDG_MEAN:-0.5}"

echo "Cropping AOI window and rasterising the building mask onto an identical grid ..."
"${BIN}/gdal_translate" -q -projwin "${ULX}" "${ULY}" "${LRX}" "${LRY}" "${ORTHO}" "${OUT}/_crop_img.tif"
read -r W H <<<"$("${BIN}/gdalinfo" "${OUT}/_crop_img.tif" | awk '/Size is/{gsub(",","");print $3,$4}')"
"${BIN}/ogr2ogr" -q -f GPKG /tmp/_bld.gpkg -t_srs EPSG:32630 -where "building IS NOT NULL" -nln b "${OSM_GPKG}" multipolygons
# Build the mask on a ZEROED copy of the crop so it inherits the crop's exact
# geotransform. Creating a fresh raster via -te/-ts mis-burns and floods the whole
# frame with 255; rasterising onto a translate-derived grid is correct.
"${BIN}/gdal_translate" -q -b 1 -ot Byte -scale 0 255 0 0 "${OUT}/_crop_img.tif" "${OUT}/_crop_mask.tif"
"${BIN}/gdal_rasterize" -q -burn 255 -l b /tmp/_bld.gpkg "${OUT}/_crop_mask.tif"

echo "Tessellating into ${TILE}x${TILE} pairs (grid ${W}x${H}) ..."
echo "tile,col,row,building_fraction" > "${OUT}/manifest.csv"
kept=0; total=0
for ((y=0; y+TILE<=H; y+=TILE)); do
  for ((x=0; x+TILE<=W; x+=TILE)); do
    total=$((total+1)); id=$(printf "t_%05d_%05d" "${x}" "${y}"); m="${MSKDIR}/${id}.png"
    "${BIN}/gdal_translate" -q -of PNG -srcwin "${x}" "${y}" "${TILE}" "${TILE}" "${OUT}/_crop_mask.tif" "${m}"
    mean=$("${BIN}/gdalinfo" -stats "${m}" 2>/dev/null | grep -oE "Mean=[0-9.]+" | head -1 | cut -d= -f2) || true
    if awk "BEGIN{exit !(${mean:-0} > ${MIN_BLDG_MEAN})}"; then
      "${BIN}/gdal_translate" -q -of PNG -b 1 -b 2 -b 3 -mask none -srcwin "${x}" "${y}" "${TILE}" "${TILE}" "${OUT}/_crop_img.tif" "${IMGDIR}/${id}.png"
      printf "%s,%s,%s,%.4f\n" "${id}" "${x}" "${y}" "$(awk "BEGIN{print ${mean}/255}")" >> "${OUT}/manifest.csv"
      kept=$((kept+1))
    else
      rm -f "${m}"
    fi
  done
done
rm -f "${OUT}/_crop_img.tif" "${OUT}/_crop_mask.tif" \
      "${IMGDIR}"/*.aux.xml "${MSKDIR}"/*.aux.xml \
      "${IMGDIR}"/*.wld "${MSKDIR}"/*.wld \
      "${IMGDIR}"/*.msk "${MSKDIR}"/*.msk 2>/dev/null || true
echo "Done: retained ${kept} of ${total} candidate tiles -> ${OUT}/{images,masks}"
