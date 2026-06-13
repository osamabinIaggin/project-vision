#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-1 product: acquisition of the multi-temporal centimetre-scale
# orthomosaic pair over the Old Fadama AOI from OpenAerialMap (CC-BY 4.0).
# The 2020 and 2024 epochs constitute the change-detection baseline. The
# 2.9 GB uncompressed 2024 variant is deliberately eschewed in favour of the
# perceptually equivalent high-compression rendition.
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
IMG_DIR="${AOI_DIR}/imagery"
mkdir -p "${IMG_DIR}"

declare -A SCENES=(
  [oldfadama_2020_5cm.tif]="https://oin-hotosm-temp.s3.amazonaws.com/5ed24f7157676e00057f5843/0/5ed24f7157676e00057f5844.tif"
  [oldfadama_2024_5cm.tif]="https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/66e402dfcd0baa0001b62128/0/66e402dfcd0baa0001b62129.tif"
)
for out in "${!SCENES[@]}"; do
  echo "Retrieving ${out} ..."
  curl -fsSL -o "${IMG_DIR}/${out}" "${SCENES[$out]}"
  "${BIN}/gdalinfo" "${IMG_DIR}/${out}" | grep -E "Size is|Pixel Size|EPSG\",326" | head -3
done
echo "Orthomosaic acquisition complete."
