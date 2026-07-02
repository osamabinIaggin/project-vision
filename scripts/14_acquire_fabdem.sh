#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-3 substrate (v2): acquisition of FABDEM V1-2 tiles over the Odaw
# basin. FABDEM is Copernicus GLO-30 with forest and BUILDING artefacts
# removed (Hawker et al. 2022, 10.1088/1748-9326/ac4d4f) — essential here
# because GLO-30 is a surface model: over the dense Accra fabric its "terrain"
# embeds roof heights, which corrupts flow routing and any
# height-above-drainage statistic. Distributed by the University of Bristol
# under CC-BY-NC-SA 4.0 (non-commercial research use).
#
# The 10x10-degree region zip (~1.2 GB) is fetched to a scratch location and
# only the two 1-degree tiles subtending the basin are retained.
# ----------------------------------------------------------------------------
source "$(dirname "$0")/_env.sh"
mkdir -p "${DATA_DIR}"

ZIP_URL="https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/N00W010-N10E000_FABDEM_V1-2.zip"
TILES=(N05W001_FABDEM_V1-2.tif N06W001_FABDEM_V1-2.tif)

missing=0
for t in "${TILES[@]}"; do [[ -f "${DATA_DIR}/${t}" ]] || missing=1; done
if [[ "${missing}" -eq 0 ]]; then
  echo "FABDEM tiles already present in ${DATA_DIR}; nothing to do."
  exit 0
fi

TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
echo "Retrieving FABDEM region zip (~1.2 GB) ..."
curl -fsSL --retry 3 -o "${TMP}/fabdem_region.zip" "${ZIP_URL}"
unzip -o -q "${TMP}/fabdem_region.zip" "${TILES[@]}" -d "${DATA_DIR}"
echo "FABDEM acquisition complete: ${TILES[*]} -> ${DATA_DIR}"
