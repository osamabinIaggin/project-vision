#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Stage-2 corpus extension: catalogue of the Open Cities Africa drone
# orthomosaics over the middle-Odaw communities, from OpenAerialMap (CC-BY 4.0).
#
# Old Fadama sits at the catchment outlet and is morphologically atypical — a
# lagoon-mouth informal settlement. A segmentation model fit there alone risks
# learning that specific fabric rather than a transferable representation of
# Accra's dense built environment, a limitation carried since Stage 2. These
# four scenes cover the middle-Odaw communities upstream (Alogboshie,
# Akweteyman, Alajo, Nima), which are also precisely the communities whose
# drains the Open Cities / GARID campaigns surveyed (scripts/21) — so the same
# imagery underwrites both the corpus extension and, later, observation of the
# drain siltation that Stage 5 can currently only sweep as a scenario.
#
# NOTHING IS BULK-DOWNLOADED. The four scenes total ~2.5 GB, of which the
# analysis needs four 1024 m windows. Inspection shows they are cloud-optimised
# GeoTIFFs — 512x512 internal blocks with a full overview pyramid — so GDAL can
# fetch only the blocks a window intersects over HTTP range requests. This
# script therefore records the catalogue and verifies remote readability;
# scripts/26 warps the windows straight from these URLs via /vsicurl. Retaining
# 2.5 GB of raster to extract ~1% of it would also contradict the repository's
# standing policy of versioning the methodology rather than the payload.
# ----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/_env.sh"
OUT_DIR="${REPO_ROOT}/accra_flood/middleodaw"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/scenes.json"

cat > "${MANIFEST}" <<'EOF'
{
  "source": "OpenAerialMap, Open Cities Africa (CC-BY 4.0)",
  "note": "read remotely via /vsicurl; cloud-optimised, 512x512 blocks + overviews",
  "scenes": {
    "alogboshie": {
      "title": "Open Cities Africa - Alogboshie, Accra",
      "acquired": "2018-08-05", "gsd_m": 0.0201, "size_mb": 1184,
      "bbox": [-0.2435, 5.6197, -0.2270, 5.6479],
      "url": "https://oin-hotosm-temp.s3.amazonaws.com/5b694a0f4b87366cc0f0fa70/0/5b694a0f4b8736ebfff0fa71.tif"
    },
    "akweteman": {
      "title": "Open Cities Africa - Akweteman, Accra",
      "acquired": "2018-10-06", "gsd_m": 0.0322, "size_mb": 366,
      "bbox": [-0.2500, 5.6074, -0.2327, 5.6198],
      "url": "https://oin-hotosm-temp.s3.amazonaws.com/5bb9323e9ed15b0006d24f33/0/5bb9323e9ed15b0006d24f34.tif"
    },
    "alajo": {
      "title": "Open Cities Africa - Alajo, Accra",
      "acquired": "2018-11-12", "gsd_m": 0.0360, "size_mb": 624,
      "bbox": [-0.2349, 5.5826, -0.1999, 5.6111],
      "url": "https://oin-hotosm-temp.s3.amazonaws.com/5be9bb18080ac000051474fd/0/5be9bb18080ac000051474fe.tif"
    },
    "nima": {
      "title": "Open Cities Africa - Nima, Accra",
      "acquired": "2019-07-07", "gsd_m": 0.0520, "size_mb": 295,
      "bbox": [-0.2100, 5.5721, -0.1863, 5.5950],
      "url": "https://oin-hotosm-temp.s3.amazonaws.com/5d2d0ab4f416f40006cffcc1/0/5d2d0ab4f416f40006cffcc2.tif"
    }
  }
}
EOF

echo "Verifying remote readability of each scene ..."
"${REPO_ROOT}/.venv/bin/python" - "${MANIFEST}" "${BIN}" <<'PYEOF'
import json, subprocess, sys
man, BIN = sys.argv[1], sys.argv[2]
d = json.load(open(man))
ok = 0
for name, s in d["scenes"].items():
    r = subprocess.run([f"{BIN}/gdalinfo", "/vsicurl/" + s["url"]],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {name:12s} UNREADABLE"); continue
    size = next((l for l in r.stdout.splitlines() if l.startswith("Size is")), "?")
    ovr = "Overviews:" in r.stdout
    blk = "Block=512x512" in r.stdout
    print(f"  {name:12s} {size:22s} overviews={ovr} tiled512={blk}")
    ok += 1
print(f"{ok}/{len(d['scenes'])} scenes readable remotely")
PYEOF
echo "Catalogue written: ${MANIFEST}"
