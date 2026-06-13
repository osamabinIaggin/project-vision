# Project VISION — Vulnerability Inference for Submersion-prone Informal Settlements via Orthoimagery and Networks

A reproducible geospatial pipeline for the high-resolution identification of
anthropogenic pluvial-flood drivers in the Odaw–Korle catchment of Accra, Ghana.

---

## Abstract

Recurrent and frequently catastrophic inundation in metropolitan Accra is, on the
preponderance of the evidence, an anthropogenic rather than a climatological
phenomenon: its proximate determinants are the obstruction of drainage
infrastructure by solid-waste accumulation and the unregulated encroachment of
built structures onto watercourses and antecedent wetlands. Such determinants are
spatially fine-grained and therefore largely irresolvable by conventional
moderate-resolution satellite remote sensing. This project advances a multi-stage,
fully reproducible computational framework that exploits centimetre-scale
orthorectified aerial imagery to localise, quantify, and temporally track these
human-induced hazard drivers, with the lagoon-mouth informal settlement of
Old Fadama / Agbogbloshie — the hydraulic terminus of the entire ~400 km²
catchment and the epicentre of the June 2015 flood disaster — adopted as the
inaugural area of interest.

## 1. Motivation and Problem Formulation

The hydrological dysfunction of Accra is not principally a deficit of
precipitation forecasting but a deficit of *spatial intelligence*: municipal
authorities presently enumerate obstructed drains and encroaching structures by
post-hoc pedestrian survey conducted in the aftermath of inundation events. The
central hypothesis of this work is that such enumeration can be performed
*a priori*, automatically, and at scale, by interrogating sub-decimetre aerial
imagery with a combination of photogrammetric reconstruction, semantic
segmentation, and topology-aware geospatial overlay analysis.

## 2. Study Area

Old Fadama occupies the confluence of the terminal reach of the Odaw River and the
Korle Lagoon. As the downstream sink of the metropolitan drainage network it
concentrates, within a single contiguous tract of approximately 1.8 × 2.6 km,
every causal mechanism under investigation: progressive colonisation of former
wetland, dense riparian and on-drain construction, and the chronic occlusion of the
sea-outfall culverts by silt and municipal refuse.

## 3. Methodological Framework

The system is deliberately decomposed into loosely-coupled stages rather than
conceived as a monolithic end-to-end estimator:

| Stage | Operation | Paradigm |
|-------|-----------|----------|
| 1 | Photogrammetric assembly of overlapping frames into a georeferenced orthomosaic and surface model | Classical multi-view geometry |
| 2 | Pixel-wise semantic segmentation of the built environment, drainage network, and encroachment | Supervised deep learning (encoder–decoder CNN) |
| 3 | Terrain-derived flood-susceptibility inference from morphometric covariates (elevation, slope, flow accumulation, Topographic Wetness Index) | Tabular ensemble learning |
| 4 | Topology-aware overlay reconciling extracted structures against the drainage ontology to yield prioritised hazard loci | Deterministic geospatial computation |

The present repository operationalises Stages 1, 3 (conceptual), and 4, and
furnishes the labelled corpus required to instantiate Stage 2.

## 4. Data Provenance

| Asset | Source | Licence | Specification |
|-------|--------|---------|---------------|
| Orthomosaics (2020, 2024) | OpenAerialMap | CC-BY 4.0 | ~5 cm GSD, RGB, EPSG:32630 |
| Vector labels (buildings, drainage) | OpenStreetMap (via Overpass) | ODbL | 25,286 building footprints; Odaw, drains, canal |
| Digital elevation model | Copernicus GLO-30 | Copernicus open | 30 m, 0–739 m relief over the metropolitan extent |

Raster payloads are excluded from version control (see `.gitignore`) and are
regenerated deterministically by the acquisition scripts in `scripts/`.

## 5. Preliminary Results

A purely deterministic Stage-4 overlay, executed without recourse to any learned
model, adjudicates two complementary modalities of encroachment within the 2020
epoch: *on-drain* construction (a footprint intersecting a buffer about a mapped
drain/canal centreline) and *riparian* construction (a footprint intersecting a
buffer about a water-body polygon — Korle Lagoon and the Odaw channel). Because the
tally is a monotone function of the chosen tolerance, a sensitivity sweep is
reported in lieu of a single figure:

| Modality | Tolerance | Encroaching structures |
|----------|-----------|------------------------|
| On-drain | 5 m | 114 |
| Riparian | 5 m | 51 |
| Riparian | 10 m | 159 |
| Riparian | 20 m | 470 |
| **Union (drain 5 m ∪ water 10 m)** | — | **273** |

No structure lies *within* a mapped water polygon, an internal consistency check on
the geometry. The union of **273 structures** encroaching upon the drainage system
constitutes a proof of concept that the defining hazard signature is computationally
legible from open data alone (see `docs/figures/encroachment_oldfadama_2020.png`).

## 6. Repository Structure

```
.
├── README.md                  # this document
├── CITATION.cff               # scholarly citation metadata
├── LICENSE                    # MIT (source code)
├── scripts/                   # reproducible acquisition & analysis pipeline
│   ├── _env.sh                # GDAL/PROJ environment resolution
│   ├── 01_acquire_dem.sh
│   ├── 02_acquire_oldfadama_imagery.sh
│   ├── 03_acquire_osm_labels.sh
│   └── 04_encroachment_overlay.sh
└── accra_flood/               # working tree (data dirs are gitignored)
    ├── data/                  # DEM (regenerated)
    └── oldfadama/             # pilot AOI imagery, labels, metadata
```

## 7. Reproducibility

The pipeline presumes a QGIS distribution providing the GDAL/PROJ toolchain; the
environment-resolution shim `scripts/_env.sh` locates the bundled binaries.
Execute the numbered scripts in sequence to reconstitute the full data corpus and
the encroachment analysis from first principles.

## 8. Limitations and Prospective Work

As the catchment's terminal sink, Old Fadama's inundation regime is partially
governed by upstream forcing and outfall occlusion rather than purely local
hydraulics; a model fit exclusively here risks learning a settlement-specific
rather than a generalisable representation. Subsequent campaigns will extend the
corpus to upstream communities (Alogboshie, Alajo, Akweteyman) and instantiate the
Stage-2 segmentation network to obviate dependence on pre-existing vector labels,
thereby enabling encroachment change-detection across the 2020 and 2024 epochs.

## 9. Licence and Citation

Source code is released under the MIT Licence (`LICENSE`). Derived data inherit the
upstream licences enumerated in §4. Please cite this work via `CITATION.cff`.
