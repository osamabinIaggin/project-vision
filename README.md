# VISION — Geospatial Deep Learning for Flood-Driver Detection

> Computer vision on centimetre-scale aerial imagery to localise the *human* causes
> of urban flooding — drainage encroachment and obstructed waterways — that
> conventional satellite remote sensing cannot resolve.

![status](https://img.shields.io/badge/status-active%20research-1f6feb)
![method](https://img.shields.io/badge/deep%20learning-U--Net%20segmentation-d29922)
![vision](https://img.shields.io/badge/computer%20vision-change%20detection-2da44e)
![data](https://img.shields.io/badge/remote%20sensing-5%20cm%20GSD-8957e5)
![license](https://img.shields.io/badge/license-MIT-8b949e)

**VISION** (*Vulnerability Inference for Submersion-prone Informal settlements via
Orthoimagery and Networks*) is an applied geospatial-AI system that couples
**deep-learning semantic segmentation** — a **U-Net convolutional neural network** —
with **multi-temporal change detection** and **topology-aware spatial reasoning** to
detect and quantify the anthropogenic drivers of pluvial flooding in Accra, Ghana,
at a resolution roughly two orders of magnitude finer than the satellite imagery on
which prior work has relied.

### Technical approach

- **Semantic segmentation (U-Net / CNN)** — pixel-wise extraction of buildings,
  drainage, and encroachment from 5 cm RGB orthomosaics
- **Multi-temporal change detection** — 2020 vs 2024 epochs to quantify the
  *growth* of encroachment onto watercourses
- **Topographic flood-susceptibility modelling** — gradient-boosted / random-forest
  ensembles over terrain morphometrics (slope, flow accumulation, TWI)
- **Topology-aware geospatial overlay** — reconciling segmented structures against
  the hydrographic network to rank hazard loci
- **Open-channel hydraulics** — Manning conveyance capacity of field-surveyed
  drain cross-sections, and its collapse under progressive siltation
- **Reproducible by construction** — open data, scripted acquisition, and a
  version-controlled methodology rather than committed binary payloads

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
| 5 | Hydraulic conveyance analysis of the field-surveyed drainage network, and its degradation under siltation | Open-channel hydraulics (Manning) |

All five stages are operationalised in this repository. Stage 1 is satisfied by
the published OpenAerialMap orthomosaics rather than by re-running
photogrammetry; Stage 3 is realised as physically-based terrain hydrology
(HAND) in place of the originally-envisaged tabular ensemble, the AOI having
proved to admit no within-basin hazard gradient for such a model to learn.

## 4. Data Provenance

| Asset | Source | Licence | Specification |
|-------|--------|---------|---------------|
| Orthomosaics (2020, 2024) | OpenAerialMap | CC-BY 4.0 | ~5 cm GSD, RGB, EPSG:32630 |
| Vector labels (buildings, drainage) | OpenStreetMap (via Overpass) | ODbL | 25,286 building footprints; Odaw, drains, canal |
| Digital elevation model | Copernicus GLO-30 | Copernicus open | 30 m, 0–739 m relief over the metropolitan extent |
| Bare-earth terrain | FABDEM V1-2 | CC-BY-NC-SA | 30 m, building/forest artefacts removed |
| Drainage cross-sections | Open Cities Accra / GARID field survey, via OSM | ODbL | 2,107 drain/ditch ways; 760 with surveyed width and depth |
| Built-up time series | Google Open Buildings 2.5D Temporal | CC-BY 4.0 | presence, count, height; annual 2016–2023 |
| Rainfall | ERA5 hourly reanalysis (Open-Meteo) | CC-BY 4.0 | ~31 km, 1980–2024 |

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

### Learned building segmentation (Stage 2)

A U-Net was trained to segment built-up area from the 5 cm orthoimagery, supervised
by the OSM footprints (`scripts/05`–`07`). On an identical 1,081-tile evaluation
split, a **pretrained ResNet-34 encoder (transfer learning) outperformed a
from-scratch U-Net — validation IoU 0.579 vs 0.530** — consistent with the
literature on small, label-noisy datasets. Both models converge to *coarse,
region-level* masks rather than crisp per-footprint boundaries; controlled diagnosis
attributes this ceiling to **label quality** (OSM footprints are offset ~1–2 m and
merge adjacent structures in the dense settlement) rather than model capacity — train
loss falls freely while validation plateaus.

The label-quality hypothesis was subsequently stress-tested from three directions
(`scripts/09`–`12`). (i) *Recipe ablations*: stronger training recipes —
native-resolution multi-scale crops, photometric jitter, differential
encoder/decoder learning rates, AdamW under a warmup-cosine schedule over a
60-epoch budget — score **0.561/0.566, below the 0.579 baseline**; the added
optimisation capacity only memorises label noise (`scripts/09`). (ii) *Independent
labels*: Google Open Buildings v3 footprints for the identical window
(`scripts/11`) agree with the OSM mask at only **IoU 0.611** on the same grid, so
the model already operates at the inter-source noise floor; Open Buildings is not
a drop-in replacement, as it draws cleaner per-building shapes but under-detects
in the densest blocks (`docs/figures/label_sources_oldfadama.png`).
(iii) *Registration*: a global-shift search of the OSM mask against model
predictions peaks at ~0.2 m — the label error is per-building and random, not a
correctable systematic offset (`scripts/12`). The one genuine post-hoc gain is
inference-time: **4-way flip test-time augmentation with a calibrated 0.45
threshold lifts the transfer-learning model to IoU 0.593** at zero training cost
(`scripts/10`). Surpassing the ≈0.6 ceiling requires cleaner supervision (e.g. a
hand-verified label subset), not further architecture or recipe work.

That cleaner supervision was then constructed by **cross-source consensus
verification** (`scripts/13`): pixels on which OSM and Open Buildings agree are
trusted (77% of the corpus; spot-verified visually against the imagery), while
disagreement pixels are marked *ignore* and excluded from loss and metric alike.
Two results follow. First, the benchmark itself was hiding model quality: scored
on verified pixels only, the existing transfer-learning model achieves **pooled
IoU 0.77** — most of its apparent error was disagreement with label noise, not
with reality. Second, retraining with disputed pixels masked out of the loss
yields a further genuine gain: **pooled verified IoU 0.80** (per-tile mean 0.70),
and the new model now *correctly* rejects OSM's phantom buildings on open ground
(its legacy score against raw OSM masks drops to 0.55 for exactly that reason —
see `docs/figures/unet_predictions_consensus.png`). The verified metric excludes
the hardest boundary pixels by construction, so these figures characterise
region-level built-up mapping, consistent with the model's downstream use in
Stage-2/4; per-footprint delineation in the dense core remains open pending
instance-level labels.

Applied to the **unseen 2024 epoch** (`scripts/08`), the model maps built-up extent
*without any 2024 labels* (17.6 ha over the test patch), confirming label-free
generalisation. However, naïve inter-epoch mask differencing is dominated by
prediction flicker between independent flights and does **not** reliably localise
individual new structures; the deterministic overlay remains the higher-fidelity
encroachment estimator. Figures: `docs/figures/unet_predictions_resnet.png`,
`docs/figures/change_detection_oldfadama.png`.

### Hydrological hazard and integrated exposure (Stages 3-4)

Stage-3 terrain analysis (`scripts/14`-`15`) is computed from **FABDEM V1-2**
(Copernicus GLO-30 with building/forest artefacts removed — necessary because
over the dense urban fabric the raw GLO-30 surface embeds roof heights and
corrupts flow routing). Least-cost depression breaching, D8 flow accumulation,
stream extraction, and **height above nearest drainage (HAND)** are derived over
the full Odaw catchment, from the Akwapim foothills to the Korle Lagoon outfall.

**Validation** proceeds on three independent lines, all from primary data (no
agency ever published a flood extent for an Odaw event: UNOSAT's only Ghana
product covers the Oct-2023 lower-Volta event, verified to lie outside the
basin; the 2015 event is absent from the Global Flood Database; Copernicus EMS
was never activated for Ghana).

1. **Locality-level statistical test** (`scripts/19`). Every Odaw-basin
   locality named in multi-year flood reporting (Old Fadama, Circle, Odawna,
   Adabraka, Kaneshie, Nima, Alajo, Abeka Lapaz) versus eight elevated control
   districts absent from flood reporting, geocoded independently via OSM
   Nominatim: median HAND **1.9 m vs 23.0 m**, AUC 0.86, exact one-sided
   Mann-Whitney **p = 0.0074**. The two flooded localities sampled high are
   centroid artefacts (Nima and Lapaz geocode to ridge tops above their valley
   frontage) — noise that biases *against* the hypothesis and is retained.

2. **Areal open-water SAR mapping** (`scripts/17`). Sentinel-1 RTC scenes
   (Planetary Computer, anonymous access) were processed for the 18 June 2018
   (D+2) and 9 June 2020 (same-evening pass) floods against dry-season
   reference stacks. The detector is demonstrably sound — 121 ha of water
   change on 2020-06-09 at 99.5% HAND < 2 m (AUC 0.95), dominated by the
   Panbros salt-pan cycle, a positive control — but inside the Odaw floodplain
   it detects almost nothing: the surface is near-continuously roofed and
   flooded streets raise VV backscatter (double-bounce) instead of darkening
   it. Open-water SAR is physically blind to this basin's street flooding;
   figures `docs/figures/sar_flood_20180620_vs_hand.png`, `..20200609..`.

3. **SAR double-bounce statistics** (`scripts/18`). Testing the complementary
   urban signature across all 29 June acquisitions 2015-2024: the one
   same-evening flood acquisition ranks 5/29 on floodplain backscatter anomaly
   (p = 0.17), and the anomaly does not correlate with pre-pass ERA5 rainfall
   (Spearman ρ = -0.02). Reported as the null results they are: at 20 m VV
   gamma0, the urban double-bounce channel carries no usable flood signal
   here either. The terrain-based validation (line 1) is therefore the
   operative evidence, and it is significant.

The integrated exposure product (`scripts/16`) intersects the HAND hazard
classes with the consensus-model built-up extent: **the entire 37.7 ha of
built-up Old Fadama lies in the severe class (HAND < 2 m)** — within-AOI hazard
gradation is physically absent (mean HAND 0.03 m), not merely unresolved at the
30 m DEM. The settlement-scale statement is therefore uniform severe fluvial
exposure at the basin outlet, aggravated by the drainage encroachment quantified
in Stage 4. Figure: `docs/figures/flood_risk_oldfadama.png` (corridor HAND map,
2015 sites, and AOI). Pluvial ponding and within-settlement micro-topography
would require a drone-derived DSM, which the OpenAerialMap missions did not
publish.

### Decadal built-up and encroachment trend (Stage 4, temporal)

The two-epoch drone change detection could not separate new construction from
inter-flight prediction flicker. **Google Open Buildings 2.5D Temporal**
(CC-BY 4.0, anonymous access; presence, count, and *height* at 4 m effective
resolution, annually 2016-2023) replaces the two-epoch problem with eight
consistent epochs (`scripts/20`). Over the AOI: built-up area grew **22.4 →
33.3 ha (2016-2021, +49%)**, contracted to 27.4 ha in 2022, and rebounded to
30.0 ha by 2023; built-up within the 15 m riparian buffer of the drains grew
**tenfold** (0.03 → 0.31 ha). The 2021→2022 contraction is itself a
validation of the source: the epochs are dated 30 June, and the
police-backed Agbogbloshie clearance began **1 July 2021** — the series drops
in exactly the epoch it should, and the 2023 rebound matches documented
reoccupation. Mean structure height holds at 5-6 m (one-to-two storey),
confirming minimal vertical refuge in the settlement. Cross-check: the 2020
epoch reports 30.1 ha built-up vs 37.7 ha from the 5 cm consensus model —
consistent given the 4 m footprint floor. Figure:
`docs/figures/builtup_timeseries_oldfadama.png`.

On the remaining terrain gap: no open online product resolves ground
micro-topography for Accra (FABDEM/GLO-30 at 30 m is the open ceiling;
TanDEM-X 12 m is proposal-gated; no national LiDAR exists). ICESat-2
laser altimetry offers sparse cm-accurate ground tracks behind a free
Earthdata registration and is the only open avenue toward sub-30 m vertical
constraint short of a new drone flight.

### Cross-site transfer: the Stage-2 representation is settlement-specific

Every Stage-2 number above was obtained at Old Fadama, and §8 has recorded from the
outset the risk that a model fit there learns a lagoon-mouth informal settlement rather
than Accra's dense built fabric in general. That risk is now measured rather than
asserted. The Open Cities Africa drone missions cover four middle-Odaw communities
upstream — Alogboshie, Akweteyman, Alajo and Nima — which are also the communities whose
drains Stage 5 analyses, so the same imagery serves both purposes (`scripts/24`–`26`).
The scenes are cloud-optimised, so the windows are read remotely rather than downloaded;
each is resampled to the 5 cm training GSD, since a transfer test conducted at the wrong
scale would measure the scale mismatch instead. Supervision is rebuilt identically:
the same two sources, the same consensus rule, the same ignore band.

The unmodified consensus checkpoint was then applied zero-shot to 1,575 tiles
(`scripts/27`):

| Site | tiles | pooled verified IoU | per-tile | label floor | predicted built-up | actual (OSM) |
|---|---|---|---|---|---|---|
| Old Fadama (control, own val split) | 217 | **0.798** | 0.695 | 0.611 | 53.2% | 49.8% |
| Akweteyman | 400 | 0.641 | 0.603 | 0.519 | 38.2% | 43.6% |
| Alajo | 398 | 0.603 | 0.558 | 0.488 | 38.1% | 41.6% |
| Nima | 400 | 0.507 | 0.494 | 0.713 | 43.1% | 65.5% |
| Alogboshie | 377 | 0.297 | 0.359 | 0.543 | 11.0% | 36.8% |
| **All four, pooled** | **1,575** | **0.523** | 0.505 | 0.578 | — | — |

**The representation degrades substantially but does not collapse.** Pooled verified IoU
falls from 0.798 to 0.523 — a loss of 0.28 — while the label-noise floor moves only from
0.611 to 0.578. The degradation is therefore attributable to the model rather than to
worse supervision upstream, but it is partial: three of the four communities retain
0.51–0.64, which is materially above chance and, for Akweteyman and Alajo, close to the
label floor itself. Predicted built-up fraction tracks the actual fraction at those two
sites (38.2% against 43.6%, 38.1% against 41.6%), so the model remains calibrated there
and simply delineates less precisely.

Alogboshie is a genuine outlier and is reported as such rather than averaged away: 0.297
pooled, with a predicted built-up fraction of 11.0% against 36.8% present — it
under-segments by more than threefold, where the other three do not.

A preprocessing explanation was the obvious first suspect and was tested rather than
assumed. Alogboshie is the finest source in the set at 2.01 cm and must be reduced 2.49×
to reach the 5 cm training GSD, whereas Akweteyman, Alajo and Nima sit at 3.2, 3.6 and
5.2 cm and are reduced by 1.55×, 1.39× and 1.04×. Bilinear resampling reads only a 2×2
neighbourhood, so beyond roughly a factor of two it under-filters and folds discarded
detail back as aliasing — precisely the high-frequency texture a CNN keys on — and those
reduction factors happen to rank the four sites in the same order as their scores. The
scene was therefore re-fetched with area-average resampling, which integrates the full
source footprint and is the correct kernel for a true downsample (`scripts/26` now
selects the kernel from the reduction factor). **The score moved from 0.297 to 0.298.**

The hypothesis is refuted, and that negative is the useful result: Alogboshie's difficulty
is a real domain difference — settlement morphology, roof material and condition,
radiometry, season, solar geometry — and not an artefact of how the imagery was prepared.
This governs what follows. Since no data defect underlies the outlier, excluding it from
the reported evaluation would be selection on the outcome, and it is retained. For
training the implication inverts: the hardest domain is the informative one, and a corpus
assembled to fix cross-site generalisation has more need of Alogboshie than of the sites
that already transfer.

That the comparison means anything at all rests on a control, and the control is the
methodological point. A low score at a new site is uninterpretable in isolation — as
consistent with a preprocessing defect as with a domain shift. `scripts/27 --control`
therefore scores Old Fadama's own validation split through the identical code path, and
recovers 0.798 / 0.695 / 0.611 against the 0.80 / 0.70 / 0.611 published from
`scripts/13`. The pipeline is sound; the gap is real. Figure:
`docs/figures/transfer_middleodaw.png` (imagery, verified label, prediction, three tiles
per community).

The practical consequence is that the existing checkpoint should not be deployed outside
Old Fadama without fine-tuning or joint training, and this bears directly on the siltation
work Stage 5 designates as the next milestone, since that work is sited in these very
communities. What was a caveat in §8 is now a quantified prerequisite — and the corpus
needed to remove it, 1,575 consensus-verified tiles across four settlements, is now built.

### The transfer gap is a coverage limitation, not a ceiling (Stage 2, joint)

Measuring the gap leaves open whether it can be closed. Stage 2 had already met one hard
ceiling at Old Fadama — label granularity, immovable by architecture or recipe — so the
possibility that cross-site degradation was a second such ceiling had to be excluded
rather than assumed. `scripts/28` settles it by leave-one-site-out: each of the five
sites is withheld *entirely*, the consensus checkpoint is fine-tuned for eight epochs on
the other four, and the withheld site is scored under the identical verified-pixel
metric. Holding out whole sites rather than random tiles is what makes this a test of
transfer rather than of interpolation — tiles drawn from a single 512 m window are far
too spatially correlated for a random split to say anything about a new settlement.

| Held-out site | tiles | before | after fine-tuning | change | held-in sites |
|---|---|---|---|---|---|
| Alogboshie | 377 | 0.298 | **0.616** | +0.318 | 0.885 |
| Nima | 400 | 0.507 | **0.863** | +0.355 | 0.866 |
| Alajo | 398 | 0.603 | **0.833** | +0.231 | 0.888 |
| Akweteyman | 400 | 0.641 | **0.883** | +0.242 | 0.881 |
| Old Fadama | 1,081 | 0.904 † | 0.681 | — † | 0.912 |

Every genuinely unseen site improves, by between 0.23 and 0.36 pooled verified IoU, for a
mean of **+0.287** — and it does so without the model ever seeing a tile from the site it
is scored on. Alogboshie, the outlier that resisted the preprocessing explanation, more
than doubles from 0.298 to 0.616. It remains the hardest of the five, so the domain
difference established above is real, but it is plainly learnable rather than intractable.
Recalibration accompanies the accuracy: Alogboshie's predicted built-up fraction moves
from 11.0% against 26.2% actual to 30.4%, i.e. from threefold under-segmentation to
approximate agreement. Held-in performance sits at 0.866–0.912 throughout, so none of this
is bought by forgetting the sites already learned.

† The Old Fadama row is not comparable and is marked accordingly. Its "before" figure of
0.904 is not a transfer score at all: the starting checkpoint was trained on 80% of those
very tiles, so scoring the whole site recovers a training-set number, and the apparent
−0.222 is an artefact of that contamination rather than a regression. The interpretable
quantity in that fold is the 0.681 — what four middle-Odaw communities alone generalise to
Old Fadama, having never seen it. That figure is reported for what it is.

The conclusion is that the degradation documented above reflects the *coverage* of the
training corpus rather than a limit of the representation. Roughly two thousand tiles from
other settlements and eight epochs recover most of the loss, which places cross-site
generalisation firmly in the category of problems this project can solve with data it can
obtain, and distinguishes it from the per-footprint delineation ceiling, which it cannot.

### Conveyance capacity of the surveyed drainage network (Stage 5)

Stages 2–4 establish where water concentrates and where structures encroach on
the drains. Neither asks whether the drains that remain can carry what their
catchments deliver. The Open Cities Accra / GARID field campaigns (2018–2020)
surveyed the middle-Odaw drains segment by segment and committed engineering
attributes to OpenStreetMap — width, depth, cross-section profile, material,
invert smoothness, culvert status — which is precisely the geometry Manning's
equation requires. `scripts/21` retrieves 2,107 drain and ditch ways over
central Accra, 760 of them carrying both width and depth.

**No design storm is used, because none can be defended.** `scripts/22` tests
whether an open rainfall product can supply one, and reports a clean negative:
fitted to 45 years of ERA5 hourly reanalysis, the Gumbel relation is internally
well-behaved and reproduces Accra's annual total (~900 mm), yet on every
documented flood day the reanalysis returns a depth *below every annual maximum
in the record* — 3 June 2015, which killed some 150 people, appears as a
5.1 mm/h, 10.8 mm/24h day. Six sample points spanning the metropolis return
byte-identical series, so this is grid resolution, not storm displacement: one
~31 km cell covers the city, and coastal Accra's convective rainfall is beneath
it. The model is therefore inverted. For each segment the rational method is
solved not for discharge but for the **critical intensity** at which the drain
reaches capacity, `i_crit = Q_cap · 3.6·10⁶ / (C · A)` — a property of the drain
and its catchment alone, requiring no rainfall input and no return period. The
ERA5 relation is retained only as an acknowledged *floor* (1-hour, T = 10 yr:
20.8 mm/h) against which those critical intensities are read.

Contributing area is accumulated on the drain network itself rather than routed
over the DEM, for a reason this project has already established: no open product
resolves Accra's street-level micro-topography, so D8 routing over a 30 m
surface cannot determine which side of a street drains to which gutter. Each
10 m cell is instead allocated to its nearest drain and the resulting local
catchments are accumulated downstream through the network graph, reconstructed
from OSM endpoints snapped at 3 m and oriented by a 300 m-smoothed elevation
field. Longitudinal grade is taken across a 300 m chord and clamped to the range
urban drains are actually laid to (0.1–2%); the DEM is used only to place a
segment within that range, never to assert a grade outside it. Because grade
enters as √S, the *ranking* is invariant to it and only absolute counts shift.

Two data-quality findings preceded any hydraulic result, and both materially
changed it. First, 108 surveyed segments (4.3 km) carry widths of 4–20 cm —
median 6 cm — against depths of 0.2–0.7 m, aspect ratios reaching 13:1. These
are not street drains but unit or entry errors in the field campaign, and left
in they dominated the ranking completely, supplying 91 of the 107 apparently
critical segments. They are flagged in the output and held out of the headline
statistics rather than silently discarded. Second, the surveyed network is
topologically fragmented — 484 connected components before endpoint snapping,
293 of them isolated single ways — so accumulated catchments remain local
(median 0.64 ha, maximum 50 ha) and no trunk conveyance is represented.

On the 652 segments (45.2 km) with credible geometry the result is
counter-intuitive and, for this project's thesis, the point:

| Invert condition | Segments failing | Length | Share of network |
|------------------|------------------|--------|------------------|
| Clean section | 16 | 1.0 km | 2.3% |
| 25% silted | 27 | 1.9 km | 4.1% |
| 50% silted | 46 | 3.1 km | 6.9% |
| 75% silted | 146 | 11.6 km | **25.8%** |

**The drains are not undersized.** At full section the median segment conveys
0.61 m³/s against a median catchment of 0.64 ha and tolerates 309 mm/h — an
intensity no Accra storm approaches — and only 2.3% of the network fails even
the deliberately conservative ERA5 floor. Capacity is lost to the *state* of the
invert, not its geometry: siltation to three-quarters depth multiplies the
failing length elevenfold, and the median segment fails once 85% silted, with
the most fragile fifth going at 36%. The hazard is therefore a maintenance
variable rather than a construction one, which is exactly the quantity that
centimetre-resolution overhead imagery can observe and a design study cannot.
It also locates Stage 5 squarely on the project's central hypothesis: Accra's
flooding is anthropogenic, and the anthropogenic term here is refuse in the
channel, not an engineering deficit. Figure:
`docs/figures/drain_capacity_accra.png`; per-segment results, ranked and with
the plausibility flag retained, in `accra_flood/output/drain_capacity.csv` and a
QGIS-ready `drain_capacity.gpkg`.

The principal caveats are stated rather than absorbed. The siltation sweep is a
scenario, not an observation — the survey's `smoothness` tag is the only
recorded proxy for invert condition and it does not separate the fragile
segments (median 398 mm/h against 305 for the remainder), so which drains are
*actually* silted remains unmeasured and is the natural target for Stage-2
imagery. Fragmentation means the trunk drains that would carry the accumulated
flow are absent from the surveyed set, so basin-scale backwater — a documented
contributor at the Odaw outfall — is outside this model. And every intensity
here is anchored to a floor known to understate the truth, so the failure shares
are lower bounds.

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
│   ├── 04_encroachment_overlay.sh
│   ├── 05_build_training_tiles.sh   # Stage-2 (image, mask) tile corpus
│   ├── 06_train_unet.py             # from-scratch U-Net baseline
│   ├── 07_train_unet_resnet.py      # pretrained ResNet-34 U-Net (transfer learning)
│   ├── 08_change_detection.{sh,py}  # apply model to 2020/2024, map built-up change
│   ├── 09_train_unet_v3.py          # recipe ablations (documented negative result)
│   ├── 10_eval_tta.py               # flip TTA + threshold calibration (IoU 0.593)
│   ├── 11_acquire_open_buildings.sh # Google Open Buildings v3 footprints for the AOI
│   ├── 12_label_noise_audit.sh      # quantifies the OSM label-noise floor
│   ├── 13_train_unet_consensus.py   # consensus-verified labels (verified IoU 0.80)
│   ├── 14_acquire_fabdem.sh         # FABDEM V1-2 tiles (bare-earth GLO-30)
│   ├── 15_hydrology_odaw.py         # breach/D8/streams/HAND/TWI over the basin
│   ├── 16_flood_risk_surface.py     # HAND hazard x built-up exposure product
│   ├── 17_sar_flood_validation.py   # Sentinel-1 flood mapping, 2018/2020 events
│   ├── 18_sar_flood_statistics.py   # 29-scene double-bounce + rainfall tests
│   ├── 19_hand_locality_test.py     # Mann-Whitney HAND test, flood localities
│   ├── 20_builtup_timeseries.py     # 2016-2023 built-up/encroachment trend (2.5D)
│   ├── 21_acquire_drain_survey.sh   # Open Cities/GARID surveyed drain network
│   ├── 22_design_storm_idf.py       # ERA5 IDF (documented negative result)
│   ├── 23_drain_capacity.py         # Manning capacity, siltation response (Stage 5)
│   ├── 24_acquire_middle_odaw_imagery.sh  # Open Cities scene catalogue (remote-read)
│   ├── 25_acquire_middle_odaw_labels.sh   # OSM + Open Buildings for the new AOI
│   ├── 26_build_middle_odaw_tiles.py      # consensus tile corpus at 5 cm
│   ├── 27_transfer_eval.py                # zero-shot transfer test (+ --control)
│   └── 28_joint_finetune.py               # leave-one-site-out fine-tuning
└── accra_flood/               # working tree (data dirs are gitignored)
    ├── data/                  # DEM (regenerated)
    ├── drains/                # surveyed drainage network (regenerated)
    ├── middleodaw/            # four upstream communities; scenes.json is versioned
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
rather than a generalisable representation. The principal empirical finding of the
Stage-2 work is that segmentation fidelity here is **label-bound, not
capacity-bound**: the binding constraint is the geometric quality of the OSM
supervision, not the network. Prospective work therefore prioritises (i) refining a
modest corpus of crisp, manually corrected building labels to lift the ceiling;
(ii) temporally consistent change detection (co-registration plus siamese or
post-classification methods) in place of naïve mask differencing; and (iii)
extending the corpus to upstream communities (Alogboshie, Alajo, Akweteyman) to
guard against a settlement-specific representation.

Item (iii) has since been carried out and then resolved. It first converted a
suspicion into a measurement — the Old Fadama checkpoint loses 0.28 pooled
verified IoU on four upstream communities against an essentially unchanged label
floor — and leave-one-site-out fine-tuning then showed that loss to be a
coverage limitation rather than a ceiling, recovering a mean of +0.287 on sites
the model has never seen. What remains is engineering rather than diagnosis:
train a single production checkpoint on all five sites (the per-fold models
exist only to measure generalisation), and widen the middle-Odaw windows from
512 m to Old Fadama's 1024 m via `MIDODAW_WINDOW` for the extra volume. Two
limits are worth restating because they did not move. Alogboshie remains the
hardest site after fine-tuning, so settlement morphology still costs accuracy
even when it is represented in training. And per-footprint delineation in dense
fabric remains bounded by label granularity, which no amount of additional
sites addresses; that requires instance-level annotation.

The Stage-5 hydraulics sharpen that third priority into a specific and testable
objective. Having shown that the surveyed cross-sections are adequate when clean
and that failure is governed instead by siltation, the decisive unmeasured
quantity is *which* drains are obstructed — a state variable, varying on the
timescale of a wet season, that no cross-section survey can capture and that the
OSM `smoothness` tag does not resolve. The two halves of this repository meet
there: the Open Cities drone imagery covering Alogboshie, Akweteyman, Alajo and
Nima is the same 2–5 cm material Stage 2 already trains on, and it overlaps the
surveyed network exactly. Segmenting refuse and silt within the drain corridor,
and driving the Stage-5 blockage parameter from that observation rather than
from a scenario sweep, would convert the present conditional result into a
measured one. Three further gaps are noted without being minimised: the trunk
drains that carry accumulated flow are absent from the surveyed set, so
basin-scale backwater is unmodelled; no open rainfall product resolves the
convective intensities that actually cause the flooding, which bounds every
absolute figure reported here from below; and the longitudinal grades are
constrained to an engineering range rather than measured, though the ranking is
invariant to that choice.

## 9. Licence and Citation

Source code is released under the MIT Licence (`LICENSE`). Derived data inherit the
upstream licences enumerated in §4. Please cite this work via `CITATION.cff`.
