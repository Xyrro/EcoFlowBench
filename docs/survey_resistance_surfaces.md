# Survey: published resistance surfaces that are downloadable under a permissive licence

Scope (external review item 5, 2026-09-06, ≤ 2 h): find Circuitscape / Omniscape studies whose
*resistance surfaces* (not only current maps) can be downloaded under CC0 / CC BY / public domain,
as candidate real-world inputs for a "published-resistance" evaluation set. **Nothing was downloaded.**
Licences were read from the repository metadata (Zenodo / figshare / Dryad APIs, USGS data page);
items marked "unverified" could not be checked from the cluster (403) and rely on catalogue text.

| # | Study / dataset | Repository, DOI | Licence | Format, resolution, extent | Resistance surface included? | Notes |
|---|---|---|---|---|---|---|
| 1 | Marsoner, Simion, Giombini, Egarter Vigl (Eurac) — *Maps of ecosystem multifunctionality and ecological connectivity for identifying Green Infrastructure networks in the European Alps* | Zenodo 10.5281/zenodo.6602481 | **CC BY 4.0** | GeoTIFF; "Landscape permeability.zip" 159 MB; 10 Alpine pilot regions (FR, DE, AT, SI, IT, CH); resistance reclassified from a 5 m LULC map | **Yes** — forest-mammal landscape resistance + ecological network outputs | best candidate: multiple regions, explicit resistance layer, permissive |
| 2 | Northern raccoon range-expansion study (Europe) — *Crossing borders: connectivity analyses reveal potential patterns of range expansion of the Northern raccoon in Europe* | figshare 10.6084/m9.figshare.27311484.v1 | **CC BY 4.0** | GeoTIFF in "output maps.zip" (121 MB): suitability, **resistance**, connectivity for Europe (+ QGIS styles) | **Yes** (continental Europe, SDM-derived resistance) | single species, SDM-based; good contrast to expert tables |
| 3 | van Rees et al. — *Landscape genetics identifies streams and drainage infrastructure as dispersal corridors for an endangered wetland bird* (Hawaiian gallinule, Oʻahu) | Dryad 10.5061/dryad.p90b87p; paper 10.1002/ece3.4296 | **CC0 1.0** | ASCII grids, ~299 MB, several alternative resistance hypotheses; island of Oʻahu | **Yes** — multiple resistance surfaces (streams/ditches as corridors) | small extent; Circuitscape used in the paper |
| 4 | Saltwater crocodile landscape genetics (Northern Territory, Australia) — *Landscape layer for resistance* | Dryad 10.5061/dryad.q2bvq83gb; paper 10.1111/mec.16310 | **CC0 1.0** | ASCII, 3 km cells, 325 × 202 | **Yes** — categorical base layer for ResistanceGA (optimised surfaces derived from it) | very coarse (3 km); categorical only |
| 5 | Buchholtz & Kreitler (USGS) — *Circuit-based potential fire connectivity and relative flow patterns in the Great Basin, United States, 270 m* | USGS 10.5066/P9EA3E00 | **CC0 1.0 / public domain** | raster, 270 m, Great Basin | Partial: outputs (cumulative + normalised current) with fire-spread *conductance* as input; resistance recoverable if the conductance raster is in the release (to verify at download) | Omniscape-based; non-animal "resistance" (fire) |
| 6 | Brennan et al. 2022 — *Functional connectivity of the world's protected areas* | Zenodo 10.5281/zenodo.6473366 | **CC BY 4.0** | Global_MMP.tif (9.6 MB, global mammal movement probability = current density), PAI csv | **No** — outputs only (resistance is derived from the Human Footprint by a documented transform) | useful as an *output* reference, not as input |
| 7 | Omniscape results for the Western Washington Habitat Connectivity Assessment (WA DFW / WHCWG) | USGS ScienceBase item 5d2cd26be4b038fabe22cf2d | unverified (403 from the cluster); USGS-hosted items are usually public domain but WHCWG products carry their own terms | raster outputs | unverified whether resistance layers are included | check from a browser |
| 8 | McRae et al. 2016 — *Omnidirectional connectivity for resilient terrestrial landscapes in the Pacific Northwest* (TNC) | Data Basin f5ff92497fb44012869981b4bbebd2eb | unverified (Data Basin per-dataset terms) | raster outputs | unverified | the original Omniscape application; worth a manual check |
| 9 | Data Basin "Functional Connectivity Index" (South Atlantic LCC: black bear / red wolf / cougar / timber rattlesnake) | Data Basin 19b6b3574c554433a41197e0b8bc170e, ff279bb7… | unverified | raster | unverified | expert-opinion resistance mentioned in the description |
| 10 | Circuitscape.jl / Omniscape.jl test inputs (10 × 10, 30 × 30 rasters) | GitHub, MIT | **MIT** | ASCII/TIF | yes (toy) | already used in our smoke tests; not landscapes |

Not found within the time box: a CC-licensed multi-region *expert-table* resistance set for large
mammals at ≤ 100 m outside Europe; Belote et al. 2016 and Dickson et al. 2017 (US corridors / ecological
flow) publish results through PLoS/CSP without a repository link for the resistance rasters.

## Download status (2026-09-07, owner-approved)

| candidate | status |
|---|---|
| 1 Eurac Alps landscape permeability | downloaded (159 MB, sha256 in `data/sources/manifest.json`) |
| 2 Raccoon Europe output maps | downloaded (121 MB, sha256 in the manifest) |
| 3 Hawaiian gallinule resistance layers (Dryad, 299 MB `f_wet_negbin.zip`) | **blocked**: Dryad answers 401 on the API file download and 403 on the web file stream for scripted requests from the cluster; needs a browser download by the owner (or a Dryad API token), then `data/sources/published_hawaiian_gallinule_dryad.zip` + a manifest entry |

## Recommendation (no action taken)

Candidates 1–3 (CC BY 4.0 / CC0, explicit resistance rasters, ≥ 3 regions) are suitable for a small
**"published-resistance" evaluation set**: tile them at tier S/M, solve with the reference pipeline,
and report them as an additional OOD test (real resistance rather than AmpScape's tables). Total
download ≈ 0.6 GB, within the gate. Candidate 5 is a distinct "non-ecological resistance" case. Items
7–9 need a browser check of the terms before any download.
