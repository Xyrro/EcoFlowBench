# Upstream data sources: availability, licences, redistribution

Verified 2026-09-05 from the primary download pages / bucket manifests (Phase 2 pilot scope).
"Derived rasters" = per-tile covariate channels EcoFlowBench stores (class codes, elevation,
slope, distance-to-road, distance-to-river, gHM) — never the original source tiles.

| Layer | Version / year | Access (verified) | Licence (verified) | Redistribute derived rasters? | Notes |
|---|---|---|---|---|---|
| **ESA WorldCover** land cover, 10 m | 2021 v200 | public S3 `s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{N|S}yy{E|W}xxx_Map.tif` (3°×3° COGs, EPSG:4326; e.g. N00E006 = 2.5 MB, N00E009 = 10 MB; grid: `esa_worldcover_grid.geojson` in the same bucket; region eu-central-1, `--no-sign-request`) | **CC BY 4.0** (ESA WorldCover data-access page) | **Yes** | attribution: "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium" |
| **Copernicus DEM GLO-30** | 2021 release (WorldDEM-30 based, © DLR 2010-2014, © Airbus 2014-2018) | public S3 `s3://copernicus-dem-30m/Copernicus_DSM_COG_10_{N|S}yy_00_{E|W}xxx_00_DEM/…_DEM.tif` (1°×1° COGs, ~45 MB each, `tileList.txt` lists 26 450 tiles) | **Copernicus WorldDEM-30 licence** (COP-DEM-GLO-30-F): Art. 4 grants reproduction, distribution, communication to the public, adaptation/modification; Art. 6 requires notices | **Yes, with notices** | required notice on modified data: "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" + liability sentence "The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30". A few tiles (some countries) are withheld from GLO-30 Public; fall back to GLO-90 (`s3://copernicus-dem-90m`) for those |
| **GRIP4** global roads | v4, 2018 (Meijer et al. 2018, ERL 13:064006) | `https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_Region{1..7}_vector_shp.zip` (regional shapefiles; global FGDB > 2 GB); 5-arcmin density rasters also available | globio.info states verbatim: "GRIP4 is provided under a Creative Commons License (CC-0) and is free to use." Secondary catalogues (FAO, UNDP GeoHub) label it CC BY 4.0; one catalogue says ODbL. **We treat it as CC BY 4.0 (the stricter of the two Creative Commons readings) and attribute.** | **Yes** (as distance / density rasters) | GRIP4 is compiled partly from OpenStreetMap; PBL states they verified public availability of sources. Because we store only distance-to-road rasters (non-reversible), residual ODbL exposure is minimal; documented for the datasheet |
| **HydroRIVERS** | v1.0 (HydroSHEDS) | `https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip` (544 MB global) or per-region (e.g. `HydroRIVERS_v10_na_shp.zip`) | HydroSHEDS licence = **CC BY 4.0** (hydrosheds.org/pages/license), "freely available for scientific, educational and commercial use" | **Yes** (distance-to-river raster) | citation Lehner & Grill 2013, Hydrol. Process. 27:2171 |
| **gHM** global Human Modification | v1, 2016 median year, 1 km (Kennedy et al. 2019, GCB) | figshare `https://ndownloader.figshare.com/files/13448294` (`gHM.zip`, 415 MB), DOI 10.6084/m9.figshare.7283087.v1 | **CC BY 4.0** (figshare record) | **Yes** | 300 m temporal version (Theobald et al. 2020, ESSD; Zenodo 3901815) is an option for tiers S/M later |
| **RESOLVE Ecoregions 2017** | 2017 (Dinerstein et al. 2017, BioScience 67:534) | `https://storage.googleapis.com/teow2016/Ecoregions2017.zip` (150 MB shapefile) | **CC BY 4.0** (ecoregions.appspot.com) | used only for stratification (biome / realm labels in the manifest) | |
| **WDPA** protected areas | monthly | protectedplanet.net, registration + terms acceptance | **Restrictive**: no redistribution of WDPA data, non-commercial only without written permission (protectedplanet.net/en/legal) | **No polygons.** Only derived focal-node masks (rasterised, unlabelled) may be stored, and the datasheet must say so | Phase 4 item; not downloaded in the Phase 2 pilot |
| OpenStreetMap | — | — | ODbL (share-alike) | **not used** (owner decision 2026-09-05) | replaced by GRIP4 |

## Combined data licence

All stored covariates derive from CC BY 4.0 sources or the Copernicus WorldDEM-30 licence
(which permits redistribution with notices). The dataset can therefore be released under
**CC BY 4.0** with an attribution block listing every upstream source above, plus the
Copernicus notices verbatim. WDPA-derived masks (if any) are stored without identifiers.

## Pilot download budget (Phase 2, 50 tiles)

| Item | Size |
|---|---|
| RESOLVE Ecoregions 2017 | 150 MB |
| gHM v1 1 km | 415 MB |
| HydroRIVERS regional shapefiles for continents with pilot tiles (≤ 6) | ≤ 480 MB |
| GRIP4 regional shapefiles for regions with pilot tiles (≤ 7) | see `docs/phase_02_report.md` (sizes recorded at download) |
| WorldCover / Copernicus DEM | windowed reads from COGs via `/vsicurl/`, only the pixels needed (≈ 2–20 MB per tile) |

Total is kept under the 5 GB gate; anything above requires owner approval.
