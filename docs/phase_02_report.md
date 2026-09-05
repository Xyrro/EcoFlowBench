# Phase 2 report — landscape instance families

Date: 2026-09-05. Status: **complete, awaiting owner confirmation** before Phase 3 (resistance
surfaces). Same summary in `docs/status/latest.md`.

## What was built

| Item | Location |
|---|---|
| Synthetic generators: GRF (ℓ ∈ {2,8,32,128}, anisotropy, orientation), midpoint-displacement fractal (H ∈ {0.2,0.5,0.8}), NLMpy random cluster / planar / edge / distance gradient / mosaic, linear-barrier and patch-mosaic overlays, NoData blobs, contrast ladder {10,100,1000,10000}, documented prior, deterministic `sample_landscape(seed)` / `regenerate` | `ecoflowbench/landscapes/synthetic.py` |
| Real-tile pipeline: UTM tile grids, windowed COG readers (WorldCover, Copernicus DEM with GLO-90 fallback), gHM resampling, GRIP4 / HydroRIVERS distance + nearest-attribute rasters, slope, QC, GeoTIFF writer/reader | `ecoflowbench/landscapes/real.py` |
| Stratified sampling (biome × realm × gHM tercile, uniform-on-sphere land points, balanced round-robin, reserve list) | `ecoflowbench/landscapes/sampling.py` |
| Scripts: source downloader with sha256 manifest, tile sampler, tile extractor (idempotent, `--retry-rejected`, `--refresh`), figure scripts | `scripts/download_sources.py`, `sample_tiles.py`, `extract_tiles.py`, `plot_tiles.py`, `preview_synthetic.py` |
| Licences and access paths for every source | `docs/licenses.md` |
| Storage plan and HF quota issue | `docs/compute_env.md` §10 |
| Tests: 38 synthetic + 16 offline real + 2 network (opt-in) + 5 scaffold | `tests/` |

## What was verified

- **Synthetic:** shapes/dtypes/ranges for 7 generators × 3 shapes, determinism per seed,
  regeneration round-trip, NLMpy global-RNG isolation, GRF smoothness and anisotropy direction,
  barrier geometry and gap fraction, patch-mosaic class values, single-connected-component NoData,
  contrast endpoints exact. Visual check: `docs/figures/synthetic_gallery.png`.
- **Real:** UTM EPSG for 6 known cities, grid reproducibility and extent, WorldCover/Copernicus
  tile naming across hemispheres and the lat-0/lon-3 corners, slope on a plane, line rasterisation
  + distance, GeoTIFF round-trip, NaN-safe merge of partially covering sources (regression test
  for the bug found in the pilot), nearest infill. Network tests read real COGs for a tile
  straddling four DEM tiles and a southern-hemisphere tile.
- **Pilot extraction (tier S, 128 px @ 100 m):** 60 / 60 tiles accepted on the final run,
  14 biomes, 5 realms, 55 distinct strata, 34 UTM zones.
  Map: `docs/figures/pilot_tiles_map.png`; gallery: `docs/figures/pilot_tiles_gallery.png`;
  per-tile quicklooks in `data/tiles/pilot/quicklooks/` (not committed).

## Measured numbers

| Quantity | Value |
|---|---|
| Source downloads (pilot set) | 12 files, 2.76 GB (under the 5 GB gate), sha256 in `data/sources/manifest.json` |
| Unzipped sources on scratch | 11 GB (GRIP4 regions 1,2,3,5,7; HydroRIVERS na/sa/af/au/eu; RESOLVE; gHM) |
| Extraction throughput (login node, 4 threads, I/O-bound) | 60 tiles in ~62 s ≈ 1 s/tile wall; ~50 MB transferred from COGs total |
| Tile GeoTIFF size (8 bands, 128², deflate) | ~230 KB each; 60 tiles = 14 MB |
| DEM seam infill | mean 0.42% of pixels, max 3.61% |
| gHM NoData infill (water/coast) | mean 4.64%, max 24.99% |
| gHM tercile edges (recorded) | 0.0329, 0.1821 |
| gHM terciles in accepted tiles (0/1/2) | 15 / 22 / 23 |
| Realms | {'Afrotropic': 8, 'Australasia': 13, 'Nearctic': 6, 'Neotropic': 20, 'Palearctic': 13} |
| Tiles with no GRIP4 road within 12.8 km | 12 (remote tundra/desert/boreal; distance channel = sentinel 128 km) |
| Tiles with no HydroRIVERS reach | 9 |

Biome coverage of the accepted pilot tiles:

| Biome | tiles |
|---|---|
| Tropical & Subtropical Moist Broadleaf Forests | 8 |
| Deserts & Xeric Shrublands | 7 |
| Temperate Grasslands, Savannas & Shrublands | 7 |
| Tropical & Subtropical Dry Broadleaf Forests | 6 |
| Mediterranean Forests, Woodlands & Scrub | 6 |
| Temperate Broadleaf & Mixed Forests | 5 |
| Tropical & Subtropical Grasslands, Savannas & Shrublands | 4 |
| Temperate Conifer Forests | 4 |
| Montane Grasslands & Shrublands | 4 |
| Boreal Forests/Taiga | 4 |
| Flooded Grasslands & Savannas | 2 |
| Mangroves | 1 |
| Tropical & Subtropical Coniferous Forests | 1 |
| Tundra | 1 |

## Findings and deviations (all logged in `DECISIONS.md`)

1. **HF private storage quota is 100 GB** on a free account (verified). The proposed v1.0 ladder
   is ≈ 215 GB compressed and even a trimmed private v1.0 would need PRO (1 TB) or a public repo.
   `docs/compute_env.md` §10.2 gives three options; **owner decision needed** before Phase 5.
2. **GRIP4 licence text** on globio.info says "Creative Commons License (CC-0)"; FAO/UNDP
   catalogues say CC BY 4.0. We attribute as CC BY 4.0 (stricter). GRIP4 is partly OSM-derived;
   we store only distance/class rasters (non-reversible). Documented in `docs/licenses.md`.
3. **Copernicus DEM** licence permits redistribution of modified data with mandatory notices
   (quoted in `docs/licenses.md`); a few country tiles are withheld from GLO-30, handled by the
   GLO-90 fallback (not triggered in the pilot).
4. **Bug found and fixed:** NaN-fill merge overwrote earlier DEM tiles when a grid straddled
   several source tiles (12/62 first-run rejections were this bug, not data gaps). Regression
   test added; all 60 tiles re-extracted with the fixed code.
5. **Bilinear seam rows** between source tiles are infilled by nearest neighbour when < 10 % of
   pixels (mean 0.4 %); fraction recorded per tile in QC.
6. Pilot restricted to GRIP4 regions 1,2,3,5,7 to stay under the download gate; Europe and
   S/E Asia (1.9 GB more) are needed for full-scale global coverage.
7. **WDPA** cannot be redistributed; Phase 4 will store only unlabelled focal masks (or skip WDPA).

## Risks / open items

- HF storage decision (above).
- Full-scale real tiling will need the remaining GRIP4/HydroRIVERS regions (~2.3 GB → total
  ≈ 5.1 GB of sources, i.e. **just over the 5 GB gate; asking now** so it is settled before scale-up).
- Extraction runs on the login node because it is network-bound (≈ 1 s CPU per tile); at
  thousands of tiles I will split it into a login-node COG-window prefetch stage and a Slurm
  reprojection stage, as noted in `docs/compute_env.md`.
- Tercile edges were computed on 2 883 candidates from the pilot realms; they will be recomputed
  once on the global candidate pool at full scale and then frozen.

## Next step (Phase 3, on confirmation)

Resistance tables (`configs/resistance_tables/*.yaml`, ≥ 4 literature-sourced) and the
covariates → resistance mapping in `ecoflowbench/resistance/`, applied to all 60 pilot tiles
with range/mask unit tests.
