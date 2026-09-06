# AmpScape shard schema

Schema version **0.2** (`ampscape.io.schema`). One HDF5 file per shard, one group per sample.

## Root attributes

`dataset_version`, `schema_version`, `pipeline_git_sha`, `created_at`.

## Sample group `/<sample_id>/`

Attribute `meta` (JSON) with the fields of `MetaModel`:

| field | type | note |
|---|---|---|
| `sample_id` | `<class 'str'>` | required |
| `dataset_id` | `<class 'str'>` | required |
| `family` | `<class 'str'>` | required |
| `tier` | `<class 'str'>` | required |
| `H` | `<class 'int'>` | required |
| `W` | `<class 'int'>` | required |
| `pixel_size_m` | `<class 'float'>` | required |
| `seed` | `<class 'int'>` | required |
| `generator` | `str | None` | optional |
| `generator_params` | `dict | None` | optional |
| `contrast` | `float | None` | optional |
| `resistance_table_id` | `str | None` | optional |
| `tile_id` | `str | None` | optional |
| `lat` | `float | None` | optional |
| `lon` | `float | None` | optional |
| `crs` | `str | None` | optional |
| `graph_connectivity` | `<class 'int'>` | optional |
| `source_config` | `<class 'dict'>` | required |
| `omniscape` | `<class 'dict'>` | required |
| `solver_name` | `<class 'str'>` | required |
| `solver_versions` | `<class 'dict'>` | required |
| `solver_preset` | `<class 'dict'>` | required |
| `qc_flags` | `list[str]` | optional |
| `created_at` | `<class 'str'>` | required |
| `pipeline_git_sha` | `<class 'str'>` | required |
| `dataset_version` | `<class 'str'>` | required |
| `resampling` | `dict | None` | optional |

### `inputs/`

| dataset | dtype | shape |
|---|---|---|
| `resistance` | float32 | (H, W) — R ∈ [1, r_max], 1.0 at NoData |
| `nodata_mask` | uint8 | (H, W) — 1 = NoData |
| `covariates` | float32 | (C, H, W) — real tiles; `attrs.channels` |

### `configs/<name>/`

Attributes `kind`, `task_ids`, `focal_table` (JSON), `source_meta` (JSON).

| kind | tasks | inputs | outputs (raw solver output) |
|---|---|---|---|
| points | T1,T2 | `focal_mask` int32 | `cum_current` float32, `reff` float64, `labels` int32, `pair_index` int32; K ≤ 4 also `pairwise_current`, `voltage` float32 (P, H, W) |
| wall_to_wall | T1W | `focal_mask` int32 | `cum_current` float32, `reff` float64, `labels` int32, `pair_index` int32; K ≤ 4 also `pairwise_current`, `voltage` float32 (P, H, W) |
| regions | T1R | `focal_mask` int32 | `cum_current` float32, `reff` float64, `labels` int32, `pair_index` int32; K ≤ 4 also `pairwise_current`, `voltage` float32 (P, H, W) |
| advanced | T3 | `source_strength` float32, `ground` int8 | `current` float32, `voltage` float32 |
| omniscape | T4 | `source_strength` float32 | `cum_current` float32, `flow_potential` float32, `normalized` float32 |

Output group attributes: `solver_stats` (JSON `SolveStats`), `qc_flags` (JSON list), `qc_pass` (bool).

Conventions: north-up row-major rasters; pair (i, j): node i grounded, 1 A injected at j; NoData pixels
hold 0 in output maps; nothing is normalised or clipped. See `docs/task_specification.md`.

## Parquet index

One row per (sample, config): identifiers, family/tier/generator/table/tile, K, placement, seed,
solver, timings, residuals, `qc_flags`, `qc_pass`, `qc_trainval`, split and OOD flags, shard file.
