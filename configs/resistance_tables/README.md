# Resistance tables

YAML tables consumed by `ecoflowbench.resistance` (schema: `tables.py::ResistanceTable`).
Values are **EcoFlowBench's own expert-style parameterisations**: the *ordering* of classes and
the *form* of each term follow the cited literature, but the numeric values are not copied
from any single study (published tables are species- and region-specific). Bowman et al. (2020)
show current-density maps are robust to the magnitudes as long as ranks are preserved, which
is the property we rely on. Every table is versioned and its sha256 is stored per sample.

| table_id | base | r_max | roads | slope | water | purpose |
|---|---|---|---|---|---|---|
| generic_hm | 1 + 999·gHM² | 1000 | — | — | barrier 1000 | continuous human-modification transform (Brennan et al. 2022 style) |
| large_mammal | WorldCover classes | 1000 | additive by GRIP4 class | ×(1+0.03·deg) | barrier 800 | forest-low / crop-moderate / urban-very-high, roads strong |
| amphibian | WorldCover classes | 1000 | additive, extreme | ×(1+0.06·deg) | permeable (2) | wetlands/forest low, dry open high, roads extreme |
| forest_bird | WorldCover classes | 100 | additive, mild | — | 20 | forest low, open moderate, urban high; elevation bands |
| random_lm_20260905 | perturbed large_mammal | 1000 | perturbed | perturbed | barrier 800 | decorrelation table (held out as `test_ood_table` candidate) |

WorldCover 2021 class codes: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare,
70 snow/ice, 80 water, 90 wetland, 95 mangrove, 100 moss/lichen. GRIP4 road types: 1 highway,
2 primary, 3 secondary, 4 tertiary, 5 local.
