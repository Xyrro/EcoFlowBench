"""Randomly perturbed resistance tables (brief §5, item 5).

``random_table`` decorrelates surrogates from any single expert table: starting from a base
class table, every landcover value and road penalty is multiplied by ``exp(N(0, log_sd))``
(log-normal jitter), values are re-clipped to ``[1, r_max]`` and the result is a fully
specified :class:`ResistanceTable` whose ``random`` field records ``(base_table_id, seed,
log_sd)`` so it can be regenerated bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from ampscape.resistance.tables import ResistanceTable


def perturb_table(base: ResistanceTable, seed: int, log_sd: float = 0.5,
                  table_id: str | None = None) -> ResistanceTable:
    """Return a new table with log-normally jittered class values and road penalties."""
    if base.base != "landcover":
        raise ValueError("perturb_table needs a landcover-based table")
    rng = np.random.default_rng(int(seed))
    lc = {c: float(np.clip(v * np.exp(rng.normal(0.0, log_sd)), 1.0, base.r_max)) for c, v in base.landcover.items()}
    d = base.model_dump()
    d["landcover"] = lc
    if base.roads is not None:
        d["roads"]["by_class"] = {c: float(np.clip(v * np.exp(rng.normal(0.0, log_sd)), 0.0, base.r_max))
                                  for c, v in base.roads.by_class.items()}
    if base.slope is not None:
        d["slope"]["per_degree"] = float(np.clip(base.slope.per_degree * np.exp(rng.normal(0.0, log_sd)), 0.0, 1.0))
    d["table_id"] = table_id or f"random_{base.table_id}_{int(seed)}"
    d["description"] = f"log-normal perturbation (sd={log_sd}) of {base.table_id} v{base.version}, seed {int(seed)}"
    d["random"] = {"base_table_id": base.table_id, "base_version": base.version, "base_sha256": base.sha256,
                   "seed": int(seed), "log_sd": float(log_sd)}
    t = ResistanceTable.model_validate(d)
    t._sha256 = base.sha256  # type: ignore[attr-defined]
    return t
