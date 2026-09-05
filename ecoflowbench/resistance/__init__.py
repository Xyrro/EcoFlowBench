"""Covariates -> resistance surfaces via YAML-driven resistance tables (brief §5)."""

from ecoflowbench.resistance.random_table import perturb_table
from ecoflowbench.resistance.tables import ResistanceTable, apply_table, load_tables

__all__ = ["ResistanceTable", "apply_table", "load_tables", "perturb_table"]
