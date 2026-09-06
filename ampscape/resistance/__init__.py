"""Covariates -> resistance surfaces via YAML-driven resistance tables (brief §5)."""

from ampscape.resistance.random_table import perturb_table
from ampscape.resistance.tables import ResistanceTable, apply_table, load_tables

__all__ = ["ResistanceTable", "apply_table", "load_tables", "perturb_table"]
