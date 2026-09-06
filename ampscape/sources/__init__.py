"""Focal node / source-strength / ground configuration generators (brief §6)."""

from ampscape.sources.config import SourceConfig
from ampscape.sources.generators import (
    SourceSample,
    generate_all,
    sample_advanced,
    sample_omniscape,
    sample_points,
    sample_regions,
    sample_wall_to_wall,
)
from ampscape.sources.graph import build_conductance_graph, component_labels, laplacian

__all__ = ["SourceConfig", "SourceSample", "generate_all", "sample_advanced", "sample_omniscape", "sample_points",
           "sample_regions", "sample_wall_to_wall", "build_conductance_graph", "component_labels", "laplacian"]
