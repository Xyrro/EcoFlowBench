"""Phase 1 sanity tests: package imports, pinned versions, and config directories exist."""

from __future__ import annotations

import importlib
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SUBPACKAGES = [
    "landscapes", "resistance", "sources", "solve", "io", "splits",
    "data", "models", "metrics", "eval", "viz",
]


def test_package_imports():
    import ecoflowbench

    assert ecoflowbench.__version__ == "0.0.1"
    for name in SUBPACKAGES:
        mod = importlib.import_module(f"ecoflowbench.{name}")
        assert mod.__doc__, f"ecoflowbench.{name} has no docstring"


def test_layout_matches_brief():
    for d in ["configs/tasks", "configs/landscapes", "configs/resistance_tables",
              "configs/solver", "configs/datasets", "julia/EcoFlowBenchSolve.jl/src",
              "scripts", "tests", "docs", "notebooks", "paper"]:
        assert (ROOT / d).is_dir(), d


def test_dependencies_are_pinned():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"] + data["project"]["optional-dependencies"]["dev"]
    unpinned = [d for d in deps if "==" not in d]
    assert not unpinned, f"unpinned dependencies: {unpinned}"


def test_julia_project_pins_solvers():
    proj = tomllib.loads((ROOT / "julia/EcoFlowBenchSolve.jl/Project.toml").read_text())
    assert {"Circuitscape", "Omniscape"} <= set(proj["deps"])
    manifest = (ROOT / "julia/EcoFlowBenchSolve.jl/Manifest.toml").read_text()
    assert "[[deps.Circuitscape]]" in manifest and "[[deps.Omniscape]]" in manifest


def test_solver_preset_matches_spec():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/solver/circuitscape_reference.yaml").read_text())
    assert cfg["solver"] == "cholmod"
    assert cfg["precision"] == "double"
    assert cfg["connect_four_neighbors_only"] is False
    assert cfg["connect_using_avg_resistances"] is False
