"""Network tests for the real-tile pipeline; run with EFB_NETWORK_TESTS=1 on the login node."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ecoflowbench.landscapes import real

pytestmark = pytest.mark.skipif(os.environ.get("EFB_NETWORK_TESTS") != "1",
                                reason="set EFB_NETWORK_TESTS=1 to run tests that read public COGs")


@pytest.mark.network
def test_worldcover_and_copdem_two_tile_straddle():
    # Ural foothills tile straddles a 1° DEM boundary in both axes (4 DEM tiles)
    g = real.make_grid(57.94, 62.935, 128, 100.0)
    lc = real.read_worldcover(g)
    dem_raw = real.read_copdem(g)
    dem, filled = real.fill_nearest(dem_raw)
    assert (lc == 0).mean() < 0.05
    assert np.isnan(dem_raw).mean() < 0.05          # only seam rows may be missing
    assert filled < 0.05 and np.isfinite(dem).all()
    assert 40 < dem.min() < dem.max() < 300


@pytest.mark.network
def test_copdem_southern_hemisphere():
    dem, _ = real.fill_nearest(real.read_copdem(real.make_grid(-9.033, -39.617, 128, 100.0)))
    assert np.isfinite(dem).all() and dem.max() > 300
