"""log10(C + ε·max C) transform: scale invariance, zero handling, inverse."""

from __future__ import annotations

import numpy as np

from ampscape.metrics import EPS, inverse_log10_eps, log10_eps


def test_log10_eps_properties():
    c = np.array([[0.0, 1e-4], [1e-2, 1.0]])
    y = log10_eps(c)
    assert y.dtype == np.float32 and y[1, 1] == np.float32(np.log10(1 + EPS))
    assert np.isclose(y[0, 0], np.log10(EPS))                       # zeros map to log10(eps·max) = -6
    # scale invariance up to a constant shift: same map ×1000 differs by exactly 3 decades
    np.testing.assert_allclose(log10_eps(1000 * c) - y, 3.0, atol=1e-5)
    # log1p is NOT scale invariant (the property that motivated the change)
    assert not np.allclose(np.log1p(1000 * c) - np.log1p(c), (np.log1p(1000 * c) - np.log1p(c))[1, 1])
    assert np.all(log10_eps(np.zeros((3, 3))) == 0)
    np.testing.assert_allclose(inverse_log10_eps(y, c.max()), c, atol=1e-7)
    # batched: per-map maximum
    b = np.stack([c, 10 * c])
    yb = log10_eps(b, axis=(1, 2))
    np.testing.assert_allclose(yb[1] - yb[0], 1.0, atol=1e-5)
