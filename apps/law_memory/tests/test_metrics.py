from __future__ import annotations

import numpy as np

from ..metrics import deformed_metric_profile, inject_into_metrics, interpolate_law_at_r


def test_interpolate_law_at_r() -> None:
    r = np.linspace(1.0, 3.0, 5)
    law = np.linspace(0.0, 1.0, 5)
    query = np.array([1.5, 2.5])
    interp = interpolate_law_at_r(r, law, query)
    expected = np.interp(query, r, law)
    np.testing.assert_allclose(interp, expected)


def test_deformed_metric_profile_shapes() -> None:
    r = np.linspace(0.5, 2.5, 6)
    law = np.sin(r)
    phi, gamma = deformed_metric_profile(r, law)
    assert phi.shape == r.shape
    assert gamma.shape == r.shape


def test_inject_into_metrics_summary() -> None:
    r = np.linspace(0.5, 2.5, 6)
    law = np.sin(r)
    summary = inject_into_metrics(r, law)
    assert "phi_mean" in summary
    assert "gamma_mean" in summary

