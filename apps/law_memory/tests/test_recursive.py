from __future__ import annotations

from ..recursive import ParameterSet, build_parameter_grid


def test_build_parameter_grid() -> None:
    params = build_parameter_grid([0.3, 0.4], [0.1], [0.0, 0.1])
    assert len(params) == 4
    assert isinstance(params[0], ParameterSet)

