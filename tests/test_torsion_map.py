import numpy as np
import pytest

pytest.importorskip("plotly.graph_objects", reason="plotly is required for torsion map tests")

from law_tools.dashboard.torsion_map import TorsionMapResult, make_torsion_heatmap


def test_make_torsion_heatmap_marks_values_above_threshold():
    result = TorsionMapResult(
        r=np.array([[1.0, 2.0], [1.0, 2.0]]),
        theta=np.array([[0.0, 0.0], [np.pi, np.pi]]),
        torsion=np.array([[0.4, 0.6], [0.7, 0.3]]),
    )

    fig = make_torsion_heatmap(result, chaos_threshold=0.5)

    assert fig.data[0].type == "heatmap"
    assert any(ann.text == "chaos > 0.5" for ann in fig.layout.annotations)

    assert len(fig.data) == 2
    scatter = fig.data[1]
    assert scatter.type == "scatter"
    assert len(scatter.x) == 2
    assert sorted(scatter.x) == [1.0, 2.0]
    assert sorted(scatter.y) == pytest.approx([0.0, np.pi])


def test_make_torsion_heatmap_without_values_above_threshold():
    result = TorsionMapResult(
        r=np.array([[1.0, 2.0], [1.0, 2.0]]),
        theta=np.array([[0.0, 0.0], [np.pi, np.pi]]),
        torsion=np.array([[0.2, 0.3], [0.1, 0.2]]),
    )

    fig = make_torsion_heatmap(result, chaos_threshold=0.5)

    assert len(fig.data) == 1
    assert any(ann.text == "chaos > 0.5" for ann in fig.layout.annotations)
