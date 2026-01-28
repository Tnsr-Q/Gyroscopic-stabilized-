from __future__ import annotations

import numpy as np

from ..analysis import (
    classify_trajectory,
    compute_coherence_embedding,
    detect_fixed_point,
)


def test_detect_fixed_point_constant_profile() -> None:
    final = np.ones((10, 5))
    result = detect_fixed_point(final)
    assert result.is_fixed
    assert np.all(result.delta == 0.0)


def test_classify_trajectory_oscillation() -> None:
    t = np.linspace(0, 4 * np.pi, 40)
    x = np.sin(t)[:, None]
    label = classify_trajectory(x)
    assert label == "oscillation"


def test_embedding_is_symmetric(law_stack: np.ndarray) -> None:
    embedding = compute_coherence_embedding(law_stack)
    assert embedding.shape[0] == embedding.shape[1]
    np.testing.assert_allclose(embedding, embedding.T)

