import numpy as np
import yaml
import pytest

h5py = pytest.importorskip("h5py")

from law_operator_training import LawOperatorTrainer


def _create_law_checkpoint(path):
    time_steps, radial, angular = 8, 3, 2
    grid = np.linspace(0.0, 1.0, time_steps * radial * angular)
    law_tensor = grid.reshape(time_steps, radial, angular)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("law_tensor", data=law_tensor)
        handle.attrs["time"] = time_steps
        handle.attrs["radial"] = radial
        handle.attrs["angular"] = angular
    return law_tensor


def test_operator_encoding_determinism(tmp_path):
    law_path = tmp_path / "chk_law_tensor.h5"
    _create_law_checkpoint(law_path)

    trainer = LawOperatorTrainer(str(law_path), n_components=3)
    trainer.load_law_field()
    trainer.reduce_operators()

    tokens_a = trainer.export_gpt_tokens(tmp_path / "tokens_a.yaml")
    tokens_b = trainer.export_gpt_tokens(tmp_path / "tokens_b.yaml")

    assert tokens_a == tokens_b
    with (
        open(tmp_path / "tokens_a.yaml", "r", encoding="utf-8") as handle_a,
        open(tmp_path / "tokens_b.yaml", "r", encoding="utf-8") as handle_b,
    ):
        assert yaml.safe_load(handle_a) == yaml.safe_load(handle_b)


def test_operator_reprojection_consistency(tmp_path):
    law_path = tmp_path / "chk_law_tensor.h5"
    law_tensor = _create_law_checkpoint(law_path)

    suggestion = tmp_path / "suggestion.yaml"
    with open(suggestion, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "mutate": {
                    "operator_weights": [0.0, 0.0, 0.0],
                    "shift_basis": True,
                    "reproject_after": True,
                }
            },
            handle,
        )

    trainer = LawOperatorTrainer(str(law_path), n_components=3)
    trainer.load_law_field()
    trainer.reduce_operators()

    baseline = trainer.reproject_operators_to_law().copy()
    trainer.attach_gpt_guidance(str(suggestion), influence_strength=0.5)
    reprojection = trainer.reproject_operators_to_law()

    assert np.allclose(baseline, law_tensor)
    assert np.allclose(reprojection, law_tensor)
