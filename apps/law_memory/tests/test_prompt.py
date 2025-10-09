from __future__ import annotations

from pathlib import Path

from ..prompt_synth import PromptSynthesizer


def test_prompt_generation(tmp_path: Path) -> None:
    grammar = tmp_path / "grammar.yaml"
    grammar.write_text(
        """
variants:
  default:
    templates:
      - "Use {0} and {1}"
    mutations:
      - ["A"]
      - ["B"]
"""
    )

    synth = PromptSynthesizer(grammar)
    prompt = synth.generate_prompt()
    assert "Use" in prompt


def test_batch_generation(tmp_path: Path) -> None:
    grammar = tmp_path / "grammar.yaml"
    grammar.write_text(
        """
variants:
  default:
    templates:
      - "Case {0}"
    mutations:
      - ["X", "Y"]
"""
    )

    synth = PromptSynthesizer(grammar, seed=1)
    prompts = synth.batch_generate(3)
    assert len(prompts) == 3
    assert all(prompt.startswith("Case") for prompt in prompts)

