"""Prompt synthesis helpers for GPT assisted law evolution."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import yaml


class PromptSynthesizer:
    """Generate mutation prompts from a small grammar specification."""

    def __init__(self, grammar_path: Path, seed: int | None = None) -> None:
        self.grammar_path = Path(grammar_path)
        self.random = random.Random(seed)
        self.grammar = yaml.safe_load(self.grammar_path.read_text())

    def generate_prompt(self, variant: str = "default", context: str | None = None) -> str:
        variants = self.grammar.get("variants", {})
        if variant not in variants:
            variant = "default"
        spec = variants[variant]
        template = self.random.choice(spec["templates"])
        substitutions = [self.random.choice(options) for options in spec["mutations"]]
        prompt = template.format(*substitutions)
        if context:
            prompt = context + "\n\n" + prompt
        return prompt

    def batch_generate(
        self, count: int, variant: str = "default", context: str | None = None
    ) -> List[str]:
        return [self.generate_prompt(variant=variant, context=context) for _ in range(count)]


__all__ = ["PromptSynthesizer"]

