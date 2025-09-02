# control/recursive_loop.py
from __future__ import annotations
from typing import Callable, Dict, Any, Sequence
import math

class RecursiveController:
    """
    Multi-scale recursive loop:
      - outer scales feed inner as targets
      - inner returns residuals to adjust outer
    """
    def __init__(self,
                 step_fns: Sequence[Callable[[Dict[str, Any]], Dict[str, Any]]],
                 max_depth: int = 3):
        self.step_fns = list(step_fns)
        self.max_depth = max(1, max_depth)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Depth-first recursion with residual feedback."""
        def _recurse(depth: int, s: Dict[str, Any]) -> Dict[str, Any]:
            s_local = dict(s)
            if depth >= self.max_depth:
                # base step at finest scale
                for fn in self.step_fns:
                    s_local.update(fn(s_local))
                return s_local

            # coarse → fine
            for fn in self.step_fns:
                s_local.update(fn(s_local))

            # feed forward as "target"
            s_local['target'] = s_local.get('target', {})  # ensure dict
            s_child = _recurse(depth + 1, s_local)

            # residual correction back to parent
            resid = {}
            for k in ('ane_margin', 'phase_indicator', 'conditioning'):
                if k in s_child and k in s_local:
                    resid[k] = s_local[k] - s_child[k]
            s_local['residual'] = resid
            return s_local

        return _recurse(0, state)