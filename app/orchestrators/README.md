# orchestrators

End-to-end pipelines that string services together.

### `higgs_pipeline.py`
```python
from orchestrators.higgs_pipeline import run_higgs_program
import numpy as np

scenario = {"model":"SM","loops":1,"inputs":{"xi_0":0.1}}
mu = np.logspace(2, 19, 400)
h  = np.logspace(2, 19, 300)

class MiniCube:
    def __init__(self, edges=8): self._edges = [object()]*edges
    def minimal_cut_edges(self, _): return self._edges
    @property
    def edges(self): return self._edges

results = run_higgs_program(scenario, mu, h, {"einstein_frame":True}, np.random.randn(512)+1j*np.random.randn(512), MiniCube())
```