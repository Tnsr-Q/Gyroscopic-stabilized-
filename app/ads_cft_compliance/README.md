# ads_cft_compliance

Holography checkpack: RT-like area scaling, complementary recovery, compression law.

### API
```python
from ads_cft_compliance import holography_checkpack
import numpy as np

psi = np.random.randn(256) + 1j*np.random.randn(256)

class MiniCube:
    def __init__(self, edges=8): self._edges = [object()]*edges
    def minimal_cut_edges(self, _): return self._edges

res = holography_checkpack(psi, MiniCube())
```