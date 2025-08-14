# veff_service

RG-improved Higgs effective potential with optional Einstein frame mapping.

### API
```python
from veff_service import higgs_veff
from rge_service import run_rge
import numpy as np

mu = np.logspace(2, 10, 200)
running = run_rge({"model":"SM","loops":1,"inputs":{"xi_0":0.1}}, mu)
h = np.logspace(2, 10, 150)
res = higgs_veff(h, running, {"einstein_frame":True, "loops":1})
# res keys: h, Veff, K (if Einstein frame), meta_scale, lambda_at
```
