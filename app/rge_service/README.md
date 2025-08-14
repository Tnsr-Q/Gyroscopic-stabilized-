# rge_service

Minimal 1-loop SM RGEs for (λ, y_t, g1, g2, g3, ξ).

### API
```python
from rge_service import run_rge
import numpy as np

scenario = {"model":"SM", "loops":1, "inputs":{"mt":172.5,"mh":125.1,"alpha_s":0.1181,"xi_0":0.1}}
mu = np.logspace(2, 10, 200)
out = run_rge(scenario, mu)
# out keys: "mu","lambda","yt","g"(n,3),"xi"
```

### Guarantees

* Outputs are finite (`nan_to_num` guard).
* Asymptotic freedom rough check: g3 typically decreases with μ, g1 increases.

---