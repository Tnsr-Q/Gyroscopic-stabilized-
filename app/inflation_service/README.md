# inflation_service

Higgs-inflation slow-roll (Einstein frame): returns (n_s, r, A_s, h_star, h_end).

### API
```python
from rge_service import run_rge
from inflation_service import higgs_inflation_observables
import numpy as np

mu = np.logspace(2, 19, 600)
running = run_rge({"model":"SM","loops":1,"inputs":{"xi_0":10.0}}, mu)
obs = higgs_inflation_observables(running, N_star=50)
```
