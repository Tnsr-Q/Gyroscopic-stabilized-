# bounce_service

Thin-wall O(4) shooting (no gravity) to estimate bounce action S4.

### API
```python
from bounce_service import bounce_action

# Simple double-well-ish toy potential
def V(h): return 0.25*(h**2-1.0)**2 - 0.02*h
out = bounce_action(V)
# out keys: S4, profile(None), rate(str), h_false
```