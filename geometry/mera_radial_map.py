# geometry/mera_radial_map.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

def calibrate_layer_to_radius(n_layers: int,
                              r_throat: float,
                              r_max: float,
                              scheme: str = "log") -> Dict[int, float]:
    """
    Produce a monotone map layer -> radius.
      - layer 0 near throat (UV), layer n-1 near IR boundary.
      - "log" : r(l) = r_throat * (r_max/r_throat)^{l/(n-1)}
      - "affine": linear spacing
    """
    if n_layers <= 1:
        return {0: r_throat}
    idx = np.linspace(0, 1, n_layers)
    if scheme == "affine":
        radii = r_throat + idx * (r_max - r_throat)
    else:
        radii = r_throat * (r_max / max(r_throat, 1e-9))**idx
    return {int(l): float(r) for l, r in enumerate(radii)}

def map_radius_to_layer(r: float, layer_to_radius: Dict[int, float]) -> int:
    """Return nearest layer to given radius."""
    items = sorted(layer_to_radius.items(), key=lambda kv: kv[1])
    best = min(items, key=lambda kv: abs(kv[1] - r))
    return int(best[0])