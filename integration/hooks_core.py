# integration/hooks_core.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch

from physics.wormhole_mt import MTWormhole
from physics.torsion_ec import torsion_spring_force
from physics.coherence_gamma import gamma_feedback_step
from geometry.mera_radial_map import calibrate_layer_to_radius
from services.rt_metrics import RealTimeGaugeMetrics

def make_mt_wormhole(rcc, r0: float = 0.5) -> MTWormhole:
    # Plug your Planck-factor hook into Φ(r) if available
    def Phi(r: float) -> float:
        try:
            return float(rcc.planck_redshift(float(r)))
        except Exception:
            return 0.0
    return MTWormhole(r0=r0, Phi=Phi)

def sense_fn_factory(rcc):
    rt = RealTimeGaugeMetrics(rcc)
    def _sense() -> Dict[str, Any]:
        m = rt.compute_realtime_metrics()
        # add center-law-field gradient proxy for proper-time slope
        try:
            if hasattr(rcc.state, 'law_field'):
                lf = torch.as_tensor(rcc.state.law_field, dtype=torch.float32)
                gx = torch.gradient(lf, dim=0)[0].abs().mean().item()
                gy = torch.gradient(lf, dim=1)[0].abs().mean().item()
                proper_time_grad = 0.5*(gx+gy)
            else:
                # Fallback: use entropy rate as proxy
                proper_time_grad = float(getattr(rcc.state, 'entropy', 0.1) * 0.1)
        except Exception:
            proper_time_grad = 0.1  # Safe default
        return dict(m, proper_time_grad=proper_time_grad)
    return _sense

def compute_fn_factory(rcc, mt: MTWormhole, spin_density: float = 0.02):
    # map MERA layers to radius once (updated if layers change)
    layers = getattr(rcc.mera, "layers", 6)
    layer_to_radius = calibrate_layer_to_radius(layers, r_throat=mt.r0, r_max=4.0*mt.r0)

    def _compute(obs: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal policy:
        # - If QEI headroom negative, soften phases near throat
        # - If conditioning high, reduce gains
        ane_margin = obs.get('qei_headroom', 0.0)
        conditioning = obs.get('total_conditioning', 1.0)
        phase_indicator = obs.get('phase_indicator', 0.0)

        # Torsion spring near throat
        F_t = torsion_spring_force(spin_density, mt.r0, mt.r0)

        # Gamma feedback (use stored gamma if present)
        gamma = float(getattr(rcc, "gamma_state", 0.3))
        proper = float(obs.get('proper_time_grad', 0.1))
        gamma_next, m_eff = gamma_feedback_step(gamma, proper, m0=1.0, k=0.8, eta=0.5, dt=1e-2)

        # MERA scaling mask: damp near-throat when margin low
        throat_layer = 0
        scaling = 1.0 if ane_margin > 0 else max(0.5, 1.0 + 0.3*ane_margin)  # shrink if negative
        layer_scale = {l: (scaling if l <= throat_layer+1 else 1.0) for l in range(layers)}

        return {
            "ane_margin": ane_margin,
            "conditioning": conditioning,
            "phase_indicator": phase_indicator,
            "torsion_force": F_t,
            "gamma_next": gamma_next,
            "m_eff": m_eff,
            "layer_scale": layer_scale,
        }
    return _compute

def actuate_fn_factory(rcc):
    def _actuate(action: Dict[str, Any]) -> None:
        # Apply MERA layer rescaling (wormhole-specific tensor op)
        layer_scale = action.get("layer_scale", {})
        if hasattr(rcc, "mera") and layer_scale:
            for l, s in layer_scale.items():
                try:
                    rcc.mera.tensors[l] *= float(s)
                except Exception:
                    pass

        # Update gamma state
        if "gamma_next" in action:
            rcc.gamma_state = float(action["gamma_next"])

        # Gentle gain nudge based on conditioning
        cond = float(action.get("conditioning", 1.0))
        if hasattr(rcc, "params"):
            b1, b2, b3 = rcc.params.get("beta_vec", (0.15, 0.05, 0.01))
            if cond > 300:
                rcc.params["beta_vec"] = (0.9*b1, 0.9*b2, 0.9*b3)
            elif cond < 50:
                rcc.params["beta_vec"] = (min(1.1*b1, 0.5), min(1.1*b2, 0.2), min(1.1*b3, 0.1))
    return _actuate