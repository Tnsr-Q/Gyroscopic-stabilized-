# integration/time_echo_hooks.py
from __future__ import annotations
import torch
from typing import Dict, Any, Tuple
from timeops.three_time_ops import ThreeTimeClock
from protocols.time_echo import TimeRecompressionProtocol

def time_echo_sense(rcc, clock: ThreeTimeClock):
    def _sense() -> Dict[str,Any]:
        # Pull your existing metrics plus a seed clock state
        try:
            rt = rcc.rt_metrics.compute_realtime_metrics()
        except AttributeError:
            # Fallback to basic metrics from rcc state
            rt = {
                "control_sensitivity": getattr(rcc.state, 'coherence_gamma', 0.1),
                "flow_stability": 0.1,  # default
                "entanglement_rate": 0.1  # default
            }
        # normalized random clock state (can be cached or learned)
        psi = torch.randn(clock.d, dtype=torch.complex128)
        psi = psi / torch.linalg.norm(psi)
        rt["psi0"] = psi
        return rt
    return _sense

def time_echo_compute(rcc, clock: ThreeTimeClock):
    proto = TimeRecompressionProtocol(clock)
    def _compute(obs: Dict[str,Any]) -> Dict[str,Any]:
        psi0 = obs["psi0"]
        # Map geometric rates into λ's along 3 time axes:
        # e.g., proper-time accumulations per axis (use your A_mu or Jacobian signatures)
        lam1 = float(obs.get("control_sensitivity", 0.1))*1e-2
        lam2 = float(obs.get("flow_stability", 0.1))*1e-2
        lam3 = float(obs.get("entanglement_rate", 0.1))*1e-2
        res = proto.run_echo(psi0, (lam1,lam2,lam3), (lam1*1.0, lam2*1.1, lam3*0.9))
        # If visibility improved, emit Δτ for an echo gate to apply across the runtime
        return {"delta_tau_vec": (res["d_tau_1"], res["d_tau_2"], res["d_tau_3"]),
                "vis_before": res["vis_before"], "vis_after": res["vis_after"]}
    return _compute

def time_echo_actuate(rcc, clock: ThreeTimeClock):
    def _actuate(action: Dict[str,Any]) -> None:
        Δ = action.get("delta_tau_vec", None)
        if Δ is None: return
        # Apply echo by imprinting phases on the "internal clock" channel.
        # Here we route it to your phase-imprint / bond-phase kernel as a small correction.
        try:
            rcc.phase_imprint_from_clock_echo(Δ)
        except AttributeError:
            # fallback: store for contractor; many of your modules already read rcc.params
            rcc.params["time_echo_delta"] = Δ
    return _actuate