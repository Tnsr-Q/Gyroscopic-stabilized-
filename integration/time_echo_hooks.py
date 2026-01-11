# integration/time_echo_hooks.py
from __future__ import annotations
import torch
import warnings
from typing import Dict, Any
from timeops.three_time_ops import ThreeTimeClock
from protocols.time_echo import TimeRecompressionProtocol

def time_echo_sense(rcc, clock: ThreeTimeClock):
    # Initialize reference state once for consistent coherence tracking
    psi_ref = torch.randn(clock.d**3, dtype=torch.complex128, device=clock.device)
    psi_ref = psi_ref / torch.linalg.norm(psi_ref)
    
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
        # Use cached reference state for visibility tracking across echo cycles
        rt["psi0"] = psi_ref
        return rt
    return _sense

def time_echo_compute(rcc, clock: ThreeTimeClock, path2_scales=(1.0, 1.1, 0.9)):
    """
    Create echo compute function with configurable path scaling.
    
    Args:
        rcc: RecursiveConformalComputing instance
        clock: ThreeTimeClock instance
        path2_scales: Tuple of scaling factors for second path (default: (1.0, 1.1, 0.9))
                     These introduce differential evolution to probe phase differences.
    """
    proto = TimeRecompressionProtocol(clock)
    def _compute(obs: Dict[str,Any]) -> Dict[str,Any]:
        psi0 = obs["psi0"]
        # Map geometric rates into λ's along 3 time axes:
        # e.g., proper-time accumulations per axis (use your A_mu or Jacobian signatures)
        lam1 = float(obs.get("control_sensitivity", 0.1))*1e-2
        lam2 = float(obs.get("flow_stability", 0.1))*1e-2
        lam3 = float(obs.get("entanglement_rate", 0.1))*1e-2
        # Apply configurable scaling to create path difference for phase tomography
        res = proto.run_echo(psi0, (lam1,lam2,lam3), 
                           (lam1*path2_scales[0], lam2*path2_scales[1], lam3*path2_scales[2]))
        # If visibility improved, emit Δτ for an echo gate to apply across the runtime
        return {"delta_tau_vec": (res["d_tau_1"], res["d_tau_2"], res["d_tau_3"]),
                "vis_before": res["vis_before"], "vis_after": res["vis_after"]}
    return _compute

def time_echo_actuate(rcc, clock: ThreeTimeClock):
    def _actuate(action: Dict[str,Any]) -> None:
        Δ = action.get("delta_tau_vec", None)
        if Δ is None:
            warnings.warn("Echo actuation skipped: delta_tau_vec not provided in action", 
                        RuntimeWarning, stacklevel=2)
            return
        # Apply echo by imprinting phases on the "internal clock" channel.
        # Here we route it to your phase-imprint / bond-phase kernel as a small correction.
        try:
            rcc.phase_imprint_from_clock_echo(Δ)
        except AttributeError:
            # fallback: store for contractor; many of your modules already read rcc.params
            rcc.params["time_echo_delta"] = Δ
    return _actuate