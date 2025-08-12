"""Recursive Conformal Computing orchestration class."""

import numpy as np
import torch
from typing import Dict, Optional, Set, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

import sys
import os
# Add the quantum-core directory to Python path for absolute imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import core physics components
from pkgs.core_physics import (
    TimeOperator, TimeRecompressionGate, ProperTimeGaugeField,
    MERASpacetime, build_tesseract_lattice, UVIRRegulator, AlenaSoul,
    GyroscopeFeeler, SignatureNormalizer, JacobianHypercube,
    estimate_spectral_dim_from_cut, Edge, BondGraph
)

logger = logging.getLogger(__name__)

class DecoherenceState(Enum):
    COHERENT = "coherent"
    DECOHERING = "decohering"
    DECOHERENT = "decoherent"
    RECOMPRESSING = "recompressing"
    STABILIZED = "stabilized"

@dataclass
class QuantumState:
    coherence_gamma: float
    proper_time: float
    torsion: float
    law_field: np.ndarray
    decoherence_state: DecoherenceState
    entropy: float
    phase_memory: float = 0.0

class RGPlanner:
    """Renormalization Group Planner for boundary-driven scaling and Planck factor."""
    
    def __init__(self, mu0: float = 1.0, L_rg: float = 1.0, d_boundary: int = 2, eps: float = 1e-6):
        self.mu0 = mu0
        self.L_rg = L_rg
        self.d = d_boundary
        self.eps = eps
        self.c_ref = None  # set on first call
        self.prev_c_rel = 1.0  # for monotonicity enforcement

    def mu_of_r(self, r: float) -> float:
        """Renormalization scale as function of radius."""
        return self.mu0 * np.exp(-r / max(self.L_rg, self.eps))

    def c_hat(self, r: float, dH_est: Optional[float], S_slope: Optional[float] = None) -> float:
        """Relative speed parameter based on entanglement or spectral dimension."""
        # Simple proxy: prefer entanglement slope if available; else d_H
        if S_slope is not None:
            c_raw = max(S_slope, self.eps)
        elif dH_est is not None:
            c_raw = max(dH_est, self.eps)
        else:
            c_raw = 1.0
            
        # Set reference on first call
        if self.c_ref is None:
            self.c_ref = c_raw
            
        # Compute relative speed and enforce monotonicity
        c_rel = c_raw / max(self.c_ref, self.eps)
        c_rel = min(c_rel, self.prev_c_rel)  # Enforce non-increasing
        self.prev_c_rel = c_rel  # Update for next call
        return c_rel

    def planck_scale_factor(self, r: float, c_rel: float) -> float:
        """Compute Planck scale factor: M_Pl / M_Pl0 = (c_rel)^{1/(d-1)}."""
        exponent = 1.0 / max(1, self.d - 1)
        return float((c_rel + self.eps) ** exponent)

class RecursiveConformalComputing:
    """Main orchestration class for recursive conformal computing."""
    
    _THETA_ALIASES = {"lambda_phi": "λφ", "beta1": "β1", "beta2": "β2", "beta3": "β3"}

    def __init__(self, r0: float = 0.5, m0: float = 1.0, params: Optional[Dict] = None):
        self.r0 = r0
        self.m0 = m0
        self.params = params if params else {}
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize core physics components
        self.time_op = TimeOperator()
        self.recompression_gate = TimeRecompressionGate(self.time_op)
        self.gauge_field = ProperTimeGaugeField(device=device)
        self.mera = MERASpacetime()
        self.graph = build_tesseract_lattice()
        self.loom = UVIRRegulator(
            d_H=self.params.get('d_H', 3.12), 
            C=self.params.get('C_RT', 91.64)
        )
        self.soul = AlenaSoul(kappa=self.params.get('kappa', 0.015))
        self.feeler = GyroscopeFeeler(self.graph)
        self.hypercube = JacobianHypercube().to(device)
        self.sig_norm = SignatureNormalizer()
        
        # Initialize quantum state
        self.state = QuantumState(
            coherence_gamma=1.0,
            proper_time=0.0,
            torsion=0.0,
            law_field=np.ones((32, 32)),
            decoherence_state=DecoherenceState.COHERENT,
            entropy=0.0
        )
        
        # Initialize clock state
        psi = torch.randn(self.time_op.dim_clock, dtype=torch.complex64, device=device)
        self.psi_clock = psi / torch.norm(psi)
        
        # Initialize state tracking
        self._dH_estimate = None  # Spectral dimension cache
        self.rg_planner = RGPlanner(
            mu0=self.params.get('mu0', 1.0),
            L_rg=self.params.get('L_rg', 1.0),
            d_boundary=self.params.get('d_boundary', 2)
        )
        self._last_s_Pl = 1.0  # Last Planck scale factor
        self._c_rel_history = []  # Track c_rel for stability
        self._dH_history = []     # Track spectral dimension

    @property
    def recorder(self) -> Optional["SimpleRecorder"]:
        """Get recorder from params if available."""
        return self.params.get("recorder")

    def layer_radius(self, layer: int) -> float:
        """Map layer index to radial position in throat geometry."""
        L = self.mera.layers
        return self.r0 * (layer + 0.5) / max(1, L)

    def target_bond_dim(self, r: float, Dmin: int = 2, Dmax: int = 32, alpha: float = 3.0, 
                       R: Optional[float] = None, eps: float = 1e-6) -> int:
        """Compute target bond dimension for a given radial position."""
        if R is None:
            R = 1.0 * self.r0
        num = np.log(1.0 + alpha * r / max(self.r0, eps))
        den = np.log(1.0 + alpha * R / max(self.r0, eps)) + eps
        frac = np.clip(num/den, 0.0, 1.0)
        # More capacity near the throat (r small)
        val = Dmin + (Dmax - Dmin) * (1.0 - frac)
        return int(np.clip(round(val), Dmin, Dmax))

    def update_bond_dimensions(self):
        """Update target bond dimensions based on radial position."""
        new_dims = []
        for l in range(self.mera.layers):
            r = self.layer_radius(l)
            new_dim = self.target_bond_dim(r)
            
            # Apply cap: max 2x change per step
            current_dim = self.mera.target_dims[l]
            max_change = min(2 * current_dim, self.params.get('max_bond_dim', 32))
            new_dim = min(new_dim, max_change)
            
            new_dims.append(new_dim)
        
        self.mera.target_dims = new_dims
        logger.info(f"Updated bond dimensions: {self.mera.target_dims}")

    def capacity_throttle(self, N: int) -> int:
        """Capacity throttle with log-compression for MERA bond dims."""
        # Compute geometric mean of bond dimensions in minimal-cut layers
        cut_layers = [
            max(0, self.mera.layers//2 - 1), 
            self.mera.layers//2,
            min(self.mera.layers-1, self.mera.layers//2 + 1)
        ]
        dims = [self.mera.target_dims[l] for l in cut_layers]
        log_dim = np.log(np.mean(dims) + 1e-6)
        
        # Adjust capacity with log-compression
        Nb = max(4, int(N))
        Cc = float(self.params.get("cap_C", 1.0))
        b = float(self.params.get("cap_b", -0.3))
        K = int(max(1, np.floor(Cc * (np.log(Nb * log_dim + 1e-6)**b))))
        
        cap_min = self.params.get("cap_min", 1)
        cap_max = self.params.get("cap_max", len(self.mera.minimal_cut_edges(self.graph)))
        return int(np.clip(K, cap_min, cap_max))

    def select_cut_bonds(self, K: int) -> Tuple[List[Edge], Dict[Edge, float]]:
        """Select bonds to cut and their phase mappings."""
        cut = self.mera.minimal_cut_edges(self.graph)
        ups = self.soul.upsilon_from_A(self.gauge_field.A_mu)
        phi_map = self.soul.phase_imprint(self.graph, ups, self.loom.lambda_phi)
        
        # Sort by absolute phase value and edge ordering
        cut_sorted = sorted(
            cut, key=lambda e: (-abs(phi_map.get(e, 0.0)), (min(e.u, e.v), max(e.u, e.v)))
        )
        return cut_sorted[:max(1, K)], phi_map

    def _h_profile(self) -> float:
        """Compute h profile from law field gradients."""
        gx, gy = np.gradient(self.state.law_field)
        chi = float(np.sqrt((gx**2 + gy**2).mean()))
        Tn = float(abs(self.state.torsion))
        phase_div = 0.0
        b1, b2, b3 = self.params.get("beta_vec", (0.15, 0.05, 0.01))
        g1 = self.params.get("gamma1", 0.2)
        return 1.0 + b1*np.tanh(chi-1.0) + b2*(Tn*Tn)/(1.0+g1*chi) + b3*phase_div

    def _Tuu_from_h(self, dt: float) -> float:
        """Compute stress-energy component from h profile."""
        h_now = float(self._h_profile())
        # Stabilize logs near tails
        h_clamped = np.clip(h_now, 1e-12, 1e12)
        ln_now = np.log1p(h_clamped - 1)  # More stable near h=1
        
        if not hasattr(self, "_ema"):
            self._ema = {"ln": ln_now, "d1": 0.0, "d2": 0.0}
            
        ema = self._ema
        if dt <= 0:
            return 0.0
            
        tau = 5.0 * self.params.get('sigma_u', 4e-3)
        alpha = dt/(tau+dt)
        d1 = (1-alpha)*ema["d1"] + alpha*((ln_now - ema["ln"])/dt)
        ema["d2"] = (1-alpha)*ema["d2"] + alpha*((d1 - ema["d1"])/dt)
        ema["ln"], ema["d1"] = ln_now, d1
        
        a1, a2 = self.params.get('hbar_weights', (0.0, -1.0))
        return float(a1*ema["d2"] + a2*(d1*d1))

    def compute_Tuu_eff(self, dt: float, s_Pl: float = 1.0) -> Dict[str, float]:
        """Compute effective stress-energy with Planck scale factor."""
        T_hbar = self._Tuu_from_h(dt)
        # Apply Planck scale factor: geometric prefactor scales as 1/M_Pl^2
        T_hbar *= (1.0 / max(s_Pl, 1e-6))**2

        T_ups = float(self.soul.upsilon_from_A(self.gauge_field.A_mu)[0,1].abs().mean().item())
        T_ups *= self.params.get("upsilon_to_Tuu", 1.0)

        T_sqz = -abs(self.params.get("squeeze_amp", 0.0)) * (float(abs(self.state.torsion))**2)
        return {
            "Tuu_hbar": T_hbar,
            "Tuu_Upsilon": T_ups,
            "Tuu_sqz": T_sqz,
            "Tuu_total": T_hbar + T_ups + T_sqz
        }
    
    def smeared_ANE(self, dt: float, sigma_u: float, s_Pl: float = 1.0) -> float:
        """Smeared ANE with Planck scale threading."""
        if not hasattr(self, "_ane_hist"):
            self._ane_hist, self._t_clock = [], 0.0
            
        self._t_clock += dt
        Tuu_total = self.compute_Tuu_eff(dt, s_Pl)["Tuu_total"]
        self._ane_hist.append((self._t_clock, Tuu_total))
        
        # Keep only recent history within 6 sigma
        max_age = 6.0 * sigma_u
        self._ane_hist = [(t, v) for (t, v) in self._ane_hist if (self._t_clock - t) <= max_age]
        
        # Gaussian-weighted average
        num = 0.0
        den = 0.0
        sigma_u_safe = max(1e-6, sigma_u)
        for t, v in self._ane_hist:
            tau = self._t_clock - t
            weight = np.exp(-0.5 * (tau / sigma_u_safe)**2)
            num += v * weight
            den += weight
            
        return float(num / max(den, 1e-12))

    def qei_guard(self, ane: float, sigma_u: float, margin: float = 0.05) -> bool:
        """Quantum Energy Inequality guard with dynamic RHS."""
        # Compute RHS based on current envelope f(t)
        t_vals = [t for t, _ in self._ane_hist]
        if len(t_vals) < 2:
            return True  # Not enough data
        
        # Compute min and max times
        t_min, t_max = min(t_vals), max(t_vals)
        duration = t_max - t_min
        
        # Compute dynamic constant based on envelope
        qei_const = 1.0 / max(duration, 1e-6)
        sigma_u_safe = max(1e-6, sigma_u)
        threshold = (-qei_const / sigma_u_safe**4) * (1.0 - margin)
        return ane >= threshold

    def estimate_spectral_dim(self, cut_edges: List[Edge]) -> Optional[float]:
        """Estimate spectral dimension from interface graph."""
        self._dH_estimate = estimate_spectral_dim_from_cut(self.graph, cut_edges)
        if self.recorder and self._dH_estimate is not None:
            self.recorder.log({"spectral_dim": self._dH_estimate})
            self._dH_history.append(self._dH_estimate)
        return self._dH_estimate

    def zero_field_baseline(self):
        """Zero EM potentials and Υ; snapshot MERA baseline."""
        self.gauge_field.zero_fields()
        self.mera.snapshot_flat_baseline()
        logger.info("Zeroed gauge fields and set MERA baseline")

    def flat_baseline_check(self, dt: float, steps: int = 128) -> float:
        """Run with zero fields; ANE should hover near ~0 and QEI-safe."""
        sigma_u = self.params.get('sigma_u', 4e-3)
        for _ in range(steps):
            self.smeared_ANE(dt, sigma_u, self._last_s_Pl)  # integrates history at zero drive
        final_ane = self.smeared_ANE(dt, sigma_u, self._last_s_Pl)
        logger.info(f"Flat baseline check: ANE = {final_ane:.2e}")
        return final_ane