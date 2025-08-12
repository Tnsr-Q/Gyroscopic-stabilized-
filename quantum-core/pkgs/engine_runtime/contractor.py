"""Precision Traversal Contractor for physics step orchestration."""

import logging
from typing import Dict, Optional
from .rcc import RecursiveConformalComputing

logger = logging.getLogger(__name__)

class PrecisionTraversalContractor:
    """Orchestrates physics steps using RecursiveConformalComputing."""
    
    def __init__(self, rcc: RecursiveConformalComputing):
        self.rcc = rcc
        self.rcc.mera.snapshot_flat_baseline()
        logger.info("Precision Traversal Contractor initialized")
        
    def step(self, dt: float, r: float, do_contract: bool = True) -> Dict:
        """Execute a single physics step."""
        # Update bond dimensions based on radial position
        self.rcc.update_bond_dimensions()
        
        # Calculate bond capacity
        Nb = max(8, min(4096, int(10*(1 + r/max(self.rcc.r0, 1e-6))*4)))
        K = self.rcc.capacity_throttle(Nb)
        
        # Select bonds and apply phase imprint
        bonds, phi_map = self.rcc.select_cut_bonds(K)
        self.rcc.mera.imprint_bond_phases(self.rcc.graph, bonds, phi_map)
        
        # Estimate spectral dimension from cut
        dH = self.rcc.estimate_spectral_dim(bonds)
        
        # Compute RG parameters and Planck scale factor
        c_rel = self.rcc.rg_planner.c_hat(r, dH_est=dH)
        s_Pl = self.rcc.rg_planner.planck_scale_factor(r, c_rel)
        self.rcc._last_s_Pl = s_Pl  # Store for consistent access
        
        if dH:
            logger.debug(f"Spectral dimension estimate: dH={dH:.2f}, s_Pl={s_Pl:.4f}")
        
        # Calculate ANE with Planck scale factor and check QEI guard
        sigma_u = self.rcc.params.get('sigma_u', 4e-3)
        ane = self.rcc.smeared_ANE(dt, sigma_u, s_Pl)
        guard_ok = self.rcc.qei_guard(ane, sigma_u)
        
        # Apply emergency damping if guard fails
        if not guard_ok:
            import numpy as np
            beta_vec = np.array(self.rcc.params.get("beta_vec", (0.15, 0.05, 0.01)))
            self.rcc.params["beta_vec"] = tuple(0.9 * beta_vec)
            logger.warning(f"QEI guard violated! ANE={ane:.2e} - Applying emergency damping")
            
        # Run contractor if allowed
        if do_contract and guard_ok:
            target = -0.25 * abs(ane) - 1e-4
            ane_new = self._contractor_step(dt, target, s_Pl)
        else:
            ane_new = ane
            
        # Enforce tensor drift guard (with dtype preservation)
        self.rcc.mera.enforce_tensor_drift_guard(max_drift=1e-3)
            
        # Log results
        if self.rcc.recorder:
            delta_energy = self.rcc.mera.delta_tensor_energy()
            Tuu_total = self.rcc.compute_Tuu_eff(dt, s_Pl)["Tuu_total"]
            self.rcc.recorder.log({
                "ANE_smear": ane_new,
                "K": K,
                "Δtensor": delta_energy,
                "Tuu_total": Tuu_total,
                "spectral_dim": dH if dH else 0.0,
                "s_Pl": s_Pl,
                "c_rel": c_rel
            })
            
        return {
            "K": K,
            "ane_smear": ane_new,
            "guard_ok": guard_ok,
            "spectral_dim": dH,
            "s_Pl": s_Pl
        }

    def _contractor_step(self, dt: float, target: float, s_Pl: float, 
                        step: float = 0.25, max_ls: int = 5) -> float:
        """Internal contractor step with line search."""
        sigma_u = self.rcc.params.get('sigma_u', 4e-3)
        base_ane = self.rcc.smeared_ANE(dt, sigma_u, s_Pl)
        grads = self._finite_diff_grads(dt)
        current_thetas = {k: self._get_theta(k) for k in grads}
        trial = {}
        obj_sign = 1.0 if base_ane > target else -1.0
        
        # Initial trial step
        for k, g in grads.items():
            new_val = current_thetas[k] - step * obj_sign * g
            trial[k] = self._project_theta(k, new_val)
            
        # Line search
        for _ in range(max_ls):
            self._apply_theta(trial)
            ane_new = self.rcc.smeared_ANE(dt, sigma_u, s_Pl)
            
            improvement_condition = (
                (target < base_ane and ane_new < base_ane) or 
                (target > base_ane and ane_new > base_ane)
            )
            
            if improvement_condition and self.rcc.qei_guard(ane_new, sigma_u):
                return ane_new
                
            step *= 0.5
            for k, g in grads.items():
                new_val = current_thetas[k] - step * obj_sign * g
                trial[k] = self._project_theta(k, new_val)
        
        # Restore original parameters if no improvement
        self._apply_theta(current_thetas)
        return base_ane

    def _finite_diff_grads(self, dt: float, delta_scale: float = 1e-3) -> Dict[str, float]:
        """Central-diff grads d(ANE)/dθ for θ in {g, λφ, β1, β2, β3}."""
        theta_keys = ["g", "λφ", "β1", "β2", "β3"]
        base_sigma = self.rcc.params.get('sigma_u', 4e-3)
        base_ane = self.rcc.smeared_ANE(dt, base_sigma, self.rcc._last_s_Pl)
        grads: Dict[str, float] = {}

        # Snapshot current parameters
        g0 = self.rcc.params.get("double_trace_gain", 0.0)
        lphi = self.rcc.loom.lambda_phi
        b1, b2, b3 = self.rcc.params.get("beta_vec", (0.15, 0.05, 0.01))
        param_backup = {"g": g0, "λφ": lphi, "β1": b1, "β2": b2, "β3": b3}

        for k in theta_keys:
            v0 = param_backup[k]
            h = max(1e-6, abs(v0)*delta_scale)

            # θ+ step
            self._apply_theta({k: self._project_theta(k, v0 + h)})
            ane_plus = self.rcc.smeared_ANE(dt, base_sigma, self.rcc._last_s_Pl)

            # θ- step
            self._apply_theta({k: self._project_theta(k, v0 - h)})
            ane_minus = self.rcc.smeared_ANE(dt, base_sigma, self.rcc._last_s_Pl)

            # Restore original value
            self._apply_theta({k: v0})

            # Central difference gradient
            grads[k] = (ane_plus - ane_minus) / (2.0 * h)

        # Restore all parameters to original state
        self._apply_theta(param_backup)
        return grads

    def _get_theta(self, key: str) -> float:
        """Get parameter value by key."""
        key = self.rcc._THETA_ALIASES.get(key, key)
        if key == "g":
            return self.rcc.params.get("double_trace_gain", 0.0)
        if key == "λφ":
            return self.rcc.loom.lambda_phi
        beta_vec = self.rcc.params.get("beta_vec", (0.15, 0.05, 0.01))
        return list(beta_vec)[{"β1": 0, "β2": 1, "β3": 2}[key]]

    def _project_theta(self, key: str, val: float) -> float:
        """Project parameter to valid range."""
        import numpy as np
        key = self.rcc._THETA_ALIASES.get(key, key)
        if key == "g":
            return float(np.clip(val, 0.0, self.rcc.params.get("g_max", 0.05)))
        if key == "λφ":
            return float(np.clip(val, 0.0, 2.0))
        return float(np.clip(val, -0.5, 0.5))

    def _apply_theta(self, trial: Dict):
        """Apply parameter changes."""
        beta_vec = list(self.rcc.params.get("beta_vec", (0.15, 0.05, 0.01)))
        
        for key, val in trial.items():
            key = self.rcc._THETA_ALIASES.get(key, key)
            if key == "g":
                self.rcc.params["double_trace_gain"] = val
            elif key == "λφ":
                self.rcc.loom.lambda_phi = val
            elif key == "β1":
                beta_vec[0] = val
            elif key == "β2":
                beta_vec[1] = val
            elif key == "β3":
                beta_vec[2] = val
                
        self.rcc.params["beta_vec"] = tuple(beta_vec)