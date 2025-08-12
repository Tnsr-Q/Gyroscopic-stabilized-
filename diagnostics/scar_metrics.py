"""
Quantum scarring and ETH diagnostic metrics.

Functions for detecting quantum scars and ETH violations.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple


def participation_ratio(psi: torch.Tensor) -> float:
    """
    Calculate participation ratio PR = 1/sum |psi_i|^4 (basis-dependent; use clock basis).
    
    Args:
        psi: Quantum state tensor
        
    Returns:
        Participation ratio as a float
    """
    v = psi.detach().cpu().numpy().ravel()
    p2 = np.sum(np.abs(v)**2)
    p4 = np.sum(np.abs(v)**4) + 1e-12
    return float((p2**2)/p4)


def cheap_otoc_rate(time_op, psi: torch.Tensor, 
                    steps: int = 8, dt: float = 0.01) -> float:
    """
    Crude OTOC growth proxy using diagonal clock-H and a diagonal 'V'.
    C(t) ~ -<[W(t),V]^2>; we approximate slope of log C(t).
    
    Args:
        time_op: Time operator with H_int attribute
        psi: Quantum state tensor
        steps: Number of time steps for calculation
        dt: Time step size
        
    Returns:
        OTOC growth rate as a float
    """
    H = time_op.H_int.detach().cpu().numpy().diagonal().astype(np.float64)
    d = H.shape[0]
    V = np.diag(np.linspace(-1.0, 1.0, d))  # simple diagonal
    psi0 = psi.detach().cpu().numpy()
    
    def U(t): 
        return np.diag(np.exp(-1j*H*t))
    
    Cvals, ts = [], []
    for k in range(1, steps+1):
        t = k*dt
        Wt = U(t) @ V @ np.conj(U(t)).T
        comm = Wt@V - V@Wt
        C = np.vdot(psi0, (comm.conj().T @ comm) @ psi0).real
        Cvals.append(max(C, 1e-16))
        ts.append(t)
    
    ts = np.array(ts)
    Cvals = np.array(Cvals)
    slope = np.polyfit(ts[1:], np.log(Cvals[1:]), 1)[0]
    return float(slope)


def eth_deviation(energies: np.ndarray, obs_vals: np.ndarray, 
                  window: float = 0.05) -> float:
    """
    ETH variance in a microcanonical window centered at median energy.
    
    Args:
        energies: Array of eigenstate energies
        obs_vals: Array of observable expectation values
        window: Energy window as fraction of total energy range
        
    Returns:
        ETH deviation (variance) as a float
    """
    E = energies.reshape(-1)
    O = obs_vals.reshape(-1)
    Em = np.median(E)
    sel = np.abs(E - Em) < window*np.ptp(E)
    if sel.sum() < 3:
        return 0.0
    return float(np.var(O[sel]))