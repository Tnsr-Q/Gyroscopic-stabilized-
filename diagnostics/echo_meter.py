"""
Loschmidt echo measurements for quantum chaos diagnostics.

Functions for measuring sensitivity to perturbations (quantum chaos/stability).
"""
import numpy as np
import torch
from typing import Dict


@torch.no_grad()
def loschmidt_echo(time_op, psi: torch.Tensor, 
                   t: float = 0.05, eps: float = 1e-2) -> float:
    """
    Calculate Loschmidt echo L(t) = |<psi| U_0†(t) U_eps(t) |psi>|^2 
    with small H-perturbation.
    
    Args:
        time_op: Time operator with H_int attribute
        psi: Quantum state tensor
        t: Evolution time
        eps: Perturbation strength
        
    Returns:
        Loschmidt echo value as a float
    """
    H = time_op.H_int
    U0 = torch.matrix_exp(-1j*H*t)
    He = (1.0 + eps)*H
    Ue = torch.matrix_exp(-1j*He*t)
    amp = (psi.conj() @ U0.conj().T @ Ue @ psi).abs().item()
    return float(amp*amp)


def echo_report(time_op, psi, t_list=(0.02, 0.05, 0.1)) -> Dict[str, float]:
    """
    Generate a report of Loschmidt echo measurements at multiple times.
    
    Args:
        time_op: Time operator with H_int attribute
        psi: Quantum state tensor
        t_list: Tuple of time values for echo measurements
        
    Returns:
        Dictionary with echo measurements and decay rate
    """
    out = {}
    for t in t_list:
        out[f"echo_t{t}"] = loschmidt_echo(time_op, psi, t=t)
    
    # crude "rate": 1 - L(0.1)  
    out["echo_rate"] = 1.0 - out[f"echo_t{t_list[-1]}"]
    return out