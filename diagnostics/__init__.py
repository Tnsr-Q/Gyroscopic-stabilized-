"""
Diagnostic tools for quantum scarring, ETH testing, and Loschmidt echo measurements.

This module provides report-only diagnostic functions for analyzing quantum
systems without modifying the core quantum state evolution.
"""

from .scar_metrics import participation_ratio, cheap_otoc_rate, eth_deviation
from .echo_meter import loschmidt_echo, echo_report

__all__ = [
    'participation_ratio',
    'cheap_otoc_rate', 
    'eth_deviation',
    'loschmidt_echo',
    'echo_report'
]