"""
Diagnostics module for quantum contextuality analysis.

This module provides contextuality detection using the Kochen-Specker theorem 
and Lovász theta function, with graceful fallbacks for when advanced computations fail.
"""

from .contextuality import (
    kochen_specker_violation,
    lovasz_theta_bound,
    contextuality_graph_from_state,
    contextuality_certificate
)

__all__ = [
    'kochen_specker_violation',
    'lovasz_theta_bound', 
    'contextuality_graph_from_state',
    'contextuality_certificate'
]