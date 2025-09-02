"""Contextuality and quantum advantage certificates."""

try:
    from .contextuality import KSCert, ks_cert
    __all__ = ['KSCert', 'ks_cert']
except ImportError:
    # Graceful degradation if dependencies missing
    __all__ = []