"""
Diagnostics package for stress testing and validation.

This package contains stress testing utilities to validate tensor network
implementations under extreme conditions and edge cases.
"""

from .stress_tests import StressTestSuite, run_comprehensive_stress_test

__all__ = ['StressTestSuite', 'run_comprehensive_stress_test']