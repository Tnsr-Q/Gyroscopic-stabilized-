"""
Lattice construction and bond graph operations for quantum computing.

Contains core lattice structures including BondGraph and tesseract/cube builders.
"""
from typing import List
from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    u: int
    v: int


@dataclass
class BondGraph:
    V: int
    edges: List[Edge]
    baseline_chi: int = 2


def build_tesseract_lattice(baseline_chi: int = 2) -> BondGraph:
    """Build a tesseract (4D hypercube) lattice structure."""
    V, edges = 16, []
    for i in range(V):
        for bit in range(4):
            if i < (j := i ^ (1 << bit)): 
                edges.append(Edge(i, j))
    return BondGraph(V=V, edges=edges, baseline_chi=baseline_chi)