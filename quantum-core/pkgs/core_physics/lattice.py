"""Lattice structures and bond graph utilities."""

from dataclasses import dataclass
from typing import List

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
    """Build a 4D tesseract (hypercube) lattice with 16 vertices."""
    V, edges = 16, []
    for i in range(V):
        for bit in range(4):
            if i < (j := i ^ (1 << bit)):
                edges.append(Edge(i, j))
    return BondGraph(V=V, edges=edges, baseline_chi=baseline_chi)