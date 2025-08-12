"""Pydantic schemas for service APIs."""

from pydantic import BaseModel
from typing import Dict, Optional, Any

class InitRequest(BaseModel):
    """Request schema for engine initialization."""
    seed: int = 0
    grid_size: int = 32
    device: str = "auto"
    params: Dict[str, float] = {}

class StepRequest(BaseModel):
    """Request schema for physics step."""
    dt: float
    r: float
    do_contract: bool = True

class StepResult(BaseModel):
    """Result schema from physics step."""
    K: int
    ane_smear: float
    guard_ok: bool
    spectral_dim: Optional[float] = None
    metrics: Dict[str, Any] = {}  # Changed from Dict[str, float] to allow mixed types