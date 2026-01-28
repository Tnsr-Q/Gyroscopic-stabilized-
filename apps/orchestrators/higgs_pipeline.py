# orchestrators/higgs_pipeline.py
import numpy as np
from rge_service import run_rge
from veff_service import higgs_veff
from bounce_service import bounce_action
from inflation_service import higgs_inflation_observables
from ads_cft_compliance import holography_checkpack

def run_higgs_program(scenario: dict, mu_grid, h_grid, opts: dict, boundary_state, hypercube):
    running = run_rge(scenario, np.asarray(mu_grid))
    veff = higgs_veff(np.asarray(h_grid), running, opts)
    V_interp = lambda x: float(np.interp(x, veff["h"], veff["Veff"]))
    bounce = bounce_action(V_interp, include_gravity=False)
    infl = higgs_inflation_observables(running, N_star=opts.get("N_star", 50))
    holo = holography_checkpack(boundary_state, hypercube)
    return {"running": running, "veff": veff, "bounce": bounce, "infl": infl, "holo": holo}

# example_usage.py
import numpy as np
from orchestrators.higgs_pipeline import run_higgs_program

scenario = {
    "model": "SM",
    "loops": 1,
    "inputs": {"mt": 172.5, "mh": 125.10, "alpha_s": 0.1181, "xi_0": 0.1},
    "thresholds": []
}

mu_grid = np.logspace(2, 19, 1500)
h_grid  = np.logspace(2, 19, 1200)
opts = {"scheme":"MSbar","loops":1,"einstein_frame":True,"N_star":50}

# plumb real ones in your core:
boundary_state = np.random.randn(1024) + 1j*np.random.randn(1024)

class MiniCube:
    def __init__(self, edges=8): self._edges = [object()]*edges
    def minimal_cut_edges(self, _): return self._edges
    @property
    def edges(self): return self._edges

hypercube = MiniCube()

results = run_higgs_program(scenario, mu_grid, h_grid, opts, boundary_state, hypercube)
print("λ(μ) at top scale:", results["running"]["lambda"][-1])
print("meta-scale (λ=0) ≈", results["veff"]["meta_scale"])
print("Bounce S4:", results["bounce"]["S4"])
print("Inflation ns, r:", results["infl"]["ns"], results["infl"]["r"])
print("Holography checks:", results["holo"])
