"""Costanti e helper condivisi dai test del progetto."""
from pathlib import Path

import numpy as np

from src.instance import (Instance, load_instance, load_solution,
                          sol_to_internal_candidates)

DATA = Path(__file__).resolve().parents[1] / "data"

INSTANCES = [
    "A-n45-k7", "A-n60-k9", "A-n80-k10",
    "B-n56-k7", "B-n66-k9", "B-n78-k10",
    "E-n76-k8", "E-n101-k14",
    "P-n50-k10", "P-n101-k4",
]

# Ottimi certificati — fonte: sito ufficiale CVRPLIB (file .sol scaricati).
EXPECTED_BKS = {
    "A-n45-k7": 1146, "A-n60-k9": 1354, "A-n80-k10": 1763,
    "B-n56-k7": 707,  "B-n66-k9": 1316, "B-n78-k10": 1221,
    "E-n76-k8": 735,  "E-n101-k14": 1067,
    "P-n50-k10": 696, "P-n101-k4": 681,
}


def make_instance(coords, demands, capacity, k=None, name="toy") -> Instance:
    """Costruisce direttamente un'istanza (per test sintetici), con la
    stessa matrice EUC_2D/nint del parser."""
    coords = np.asarray(coords, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.int64)
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    D = np.floor(np.sqrt(dx * dx + dy * dy) + 0.5).astype(np.int64)
    return Instance(name=name, n=len(demands), capacity=int(capacity), k=k,
                    coords=coords, demands=demands, D=D)


def make_toy_instance(rng, n_customers, capacity) -> Instance:
    """Istanza giocattolo casuale (domande in 1..Q/2 cosi' ogni cliente
    sta sempre in un veicolo da solo)."""
    n = n_customers + 1
    coords = rng.integers(0, 100, size=(n, 2)).astype(np.float64)
    demands = np.zeros(n, dtype=np.int64)
    demands[1:] = rng.integers(1, capacity // 2 + 1, size=n_customers)
    return make_instance(coords, demands, capacity, name=f"toy-n{n}")


def load_bks_internal(name):
    """Ritorna (istanza, rotte ottime in indici interni, costo ottimo),
    identificando empiricamente la convenzione degli indici del .sol
    (stessa logica del Test A della Fase 1)."""
    inst = load_instance(DATA / f"{name}.vrp")
    routes_raw, declared = load_solution(DATA / f"{name}.sol")
    for _label, cand in sol_to_internal_candidates(routes_raw, inst.n):
        ok, _ = inst.is_feasible(cand, max_routes=inst.k)
        if ok and inst.solution_cost(cand) == declared:
            return inst, cand, declared
    raise AssertionError(f"{name}: convenzione degli indici .sol non identificata")
