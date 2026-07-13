"""
test_immune.py — Test di accettazione della Fase 5 (motore immunologico).

Copre: (1) invarianti degli operatori di ipermutazione (la permutazione
resta una permutazione; la legge esponenziale ha i valori estremi e la
monotonia attesi); (2) riproducibilita' bit a bit a parita' di seed;
(3) smoke run su A-n45-k7 con soglia di qualita' anti-regressione e
arresto ESATTO al budget; (4) il caso "a rischio" B-n66-k9: il best
riportato deve rispettare k rotte (vincolo m della definizione formale).
"""
import numpy as np
import pytest

from src.immune import ImmuneAlgorithm, ImmuneParams
from src.instance import load_instance
from src.operators import mutate, mutations_count, normalized_fitness
from tests.common import DATA, EXPECTED_BKS


# ---------------------------------------------------------------------------
# Operatori di ipermutazione
# ---------------------------------------------------------------------------

def test_mutate_preserva_permutazione():
    rng = np.random.default_rng(0)
    nc = 60
    base = np.arange(1, nc + 1, dtype=np.int64)
    atteso = list(range(1, nc + 1))
    t = rng.permutation(base)
    for _ in range(500):
        m = int(rng.integers(1, nc + 1))
        t = mutate(t, m, rng)
        assert sorted(int(c) for c in t) == atteso     # sempre una permutazione


def test_mutate_non_modifica_input():
    rng = np.random.default_rng(1)
    t = rng.permutation(np.arange(1, 31, dtype=np.int64))
    before = t.copy()
    _ = mutate(t, 10, rng)
    assert np.array_equal(t, before)                   # lavora su una copia


def test_mutations_count_estremi_e_monotonia():
    import math
    nc = 100
    rho = math.log(nc)
    assert mutations_count(1.0, nc, rho) == 1          # il migliore: 1 mutazione
    assert mutations_count(0.0, nc, rho) == nc         # il peggiore: nc mutazioni
    vals = [mutations_count(f, nc, rho) for f in np.linspace(0, 1, 21)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))  # non crescente in f_hat


def test_normalized_fitness():
    f = normalized_fitness(np.array([100, 150, 200]))
    assert f[0] == 1.0 and f[2] == 0.0 and 0.0 < f[1] < 1.0
    assert np.all(normalized_fitness(np.array([42, 42, 42])) == 1.0)  # degenere


# ---------------------------------------------------------------------------
# Riproducibilita': stesso seed -> stessa run, seed diversi -> run diverse
# ---------------------------------------------------------------------------

def _clean(res):
    return {k: v for k, v in res.items() if k != "wall_time"}


def test_riproducibilita():
    # Nota: il budget deve essere abbastanza grande da far divergere le
    # traiettorie stocastiche; con budget minuscoli la history puo'
    # contenere solo la prima valutazione (il Clarke-Wright puro), che e'
    # deterministica e quindi identica per ogni seed.
    inst = load_instance(DATA / "A-n45-k7.vrp")
    r1 = ImmuneAlgorithm(inst, seed=7, fe_max=25_000).run()
    r2 = ImmuneAlgorithm(inst, seed=7, fe_max=25_000).run()
    assert _clean(r1) == _clean(r2)

    r3 = ImmuneAlgorithm(inst, seed=8, fe_max=25_000).run()
    assert r3["history"] != r1["history"]


# ---------------------------------------------------------------------------
# Smoke run: qualita' anti-regressione + arresto esatto + coerenza output
# ---------------------------------------------------------------------------

def test_smoke_A45():
    inst = load_instance(DATA / "A-n45-k7.vrp")
    bks = EXPECTED_BKS["A-n45-k7"]
    res = ImmuneAlgorithm(inst, seed=0, fe_max=50_000).run()

    # arresto ESATTO al budget imposto
    assert res["fe_used"] == 50_000
    # qualita' minima anti-regressione (attesa molto migliore)
    gap = 100.0 * (res["best_cost"] - bks) / bks
    assert gap < 10.0, f"gap {gap:.2f}%"
    # vincolo sul numero di veicoli
    assert res["respects_k"] and res["best_n_routes"] <= inst.k
    # coerenza: costo dichiarato == ricalcolo, soluzione ammissibile
    ok, msg = inst.is_feasible(res["best_routes"], max_routes=inst.k)
    assert ok, msg
    assert inst.solution_cost(res["best_routes"]) == res["best_cost"]
    # curva di convergenza: costi strettamente decrescenti, FE non decrescenti
    costs = [c for _, c in res["history"]]
    fes = [fe for fe, _ in res["history"]]
    assert costs == sorted(costs, reverse=True) and len(set(costs)) == len(costs)
    assert fes == sorted(fes)
    assert res["history"][-1][1] == res["best_cost"]
    assert res["gen_of_best"] is not None and res["fe_of_best"] == fes[-1]


def test_smoke_B66_rispetta_k():
    """B-n66-k9 e' l'istanza dove CW+LS da soli restavano a 10 rotte (>k):
    il motore con decodifica Split deve riportare un best con <= 9 rotte."""
    inst = load_instance(DATA / "B-n66-k9.vrp")
    res = ImmuneAlgorithm(inst, seed=0, fe_max=50_000).run()
    assert res["respects_k"] and res["best_n_routes"] <= inst.k
    ok, msg = inst.is_feasible(res["best_routes"], max_routes=inst.k)
    assert ok, msg
    gap = 100.0 * (res["best_cost"] - EXPECTED_BKS["B-n66-k9"]) / EXPECTED_BKS["B-n66-k9"]
    assert gap < 10.0, f"gap {gap:.2f}%"
