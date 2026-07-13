"""
test_local_search.py — Test di accettazione della Fase 4.

Il test centrale (Delta == ricalcolo) applica centinaia di mosse CASUALI
di ogni tipo — anche peggiorative — e dopo OGNUNA verifica che il costo
mantenuto per differenze coincida esattamente con il ricalcolo da zero,
che la soluzione resti una partizione valida dei clienti e che i carichi
mantenuti coincidano con quelli ricomputati. E' il test che stana la
quasi totalita' dei bug nelle formule Delta e negli aggiornamenti di
stato.

Seguono: monotonia/idempotenza della discesa, rispetto del cap di FE per
invocazione e del budget globale (arresto esatto), feasibility di
Clarke-Wright su tutte le istanze e qualita' della discesa a partire da
Clarke-Wright (gap dal BKS).
"""
import numpy as np
import pytest

import src.local_search as ls
from src.budget import Budget
from src.greedy import clarke_wright
from src.instance import load_instance
from src.local_search import LocalSearch
from src.split import split
from tests.common import DATA, EXPECTED_BKS, INSTANCES


# ---------------------------------------------------------------------------
# Generatori di mosse casuali VALIDE (per il test Delta == ricalcolo)
# ---------------------------------------------------------------------------

def _random_two_opt(state, rng):
    cands = [ri for ri, r in enumerate(state.routes) if len(r) >= 2]
    if not cands:
        return None
    ri = int(rng.choice(cands))
    m = len(state.routes[ri])
    i = int(rng.integers(0, m - 1))
    j = int(rng.integers(i + 1, m))
    return ("two_opt", ri, i, j)


def _random_chain(state, rng):
    for _ in range(60):
        rA = int(rng.integers(0, len(state.routes)))
        A = state.routes[rA]
        L = int(rng.integers(1, 4))
        if L > len(A):
            continue
        i = int(rng.integers(0, len(A) - L + 1))
        rB = int(rng.integers(0, len(state.routes)))
        if rA == rB and L == len(A):
            continue
        if rA != rB and not ls._chain_ok(state, rA, i, L, rB):
            continue
        lenR = (len(A) - L) if rA == rB else len(state.routes[rB])
        gap = int(rng.integers(0, lenR + 1))
        rev = bool(rng.integers(0, 2))
        if rA == rB and gap == i and not rev:
            continue                                   # no-op
        return ("chain", rA, i, L, rB, gap, rev)
    return None


def _random_swap(state, rng):
    if len(state.routes) < 2:
        return None
    n = state.inst.n
    for _ in range(60):
        u = int(rng.integers(1, n))
        v = int(rng.integers(1, n))
        if u == v or int(state.route_of[u]) == int(state.route_of[v]):
            continue
        if not ls._swap_ok(state, u, v):
            continue
        return ("swap", u, v)
    return None


def _random_star(state, rng):
    if len(state.routes) < 2:
        return None
    for _ in range(60):
        rA, rB = rng.choice(len(state.routes), size=2, replace=False)
        rA, rB = int(rA), int(rB)
        lenA, lenB = len(state.routes[rA]), len(state.routes[rB])
        i = int(rng.integers(-1, lenA))
        j = int(rng.integers(-1, lenB))
        if (i == -1 and j == -1) or (i == lenA - 1 and j == lenB - 1):
            continue
        if not ls._star_ok(state, rA, i, rB, j):
            continue
        return ("star", rA, i, rB, j)
    return None


def _apply(state, mv):
    kind = mv[0]
    if kind == "two_opt":
        return ls._apply_two_opt(state, *mv[1:])
    if kind == "chain":
        return ls._apply_chain(state, *mv[1:])
    if kind == "swap":
        return ls._apply_swap(state, *mv[1:])
    if kind == "star":
        return ls._apply_two_opt_star(state, *mv[1:])
    raise ValueError(kind)


def _check_invariants(state, inst):
    # costo per differenze == ricalcolo da zero (uguaglianza intera esatta)
    assert state.cost == inst.solution_cost(state.routes)
    # partizione valida dei clienti + capacita' rispettate
    ok, msg = inst.is_feasible(state.routes)
    assert ok, msg
    # carichi mantenuti == carichi ricomputati
    for ri, r in enumerate(state.routes):
        assert state.loads[ri] == inst.route_load(r)


# ---------------------------------------------------------------------------
# Delta == ricalcolo, su centinaia di mosse casuali di ogni tipo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["A-n45-k7", "B-n66-k9"])
def test_delta_uguale_ricalcolo(name):
    inst = load_instance(DATA / f"{name}.vrp")
    rng = np.random.default_rng(123)
    tour = rng.permutation(np.arange(1, inst.n, dtype=np.int64))
    res = split(tour, inst)
    state = ls._State(inst, res.routes, res.cost)
    _check_invariants(state, inst)

    generators = [_random_two_opt, _random_chain, _random_swap, _random_star]
    applied = {g.__name__: 0 for g in generators}
    total = 0
    for step in range(4000):
        gen = generators[step % 4]
        mv = gen(state, rng)
        if mv is None:
            continue
        _apply(state, mv)
        applied[gen.__name__] += 1
        total += 1
        _check_invariants(state, inst)          # dopo OGNI mossa
        if total >= 600:
            break

    assert total >= 500                          # il test ha davvero lavorato
    for g, cnt in applied.items():
        assert cnt >= 50, f"{g}: esercitato solo {cnt} volte"


# ---------------------------------------------------------------------------
# Discesa: monotonia, correttezza, idempotenza
# ---------------------------------------------------------------------------

def test_discesa_monotona_e_idempotenza():
    inst = load_instance(DATA / "P-n50-k10.vrp")
    rng = np.random.default_rng(9)
    budget = Budget(fe_max=300_000)
    searcher = LocalSearch(inst, budget, rng=rng)

    for _ in range(3):
        tour = rng.permutation(np.arange(1, inst.n, dtype=np.int64))
        res = split(tour, inst)
        routes, cost = searcher.improve(res.routes, res.cost)
        assert cost <= res.cost                          # mai peggio dell'ingresso
        assert cost == inst.solution_cost(routes)        # costo coerente
        ok, msg = inst.is_feasible(routes)
        assert ok, msg
        # seconda invocazione sulla stessa soluzione: mai un peggioramento
        routes2, cost2 = searcher.improve(routes, cost)
        assert cost2 <= cost


# ---------------------------------------------------------------------------
# Contabilita' FE: cap per invocazione e budget globale esatto
# ---------------------------------------------------------------------------

def test_cap_fe_per_invocazione():
    inst = load_instance(DATA / "B-n56-k7.vrp")
    budget = Budget(fe_max=100_000)
    searcher = LocalSearch(inst, budget, rng=np.random.default_rng(2))
    tour = np.random.default_rng(5).permutation(
        np.arange(1, inst.n, dtype=np.int64))
    res = split(tour, inst)

    start = budget.used
    routes, cost = searcher.improve(res.routes, res.cost, max_fe=200)
    spesa = budget.used - start
    assert 0 < spesa <= 200
    assert cost <= res.cost
    assert cost == inst.solution_cost(routes)


def test_budget_globale_esatto():
    """Da una soluzione casuale la discesa vorrebbe molte piu' di 150
    valutazioni: deve fermarsi ESATTAMENTE al tetto, senza eccezioni."""
    inst = load_instance(DATA / "B-n56-k7.vrp")
    budget = Budget(fe_max=150)
    searcher = LocalSearch(inst, budget, rng=np.random.default_rng(4))
    tour = np.random.default_rng(6).permutation(
        np.arange(1, inst.n, dtype=np.int64))
    res = split(tour, inst)

    routes, cost = searcher.improve(res.routes, res.cost)
    assert budget.used == 150
    assert budget.exhausted
    assert cost <= res.cost
    ok, msg = inst.is_feasible(routes)
    assert ok, msg


# ---------------------------------------------------------------------------
# Clarke-Wright: feasibility su tutte le istanze
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", INSTANCES)
def test_clarke_wright_feasible(name):
    inst = load_instance(DATA / f"{name}.vrp")
    routes = clarke_wright(inst)
    ok, msg = inst.is_feasible(routes)
    assert ok, msg
    # il greedy deve battere largamente una soluzione casuale
    rng = np.random.default_rng(0)
    tour = rng.permutation(np.arange(1, inst.n, dtype=np.int64))
    assert inst.solution_cost(routes) < split(tour, inst).cost


# ---------------------------------------------------------------------------
# Qualita': Clarke-Wright + discesa completa vicino al BKS
# ---------------------------------------------------------------------------

def test_qualita_da_clarke_wright():
    inst = load_instance(DATA / "A-n45-k7.vrp")
    bks = EXPECTED_BKS["A-n45-k7"]
    cw = clarke_wright(inst)
    cw_cost = inst.solution_cost(cw)

    budget = Budget(fe_max=200_000)
    searcher = LocalSearch(inst, budget, rng=np.random.default_rng(1))
    routes, cost = searcher.improve(cw, cw_cost)

    assert cost <= cw_cost
    assert cost == inst.solution_cost(routes)
    gap = 100.0 * (cost - bks) / bks
    assert gap < 5.0, f"gap {gap:.2f}% dal BKS ({cost} vs {bks})"
