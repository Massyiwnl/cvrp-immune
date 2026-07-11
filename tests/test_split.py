"""
test_split.py — Test di accettazione della Fase 2 (decoder Split).

1) Brute-force: su istanze giocattolo casuali (nc <= 8) si enumerano
   TUTTE le 2^(nc-1) partizioni contigue del tour; lo Split (illimitato
   e limitato, per ogni valore di max_routes) deve restituire esattamente
   il minimo ammissibile.

2) Feasibility di massa: centinaia di permutazioni casuali delle istanze
   reali; ogni decodifica deve essere ammissibile, con costo identico al
   ricalcolo indipendente, e il round-trip rotte -> tour deve restituire
   il tour originale.

3) Proprieta' matematica garantita: sul giant tour ottenuto concatenando
   le rotte della soluzione OTTIMA, split_limited(., k) restituisce
   esattamente il costo ottimo. Dimostrazione: la partizione BKS e' una
   delle partizioni contigue con <= k rotte, quindi costo_DP <= BKS; ogni
   partizione con <= k rotte e' una soluzione ammissibile del CVRP,
   quindi costo_DP >= BKS. Ne segue costo_DP == BKS.

4) Coerenza fra varianti e casi degeneri (infeasibility rilevata).
"""
import numpy as np
import pytest

from src.instance import load_instance
from src.split import split, split_limited, tour_from_routes
from tests.common import (DATA, INSTANCES, load_bks_internal,
                          make_instance, make_toy_instance)


# ---------------------------------------------------------------------------
# 1) Confronto con enumerazione esaustiva
# ---------------------------------------------------------------------------

def brute_force_split(tour, inst, max_routes=None):
    """Enumera tutte le partizioni contigue del tour tramite maschere di
    taglio (bit t = taglio dopo la posizione t) e restituisce il costo
    minimo ammissibile, o None se non esiste."""
    nc = len(tour)
    best = None
    for mask in range(1 << (nc - 1)):
        routes, start = [], 0
        for t in range(nc - 1):
            if (mask >> t) & 1:
                routes.append(list(tour[start:t + 1]))
                start = t + 1
        routes.append(list(tour[start:nc]))
        if max_routes is not None and len(routes) > max_routes:
            continue
        if any(inst.route_load(r) > inst.capacity for r in routes):
            continue
        c = inst.solution_cost(routes)
        if best is None or c < best:
            best = c
    return best


def test_brute_force_illimitato_e_limitato():
    rng = np.random.default_rng(42)
    for _trial in range(20):
        nc = int(rng.integers(4, 9))                    # 4..8 clienti
        inst = make_toy_instance(rng, nc, capacity=20)
        tour = rng.permutation(np.arange(1, nc + 1)).astype(np.int64)

        # --- variante illimitata --------------------------------------
        res = split(tour, inst)
        bf = brute_force_split(tour, inst)
        assert res.feasible and bf is not None
        assert res.cost == bf
        ok, msg = inst.is_feasible(res.routes)
        assert ok, msg
        assert inst.solution_cost(res.routes) == res.cost

        # --- variante limitata, per ogni max_routes da 1 a nc ----------
        for m in range(1, nc + 1):
            resm = split_limited(tour, inst, max_routes=m)
            bfm = brute_force_split(tour, inst, max_routes=m)
            if bfm is None:
                assert not resm.feasible
            else:
                assert resm.feasible and resm.cost == bfm
                assert resm.n_routes <= m
                ok, msg = inst.is_feasible(resm.routes, max_routes=m)
                assert ok, msg


# ---------------------------------------------------------------------------
# 2) Feasibility di massa su permutazioni casuali delle istanze reali
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["A-n45-k7", "E-n101-k14"])
def test_permutazioni_casuali(name):
    inst = load_instance(DATA / f"{name}.vrp")
    rng = np.random.default_rng(0)
    customers = np.arange(1, inst.n, dtype=np.int64)
    for _ in range(300):
        tour = rng.permutation(customers)
        res = split(tour, inst)
        assert res.feasible
        ok, msg = inst.is_feasible(res.routes)
        assert ok, msg
        # il costo della DP coincide con il ricalcolo indipendente
        assert inst.solution_cost(res.routes) == res.cost
        # round-trip: concatenare le rotte restituisce il tour originale
        assert np.array_equal(tour_from_routes(res.routes), tour)


# ---------------------------------------------------------------------------
# 3) Proprieta' garantita sul giant tour della soluzione ottima
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", INSTANCES)
def test_split_sul_tour_ottimo(name):
    inst, bks_routes, bks_cost = load_bks_internal(name)
    tour = tour_from_routes(bks_routes)

    # split limitato a k rotte: uguaglianza matematica con l'ottimo
    resk = split_limited(tour, inst, max_routes=inst.k)
    assert resk.feasible
    assert resk.cost == bks_cost, (
        f"{name}: split_limited sul tour ottimo da' {resk.cost}, "
        f"atteso {bks_cost}")
    assert resk.n_routes <= inst.k
    ok, msg = inst.is_feasible(resk.routes, max_routes=inst.k)
    assert ok, msg

    # split illimitato: e' un rilassamento, quindi costo <= ottimo
    res = split(tour, inst)
    assert res.feasible and res.cost <= bks_cost


# ---------------------------------------------------------------------------
# 4) Coerenza fra varianti e casi degeneri
# ---------------------------------------------------------------------------

def test_coerenza_limitato_illimitato():
    """Con max_routes = nc (nessun vincolo effettivo) la variante limitata
    deve coincidere con quella illimitata."""
    inst = load_instance(DATA / "A-n45-k7.vrp")
    rng = np.random.default_rng(7)
    customers = np.arange(1, inst.n, dtype=np.int64)
    for _ in range(30):
        tour = rng.permutation(customers)
        assert split(tour, inst).cost == \
            split_limited(tour, inst, max_routes=inst.n_customers).cost


def test_limitato_infeasible_rilevata():
    """Tre clienti con domanda 6 e Q = 10: nessuna coppia sta in un
    veicolo, quindi servono almeno 3 rotte qualunque sia l'ordine."""
    inst = make_instance(coords=[[0, 0], [10, 0], [0, 10], [10, 10]],
                         demands=[0, 6, 6, 6], capacity=10)
    tour = np.array([1, 2, 3], dtype=np.int64)

    assert not split_limited(tour, inst, max_routes=2).feasible

    r3 = split_limited(tour, inst, max_routes=3)
    atteso = sum(2 * int(inst.D[0, c]) for c in (1, 2, 3))  # andata+ritorno
    assert r3.feasible and r3.n_routes == 3 and r3.cost == atteso
    # anche l'illimitato deve usare 3 rotte singole allo stesso costo
    assert split(tour, inst).cost == atteso


def test_domanda_superiore_a_Q():
    """Caso degenere: una domanda supera Q -> nessuna partizione esiste."""
    inst = make_instance(coords=[[0, 0], [5, 5]], demands=[0, 15], capacity=10)
    res = split(np.array([1], dtype=np.int64), inst)
    assert not res.feasible
