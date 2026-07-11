"""
test_instance.py — Test di accettazione della Fase 1 (parser + matrice).

Test A (il piu' importante del progetto): per ognuna delle 10 istanze,
il costo della soluzione ottima ufficiale (.sol di CVRPLIB) ricalcolato
con la NOSTRA matrice delle distanze deve coincidere ESATTAMENTE con il
costo dichiarato nel file. Se questo test e' verde, ogni costo prodotto
dal nostro algoritmo sara' direttamente confrontabile con le best-known
solutions (gap % attendibili).

Test B (sanity): ceil(domanda_totale / Q) <= k per ogni istanza.

In piu': test unitari sull'arrotondamento nint e su una mini-istanza
sintetica con distanze verificabili a mano.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from src.instance import (load_instance, load_solution,
                          sol_to_internal_candidates, _nint)

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


# ---------------------------------------------------------------------------
# Convenzione di arrotondamento
# ---------------------------------------------------------------------------

def test_nint_convention():
    # round() di Python usa il banker's rounding: round(2.5) == 2.
    # La convenzione TSPLIB richiede half-up: nint(2.5) == 3.
    assert _nint(2.5) == 3
    assert _nint(3.5) == 4
    assert _nint(2.49) == 2
    assert round(2.5) == 2   # ecco perche' NON si usa round()


# ---------------------------------------------------------------------------
# Mini-istanza sintetica: distanze verificabili a mano
# ---------------------------------------------------------------------------

MINI = """NAME : mini-n4-k2
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 3 4
3 0 5
4 6 8
DEMAND_SECTION
1 0
2 5
3 5
4 5
DEPOT_SECTION
1
-1
EOF
"""


def test_mini_distanze(tmp_path):
    p = tmp_path / "mini-n4-k2.vrp"
    p.write_text(MINI)
    inst = load_instance(p)
    assert inst.n == 4 and inst.k == 2 and inst.capacity == 10
    # (0,0)-(3,4) = 5 esatto ; (0,0)-(6,8) = 10 esatto
    assert inst.D[0, 1] == 5
    assert inst.D[0, 3] == 10
    # (3,4)-(0,5) = sqrt(10) = 3.162... -> 3
    assert inst.D[1, 2] == 3
    # (0,5)-(6,8) = sqrt(45) = 6.708... -> 7
    assert inst.D[2, 3] == 7
    # costo rotta [1,2]: depot->1 (5) + 1->2 (3) + 2->depot (5) = 13
    assert inst.route_cost([1, 2]) == 13
    ok, msg = inst.is_feasible([[1, 2], [3]])
    assert ok, msg
    # capacita' violata se tutti e tre i clienti in una rotta (15 > 10)
    ok, _ = inst.is_feasible([[1, 2, 3]])
    assert not ok


# ---------------------------------------------------------------------------
# Parsing delle 10 istanze del progetto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", INSTANCES)
def test_parsing_base(name):
    inst = load_instance(DATA / f"{name}.vrp")
    # dimensione e k coerenti con il nome del file
    assert inst.n == int(re.search(r"-n(\d+)-", name).group(1))
    assert inst.k == int(re.search(r"-k(\d+)", name).group(1))
    # depot e domande
    assert inst.demands[0] == 0
    assert (inst.demands[1:] >= 1).all()
    # proprieta' della matrice delle distanze
    assert inst.D.shape == (inst.n, inst.n)
    assert inst.D.dtype == np.int64
    assert (np.diag(inst.D) == 0).all()
    assert (inst.D == inst.D.T).all()
    assert (inst.D >= 0).all()


# ---------------------------------------------------------------------------
# Test B — sanity su domande e capacita'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", INSTANCES)
def test_sanity_capacita(name):
    inst = load_instance(DATA / f"{name}.vrp")
    assert inst.min_vehicles <= inst.k, (
        f"{name}: servono almeno {inst.min_vehicles} veicoli ma k={inst.k}")


# ---------------------------------------------------------------------------
# Test A — riproduzione ESATTA del costo delle soluzioni ottime ufficiali
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", INSTANCES)
def test_A_riproduzione_costo_sol(name):
    inst = load_instance(DATA / f"{name}.vrp")
    routes_raw, declared = load_solution(DATA / f"{name}.sol")

    # 1) il costo dichiarato nel .sol deve essere l'ottimo noto di CVRPLIB
    assert declared == EXPECTED_BKS[name], (
        f"{name}: il .sol dichiara {declared}, atteso {EXPECTED_BKS[name]} "
        f"(file scaricato da fonte non ufficiale?)")

    # 2) almeno una interpretazione degli indici deve dare una soluzione
    #    ammissibile il cui costo, ricalcolato con la NOSTRA matrice,
    #    coincide esattamente con quello dichiarato
    match = None
    for label, cand in sol_to_internal_candidates(routes_raw, inst.n):
        ok, _ = inst.is_feasible(cand, max_routes=inst.k)
        if ok and inst.solution_cost(cand) == declared:
            match = label
            break
    assert match is not None, (
        f"{name}: nessuna interpretazione degli indici del .sol riproduce "
        f"il costo dichiarato {declared} — matrice delle distanze o "
        f"parsing da rivedere")
