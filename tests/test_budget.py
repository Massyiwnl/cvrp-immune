"""
test_budget.py — Test di accettazione della Fase 3 (contatore FE).

Il requisito centrale della consegna: gli esperimenti si arrestano al
numero massimo di valutazioni della funzione fitness FE = 3.5 x 10^5.
Qui si verifica che il contatore garantisca l'arresto ESATTO, che il
lavoro fuori budget venga rifiutato (non nascosto), che la riemersione
da cicli annidati funzioni e che il log di convergenza registri
correttamente i miglioramenti del best.
"""
import json

import numpy as np
import pytest

from src.budget import (FE_MAX_DEFAULT, Budget, BudgetExhausted,
                        ConvergenceLog, evaluate_tour)
from src.instance import load_instance
from src.split import split
from tests.common import DATA


# ---------------------------------------------------------------------------
# Arresto esatto al budget della consegna
# ---------------------------------------------------------------------------

def test_arresto_esatto_a_350000():
    assert FE_MAX_DEFAULT == 350_000
    b = Budget()                       # default: il valore della consegna
    while not b.exhausted:             # pattern canonico "controlla, valuta, registra"
        b.spend(1)
    assert b.used == 350_000
    assert b.remaining == 0
    assert b.exhausted


# ---------------------------------------------------------------------------
# Semantica di spend / can_spend
# ---------------------------------------------------------------------------

def test_spend_semantica():
    b = Budget(fe_max=10)
    for _ in range(9):
        assert b.spend(1) is True      # budget non ancora esaurito
    assert b.used == 9 and not b.exhausted
    assert b.spend(1) is False         # la decima raggiunge il tetto esatto
    assert b.used == 10 and b.exhausted
    with pytest.raises(BudgetExhausted):
        b.spend(1)                     # lavoro fuori budget: rifiutato...
    assert b.used == 10                # ...senza toccare il contatore


def test_batch_e_overshoot():
    b = Budget(fe_max=10)
    assert b.can_spend(10) and not b.can_spend(11)
    assert b.spend(7) is True and b.used == 7
    with pytest.raises(BudgetExhausted):
        b.spend(4)                     # 7 + 4 > 10: rifiutata senza registrare
    assert b.used == 7
    assert b.spend(3) is False and b.used == 10


# ---------------------------------------------------------------------------
# checkpoint(): riemersione da cicli annidati
# ---------------------------------------------------------------------------

def test_checkpoint_riemersione():
    """Simula la scansione annidata dei vicinati: una sola try/except al
    livello esterno intercetta l'esaurimento del budget."""
    b = Budget(fe_max=25)
    valutate = 0
    try:
        for _ in range(10):
            for _ in range(10):
                b.checkpoint()         # solleva quando il budget e' finito
                valutate += 1          # (qui avverrebbe la valutazione)
                b.spend(1)
    except BudgetExhausted:
        pass
    assert valutate == 25 == b.used    # esattamente il budget, non una in piu'


# ---------------------------------------------------------------------------
# Log di convergenza
# ---------------------------------------------------------------------------

def test_convergence_log():
    b = Budget(fe_max=100)
    log = ConvergenceLog(b)
    # sequenza (costo, generazione): 60 peggiora, 40 pareggia -> ignorati
    sequenza = [(50, 0), (60, 0), (40, 1), (40, 2), (30, 5)]
    migliorie = 0
    for cost, gen in sequenza:
        b.spend(1)
        if log.update(cost, generation=gen):
            migliorie += 1

    assert migliorie == 3
    assert log.best_cost == 30
    assert log.gen_of_best == 5
    fes = [fe for fe, _ in log.history]
    costs = [c for _, c in log.history]
    assert costs == [50, 40, 30]               # solo miglioramenti stretti
    assert fes == sorted(fes)                  # FE non decrescenti
    assert log.fe_of_best == fes[-1] == 5
    json.dumps(log.as_dict())                  # serializzabile per i risultati


# ---------------------------------------------------------------------------
# Integrazione con lo Split: 1 valutazione completa = 1 FE
# ---------------------------------------------------------------------------

def test_evaluate_tour_conta_una_fe():
    inst = load_instance(DATA / "A-n45-k7.vrp")
    b = Budget(fe_max=50)
    rng = np.random.default_rng(3)
    customers = np.arange(1, inst.n, dtype=np.int64)
    for i in range(1, 6):
        tour = rng.permutation(customers)
        res = evaluate_tour(tour, inst, b)
        assert b.used == i                      # esattamente 1 FE a chiamata
        assert res.cost == split(tour, inst).cost


def test_mini_run_integrazione():
    """Mini-run realistica: valuta permutazioni casuali finche' il budget
    non si esaurisce, tracciando il best — lo scheletro che il motore
    immunologico riempira' nella Fase 5."""
    inst = load_instance(DATA / "A-n45-k7.vrp")
    b = Budget(fe_max=500)
    log = ConvergenceLog(b)
    rng = np.random.default_rng(11)
    customers = np.arange(1, inst.n, dtype=np.int64)

    best_visto = None
    while not b.exhausted:
        res = evaluate_tour(rng.permutation(customers), inst, b)
        log.update(res.cost)
        best_visto = res.cost if best_visto is None else min(best_visto, res.cost)

    assert b.used == 500                        # arresto esatto
    assert log.best_cost == best_visto          # il log non perde nulla
    assert 0 < log.fe_of_best <= 500
    costs = [c for _, c in log.history]
    assert costs == sorted(costs, reverse=True)  # curva strettamente decrescente
