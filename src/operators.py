"""
operators.py — Operatori di ipermutazione sul giant tour.
Argomento del corso: sistemi immunitari artificiali (selezione clonale).

Legge di ipermutazione inversamente proporzionale alla fitness
--------------------------------------------------------------
Dato il costo c_i dell'anticorpo i-esimo nella popolazione corrente:

    f_hat_i = (c_max - c_i) / (c_max - c_min)   in [0, 1]   (1 = migliore)
    alpha_i = e^(-rho * f_hat_i)
    M_i     = max(1, ceil(nc * alpha_i))        mutazioni elementari

Con rho = ln(nc) (default): l'anticorpo MIGLIORE riceve esattamente 1
mutazione (exploitation: piccoli aggiustamenti intorno alle soluzioni
buone), il PEGGIORE ne riceve nc (exploration: di fatto una forte
randomizzazione). E' il tratto distintivo del paradigma immunologico:
l'intensita' della mutazione e' modulata dalla qualita', al contrario
dei GA classici dove il tasso di mutazione e' uniforme.

Caso degenere: se tutti i costi coincidono (c_max == c_min) si pone
f_hat = 1 per tutti (mutazione minima); la diversita' viene reintrodotta
dall'aging e dalle nascite, non da mutazioni piu' violente.

Mutazioni elementari sulla permutazione
---------------------------------------
* inversione di segmento (peso 0.50) — inverte t[i..j]: e' la mossa piu'
  efficace sulle permutazioni "di percorso" perche' preserva
  l'adiacenza interna del segmento;
* swap (peso 0.25) — scambia due posizioni;
* insertion (peso 0.25) — sposta un cliente in un'altra posizione.

Tutte le mutazioni preservano l'invariante di permutazione (ogni cliente
esattamente una volta): la feasibility della soluzione decodificata e'
poi garantita per costruzione dallo Split.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_WEIGHTS = (0.50, 0.25, 0.25)      # inversione, swap, insertion


def normalized_fitness(costs: np.ndarray) -> np.ndarray:
    """Fitness normalizzata in [0, 1]: 1 = costo minimo (migliore),
    0 = costo massimo. Se tutti i costi coincidono, tutti a 1."""
    costs = np.asarray(costs, dtype=np.float64)
    cmin, cmax = costs.min(), costs.max()
    if cmax == cmin:
        return np.ones_like(costs)
    return (cmax - costs) / (cmax - cmin)


def mutations_count(f_hat: float, nc: int, rho: float) -> int:
    """Numero di mutazioni elementari M = max(1, ceil(nc * e^(-rho*f_hat)))."""
    return max(1, math.ceil(nc * math.exp(-rho * f_hat)))


def mutate(tour: np.ndarray, n_mut: int, rng,
           weights=DEFAULT_WEIGHTS) -> np.ndarray:
    """Applica n_mut mutazioni elementari a una COPIA del tour.

    Le tre mutazioni sono estratte con le probabilita' in `weights`.
    Nota implementativa: le assegnazioni su slice NumPy sovrapposte non
    sono sicure senza copia esplicita del lato destro (.copy())."""
    t = tour.copy()
    nc = len(t)
    kinds = rng.choice(3, size=n_mut, p=list(weights))
    for k in kinds:
        i, j = rng.choice(nc, size=2, replace=False)
        if k == 0:                               # inversione di segmento
            if i > j:
                i, j = j, i
            t[i:j + 1] = t[i:j + 1][::-1].copy()
        elif k == 1:                             # swap
            t[i], t[j] = t[j], t[i]
        else:                                    # insertion: sposta t[i] in j
            c = t[i]
            if i < j:
                t[i:j] = t[i + 1:j + 1].copy()   # shift a sinistra
            else:
                t[j + 1:i + 1] = t[j:i].copy()   # shift a destra
            t[j] = c
    return t
