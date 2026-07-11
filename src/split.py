"""
split.py — Decoder "Split": partizione ottima di un giant tour in rotte
CVRP tramite programmazione dinamica.

Idea (argomento del corso: metodi esatti / dynamic programming)
---------------------------------------------------------------
Una soluzione e' rappresentata come "giant tour": una permutazione
S = (s_1, ..., s_nc) di tutti i clienti, SENZA separatori di rotta.
Si costruisce un grafo ausiliario aciclico con nodi 0..nc: l'arco
(i, j), con i < j, rappresenta "un veicolo serve i clienti
s_{i+1}, ..., s_j in quest'ordine" ed esiste solo se il carico del
segmento non supera Q. Il suo costo e':

    cost(i,j) = D[0, s_{i+1}] + sum_{t=i+1}^{j-1} D[s_t, s_{t+1}] + D[s_j, 0]

Il cammino di costo minimo da 0 a nc individua la partizione in rotte
di costo minimo per QUELLA permutazione. Il grafo e' un DAG gia'
ordinato topologicamente (0, 1, ..., nc), quindi il cammino minimo si
calcola con la ricorrenza di Bellman:

    V[0] = 0
    V[j] = min_{i < j : load(i+1..j) <= Q}  V[i] + cost(i,j)

Complessita': O(nc * span), dove span e' il massimo numero di clienti
consecutivi che stanno in un veicolo (il break sulla capacita' tronca
il ciclo interno); cost(i,j) e' aggiornato in O(1) estendendo il
segmento di un cliente alla volta.

Due varianti pubbliche:

* split(tour, inst)
    Flotta illimitata: partizione a costo minimo senza vincolo sul
    numero di rotte. E' la valutazione veloce usata come fitness.

* split_limited(tour, inst, max_routes)
    Al piu' max_routes rotte, tramite DP bidimensionale
    V[r][j] = min_i V[r-1][i] + cost(i,j),  O(max_routes * nc * span).
    Serve a garantire la conformita' alla definizione formale del
    problema (m veicoli disponibili): sulle 10 istanze del progetto la
    capacita' e' "stretta" (ceil(domanda_tot / Q) = k), quindi il
    controllo sul numero di rotte non e' opzionale.

Proprieta' chiave (verificate nei test):

* Feasibility per costruzione: ogni cliente compare esattamente una
  volta e ogni rotta rispetta Q.
* split() e' totale se ogni singola domanda e' <= Q (ogni cliente puo'
  sempre costituire una rotta da solo); altrimenti feasible = False.
* Sul giant tour ottenuto concatenando le rotte di una soluzione
  OTTIMA, split_limited(., k) restituisce esattamente il costo ottimo:
  la partizione BKS e' una delle partizioni contigue con <= k rotte
  (quindi costo_DP <= BKS) e ogni partizione con <= k rotte e' una
  soluzione ammissibile del CVRP (quindi costo_DP >= BKS).

Nota implementativa: i nuclei _split_dp e _split_dp_limited sono
scritti con cicli espliciti su array NumPy (nessuna vettorizzazione)
perche' sono i candidati designati alla compilazione con Numba nel
checkpoint prestazioni delle fasi successive: la versione a cicli e'
gia' nella forma che Numba compila in modo ottimale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance

# Valore "infinito" sicuro per int64: si evita comunque di sommarvi
# alcunche' (gli stati irraggiungibili vengono saltati).
INF = np.int64(2) ** 62


# ---------------------------------------------------------------------------
# Nuclei di programmazione dinamica (numba-ready: soli array e cicli)
# ---------------------------------------------------------------------------

def _split_dp(tour, demands, D, capacity):
    """Bellman sul grafo ausiliario, flotta illimitata.

    Ritorna (V, pred):
      V[j]    = costo minimo per servire i primi j clienti del tour;
      pred[j] = indice i di inizio (esclusivo) dell'ultima rotta nel
                cammino ottimo che raggiunge j.
    """
    nc = len(tour)
    V = np.full(nc + 1, INF, dtype=np.int64)
    pred = np.full(nc + 1, -1, dtype=np.int64)
    V[0] = 0
    for i in range(nc):
        if V[i] >= INF:                      # stato irraggiungibile
            continue
        load = 0
        cost = 0
        for j in range(i + 1, nc + 1):
            c = tour[j - 1]
            load += demands[c]
            if load > capacity:              # pruning: inutile estendere
                break
            if j == i + 1:
                cost = D[0, c] + D[c, 0]     # rotta con un solo cliente
            else:
                p = tour[j - 2]              # cliente precedente del segmento
                # estensione O(1): sostituisce l'arco p->depot con p->c->depot
                cost += D[p, c] + D[c, 0] - D[p, 0]
            if V[i] + cost < V[j]:
                V[j] = V[i] + cost
                pred[j] = i
    return V, pred


def _split_dp_limited(tour, demands, D, capacity, rmax):
    """Bellman a livelli: V[r][j] = costo minimo per servire i primi j
    clienti con ESATTAMENTE r rotte. Stessa logica di _split_dp, con la
    dimensione aggiuntiva del numero di rotte."""
    nc = len(tour)
    V = np.full((rmax + 1, nc + 1), INF, dtype=np.int64)
    pred = np.full((rmax + 1, nc + 1), -1, dtype=np.int64)
    V[0, 0] = 0
    for r in range(1, rmax + 1):
        for i in range(nc):
            if V[r - 1, i] >= INF:
                continue
            load = 0
            cost = 0
            for j in range(i + 1, nc + 1):
                c = tour[j - 1]
                load += demands[c]
                if load > capacity:
                    break
                if j == i + 1:
                    cost = D[0, c] + D[c, 0]
                else:
                    p = tour[j - 2]
                    cost += D[p, c] + D[c, 0] - D[p, 0]
                if V[r - 1, i] + cost < V[r, j]:
                    V[r, j] = V[r - 1, i] + cost
                    pred[r, j] = i
    return V, pred


# ---------------------------------------------------------------------------
# Estrazione delle rotte dai predecessori
# ---------------------------------------------------------------------------

def _extract_routes(tour, pred):
    """Risale i predecessori da j = nc fino a 0 e ricostruisce le rotte
    (in ordine, come segmenti contigui del tour)."""
    routes = []
    j = len(tour)
    while j > 0:
        i = int(pred[j])
        routes.append([int(c) for c in tour[i:j]])
        j = i
    routes.reverse()
    return routes


def _extract_routes_limited(tour, pred, r_best):
    routes = []
    j = len(tour)
    r = r_best
    while j > 0:
        i = int(pred[r, j])
        routes.append([int(c) for c in tour[i:j]])
        j = i
        r -= 1
    routes.reverse()
    return routes


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    cost: int          # costo totale della partizione ottima (INF se infeasible)
    routes: list       # lista di rotte (liste di clienti interni), [] se infeasible
    n_routes: int      # numero di rotte usate
    feasible: bool     # False se nessuna partizione ammissibile esiste


def split(tour, inst: Instance) -> SplitResult:
    """Partizione ottima del giant tour, flotta illimitata.

    feasible = False solo nel caso degenere in cui una domanda supera Q
    (nessuna rotta puo' contenere quel cliente)."""
    tour = np.asarray(tour, dtype=np.int64)
    V, pred = _split_dp(tour, inst.demands, inst.D, inst.capacity)
    nc = len(tour)
    if V[nc] >= INF:
        return SplitResult(int(INF), [], 0, False)
    routes = _extract_routes(tour, pred)
    return SplitResult(int(V[nc]), routes, len(routes), True)


def split_limited(tour, inst: Instance, max_routes: int | None = None) -> SplitResult:
    """Partizione ottima del giant tour usando AL PIU' max_routes rotte
    (default: inst.k, il numero di veicoli m della definizione formale).

    feasible = False se il tour, in QUEST'ORDINE, non e' partizionabile
    in <= max_routes segmenti contigui che rispettano Q — evento possibile
    con capacita' stretta, anche quando una diversa permutazione degli
    stessi clienti lo sarebbe. A parita' di costo minimo viene preferito
    il numero di rotte piu' basso."""
    if max_routes is None:
        max_routes = inst.k
    tour = np.asarray(tour, dtype=np.int64)
    V, pred = _split_dp_limited(tour, inst.demands, inst.D, inst.capacity,
                                int(max_routes))
    nc = len(tour)
    col = V[:, nc]
    r_best = int(np.argmin(col))          # primo minimo -> meno rotte a parita'
    if col[r_best] >= INF:
        return SplitResult(int(INF), [], 0, False)
    routes = _extract_routes_limited(tour, pred, r_best)
    return SplitResult(int(col[r_best]), routes, len(routes), True)


def tour_from_routes(routes) -> np.ndarray:
    """Concatena le rotte in un giant tour: operazione inversa dello Split.

    Poiche' lo Split estrae segmenti contigui nell'ordine del tour,
    tour_from_routes(split(t).routes) restituisce esattamente t.
    Sara' usata per il writeback lamarckiano dopo la ricerca locale."""
    flat = [c for r in routes for c in r]
    return np.array(flat, dtype=np.int64)
