"""
local_search.py — Ricerca locale a vicinati multipli su rotte CVRP.
Argomento del corso: meta-euristiche a singola soluzione / local search.

Schema
------
Discesa first-improvement su una pipeline di cinque vicinati, ripetuta
finche' un giro completo non produce alcun miglioramento (ottimo locale
rispetto all'unione dei vicinati esplorati) oppure finche' il budget di
FE — globale o il cap per invocazione — si esaurisce.

Vicinati e formule Delta (distanze simmetriche; p/s = predecessore e
successore nella rotta, con il depot 0 agli estremi):

1. 2-opt intra-rotta — inversione del segmento r[i..j]:
       Delta = D[a, r[j]] + D[r[i], b] - D[a, r[i]] - D[r[j], b]
   con a = nodo prima di i, b = nodo dopo j (gli archi interni al
   segmento non cambiano costo per simmetria).

2/3. Or-opt / relocate (catena di L in {1,2,3}, stessa rotta o rotta
   diversa; relocate = caso L=1 inter-rotta) — rimozione della catena
   c1..cL tra p ed s, inserimento tra x e y (eventualmente invertita):
       Delta = (D[x,c1'] + D[cL',y] - D[x,y]) - (D[p,c1] + D[cL,s] - D[p,s])
   dove (c1',cL') = (c1,cL) o (cL,c1) se invertita.

4. swap inter-rotta — scambio di u (rotta A) e v (rotta B):
       Delta = D[pu,v] + D[v,su] - D[pu,u] - D[u,su]
             + D[pv,u] + D[u,sv] - D[pv,v] - D[v,sv]

5. 2-opt* — scambio delle code di due rotte dopo le posizioni i e j
   (i, j = -1 significa taglio subito dopo il depot; il caso con una
   coda vuota realizza la FUSIONE di due rotte):
       Delta = D[a_i, b_{j+1}] + D[b_j, a_{i+1}] - D[a_i, a_{i+1}] - D[b_j, b_{j+1}]
   Ammissibilita' via somme prefisse dei carichi:
       pref_A[i+1] + (load_B - pref_B[j+1]) <= Q   (e simmetrico).

Contabilita' delle FE (convenzione del progetto, vedi budget.py)
----------------------------------------------------------------
OGNI Delta calcolato e' la valutazione del costo di una soluzione
candidata e vale 1 FE (budget.spend(1)), che il vicino venga accettato o
no. I controlli di ammissibilita' (capacita') che scartano un candidato
PRIMA di calcolarne il costo non consumano FE: nessuna valutazione di
costo e' avvenuta. Prima di ogni valutazione viene controllato lo stato
del budget: l'arresto e' esatto.

Liste di candidati (riduzione dei vicinati)
-------------------------------------------
I vicinati inter-rotta completi costano O(n^2) valutazioni per passata:
insostenibile con FE = 3.5e5. Tecnica implementativa standard della
local search: per ogni cliente u si considerano solo mosse che creano
archi verso uno dei suoi h vicini piu' prossimi (default h = 20).
Il 2-opt intra-rotta resta a scansione completa per rotta (economico,
le rotte hanno pochi clienti).

Le scansioni iterano sui CLIENTI in ordine casuale (rng riproducibile):
dopo ogni mossa applicata le posizioni vengono riconsultate dallo stato
aggiornato, quindi la passata prosegue senza indici pendenti.
"""
from __future__ import annotations

import numpy as np

from .budget import Budget, BudgetExhausted
from .instance import Instance


# ---------------------------------------------------------------------------
# Liste di candidati: gli h clienti piu' vicini a ciascun cliente
# ---------------------------------------------------------------------------

def build_neighbors(inst: Instance, h: int = 20) -> np.ndarray:
    """Ritorna un array (n-1, h): la riga u-1 contiene gli h clienti piu'
    vicini a u (se stesso e il depot esclusi; i cloni a distanza 0 sono
    vicini legittimi). Costruito una volta per istanza."""
    n = inst.n
    h = min(h, n - 2)
    neigh = np.empty((n - 1, h), dtype=np.int64)
    big = np.iinfo(np.int64).max
    for u in range(1, n):
        d = inst.D[u, 1:].copy()
        d[u - 1] = big                       # esclude u stesso (non i cloni)
        idx = np.argsort(d, kind="stable")[:h]
        neigh[u - 1] = idx + 1
    return neigh


# ---------------------------------------------------------------------------
# Stato mutabile della soluzione durante la discesa
# ---------------------------------------------------------------------------

class _State:
    """Rotte + strutture derivate (carichi, somme prefisse, posizioni).

    Dopo OGNI mossa applicata lo stato derivato viene ricostruito da zero
    (_finalize): con n <= 101 costa O(n), trascurabile rispetto alle
    migliaia di valutazioni, e azzera la superficie di bug della
    manutenzione incrementale degli indici."""

    __slots__ = ("inst", "routes", "loads", "prefs", "route_of", "pos_of", "cost")

    def __init__(self, inst: Instance, routes, cost=None):
        self.inst = inst
        self.routes = [[int(c) for c in r] for r in routes if len(r) > 0]
        self.cost = int(cost) if cost is not None \
            else inst.solution_cost(self.routes)
        self.reindex()

    def reindex(self):
        n = self.inst.n
        dem = self.inst.demands
        self.loads = []
        self.prefs = []
        self.route_of = np.full(n, -1, dtype=np.int64)
        self.pos_of = np.full(n, -1, dtype=np.int64)
        for ri, r in enumerate(self.routes):
            pref = [0]
            for p, c in enumerate(r):
                self.route_of[c] = ri
                self.pos_of[c] = p
                pref.append(pref[-1] + int(dem[c]))
            self.prefs.append(pref)
            self.loads.append(pref[-1])


def _finalize(state: _State):
    """Da chiamare dopo ogni mossa applicata: elimina eventuali rotte
    svuotate e ricostruisce lo stato derivato."""
    if any(len(r) == 0 for r in state.routes):
        state.routes = [r for r in state.routes if r]
    state.reindex()


def _prev(route, p):
    return route[p - 1] if p > 0 else 0


def _next(route, p):
    return route[p + 1] if p + 1 < len(route) else 0


# ---------------------------------------------------------------------------
# 1) 2-opt intra-rotta
# ---------------------------------------------------------------------------

def _delta_two_opt(state, ri, i, j):
    """Inversione del segmento route[i..j] (inclusi), 0 <= i < j."""
    r = state.routes[ri]
    D = state.inst.D
    a = _prev(r, i)
    b = _next(r, j)
    return int(D[a, r[j]] + D[r[i], b] - D[a, r[i]] - D[r[j], b])


def _apply_two_opt(state, ri, i, j):
    d = _delta_two_opt(state, ri, i, j)
    r = state.routes[ri]
    r[i:j + 1] = r[i:j + 1][::-1]
    state.cost += d
    _finalize(state)
    return d


# ---------------------------------------------------------------------------
# 2/3) Or-opt / relocate: spostamento di una catena di L clienti
# ---------------------------------------------------------------------------

def _chain_ok(state, rA, i, L, rB):
    """Capacita': se la rotta di destinazione e' diversa, la catena deve
    entrarci. Controllo O(1) via somme prefisse (nessuna FE)."""
    if rA == rB:
        return True
    chain_load = state.prefs[rA][i + L] - state.prefs[rA][i]
    return state.loads[rB] + chain_load <= state.inst.capacity


def _delta_chain(state, rA, i, L, rB, gap, rev):
    """Rimuove la catena A[i..i+L-1] e la inserisce nella rotta rB alla
    posizione 'gap' della sequenza DOPO la rimozione (0 <= gap <= lenR);
    se rev la catena viene invertita. Precondizione: mossa ammissibile."""
    D = state.inst.D
    A = state.routes[rA]
    c1, cL = A[i], A[i + L - 1]
    p = _prev(A, i)
    s = _next(A, i + L - 1)
    gain_remove = D[p, c1] + D[cL, s] - D[p, s]

    if rA == rB:
        lenR = len(A) - L
        # elemento t-esimo della sequenza dopo la rimozione della catena
        def seq(t):
            return A[t] if t < i else A[t + L]
    else:
        B = state.routes[rB]
        lenR = len(B)

        def seq(t):
            return B[t]

    x = 0 if gap == 0 else seq(gap - 1)
    y = 0 if gap == lenR else seq(gap)
    h1, h2 = (cL, c1) if rev else (c1, cL)
    cost_insert = D[x, h1] + D[h2, y] - D[x, y]
    return int(cost_insert - gain_remove)


def _apply_chain(state, rA, i, L, rB, gap, rev):
    d = _delta_chain(state, rA, i, L, rB, gap, rev)
    A = state.routes[rA]
    chain = A[i:i + L]
    if rev:
        chain = chain[::-1]
    del A[i:i + L]
    target = A if rA == rB else state.routes[rB]
    target[gap:gap] = chain
    state.cost += d
    _finalize(state)
    return d


# ---------------------------------------------------------------------------
# 4) swap inter-rotta
# ---------------------------------------------------------------------------

def _swap_ok(state, u, v):
    """Capacita' in entrambe le direzioni (precondizione: rotte diverse)."""
    q = state.inst.demands
    Q = state.inst.capacity
    ru = int(state.route_of[u])
    rv = int(state.route_of[v])
    return (state.loads[ru] - int(q[u]) + int(q[v]) <= Q and
            state.loads[rv] - int(q[v]) + int(q[u]) <= Q)


def _delta_swap(state, u, v):
    """Scambio dei clienti u e v (precondizione: rotte diverse)."""
    D = state.inst.D
    A = state.routes[int(state.route_of[u])]
    B = state.routes[int(state.route_of[v])]
    pu_, su_ = _prev(A, int(state.pos_of[u])), _next(A, int(state.pos_of[u]))
    pv_, sv_ = _prev(B, int(state.pos_of[v])), _next(B, int(state.pos_of[v]))
    return int(D[pu_, v] + D[v, su_] - D[pu_, u] - D[u, su_]
               + D[pv_, u] + D[u, sv_] - D[pv_, v] - D[v, sv_])


def _apply_swap(state, u, v):
    d = _delta_swap(state, u, v)
    ru, rv = int(state.route_of[u]), int(state.route_of[v])
    pu, pv = int(state.pos_of[u]), int(state.pos_of[v])
    state.routes[ru][pu] = v
    state.routes[rv][pv] = u
    state.cost += d
    _finalize(state)
    return d


# ---------------------------------------------------------------------------
# 5) 2-opt*: scambio delle code di due rotte
# ---------------------------------------------------------------------------

def _star_ok(state, rA, i, rB, j):
    """Capacita' delle due nuove rotte, O(1) via somme prefisse."""
    Q = state.inst.capacity
    headA = state.prefs[rA][i + 1]           # carico di A[0..i] (0 se i = -1)
    headB = state.prefs[rB][j + 1]
    tailA = state.loads[rA] - headA
    tailB = state.loads[rB] - headB
    return headA + tailB <= Q and headB + tailA <= Q


def _delta_two_opt_star(state, rA, i, rB, j):
    """Nuove rotte: A[0..i] + B[j+1..] e B[0..j] + A[i+1..]
    (precondizione: rA != rB; i in -1..lenA-1, j in -1..lenB-1)."""
    D = state.inst.D
    A, B = state.routes[rA], state.routes[rB]
    a_i = A[i] if i >= 0 else 0
    a_i1 = A[i + 1] if i + 1 < len(A) else 0
    b_j = B[j] if j >= 0 else 0
    b_j1 = B[j + 1] if j + 1 < len(B) else 0
    return int(D[a_i, b_j1] + D[b_j, a_i1] - D[a_i, a_i1] - D[b_j, b_j1])


def _apply_two_opt_star(state, rA, i, rB, j):
    d = _delta_two_opt_star(state, rA, i, rB, j)
    A, B = state.routes[rA], state.routes[rB]
    newA = A[:i + 1] + B[j + 1:]
    newB = B[:j + 1] + A[i + 1:]
    state.routes[rA] = newA
    state.routes[rB] = newB
    state.cost += d
    _finalize(state)
    return d


# ---------------------------------------------------------------------------
# Discesa a vicinati multipli
# ---------------------------------------------------------------------------

class LocalSearch:
    """Discesa first-improvement sulla pipeline dei cinque vicinati.

    Le liste dei candidati vengono costruite UNA volta per istanza:
    creare un solo oggetto LocalSearch e riusarlo per tutte le chiamate
    della run. La discesa consuma FE dal budget condiviso; max_fe (se
    dato) limita le FE della singola invocazione."""

    def __init__(self, inst: Instance, budget: Budget, h: int = 20, rng=None):
        self.inst = inst
        self.budget = budget
        self.neigh = build_neighbors(inst, h)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---- ciclo principale ---------------------------------------------

    def improve(self, routes, cost=None, max_fe: int | None = None):
        """Migliora la soluzione data. Ritorna (routes, cost).

        Il costo in ingresso (se noto, es. da SplitResult) non viene
        ri-conteggiato: la sua valutazione e' gia' stata pagata dal
        chiamante. La discesa termina all'ottimo locale della pipeline
        oppure all'esaurimento del budget/cap, restituendo comunque una
        soluzione valida (mai peggiore di quella in ingresso)."""
        state = _State(self.inst, routes, cost)
        fe_start = self.budget.used

        def stop():
            if self.budget.exhausted:
                return True
            return max_fe is not None and self.budget.used - fe_start >= max_fe

        try:
            improved = True
            while improved and not stop():
                improved = False
                for sweep in (self._sweep_two_opt,
                              self._sweep_oropt1,
                              self._sweep_oropt2,
                              self._sweep_oropt3,
                              self._sweep_swap,
                              self._sweep_two_opt_star):
                    if sweep(state, stop):
                        improved = True
                    if stop():
                        break
        except BudgetExhausted:      # cintura di sicurezza: mai oltre il tetto
            pass
        return state.routes, state.cost

    # ---- scansioni first-improvement -----------------------------------

    def _sweep_two_opt(self, state, stop):
        """2-opt intra: scansione completa (i, j) per ogni rotta, ripetuta
        sulla rotta fino a quando non migliora piu'."""
        applied = False
        for ri in self.rng.permutation(len(state.routes)):
            ri = int(ri)
            moved = True
            while moved:
                moved = False
                r = state.routes[ri]
                m = len(r)
                for i in range(m - 1):
                    for j in range(i + 1, m):
                        if i == 0 and j == m - 1:
                            continue          # inversione totale: Delta = 0
                        if stop():
                            return applied
                        d = _delta_two_opt(state, ri, i, j)
                        self.budget.spend(1)
                        if d < 0:
                            _apply_two_opt(state, ri, i, j)
                            applied = moved = True
                            break
                    if moved:
                        break
        return applied

    def _sweep_oropt(self, state, L, stop):
        """Catene di L clienti ancorate a u, inserite prima/dopo uno dei
        suoi vicini v (stessa rotta o rotta diversa), dritte o invertite."""
        applied = False
        revs = (False,) if L == 1 else (False, True)
        for u in self.rng.permutation(np.arange(1, self.inst.n)):
            u = int(u)
            rA = int(state.route_of[u])
            i = int(state.pos_of[u])
            A = state.routes[rA]
            if i + L > len(A):
                continue                       # la catena non ci sta
            if L == len(A):
                pass                           # catena = intera rotta: solo inter (fusione)
            chain = A[i:i + L]
            found = False
            for v in self.neigh[u - 1]:
                v = int(v)
                if v in chain:
                    continue
                rB = int(state.route_of[v])
                if rA == rB and L == len(A):
                    continue                   # reinserire l'intera rotta in se': inutile
                if rA != rB and not _chain_ok(state, rA, i, L, rB):
                    continue                   # capacita' violata: nessuna FE
                pv = int(state.pos_of[v])
                idx_v = (pv if pv < i else pv - L) if rA == rB else pv
                for gap in (idx_v, idx_v + 1):
                    for rev in revs:
                        if rA == rB and gap == i and not rev:
                            continue           # reinserimento identico: no-op
                        if stop():
                            return applied
                        d = _delta_chain(state, rA, i, L, rB, gap, rev)
                        self.budget.spend(1)
                        if d < 0:
                            _apply_chain(state, rA, i, L, rB, gap, rev)
                            applied = found = True
                            break
                    if found:
                        break
                if found:
                    break
        return applied

    def _sweep_oropt1(self, state, stop):
        return self._sweep_oropt(state, 1, stop)

    def _sweep_oropt2(self, state, stop):
        return self._sweep_oropt(state, 2, stop)

    def _sweep_oropt3(self, state, stop):
        return self._sweep_oropt(state, 3, stop)

    def _sweep_swap(self, state, stop):
        applied = False
        for u in self.rng.permutation(np.arange(1, self.inst.n)):
            u = int(u)
            ru = int(state.route_of[u])
            for v in self.neigh[u - 1]:
                v = int(v)
                if int(state.route_of[v]) == ru:
                    continue
                if not _swap_ok(state, u, v):
                    continue                   # capacita': nessuna FE
                if stop():
                    return applied
                d = _delta_swap(state, u, v)
                self.budget.spend(1)
                if d < 0:
                    _apply_swap(state, u, v)
                    applied = True
                    break                      # prossimo u
        return applied

    def _sweep_two_opt_star(self, state, stop):
        applied = False
        for u in self.rng.permutation(np.arange(1, self.inst.n)):
            u = int(u)
            rA = int(state.route_of[u])
            pu = int(state.pos_of[u])
            found = False
            for v in self.neigh[u - 1]:
                v = int(v)
                rB = int(state.route_of[v])
                if rB == rA:
                    continue
                pv = int(state.pos_of[v])
                lenA = len(state.routes[rA])
                lenB = len(state.routes[rB])
                # variante 1: crea l'arco (u, v)  -> tagli (dopo u, prima di v)
                # variante 2: crea l'arco (v, u)  -> tagli (prima di u, dopo v)
                for i, j in ((pu, pv - 1), (pu - 1, pv)):
                    if i == -1 and j == -1:
                        continue               # scambio integrale: Delta = 0
                    if i == lenA - 1 and j == lenB - 1:
                        continue               # code vuote: no-op
                    if not _star_ok(state, rA, i, rB, j):
                        continue               # capacita': nessuna FE
                    if stop():
                        return applied
                    d = _delta_two_opt_star(state, rA, i, rB, j)
                    self.budget.spend(1)
                    if d < 0:
                        _apply_two_opt_star(state, rA, i, rB, j)
                        applied = found = True
                        break
                if found:
                    break
        return applied
