"""
greedy.py — Euristica costruttiva di Clarke-Wright (savings), versione
parallela. Argomento del corso: algoritmi greedy.

Usi nel progetto: (1) semina di una parte della popolazione iniziale del
motore immunologico (Fase 5); (2) punto di partenza ragionevole nei test
di qualita' della ricerca locale (Fase 4).

Idea
----
Si parte dalla soluzione banale con n-1 rotte (depot -> cliente -> depot).
Fondere le rotte di i e j rendendo (i, j) un arco interno fa risparmiare

    s(i, j) = D[0,i] + D[0,j] - D[i,j]  >= 0   (disuguaglianza triangolare)

Strategia greedy: si ordinano i risparmi in senso decrescente e si
applica ogni fusione ammissibile, cioe' quando i e j (1) appartengono a
rotte diverse, (2) sono entrambi ESTREMI delle rispettive rotte (l'arco
(i,j) deve poter essere creato senza spezzare archi interni), e (3) la
somma dei carichi delle due rotte rispetta Q.

L'ordinamento con tie-break deterministico rende l'euristica
riproducibile: la diversificazione della popolazione iniziale sara'
ottenuta perturbando la soluzione CW con mutazioni casuali (Fase 5),
non randomizzando il greedy.
"""
from __future__ import annotations

from .instance import Instance


def clarke_wright(inst: Instance) -> list[list[int]]:
    """Costruisce una soluzione CVRP con il metodo dei savings.

    Ritorna una lista di rotte (liste di clienti interni). La soluzione
    e' feasible per costruzione rispetto alla capacita'; il numero di
    rotte NON e' garantito essere <= k (di norma ci si avvicina molto)."""
    D, Q, dem, n = inst.D, inst.capacity, inst.demands, inst.n

    routes: list[list[int]] = [[c] for c in range(1, n)]   # rotte banali
    load = [int(dem[c]) for c in range(1, n)]              # carico per rotta
    rid = {c: c - 1 for c in range(1, n)}                  # cliente -> indice rotta

    # savings s(i,j) per ogni coppia di clienti, ordinati in senso
    # decrescente con tie-break deterministico su (i, j)
    savings = []
    for i in range(1, n):
        Di0 = int(D[0, i])
        for j in range(i + 1, n):
            s = Di0 + int(D[0, j]) - int(D[i, j])
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda t: (-t[0], t[1], t[2]))

    for _s, i, j in savings:
        ri, rj = rid[i], rid[j]
        if ri == rj:
            continue                                        # gia' nella stessa rotta
        if load[ri] + load[rj] > Q:
            continue                                        # capacita' violata
        A, B = routes[ri], routes[rj]
        # i deve diventare la CODA di A ...
        if A[0] == i:
            A.reverse()
        elif A[-1] != i:
            continue                                        # i e' interno: fusione impossibile
        # ... e j la TESTA di B, cosi' l'arco (i, j) diventa interno
        if B[-1] == j:
            B.reverse()
        elif B[0] != j:
            continue                                        # j e' interno
        A.extend(B)
        load[ri] += load[rj]
        for c in B:
            rid[c] = ri
        routes[rj] = []                                     # rotta assorbita

    return [r for r in routes if r]
