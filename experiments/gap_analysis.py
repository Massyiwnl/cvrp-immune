"""
gap_analysis.py — Scomposizione del gap dall'ottimo, ovvero, quanto si perde
nell'ORDINAMENTO dei clienti dentro le rotte (sotto-problema TSP) e
quanto nell'ASSEGNAMENTO dei clienti alle rotte (partizione)?

Metodo
------
Data la migliore soluzione trovata su un'istanza, si ri-ottimizza in modo
ESATTO l'ordine di visita all'interno di ciascuna rotta, lasciando
INVARIATO l'insieme dei clienti assegnati a ogni veicolo:

  costo_nostro     costo della soluzione trovata
  costo_riordinato costo della STESSA partizione, con ogni rotta risolta
                   all'ottimo come TSP (Held-Karp esatto fino a 13
                   clienti per rotta; multi-start 2-opt + Or-opt oltre)
  BKS              ottimo di CVRPLIB

  perdita_ordinamento  = costo_nostro     - costo_riordinato   (>= 0)
  perdita_assegnamento = costo_riordinato - BKS                (>= 0)
  gap totale           = perdita_ordinamento + perdita_assegnamento

La scomposizione e' esatta e additiva: dice se il nostro algoritmo
sbaglia soprattutto "come ordina" o "come raggruppa" i clienti — e la
risposta cambia con la lunghezza delle rotte (7 clienti in E-n101-k14
contro 25 in P-n101-k4, a parita' di geometria).

In piu': conteggio degli AUTO-INCROCI delle rotte. In una soluzione
2-opt-ottima (distanze euclidee) nessuna rotta puo' intersecare se
stessa: gli incroci residui misurano quanto la ricerca locale sia
lontana dalla convergenza sul best riportato.

Uso:  python -m experiments.gap_analysis
"""
from __future__ import annotations

import csv
import json
from itertools import combinations

import numpy as np

from experiments.expcommon import (DATA, FINAL_SEEDS, INSTANCES, RESULTS,
                                   bks_of, gap_pct)
from src.instance import load_instance

RUNS_DIR = RESULTS / "runs"
EXACT_MAX = 13          # oltre questa taglia, TSP euristico multi-start
RESTARTS = 300


# ---------------------------------------------------------------------------
# TSP di una singola rotta (ciclo: depot -> clienti -> depot)
# ---------------------------------------------------------------------------

def _tsp_exact(route, D):
    """Held-Karp: costo minimo del ciclo depot -> tutti i clienti -> depot."""
    m = len(route)
    if m <= 2:
        return int(D[0, route[0]] + D[route[0], route[-1]] + D[route[-1], 0]) \
            if m == 2 else int(2 * D[0, route[0]])
    INF = float("inf")
    dp = [[INF] * m for _ in range(1 << m)]
    for i in range(m):
        dp[1 << i][i] = int(D[0, route[i]])
    for S in range(1 << m):
        row = dp[S]
        for i in range(m):
            cur = row[i]
            if cur == INF or not (S >> i) & 1:
                continue
            di = D[route[i]]
            for j in range(m):
                if (S >> j) & 1:
                    continue
                nS = S | (1 << j)
                val = cur + int(di[route[j]])
                if val < dp[nS][j]:
                    dp[nS][j] = val
    full = (1 << m) - 1
    return int(min(dp[full][i] + int(D[route[i], 0]) for i in range(m)))


def _tsp_heuristic(route, D, rng):
    """Multi-start 2-opt + Or-opt (rotte troppo lunghe per Held-Karp)."""
    def cost(seq):
        c = int(D[0, seq[0]] + D[seq[-1], 0])
        for a, b in zip(seq[:-1], seq[1:]):
            c += int(D[a, b])
        return c

    def descend(seq):
        seq = list(seq)
        improved = True
        while improved:
            improved = False
            m = len(seq)
            # 2-opt
            for i in range(m - 1):
                a = seq[i - 1] if i > 0 else 0
                for j in range(i + 1, m):
                    b = seq[j + 1] if j + 1 < m else 0
                    d = (D[a, seq[j]] + D[seq[i], b]
                         - D[a, seq[i]] - D[seq[j], b])
                    if d < 0:
                        seq[i:j + 1] = seq[i:j + 1][::-1]
                        improved = True
            # Or-opt (catene di 1..3, anche invertite)
            for L in (1, 2, 3):
                m = len(seq)
                for i in range(m - L + 1):
                    chain = seq[i:i + L]
                    p = seq[i - 1] if i > 0 else 0
                    s = seq[i + L] if i + L < m else 0
                    gain = D[p, chain[0]] + D[chain[-1], s] - D[p, s]
                    rest = seq[:i] + seq[i + L:]
                    for g in range(len(rest) + 1):
                        x = rest[g - 1] if g > 0 else 0
                        y = rest[g] if g < len(rest) else 0
                        for ch in (chain, chain[::-1]):
                            add = D[x, ch[0]] + D[ch[-1], y] - D[x, y]
                            if add - gain < 0:
                                seq = rest[:g] + list(ch) + rest[g:]
                                improved = True
                                break
                        else:
                            continue
                        break
                    if improved:
                        break
                if improved:
                    break
        return seq, cost(seq)

    best = cost(route)
    _, c = descend(route)                       # dalla soluzione corrente
    best = min(best, c)
    arr = np.array(route)
    for _ in range(RESTARTS):                   # da permutazioni casuali
        _, c = descend(list(rng.permutation(arr)))
        best = min(best, c)
    return int(best)


# ---------------------------------------------------------------------------
# Auto-incroci di una rotta (test di 2-opt-ottimalita' geometrica)
# ---------------------------------------------------------------------------

def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        v = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return int(v > 1e-12) - int(v < -1e-12)
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 != o2 and o3 != o4 and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0


def _self_crossings(routes, coords):
    n = 0
    for r in routes:
        cyc = [0] + list(r) + [0]
        segs = list(zip(cyc[:-1], cyc[1:]))
        for (a, b), (c, d) in combinations(segs, 2):
            if len({a, b, c, d}) < 4:           # segmenti adiacenti: mai "incrocio"
                continue
            if _segments_cross(coords[a], coords[b], coords[c], coords[d]):
                n += 1
    return n


# ---------------------------------------------------------------------------

def analyze(name, rng):
    inst = load_instance(DATA / f"{name}.vrp")
    runs = [json.loads((RUNS_DIR / f"{name}_seed{s}.json").read_text())
            for s in FINAL_SEEDS]
    best = min(runs, key=lambda r: r["best_cost"])
    routes = best["best_routes"]
    ours = best["best_cost"]
    bks = bks_of(name)

    reordered = 0
    exact = True
    for r in routes:
        if len(r) <= EXACT_MAX:
            reordered += _tsp_exact(r, inst.D)
        else:
            reordered += _tsp_heuristic(r, inst.D, rng)
            exact = False

    lens = [len(r) for r in routes]
    return {
        "instance": name,
        "ours": ours,
        "reordered": reordered,
        "bks": bks,
        "loss_ordering": ours - reordered,
        "loss_assignment": reordered - bks,
        "gap_pct": round(gap_pct(ours, bks), 2),
        "pct_ordering": round(100 * (ours - reordered) / max(1, ours - bks), 0),
        "route_len_avg": round(sum(lens) / len(lens), 1),
        "route_len_max": max(lens),
        "crossings": _self_crossings(routes, inst.coords),
        "tsp_exact": exact,
    }


def main():
    rng = np.random.default_rng(0)
    rows = [analyze(n, rng) for n in INSTANCES]

    out = RESULTS / "gap_decomposition.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    head = (f"{'istanza':<12} {'clienti/rotta':>13} {'nostro':>7} {'riordin.':>8} "
            f"{'BKS':>5} {'perdita ord.':>12} {'perdita assegn.':>15} "
            f"{'% ord.':>7} {'incroci':>7}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['instance']:<12} {r['route_len_avg']:>7.1f} (max {r['route_len_max']:>2}) "
              f"{r['ours']:>7} {r['reordered']:>8} {r['bks']:>5} "
              f"{r['loss_ordering']:>12} {r['loss_assignment']:>15} "
              f"{str(int(r['pct_ordering']))+'%':>7} {r['crossings']:>7}")
    print("-" * len(head))
    tot_o = sum(r["loss_ordering"] for r in rows)
    tot_a = sum(r["loss_assignment"] for r in rows)
    print(f"totale: perdita ordinamento {tot_o} ({100*tot_o/(tot_o+tot_a):.0f}%), "
          f"perdita assegnamento {tot_a} ({100*tot_a/(tot_o+tot_a):.0f}%)")
    print(f"\nsalvato: {out}")


if __name__ == "__main__":
    main()
