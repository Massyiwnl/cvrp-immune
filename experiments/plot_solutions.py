"""
plot_solutions.py — Diagrammi delle soluzioni sul piano.

Produce due tipi di figura in report/figures/:

1. solution_<istanza>.pdf — due pannelli affiancati: la MIGLIORE soluzione
   trovata dall'algoritmo (sinistra) e la soluzione OTTIMA di CVRPLIB
   (destra), con costi e gap. Rende visibile *dove* la nostra soluzione
   si discosta dall'ottimo, e mostra la struttura dei diversi set
   (clienti uniformi nel set A, clusterizzati nel set B).

2. structure_E-vs-P.pdf — confronto strutturale fra E-n101-k14 e
   P-n101-k4: le due istanze condividono ESATTAMENTE la stessa geometria
   (coordinate e domande identiche: stesso set di 100 clienti di
   Christofides & Eilon) e differiscono SOLO per la capacita' dei veicoli
   (112 contro 400), quindi per il numero di rotte imposto (14 contro 4).
   E' un esperimento naturale controllato: l'unica variabile e' la
   lunghezza delle rotte (~7 contro ~25 clienti), il che permette di
   isolare l'effetto della componente TSP intra-rotta sulla difficolta'
   del problema per l'algoritmo.

Uso:  python -m experiments.plot_solutions
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.expcommon import DATA, FINAL_SEEDS, RESULTS, ROOT, bks_of, gap_pct
from src.instance import (load_instance, load_solution,
                          sol_to_internal_candidates)

RUNS_DIR = RESULTS / "runs"
FIG_DIR = ROOT / "report" / "figures"

# una istanza per set (strutture diverse) + le due grandi
SOLUTION_INSTANCES = ["A-n45-k7", "B-n78-k10", "E-n101-k14", "P-n101-k4"]


def _best_routes(name):
    """Rotte della migliore soluzione fra le 5 run, e suo costo."""
    runs = [json.loads((RUNS_DIR / f"{name}_seed{s}.json").read_text())
            for s in FINAL_SEEDS]
    best = min(runs, key=lambda r: r["best_cost"])
    return best["best_routes"], best["best_cost"], best["seed"]


def _bks_routes(inst, name):
    """Rotte della soluzione ottima ufficiale, in indici interni."""
    raw, cost = load_solution(DATA / f"{name}.sol")
    for _label, cand in sol_to_internal_candidates(raw, inst.n):
        ok, _ = inst.is_feasible(cand, max_routes=inst.k)
        if ok and inst.solution_cost(cand) == cost:
            return cand, cost
    raise AssertionError(f"{name}: convenzione .sol non identificata")


def _draw(ax, inst, routes, title):
    """Disegna depot, clienti e rotte (ogni rotta di un colore diverso)."""
    xy = inst.coords
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(routes):
        path = [0] + list(r) + [0]              # il ciclo parte e torna al depot
        ax.plot(xy[path, 0], xy[path, 1], "-", linewidth=1.1,
                color=cmap(i % 20), zorder=1)
    ax.scatter(xy[1:, 0], xy[1:, 1], s=14, color="0.25", zorder=2)
    ax.scatter(xy[0, 0], xy[0, 1], s=130, marker="s", color="black",
               zorder=3, label="depot")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)


def plot_solution(name: str):
    inst = load_instance(DATA / f"{name}.vrp")
    ours, cost, seed = _best_routes(name)
    opt, bks = _bks_routes(inst, name)
    gap = gap_pct(cost, bks)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    _draw(axes[0], inst, ours,
          f"Soluzione trovata — costo {cost}  (gap {gap:.2f}%)\n"
          f"{len(ours)} rotte, seed {seed}")
    _draw(axes[1], inst, opt,
          f"Soluzione ottima (CVRPLIB) — costo {bks}\n{len(opt)} rotte")
    fig.suptitle(f"{name}  (n = {inst.n_customers} clienti, Q = {inst.capacity}, "
                 f"k = {inst.k})", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"solution_{name}.pdf")
    fig.savefig(FIG_DIR / f"solution_{name}.png", dpi=140)
    plt.close(fig)
    print(f"{name:<12} nostra {cost:>5} (gap {gap:>5.2f}%)   ottimo {bks:>5}   "
          f"rotte {len(ours)}/{inst.k}")


def plot_structure_E_vs_P():
    """Stessa geometria, capacita' diversa: 14 rotte corte contro 4 lunghe."""
    e = load_instance(DATA / "E-n101-k14.vrp")
    p = load_instance(DATA / "P-n101-k4.vrp")
    assert np.array_equal(e.coords, p.coords) and np.array_equal(e.demands, p.demands)

    e_routes, e_cost, _ = _best_routes("E-n101-k14")
    p_routes, p_cost, _ = _best_routes("P-n101-k4")
    e_gap = gap_pct(e_cost, bks_of("E-n101-k14"))
    p_gap = gap_pct(p_cost, bks_of("P-n101-k4"))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    _draw(axes[0], e, e_routes,
          f"E-n101-k14 — Q = 112, k = 14\ncosto {e_cost} (gap {e_gap:.2f}%), "
          f"~{e.n_customers/e.k:.0f} clienti/rotta")
    _draw(axes[1], p, p_routes,
          f"P-n101-k4 — Q = 400, k = 4\ncosto {p_cost} (gap {p_gap:.2f}%), "
          f"~{p.n_customers/p.k:.0f} clienti/rotta")
    fig.suptitle("Stessa geometria (100 clienti e domande identiche), "
                 "capacit\u00e0 diversa: rotte corte contro rotte lunghe",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "structure_E-vs-P.pdf")
    fig.savefig(FIG_DIR / "structure_E-vs-P.png", dpi=140)
    plt.close(fig)
    print("\nstructure_E-vs-P: stessa geometria, Q = 112 (14 rotte) contro "
          "Q = 400 (4 rotte)")


def main():
    for name in SOLUTION_INSTANCES:
        plot_solution(name)
    plot_structure_E_vs_P()
    print(f"\nfigure salvate in {FIG_DIR}/")


if __name__ == "__main__":
    main()
