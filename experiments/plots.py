"""
plots.py — Fase 8: grafici di convergenza per la relazione.

Per ciascuna istanza rappresentativa (una per set, strutture diverse):

    A-n80-k10   clienti uniformi, istanza media-grande del set A
    B-n78-k10   clienti CLUSTERIZZATI (caratteristica del set B)
    E-n101-k14  grande, 14 rotte corte (capacita' 112)
    P-n101-k4   grande, 4 rotte LUNGHE (capacita' 400, quasi-TSP)

il grafico mostra il costo della migliore soluzione trovata (best-so-far)
in funzione delle FE consumate: 5 curve sottili (una per run, funzioni a
gradini ricostruite dalla history dei miglioramenti), la curva MEDIA in
evidenza (campionata su una griglia comune di FE, tracciata solo da
quando tutte e 5 le run hanno un best definito) e la linea tratteggiata
del BKS. Esporta PDF e PNG (anteprima) in report/figures/.

Uso:  python -m experiments.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.expcommon import FINAL_SEEDS, RESULTS, ROOT, bks_of

RUNS_DIR = RESULTS / "runs"
FIG_DIR = ROOT / "report" / "figures"

# una istanza per set, con strutture volutamente diverse (vedi docstring)
PLOT_INSTANCES = ["A-n80-k10", "B-n78-k10", "E-n101-k14", "P-n101-k4"]

FE_MAX = 350_000
GRID = np.arange(0, FE_MAX + 1, 1000)


def _step_on_grid(history, grid):
    """Valuta la funzione a gradini best-so-far sui punti della griglia.
    Prima del primo miglioramento il best non e' definito -> NaN."""
    fes = np.array([fe for fe, _ in history], dtype=np.int64)
    costs = np.array([c for _, c in history], dtype=np.float64)
    idx = np.searchsorted(fes, grid, side="right") - 1
    out = np.where(idx >= 0, costs[np.clip(idx, 0, None)], np.nan)
    return out


def plot_instance(name: str) -> Path:
    bks = bks_of(name)
    histories = []
    for seed in FINAL_SEEDS:
        res = json.loads((RUNS_DIR / f"{name}_seed{seed}.json").read_text())
        histories.append(res["history"])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # curve delle singole run (funzioni a gradini)
    for i, hist in enumerate(histories):
        fes = [fe for fe, _ in hist] + [FE_MAX]
        costs = [c for _, c in hist] + [hist[-1][1]]
        ax.plot(fes, costs, drawstyle="steps-post", linewidth=0.9,
                alpha=0.45, color="tab:blue",
                label="singole run (5)" if i == 0 else None)

    # curva media sulla griglia comune (solo dove tutte e 5 sono definite)
    curves = np.vstack([_step_on_grid(h, GRID) for h in histories])
    defined = ~np.isnan(curves).any(axis=0)
    mean_curve = curves.mean(axis=0)
    ax.plot(GRID[defined], mean_curve[defined], linewidth=2.2,
            color="tab:red", label="media delle 5 run")

    # best-known solution
    ax.axhline(bks, linestyle="--", linewidth=1.2, color="black",
               label=f"BKS = {bks}")

    ax.set_xlabel("Valutazioni di fitness (FE)")
    ax.set_ylabel("Costo della migliore soluzione")
    ax.set_title(f"Convergenza su {name}")
    ax.set_xlim(0, FE_MAX)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"convergence_{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIG_DIR / f"convergence_{name}.png", dpi=150)
    plt.close(fig)

    # statistiche utili per i commenti in relazione
    fe_best = [h[-1][0] for h in histories]
    at50k = curves[:, GRID.searchsorted(50_000)]
    final = curves[:, -1]
    print(f"{name:<12} FE medie al best: {np.mean(fe_best):>9,.0f}   "
          f"gap medio a 50k FE: {100*(np.nanmean(at50k)-bks)/bks:5.2f}%   "
          f"finale: {100*(np.mean(final)-bks)/bks:5.2f}%")
    return pdf


def main():
    print(f"{'istanza':<12} {'':>24} {'':>22}")
    for name in PLOT_INSTANCES:
        plot_instance(name)
    print(f"\nfigure salvate in {FIG_DIR}/ (PDF per LaTeX + PNG anteprima)")


if __name__ == "__main__":
    main()
