"""
tuning.py — Fase 6: taratura sistematica degli iperparametri.

Metodologia
-----------
Coordinate descent in due round su una griglia ridotta (la consegna
chiede scelte GIUSTIFICATE, non una ricerca esaustiva):

* Round 1 — budget di ricerca locale per generazione (la leva dominante
  emersa in Fase 5): cap di FE per invocazione (ls_cap_fe) e numero di
  cloni sottoposti a LS (ls_clones), a parita' di budget totale.
* Round 2 — struttura della popolazione (pop_size, dup) e aging (tau_b)
  sopra la configurazione vincente del Round 1.

Protocollo: 4 istanze eterogenee (A-n60-k9, B-n66-k9, E-n101-k14,
P-n101-k4: una per set, strutture diverse) x 3 seed di TUNING
(100..102, disgiunti dai seed 0..4 della campagna finale: non si tarano
i parametri sugli stessi campioni casuali usati per i risultati) x
FE = 3.5e5. Criterio: gap % medio dal BKS sulle 12 run; a parita',
si preferisce la deviazione standard minore.

Uso (dalla radice della repo):

    python -m experiments.tuning --round 1
    python -m experiments.tuning --round 2

Output: results/tuning_round<N>.csv (una riga per run) + tabella
riassuntiva a video. I CSV alimentano la tabella del tuning in relazione.
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool, cpu_count

from experiments.expcommon import DATA, RESULTS, TUNING_SEEDS, bks_of, gap_pct
from src.immune import ImmuneAlgorithm, ImmuneParams
from src.instance import load_instance

TUNING_INSTANCES = ["A-n60-k9", "B-n66-k9", "E-n101-k14", "P-n101-k4"]

# ---------------------------------------------------------------------------
# Configurazioni dei round (dict di override su ImmuneParams).
#
# IMPORTANTE — riproducibilita' storica: le configurazioni ancorano
# ESPLICITAMENTE la baseline pre-tuning (dup=2, tau_b=20, i default della
# Fase 5). Senza questo ancoraggio, ri-eseguire il tuning DOPO il
# congelamento dei parametri vincenti nei default farebbe ereditare alle
# configurazioni i default nuovi, rendendo alcune righe identiche fra
# loro (es. "dup3" == "base" se dup=3 e' gia' il default) e i round non
# piu' confrontabili con quelli originali.
# ---------------------------------------------------------------------------

BASELINE_PRE_TUNING = {"pop_size": 15, "dup": 2, "tau_b": 20}

ROUND1 = {
    # forma del budget LS per generazione: K x cap = FE di LS a generazione
    name: {**BASELINE_PRE_TUNING, **ov} for name, ov in {
        "K3_cap1500": {"ls_clones": 3, "ls_cap_fe": 1500},  # 4.5k/gen (griglia guida)
        "K3_cap2000": {"ls_clones": 3, "ls_cap_fe": 2000},  # 6k/gen  (default Fase 5)
        "K3_cap3000": {"ls_clones": 3, "ls_cap_fe": 3000},  # 9k/gen  (griglia guida)
        "K3_cap5000": {"ls_clones": 3, "ls_cap_fe": 5000},  # 15k/gen (sonda Fase 5)
        "K2_cap5000": {"ls_clones": 2, "ls_cap_fe": 5000},  # 10k/gen, discese profonde
        "K4_cap1500": {"ls_clones": 4, "ls_cap_fe": 1500},  # 6k/gen, discese ampie
    }.items()
}

# Round 2: sopra il vincitore del Round 1 (K=2, cap=5000; gap medio 3.57%
# sulla baseline pre-tuning), variazioni di popolazione e aging.
ROUND2_BASE = {**BASELINE_PRE_TUNING, "ls_clones": 2, "ls_cap_fe": 5000}
ROUND2 = {
    "base":   {},
    "pop10":  {"pop_size": 10},
    "pop20":  {"pop_size": 20},
    "dup3":   {"dup": 3},
    "tau10":  {"tau_b": 10},
    "tau30":  {"tau_b": 30},
}

# Round 3: verifica dell'interazione fra le due migliori direzioni del
# Round 2 (dup=3 e tau_b=10). Vincitore sul motore della Fase 5: 2.66%.
ROUND3 = {
    "dup3_tau10": {**ROUND2_BASE, "dup": 3, "tau_b": 10},
}

# Round 4 — RI-VALIDAZIONE sul motore FINALE. I Round 1-3 furono eseguiti
# sulla versione del motore con fitness = Split illimitato; l'analisi di
# P-n50-k10 (Fase 7) ha poi cambiato fitness (split_fitness: decodifica
# vincolata a k rotte) e selezione (lessicografica feasibility-first).
# I round storici documentano il percorso, ma la scelta congelata va
# confermata sull'algoritmo definitivo: qui si confrontano le quattro
# configurazioni di frontiera emerse dai round precedenti.
ROUND4 = {
    "K2c5000_d3_t20": {**BASELINE_PRE_TUNING, "ls_clones": 2, "ls_cap_fe": 5000, "dup": 3},
    "K2c5000_d3_t10": {**BASELINE_PRE_TUNING, "ls_clones": 2, "ls_cap_fe": 5000, "dup": 3, "tau_b": 10},
    "K3c5000_d3_t20": {**BASELINE_PRE_TUNING, "ls_clones": 3, "ls_cap_fe": 5000, "dup": 3},
    "K3c5000_d3_t10": {**BASELINE_PRE_TUNING, "ls_clones": 3, "ls_cap_fe": 5000, "dup": 3, "tau_b": 10},
}

# ---------------------------------------------------------------------------

def _run_job(job):
    """Una singola run (worker del pool). Ritorna la riga di risultato."""
    cfg_name, overrides, name, seed = job
    inst = load_instance(DATA / f"{name}.vrp")
    params = ImmuneParams(**overrides)
    t0 = time.perf_counter()
    res = ImmuneAlgorithm(inst, params=params, seed=seed).run()
    wall = time.perf_counter() - t0
    bks = bks_of(name)
    return {
        "config": cfg_name, "instance": name, "seed": seed,
        "best": res["best_cost"], "bks": bks,
        "gap": round(gap_pct(res["best_cost"], bks), 3),
        "n_routes": res["best_n_routes"], "respects_k": res["respects_k"],
        "gen_of_best": res["gen_of_best"], "fe_of_best": res["fe_of_best"],
        "generations": res["generations"], "wall": round(wall, 2),
        **{f"p_{k}": v for k, v in overrides.items()},
    }


def run_round(configs: dict, out_csv, workers: int | None = None):
    jobs = [(cfg, ov, name, seed)
            for cfg, ov in configs.items()
            for name in TUNING_INSTANCES
            for seed in TUNING_SEEDS]
    workers = workers or min(cpu_count(), 8)
    print(f"{len(jobs)} run su {workers} processi...")
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        rows = pool.map(_run_job, jobs)
    print(f"completate in {time.perf_counter() - t0:.0f}s")

    RESULTS.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("p_"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"salvato: {out_csv}")
    _summary(rows)
    return rows


def _summary(rows):
    """Tabella: gap medio (+/- std) per configurazione, totale e per istanza."""
    configs = sorted({r["config"] for r in rows})
    instances = sorted({r["instance"] for r in rows})

    def stats(sel):
        gaps = [r["gap"] for r in sel]
        m = sum(gaps) / len(gaps)
        s = (sum((g - m) ** 2 for g in gaps) / len(gaps)) ** 0.5
        return m, s

    head = f"{'config':<14} {'gap medio':>10} {'std':>6}  " + \
        "  ".join(f"{i.split('-')[0]+i.split('-')[2]:>8}" for i in instances)
    print("\n" + head)
    print("-" * len(head))
    ranking = []
    for cfg in configs:
        sel = [r for r in rows if r["config"] == cfg]
        m, s = stats(sel)
        per_inst = []
        for name in instances:
            mi, _ = stats([r for r in sel if r["instance"] == name])
            per_inst.append(f"{mi:>8.2f}")
        ranking.append((m, s, cfg))
        print(f"{cfg:<14} {m:>10.2f} {s:>6.2f}  " + "  ".join(per_inst))
    ranking.sort()
    print(f"\nmigliore: {ranking[0][2]}  (gap medio {ranking[0][0]:.2f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tuning Fase 6")
    ap.add_argument("--round", type=int, choices=(1, 2, 3, 4), required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    if args.round == 1:
        run_round(ROUND1, RESULTS / "tuning_round1.csv", args.workers)
    elif args.round == 2:
        configs = {name: {**ROUND2_BASE, **ov} for name, ov in ROUND2.items()}
        run_round(configs, RESULTS / "tuning_round2.csv", args.workers)
    elif args.round == 3:
        run_round(ROUND3, RESULTS / "tuning_round3.csv", args.workers)
    else:
        run_round(ROUND4, RESULTS / "tuning_round4.csv", args.workers)


if __name__ == "__main__":       # obbligatorio per multiprocessing su Windows
    main()
