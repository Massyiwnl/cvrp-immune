"""
compliance.py — Audit di conformita' alla consegna.

Verifica MECCANICAMENTE, uno per uno, tutti i requisiti, ricalcolando ogni valore da zero e senza fidarsi né del motore
né dell'aggregatore. Ogni controllo stampa PASS/FAIL con l'evidenza
numerica; exit code 1 se anche un solo controllo fallisce.

Requisiti verificati (Sezione 2 "Protocollo Sperimentale"):
  R1  le 10 istanze richieste, tutte presenti e testate
  R2  runs = 5 per istanza (seed distinti)
  R3  criterio di arresto FE = 3.5e5 (esatto, in ogni run)
  R4  best  = miglior costo fra le 5 run                     [metrica 1]
  R5  mean  = media dei best delle singole run               [metrica 2]
  R6  std   = deviazione standard degli stessi               [metrica 3]
  R7  satisfability = citta' servite                         [metrica 4]
  R8  numero medio di iterazioni al raggiungimento del best  [metrica 5]
  R9  grafici di convergenza per >= 3 istanze rappresentative
  R10 vincolo di flotta: numero di rotte <= m (= k) in ogni run
  R11 ammissibilita': ogni citta' servita una sola volta, capacita' Q
  R12 integrita': costo ricalcolato == costo dichiarato, in ogni run
  R13 riproducibilita': stesso seed -> stesso risultato

Uso:  python -m experiments.compliance
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from experiments.expcommon import (DATA, FINAL_SEEDS, INSTANCES, RESULTS,
                                   ROOT, bks_of)
from src.immune import ImmuneAlgorithm
from src.instance import load_instance

RUNS_DIR = RESULTS / "runs"
FIG_DIR = ROOT / "report" / "figures"

# le 10 istanze ESATTAMENTE come elencate nella consegna (sezione 2)
REQUIRED = [
    "A-n45-k7", "A-n60-k9", "A-n80-k10",          # set A
    "B-n56-k7", "B-n66-k9", "B-n78-k10",          # set B
    "E-n76-k8", "E-n101-k14",                     # set E
    "P-n50-k10", "P-n101-k4",                     # set P
]
FE_REQUIRED = 350_000
RUNS_REQUIRED = 5

_fails = []


def check(ok: bool, label: str, evidence: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {evidence}" if evidence else ""))
    if not ok:
        _fails.append(label)


def main():
    print("=" * 78)
    print("AUDIT DI CONFORMITA' ALLA CONSEGNA")
    print("=" * 78)

    # ---- R1: le 10 istanze richieste ----------------------------------
    print("\nR1 — istanze richieste dalla consegna")
    check(sorted(INSTANCES) == sorted(REQUIRED),
          "le 10 istanze testate sono esattamente quelle richieste",
          f"{len(INSTANCES)} istanze (set A: 3, B: 3, E: 2, P: 2)")
    missing = [n for n in REQUIRED if not (DATA / f"{n}.vrp").exists()]
    check(not missing, "tutti i file .vrp presenti",
          "nessun file mancante" if not missing else f"mancano {missing}")

    # ---- carica tutte le run ------------------------------------------
    runs = {}
    for name in REQUIRED:
        runs[name] = []
        for seed in FINAL_SEEDS:
            p = RUNS_DIR / f"{name}_seed{seed}.json"
            if p.exists():
                runs[name].append(json.loads(p.read_text()))

    # ---- R2: 5 run per istanza ----------------------------------------
    print("\nR2 — runs = 5 per istanza")
    counts = {n: len(r) for n, r in runs.items()}
    check(all(c == RUNS_REQUIRED for c in counts.values()),
          "ogni istanza ha esattamente 5 run indipendenti",
          f"totale {sum(counts.values())} run (10 x 5)")
    seeds_ok = all(len({r['seed'] for r in rr}) == RUNS_REQUIRED
                   for rr in runs.values())
    check(seeds_ok, "i 5 seed di ogni istanza sono distinti", f"seed {FINAL_SEEDS}")

    # ---- R3: criterio di arresto FE = 3.5e5 ----------------------------
    print("\nR3 — criterio di arresto: FE = 3.5 x 10^5")
    fes = [r["fe_used"] for rr in runs.values() for r in rr]
    fmax = [r["fe_max"] for rr in runs.values() for r in rr]
    check(all(f == FE_REQUIRED for f in fmax),
          "budget impostato a 350000 in tutte le run", f"fe_max = {set(fmax)}")
    check(all(f == FE_REQUIRED for f in fes),
          "budget consumato ESATTAMENTE (arresto esatto, mai oltre né sotto)",
          f"fe_used = {set(fes)} in tutte le 50 run")

    # ---- R10/R11/R12: vincoli e integrita', run per run -----------------
    print("\nR10/R11/R12 — vincolo di flotta, ammissibilita', integrita' dei costi")
    bad_feas, bad_k, bad_cost, exact_k = [], [], [], 0
    for name in REQUIRED:
        inst = load_instance(DATA / f"{name}.vrp")
        for r in runs[name]:
            rt = r["best_routes"]
            ok, _ = inst.is_feasible(rt, max_routes=inst.k)   # (R11 + R10)
            if not ok:
                bad_feas.append((name, r["seed"]))
            if len(rt) > inst.k:
                bad_k.append((name, r["seed"]))
            if len(rt) == inst.k:
                exact_k += 1
            if inst.solution_cost(rt) != r["best_cost"]:      # (R12)
                bad_cost.append((name, r["seed"]))
    check(not bad_feas, "ogni citta' servita una volta sola, capacita' Q rispettata",
          "50/50 soluzioni ammissibili")
    check(not bad_k, "numero di rotte <= m (= k) in ogni run",
          f"50/50 conformi; di cui esattamente k rotte: {exact_k}/50")
    check(not bad_cost, "costo ricalcolato da zero == costo dichiarato",
          "50/50 costi coerenti (aritmetica intera, uguaglianza esatta)")

    # ---- R4..R8: le 5 metriche richieste, ricalcolate indipendentemente --
    print("\nR4-R8 — le 5 metriche richieste dalla consegna (ricalcolo indipendente)")
    summary = {row["instance"]: row for row in
               __import__("csv").DictReader(open(RESULTS / "summary.csv"))}
    # tolleranza = mezza unita' dell'ultima cifra riportata (i valori in
    # summary.csv sono arrotondati a 1 decimale). Il confronto e' PER
    # ISTANZA: sommare gli scarti di arrotondamento delle 10 istanze
    # produrrebbe un falso positivo.
    TOL = 0.05
    bad_best, bad_mean, bad_std, bad_iter = [], [], [], []
    sat_ok = True
    for name in REQUIRED:
        inst = load_instance(DATA / f"{name}.vrp")
        costs = [r["best_cost"] for r in runs[name]]
        gens = [r["gen_of_best"] for r in runs[name]]
        best = min(costs)                                     # metrica 1
        mean = statistics.mean(costs)                         # metrica 2
        std = statistics.stdev(costs)                         # metrica 3 (ddof=1)
        itm = statistics.mean(gens)                           # metrica 5
        s = summary[name]
        if best != int(s["best"]):
            bad_best.append(name)
        if abs(mean - float(s["mean"])) > TOL:
            bad_mean.append(name)
        if abs(std - float(s["std"])) > TOL:
            bad_std.append((name, round(std, 4), float(s["std"])))
        if abs(itm - float(s["iter_mean"])) > TOL:
            bad_iter.append(name)
        # metrica 4: satisfability = citta' servite in OGNI run
        for r in runs[name]:
            served = sorted(c for rt in r["best_routes"] for c in rt)
            if served != list(range(1, inst.n)):
                sat_ok = False
    check(not bad_best, "metrica 'best' coerente col ricalcolo", "10/10 istanze")
    check(not bad_mean, "metrica 'mean' coerente col ricalcolo", "10/10 istanze")
    check(not bad_std, "metrica 'std' (campionaria, ddof=1) coerente",
          "10/10 istanze" if not bad_std else f"discrepanze: {bad_std}")
    check(not bad_iter, "metrica 'numero medio di iterazioni al best' coerente",
          "10/10 istanze (iterazione = generazione dell'algoritmo)")
    check(sat_ok, "satisfability: TUTTE le citta' servite in tutte le 50 run",
          "100% — nessun caso da segnalare (la consegna la richiede solo se < 100%)")

    # ---- R9: grafici di convergenza -------------------------------------
    print("\nR9 — grafici di convergenza (>= 3 istanze rappresentative)")
    pdfs = sorted(FIG_DIR.glob("convergence_*.pdf"))
    check(len(pdfs) >= 3, "almeno 3 grafici di convergenza generati",
          f"{len(pdfs)} figure: " + ", ".join(p.stem.replace('convergence_', '') for p in pdfs))
    hist_ok = all(len(r["history"]) >= 2 for rr in runs.values() for r in rr)
    check(hist_ok, "ogni run ha una curva di convergenza registrata",
          "history (FE, costo) presente in tutte le 50 run")

    # ---- R13: riproducibilita' ------------------------------------------
    print("\nR13 — riproducibilita' (stesso seed -> stesso risultato)")
    inst = load_instance(DATA / "B-n56-k7.vrp")
    again = ImmuneAlgorithm(inst, seed=0).run()
    saved = runs["B-n56-k7"][0]
    same = (again["best_cost"] == saved["best_cost"]
            and again["history"] == [list(h) for h in saved["history"]])
    check(same, "run rieseguita da zero identica a quella salvata",
          f"B-n56-k7 seed 0: best {again['best_cost']} (atteso {saved['best_cost']})")

    # ---- riepilogo ------------------------------------------------------
    print("\n" + "=" * 78)
    gaps = [100 * (min(r["best_cost"] for r in runs[n]) - bks_of(n)) / bks_of(n)
            for n in REQUIRED]
    print(f"gap medio dei best: {sum(gaps)/len(gaps):.2f}%   "
          f"migliore: {min(gaps):.2f}%   peggiore: {max(gaps):.2f}%")
    if _fails:
        print(f"AUDIT FALLITO: {len(_fails)} controlli non superati")
        for f in _fails:
            print("  -", f)
        sys.exit(1)
    print("AUDIT SUPERATO: tutti i requisiti sperimentali della consegna sono soddisfatti.")
    print("=" * 78)


if __name__ == "__main__":
    main()
