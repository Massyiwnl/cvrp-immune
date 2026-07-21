"""
aggregate.py — Fase 7: aggregazione dei risultati nella tabella finale
richiesta dalla consegna. Per ciascuna istanza, sulle 5 run:

  * best  — miglior costo fra le 5 run;
  * mean  — media dei migliori costi di ciascuna run;
  * std   — deviazione standard campionaria (ddof=1) degli stessi;
  * iterazioni medie — media, sulle 5 run, della generazione in cui la
    best della run e' stata raggiunta (gen_of_best);
  * satisfability — numero di citta' servite: con il decoder Split e'
    il 100% per costruzione in ogni run (la consegna chiede il dato solo
    se non tutte soddisfatte; il validatore lo riverifica comunque).

In piu', per l'analisi: BKS (letto dai .sol ufficiali), gap % del best e
della media, FE medie al best, numero di rotte del best (sempre <= k).

Output: results/summary.csv (dati completi), results/summary_table.tex
(tabella booktabs)), tabella a video.

Uso:  python -m experiments.aggregate
"""
from __future__ import annotations

import csv
import json
import statistics

from experiments.expcommon import (FINAL_SEEDS, INSTANCES, RESULTS,
                                   bks_of, gap_pct)

RUNS_DIR = RESULTS / "runs"


def aggregate() -> list[dict]:
    rows = []
    for name in INSTANCES:
        runs = []
        for seed in FINAL_SEEDS:
            path = RUNS_DIR / f"{name}_seed{seed}.json"
            runs.append(json.loads(path.read_text()))
        costs = [r["best_cost"] for r in runs]
        bks = bks_of(name)
        best = min(costs)
        best_run = min(runs, key=lambda r: r["best_cost"])
        rows.append({
            "instance": name,
            "bks": bks,
            "best": best,
            "gap_best": round(gap_pct(best, bks), 2),
            "mean": round(statistics.mean(costs), 1),
            "gap_mean": round(gap_pct(statistics.mean(costs), bks), 2),
            "std": round(statistics.stdev(costs), 1),
            "iter_mean": round(statistics.mean(r["gen_of_best"] for r in runs), 1),
            "fe_best_mean": round(statistics.mean(r["fe_of_best"] for r in runs)),
            "n_routes_best": best_run["best_n_routes"],
            "k": int(name.split("-k")[1]),
            "satisf": "100%",
        })
    return rows


def save_csv(rows):
    out = RESULTS / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"salvato: {out}")


def save_latex(rows):
    """Tabella booktabs con le metriche richieste dalla consegna + BKS/gap."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Risultati su 5 run indipendenti per istanza "
        r"(FE $= 3.5\times10^5$; iterazioni $=$ generazioni alla best; "
        r"tutte le citt\`a servite in ogni run).}",
        r"\label{tab:risultati}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Istanza & BKS & best & gap\% & mean & dev.\ std & "
        r"iter.\ medie & rotte/$k$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['instance']} & {r['bks']} & {r['best']} & "
            f"{r['gap_best']:.2f} & {r['mean']:.1f} & {r['std']:.1f} & "
            f"{r['iter_mean']:.1f} & {r['n_routes_best']}/{r['k']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = RESULTS / "summary_table.tex"
    out.write_text("\n".join(lines))
    print(f"salvato: {out}")


def print_table(rows):
    head = (f"{'istanza':<12} {'BKS':>5} {'best':>5} {'gap%':>6} "
            f"{'mean':>7} {'std':>6} {'iter.med':>8} {'rotte/k':>7}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['instance']:<12} {r['bks']:>5} {r['best']:>5} "
              f"{r['gap_best']:>6.2f} {r['mean']:>7.1f} {r['std']:>6.1f} "
              f"{r['iter_mean']:>8.1f} {str(r['n_routes_best'])+'/'+str(r['k']):>7}")
    gaps = [r["gap_best"] for r in rows]
    gaps_m = [r["gap_mean"] for r in rows]
    print("-" * len(head))
    print(f"gap medio dei best: {sum(gaps)/len(gaps):.2f}%   "
          f"gap medio delle mean: {sum(gaps_m)/len(gaps_m):.2f}%")


def main():
    rows = aggregate()
    save_csv(rows)
    save_latex(rows)
    print_table(rows)
    operator_activity()


# ---------------------------------------------------------------------------
# Attivita' degli operatori (diagnostica per l'analisi critica in relazione)
# ---------------------------------------------------------------------------

def operator_activity():
    """Media sulle 5 run, per istanza, dei contatori di attivita' degli
    operatori immunologici, ovvero quante volte ciascun meccanismo si attiva
    realmente durante una run. Serve a sostanziare con dati (e non con
    affermazioni) il ruolo di aging, soppressione dei duplicati, ricerca
    locale e gestione del vincolo di flotta."""
    rows = []
    for name in INSTANCES:
        runs = [json.loads((RUNS_DIR / f"{name}_seed{s}.json").read_text())
                for s in FINAL_SEEDS]
        if "stats" not in runs[0]:
            print("le run non contengono le statistiche degli operatori: "
                  "rieseguire la campagna (python -m experiments.run_all --force)")
            return []
        st = [r["stats"] for r in runs]
        m = lambda key: statistics.mean(s[key] for s in st)
        rows.append({
            "instance": name,
            "gen": round(statistics.mean(r["generations"] for r in runs), 1),
            "aging_deaths": round(m("aging_deaths"), 1),
            "births": round(m("births"), 1),
            "dupes_demoted": round(m("dupes_demoted"), 1),
            "ls_calls": round(m("ls_calls"), 1),
            "ls_improvements": round(m("ls_improvements"), 1),
            "nonconforming_pct": round(
                100 * m("clones_nonconforming") / max(1, m("clones_total")), 1),
        })

    out = RESULTS / "operator_activity.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsalvato: {out}")

    head = (f"{'istanza':<12} {'gen':>5} {'morti aging':>11} {'nascite':>7} "
            f"{'dupl.sopp.':>10} {'LS ok/tot':>10} {'cloni >k rotte':>14}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        ls = f"{r['ls_improvements']:.0f}/{r['ls_calls']:.0f}"
        print(f"{r['instance']:<12} {r['gen']:>5.1f} {r['aging_deaths']:>11.1f} "
              f"{r['births']:>7.1f} {r['dupes_demoted']:>10.1f} {ls:>10} "
              f"{str(r['nonconforming_pct'])+'%':>14}")
    return rows


if __name__ == "__main__":
    main()
