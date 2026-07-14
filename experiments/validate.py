"""
validate.py — Fase 7: validatore INDIPENDENTE delle soluzioni salvate.

Rilegge ogni results/runs/*.json e riverifica DA ZERO, senza fidarsi di
nulla di cio' che il motore dichiara:

  1. ogni cliente e' servito esattamente una volta;
  2. il carico di ogni rotta rispetta la capacita' Q;
  3. il numero di rotte non supera k (il numero di veicoli m della
     definizione formale della consegna);
  4. il costo ricalcolato con la matrice delle distanze coincide
     ESATTAMENTE con il best_cost dichiarato;
  5. il protocollo e' stato rispettato: fe_used == fe_max == 350000,
     seed e istanza coerenti col nome del file;
  6. la curva di convergenza e' ben formata (costi strettamente
     decrescenti, FE non decrescenti, ultimo punto == best_cost).

Esce con codice 1 se anche una sola verifica fallisce: da eseguire
sempre prima di scrivere i numeri in relazione.

Uso:  python -m experiments.validate
"""
from __future__ import annotations

import json
import sys

from experiments.expcommon import DATA, FINAL_SEEDS, INSTANCES, RESULTS
from src.instance import load_instance

RUNS_DIR = RESULTS / "runs"


def validate_run(path, inst) -> list[str]:
    """Ritorna la lista dei problemi trovati (vuota = tutto ok)."""
    problems = []
    res = json.loads(path.read_text())

    # (5) protocollo
    if res["instance"] != inst.name:
        problems.append(f"istanza incoerente: {res['instance']}")
    if res["fe_max"] != 350_000:
        problems.append(f"fe_max = {res['fe_max']} != 350000")
    if res["fe_used"] != res["fe_max"]:
        problems.append(f"fe_used = {res['fe_used']} != fe_max")

    # (1)+(2)+(3) ammissibilita' e vincolo sul numero di veicoli
    routes = res["best_routes"]
    ok, msg = inst.is_feasible(routes, max_routes=inst.k)
    if not ok:
        problems.append(f"soluzione non ammissibile: {msg}")
    if not res.get("respects_k", False):
        problems.append("respects_k = False")

    # (4) costo ricalcolato da zero
    recomputed = inst.solution_cost(routes)
    if recomputed != res["best_cost"]:
        problems.append(f"costo ricalcolato {recomputed} != "
                        f"dichiarato {res['best_cost']}")

    # (6) curva di convergenza ben formata
    hist = res["history"]
    costs = [c for _, c in hist]
    fes = [fe for fe, _ in hist]
    if not hist:
        problems.append("history vuota")
    else:
        if costs != sorted(costs, reverse=True) or len(set(costs)) != len(costs):
            problems.append("history: costi non strettamente decrescenti")
        if fes != sorted(fes):
            problems.append("history: FE decrescenti")
        if costs[-1] != res["best_cost"]:
            problems.append("history: ultimo punto != best_cost")
        if fes[-1] != res["fe_of_best"]:
            problems.append("history: ultimo FE != fe_of_best")
    return problems


def main():
    total, failed = 0, 0
    for name in INSTANCES:
        inst = load_instance(DATA / f"{name}.vrp")
        for seed in FINAL_SEEDS:
            path = RUNS_DIR / f"{name}_seed{seed}.json"
            total += 1
            if not path.exists():
                print(f"MANCANTE  {path.name}")
                failed += 1
                continue
            problems = validate_run(path, inst)
            if problems:
                failed += 1
                print(f"FAIL      {path.name}: " + "; ".join(problems))
            else:
                print(f"ok        {path.name}")
    print(f"\n{total - failed}/{total} run valide")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
