"""
run_all.py — Fase 7: campagna sperimentale completa richiesta dalla
consegna: 10 istanze x runs = 5 (seed 0..4), criterio di arresto
FE = 3.5e5, parametri CONGELATI dal tuning (i default di ImmuneParams).

Ogni run salva il proprio JSON in results/runs/<istanza>_seed<seed>.json
(best, rotte, curva di convergenza, generazione e FE del best, parametri):
sono i dati grezzi da cui validatore (validate.py), aggregatore
(aggregate.py) e grafici (Fase 8) derivano tutto il resto.

Uso (dalla radice della repo):

    python -m experiments.run_all              # salta le run gia' presenti
    python -m experiments.run_all --force      # riesegue tutto
    python -m experiments.run_all --workers 4

Le run gia' salvate vengono saltate (ripresa incrementale dopo
un'interruzione); --force le riesegue. Grazie ai seed fissi l'intera
campagna e' riproducibile bit a bit su qualunque macchina.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from experiments.expcommon import (DATA, FINAL_SEEDS, INSTANCES, RESULTS,
                                   bks_of, gap_pct)
from src.main import run_one

RUNS_DIR = RESULTS / "runs"


def _job_path(name: str, seed: int) -> Path:
    return RUNS_DIR / f"{name}_seed{seed}.json"


def _run_job(job):
    name, seed = job
    t0 = time.perf_counter()
    res = run_one(DATA / f"{name}.vrp", seed, out_dir=RUNS_DIR)
    wall = time.perf_counter() - t0
    gap = gap_pct(res["best_cost"], bks_of(name))
    gb = res["gen_of_best"]
    return (f"{name:<12} seed={seed}  best={res['best_cost']:>5}  "
            f"gap={gap:>5.2f}%  rotte={res['best_n_routes']}  "
            f"gen_best={str(gb) if gb is not None else '-':>3}  t={wall:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Campagna completa 10x5")
    ap.add_argument("--force", action="store_true",
                    help="riesegue anche le run gia' salvate")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(name, seed) for name in INSTANCES for seed in FINAL_SEEDS
            if args.force or not _job_path(name, seed).exists()]
    skipped = len(INSTANCES) * len(FINAL_SEEDS) - len(jobs)
    if skipped:
        print(f"{skipped} run gia' presenti, saltate (usa --force per rieseguirle)")
    if not jobs:
        print("nulla da fare")
        return

    workers = args.workers or min(cpu_count(), 8)
    print(f"{len(jobs)} run su {workers} processi (FE = 350000, seed fissi)...")
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        for line in pool.imap_unordered(_run_job, jobs):
            print(line, flush=True)
    print(f"\ncampagna completata in {time.perf_counter() - t0:.0f}s "
          f"-> {RUNS_DIR}/")


if __name__ == "__main__":       # obbligatorio per multiprocessing su Windows
    main()
