"""
main.py — Runner da riga di comando: una run dell'algoritmo immunologico
su un'istanza con un seed, con salvataggio del risultato in JSON.

Uso (dalla radice della repo):

    python -m src.main --instance data/A-n45-k7.vrp --seed 0

Parametri opzionali per sovrascrivere gli iperparametri di default (vedi
ImmuneParams): --pop, --dup, --rho, --tau, --ls-clones, --ls-cap, --h,
--greedy-frac; --fe cambia il budget (default: 350000, il valore della
consegna); --out-dir la cartella dei risultati (default: results/).

Il JSON per-run contiene tutto cio' che serve alle fasi successive:
best_cost, best_routes, best_n_routes, gen_of_best, fe_of_best,
generations, fe_used, wall_time, history (curva di convergenza) e i
parametri usati.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .budget import FE_MAX_DEFAULT
from .immune import ImmuneAlgorithm, ImmuneParams
from .instance import load_instance


def run_one(instance_path, seed: int, fe_max: int = FE_MAX_DEFAULT,
            params: ImmuneParams | None = None,
            out_dir: str | Path | None = "results") -> dict:
    """Esegue una run e (se out_dir non e' None) salva il JSON."""
    inst = load_instance(instance_path)
    algo = ImmuneAlgorithm(inst, params=params, seed=seed, fe_max=fe_max)
    result = algo.run()
    if out_dir is not None:
        out = Path(out_dir) / f"{inst.name}_seed{seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        result["saved_to"] = str(out)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Algoritmo immunologico a selezione clonale per il CVRP")
    ap.add_argument("--instance", required=True, help="file .vrp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fe", type=int, default=FE_MAX_DEFAULT)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--dup", type=int, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--tau", type=int, default=None)
    ap.add_argument("--ls-clones", type=int, default=None)
    ap.add_argument("--ls-cap", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--greedy-frac", type=float, default=None)
    args = ap.parse_args(argv)

    params = ImmuneParams()
    overrides = {"pop_size": args.pop, "dup": args.dup, "rho": args.rho,
                 "tau_b": args.tau, "ls_clones": args.ls_clones,
                 "ls_cap_fe": args.ls_cap, "h_neighbors": args.h,
                 "greedy_fraction": args.greedy_frac}
    for name, val in overrides.items():
        if val is not None:
            setattr(params, name, val)

    res = run_one(args.instance, args.seed, fe_max=args.fe,
                  params=params, out_dir=args.out_dir)
    print(f"{res['instance']}  seed={res['seed']}  "
          f"best={res['best_cost']}  rotte={res['best_n_routes']}  "
          f"gen_best={res['gen_of_best']}  fe_best={res['fe_of_best']}  "
          f"gen_tot={res['generations']}  fe={res['fe_used']}  "
          f"t={res['wall_time']:.1f}s  -> {res.get('saved_to', '(non salvato)')}")


if __name__ == "__main__":
    main()
