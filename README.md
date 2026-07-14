# cvrp-immune — CVRP con Algoritmo Immunologico a Selezione Clonale + Ricerca Locale

Progetto per il corso *Heuristics & Metaheuristics for Optimization & Learning*
(Prof. M. Pavone) — Capacitated Vehicle Routing Problem su istanze CVRPLIB
(set A, B, E, P).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Esecuzione di una run

```bash
python -m src.main --instance data/A-n45-k7.vrp --seed 0
```

Budget di default: FE = 350000 (consegna). Risultato salvato in `results/<istanza>_seed<seed>.json`
con best, rotte, curva di convergenza e parametri. Nota: Numba era previsto per i punti caldi,
ma le misure hanno mostrato che Python puro completa una run in 2-4 s — dipendenza rimossa.

## Test

```bash
pytest -v
```

Il test piu' importante (`test_A_riproduzione_costo_sol`) verifica che la
matrice delle distanze EUC_2D riproduca **esattamente** i costi delle
soluzioni ottime ufficiali di CVRPLIB: garantisce che tutti i risultati
del progetto siano confrontabili con le best-known solutions.

## Dati

`data/` contiene le 10 istanze richieste dalla consegna (.vrp) e le
rispettive soluzioni ottime (.sol), scaricate dal sito ufficiale CVRPLIB.
Per ri-scaricarle: `python data/download_instances.py`.

## Struttura

- `src/instance.py` — parser TSPLIB + matrice distanze EUC_2D (nint)
- `src/split.py` — decoder Split (DP): partizione ottima del giant tour in rotte
- `src/budget.py` — contatore FE (criterio di arresto) + log di convergenza
- `src/greedy.py` — euristica costruttiva Clarke-Wright (savings)
- `src/local_search.py` — discesa first-improvement a 5 vicinati con liste di candidati
- `src/operators.py` — ipermutazione inversamente proporzionale alla fitness
- `src/immune.py` — motore a selezione clonale (il cuore del progetto)
- `src/main.py` — runner CLI: `python -m src.main --instance data/A-n45-k7.vrp --seed 0`
- `experiments/tuning.py` — harness di tuning Fase 6: `python -m experiments.tuning --round {1,2,3}`
- `experiments/run_all.py` — campagna completa 10×5: `python -m experiments.run_all`
- `experiments/validate.py` — validatore indipendente delle 50 best: `python -m experiments.validate`
- `experiments/aggregate.py` — tabella finale (CSV + LaTeX) + attività operatori
- `experiments/plots.py` — grafici di convergenza (PDF per LaTeX)
- `experiments/plot_solutions.py` — diagrammi delle soluzioni sul piano (trovata vs ottimo)
- `experiments/compliance.py` — audit di conformità alla consegna (17 controlli)
- `experiments/gap_analysis.py` — scomposizione del gap: ordinamento (TSP intra-rotta,
  esatto via Held-Karp) vs assegnamento (partizione). Risultato: 100% assegnamento,
  0% ordinamento — ogni rotta prodotta è TSP-ottima.
- `tests/` — test di accettazione per ogni fase (`tests/common.py`: helper condivisi)
- `results/` — JSON per-run, tabelle aggregate, CSV del tuning
- `report/figures/` — tutte le figure (PDF per LaTeX + PNG di anteprima)

## Stato

- [x] Fase 0 — setup repo + istanze
- [x] Fase 1 — parser + matrice distanze (Test A verde su 10/10)
- [x] Fase 2 — decoder Split, varianti illimitata e limitata a m rotte (brute-force test verde)
- [x] Fase 3 — contatore FE (arresto esatto a 3.5e5), convenzione di conteggio, log di convergenza
- [x] Fase 4 — ricerca locale a 5 vicinati (2-opt, Or-opt 1-3, swap, 2-opt*) + Clarke-Wright
- [x] Fase 5 — motore immunologico (cloning, ipermutazione, aging elitista, selezione (μ+λ) con soppressione duplicati, LS lamarckiana)
- [x] Fase 6 — tuning (coordinate descent, 2 round + interazione): parametri congelati
      B=15, dup=3, ρ=ln(n), τ_B=10, K=2, cap_LS=5000, h=20, greedy=25%
      (gap medio sul set di tuning: 5.57% → 2.66%; dettagli in results/tuning_round*.csv)
- [x] Fase 6-bis — ri-validazione del tuning sul motore FINALE (Round 4): dopo le modifiche
      di fitness/selezione della Fase 7 la scelta va confermata sull'algoritmo definitivo.
      Vincitore: K=3, cap=5000, dup=3, τ_B=20 (2.90% sui seed di tuning). Le configurazioni di
      frontiera sono statisticamente indistinguibili: si mantiene la scelta fatta sui seed di
      tuning (scegliere sui seed finali sarebbe selezione sul test set).
- [x] Fase 7 — campagna completa 10×5 a FE=3.5e5: 50/50 run valide (validatore indipendente);
      gap medio dei best 1.82%, tutte le run con ≤ k rotte, 7 istanze su 10 sotto l'1.5%.
      Nota di design: fitness = decodifica vincolata a k rotte (split_fitness) + selezione
      lessicografica feasibility-first — necessari sulle istanze a capacità stretta (P-n50-k10).
- [x] Fase 8 — grafici di convergenza (4 istanze, una per set) in report/figures/:
      `python -m experiments.plots`
- [x] Audit di conformità — `python -m experiments.compliance`: verifica meccanica di tutti
      i requisiti sperimentali della consegna (17/17 PASS). Diagnostica di attività degli
      operatori in results/operator_activity.csv.
- [ ] Fase 9 — relazione LaTeX (4-12 pagine, pseudocodice + diagrammi, punti i-v)
- [ ] Fase 10 — consegna: sorgenti + relazione .pdf/.tex + figure → mario.pavone@unict.it
      entro le 16:00 del 22/07
