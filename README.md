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
- `tests/` — test di accettazione per ogni fase (`tests/common.py`: helper condivisi)
- `experiments/`, `results/`, `plots/`, `report/` — fasi successive

## Stato

- [x] Fase 0 — setup repo + istanze
- [x] Fase 1 — parser + matrice distanze (Test A verde su 10/10)
- [x] Fase 2 — decoder Split, varianti illimitata e limitata a m rotte (brute-force test verde)
- [x] Fase 3 — contatore FE (arresto esatto a 3.5e5), convenzione di conteggio, log di convergenza
- [x] Fase 4 — ricerca locale a 5 vicinati (2-opt, Or-opt 1-3, swap, 2-opt*) + Clarke-Wright
- [x] Fase 5 — motore immunologico (cloning, ipermutazione, aging elitista, selezione (μ+λ) con soppressione duplicati, LS lamarckiana)
- [ ] Fasi 6-10 — tuning, esperimenti, analisi, relazione, consegna
