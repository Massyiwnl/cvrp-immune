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
- `tests/` — test di accettazione per ogni fase
- `experiments/`, `results/`, `plots/`, `report/` — fasi successive

## Stato

- [x] Fase 0 — setup repo + istanze
- [x] Fase 1 — parser + matrice distanze (Test A verde su 10/10)
- [ ] Fase 2 — decoder Split (programmazione dinamica)
- [ ] Fase 3 — contatore FE
- [ ] Fase 4 — ricerca locale multi-vicinato
- [ ] Fase 5 — motore immunologico (selezione clonale)
- [ ] Fasi 6-10 — tuning, esperimenti, analisi, relazione, consegna
