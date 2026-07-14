"""
expcommon.py — Costanti e helper condivisi dagli script sperimentali
(tuning, campagna completa, aggregazione).

I BKS non sono hardcodati: vengono letti dalla riga "Cost" dei file .sol
ufficiali di CVRPLIB presenti in data/ (fonte autorevole, gia' validata
dal Test A della Fase 1).
"""
from __future__ import annotations

from pathlib import Path

from src.instance import load_solution

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

INSTANCES = [
    "A-n45-k7", "A-n60-k9", "A-n80-k10",
    "B-n56-k7", "B-n66-k9", "B-n78-k10",
    "E-n76-k8", "E-n101-k14",
    "P-n50-k10", "P-n101-k4",
]

# Seed della campagna finale (5 run per istanza, come da consegna)
FINAL_SEEDS = [0, 1, 2, 3, 4]

# Seed usati SOLO per il tuning: disgiunti da quelli finali, per non
# tarare i parametri sugli stessi campioni casuali degli esperimenti.
TUNING_SEEDS = [100, 101, 102]


def bks_of(name: str) -> int:
    """Costo della best-known solution, letto dal .sol ufficiale."""
    _routes, cost = load_solution(DATA / f"{name}.sol")
    return cost


def gap_pct(cost: int, bks: int) -> float:
    return 100.0 * (cost - bks) / bks
