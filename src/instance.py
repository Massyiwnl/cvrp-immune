"""
instance.py — Parser di istanze CVRP in formato TSPLIB/CVRPLIB e
costruzione della matrice delle distanze EUC_2D.

Convenzioni interne del progetto
--------------------------------
* I nodi vengono re-indicizzati:  depot = 0,  clienti = 1 .. n-1
  (nei file .vrp dei set A/B/E/P i nodi sono numerati 1 .. n e il
  depot e' sempre il nodo 1: indice interno = id del file - 1).

* Le distanze seguono la convenzione TSPLIB "EUC_2D":

      d(i, j) = nint( sqrt( (xi-xj)^2 + (yi-yj)^2 ) )

  dove nint(x) = int(x + 0.5), ovvero arrotondamento all'intero piu'
  vicino con il caso .5 arrotondato verso l'alto.

  ATTENZIONE: round() di Python e numpy.round usano il "banker's
  rounding" (round-half-to-even: round(2.5) == 2), che su alcune coppie
  di punti produce un valore diverso da nint e rende i costi NON
  confrontabili con le best-known solutions di CVRPLIB. Per questo qui
  si usa sempre floor(x + 0.5).

* Una "rotta" e' una sequenza di clienti (indici interni) SENZA il depot
  agli estremi: il costo della rotta include implicitamente gli archi
  depot -> primo cliente e ultimo cliente -> depot.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Arrotondamento TSPLIB
# ---------------------------------------------------------------------------

def _nint(x: float) -> int:
    """Arrotondamento all'intero piu' vicino secondo TSPLIB (half-up)."""
    return int(x + 0.5)


# ---------------------------------------------------------------------------
# Struttura dati dell'istanza
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    name: str            # nome dell'istanza (es. "A-n45-k7")
    n: int               # numero totale di nodi, depot incluso
    capacity: int        # capacita' Q di ogni veicolo
    k: int | None        # numero di veicoli indicato nel nome (…-k<numero>)
    coords: np.ndarray   # shape (n, 2), float64 — coords[0] = depot
    demands: np.ndarray  # shape (n,), int64 — demands[0] = 0
    D: np.ndarray        # shape (n, n), int64 — distanze EUC_2D arrotondate

    # ---- proprieta' di comodo -------------------------------------------

    @property
    def n_customers(self) -> int:
        return self.n - 1

    @property
    def total_demand(self) -> int:
        return int(self.demands.sum())

    @property
    def min_vehicles(self) -> int:
        """Lower bound banale sul numero di veicoli: ceil(domanda_tot / Q)."""
        return math.ceil(self.total_demand / self.capacity)

    # ---- valutazione di rotte e soluzioni --------------------------------

    def route_cost(self, route) -> int:
        """Costo di una rotta (lista di clienti interni, senza depot)."""
        if len(route) == 0:
            return 0
        c = int(self.D[0, route[0]]) + int(self.D[route[-1], 0])
        for a, b in zip(route[:-1], route[1:]):
            c += int(self.D[a, b])
        return c

    def route_load(self, route) -> int:
        """Carico totale di una rotta."""
        return int(self.demands[list(route)].sum()) if len(route) else 0

    def solution_cost(self, routes) -> int:
        """Costo totale di una soluzione (lista di rotte)."""
        return sum(self.route_cost(r) for r in routes)

    def is_feasible(self, routes, max_routes: int | None = None):
        """Verifica di ammissibilita' di una soluzione.

        Controlla che: (1) ogni cliente 1..n-1 sia visitato esattamente
        una volta; (2) il carico di ogni rotta non superi Q; (3) se
        max_routes e' indicato, il numero di rotte non vuote non lo superi.

        Ritorna (True, "ok") oppure (False, <motivo>).
        """
        visited = sorted(c for r in routes for c in r)
        if visited != list(range(1, self.n)):
            return False, "clienti mancanti, duplicati o indici non validi"
        for idx, r in enumerate(routes):
            load = self.route_load(r)
            if load > self.capacity:
                return False, (f"capacita' violata nella rotta {idx}: "
                               f"{load} > {self.capacity}")
        if max_routes is not None:
            n_used = sum(1 for r in routes if len(r) > 0)
            if n_used > max_routes:
                return False, f"troppe rotte: {n_used} > {max_routes}"
        return True, "ok"


# ---------------------------------------------------------------------------
# Parser del file .vrp (formato TSPLIB)
# ---------------------------------------------------------------------------

def load_instance(path) -> Instance:
    """Legge un file .vrp (TSPLIB, EDGE_WEIGHT_TYPE = EUC_2D) e restituisce
    un oggetto Instance con la matrice delle distanze gia' precalcolata."""
    path = Path(path)
    header: dict[str, str] = {}
    coords_raw: dict[int, tuple[float, float]] = {}
    demands_raw: dict[int, int] = {}
    depot_ids: list[int] = []
    section = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()

        # cambi di sezione
        if upper.startswith("NODE_COORD_SECTION"):
            section = "coords";  continue
        if upper.startswith("DEMAND_SECTION"):
            section = "demand";  continue
        if upper.startswith("DEPOT_SECTION"):
            section = "depot";   continue
        if upper.startswith("EOF"):
            break

        if section is None:
            # riga di intestazione "CHIAVE : valore" (lo split avviene solo
            # sul primo ':' cosi' il COMMENT puo' contenere altri due punti)
            if ":" in line:
                key, val = line.split(":", 1)
                header[key.strip().upper()] = val.strip()
            continue

        parts = line.split()
        if section == "coords":
            i = int(parts[0])
            coords_raw[i] = (float(parts[1]), float(parts[2]))
        elif section == "demand":
            i = int(parts[0])
            demands_raw[i] = int(float(parts[1]))
        elif section == "depot":
            v = int(float(parts[0]))
            if v == -1:
                section = None
            else:
                depot_ids.append(v)

    # ---- controlli sul formato -------------------------------------------
    n = int(header["DIMENSION"])
    capacity = int(float(header["CAPACITY"]))
    ewt = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").strip().upper()
    if ewt != "EUC_2D":
        raise ValueError(f"{path.name}: EDGE_WEIGHT_TYPE '{ewt}' non "
                         f"supportato (atteso EUC_2D)")
    if sorted(coords_raw) != list(range(1, n + 1)):
        raise ValueError(f"{path.name}: id dei nodi non contigui 1..{n}")
    if sorted(demands_raw) != list(range(1, n + 1)):
        raise ValueError(f"{path.name}: DEMAND_SECTION incompleta")
    if depot_ids != [1]:
        raise ValueError(f"{path.name}: atteso depot = nodo 1 "
                         f"(convenzione dei set A/B/E/P), trovato {depot_ids}")

    # ---- costruzione array interni (indice interno = id file - 1) --------
    coords = np.array([coords_raw[i] for i in range(1, n + 1)],
                      dtype=np.float64)
    demands = np.array([demands_raw[i] for i in range(1, n + 1)],
                       dtype=np.int64)
    if demands[0] != 0:
        raise ValueError(f"{path.name}: la domanda del depot deve essere 0")

    # ---- matrice distanze EUC_2D (vettorizzata) ---------------------------
    # nint(x) = floor(x + 0.5) per x >= 0: identico a int(x + 0.5) ed evita
    # il banker's rounding di np.round.
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    D = np.floor(np.sqrt(dx * dx + dy * dy) + 0.5).astype(np.int64)

    # ---- numero di veicoli dal nome (…-k<numero>) -------------------------
    name = header.get("NAME", path.stem)
    m = re.search(r"-k(\d+)", name) or re.search(r"-k(\d+)", path.stem)
    k = int(m.group(1)) if m else None

    return Instance(name=name, n=n, capacity=capacity, k=k,
                    coords=coords, demands=demands, D=D)


# ---------------------------------------------------------------------------
# Lettura dei file .sol di CVRPLIB
# ---------------------------------------------------------------------------

def load_solution(path):
    """Legge un file .sol di CVRPLIB.

    Ritorna (routes, cost) dove `routes` sono le rotte con la numerazione
    ORIGINALE del file (da convertire con sol_to_internal_candidates) e
    `cost` e' il costo dichiarato nella riga "Cost <valore>".
    """
    routes: list[list[int]] = []
    cost = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("route"):
            right = line.split(":", 1)[1]
            routes.append([int(t) for t in right.split()])
        else:
            m = re.match(r"(?i)^cost\s*:?\s*([\d.]+)", line)
            if m:
                cost = int(float(m.group(1)))
    if cost is None or not routes:
        raise ValueError(f"file .sol non riconosciuto: {path}")
    return routes, cost


def sol_to_internal_candidates(routes, n):
    """Converte le rotte di un .sol in indici interni (clienti 1..n-1).

    A seconda della fonte, i file .sol possono numerare i clienti in tre
    modi diversi. Vengono restituite tutte le interpretazioni compatibili
    con il range degli indici osservati; quella corretta viene identificata
    dal chiamante confrontando il costo ricalcolato con quello dichiarato
    (verifica empirica che elimina ogni ambiguita').

    Ritorna una lista di tuple (etichetta, rotte_interne).
    """
    ids = [j for r in routes for j in r]
    lo, hi = min(ids), max(ids)
    out = []
    # (a) convenzione CVRPLIB: cliente j del .sol = nodo j+1 del .vrp
    #     -> indice interno j          (indici del .sol in 1..n-1)
    if lo >= 1 and hi <= n - 1:
        out.append(("cvrplib: j -> j", [list(r) for r in routes]))
    # (b) id del .vrp usati direttamente: cliente j = nodo j del .vrp
    #     -> indice interno j-1        (indici del .sol in 2..n)
    if lo >= 2 and hi <= n:
        out.append(("id .vrp: j -> j-1", [[j - 1 for j in r] for r in routes]))
    # (c) numerazione 0-based dei clienti: cliente j -> indice interno j+1
    if lo >= 0 and hi <= n - 2:
        out.append(("0-based: j -> j+1", [[j + 1 for j in r] for r in routes]))
    return out
