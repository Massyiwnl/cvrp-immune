"""
immune.py — Algoritmo immunologico a selezione clonale ibridato con
ricerca locale (opzione 2 della consegna).
Argomenti del corso: sistemi immunitari artificiali + hybrid metaheuristics.

Pipeline di una generazione
---------------------------
1. CLONING statico: ogni anticorpo produce `dup` cloni.
2. IPERMUTAZIONE inversamente proporzionale alla fitness (operators.py):
   i migliori mutano poco (exploitation), i peggiori molto (exploration).
3. VALUTAZIONE dei cloni: decodifica Split del giant tour (1 FE l'una).
4. RICERCA LOCALE lamarckiana sui migliori `ls_clones` cloni della
   generazione, con cap di FE per invocazione: la soluzione migliorata
   viene riscritta nel genotipo concatenando le rotte nel giant tour
   (apprendimento lamarckiano: il fenotipo appreso rientra nel genotipo).
5. AGING: l'eta' di tutti aumenta di 1; il clone eredita l'eta' del
   genitore ma si azzera se lo ha migliorato (protegge le linee
   promettenti); chi supera tau_b viene rimosso. Aging ELITISTA: il
   miglior anticorpo corrente non muore mai.
6. SELEZIONE (mu+lambda): dai sopravvissuti (genitori + cloni) passano i
   migliori pop_size; se sono meno di pop_size, NASCITE di anticorpi
   casuali (reintroduzione di diversita').

Inizializzazione: una frazione `greedy_fraction` della popolazione e'
seminata da Clarke-Wright (il primo individuo e' CW puro, gli altri sue
perturbazioni); il resto sono permutazioni casuali. Anche le valutazioni
dell'inizializzazione consumano FE (generazione 0).

Vincolo sul numero di veicoli (definizione formale: m = k)
----------------------------------------------------------
La fitness di ricerca e' lo Split ILLIMITATO (veloce; le soluzioni
intermedie con piu' di k rotte sono trampolini legittimi). Il BEST
GLOBALE riportato, pero', accetta solo candidati con al piu' k rotte:
curva di convergenza e risultati finali rispettano sempre la flotta m.
Empiricamente (Fase 2) all'ottimo lo Split illimitato usa esattamente k
rotte su tutte le 10 istanze, quindi il vincolo non penalizza la
ricerca; in mancanza di candidati conformi (mai osservato) il risultato
segnala respects_k = False riportando il migliore assoluto.

Riproducibilita': un unico generatore numpy.random.default_rng(seed)
alimenta inizializzazione, ipermutazione e ricerca locale; a parita' di
seed la run e' identica bit a bit.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .budget import (FE_MAX_DEFAULT, Budget, BudgetExhausted, ConvergenceLog,
                     evaluate_tour)
from .greedy import clarke_wright
from .instance import Instance
from .local_search import LocalSearch
from .operators import (DEFAULT_WEIGHTS, mutate, mutations_count,
                        normalized_fitness)
from .split import tour_from_routes


@dataclass
class ImmuneParams:
    """Iperparametri dell'algoritmo (valori di partenza; tuning in Fase 6)."""
    pop_size: int = 15               # B: dimensione della popolazione
    dup: int = 2                     # cloni per anticorpo
    rho: float | None = None         # legge di ipermutazione (None -> ln(nc))
    tau_b: int = 20                  # eta' massima (aging)
    ls_clones: int = 3               # K: migliori cloni sottoposti a LS per generazione
    ls_cap_fe: int = 2000            # cap di FE per invocazione di LS
    h_neighbors: int = 20            # ampiezza liste di candidati della LS
    greedy_fraction: float = 0.25    # frazione di popolazione seminata da CW
    mut_weights: tuple = DEFAULT_WEIGHTS


@dataclass
class Antibody:
    tour: np.ndarray                 # genotipo: giant tour
    cost: int                        # costo della decodifica Split (fitness)
    n_routes: int
    routes: list                     # fenotipo: rotte decodificate
    age: int = 0


class ImmuneAlgorithm:
    """Una run completa dell'algoritmo su un'istanza, con un seed."""

    def __init__(self, inst: Instance, params: ImmuneParams | None = None,
                 seed: int = 0, fe_max: int = FE_MAX_DEFAULT):
        self.inst = inst
        self.params = params if params is not None else ImmuneParams()
        self.seed = int(seed)
        self.fe_max = int(fe_max)
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------

    def run(self) -> dict:
        inst, P = self.inst, self.params
        nc = inst.n_customers
        kmax = inst.k if inst.k is not None else nc
        rho = P.rho if P.rho is not None else math.log(nc)
        customers = np.arange(1, inst.n, dtype=np.int64)

        budget = Budget(self.fe_max)
        log = ConvergenceLog(budget)
        searcher = LocalSearch(inst, budget, h=P.h_neighbors, rng=self.rng)

        # best riportato (<= k rotte) e migliore assoluto (fallback)
        best = {"routes": None, "n_routes": None}
        anybest = {"cost": None, "routes": None, "n_routes": None}

        def consider(cost, routes, n_routes, gen):
            """Candida una soluzione appena valutata al best globale."""
            if anybest["cost"] is None or cost < anybest["cost"]:
                anybest.update(cost=int(cost),
                               routes=[list(r) for r in routes],
                               n_routes=int(n_routes))
            if n_routes <= kmax and log.update(cost, generation=gen):
                best["routes"] = [list(r) for r in routes]
                best["n_routes"] = int(n_routes)

        def new_antibody(tour, gen, age=0):
            """Valuta un tour (1 FE, con controllo PREVENTIVO del budget)."""
            budget.checkpoint()
            res = evaluate_tour(tour, inst, budget)
            consider(res.cost, res.routes, res.n_routes, gen)
            return Antibody(tour=tour, cost=res.cost, n_routes=res.n_routes,
                            routes=res.routes, age=age)

        gen = 0
        t0 = time.perf_counter()
        pop: list[Antibody] = []
        try:
            # ---- inizializzazione (generazione 0) ----------------------
            n_greedy = max(1, round(P.pop_size * P.greedy_fraction))
            cw_tour = tour_from_routes(clarke_wright(inst))
            m_pert = max(2, nc // 10)            # entita' delle perturbazioni
            for i in range(n_greedy):
                t = cw_tour.copy() if i == 0 else \
                    mutate(cw_tour, m_pert, self.rng, P.mut_weights)
                pop.append(new_antibody(t, gen))
            while len(pop) < P.pop_size:
                pop.append(new_antibody(self.rng.permutation(customers), gen))

            # ---- ciclo generazionale -----------------------------------
            while not budget.exhausted:
                gen += 1

                # (1)+(2)+(3) cloning, ipermutazione, valutazione
                fhat = normalized_fitness(np.array([a.cost for a in pop]))
                clones: list[tuple[Antibody, int]] = []   # (clone, costo genitore)
                for parent, f in zip(pop, fhat):
                    m_mut = mutations_count(float(f), nc, rho)
                    for _ in range(P.dup):
                        t = mutate(parent.tour, m_mut, self.rng, P.mut_weights)
                        clones.append((new_antibody(t, gen, age=parent.age),
                                       parent.cost))

                # (4) ricerca locale lamarckiana sui migliori K cloni
                clones.sort(key=lambda pc: pc[0].cost)
                for c, _ in clones[:P.ls_clones]:
                    if budget.exhausted:
                        break
                    routes, cost = searcher.improve(c.routes, c.cost,
                                                    max_fe=P.ls_cap_fe)
                    if cost < c.cost:
                        c.routes = routes
                        c.cost = int(cost)
                        c.n_routes = len(routes)
                        c.tour = tour_from_routes(routes)   # writeback
                        consider(c.cost, routes, c.n_routes, gen)

                # reset dell'eta' per i cloni che hanno migliorato il genitore
                for c, parent_cost in clones:
                    if c.cost < parent_cost:
                        c.age = 0

                # (5) invecchiamento + rimozione per vecchiaia (elitista)
                merged = pop + [c for c, _ in clones]
                for a in merged:
                    a.age += 1
                best_ab = min(merged, key=lambda a: a.cost)
                survivors = [a for a in merged
                             if a.age <= P.tau_b or a is best_ab]

                # (6) selezione (mu+lambda) con SOPPRESSIONE DEI DUPLICATI:
                # a parita' di costo passa un solo anticorpo; i duplicati
                # rientrano solo se mancano soluzioni distinte. Senza questa
                # regola la popolazione collassa su copie dello stesso ottimo
                # locale, la fitness normalizzata degenera (tutti f_hat = 1),
                # l'ipermutazione si riduce al minimo per tutti e la ricerca
                # stagna (convergenza prematura osservata empiricamente).
                survivors.sort(key=lambda a: a.cost)
                pop, seen, dupes = [], set(), []
                for a in survivors:
                    if len(pop) >= P.pop_size:
                        break
                    if a.cost in seen:
                        dupes.append(a)
                    else:
                        seen.add(a.cost)
                        pop.append(a)
                for a in dupes:
                    if len(pop) >= P.pop_size:
                        break
                    pop.append(a)
                # nascite: reintroduzione di diversita' se sotto organico
                while len(pop) < P.pop_size and not budget.exhausted:
                    pop.append(new_antibody(self.rng.permutation(customers),
                                            gen, age=0))
        except BudgetExhausted:
            pass                                  # arresto esatto al tetto
        wall = time.perf_counter() - t0

        # ---- risultato + validazione finale indipendente ----------------
        respects_k = best["routes"] is not None
        out_routes = best["routes"] if respects_k else anybest["routes"]
        out_cost = log.best_cost if respects_k else anybest["cost"]
        out_nroutes = best["n_routes"] if respects_k else anybest["n_routes"]

        feas, msg = inst.is_feasible(out_routes,
                                     max_routes=kmax if respects_k else None)
        if not feas or inst.solution_cost(out_routes) != out_cost:
            raise RuntimeError(f"incoerenza interna sul best finale: {msg}")

        return {
            "instance": inst.name,
            "seed": self.seed,
            "fe_max": self.fe_max,
            "params": asdict(P),
            "best_cost": out_cost,
            "best_n_routes": out_nroutes,
            "best_routes": out_routes,
            "respects_k": respects_k,
            "gen_of_best": log.gen_of_best,
            "fe_of_best": log.fe_of_best,
            "generations": gen,
            "fe_used": budget.used,
            "wall_time": wall,
            "history": [[fe, c] for fe, c in log.history],
        }
