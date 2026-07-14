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
Sulle istanze a capacita' strettissima (P-n50-k10: riempimento 95%) lo
Split illimitato puo' non produrre MAI partizioni con <= k rotte, perche'
minimizza il costo e non il numero di rotte: per questo, quando un
candidato con piu' di k rotte migliora il best corrente, si tenta la
decodifica vincolata split_limited sul suo tour (lazy, 1 FE; il filtro
e' esatto perche' l'illimitato e' un lower bound del limitato). In
mancanza assoluta di candidati conformi il risultato segnala
respects_k = False riportando il migliore assoluto.

Riproducibilita': un unico generatore numpy.random.default_rng(seed)
alimenta inizializzazione, ipermutazione e ricerca locale; a parita' di
seed la run e' identica bit a bit.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .budget import FE_MAX_DEFAULT, Budget, BudgetExhausted, ConvergenceLog
from .greedy import clarke_wright
from .instance import Instance
from .local_search import LocalSearch
from .operators import (DEFAULT_WEIGHTS, mutate, mutations_count,
                        normalized_fitness)
from .split import split_fitness, split_limited, tour_from_routes


@dataclass
class ImmuneParams:
    """Iperparametri dell'algoritmo.

    I default sono la configurazione CONGELATA dal tuning (Fase 6,
    Round 1-3: coordinate descent su 4 istanze eterogenee x 3 seed
    disgiunti da quelli finali) e RI-VALIDATA sul motore definitivo
    (Round 4) dopo le modifiche di fitness e selezione introdotte in
    Fase 7; dettagli in results/tuning_round*.csv. Sul motore finale:
    K=3/cap=5000/dup=3/tau_b=20 -> 2.90% di gap medio sul set di
    tuning; il tau_b=10 emerso sul motore della Fase 5 peggiora qui
    (3.27-3.40%): con la selezione feasibility-first le linee conformi
    maturano piu' lentamente e l'aging aggressivo le rimuove troppo
    presto."""
    pop_size: int = 15               # B: dimensione della popolazione
    dup: int = 3                     # cloni per anticorpo           (tuning R2)
    rho: float | None = None         # legge di ipermutazione (None -> ln(nc))
    tau_b: int = 20                  # eta' massima (aging)          (tuning R4)
    ls_clones: int = 3               # K: cloni sottoposti a LS/gen  (tuning R4)
    ls_cap_fe: int = 5000            # cap di FE per invocazione LS  (tuning R1)
    h_neighbors: int = 20            # ampiezza liste di candidati della LS
    greedy_fraction: float = 0.25    # frazione di popolazione seminata da CW
    mut_weights: tuple = DEFAULT_WEIGHTS


@dataclass
class Antibody:
    tour: np.ndarray                 # genotipo: giant tour
    cost: int                        # costo della decodifica (fitness)
    n_routes: int
    routes: list                     # fenotipo: rotte decodificate
    conforming: bool = True          # True se n_routes <= k (decodifica vincolata riuscita)
    age: int = 0


def _rank(a: Antibody):
    """Ordinamento lessicografico feasibility-first (regola di dominanza):
    un anticorpo CONFORME (<= k rotte) precede sempre un non conforme;
    a parita' di classe decide il costo. E' gestione del vincolo di
    flotta in selezione, non un approccio a penalty (nessun coefficiente,
    nessuna esplorazione calibrata dell'inammissibile)."""
    return (not a.conforming, a.cost)


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

        def consider(cost, routes, n_routes, gen, tour=None):
            """Candida una soluzione appena valutata al best globale.

            Le soluzioni con piu' di k rotte non possono diventare il
            best riportato; se pero' il loro costo (Split ILLIMITATO)
            batte il best corrente, lo stesso tour potrebbe ammettere una
            partizione con <= k rotte, piu' costosa del rilassamento ma
            comunque migliorativa: si tenta allora la decodifica
            vincolata split_limited (1 FE, convenzione rispettata).
            Il filtro "solo se il rilassamento migliora il best" e'
            ESATTO perche' l'illimitato e' un lower bound del limitato
            sullo stesso tour: i tentativi inutili sono impossibili e il
            costo in FE resta trascurabile (poche decine per run).
            Meccanismo necessario sulle istanze a capacita' strettissima:
            su P-n50-k10 (riempimento 95%) senza questa decodifica lazy
            intere run non producono alcun best conforme a k."""
            if anybest["cost"] is None or cost < anybest["cost"]:
                anybest.update(cost=int(cost),
                               routes=[list(r) for r in routes],
                               n_routes=int(n_routes))
            if n_routes > kmax:
                if (tour is None or budget.exhausted
                        or (log.best_cost is not None
                            and cost >= log.best_cost)):
                    return
                budget.checkpoint()
                lim = split_limited(tour, inst, max_routes=kmax)
                budget.spend(1)
                if not lim.feasible:
                    return
                cost, routes, n_routes = lim.cost, lim.routes, lim.n_routes
            if log.update(cost, generation=gen):
                best["routes"] = [list(r) for r in routes]
                best["n_routes"] = int(n_routes)

        def new_antibody(tour, gen, age=0):
            """Valuta un tour (1 FE, con controllo PREVENTIVO del budget).
            La fitness e' split_fitness: decodifica vincolata a k rotte
            quando esiste, rilassamento illimitato altrimenti. Il tour
            NON viene ripassato a consider: se la decodifica vincolata e'
            fallita qui, ritentarla la' sarebbe lavoro inutile."""
            budget.checkpoint()
            res = split_fitness(tour, inst, kmax)
            budget.spend(1)
            consider(res.cost, res.routes, res.n_routes, gen, tour=None)
            return Antibody(tour=tour, cost=res.cost, n_routes=res.n_routes,
                            routes=res.routes,
                            conforming=res.n_routes <= kmax, age=age)

        gen = 0
        t0 = time.perf_counter()
        pop: list[Antibody] = []
        # contatori di attivita' degli operatori (diagnostica: non
        # consumano numeri casuali, i risultati restano bit-identici)
        stats = {"aging_deaths": 0, "births": 0, "dupes_demoted": 0,
                 "ls_calls": 0, "ls_improvements": 0, "lazy_limited": 0,
                 "clones_nonconforming": 0, "clones_total": 0,
                 "elite_saved_by_aging": 0}
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
                        c_new = new_antibody(t, gen, age=parent.age)
                        stats["clones_total"] += 1
                        if not c_new.conforming:
                            stats["clones_nonconforming"] += 1
                        clones.append((c_new, parent.cost))

                # (4) ricerca locale lamarckiana sui migliori K cloni
                # (feasibility-first: la LS raffina prima i conformi)
                clones.sort(key=lambda pc: _rank(pc[0]))
                for c, _ in clones[:P.ls_clones]:
                    if budget.exhausted:
                        break
                    stats["ls_calls"] += 1
                    routes, cost = searcher.improve(c.routes, c.cost,
                                                    max_fe=P.ls_cap_fe)
                    if cost < c.cost:
                        stats["ls_improvements"] += 1
                        c.routes = routes
                        c.cost = int(cost)
                        c.n_routes = len(routes)
                        c.conforming = c.n_routes <= kmax
                        c.tour = tour_from_routes(routes)   # writeback
                        consider(c.cost, routes, c.n_routes, gen, c.tour)

                # reset dell'eta' per i cloni che hanno migliorato il genitore
                for c, parent_cost in clones:
                    if c.cost < parent_cost:
                        c.age = 0

                # (5) invecchiamento + rimozione per vecchiaia (elitista)
                merged = pop + [c for c, _ in clones]
                for a in merged:
                    a.age += 1
                best_ab = min(merged, key=_rank)
                survivors = [a for a in merged
                             if a.age <= P.tau_b or a is best_ab]
                stats["aging_deaths"] += len(merged) - len(survivors)
                if best_ab.age > P.tau_b:
                    stats["elite_saved_by_aging"] += 1   # elitismo: il best non muore

                # (6) selezione (mu+lambda) con SOPPRESSIONE DEI DUPLICATI:
                # a parita' di costo passa un solo anticorpo; i duplicati
                # rientrano solo se mancano soluzioni distinte. Senza questa
                # regola la popolazione collassa su copie dello stesso ottimo
                # locale, la fitness normalizzata degenera (tutti f_hat = 1),
                # l'ipermutazione si riduce al minimo per tutti e la ricerca
                # stagna (convergenza prematura osservata empiricamente).
                survivors.sort(key=_rank)
                pop, seen, dupes = [], set(), []
                for a in survivors:
                    if len(pop) >= P.pop_size:
                        break
                    if _rank(a) in seen:
                        dupes.append(a)
                        stats["dupes_demoted"] += 1
                    else:
                        seen.add(_rank(a))
                        pop.append(a)
                for a in dupes:
                    if len(pop) >= P.pop_size:
                        break
                    pop.append(a)
                # nascite: reintroduzione di diversita' se sotto organico
                while len(pop) < P.pop_size and not budget.exhausted:
                    pop.append(new_antibody(self.rng.permutation(customers),
                                            gen, age=0))
                    stats["births"] += 1
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
            "stats": stats,
            "history": [[fe, c] for fe, c in log.history],
        }
