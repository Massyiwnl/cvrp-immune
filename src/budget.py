"""
budget.py — Contatore delle valutazioni di fitness (FE), criterio di
arresto e log di convergenza.

Convenzione di conteggio adottata dal progetto
----------------------------------------------
La consegna impone come criterio di arresto il numero massimo di
valutazioni della funzione fitness: FE = 3.5 x 10^5.

    1 FE = una valutazione del costo di UNA soluzione candidata,
           indipendentemente da come il costo viene calcolato:
           (a) decodifica Split completa di un giant tour;
           (b) valutazione incrementale (Delta) del costo di un vicino
               esaminato dalla ricerca locale.

Motivazione: il budget di FE serve a rendere confrontabile lo sforzo di
ricerca a parita' di condizioni; contare ogni soluzione candidata
esaminata — anche quelle il cui costo si ottiene in O(1) per differenza —
e' la posizione piu' conservativa e difendibile. Il calcolo incrementale
del Delta e' un'ottimizzazione di VELOCITA', non una ragione per non
contare la valutazione. (Convenzione alternativa piu' permissiva:
contare solo le decodifiche Split complete; qualunque sia la scelta, va
dichiarata esplicitamente nella relazione.)

Garanzie di esattezza
---------------------
* spend(k) registra k valutazioni GIA' eseguite e solleva
  BudgetExhausted se k valutazioni non rientrano nel budget residuo:
  contabilizzare lavoro fuori budget e' un bug, e viene reso visibile
  invece di essere nascosto.
* Con il pattern per-valutazione ("controlla, valuta, registra") il
  contatore si ferma ESATTAMENTE a fe_max:

      while not budget.exhausted:
          costo = valuta(candidato)      # una valutazione
          budget.spend(1)

* checkpoint() solleva BudgetExhausted a budget esaurito: permette di
  riemergere con una sola try/except da cicli annidati in profondita'
  (es. la scansione dei vicinati nella ricerca locale della Fase 4).

Compatibilita' con Numba (checkpoint prestazioni delle fasi successive)
-----------------------------------------------------------------------
Il contatore vive in un array NumPy int64 di un solo elemento
(budget.state): un kernel compilato con Numba puo' ricevere state e
fe_max, incrementare state[0] ad ogni valutazione e interrompersi quando
state[0] >= fe_max; al ritorno, il livello Python riprende i controlli
con exhausted / checkpoint(). Un'unica fonte di verita' per il conteggio,
nessun oggetto Python nel percorso caldo.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .split import SplitResult, split

# Valore imposto dalla consegna: 3.5 x 10^5 valutazioni di fitness.
FE_MAX_DEFAULT = 350_000


class BudgetExhausted(Exception):
    """Sollevata quando si tenta di superare il budget di FE, oppure da
    checkpoint() quando il budget e' esaurito (riemersione controllata)."""


class Budget:
    """Contatore delle valutazioni di fitness con criterio di arresto."""

    def __init__(self, fe_max: int = FE_MAX_DEFAULT):
        self.fe_max = int(fe_max)
        # array condiviso con eventuali kernel Numba (vedi docstring modulo)
        self.state = np.zeros(1, dtype=np.int64)

    # ---- lettura -----------------------------------------------------

    @property
    def used(self) -> int:
        """FE consumate finora."""
        return int(self.state[0])

    @property
    def remaining(self) -> int:
        """FE ancora disponibili."""
        return max(0, self.fe_max - self.used)

    @property
    def exhausted(self) -> bool:
        """True se il budget e' esaurito (used >= fe_max)."""
        return self.used >= self.fe_max

    def can_spend(self, k: int = 1) -> bool:
        """True se k ulteriori valutazioni rientrano nel budget."""
        return self.used + k <= self.fe_max

    # ---- scrittura ----------------------------------------------------

    def spend(self, k: int = 1) -> bool:
        """Registra k valutazioni di fitness appena eseguite.

        Ritorna True se dopo la registrazione il budget NON e' esaurito
        (si puo' continuare), False se e' stato raggiunto esattamente il
        tetto. Solleva BudgetExhausted — senza toccare il contatore — se
        le k valutazioni non rientrano nel budget: significa che il
        chiamante ha eseguito lavoro fuori budget (bug da correggere,
        controllare prima con exhausted / can_spend / checkpoint)."""
        if self.used + k > self.fe_max:
            raise BudgetExhausted(
                f"tentativo di registrare {k} FE con {self.remaining} residue")
        self.state[0] += k
        return not self.exhausted

    def checkpoint(self) -> None:
        """Solleva BudgetExhausted se il budget e' esaurito; altrimenti
        non fa nulla. Da inserire in testa ai cicli caldi per riemergere
        con una sola try/except al livello della run."""
        if self.exhausted:
            raise BudgetExhausted(f"budget esaurito: {self.used}/{self.fe_max}")


class ConvergenceLog:
    """Traccia il best-so-far in funzione delle FE consumate.

    Registra un punto (fe, costo) per OGNI miglioramento stretto del
    best globale: la spezzata risultante e' esattamente la curva di
    convergenza richiesta dalla consegna (costo del best vs FE). Tiene
    inoltre la generazione dell'ultimo miglioramento, che alimenta la
    metrica "numero medio di iterazioni per il raggiungimento della
    migliore soluzione"."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.best_cost: int | None = None
        self.fe_of_best: int | None = None
        self.gen_of_best: int | None = None
        self.history: list[tuple[int, int]] = []   # (fe, costo) ai miglioramenti

    def update(self, cost, generation: int | None = None) -> bool:
        """Propone un candidato al best globale. Registra solo i
        miglioramenti STRETTI (i pareggi non alterano il log).
        Ritorna True se il best e' migliorato."""
        cost = int(cost)
        if self.best_cost is None or cost < self.best_cost:
            self.best_cost = cost
            self.fe_of_best = self.budget.used
            self.gen_of_best = generation
            self.history.append((self.budget.used, cost))
            return True
        return False

    def as_dict(self) -> dict:
        """Serializzazione JSON-friendly per i file di risultato per-run."""
        return {
            "best_cost": self.best_cost,
            "fe_of_best": self.fe_of_best,
            "gen_of_best": self.gen_of_best,
            "history": [[fe, c] for fe, c in self.history],
        }


# ---------------------------------------------------------------------------
# Valutazione completa di un giant tour (convenzione: 1 FE)
# ---------------------------------------------------------------------------

def evaluate_tour(tour, inst: Instance, budget: Budget) -> SplitResult:
    """Decodifica Split di un giant tour + registrazione di ESATTAMENTE
    1 FE, secondo la convenzione del progetto. Il chiamante deve
    verificare il budget PRIMA di invocare (exhausted / checkpoint)."""
    res = split(tour, inst)
    budget.spend(1)
    return res
