"""GTO poker solver — CPU/GPU with multiprocessing.

Public API
----------
.. rubric:: Hand evaluation

- :func:`evaluate_hands` – vectorised 7-card hand evaluator.

.. rubric:: Equity

- :func:`compute_equity`           – basic MC equity vs random opponents.
- :func:`compute_equity_vs_ranges` – range-aware MC with weighted sampling.
- :func:`compute_equity_per_combo` – per-combo equity for response modelling.

.. rubric:: Expected value

- :func:`compute_ev` – fold / call / raise EV with opponent response
  modelling, EQR, and action-weighted sampling.  Accepts multiple raise
  fractions in a single call to share expensive equity computation.

.. rubric:: Ranges

- :data:`HAND_RANKINGS`   – 169 canonical types (strongest first).
- :func:`get_position_range` – opening range for a seat.
- :func:`narrow_range`       – narrow by observed action.
- :func:`expand_range`       – hand types → concrete card combos.
- :func:`expand_hand_type`   – single hand type → combos.
- :func:`compute_combo_weights` – action-weighted sampling weights.

.. rubric:: Pool management

- :func:`get_pool`      – get / create the shared worker pool.
- :func:`shutdown_pool` – terminate the shared worker pool.
"""

from solver.evaluator import evaluate_hands
from solver.equity import (
    DeviceLike,
    compute_equity,
    compute_equity_per_combo,
    compute_equity_vs_ranges,
    get_pool,
    shutdown_pool,
)
from solver.ev import compute_ev
from solver.ranges import (
    HAND_RANKINGS,
    compute_combo_weights,
    expand_hand_type,
    expand_range,
    get_position_range,
    narrow_range,
)

__all__ = [
    # evaluator
    "evaluate_hands",
    # equity
    "DeviceLike",
    "compute_equity",
    "compute_equity_vs_ranges",
    "compute_equity_per_combo",
    # pool
    "get_pool",
    "shutdown_pool",
    # ev
    "compute_ev",
    # ranges
    "HAND_RANKINGS",
    "get_position_range",
    "narrow_range",
    "expand_range",
    "expand_hand_type",
    "compute_combo_weights",
]
