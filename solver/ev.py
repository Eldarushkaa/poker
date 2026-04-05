"""Expected value computation with opponent response modelling.

Merges the three levels from the original solver versions:

* **Simple EV** – fold / call / raise using raw equity.
* **Range-aware EV** – MDF-based fold/call/reraise probabilities.
* **Full EV** – per-combo opponent response classification, equity
  realisation (EQR) multipliers, and action-weighted sampling.

The main entry point is :func:`compute_ev` which accepts **multiple
raise fractions** in a single call so that the expensive equity / per-combo
response computation is done only once and reused across all sizings.

Supports an optional *device* parameter (``"cpu"`` or ``"cuda"``) that is
forwarded to the equity layer.  On GPU the multiprocessing pool is bypassed
and the GPU handles parallelism natively.
"""

from __future__ import annotations

from typing import Optional

import torch

from solver.equity import (
    DeviceLike,
    compute_equity_per_combo,
    compute_equity_vs_ranges,
)
from solver.ranges import (
    compute_combo_weights,
    expand_range,
)


# ─── Equity Realisation lookup ────────────────────────────────────────

EQR_TABLE: dict[tuple[bool, int], float] = {
    (True,  0): 1.05,   # IP preflop
    (False, 0): 0.90,   # OOP preflop
    (True,  1): 1.03,   # IP flop
    (False, 1): 0.93,   # OOP flop
    (True,  2): 1.01,   # IP turn
    (False, 2): 0.97,   # OOP turn
    (True,  3): 1.00,   # IP river
    (False, 3): 1.00,   # OOP river
}


def _get_eqr(
    hero_position: int,
    street: int,
    n_players: int = 6,
    active_positions: Optional[list[int]] = None,
) -> float:
    """Equity-realisation multiplier based on position and street."""
    if active_positions and len(active_positions) > 1:
        is_ip = hero_position == max(active_positions)
    else:
        if n_players == 2:
            is_ip = (hero_position == 0) if street > 0 else (hero_position == 1)
        elif n_players <= 6:
            is_ip = hero_position == 5   # BTN for 6-max
        else:
            is_ip = hero_position == 8   # BTN for 9-max
    return EQR_TABLE.get((is_ip, street), 1.0)


# ─── main entry point ────────────────────────────────────────────────

def compute_ev(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    opponent_range_hand_types: list[list[str]],
    *,
    pot: float,
    facing_bet: float,
    stack: float,
    hero_invested: float,
    raise_fracs: list[float] | None = None,
    raise_frac: float | None = None,
    n_iters: int = 3_000,
    n_workers: int = 0,
    device: DeviceLike = None,
    # positional / EQR
    hero_position: int = 0,
    street: int = 0,
    n_players: int = 6,
    eqr_enabled: bool = True,
    # per-combo response
    combo_response_iters: int = 30,
    reraise_threshold: float = 0.75,
    # weighted sampling
    weighted_sampling: bool = True,
    action_history: Optional[list[tuple[int, str]]] = None,
    opponent_positions: Optional[list[int]] = None,
) -> dict[str, float]:
    """Compute EV for fold / call / multiple raise sizes.

    The expensive equity computation (range-aware MC + per-combo response)
    is done **once** and reused across all raise fractions.

    Parameters
    ----------
    hero_cards                : (2,) long tensor.
    board_cards               : (B,) long tensor, B ∈ {0, 3, 4, 5}.
    opponent_range_hand_types : list of hand-type lists per opponent.
    pot           : current total pot.
    facing_bet    : amount hero must call (0 if check).
    stack         : hero's remaining stack.
    hero_invested : total hero has put in so far.
    raise_fracs   : list of raise sizes as fractions of (pot + facing_bet).
                    e.g. [0.33, 0.50, 0.75, 1.00] for 33%/50%/75%/pot.
    raise_frac    : single raise fraction (legacy compat). Ignored if
                    raise_fracs is provided.
    n_iters       : MC iterations for equity.
    n_workers     : parallel workers (0 = all cores, CPU only).
    device        : ``"cpu"`` (default), ``"cuda"``, etc.
    hero_position : seat index.
    street        : 0 = preflop … 3 = river.
    n_players     : table size.
    eqr_enabled   : apply EQR multiplier.
    combo_response_iters : MC iters per combo for response model.
    reraise_threshold    : opponent equity above which they reraise.
    weighted_sampling    : use action-weighted combo sampling.
    action_history       : list of (position, action_type) tuples.
    opponent_positions   : seat indices of opponents.

    Returns
    -------
    dict mapping action names to EV values.  Keys always include
    ``"fold"`` and ``"call"``.  For each raise_frac ``f``, the key is
    ``f"raise_{f}"`` (e.g. ``"raise_0.33"``).  Also includes ``"best_ev"``
    and ``"best_action"`` (the key name of the best action).
    """
    # ── resolve raise_fracs ────────────────────────────────────────
    if raise_fracs is None:
        if raise_frac is not None:
            raise_fracs = [raise_frac]
        else:
            raise_fracs = [1.0]

    # ── dead cards ────────────────────────────────────────────────
    dead: set[int] = set(hero_cards.tolist())
    if len(board_cards) > 0:
        dead.update(board_cards.tolist())

    # ── expand opponent ranges ────────────────────────────────────
    opp_combos = [expand_range(ht, dead) for ht in opponent_range_hand_types]

    # ── combo weights ─────────────────────────────────────────────
    cw: Optional[list[Optional[torch.Tensor]]] = None
    if weighted_sampling and action_history and opponent_positions:
        cw = []
        for i, ht_list in enumerate(opponent_range_hand_types):
            if i < len(opponent_positions):
                opp_pos = opponent_positions[i]
                opp_acts = [a for p, a in action_history if p == opp_pos]
            else:
                opp_acts = []
            cw.append(compute_combo_weights(ht_list, opp_acts, dead))

    # ── EQR ───────────────────────────────────────────────────────
    active = (
        list(opponent_positions) + [hero_position]
        if opponent_positions
        else None
    )
    eqr = _get_eqr(hero_position, street, n_players, active) if eqr_enabled else 1.0

    # ══════════════════════════════════════════════════════════════
    # SHARED: compute equity ONCE for all raise sizes
    # ══════════════════════════════════════════════════════════════

    raw_eq = compute_equity_vs_ranges(
        hero_cards, board_cards, opp_combos, n_iters, cw, n_workers,
        device=device,
    )
    eff_eq = _clamp_equity(raw_eq * eqr)

    # per-combo opponent response (computed ONCE)
    primary_idx = 0
    has_primary = bool(opp_combos and opp_combos[primary_idx].shape[0] > 0)
    hero_eq_per = None
    opp_eq_per = None

    if has_primary:
        primary = opp_combos[primary_idx]
        hero_eq_per = compute_equity_per_combo(
            hero_cards, board_cards, primary,
            combo_response_iters, n_workers, device=device,
        )
        opp_eq_per = 1.0 - hero_eq_per

    # ── results dict ──────────────────────────────────────────────
    results: dict[str, float] = {}

    # ── fold EV ───────────────────────────────────────────────────
    fold_ev = -hero_invested
    results["fold"] = fold_ev

    # ── call EV ───────────────────────────────────────────────────
    total_call = hero_invested + facing_bet
    call_ev = eff_eq * (pot - hero_invested) + (1 - eff_eq) * (-total_call)
    results["call"] = call_ev

    # ── raise EVs (reusing shared equity + per-combo data) ────────
    for frac in raise_fracs:
        raise_amount = min(facing_bet + frac * (pot + facing_bet), stack)
        total_raise = hero_invested + raise_amount
        new_pot = pot + facing_bet + raise_amount

        # pot-odds fold threshold
        fold_threshold = raise_amount / new_pot if new_pot > 0 else 0.5

        if has_primary:
            # classify opponent response using shared per-combo data
            opp_cpu = opp_eq_per.cpu()
            fold_mask = opp_cpu < fold_threshold
            reraise_mask = opp_cpu > reraise_threshold
            call_mask = ~fold_mask & ~reraise_mask

            n_total = float(len(opp_cpu))
            p_fold = fold_mask.float().sum().item() / n_total if n_total else 0.0
            p_reraise = reraise_mask.float().sum().item() / n_total if n_total else 0.0
            p_call = call_mask.float().sum().item() / n_total if n_total else 1.0

            # equity vs callers
            if call_mask.any():
                calling = primary[call_mask]
                all_calling = [
                    calling if j == primary_idx else c
                    for j, c in enumerate(opp_combos)
                ]

                # weights for calling subset
                call_cw: Optional[list[Optional[torch.Tensor]]] = None
                if cw is not None and cw[primary_idx] is not None:
                    w = cw[primary_idx]
                    cw_sub = w[call_mask]
                    s = cw_sub.sum()
                    if s > 0:
                        cw_sub = cw_sub / s
                    call_cw = [
                        cw_sub if j == primary_idx else (cw[j] if cw and j < len(cw) else None)
                        for j in range(len(opp_combos))
                    ]

                eq_callers = compute_equity_vs_ranges(
                    hero_cards, board_cards, all_calling, n_iters, call_cw, n_workers,
                    device=device,
                )
            else:
                eq_callers = raw_eq

            eff_eq_c = _clamp_equity(eq_callers * eqr) if eqr_enabled else eq_callers
            showdown = eff_eq_c * (new_pot - total_raise) + (1 - eff_eq_c) * (-total_raise)

            raise_ev = (
                p_fold * (pot - hero_invested)
                + p_call * showdown
                + p_reraise * (-total_raise)
            )
        else:
            raise_ev = pot - hero_invested   # no opponents → auto-win

        results[f"raise_{frac}"] = raise_ev

    # ── best action ───────────────────────────────────────────────
    best_key = max(results, key=results.get)  # type: ignore[arg-type]
    results["best_ev"] = results[best_key]
    results["best_action"] = best_key  # type: ignore[assignment]

    return results


# ── utility ──────────────────────────────────────────────────────────

def _clamp_equity(eq: float) -> float:
    return max(0.0, min(1.0, eq))
