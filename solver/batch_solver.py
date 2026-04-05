"""Batch GPU solver — processes N situations with minimal kernel launches.

The single-situation pipeline makes ~12 ``evaluate_hands`` calls per
situation.  For 100 situations that's 1,200 kernel launches — each with
~1 ms of CPU↔GPU overhead.

This module **batches** all situations together:

1. **Preprocess** — expand ranges, compute dead cards, EQR (CPU, fast).
2. **Batched range equity** — sample boards/opponents per situation on GPU,
   concatenate into one mega-tensor, **one** ``evaluate_hands`` call.
3. **Batched per-combo equity** — same pattern for per-combo response model.
4. **EV assembly** — compute fold/call/raise EVs from equities (CPU, fast).

Result: ~6 kernel launches total regardless of batch size.

Usage
-----
::

    from solver.batch_solver import batch_solve_situations

    records = batch_solve_situations(
        situations,
        raise_fracs=[0.33, 0.50, 0.75, 1.00],
        n_iters=3000,
        device="cuda",
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from solver.equity import DeviceLike, _resolve_device, _fallback_combos
from solver.evaluator import evaluate_hands
from solver.ev import EQR_TABLE, _get_eqr, _clamp_equity
from solver.ranges import compute_combo_weights, expand_range


# ═══════════════════════════════════════════════════════════════════════
# Preprocessed situation info
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _SitInfo:
    """Holds preprocessed data for one situation."""
    sit: object                             # original Situation
    hero: torch.Tensor                      # (2,) long
    board: torch.Tensor                     # (B,) long
    n_board: int
    n_board_needed: int
    opp_combos: list[torch.Tensor]          # per-opponent (n_combos, 2)
    combo_weights: Optional[list[Optional[torch.Tensor]]]
    eqr: float
    dead_set: set[int]
    raise_fracs: list[float]                # includes per-sit all-in frac
    allin_frac: float


def _preprocess(sit, raise_fracs, eqr_enabled, weighted_sampling):
    """Expand ranges, compute weights, EQR — all CPU work."""
    hero = sit.hero_cards
    board = sit.board_cards
    n_board = len(board)

    dead: set[int] = set(hero.tolist())
    if n_board > 0:
        dead.update(board.tolist())

    opp_combos = [expand_range(ht, dead) for ht in sit.opponent_ranges]

    # combo weights
    cw: Optional[list[Optional[torch.Tensor]]] = None
    if (
        weighted_sampling
        and sit.action_history
        and sit.opponent_positions
    ):
        cw = []
        for i, ht_list in enumerate(sit.opponent_ranges):
            if i < len(sit.opponent_positions):
                opp_pos = sit.opponent_positions[i]
                opp_acts = [a for p, a in sit.action_history if p == opp_pos]
            else:
                opp_acts = []
            cw.append(compute_combo_weights(ht_list, opp_acts, dead))

    # EQR
    active = (
        list(sit.opponent_positions) + [sit.hero_position]
        if sit.opponent_positions
        else None
    )
    eqr = _get_eqr(sit.hero_position, sit.street, sit.n_players, active) if eqr_enabled else 1.0

    # all-in fraction
    pot_plus_bet = sit.pot + sit.facing_bet
    allin_frac = sit.stack / pot_plus_bet if pot_plus_bet > 0 else 10.0

    # full raise fracs including all-in
    full_fracs = list(raise_fracs) + [allin_frac]

    return _SitInfo(
        sit=sit,
        hero=hero,
        board=board,
        n_board=n_board,
        n_board_needed=5 - n_board,
        opp_combos=opp_combos,
        combo_weights=cw,
        eqr=eqr,
        dead_set=dead,
        raise_fracs=full_fracs,
        allin_frac=allin_frac,
    )


# ═══════════════════════════════════════════════════════════════════════
# Batched range equity
# ═══════════════════════════════════════════════════════════════════════

def _batch_range_equity(
    infos: list[Optional[_SitInfo]],
    n_iters: int,
    device: torch.device,
) -> list[float]:
    """Compute range equity for all situations with ONE evaluate_hands call."""
    all_hero_7: list[torch.Tensor] = []
    all_opp_7: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    offsets: list[Optional[tuple[int, int, int, int]]] = []   # (h_start, o_start, n_opp, n_iters)

    h_total = 0
    o_total = 0

    for info in infos:
        if info is None or not info.opp_combos:
            offsets.append(None)
            continue

        hero = info.hero.to(device)
        board = info.board.to(device) if info.n_board > 0 else torch.tensor(
            [], dtype=torch.long, device=device
        )
        n_opp = len(info.opp_combos)
        opp_ranges = [r.to(device) for r in info.opp_combos]

        # fallback for empty ranges
        for j, r in enumerate(opp_ranges):
            if r.shape[0] == 0:
                opp_ranges[j] = _fallback_combos(hero, board, device)

        # sample opponent hands
        opp_hands = _sample_opponents(opp_ranges, info.combo_weights, n_iters, device)

        # dead card mask
        hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
        hero_board_dead[hero] = True
        if info.n_board > 0:
            hero_board_dead[board] = True

        iter_dead = hero_board_dead.unsqueeze(0).expand(n_iters, -1).clone()
        for oh in opp_hands:
            iter_dead.scatter_(1, oh, True)

        # validity
        valid = torch.ones(n_iters, dtype=torch.bool, device=device)
        if n_opp > 1:
            all_opp_cat = torch.cat(opp_hands, dim=1)
            oh_sum = F.one_hot(all_opp_cat, 52).sum(dim=1)
            valid &= oh_sum.max(dim=1).values <= 1
        for oh in opp_hands:
            for ci in range(2):
                valid &= ~hero_board_dead[oh[:, ci]]

        # board completion
        full_board = _complete_board(board, info.n_board, info.n_board_needed,
                                     iter_dead, n_iters, device)

        # hero 7-card hands
        hero_7 = torch.cat(
            [hero.unsqueeze(0).expand(n_iters, -1), full_board], dim=1
        )

        # opponent 7-card hands
        opp_stack = torch.stack(opp_hands, dim=1)   # (n_iters, n_opp, 2)
        opp_7 = torch.cat(
            [opp_stack, full_board.unsqueeze(1).expand(-1, n_opp, -1)], dim=2
        )
        opp_7_flat = opp_7.reshape(-1, 7)

        offsets.append((h_total, o_total, n_opp, n_iters))
        all_hero_7.append(hero_7)
        all_opp_7.append(opp_7_flat)
        all_valid.append(valid)

        h_total += n_iters
        o_total += opp_7_flat.shape[0]

    if not all_hero_7:
        return [0.5] * len(infos)

    # ── ONE mega evaluate_hands call ──
    hero_mega = torch.cat(all_hero_7, dim=0)
    opp_mega = torch.cat(all_opp_7, dim=0)

    hero_scores = evaluate_hands(hero_mega)
    opp_scores = evaluate_hands(opp_mega)

    # ── split and compute equity per situation ──
    equities: list[float] = []
    valid_idx = 0
    for i, info in enumerate(infos):
        if offsets[i] is None:
            equities.append(1.0 if info is not None else 0.5)
            continue

        h_start, o_start, n_opp, ni = offsets[i]
        h_pow = hero_scores[h_start : h_start + ni]
        o_pow = opp_scores[o_start : o_start + ni * n_opp].reshape(ni, n_opp)
        valid = all_valid[valid_idx]
        valid_idx += 1

        best_opp = o_pow.max(dim=1).values
        wins = (h_pow > best_opp).float()
        ties = h_pow == best_opp
        n_tied = (o_pow == h_pow.unsqueeze(1)).sum(dim=1)
        tie_share = ties.float() / (n_tied.float() + 1.0)
        results = wins + tie_share

        if valid.sum() < 10:
            equities.append(results.mean().item())
        else:
            equities.append(results[valid].mean().item())

    return equities


# ═══════════════════════════════════════════════════════════════════════
# Batched per-combo equity
# ═══════════════════════════════════════════════════════════════════════

def _batch_per_combo_equity(
    infos: list[Optional[_SitInfo]],
    n_iters_per_combo: int,
    device: torch.device,
) -> list[Optional[torch.Tensor]]:
    """Per-combo equity for primary opponent, batched across situations."""
    all_hero_7: list[torch.Tensor] = []
    all_opp_7: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    offsets: list[Optional[tuple[int, int, int]]] = []   # (start, n_combos, iters_per)

    total_evals = 0

    for info in infos:
        if info is None:
            offsets.append(None)
            continue

        primary_idx = 0
        if not info.opp_combos or info.opp_combos[primary_idx].shape[0] == 0:
            offsets.append(None)
            continue

        hero = info.hero.to(device)
        board = info.board.to(device) if info.n_board > 0 else torch.tensor(
            [], dtype=torch.long, device=device
        )
        combos = info.opp_combos[primary_idx].to(device)
        n_combos = combos.shape[0]
        total = n_combos * n_iters_per_combo

        opp_hands = combos.repeat_interleave(n_iters_per_combo, dim=0)

        hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
        hero_board_dead[hero] = True
        if info.n_board > 0:
            hero_board_dead[board] = True

        iter_dead = hero_board_dead.unsqueeze(0).expand(total, -1).clone()
        iter_dead.scatter_(1, opp_hands, True)

        valid = torch.ones(total, dtype=torch.bool, device=device)
        for ci in range(2):
            valid &= ~hero_board_dead[opp_hands[:, ci]]

        full_board = _complete_board(board, info.n_board, info.n_board_needed,
                                     iter_dead, total, device)

        hero_7 = torch.cat(
            [hero.unsqueeze(0).expand(total, -1), full_board], dim=1
        )
        opp_7 = torch.cat([opp_hands, full_board], dim=1)

        offsets.append((total_evals, n_combos, n_iters_per_combo))
        all_hero_7.append(hero_7)
        all_opp_7.append(opp_7)
        all_valid.append(valid)

        total_evals += total

    if not all_hero_7:
        return [None] * len(infos)

    # ── ONE mega evaluate_hands call ──
    hero_mega = torch.cat(all_hero_7, dim=0)
    opp_mega = torch.cat(all_opp_7, dim=0)

    hero_scores = evaluate_hands(hero_mega)
    opp_scores = evaluate_hands(opp_mega)

    # ── split and compute per-combo equity per situation ──
    results: list[Optional[torch.Tensor]] = []
    valid_idx = 0
    offset_iter = iter(offsets)

    for i, info in enumerate(infos):
        off = offsets[i]
        if off is None:
            results.append(None)
            continue

        start, n_combos, ipc = off
        end = start + n_combos * ipc

        h_pow = hero_scores[start:end]
        o_pow = opp_scores[start:end]
        valid = all_valid[valid_idx]
        valid_idx += 1

        wins = (h_pow > o_pow).float()
        ties = (h_pow == o_pow).float() * 0.5
        r = wins + ties
        r[~valid] = 0.5

        r = r.view(n_combos, ipc)
        v = valid.view(n_combos, ipc)
        v_counts = v.float().sum(dim=1).clamp(min=1)
        eq_per = (r * v.float()).sum(dim=1) / v_counts

        results.append(eq_per)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Batched caller equity (per raise frac)
# ═══════════════════════════════════════════════════════════════════════

def _batch_caller_equity(
    infos: list[Optional[_SitInfo]],
    per_combo_eqs: list[Optional[torch.Tensor]],
    raise_amount_per_sit: list[float],
    new_pot_per_sit: list[float],
    reraise_threshold: float,
    n_iters: int,
    device: torch.device,
) -> list[Optional[float]]:
    """Equity vs calling opponents for a specific raise, batched."""
    all_hero_7: list[torch.Tensor] = []
    all_opp_7: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    offsets: list[Optional[tuple[int, int, int]]] = []
    caller_infos: list[Optional[tuple[float, float, float]]] = []  # p_fold, p_call, p_reraise

    h_total = 0
    o_total = 0

    for i, info in enumerate(infos):
        if info is None or per_combo_eqs[i] is None:
            offsets.append(None)
            caller_infos.append(None)
            continue

        opp_eq = 1.0 - per_combo_eqs[i]  # opponent's equity
        fold_threshold = raise_amount_per_sit[i] / new_pot_per_sit[i] if new_pot_per_sit[i] > 0 else 0.5

        fold_mask = opp_eq < fold_threshold
        reraise_mask = opp_eq > reraise_threshold
        call_mask = ~fold_mask & ~reraise_mask

        n_total = float(len(opp_eq))
        p_fold = fold_mask.float().sum().item() / n_total if n_total else 0.0
        p_reraise = reraise_mask.float().sum().item() / n_total if n_total else 0.0
        p_call = call_mask.float().sum().item() / n_total if n_total else 1.0

        caller_infos.append((p_fold, p_call, p_reraise))

        if not call_mask.any():
            offsets.append(None)
            continue

        # Caller subset equity
        hero = info.hero.to(device)
        board = info.board.to(device) if info.n_board > 0 else torch.tensor(
            [], dtype=torch.long, device=device
        )
        primary = info.opp_combos[0].to(device)
        calling = primary[call_mask]

        # Rebuild opp_combos with calling subset
        call_combos = [
            calling if j == 0 else c.to(device) for j, c in enumerate(info.opp_combos)
        ]
        n_opp = len(call_combos)
        opp_ranges = call_combos

        for j, r in enumerate(opp_ranges):
            if r.shape[0] == 0:
                opp_ranges[j] = _fallback_combos(hero, board, device)

        # Caller combo weights
        call_cw = None
        if info.combo_weights is not None and info.combo_weights[0] is not None:
            w = info.combo_weights[0].to(device)
            cw_sub = w[call_mask]
            s = cw_sub.sum()
            if s > 0:
                cw_sub = cw_sub / s
            call_cw = [
                cw_sub if j == 0 else (info.combo_weights[j].to(device) if info.combo_weights[j] is not None else None)
                for j in range(n_opp)
            ]

        opp_hands = _sample_opponents(opp_ranges, call_cw, n_iters, device)

        hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
        hero_board_dead[hero] = True
        if info.n_board > 0:
            hero_board_dead[board] = True

        iter_dead = hero_board_dead.unsqueeze(0).expand(n_iters, -1).clone()
        for oh in opp_hands:
            iter_dead.scatter_(1, oh, True)

        valid = torch.ones(n_iters, dtype=torch.bool, device=device)
        if n_opp > 1:
            all_opp_cat = torch.cat(opp_hands, dim=1)
            oh_sum = F.one_hot(all_opp_cat, 52).sum(dim=1)
            valid &= oh_sum.max(dim=1).values <= 1
        for oh in opp_hands:
            for ci in range(2):
                valid &= ~hero_board_dead[oh[:, ci]]

        full_board = _complete_board(board, info.n_board, info.n_board_needed,
                                     iter_dead, n_iters, device)

        hero_7 = torch.cat(
            [hero.unsqueeze(0).expand(n_iters, -1), full_board], dim=1
        )
        opp_stack = torch.stack(opp_hands, dim=1)
        opp_7 = torch.cat(
            [opp_stack, full_board.unsqueeze(1).expand(-1, n_opp, -1)], dim=2
        )
        opp_7_flat = opp_7.reshape(-1, 7)

        offsets.append((h_total, o_total, n_opp))
        all_hero_7.append(hero_7)
        all_opp_7.append(opp_7_flat)
        all_valid.append(valid)

        h_total += n_iters
        o_total += opp_7_flat.shape[0]

    # Evaluate (may be empty if all situations have no callers)
    if all_hero_7:
        hero_mega = torch.cat(all_hero_7, dim=0)
        opp_mega = torch.cat(all_opp_7, dim=0)
        hero_scores = evaluate_hands(hero_mega)
        opp_scores = evaluate_hands(opp_mega)
    else:
        hero_scores = torch.tensor([])
        opp_scores = torch.tensor([])

    # Split and compute
    results: list[Optional[float]] = []
    valid_idx = 0

    for i, info in enumerate(infos):
        if caller_infos[i] is None:
            results.append(None)
            continue

        off = offsets[i]
        if off is None:
            # No callers — use raw equity
            results.append(None)
            continue

        h_start, o_start, n_opp = off
        h_pow = hero_scores[h_start : h_start + n_iters]
        o_pow = opp_scores[o_start : o_start + n_iters * n_opp].reshape(n_iters, n_opp)
        valid = all_valid[valid_idx]
        valid_idx += 1

        best_opp = o_pow.max(dim=1).values
        wins = (h_pow > best_opp).float()
        ties = h_pow == best_opp
        n_tied = (o_pow == h_pow.unsqueeze(1)).sum(dim=1)
        tie_share = ties.float() / (n_tied.float() + 1.0)
        r = wins + tie_share

        if valid.sum() < 10:
            results.append(r.mean().item())
        else:
            results.append(r[valid].mean().item())

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def batch_solve_situations(
    situations: list,
    raise_fracs: list[float],
    *,
    n_iters: int = 3000,
    combo_response_iters: int = 20,
    reraise_threshold: float = 0.75,
    device: DeviceLike = "cuda",
    eqr_enabled: bool = True,
    weighted_sampling: bool = True,
) -> list[Optional[dict]]:
    """Solve N situations with batched GPU evaluate_hands calls.

    Parameters
    ----------
    situations         : list of ``Situation`` objects.
    raise_fracs        : standard raise fractions (e.g. [0.33, 0.5, 0.75, 1.0]).
    n_iters            : MC iterations per equity computation.
    combo_response_iters : iters per combo for response model.
    reraise_threshold  : opponent equity above which they reraise.
    device             : ``"cuda"`` (default), ``"cpu"``, etc.
    eqr_enabled        : apply equity realisation multiplier.
    weighted_sampling  : use action-weighted combo sampling.

    Returns
    -------
    list of dataset records (same format as ``solve_situation``), or ``None``
    for failed situations.
    """
    dev = _resolve_device(device)

    # ── Phase 0: Preprocess (CPU) ──────────────────────────────────
    infos: list[Optional[_SitInfo]] = []
    for sit in situations:
        try:
            infos.append(_preprocess(sit, raise_fracs, eqr_enabled, weighted_sampling))
        except Exception:
            infos.append(None)

    # ── Phase 1: Batched range equity ──────────────────────────────
    raw_equities = _batch_range_equity(infos, n_iters, dev)

    # ── Phase 2: Batched per-combo equity ──────────────────────────
    per_combo_eqs = _batch_per_combo_equity(infos, combo_response_iters, dev)

    # ── Phase 3: For each raise frac, compute caller equity ────────
    # Compute raise amounts per situation per frac
    all_raise_data: dict[str, tuple[list[float], list[float], list[Optional[float]]]] = {}

    # Include standard fracs + all-in
    all_fracs_keys: list[str] = [f"raise_{f}" for f in raise_fracs] + ["allin"]

    for frac_idx, frac_key in enumerate(all_fracs_keys):
        raise_amounts = []
        new_pots = []

        for info in infos:
            if info is None:
                raise_amounts.append(0.0)
                new_pots.append(1.0)
                continue

            sit = info.sit
            if frac_key == "allin":
                frac = info.allin_frac
            else:
                frac = raise_fracs[frac_idx]

            ra = min(sit.facing_bet + frac * (sit.pot + sit.facing_bet), sit.stack)
            np_ = sit.pot + sit.facing_bet + ra
            raise_amounts.append(ra)
            new_pots.append(np_)

        caller_eq = _batch_caller_equity(
            infos, per_combo_eqs, raise_amounts, new_pots,
            reraise_threshold, n_iters, dev,
        )
        all_raise_data[frac_key] = (raise_amounts, new_pots, caller_eq)

    # ── Phase 4: Assemble records ──────────────────────────────────
    records: list[Optional[dict]] = []

    for i, info in enumerate(infos):
        if info is None:
            records.append(None)
            continue

        try:
            sit = info.sit
            raw_eq = raw_equities[i]
            eff_eq = _clamp_equity(raw_eq * info.eqr)

            # Fold EV
            fold_ev = -sit.hero_invested

            # Call EV
            total_call = sit.hero_invested + sit.facing_bet
            call_ev = eff_eq * (sit.pot - sit.hero_invested) + (1 - eff_eq) * (-total_call)

            ev_values = [fold_ev, call_ev]

            # Raise EVs
            for frac_idx, frac_key in enumerate(all_fracs_keys):
                ra_list, np_list, ceq_list = all_raise_data[frac_key]
                raise_amount = ra_list[i]
                new_pot = np_list[i]
                total_raise = sit.hero_invested + raise_amount

                # Get response probabilities from caller equity computation
                opp_eq_per = per_combo_eqs[i]
                if opp_eq_per is not None:
                    fold_threshold = raise_amount / new_pot if new_pot > 0 else 0.5
                    opp_eq = 1.0 - opp_eq_per
                    fold_mask = opp_eq < fold_threshold
                    reraise_mask = opp_eq > reraise_threshold
                    call_mask = ~fold_mask & ~reraise_mask

                    n_total = float(len(opp_eq))
                    p_fold = fold_mask.float().sum().item() / n_total if n_total else 0.0
                    p_reraise = reraise_mask.float().sum().item() / n_total if n_total else 0.0
                    p_call = call_mask.float().sum().item() / n_total if n_total else 1.0

                    eq_callers = ceq_list[i]
                    if eq_callers is None:
                        eq_callers = raw_eq
                    eff_eq_c = _clamp_equity(eq_callers * info.eqr) if eqr_enabled else eq_callers

                    showdown = eff_eq_c * (new_pot - total_raise) + (1 - eff_eq_c) * (-total_raise)
                    raise_ev = (
                        p_fold * (sit.pot - sit.hero_invested)
                        + p_call * showdown
                        + p_reraise * (-total_raise)
                    )
                else:
                    raise_ev = sit.pot - sit.hero_invested

                ev_values.append(raise_ev)

            # action_probs via softmax
            action_probs = _softmax(ev_values, temperature=2.0)
            best_action = int(max(range(len(ev_values)), key=lambda j: ev_values[j]))

            # features
            features = _encode_features(sit)

            meta = {
                "street": sit.street,
                "position": sit.hero_position,
                "n_opponents": sit.n_opponents,
                "pot": sit.pot,
                "stack": sit.stack,
                "facing_bet": sit.facing_bet,
                "hero_invested": sit.hero_invested,
                "hero_cards": sit.hero_cards.tolist(),
                "board_cards": sit.board_cards.tolist(),
            }

            records.append({
                "features": features,
                "action_probs": [round(p, 6) for p in action_probs],
                "evs": [round(v, 4) for v in ev_values],
                "best_action": best_action,
                "meta": meta,
            })
        except Exception:
            records.append(None)

    return records


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _sample_opponents(
    opp_ranges: list[torch.Tensor],
    combo_weights: Optional[list[Optional[torch.Tensor]]],
    n_iters: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Sample opponent hands from ranges."""
    opp_hands: list[torch.Tensor] = []
    for j, r in enumerate(opp_ranges):
        n_combos = r.shape[0]
        if (
            combo_weights is not None
            and j < len(combo_weights)
            and combo_weights[j] is not None
        ):
            w = combo_weights[j].to(device)
            if len(w) == n_combos:
                indices = torch.multinomial(w, n_iters, replacement=True)
            else:
                indices = torch.randint(0, n_combos, (n_iters,), device=device)
        else:
            indices = torch.randint(0, n_combos, (n_iters,), device=device)
        opp_hands.append(r[indices])
    return opp_hands


def _complete_board(
    board: torch.Tensor,
    n_board: int,
    n_board_needed: int,
    iter_dead: torch.Tensor,
    n_iters: int,
    device: torch.device,
) -> torch.Tensor:
    """Complete the board using Gumbel-top-k sampling."""
    if n_board_needed > 0:
        available_mask = ~iter_dead
        keys = torch.rand(n_iters, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_idx = keys.topk(n_board_needed, dim=1)

        if n_board > 0:
            return torch.cat(
                [board.unsqueeze(0).expand(n_iters, -1), board_idx], dim=1
            )
        return board_idx

    return board.unsqueeze(0).expand(n_iters, -1)


def _softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax."""
    scaled = [v / temperature for v in values]
    max_v = max(scaled)
    exps = [math.exp(v - max_v) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _encode_features(sit) -> list[float]:
    """Encode a situation into the 39-float feature vector."""
    feats = [0.0] * 39

    c1 = sit.hero_cards[0].item()
    rank1 = (c1 // 4) + 2
    suit1 = c1 % 4
    feats[suit1] = float(rank1)

    c2 = sit.hero_cards[1].item()
    rank2 = (c2 // 4) + 2
    suit2 = c2 % 4
    feats[4 + suit2] = float(rank2)

    for idx in range(len(sit.board_cards)):
        c = sit.board_cards[idx].item()
        r = (c // 4) + 2
        s = c % 4
        feats[8 + idx * 4 + s] = float(r)

    pot_plus_stack = sit.pot + sit.stack
    feats[28] = sit.pot / pot_plus_stack if pot_plus_stack > 0 else 0.0
    feats[29] = sit.stack / 200.0
    feats[30] = sit.facing_bet / (sit.pot + 1.0)
    feats[31] = sit.street / 3.0

    pos = sit.hero_position
    if 0 <= pos <= 5:
        feats[32 + pos] = 1.0

    feats[38] = sit.n_opponents / 5.0

    return feats
