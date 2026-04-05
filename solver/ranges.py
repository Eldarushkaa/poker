r"""Hand ranges, position-based opening ranges, and action-based narrowing.

Canonical hand types
--------------------
169 distinct hold'em starting hands ordered by preflop strength (strongest
first).  Pairs have no suffix, suited hands end with ``"s"``, offsuit with
``"o"``.

Position ranges
---------------
Each seat maps to an opening-range width (fraction of 169 types).  A 6-max
table remaps seats 0-5 to the standard 9-max positions.

Action narrowing
----------------
Observed opponent actions shrink their range by selecting a slice of the
strength-ordered hand-type list (e.g.\ a 3-bet keeps only the top 8 %).

Combo expansion
---------------
Each canonical type expands to concrete ``(card_id_1, card_id_2)`` combos,
filtering out dead cards.  Action-history-weighted sampling weights are
also available.
"""

from __future__ import annotations

from typing import Optional

import torch

# ─── 169 canonical hand types (strongest → weakest) ────────────────────

HAND_RANKINGS: list[str] = [
    # Premium
    "AA", "KK", "QQ", "AKs", "JJ", "AQs", "KQs", "AJs", "KJs", "TT",
    "AKo", "ATs", "QJs", "KTs", "QTs", "JTs", "99", "AQo", "A9s", "KQo",
    # Strong
    "88", "K9s", "T9s", "A8s", "Q9s", "J9s", "AJo", "A5s", "77", "A7s",
    "KJo", "A4s", "A3s", "A6s", "QJo", "66", "K8s", "T8s", "A2s", "98s",
    # Playable
    "J8s", "ATo", "Q8s", "K7s", "KTo", "55", "JTo", "87s", "QTo", "44",
    "33", "22", "K6s", "97s", "K5s", "76s", "T7s", "K4s", "K3s", "K2s",
    "Q7s", "86s", "65s", "J7s", "54s", "Q6s", "75s", "96s", "Q5s", "64s",
    # Marginal
    "Q4s", "Q3s", "T9o", "T6s", "Q2s", "A9o", "53s", "85s", "J6s", "J9o",
    "K9o", "J5s", "Q9o", "43s", "74s", "J4s", "J3s", "95s", "J2s", "63s",
    "A8o", "52s", "T5s", "84s", "T4s", "T3s", "42s", "T2s", "98o", "T8o",
    # Weak
    "A5o", "A7o", "73s", "A4o", "32s", "94s", "93s", "J8o", "A3o", "62s",
    "92s", "K8o", "A6o", "87o", "Q8o", "83s", "A2o", "82s", "97o", "72s",
    "76o", "K7o", "65o", "T7o", "K6o", "86o", "54o", "K5o", "J7o", "75o",
    "Q7o", "K4o", "K3o", "96o", "K2o", "64o", "Q6o", "53o", "85o", "T6o",
    "Q5o", "43o", "Q4o", "Q3o", "74o", "Q2o", "J6o", "63o", "J5o", "95o",
    "52o", "J4o", "J3o", "42o", "J2o", "84o", "T5o", "T4o", "32o", "T3o",
    "73o", "T2o", "62o", "94o", "93o", "92o", "83o", "82o", "72o",
]

# ─── position-based opening-range widths ────────────────────────────────

POSITION_RANGE_PCT: dict[int, float] = {
    0: 0.35,   # SB
    1: 0.40,   # BB (defending)
    2: 0.12,   # UTG  (9-max)
    3: 0.14,   # UTG+1
    4: 0.16,   # UTG+2 / 6-max UTG
    5: 0.20,   # LJ
    6: 0.24,   # HJ
    7: 0.30,   # CO
    8: 0.42,   # BTN
}

# 6-max seat → 9-max equivalent
_REMAP_6MAX: dict[int, int] = {0: 0, 1: 1, 2: 4, 3: 6, 4: 7, 5: 8}

# ─── action narrowing slices ───────────────────────────────────────────

ACTION_NARROWING: dict[str, tuple[float, float]] = {
    "open":          (0.0, 1.0),
    "call":          (0.10, 0.70),
    "3bet":          (0.0, 0.08),
    "call_postflop": (0.0, 0.70),
    "bet_postflop":  (0.0, 0.40),
}

# ─── rank character → index (0 = 2, …, 12 = A) ───────────────────────

_RANK_CHAR: dict[str, int] = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}


# ─── hand-type parsing & combo expansion ──────────────────────────────

def _parse_hand_type(hand_type: str) -> tuple[int, int, Optional[bool]]:
    """Parse ``"AKs"`` → ``(r1, r2, is_suited)``.  Pairs return ``None``."""
    if len(hand_type) == 2:
        r = _RANK_CHAR[hand_type[0]]
        return r, r, None
    r1 = _RANK_CHAR[hand_type[0]]
    r2 = _RANK_CHAR[hand_type[1]]
    return max(r1, r2), min(r1, r2), hand_type[2] == "s"


def expand_hand_type(hand_type: str) -> list[tuple[int, int]]:
    """Expand a canonical hand type to all concrete ``(card1, card2)`` combos.

    Card ID = rank * 4 + suit.
    """
    r1, r2, is_suited = _parse_hand_type(hand_type)
    combos: list[tuple[int, int]] = []

    if is_suited is None:                   # pair → C(4,2) = 6 combos
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                combos.append((r1 * 4 + s1, r2 * 4 + s2))
    elif is_suited:                         # suited → 4 combos
        for s in range(4):
            combos.append((r1 * 4 + s, r2 * 4 + s))
    else:                                   # offsuit → 12 combos
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    combos.append((r1 * 4 + s1, r2 * 4 + s2))
    return combos


# pre-computed combo cache
_COMBOS_CACHE: dict[str, list[tuple[int, int]]] = {
    ht: expand_hand_type(ht) for ht in HAND_RANKINGS
}


# ─── range construction ───────────────────────────────────────────────

def get_position_range(position: int, n_players: int = 6) -> list[str]:
    """Hand types in a position's opening range.

    Parameters
    ----------
    position  : seat index (0-based).
    n_players : table size (2-9).

    Returns
    -------
    Slice of :data:`HAND_RANKINGS` from the top.
    """
    if n_players <= 3:
        pct = max(POSITION_RANGE_PCT.get(position, 0.30), 0.35)
    elif n_players <= 6:
        mapped = _REMAP_6MAX.get(position, position)
        pct = POSITION_RANGE_PCT.get(mapped, 0.25)
    else:
        pct = POSITION_RANGE_PCT.get(position, 0.25)

    n_types = max(1, int(len(HAND_RANKINGS) * pct))
    return HAND_RANKINGS[:n_types]


def narrow_range(hand_types: list[str], action: str) -> list[str]:
    """Narrow a range based on an observed action.

    Parameters
    ----------
    hand_types : strength-ordered list.
    action     : key into :data:`ACTION_NARROWING`.

    Returns
    -------
    Narrowed list (may be same if action unknown).
    """
    if action not in ACTION_NARROWING:
        return hand_types
    from_pct, to_pct = ACTION_NARROWING[action]
    n = len(hand_types)
    start = int(n * from_pct)
    end = int(n * to_pct)
    return hand_types[start : max(start + 1, end)]


def expand_range(
    hand_types: list[str],
    dead_cards: set[int],
) -> torch.Tensor:
    """Expand hand types to concrete combos, filtering dead cards.

    Returns
    -------
    (n_combos, 2) long tensor, or empty (0, 2) tensor.
    """
    combos: list[tuple[int, int]] = []
    for ht in hand_types:
        for c1, c2 in _COMBOS_CACHE[ht]:
            if c1 not in dead_cards and c2 not in dead_cards:
                combos.append((c1, c2))
    if not combos:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.tensor(combos, dtype=torch.long)


# ─── action-weighted combo sampling ───────────────────────────────────

def compute_combo_weights(
    hand_types: list[str],
    action_history: list[str],
    dead_cards: Optional[set[int]] = None,
) -> Optional[torch.Tensor]:
    """Per-combo sampling weights based on action consistency.

    For each observed action the weight curve is:

    * ``"call"`` / ``"call_postflop"`` → bell curve centred on middle of range.
    * ``"3bet"`` / ``"bet_postflop"``  → exponential decay from top.
    * ``"open"``                       → uniform.

    Returns
    -------
    (n_combos,) float tensor normalised to sum 1, or ``None`` if empty.
    """
    n = len(hand_types)
    if n == 0:
        return None

    type_w = torch.ones(n, dtype=torch.float32)

    for action in action_history:
        if action in ("call", "call_postflop"):
            centre = n / 2.0
            for i in range(n):
                dist = abs(i - centre) / max(n, 1)
                type_w[i] *= max(0.1, 1.0 - dist * 1.5)
        elif action in ("3bet", "bet_postflop"):
            for i in range(n):
                frac = i / max(n - 1, 1)
                type_w[i] *= max(0.1, 1.0 - frac * 0.9)
        # "open" → uniform, no change

    dead = dead_cards or set()
    combo_w: list[float] = []
    for i, ht in enumerate(hand_types):
        n_valid = sum(
            1 for c1, c2 in _COMBOS_CACHE[ht]
            if c1 not in dead and c2 not in dead
        )
        combo_w.extend([type_w[i].item()] * n_valid)

    if not combo_w:
        return None

    w = torch.tensor(combo_w, dtype=torch.float32)
    return w / w.sum()
