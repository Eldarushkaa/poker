"""Random poker situation generator for GTO dataset creation.

Generates random 6-max NL Hold'em situations: hero cards, board state,
pot size, stack depth, facing bet, hero position, street, and opponent
count/ranges with simulated action-based narrowing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch

from solver.ranges import get_position_range, narrow_range


@dataclass
class Situation:
    """A single poker decision point."""
    hero_cards: torch.Tensor          # (2,) long
    board_cards: torch.Tensor         # (0|3|4|5,) long
    pot: float
    stack: float
    facing_bet: float
    hero_invested: float
    hero_position: int                # 0-5 for 6-max
    street: int                       # 0=preflop, 1=flop, 2=turn, 3=river
    n_players: int                    # table size (always 6)
    n_opponents: int                  # active opponents (1-5)
    opponent_positions: list[int]     = field(default_factory=list)
    opponent_ranges: list[list[str]]  = field(default_factory=list)
    action_history: list[tuple[int, str]] = field(default_factory=list)


# ─── preflop action sequences by opponent behaviour ───────────────────

_PREFLOP_ACTIONS: list[list[str]] = [
    ["open"],                        # simple open
    ["open", "call"],                # open then call a 3-bet
    ["3bet"],                        # 3-bet
    ["open"],                        # limp / open
]

# ─── postflop action sequences ────────────────────────────────────────

_POSTFLOP_ACTIONS: list[list[str]] = [
    [],                              # no prior actions this street
    ["bet_postflop"],                # opponent bet
    ["bet_postflop", "call_postflop"],  # bet-call sequence
    ["call_postflop"],               # opponent just called
]


def generate_situation(rng: Optional[random.Random] = None) -> Situation:
    """Generate one random poker situation.

    Parameters
    ----------
    rng : optional seeded Random instance for reproducibility.

    Returns
    -------
    A :class:`Situation` with random but plausible game state.
    """
    if rng is None:
        rng = random.Random()

    n_players = 6

    # choose street with bias toward postflop (more interesting decisions)
    street = rng.choices([0, 1, 2, 3], weights=[0.25, 0.30, 0.25, 0.20])[0]

    # board size by street
    board_sizes = {0: 0, 1: 3, 2: 4, 3: 5}
    board_size = board_sizes[street]

    # deal cards
    deck = list(range(52))
    rng.shuffle(deck)
    hero_cards_list = [deck[0], deck[1]]
    board_cards_list = deck[2 : 2 + board_size]

    hero_cards = torch.tensor(hero_cards_list, dtype=torch.long)
    board_cards = torch.tensor(board_cards_list, dtype=torch.long)

    # hero position
    hero_position = rng.randint(0, 5)

    # number of active opponents (1-5, biased toward fewer)
    n_opponents = rng.choices([1, 2, 3, 4, 5],
                              weights=[0.40, 0.30, 0.15, 0.10, 0.05])[0]

    # opponent positions (distinct, != hero)
    all_positions = [p for p in range(6) if p != hero_position]
    rng.shuffle(all_positions)
    opponent_positions = sorted(all_positions[:n_opponents])

    # build opponent ranges with action-based narrowing
    action_history: list[tuple[int, str]] = []
    opponent_ranges: list[list[str]] = []

    for opp_pos in opponent_positions:
        # start from position-based opening range
        hand_types = get_position_range(opp_pos, n_players)

        # simulate preflop action
        preflop_seq = rng.choice(_PREFLOP_ACTIONS)
        for action in preflop_seq:
            hand_types = narrow_range(hand_types, action)
            action_history.append((opp_pos, action))

        # simulate postflop actions for each past street
        if street >= 1:
            for past_street in range(1, street + 1):
                postflop_seq = rng.choice(_POSTFLOP_ACTIONS)
                for action in postflop_seq:
                    hand_types = narrow_range(hand_types, action)
                    action_history.append((opp_pos, action))

        opponent_ranges.append(hand_types)

    # pot size (in big blinds)
    if street == 0:
        pot = rng.uniform(2.0, 15.0)      # preflop: limps / raises
    elif street == 1:
        pot = rng.uniform(5.0, 40.0)      # flop
    elif street == 2:
        pot = rng.uniform(10.0, 80.0)     # turn
    else:
        pot = rng.uniform(15.0, 150.0)    # river

    # stack (effective stack remaining for hero)
    stack = rng.uniform(max(pot * 0.5, 10.0), 200.0)

    # facing bet
    if rng.random() < 0.3:
        facing_bet = 0.0                   # check to hero
    else:
        # bet fraction of pot
        bet_frac = rng.choice([0.33, 0.50, 0.66, 0.75, 1.0])
        facing_bet = min(pot * bet_frac, stack)

    # hero already invested (blinds + earlier bets)
    if street == 0:
        if hero_position == 0:        # SB
            hero_invested = 0.5
        elif hero_position == 1:      # BB
            hero_invested = 1.0
        else:
            hero_invested = rng.uniform(0.0, min(pot * 0.3, 5.0))
    else:
        hero_invested = rng.uniform(1.0, pot * 0.4)

    return Situation(
        hero_cards=hero_cards,
        board_cards=board_cards,
        pot=round(pot, 2),
        stack=round(stack, 2),
        facing_bet=round(facing_bet, 2),
        hero_invested=round(hero_invested, 2),
        hero_position=hero_position,
        street=street,
        n_players=n_players,
        n_opponents=n_opponents,
        opponent_positions=opponent_positions,
        opponent_ranges=opponent_ranges,
        action_history=action_history,
    )
