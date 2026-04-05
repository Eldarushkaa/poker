"""Tests for the solver package: evaluator, equity, ranges, ev."""

from __future__ import annotations

import math
import random

import torch
import pytest

from solver.evaluator import evaluate_hands
from solver.equity import compute_equity, compute_equity_vs_ranges, compute_equity_per_combo
from solver.ranges import (
    HAND_RANKINGS,
    expand_hand_type,
    expand_range,
    get_position_range,
    narrow_range,
    compute_combo_weights,
)
from solver.ev import compute_ev


# ═══════════════════════════════════════════════════════════════════════
# Evaluator tests
# ═══════════════════════════════════════════════════════════════════════


def _card(rank_char: str, suit: int) -> int:
    """Convert rank character + suit (0-3) to card ID."""
    ranks = "23456789TJQKA"
    return ranks.index(rank_char) * 4 + suit


def _hand(*cards: int) -> torch.Tensor:
    """Create a single 7-card hand tensor."""
    assert len(cards) == 7
    return torch.tensor([cards], dtype=torch.long)


class TestEvaluator:
    """Hand evaluation correctness."""

    def test_royal_flush_beats_straight_flush(self):
        # Royal flush: A♠ K♠ Q♠ J♠ T♠ + 2 fillers
        royal = _hand(
            _card("A", 0), _card("K", 0), _card("Q", 0),
            _card("J", 0), _card("T", 0), _card("3", 1), _card("2", 2),
        )
        # Straight flush: 9♠ 8♠ 7♠ 6♠ 5♠ + 2 fillers
        sf = _hand(
            _card("9", 0), _card("8", 0), _card("7", 0),
            _card("6", 0), _card("5", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(royal).item() > evaluate_hands(sf).item()

    def test_four_of_a_kind_beats_full_house(self):
        quads = _hand(
            _card("A", 0), _card("A", 1), _card("A", 2),
            _card("A", 3), _card("K", 0), _card("3", 1), _card("2", 2),
        )
        full = _hand(
            _card("K", 0), _card("K", 1), _card("K", 2),
            _card("Q", 0), _card("Q", 1), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(quads).item() > evaluate_hands(full).item()

    def test_full_house_beats_flush(self):
        full = _hand(
            _card("K", 0), _card("K", 1), _card("K", 2),
            _card("Q", 0), _card("Q", 1), _card("3", 1), _card("2", 2),
        )
        flush = _hand(
            _card("A", 0), _card("J", 0), _card("9", 0),
            _card("7", 0), _card("5", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(full).item() > evaluate_hands(flush).item()

    def test_flush_beats_straight(self):
        flush = _hand(
            _card("A", 0), _card("J", 0), _card("9", 0),
            _card("7", 0), _card("5", 0), _card("3", 1), _card("2", 2),
        )
        straight = _hand(
            _card("T", 0), _card("9", 1), _card("8", 2),
            _card("7", 3), _card("6", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(flush).item() > evaluate_hands(straight).item()

    def test_straight_beats_trips(self):
        straight = _hand(
            _card("T", 0), _card("9", 1), _card("8", 2),
            _card("7", 3), _card("6", 0), _card("3", 1), _card("2", 2),
        )
        trips = _hand(
            _card("A", 0), _card("A", 1), _card("A", 2),
            _card("K", 3), _card("Q", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(straight).item() > evaluate_hands(trips).item()

    def test_trips_beats_two_pair(self):
        trips = _hand(
            _card("7", 0), _card("7", 1), _card("7", 2),
            _card("K", 3), _card("Q", 0), _card("3", 1), _card("2", 2),
        )
        two_pair = _hand(
            _card("A", 0), _card("A", 1), _card("K", 0),
            _card("K", 1), _card("Q", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(trips).item() > evaluate_hands(two_pair).item()

    def test_two_pair_beats_one_pair(self):
        two_pair = _hand(
            _card("A", 0), _card("A", 1), _card("K", 0),
            _card("K", 1), _card("Q", 0), _card("3", 1), _card("2", 2),
        )
        one_pair = _hand(
            _card("A", 0), _card("A", 1), _card("K", 0),
            _card("Q", 1), _card("J", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(two_pair).item() > evaluate_hands(one_pair).item()

    def test_one_pair_beats_high_card(self):
        one_pair = _hand(
            _card("2", 0), _card("2", 1), _card("3", 0),
            _card("4", 1), _card("5", 0), _card("6", 1), _card("7", 2),
        )
        high = _hand(
            _card("A", 0), _card("K", 1), _card("Q", 0),
            _card("J", 1), _card("9", 0), _card("3", 1), _card("2", 2),
        )
        assert evaluate_hands(one_pair).item() > evaluate_hands(high).item()

    def test_wheel_straight(self):
        """A-2-3-4-5 wheel is a valid straight."""
        wheel = _hand(
            _card("A", 0), _card("2", 1), _card("3", 2),
            _card("4", 3), _card("5", 0), _card("9", 1), _card("T", 2),
        )
        # Wheel should beat three of a kind (trips)
        trips = _hand(
            _card("3", 0), _card("3", 1), _card("3", 2),
            _card("K", 3), _card("Q", 0), _card("9", 1), _card("2", 2),
        )
        assert evaluate_hands(wheel).item() > evaluate_hands(trips).item()

    def test_higher_pair_wins(self):
        """AA beats KK."""
        aces = _hand(
            _card("A", 0), _card("A", 1), _card("3", 0),
            _card("5", 1), _card("8", 0), _card("T", 1), _card("2", 2),
        )
        kings = _hand(
            _card("K", 0), _card("K", 1), _card("3", 0),
            _card("5", 1), _card("8", 0), _card("T", 1), _card("2", 2),
        )
        assert evaluate_hands(aces).item() > evaluate_hands(kings).item()

    def test_kicker_matters(self):
        """Pair of aces with K kicker beats pair of aces with Q kicker."""
        aa_k = _hand(
            _card("A", 0), _card("A", 1), _card("K", 0),
            _card("4", 1), _card("5", 0), _card("6", 1), _card("7", 2),
        )
        aa_q = _hand(
            _card("A", 0), _card("A", 1), _card("Q", 0),
            _card("4", 1), _card("5", 0), _card("6", 1), _card("7", 2),
        )
        assert evaluate_hands(aa_k).item() > evaluate_hands(aa_q).item()

    def test_batch_evaluation(self):
        """Evaluate multiple hands in a batch."""
        hands = torch.stack([
            _hand(
                _card("A", 0), _card("A", 1), _card("A", 2),
                _card("A", 3), _card("K", 0), _card("3", 1), _card("2", 2),
            ).squeeze(0),
            _hand(
                _card("2", 0), _card("3", 1), _card("4", 0),
                _card("7", 1), _card("9", 0), _card("J", 1), _card("K", 2),
            ).squeeze(0),
        ], dim=0)

        scores = evaluate_hands(hands)
        assert scores.shape == (2,)
        assert scores[0].item() > scores[1].item()  # quads > high card

    def test_same_hand_same_score(self):
        """Same hand in different suit patterns → same score."""
        hand1 = _hand(
            _card("A", 0), _card("K", 1), _card("Q", 2),
            _card("J", 3), _card("9", 0), _card("3", 1), _card("2", 2),
        )
        hand2 = _hand(
            _card("A", 1), _card("K", 2), _card("Q", 3),
            _card("J", 0), _card("9", 1), _card("3", 2), _card("2", 3),
        )
        assert evaluate_hands(hand1).item() == evaluate_hands(hand2).item()


# ═══════════════════════════════════════════════════════════════════════
# Ranges tests
# ═══════════════════════════════════════════════════════════════════════


class TestRanges:

    def test_hand_rankings_length(self):
        assert len(HAND_RANKINGS) == 169

    def test_expand_pair(self):
        """AA → 6 combos."""
        combos = expand_hand_type("AA")
        assert len(combos) == 6
        for c1, c2 in combos:
            assert c1 // 4 == 12 and c2 // 4 == 12  # both aces
            assert c1 % 4 != c2 % 4                   # different suits

    def test_expand_suited(self):
        """AKs → 4 combos."""
        combos = expand_hand_type("AKs")
        assert len(combos) == 4
        for c1, c2 in combos:
            assert c1 % 4 == c2 % 4  # same suit

    def test_expand_offsuit(self):
        """AKo → 12 combos."""
        combos = expand_hand_type("AKo")
        assert len(combos) == 12
        for c1, c2 in combos:
            assert c1 % 4 != c2 % 4  # different suits

    def test_get_position_range_not_empty(self):
        for pos in range(6):
            r = get_position_range(pos, 6)
            assert len(r) > 0

    def test_btn_wider_than_utg(self):
        utg = get_position_range(2, 6)   # UTG
        btn = get_position_range(5, 6)   # BTN
        assert len(btn) > len(utg)

    def test_narrow_range_3bet(self):
        full = get_position_range(5, 6)  # BTN range
        narrowed = narrow_range(full, "3bet")
        assert len(narrowed) < len(full)
        assert len(narrowed) > 0

    def test_narrow_unknown_action(self):
        full = HAND_RANKINGS[:50]
        same = narrow_range(full, "unknown_action")
        assert same == full

    def test_expand_range_filters_dead(self):
        types = ["AA"]
        dead = {48, 49}  # A♠, A♥ (card IDs for ace)
        combos = expand_range(types, dead)
        for i in range(combos.shape[0]):
            assert combos[i, 0].item() not in dead
            assert combos[i, 1].item() not in dead

    def test_combo_weights_shape(self):
        types = HAND_RANKINGS[:20]
        dead: set[int] = set()
        w = compute_combo_weights(types, ["call"], dead)
        assert w is not None
        combos = expand_range(types, dead)
        assert len(w) == combos.shape[0]
        assert abs(w.sum().item() - 1.0) < 1e-5


# ═══════════════════════════════════════════════════════════════════════
# Equity tests (quick, low-iter smoke tests)
# ═══════════════════════════════════════════════════════════════════════


class TestEquity:

    def test_basic_equity_range(self):
        """Equity should be in [0, 1]."""
        hero = torch.tensor([_card("A", 0), _card("A", 1)], dtype=torch.long)
        board = torch.tensor([], dtype=torch.long)
        eq = compute_equity(hero, board, n_opponents=1, n_iters=500, n_workers=1)
        assert 0.0 <= eq <= 1.0

    def test_aa_beats_random(self):
        """AA should have > 75% equity vs 1 random opponent."""
        hero = torch.tensor([_card("A", 0), _card("A", 1)], dtype=torch.long)
        board = torch.tensor([], dtype=torch.long)
        eq = compute_equity(hero, board, n_opponents=1, n_iters=2000, n_workers=1)
        assert eq > 0.75

    def test_equity_vs_ranges_works(self):
        hero = torch.tensor([_card("A", 0), _card("K", 0)], dtype=torch.long)
        board = torch.tensor([], dtype=torch.long)
        opp_combos = expand_range(get_position_range(2, 6), set(hero.tolist()))
        eq = compute_equity_vs_ranges(
            hero, board, [opp_combos], n_iters=500, n_workers=1,
        )
        assert 0.0 <= eq <= 1.0

    def test_per_combo_equity_shape(self):
        hero = torch.tensor([_card("A", 0), _card("K", 0)], dtype=torch.long)
        board = torch.tensor([], dtype=torch.long)
        opp = expand_range(["KK"], set(hero.tolist()))
        eq = compute_equity_per_combo(
            hero, board, opp, n_iters_per_combo=10, n_workers=1,
        )
        assert eq.shape[0] == opp.shape[0]
        assert (eq >= 0).all() and (eq <= 1).all()


# ═══════════════════════════════════════════════════════════════════════
# EV tests
# ═══════════════════════════════════════════════════════════════════════


class TestEV:

    def test_compute_ev_returns_dict(self):
        hero = torch.tensor([_card("A", 0), _card("A", 1)], dtype=torch.long)
        board = torch.tensor([], dtype=torch.long)
        opp_ranges = [get_position_range(2, 6)]

        result = compute_ev(
            hero, board, opp_ranges,
            pot=10.0, facing_bet=5.0, stack=100.0, hero_invested=2.0,
            raise_fracs=[0.5, 1.0],
            n_iters=200, n_workers=1,
            combo_response_iters=5,
        )

        assert "fold" in result
        assert "call" in result
        assert "raise_0.5" in result
        assert "raise_1.0" in result
        assert "best_ev" in result
        assert "best_action" in result

    def test_fold_ev_is_negative_invested(self):
        hero = torch.tensor([_card("7", 0), _card("2", 1)], dtype=torch.long)
        board = torch.tensor([], dtype=torch.long)
        opp_ranges = [get_position_range(5, 6)]

        result = compute_ev(
            hero, board, opp_ranges,
            pot=10.0, facing_bet=5.0, stack=100.0, hero_invested=3.0,
            raise_fracs=[1.0],
            n_iters=200, n_workers=1,
            combo_response_iters=5,
        )

        assert result["fold"] == -3.0

    def test_multiple_raise_fracs(self):
        """All requested raise fracs appear in output."""
        hero = torch.tensor([_card("A", 0), _card("K", 0)], dtype=torch.long)
        board = torch.tensor([_card("A", 1), _card("7", 2), _card("3", 3)],
                             dtype=torch.long)
        opp_ranges = [get_position_range(2, 6)]

        fracs = [0.33, 0.5, 0.75, 1.0]
        result = compute_ev(
            hero, board, opp_ranges,
            pot=20.0, facing_bet=10.0, stack=90.0, hero_invested=5.0,
            raise_fracs=fracs,
            n_iters=200, n_workers=1,
            combo_response_iters=5,
        )

        for f in fracs:
            assert f"raise_{f}" in result


# ═══════════════════════════════════════════════════════════════════════
# Situation generator tests
# ═══════════════════════════════════════════════════════════════════════


class TestSituationGen:

    def test_generate_returns_situation(self):
        from simulator.situation_gen import generate_situation, Situation
        sit = generate_situation(rng=random.Random(42))
        assert isinstance(sit, Situation)
        assert sit.hero_cards.shape == (2,)
        assert sit.n_players == 6
        assert 0 <= sit.hero_position <= 5
        assert 1 <= sit.n_opponents <= 5

    def test_board_size_matches_street(self):
        from simulator.situation_gen import generate_situation
        rng = random.Random(42)
        for _ in range(50):
            sit = generate_situation(rng=rng)
            expected = {0: 0, 1: 3, 2: 4, 3: 5}[sit.street]
            assert len(sit.board_cards) == expected

    def test_no_card_overlap(self):
        from simulator.situation_gen import generate_situation
        rng = random.Random(42)
        for _ in range(50):
            sit = generate_situation(rng=rng)
            all_cards = sit.hero_cards.tolist() + sit.board_cards.tolist()
            assert len(all_cards) == len(set(all_cards)), "Card overlap detected!"

    def test_opponent_ranges_narrowed(self):
        """Ranges should have action history and potentially be narrowed."""
        from simulator.situation_gen import generate_situation
        rng = random.Random(123)
        sit = generate_situation(rng=rng)
        assert len(sit.opponent_ranges) == sit.n_opponents
        # action_history should exist
        assert hasattr(sit, "action_history")

    def test_features_39_floats(self):
        """Feature vector should be 39 floats after the format change."""
        from simulator.situation_gen import generate_situation
        from training.dataset_gen import _encode_features

        rng = random.Random(42)
        sit = generate_situation(rng=rng)
        feats = _encode_features(sit)
        assert len(feats) == 39

    def test_position_onehot(self):
        """Position should be one-hot in features[32:38]."""
        from simulator.situation_gen import generate_situation
        from training.dataset_gen import _encode_features

        rng = random.Random(42)
        sit = generate_situation(rng=rng)
        feats = _encode_features(sit)
        pos_slice = feats[32:38]
        assert sum(pos_slice) == 1.0
        assert pos_slice[sit.hero_position] == 1.0
