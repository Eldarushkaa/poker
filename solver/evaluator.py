"""Vectorised 7-card poker hand evaluator (CPU).

Card encoding
-------------
card_id  0-51
rank     card_id // 4   (0 = 2, 1 = 3, …, 12 = Ace)
suit     card_id % 4    (0-3, arbitrary)

Score encoding
--------------
score = category * BASE + sub-ranking
BASE  = 13 ** 5 = 371 293

Categories (0 = worst → 8 = best):
  0  high card
  1  one pair
  2  two pair
  3  three of a kind
  4  straight
  5  flush
  6  full house
  7  four of a kind
  8  straight flush
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 13^5 — multiplier that separates hand categories
BASE: int = 371_293

# ── helpers ──────────────────────────────────────────────────────────────

def _top_k_ranks(rank_counts: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the *k* highest ranks present in *rank_counts*.

    Parameters
    ----------
    rank_counts : (N, 13) int  – histogram of ranks for each hand.
    k           : int          – how many top ranks to return.

    Returns
    -------
    (N, k) long tensor – rank indices (0-12), highest first.
    """
    present = (rank_counts > 0).long()                          # (N, 13)
    rank_idx = torch.arange(13, device=rank_counts.device)      # (13,)
    sort_key = present * 14 + rank_idx.unsqueeze(0)             # (N, 13)
    _, top_idx = sort_key.topk(k, dim=1)
    return top_idx


def _get_kickers(
    rank_counts: torch.Tensor,
    exclude_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Top-*k* kicker ranks, ignoring ranks flagged in *exclude_mask*.

    Parameters
    ----------
    rank_counts  : (N, 13)
    exclude_mask : (N, 13) bool – True where ranks should be skipped.
    k            : int

    Returns
    -------
    (N, k) long tensor of kicker rank indices, highest first.
    """
    available = rank_counts.clone()
    available[exclude_mask] = 0
    present = (available > 0).long()
    rank_idx = torch.arange(13, device=rank_counts.device)
    sort_key = present * 14 + rank_idx.unsqueeze(0)
    _, top_idx = sort_key.topk(k, dim=1)
    return top_idx


def _detect_straight(rank_present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the best straight from a rank-presence bitmap.

    Parameters
    ----------
    rank_present : (N, 13) bool/int

    Returns
    -------
    has_straight : (N,) bool
    straight_high: (N,) long – highest card in the best straight (0-12).
    """
    N = rank_present.shape[0]
    device = rank_present.device
    rp = rank_present.long()

    straight_high = torch.full((N,), -1, dtype=torch.long, device=device)

    # sliding window of 5 consecutive ranks (start 0..8 → high 4..12)
    for start in range(9):
        window_sum = rp[:, start : start + 5].sum(dim=1)
        is_str = window_sum == 5
        high = start + 4
        straight_high = torch.where(
            is_str,
            torch.tensor(high, device=device),
            straight_high,
        )

    # wheel: A-2-3-4-5  (high card = 3, i.e. the 5)
    wheel = (rp[:, 0] & rp[:, 1] & rp[:, 2] & rp[:, 3] & rp[:, 12]).bool()
    straight_high = torch.where(
        wheel & (straight_high < 3),
        torch.tensor(3, device=device),
        straight_high,
    )

    has_straight = straight_high >= 0
    return has_straight, straight_high


def _encode_sub(b0: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor,
                b3: torch.Tensor, b4: torch.Tensor) -> torch.Tensor:
    """Pack five sub-ranking values into one integer.

    b0·13⁴ + b1·13³ + b2·13² + b3·13 + b4
    """
    return b0 * 28561 + b1 * 2197 + b2 * 169 + b3 * 13 + b4


# ── main evaluator ──────────────────────────────────────────────────────

def evaluate_hands(cards: torch.Tensor) -> torch.Tensor:
    """Evaluate N 7-card poker hands in parallel.

    Parameters
    ----------
    cards : (N, 7) long tensor of card IDs (0-51).

    Returns
    -------
    (N,) long tensor of hand scores.  Higher is better.
    """
    N = cards.shape[0]
    device = cards.device
    ZERO = torch.zeros(N, dtype=torch.long, device=device)

    ranks = cards // 4                                          # (N, 7)
    suits = cards % 4                                           # (N, 7)

    rank_counts = F.one_hot(ranks, 13).sum(dim=1)               # (N, 13)
    suit_counts = F.one_hot(suits, 4).sum(dim=1)                # (N, 4)
    rank_present = rank_counts > 0                              # (N, 13)

    # ── high card (category 0) ──────────────────────────────────────
    top5 = _top_k_ranks(rank_counts, 5)
    score = _encode_sub(top5[:, 0], top5[:, 1], top5[:, 2],
                        top5[:, 3], top5[:, 4])

    # ── one pair (category 1) ──────────────────────────────────────
    pair_mask = rank_counts >= 2
    num_pairs = pair_mask.long().sum(dim=1)
    has_pair = num_pairs >= 1

    pair_key = pair_mask.long() * (torch.arange(13, device=device) + 1)
    pair_rank_1 = pair_key.argmax(dim=1)

    p_excl = F.one_hot(pair_rank_1, 13).bool()
    p_kick = _get_kickers(rank_counts, p_excl, 3)

    pair_score = BASE + _encode_sub(pair_rank_1, p_kick[:, 0],
                                    p_kick[:, 1], p_kick[:, 2], ZERO)
    score = torch.where(has_pair, pair_score, score)

    # ── two pair (category 2) ──────────────────────────────────────
    has_two_pair = num_pairs >= 2

    pair_key_2 = pair_key.clone()
    pair_key_2.scatter_(1, pair_rank_1.unsqueeze(1), 0)
    pair_rank_2 = pair_key_2.argmax(dim=1)

    hi_pair = torch.max(pair_rank_1, pair_rank_2)
    lo_pair = torch.min(pair_rank_1, pair_rank_2)

    tp_excl = F.one_hot(hi_pair, 13).bool() | F.one_hot(lo_pair, 13).bool()
    tp_kick = _get_kickers(rank_counts, tp_excl, 1)[:, 0]

    two_pair_score = 2 * BASE + _encode_sub(hi_pair, lo_pair, tp_kick,
                                            ZERO, ZERO)
    score = torch.where(has_two_pair, two_pair_score, score)

    # ── three of a kind (category 3) ──────────────────────────────
    trip_mask = rank_counts >= 3
    has_trips = trip_mask.any(dim=1)

    trip_key = trip_mask.long() * (torch.arange(13, device=device) + 1)
    trip_rank = trip_key.argmax(dim=1)

    t_excl = F.one_hot(trip_rank, 13).bool()
    t_kick = _get_kickers(rank_counts, t_excl, 2)

    trip_score = 3 * BASE + _encode_sub(trip_rank, t_kick[:, 0],
                                        t_kick[:, 1], ZERO, ZERO)

    # ── straight (category 4) ─────────────────────────────────────
    has_straight, straight_high = _detect_straight(rank_present)
    straight_score = 4 * BASE + _encode_sub(straight_high, ZERO, ZERO,
                                            ZERO, ZERO)

    # ── flush (category 5) ────────────────────────────────────────
    flush_suit_mask = suit_counts >= 5
    has_flush = flush_suit_mask.any(dim=1)
    flush_suit_id = flush_suit_mask.long().argmax(dim=1)

    card_is_flush = suits == flush_suit_id.unsqueeze(1)         # (N, 7)
    flush_ranks = torch.where(card_is_flush, ranks,
                              torch.tensor(-1, device=device))
    flush_sorted, _ = flush_ranks.sort(dim=1, descending=True)
    ft = flush_sorted[:, :5]                                    # top-5

    flush_score = 5 * BASE + _encode_sub(ft[:, 0], ft[:, 1],
                                         ft[:, 2], ft[:, 3], ft[:, 4])

    # ── full house (category 6) ───────────────────────────────────
    fh_rem = rank_counts.clone()
    fh_rem.scatter_(1, trip_rank.unsqueeze(1), 0)
    fh_has_pair = (fh_rem >= 2).any(dim=1)
    has_full = has_trips & fh_has_pair

    fh_pair_key = (fh_rem >= 2).long() * (torch.arange(13, device=device) + 1)
    fh_pair_rank = fh_pair_key.argmax(dim=1)

    full_score = 6 * BASE + _encode_sub(trip_rank, fh_pair_rank,
                                        ZERO, ZERO, ZERO)

    # ── four of a kind (category 7) ───────────────────────────────
    quad_mask = rank_counts >= 4
    has_quads = quad_mask.any(dim=1)

    quad_key = quad_mask.long() * (torch.arange(13, device=device) + 1)
    quad_rank = quad_key.argmax(dim=1)

    q_excl = F.one_hot(quad_rank, 13).bool()
    q_kick = _get_kickers(rank_counts, q_excl, 1)[:, 0]

    quad_score = 7 * BASE + _encode_sub(quad_rank, q_kick, ZERO, ZERO, ZERO)

    # ── straight flush (category 8) ───────────────────────────────
    flush_suit_ranks = torch.where(
        card_is_flush, ranks, torch.tensor(13, device=device),
    )
    flush_rank_oh = F.one_hot(flush_suit_ranks, 14)[:, :, :13]
    flush_rank_present = flush_rank_oh.sum(dim=1) > 0

    has_sf, sf_high = _detect_straight(flush_rank_present)
    has_sf = has_sf & has_flush

    sf_score = 8 * BASE + _encode_sub(sf_high, ZERO, ZERO, ZERO, ZERO)

    # ── assemble (weakest → strongest overwrites) ─────────────────
    score = torch.where(has_straight, straight_score, score)
    score = torch.where(has_flush, flush_score, score)
    score = torch.where(has_full, full_score, score)
    score = torch.where(has_quads, quad_score, score)
    score = torch.where(has_sf, sf_score, score)

    # pure trips (no full house, no straight/flush/quads)
    pure_trips = has_trips & ~has_full & ~has_straight & ~has_flush & ~has_quads
    score = torch.where(pure_trips, trip_score, score)

    return score
