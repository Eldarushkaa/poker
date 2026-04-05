"""Monte Carlo equity calculators with multiprocessing.

Three levels of sophistication, all sharing the same core MC engine:

1. :func:`compute_equity`           – basic equity vs *N* random opponents.
2. :func:`compute_equity_vs_ranges` – range-aware MC with optional weighted
                                      combo sampling.
3. :func:`compute_equity_per_combo` – per-combo equity for opponent-response
                                      modelling.

All heavy lifting is done on CPU tensors.  When *n_workers* > 1 the
iterations are split across ``torch.multiprocessing`` workers for parallel
execution on multi-core machines.

Pool management
---------------
Use :func:`get_pool` / :func:`shutdown_pool` to manage a persistent worker
pool that avoids the overhead of process creation on every call.  The pool
is lazily created on first use.
"""

from __future__ import annotations

import atexit
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from solver.evaluator import evaluate_hands

# Maximum evaluations per batch (tune to stay within CPU cache / RAM).
_MAX_BATCH: int = 50_000

# Default number of workers (0 = use all cores).
_DEFAULT_WORKERS: int = int(os.environ.get("SOLVER_WORKERS", "0"))


# ═══════════════════════════════════════════════════════════════════════
# Persistent worker pool
# ═══════════════════════════════════════════════════════════════════════

_pool: Optional[mp.pool.Pool] = None
_pool_size: int = 0


def get_pool(n_workers: int = 0) -> mp.pool.Pool:
    """Return the shared worker pool, creating it lazily if needed.

    Parameters
    ----------
    n_workers : desired pool size; 0 = ``os.cpu_count()``.

    Returns
    -------
    A ``multiprocessing.Pool`` instance reused across calls.
    """
    global _pool, _pool_size
    n = _resolve_workers(n_workers)
    if _pool is None or _pool_size != n:
        shutdown_pool()
        ctx = mp.get_context("spawn")
        _pool = ctx.Pool(n)
        _pool_size = n
    return _pool


def shutdown_pool() -> None:
    """Terminate and join the shared worker pool (if any)."""
    global _pool, _pool_size
    if _pool is not None:
        try:
            _pool.terminate()
            _pool.join()
        except Exception:
            pass
        _pool = None
        _pool_size = 0


# auto-cleanup on process exit
atexit.register(shutdown_pool)


# ═══════════════════════════════════════════════════════════════════════
# 1. Basic equity vs N random opponents
# ═══════════════════════════════════════════════════════════════════════

def compute_equity(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    n_opponents: int = 1,
    n_iters: int = 10_000,
    n_workers: int = _DEFAULT_WORKERS,
) -> float:
    """Hero equity via MC against *n_opponents* random hands.

    Parameters
    ----------
    hero_cards  : (2,) long – hero card IDs.
    board_cards : (B,) long – board card IDs, B ∈ {0, 3, 4, 5}.
    n_opponents : 1-5.
    n_iters     : total MC iterations (split across workers).
    n_workers   : parallel workers; 0 = ``os.cpu_count()``.

    Returns
    -------
    Equity as a float in [0, 1].
    """
    n_workers = _resolve_workers(n_workers)

    if n_workers <= 1:
        return _basic_equity_chunk(hero_cards, board_cards,
                                   n_opponents, n_iters)

    chunks = _split_iters(n_iters, n_workers)
    pool = get_pool(n_workers)
    results = pool.starmap(
        _basic_equity_chunk,
        [(hero_cards, board_cards, n_opponents, c) for c in chunks],
    )
    # weighted average (chunks may differ by ±1)
    total = sum(eq * c for eq, c in zip(results, chunks))
    return total / n_iters


def _basic_equity_chunk(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    n_opponents: int,
    n_iters: int,
) -> float:
    """Single-worker MC equity vs random opponents."""
    device = torch.device("cpu")
    hero = hero_cards.to(device)
    board = board_cards.to(device) if len(board_cards) > 0 else torch.tensor(
        [], dtype=torch.long, device=device
    )

    n_board = len(board)
    n_board_needed = 5 - n_board
    n_opp_cards = 2 * n_opponents
    n_needed = n_board_needed + n_opp_cards

    # available deck
    dead = torch.zeros(52, dtype=torch.bool, device=device)
    dead[hero] = True
    if n_board > 0:
        dead[board] = True
    available = torch.arange(52, device=device)[~dead]
    D = available.shape[0]

    # Gumbel-top-k sampling without replacement
    keys = torch.rand(n_iters, D, device=device)
    _, indices = keys.topk(n_needed, dim=1)
    dealt = available[indices]  # (n_iters, n_needed)

    # split dealt cards
    board_new = dealt[:, :n_board_needed]
    if n_board > 0:
        full_board = torch.cat(
            [board.unsqueeze(0).expand(n_iters, -1), board_new], dim=1
        )
    else:
        full_board = board_new

    opp_section = dealt[:, n_board_needed:]
    opp_cards = opp_section.reshape(n_iters, n_opponents, 2)

    # hero 7-card hands
    hero_7 = torch.cat(
        [hero.unsqueeze(0).expand(n_iters, -1), full_board], dim=1
    )

    # opponent 7-card hands
    opp_7 = torch.cat(
        [opp_cards, full_board.unsqueeze(1).expand(-1, n_opponents, -1)],
        dim=2,
    )

    hero_power = evaluate_hands(hero_7)
    opp_power = evaluate_hands(opp_7.reshape(-1, 7)).reshape(
        n_iters, n_opponents
    )

    best_opp = opp_power.max(dim=1).values
    wins = (hero_power > best_opp).float()
    ties = hero_power == best_opp
    n_tied = (opp_power == hero_power.unsqueeze(1)).sum(dim=1)
    tie_share = ties.float() / (n_tied.float() + 1.0)

    return (wins + tie_share).mean().item()


# ═══════════════════════════════════════════════════════════════════════
# 2. Range-aware equity with optional weighted sampling
# ═══════════════════════════════════════════════════════════════════════

def compute_equity_vs_ranges(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    opponent_combos: list[torch.Tensor],
    n_iters: int = 10_000,
    combo_weights: Optional[list[Optional[torch.Tensor]]] = None,
    n_workers: int = _DEFAULT_WORKERS,
) -> float:
    """Hero equity vs opponents sampled from provided ranges.

    Parameters
    ----------
    hero_cards       : (2,) long.
    board_cards      : (B,) long, B ∈ {0, 3, 4, 5}.
    opponent_combos  : list of (n_combos_i, 2) long tensors per opponent.
    n_iters          : total MC iterations.
    combo_weights    : optional per-opponent (n_combos_i,) weight tensors.
    n_workers        : parallel workers; 0 = all cores.

    Returns
    -------
    Equity float in [0, 1].
    """
    n_workers = _resolve_workers(n_workers)

    if n_workers <= 1:
        return _range_equity_chunk(
            hero_cards, board_cards, opponent_combos, n_iters, combo_weights,
        )

    chunks = _split_iters(n_iters, n_workers)
    pool = get_pool(n_workers)
    results = pool.starmap(
        _range_equity_chunk,
        [
            (hero_cards, board_cards, opponent_combos, c, combo_weights)
            for c in chunks
        ],
    )
    total = sum(eq * c for eq, c in zip(results, chunks))
    return total / n_iters


def _range_equity_chunk(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    opponent_combos: list[torch.Tensor],
    n_iters: int,
    combo_weights: Optional[list[Optional[torch.Tensor]]] = None,
) -> float:
    """Single-worker range-aware MC equity."""
    device = torch.device("cpu")
    hero = hero_cards.to(device)
    board = (
        board_cards.to(device)
        if len(board_cards) > 0
        else torch.tensor([], dtype=torch.long, device=device)
    )
    n_board = len(board)
    n_board_needed = 5 - n_board
    n_opponents = len(opponent_combos)

    if n_opponents == 0:
        return 1.0

    opp_ranges = [r.to(device) for r in opponent_combos]

    # fallback for empty ranges → random combos
    for i, r in enumerate(opp_ranges):
        if r.shape[0] == 0:
            opp_ranges[i] = _fallback_combos(hero, board, device)

    hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
    hero_board_dead[hero] = True
    if n_board > 0:
        hero_board_dead[board] = True

    # sample opponent hands
    opp_hands: list[torch.Tensor] = []
    for i, r in enumerate(opp_ranges):
        n_combos = r.shape[0]
        if (
            combo_weights is not None
            and i < len(combo_weights)
            and combo_weights[i] is not None
        ):
            w = combo_weights[i].to(device)
            if len(w) == n_combos:
                indices = torch.multinomial(w, n_iters, replacement=True)
            else:
                indices = torch.randint(0, n_combos, (n_iters,), device=device)
        else:
            indices = torch.randint(0, n_combos, (n_iters,), device=device)
        opp_hands.append(r[indices])

    # dead-card mask per iteration
    iter_dead = hero_board_dead.unsqueeze(0).expand(n_iters, -1).clone()
    for oh in opp_hands:
        iter_dead.scatter_(1, oh, True)

    # validity: no card conflicts
    valid = torch.ones(n_iters, dtype=torch.bool, device=device)
    if n_opponents > 1:
        all_opp = torch.cat(opp_hands, dim=1)
        oh_sum = F.one_hot(all_opp, 52).sum(dim=1)
        valid &= oh_sum.max(dim=1).values <= 1
    for oh in opp_hands:
        for ci in range(2):
            valid &= ~hero_board_dead[oh[:, ci]]

    # board completion
    full_board = _complete_board(board, n_board, n_board_needed,
                                 iter_dead, n_iters, device)

    # evaluate
    hero_7 = torch.cat(
        [hero.unsqueeze(0).expand(n_iters, -1), full_board], dim=1,
    )
    opp_stack = torch.stack(opp_hands, dim=1)
    opp_7 = torch.cat(
        [opp_stack, full_board.unsqueeze(1).expand(-1, n_opponents, -1)],
        dim=2,
    )

    hero_power = evaluate_hands(hero_7)
    opp_power = evaluate_hands(opp_7.reshape(-1, 7)).reshape(
        n_iters, n_opponents,
    )

    best_opp = opp_power.max(dim=1).values
    wins = (hero_power > best_opp).float()
    ties = hero_power == best_opp
    n_tied = (opp_power == hero_power.unsqueeze(1)).sum(dim=1)
    tie_share = ties.float() / (n_tied.float() + 1.0)

    results = wins + tie_share
    if valid.sum() < 10:
        return results.mean().item()
    return results[valid].mean().item()


# ═══════════════════════════════════════════════════════════════════════
# 3. Per-combo equity (for opponent response modelling)
# ═══════════════════════════════════════════════════════════════════════

def compute_equity_per_combo(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    opponent_combos: torch.Tensor,
    n_iters_per_combo: int = 30,
    n_workers: int = _DEFAULT_WORKERS,
) -> torch.Tensor:
    """Hero equity against each individual opponent combo.

    For each combo the board is sampled *n_iters_per_combo* times.

    Parameters
    ----------
    hero_cards        : (2,) long.
    board_cards       : (B,) long, B ∈ {0, 3, 4, 5}.
    opponent_combos   : (n_combos, 2) long – single opponent's range.
    n_iters_per_combo : board samples per combo.
    n_workers         : parallel workers; 0 = all cores.

    Returns
    -------
    (n_combos,) float tensor of hero equity per combo.
    """
    n_combos = opponent_combos.shape[0]
    if n_combos == 0:
        return torch.tensor([], dtype=torch.float32)

    n_workers = _resolve_workers(n_workers)

    # split combos across workers (not iters, to keep per-combo grouping)
    if n_workers <= 1 or n_combos < n_workers:
        return _per_combo_batch(hero_cards, board_cards,
                                opponent_combos, n_iters_per_combo)

    combo_chunks = _split_combos(opponent_combos, n_workers)
    pool = get_pool(n_workers)
    results = pool.starmap(
        _per_combo_batch,
        [
            (hero_cards, board_cards, chunk, n_iters_per_combo)
            for chunk in combo_chunks
        ],
    )
    return torch.cat(results, dim=0)


def _per_combo_batch(
    hero_cards: torch.Tensor,
    board_cards: torch.Tensor,
    combos: torch.Tensor,
    n_iters_per_combo: int,
) -> torch.Tensor:
    """Per-combo equity for a batch of combos (single worker)."""
    device = torch.device("cpu")
    hero = hero_cards.to(device)
    board = (
        board_cards.to(device)
        if len(board_cards) > 0
        else torch.tensor([], dtype=torch.long, device=device)
    )
    combos = combos.to(device)
    n_combos = combos.shape[0]

    n_board = len(board)
    n_board_needed = 5 - n_board
    total = n_combos * n_iters_per_combo

    # process in sub-batches if very large
    if total > _MAX_BATCH:
        results = []
        batch_combos = max(1, _MAX_BATCH // n_iters_per_combo)
        for start in range(0, n_combos, batch_combos):
            end = min(start + batch_combos, n_combos)
            chunk_eq = _per_combo_inner(
                hero, board, combos[start:end],
                n_iters_per_combo, n_board, n_board_needed, device,
            )
            results.append(chunk_eq)
        return torch.cat(results, dim=0)

    return _per_combo_inner(
        hero, board, combos,
        n_iters_per_combo, n_board, n_board_needed, device,
    )


def _per_combo_inner(
    hero: torch.Tensor,
    board: torch.Tensor,
    combos: torch.Tensor,
    n_iters_per_combo: int,
    n_board: int,
    n_board_needed: int,
    device: torch.device,
) -> torch.Tensor:
    """Core per-combo MC evaluation."""
    n_combos = combos.shape[0]
    total = n_combos * n_iters_per_combo

    opp_hands = combos.repeat_interleave(n_iters_per_combo, dim=0)  # (total, 2)

    hero_board_dead = torch.zeros(52, dtype=torch.bool, device=device)
    hero_board_dead[hero] = True
    if n_board > 0:
        hero_board_dead[board] = True

    iter_dead = hero_board_dead.unsqueeze(0).expand(total, -1).clone()
    iter_dead.scatter_(1, opp_hands, True)

    # validity
    valid = torch.ones(total, dtype=torch.bool, device=device)
    for ci in range(2):
        valid &= ~hero_board_dead[opp_hands[:, ci]]

    # board completion
    full_board = _complete_board(board, n_board, n_board_needed,
                                 iter_dead, total, device)

    hero_7 = torch.cat(
        [hero.unsqueeze(0).expand(total, -1), full_board], dim=1,
    )
    opp_7 = torch.cat([opp_hands, full_board], dim=1)

    hero_power = evaluate_hands(hero_7)
    opp_power = evaluate_hands(opp_7)

    wins = (hero_power > opp_power).float()
    ties = (hero_power == opp_power).float() * 0.5
    results = wins + ties

    results[~valid] = 0.5  # neutral for invalid

    results = results.view(n_combos, n_iters_per_combo)
    valid_r = valid.view(n_combos, n_iters_per_combo)
    valid_counts = valid_r.float().sum(dim=1).clamp(min=1)
    return (results * valid_r.float()).sum(dim=1) / valid_counts


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _resolve_workers(n_workers: int) -> int:
    if n_workers <= 0:
        return os.cpu_count() or 1
    return n_workers


def _split_iters(n_iters: int, n_workers: int) -> list[int]:
    """Split iterations as evenly as possible."""
    base = n_iters // n_workers
    remainder = n_iters % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


def _split_combos(combos: torch.Tensor, n_workers: int) -> list[torch.Tensor]:
    """Split combo tensor into roughly equal chunks."""
    n = combos.shape[0]
    size = max(1, n // n_workers)
    chunks = []
    for start in range(0, n, size):
        chunks.append(combos[start : start + size])
    return chunks


def _complete_board(
    board: torch.Tensor,
    n_board: int,
    n_board_needed: int,
    iter_dead: torch.Tensor,
    n_iters: int,
    device: torch.device,
) -> torch.Tensor:
    """Complete the board using Gumbel-top-k sampling.

    Returns
    -------
    (n_iters, 5) long tensor – full 5-card boards.
    """
    if n_board_needed > 0:
        available_mask = ~iter_dead
        keys = torch.rand(n_iters, 52, device=device)
        keys[~available_mask] = -1.0
        _, board_idx = keys.topk(n_board_needed, dim=1)

        if n_board > 0:
            return torch.cat(
                [board.unsqueeze(0).expand(n_iters, -1), board_idx],
                dim=1,
            )
        return board_idx

    return board.unsqueeze(0).expand(n_iters, -1)


def _fallback_combos(
    hero: torch.Tensor,
    board: torch.Tensor,
    device: torch.device,
    max_combos: int = 200,
) -> torch.Tensor:
    """Generate random combos as fallback for empty ranges."""
    dead = set(hero.tolist())
    if len(board) > 0:
        dead.update(board.tolist())
    available = sorted(set(range(52)) - dead)
    combos = []
    for j in range(len(available)):
        for k in range(j + 1, len(available)):
            combos.append((available[j], available[k]))
            if len(combos) >= max_combos:
                break
        if len(combos) >= max_combos:
            break
    return torch.tensor(combos, dtype=torch.long, device=device)
