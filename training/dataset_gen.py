#!/usr/bin/env python3
"""Crash-resilient GTO dataset generator.

Generates random poker situations, runs the solver on each, and stores
results as append-only ``.jsonl`` (one JSON line per situation).

Features
--------
* **Incremental**: counts existing lines in the ``.jsonl`` on startup and
  continues from that offset.  Safe to stop / restart as many times as
  needed.
* **Batch flush**: writes in small batches (default 100), flushing to disk
  after each batch — at most 100 situations lost on a crash.
* **progress.json**: persists ``total``, ``batch_id``, ``timestamp``.
* **SIGTERM / SIGINT**: catches signals, flushes the current batch, saves
  progress, and exits cleanly.
* **Telegram notifications**: sends progress updates at a configurable
  interval (requires ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``
  env vars).
* **GPU support**: pass ``--device cuda`` to run MC simulations on the GPU,
  bypassing multiprocessing for 10-50× speedup on NVIDIA cards.

Usage
-----
::

    # basic CPU run (defaults: 2M target, batch 100, 3000 MC iters)
    python -m training.dataset_gen

    # GPU run on Windows / Linux with NVIDIA card
    python -m training.dataset_gen --device cuda

    # custom
    python -m training.dataset_gen --target 2000000 --batch 200 --iters 5000

Storage
-------
``data/gto_dataset.jsonl`` — one JSON object per line::

    {"features": [...], "action_probs": [0.0, 0.1, ...],
     "evs": [fold, call, r33, r50, r75, rpot, allin],
     "best_action": 3, "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import signal
import sys
import time
import traceback
from pathlib import Path

import torch

from simulator.situation_gen import Situation, generate_situation
from solver.ev import compute_ev
from solver.equity import shutdown_pool
from training.telegram import TelegramNotifier

log = logging.getLogger("dataset_gen")

# ─── paths ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "gto_dataset.jsonl"
PROGRESS_PATH = DATA_DIR / "progress.json"

# ─── raise fractions for the 7 actions ───────────────────────────────
# fold, check/call, raise_33%, raise_50%, raise_75%, raise_pot, all-in

RAISE_FRACS: list[float] = [0.33, 0.50, 0.75, 1.00]

# Action order: fold, call, raise_33, raise_50, raise_75, raise_pot, allin
ACTION_NAMES: list[str] = [
    "fold", "call", "raise_0.33", "raise_0.5", "raise_0.75", "raise_1.0", "allin",
]

# ─── softmax temperature for EV → action probabilities ───────────────
# Lower temperature → sharper (more deterministic) distribution
# Higher temperature → softer (more exploratory) distribution

SOFTMAX_TEMPERATURE: float = 2.0

# ─── globals for signal handling ─────────────────────────────────────

_shutdown_requested = False
_signal_count = 0


def _signal_handler(signum, frame):
    global _shutdown_requested, _signal_count
    _signal_count += 1
    _shutdown_requested = True

    if _signal_count == 1:
        log.info("Shutdown signal received (%s) — terminating workers…", signum)
        # Kill the worker pool immediately to unblock any pending starmap()
        try:
            shutdown_pool()
        except Exception:
            pass
    else:
        log.warning("Second signal — forcing immediate exit")
        os._exit(1)


# ─── core: solve one situation ───────────────────────────────────────

def solve_situation(
    sit: Situation,
    n_iters: int = 3000,
    n_workers: int = 0,
    device: str = "cpu",
) -> dict | None:
    """Run the solver on a situation and return a dataset record.

    Uses a **single** ``compute_ev`` call with multiple raise fractions
    to share the expensive equity / per-combo computation.

    Returns
    -------
    dict with ``features``, ``action_probs``, ``evs``, ``best_action``,
    ``meta``; or ``None`` if the solver fails.
    """
    try:
        # single solver call with all raise fractions
        ev_result = compute_ev(
            sit.hero_cards,
            sit.board_cards,
            sit.opponent_ranges,
            pot=sit.pot,
            facing_bet=sit.facing_bet,
            stack=sit.stack,
            hero_invested=sit.hero_invested,
            raise_fracs=RAISE_FRACS,
            n_iters=n_iters,
            n_workers=n_workers,
            device=device,
            hero_position=sit.hero_position,
            street=sit.street,
            n_players=sit.n_players,
            eqr_enabled=True,
            combo_response_iters=20,
            action_history=sit.action_history if sit.action_history else None,
            opponent_positions=sit.opponent_positions if sit.opponent_positions else None,
        )

        # extract EVs in canonical order
        fold_ev = ev_result["fold"]
        call_ev = ev_result["call"]
        raise_33_ev = ev_result.get("raise_0.33", call_ev)
        raise_50_ev = ev_result.get("raise_0.5", call_ev)
        raise_75_ev = ev_result.get("raise_0.75", call_ev)
        raise_pot_ev = ev_result.get("raise_1.0", call_ev)

        # all-in EV: compute as a separate raise frac
        pot_plus_bet = sit.pot + sit.facing_bet
        allin_frac = sit.stack / pot_plus_bet if pot_plus_bet > 0 else 10.0
        allin_result = compute_ev(
            sit.hero_cards,
            sit.board_cards,
            sit.opponent_ranges,
            pot=sit.pot,
            facing_bet=sit.facing_bet,
            stack=sit.stack,
            hero_invested=sit.hero_invested,
            raise_fracs=[allin_frac],
            n_iters=n_iters,
            n_workers=n_workers,
            device=device,
            hero_position=sit.hero_position,
            street=sit.street,
            n_players=sit.n_players,
            eqr_enabled=True,
            combo_response_iters=20,
            action_history=sit.action_history if sit.action_history else None,
            opponent_positions=sit.opponent_positions if sit.opponent_positions else None,
        )
        allin_ev = allin_result.get(f"raise_{allin_frac}", call_ev)

        ev_values = [fold_ev, call_ev, raise_33_ev, raise_50_ev,
                     raise_75_ev, raise_pot_ev, allin_ev]

        # convert EVs to action probabilities via softmax
        action_probs = _softmax(ev_values, temperature=SOFTMAX_TEMPERATURE)

        # best action index (0-6)
        best_action = int(max(range(7), key=lambda i: ev_values[i]))

        # encode features
        features = _encode_features(sit)

        # metadata
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

        return {
            "features": features,
            "action_probs": [round(p, 6) for p in action_probs],
            "evs": [round(v, 4) for v in ev_values],
            "best_action": best_action,
            "meta": meta,
        }
    except Exception as exc:
        log.warning("Solver error: %s", exc)
        return None


def _softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over a list of floats."""
    scaled = [v / temperature for v in values]
    max_v = max(scaled)
    exps = [math.exp(v - max_v) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _encode_features(sit: Situation) -> list[float]:
    """Encode a situation into the 39-float feature vector.

    Layout (39 floats):
    - [0:4]   hero card 1 — rank (2-14) in suit slot, 0 elsewhere
    - [4:8]   hero card 2 — rank (2-14) in suit slot, 0 elsewhere
    - [8:28]  board cards (5 × 4 suit slots; 0 for missing cards)
    - [28]    pot ratio      = pot / (pot + stack)
    - [29]    stack ratio    = stack / 200 (normalised by max stack)
    - [30]    facing_bet_ratio = facing_bet / (pot + 1)
    - [31]    street normalised = street / 3
    - [32:38] hero position one-hot (6 values for 6-max)
    - [38]    n_opponents normalised = n_opponents / 5
    """
    feats = [0.0] * 39

    # hero card 1
    c1 = sit.hero_cards[0].item()
    rank1 = (c1 // 4) + 2        # 2-14
    suit1 = c1 % 4               # 0-3
    feats[suit1] = float(rank1)

    # hero card 2
    c2 = sit.hero_cards[1].item()
    rank2 = (c2 // 4) + 2
    suit2 = c2 % 4
    feats[4 + suit2] = float(rank2)

    # board (up to 5 cards)
    for i in range(len(sit.board_cards)):
        c = sit.board_cards[i].item()
        r = (c // 4) + 2
        s = c % 4
        feats[8 + i * 4 + s] = float(r)

    # game state
    pot_plus_stack = sit.pot + sit.stack
    feats[28] = sit.pot / pot_plus_stack if pot_plus_stack > 0 else 0.0
    feats[29] = sit.stack / 200.0
    feats[30] = sit.facing_bet / (sit.pot + 1.0)
    feats[31] = sit.street / 3.0

    # hero position one-hot (indices 32-37)
    pos = sit.hero_position
    if 0 <= pos <= 5:
        feats[32 + pos] = 1.0

    # n_opponents normalised (index 38)
    feats[38] = sit.n_opponents / 5.0

    return feats


# ─── progress persistence ────────────────────────────────────────────

def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"total": 0, "batch_id": 0, "timestamp": None}


def _save_progress(total: int, batch_id: int) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "total": total,
            "batch_id": batch_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f)


def _count_existing_lines() -> int:
    """Count lines in the dataset file (= number of situations)."""
    if not DATASET_PATH.exists():
        return 0
    count = 0
    with open(DATASET_PATH, "rb") as f:
        for _ in f:
            count += 1
    return count


# ─── main loop ───────────────────────────────────────────────────────

def run(
    target: int = 2_000_000,
    batch_size: int = 100,
    n_iters: int = 3000,
    n_workers: int = 0,
    tg_interval: float = 60.0,
    device: str = "cpu",
) -> None:
    """Main generation loop.

    Parameters
    ----------
    target     : total situations to generate (across all runs).
    batch_size : situations per batch (flushed to disk after each).
    n_iters    : MC iterations per solver call.
    n_workers  : parallel workers (0 = all cores, CPU only).
    tg_interval: seconds between Telegram notifications.
    device     : ``"cpu"`` or ``"cuda"`` (skips multiprocessing on GPU).
    """
    global _shutdown_requested, _signal_count
    _shutdown_requested = False
    _signal_count = 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # register signal handlers (cross-platform)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):          # not available on Windows
        signal.signal(signal.SIGTERM, _signal_handler)

    # resolve device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"
    if device == "cuda":
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.info("Using CPU (%d cores)", os.cpu_count() or 1)

    # init Telegram
    tg = TelegramNotifier()

    # count existing data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _count_existing_lines()
    progress = _load_progress()
    total = max(existing, progress.get("total", 0))
    batch_id = progress.get("batch_id", 0)

    log.info("Resuming from %d situations (target: %d)", total, target)

    if total >= target:
        log.info("Target already reached (%d >= %d). Exiting.", total, target)
        return

    tg.started(target)
    start_time = time.time()
    session_count = 0
    errors = 0

    try:
        while total < target and not _shutdown_requested:
            batch_records: list[str] = []
            batch_start = time.time()

            for _ in range(batch_size):
                if _shutdown_requested:
                    break

                sit = generate_situation()
                record = solve_situation(sit, n_iters=n_iters,
                                         n_workers=n_workers,
                                         device=device)
                if record is not None:
                    batch_records.append(json.dumps(record, separators=(",", ":")))
                else:
                    errors += 1

            # flush batch to disk
            if batch_records:
                with open(DATASET_PATH, "a") as f:
                    for line in batch_records:
                        f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                added = len(batch_records)
                total += added
                session_count += added
                batch_id += 1

                _save_progress(total, batch_id)

                batch_elapsed = time.time() - batch_start
                rate_batch = added / (batch_elapsed / 60) if batch_elapsed > 0 else 0

                log.info(
                    "Batch %d: +%d situations (total: %d/%d, %.1f sit/min)",
                    batch_id, added, total, target, rate_batch,
                )

            # Telegram notification (every tg_interval seconds)
            elapsed_min = (time.time() - start_time) / 60.0
            session_rate = session_count / elapsed_min if elapsed_min > 0 else 0
            msg = tg.format_progress(total, target, session_rate, elapsed_min, errors)
            tg.send_if_interval(msg, interval_seconds=tg_interval)

    except Exception as exc:
        log.error("Fatal error: %s\n%s", exc, traceback.format_exc())
        tg.error(f"{exc}")
    finally:
        # final save
        _save_progress(total, batch_id)
        elapsed_min = (time.time() - start_time) / 60.0
        reason = "signal" if _shutdown_requested else "completed"
        log.info(
            "Stopped (%s). Total: %d, Session: +%d, Errors: %d, Time: %.1f min",
            reason, total, session_count, errors, elapsed_min,
        )
        tg.stopped(total, reason)
        shutdown_pool()


# ─── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crash-resilient GTO dataset generator",
    )
    parser.add_argument("--target", type=int, default=2_000_000,
                        help="Total target situations (default: 2000000)")
    parser.add_argument("--batch", type=int, default=100,
                        help="Batch size (default: 100)")
    parser.add_argument("--iters", type=int, default=3000,
                        help="MC iterations per solver call (default: 3000)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers; 0 = all cores (default: 0)")
    parser.add_argument("--tg-interval", type=float, default=60.0,
                        help="Telegram notification interval in seconds (default: 60)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for MC simulations: cpu or cuda (default: cpu)")
    args = parser.parse_args()

    run(
        target=args.target,
        batch_size=args.batch,
        n_iters=args.iters,
        n_workers=args.workers,
        tg_interval=args.tg_interval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
