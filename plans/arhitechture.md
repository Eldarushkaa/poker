# Poker Bot Architecture Plan

## Overview

NL Texas Hold'em 6-max poker bot built in 7 phases:
1. Clean solver rewrite (CPU-optimized)
2. Situation generator
3. GTO dataset generation (distillation source)
4. Neural net GTO bot
5. Multi-player table simulator
6. Enhanced bot with game-context inputs
7. Genetic algorithm self-play evolution

## Environment

- **VPS (macOS/Linux)**: No GPU, 8 CPU guaranteed (64 limit on weekends)
- **Windows PC**: RTX 3060 Ti GPU (CUDA)
- **Framework**: PyTorch CPU + multiprocessing (VPS) / PyTorch CUDA (Windows)
- **Python**: 3.10+

## Project Structure

```
poker_bot/
├── solver/
│   ├── __init__.py          # public API + pool management exports
│   ├── evaluator.py         # vectorized 7-card hand evaluator (CPU/GPU)
│   ├── equity.py            # Monte Carlo equity + persistent worker pool + GPU
│   ├── ranges.py            # 169 hand types, position ranges, narrowing
│   ├── ev.py                # multi-raise EV with shared equity computation
│   └── batch_solver.py      # batched GPU solver (N situations → 3 kernel launches)
├── simulator/
│   ├── __init__.py
│   ├── situation_gen.py     # random situation generator with action narrowing
│   ├── deck.py              # [Phase 5] card constants, deck, dealing
│   ├── game_state.py        # [Phase 5] game state dataclass
│   └── table.py             # [Phase 5] full 6-max game loop
├── bot/
│   ├── __init__.py
│   ├── encoding.py          # [Phase 4] state to feature vector
│   ├── model.py             # [Phase 4] MLP architecture
│   ├── gto_bot.py           # [Phase 4] GTO-distilled bot
│   └── enhanced_bot.py      # [Phase 6] bot with game context
├── training/
│   ├── __init__.py
│   ├── dataset_gen.py       # crash-resilient batch solver → training data
│   ├── telegram.py          # Telegram notification helper
│   ├── train_gto.py         # [Phase 4] supervised distillation training
│   └── genetic.py           # [Phase 7] genetic algorithm evolution
├── tests/
│   ├── __init__.py
│   └── test_solver.py       # evaluator, equity, ranges, ev, situation_gen tests
├── data/                    # generated datasets (.jsonl)
├── models/                  # saved weights
├── plans/
│   └── arhitechture.md
├── .env                     # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SOLVER_WORKERS
├── .gitignore
└── requirements.txt
```

## Phase Details

### Phase 1: Solver Rewrite

Merge 3 files into 4 clean modules. Key changes:
- Remove all `device=mps` — use `device=cpu` everywhere
- Use `torch.multiprocessing` to split MC sims across cores
- Clean imports — proper package structure, no sys.path hacks
- Single version of each function with all improvements from v1-v3

**evaluator.py**: Hand evaluation engine
- `evaluate_hands(cards: Tensor) -> Tensor` — vectorized 7-card eval
- Categories: 0=high card through 8=straight flush
- Score = category * 13^5 + sub-ranking

**ranges.py**: Hand range system
- 169 canonical hand types ordered by strength
- Position-based opening ranges for 6-max
- Action-based narrowing (open/call/3bet/bet/call postflop)
- Combo expansion with dead-card filtering
- Action-weighted combo sampling

**equity.py**: Monte Carlo equity
- `compute_equity()` — basic MC equity vs N random opponents
- `compute_equity_vs_ranges()` — range-aware MC with weighted sampling
- `compute_equity_per_combo()` — per-combo equity for response modeling
- All use Gumbel-top-k for sampling without replacement
- Multiprocessing: split n_iters across workers, merge results

**ev.py**: Expected value computation
- `compute_ev()` — fold/call/raise EV with:
  - Per-combo opponent response classification
  - EQR (equity realization) multipliers by position/street
  - Pot-odds based fold/call/reraise probabilities

### Phase 2: Situation Generator

Generate random poker situations for the solver to evaluate:
- Random hero hole cards
- Random board states (preflop/flop/turn/river)
- Random pot sizes, stack depths, facing bets
- Random hero positions
- Configurable distributions to cover edge cases

### Phase 3: GTO Dataset Generation

**Crash-resilient, incremental batch pipeline:**
1. Generate situations in small batches (100 per batch)
2. For each batch: run solver, get fold_ev/call_ev/raise_ev for multiple sizings
3. Convert to (feature_vector, optimal_action_distribution) pairs
4. Append to `.jsonl` file (one JSON line per situation — atomic append, no corruption on crash)
5. Flush to disk after every batch
6. On restart: read existing `.jsonl`, count lines, resume from that offset
7. `progress.json` tracks: total_situations, last_batch_id, timestamp
8. SIGTERM/SIGINT handler: flush current batch, save progress, exit cleanly
9. Target: 2M+ situations (run/stop/resume as many times as needed)
10. **Telegram notifications** every 10 minutes via bot API:
    - Read `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from environment
    - Report: total generated, rate (situations/min), ETA to target
    - Also notify on start, graceful stop, and errors
    - Uses `requests` library (simple HTTP POST, no async needed)

**Storage format** (`data/gto_dataset.jsonl`):
```json
{"features": [8.0, 0, 0, 12.0, ...], "action_probs": [0.0, 0.1, 0.0, 0.7, 0.2, 0.0, 0.0], "evs": [fold, call, r33, r50, r75, rpot, allin], "best_action": 3, "meta": {...}}
```
- `action_probs`: softmax(evs / temperature) — probability distribution over 7 actions
- `evs`: raw EV values as a list [fold, call, raise_33, raise_50, raise_75, raise_pot, allin]
- `best_action`: argmax index (0-6)

### Phase 4: Neural Net GTO Bot

**Input encoding (39 features)**:
- 8 floats: hero hole cards (4 per card — rank value in suit slot, 0 elsewhere)
- 20 floats: board cards (5 cards x 4 suit-slots, 0 for missing cards)
- 4 floats: pot_ratio, stack_ratio, facing_bet_ratio, street_normalized
- 6 floats: hero position one-hot (6-max seats)
- 1 float: n_opponents / 5 (normalised opponent count)

**Architecture**: MLP with residual connections and batch normalization
- GTO bot: 39 → 256 → 256 → 256 → 128 → 128 → 7 (5 hidden layers, ~204K params)
- Residual skip connections every 2 layers
- BatchNorm + ReLU between layers
- Dropout 0.1 for regularization
- ~1ms inference on CPU

**Output (7 actions)**:
- fold
- check/call
- raise 33% pot
- raise 50% pot
- raise 75% pot
- raise 100% pot (pot-size)
- all-in

**Training**: Cross-entropy loss, Adam optimizer, ~50 epochs

### Phase 5: Multi-Player Table Simulator

Full 6-max game engine:
- Blind posting, dealing, 4 betting rounds
- Action validation (min-raise, all-in rules)
- Side pot computation
- Showdown and pot distribution
- Bot interface: abstract class with `decide(game_state) -> Action`

### Phase 6: Enhanced Bot

**Extended input (~62 features)**:
- 28 base features (cards + game state)
- 6x VPIP per opponent (0 if folded/absent)
- 6x PFR per opponent
- 6x Aggression Factor per opponent
- 6x stack depth per seat
- Hero position one-hot (6 values)
- Betting round action counts

**Architecture**: MLP with residual connections and batch normalization
- Enhanced bot: 62 → 256 → 256 → 256 → 256 → 128 → 128 → 7 (6 hidden layers, ~280K params)
- Same residual + BatchNorm + Dropout pattern as GTO bot
- ~1-2ms inference on CPU
- Needs 2M+ training situations for good generalization

### Phase 7: Genetic Algorithm

- Population: 50-100 bots
- Mix: ~30% pure GTO bots (frozen weights) + ~70% enhanced bots
- Tournament: round-robin within groups, track chip profit
- Selection: top 20% survive, bottom 80% replaced
- Mutation: Gaussian noise on weights (sigma=0.01-0.05)
- Crossover: weighted average of parent weights
- Generations: 100-500 depending on convergence

## Performance Notes

- Solver MC (CPU): 8-core multiprocessing, iters split into 8 chunks
- Solver MC (GPU): batched across situations, ~3 evaluate_hands mega-calls
  per batch of 100 situations, bypasses multiprocessing entirely
- CPU throughput: ~1,400 sit/min (8-core multiprocessing) or ~2,400 sit/min
  (batch solver, single-core)
- GPU throughput: ~5,000-15,000+ sit/min (RTX 3060 Ti, batch solver)
- Neural net inference: ~1ms per decision on CPU
- Dataset generation: ``--device cpu`` (VPS) or ``--device cuda`` (GPU PC)
- Both machines can run simultaneously — merge JSONL files afterwards
- Genetic algo: games can run in parallel across cores
