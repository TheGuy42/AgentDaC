# chess_perst

Train a `PersistentAgent` to play chess by predicting the **next move** for a given
position. Each rollout is a single step: the agent outputs one move, and the reward is a
**local chess engine's** evaluation of the resulting position.

## How it works

- **Task**: every sample is a single chess position (`fen`). The agent is shown the
  board, the FEN, and the legal moves, and must output one move in UCI notation
  (e.g. `g1f3`). See [format.py](format.py).
- **Agent**: `PersistentAgent` (same setup as `math_perst` — it may think and delegate to
  a persistent sub-agent before answering). See [trainer.py](trainer.py).
- **Reward** (see [chess_engine/engine.py](chess_engine/engine.py)): the engine's score of
  the board **after** the proposed move, from the moving side's perspective. By default
  (`reward: "win_prob"`) the centipawn score is squashed to an expected score in `[0, 1]`
  via the logistic `1 / (1 + 10^(-cp/400))` — bounded and well-behaved for GRPO advantages.
  Set `reward: "centipawns"` for the raw score instead. An illegal/unparseable move has no
  resulting position and scores `illegal_score`.

## Engine (local)

Reward computation runs against a local UCI engine — **Stockfish** by default. Each rollout
**runner subprocess** lazily starts **one** engine process and reuses it across all of its
rollouts (`MoveEvaluator.for_process` in
[chess_engine/evaluator.py](chess_engine/evaluator.py)), so you get `train_config.n_runners`
engine processes in total. Access to each engine is serialized, and the blocking search runs
off the event loop ([chess_engine/engine.py](chess_engine/engine.py)).

Settings live in [defaults/engine_config.json](defaults/engine_config.json):

```json
{
    "engine_path": "stockfish",
    "depth": 12,
    "threads": 1,
    "hash_mb": 64,
    "reward": "win_prob",
    "illegal_score": 0.0
}
```

`engine_path` may be a binary on `PATH` or an absolute path; any UCI engine works.
`threads`/`hash_mb` are per engine — keep `threads` low since there are `n_runners` engines.

## Data

Positions are streamed from public Lichess datasets on the Hugging Face Hub by
[data/loader.py](data/loader.py) and normalized to a common schema: each row has a `fen`
(the non-terminal position to move in) and the `ply` it occurs at. Choose one or more
sources with `--datasets` (pooled, deduplicated by FEN, then split into disjoint train/val):

- `lichess-puzzles` — tactical positions from
  [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) (~5.9M).
  The dataset's FEN is the position *before* the opponent's setup move, so we apply the
  first solution move to reach the position the solver faces. Filterable by Glicko-2 rating
  via `--min_rating` / `--max_rating`. **Default.**
- `lichess-openings` — opening positions from
  [Lichess/chess-openings](https://huggingface.co/datasets/Lichess/chess-openings) (the ECO
  book, ~3.7k); the `uci` line is replayed from the start so move counters are correct.
- `lichess-evals` — Stockfish-analyzed positions from
  [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
  (~342M unique). Its `fen` is an EPD (no move counters), so it's padded to a full FEN and
  `ply` only reflects the side to move. We stream a subset (`--num_train` + `--num_val`),
  never the whole thing.

Sizes come from `--num_train` / `--num_val`; `--data_seed` makes streaming/shuffling
reproducible. Nothing is written to disk (the Hub keeps its own cache). The reward is still
computed by the local engine (see below) — these datasets only supply positions.

## Dependencies

- **python-chess** (already a project dependency) for board logic, move parsing, and the
  UCI engine protocol.
- **A Stockfish binary** (the only extra requirement):
  ```bash
  apt-get install stockfish        # Debian/Ubuntu
  # or download from https://stockfishchess.org/download/ and set engine_path
  ```

## Configuration (`defaults/`)

- `engine_config.json` — engine + reward settings (above).
- `prompt_config.json`, `decomp_config.json`, `rollout_config.json`, `train_config.json`,
  `extra_config.json`, `verl_config.json` — shared training/agent config (copied from
  `math_perst` and tuned).
