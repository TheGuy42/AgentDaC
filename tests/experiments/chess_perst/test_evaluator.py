"""Tests for the chess move evaluator, the root-MultiPV cache, and the reward shaping.

Async tests run via `pytest-async` (a bare `async def test_...` is executed in its own event
loop — no decorator needed). Cache-logic tests use a `FakeEngine` and need no Stockfish; the
integration tests are skipped when no `stockfish` binary is on PATH.

Engine deployment (binary, threads, hash) lives in :class:`EngineConfig`; the per-analysis
search budget and reward policy live in :class:`ChessConfig`, which is passed to each
`score()` / `build()` call.
"""
from __future__ import annotations

import asyncio
import shutil

import chess
import pytest
from chess.engine import Cp, Mate, PovScore

from experiments.chess.chess_engine import ChessConfig, EngineConfig, MoveEvaluator
from experiments.chess.chess_engine.cache import RootCache
from experiments.chess.chess_engine.evaluator import MoveResult
from experiments.chess.rewards import compute_reward, compute_wp, parse_move


START_FEN = chess.STARTING_FEN
# White to move, Black queen on d4 hangs to Qd1xd4; a quiet king move lets ...Qxd1.
FREE_QUEEN_FEN = "4k3/8/8/8/3q4/8/8/3QK3 w - - 0 1"
# Black to move: ...Re1 is back-rank mate.
MATE_FEN = "4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"

requires_stockfish = pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not on PATH")


def chess_config(**overrides) -> ChessConfig:
    """A minimal-budget :class:`ChessConfig` (1-node search) for cache/reward unit tests."""
    return ChessConfig.model_validate({"limit": {"nodes": 1}, **overrides})


# --------------------------------------------------------------------------- #
# Pure / fake-engine unit tests (no Stockfish)
# --------------------------------------------------------------------------- #


class FakeEngine:
    """Stand-in for LocalEngine: returns canned MultiPV infos and counts/optionally fails calls."""

    def __init__(self, move_cps: dict[str, int], delay: float = 0.0) -> None:
        self.move_cps = move_cps
        self.delay = delay
        self.calls = 0
        self.fail_next = 0

    async def analyse(self, board, config, *, multipv=None, root_moves=None, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("boom")
        return [
            {"pv": [chess.Move.from_uci(uci)], "score": PovScore(Cp(cp), board.turn)}
            for uci, cp in self.move_cps.items()
        ]


def test_parse_move():
    board = chess.Board()
    assert parse_move(board, "e2e4") == chess.Move.from_uci("e2e4")  # uci
    assert parse_move(board, r"\boxed{e2e4}") == chess.Move.from_uci("e2e4")  # boxed wrapper
    assert parse_move(board, "I play g1f3 here") == chess.Move.from_uci("g1f3")  # single embedded uci
    assert parse_move(board, "Nf3") is None  # san not accepted (uci-only)
    assert parse_move(board, "e2e4 d2d4") is None  # ambiguous: two legal moves
    assert parse_move(board, "e2e5") is None  # illegal
    assert parse_move(board, "") is None


def test_for_validation_overwrites():
    """`for_validation` applies `validation_overwrites` on top of the base config, leaving it intact."""
    cfg = chess_config(
        mode="root_multipv",
        deterministic=False,
        validation_overwrites={"limit": {"time": 0.2}, "mode": "child", "deterministic": True},
    )
    val = cfg.for_validation()

    assert val.mode == "child" and val.deterministic is True
    assert val.limit.time == 0.2
    assert val.reward == cfg.reward  # untouched fields inherited
    # base config is not mutated
    assert cfg.mode == "root_multipv" and cfg.deterministic is False

    # no overwrites -> an equivalent copy
    assert chess_config().for_validation().model_dump() == chess_config().model_dump()


async def test_cache_dedup_single_search():
    """Concurrent requests for one FEN trigger exactly one engine search."""
    fake = FakeEngine({"e2e4": 30, "d2d4": 25, "g1f3": 20}, delay=0.02)
    cache = RootCache(fake, cache_size=10)  # type: ignore[arg-type]
    cfg = chess_config()

    results = await asyncio.gather(*[cache.build(START_FEN, cfg) for _ in range(5)])

    assert fake.calls == 1
    for scores in results:
        assert set(scores) == {"e2e4", "d2d4", "g1f3"}
        assert scores["e2e4"].pov(chess.WHITE) == Cp(30)  # White to move in START_FEN


async def test_cache_lru_eviction():
    fake = FakeEngine({"e2e4": 10})
    cache = RootCache(fake, cache_size=2)  # type: ignore[arg-type]
    cfg = chess_config()
    fens = [START_FEN, FREE_QUEEN_FEN, MATE_FEN]

    for fen in fens:
        await cache.build(fen, cfg)

    assert len(cache._tasks) == 2
    assert START_FEN not in cache._tasks  # oldest evicted
    assert MATE_FEN in cache._tasks


async def test_cache_retry_after_failure():
    """A failed search is evicted (not poisoned) so the next call recomputes."""
    fake = FakeEngine({"e2e4": 10})
    fake.fail_next = 1
    cache = RootCache(fake, cache_size=10)  # type: ignore[arg-type]
    cfg = chess_config()

    with pytest.raises(RuntimeError):
        await cache.build(START_FEN, cfg)
    assert START_FEN not in cache._tasks  # not poisoned

    scores = await cache.build(START_FEN, cfg)  # retried
    assert fake.calls == 2
    assert scores["e2e4"].pov(chess.WHITE) == Cp(10)


def test_compute_reward_illegal():
    """An illegal move scores strictly below the legal floor of each scheme."""
    wp_cfg = chess_config(reward="win_prob")
    cp_cfg = chess_config(reward="centipawns")
    illegal = MoveResult(fen=START_FEN, parse_success=False)

    assert compute_reward(illegal, wp_cfg) == -wp_cfg.mate_margin
    assert compute_reward(illegal, wp_cfg) < compute_wp(Mate(-1), wp_cfg)  # below "get mated"
    assert compute_reward(illegal, cp_cfg) < 0.0


def test_compute_wp_ordering():
    cfg = chess_config(reward="win_prob")
    r = lambda s: compute_wp(s, cfg)  # noqa: E731
    # deliver-mate > winning cp > equal > losing cp > get-mated; faster mate ranks higher
    assert r(Mate(1)) >= r(Mate(3)) > r(Cp(2000)) > r(Cp(0)) > r(Cp(-2000)) > r(Mate(-3)) >= r(Mate(-1))
    assert r(Cp(0)) == 0.5
    assert all(0.0 <= r(Mate(-n)) < 0.05 < r(Cp(0)) < 0.95 < r(Mate(n)) <= 1.0 for n in range(1, 6))


# --------------------------------------------------------------------------- #
# Integration tests (real Stockfish)
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_evaluator():
    """Factory for fresh evaluators on a deterministic engine (Threads=1); closes engines after.

    Each `score()`/`build()` call takes its own :class:`ChessConfig`, so one evaluator can
    serve any mix of modes and reward schemes — build those configs with `make_chess_config`.
    """
    created: list[MoveEvaluator] = []

    def _make(**overrides) -> MoveEvaluator:
        cfg = EngineConfig.model_validate({"threads": 1, **overrides})
        ev = MoveEvaluator(cfg)
        created.append(ev)
        return ev

    yield _make
    for ev in created:
        ev.close()


def make_chess_config(**overrides) -> ChessConfig:
    """A deterministic per-analysis config (node-limited) for the integration tests."""
    return ChessConfig.model_validate({"limit": {"nodes": 120_000}, **overrides})


@requires_stockfish
async def test_illegal_move_scores(make_evaluator):
    ev = make_evaluator()
    cfg = make_chess_config(reward="win_prob")
    board = chess.Board(START_FEN)
    move = parse_move(board, "e2e9")  # unparseable / illegal -> None
    result = await ev.score(board, move, cfg)
    assert result.parse_success is False
    assert result.score is None and result.cp is None
    assert compute_reward(result, cfg) == -cfg.mate_margin


@requires_stockfish
async def test_child_sign(make_evaluator):
    """Capturing a free queen must out-score a quiet move, with positive cp for the mover."""
    ev = make_evaluator()
    cfg = make_chess_config(mode="child")
    board = chess.Board(FREE_QUEEN_FEN)
    cap = await ev.score(board, parse_move(board, "d1d4"), cfg)
    quiet = await ev.score(board, parse_move(board, "e1f1"), cfg)
    assert cap.cp > 0
    assert compute_reward(cap, cfg) > compute_reward(quiet, cfg)


@requires_stockfish
async def test_root_multipv_covers_all_legal(make_evaluator):
    ev = make_evaluator()
    cfg = make_chess_config(mode="root_multipv")
    board = chess.Board(FREE_QUEEN_FEN)
    scores = await ev.root_cache.build(FREE_QUEEN_FEN, cfg)
    assert set(scores) == {m.uci() for m in board.legal_moves}


@requires_stockfish
async def test_root_multipv_dedup_one_search(make_evaluator):
    """Scoring many moves of one FEN concurrently runs a single engine search."""
    ev = make_evaluator()
    cfg = make_chess_config(mode="root_multipv")
    board = chess.Board(FREE_QUEEN_FEN)
    moves = list(board.legal_moves)[:5]

    calls = {"n": 0}
    original = ev.engine.analyse

    async def counting(*args, **kwargs):
        calls["n"] += 1
        return await original(*args, **kwargs)

    ev.engine.analyse = counting  # type: ignore[method-assign]
    await asyncio.gather(*[ev.score(chess.Board(FREE_QUEEN_FEN), mv, cfg) for mv in moves])
    assert calls["n"] == 1


@requires_stockfish
async def test_root_and_child_agree(make_evaluator):
    """child and root_multipv agree on ordering and (loosely) on value for the same moves."""
    ev = make_evaluator()
    child_cfg = make_chess_config(mode="child", reward="win_prob")
    root_cfg = make_chess_config(mode="root_multipv", reward="win_prob")
    board = chess.Board(FREE_QUEEN_FEN)

    good, bad = parse_move(board, "d1d4"), parse_move(board, "e1f1")  # take the queen vs. drift
    rc_good = compute_reward(await ev.score(board, good, child_cfg), child_cfg)
    rc_bad = compute_reward(await ev.score(board, bad, child_cfg), child_cfg)
    rr_good = compute_reward(await ev.score(board, good, root_cfg), root_cfg)
    rr_bad = compute_reward(await ev.score(board, bad, root_cfg), root_cfg)

    assert rc_good > rc_bad and rr_good > rr_bad  # same ordering
    assert abs(rc_good - rr_good) < 0.2 and abs(rc_bad - rr_bad) < 0.2  # loosely equal


@requires_stockfish
@pytest.mark.parametrize("mode", ["child", "root_multipv"])
async def test_mate_move_top_band(make_evaluator, mode):
    ev = make_evaluator()
    cfg = make_chess_config(mode=mode, reward="win_prob")
    board = chess.Board(MATE_FEN)
    result = await ev.score(board, parse_move(board, "e8e1"), cfg)  # ...Re1#
    assert result.score.is_mate() and result.score.mate() > 0  # Mate(1), not MateGiven
    assert compute_reward(result, cfg) > 0.95  # deliver-mate band
