"""Recover known Elo differences from simulated round robin results."""

import math
import random
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from round_robin import bradley_terry, score_for


def simulate(true_elo: Mapping[str, float], games: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    names = list(true_elo)
    records: list[dict[str, object]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for game in range(games):
                white = left if game % 2 == 0 else right
                black = right if game % 2 == 0 else left
                chance = 1.0 / (1.0 + 10 ** ((true_elo[black] - true_elo[white]) / 400.0))
                # the draw band has to shrink near the extremes or the expected score of a
                # hopeless side is the draw rate rather than its true chance
                draw = min(0.30, 2.0 * min(chance, 1.0 - chance))
                roll = rng.random()
                if roll < chance - draw / 2.0:
                    result = "white"
                elif roll < chance + draw / 2.0:
                    result = "draw"
                else:
                    result = "black"
                records.append(
                    {"left": left, "right": right, "game": game, "white": white, "result": result}
                )
    return records


def faults_from(text: str) -> dict[str, int]:
    """Read the agent name and the faults column out of a printed standings table."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 10 and parts[0].isdigit():
            found[parts[1]] = int(parts[-1])
    return found


class EloFitTest(unittest.TestCase):
    def test_recovers_spread_ratings(self) -> None:
        true_elo = {f"a{i}": value for i, value in enumerate([-600, -300, -100, 0, 150, 400, 700])}
        centred = sum(true_elo.values()) / len(true_elo)
        records = simulate(true_elo, 100, seed=7)
        fitted, _ = bradley_terry(records, list(true_elo))
        for name, value in true_elo.items():
            self.assertAlmostEqual(fitted[name], value - centred, delta=90, msg=f"{name} {fitted}")

    def test_order_is_preserved(self) -> None:
        true_elo = {f"a{i}": i * 120.0 for i in range(10)}
        records = simulate(true_elo, 60, seed=11)
        fitted, _ = bradley_terry(records, list(true_elo))
        order = sorted(fitted, key=lambda n: fitted[n])
        self.assertEqual(order, list(true_elo))

    def test_stable_with_two_games(self) -> None:
        records = simulate({"a": 0.0, "b": 0.0, "c": 0.0}, 2, seed=3)
        fitted, _ = bradley_terry(records, ["a", "b", "c"])
        for value in fitted.values():
            self.assertTrue(math.isfinite(value) and abs(value) < 1500, fitted)

    def test_an_unbeaten_agent_stays_finite(self) -> None:
        records = [
            {
                "left": "a",
                "right": "b",
                "game": g,
                "white": "a" if g % 2 == 0 else "b",
                "result": "white" if g % 2 == 0 else "black",
            }
            for g in range(100)
        ]
        fitted, _ = bradley_terry(records, ["a", "b"])
        self.assertTrue(math.isfinite(fitted["a"]))
        self.assertGreater(fitted["a"], fitted["b"])

    def test_score_for_reads_colour(self) -> None:
        record = {"left": "a", "right": "b", "game": 1, "white": "b", "result": "white"}
        self.assertEqual(score_for(record, "b"), 1.0)
        self.assertEqual(score_for(record, "a"), 0.0)
        record["result"] = "draw"
        self.assertEqual(score_for(record, "a"), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ReportTest(unittest.TestCase):
    """The report has to survive every termination, including the ones that score a fault."""

    def test_report_runs_over_every_termination(self) -> None:
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        import round_robin

        rows = [
            {
                "left": "a",
                "right": "b",
                "game": 0,
                "white": "a",
                "result": "white",
                "termination": "checkmate",
                "seconds": 1.0,
            },
            {
                "left": "a",
                "right": "b",
                "game": 1,
                "white": "b",
                "result": "white",
                "termination": "flag",
                "seconds": 1.0,
            },
            {
                "left": "a",
                "right": "c",
                "game": 0,
                "white": "a",
                "result": "draw",
                "termination": "adjudication",
                "seconds": 1.0,
            },
            {
                "left": "b",
                "right": "c",
                "game": 0,
                "white": "b",
                "result": "void",
                "termination": "both_failed",
                "seconds": 1.0,
            },
            {
                "left": "b",
                "right": "c",
                "game": 1,
                "white": "c",
                "result": "black",
                "termination": "crash",
                "seconds": 1.0,
            },
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                round_robin.report(path, ["a", "b", "c"])
        text = buffer.getvalue()
        self.assertIn("5 games played", text)
        self.assertIn("flag 1", text)
        # a flagged, c crashed, and both_failed is charged to both b and c
        self.assertEqual(faults_from(text), {"a": 1, "b": 1, "c": 2})

    def test_faults_land_on_the_agent_that_failed(self) -> None:
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        import round_robin

        # b is white in every game and loses every one of them to a flag
        rows = [
            {
                "left": "a",
                "right": "b",
                "game": 1,
                "white": "b",
                "result": "black",
                "termination": "flag",
                "seconds": 1.0,
            }
            for _ in range(3)
        ]
        for index, row in enumerate(rows):
            row["game"] = 1 + 2 * index
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                round_robin.report(path, ["a", "b"])
        self.assertEqual(faults_from(buffer.getvalue()), {"a": 0, "b": 3})


class ErrorBarTest(unittest.TestCase):
    def test_error_shrinks_as_games_are_added(self) -> None:
        true_elo = {"a": 0.0, "b": 100.0, "c": 200.0}
        _, few = bradley_terry(simulate(true_elo, 4, seed=5), list(true_elo))
        _, many = bradley_terry(simulate(true_elo, 400, seed=5), list(true_elo))
        for name in true_elo:
            self.assertLess(many[name], few[name] / 5.0, f"{name} {few} {many}")

    def test_error_is_finite_and_positive(self) -> None:
        records = simulate({"a": 0.0, "b": 0.0}, 2, seed=1)
        _, errors = bradley_terry(records, ["a", "b"])
        for value in errors.values():
            self.assertGreater(value, 0.0)
            self.assertTrue(math.isfinite(value))


class PlyCountTest(unittest.TestCase):
    def test_counts_half_moves(self) -> None:
        import chess
        import chess.pgn

        from round_robin import count_plies

        for moves in ([], ["e2e4"], ["e2e4", "e7e5", "g1f3"], ["f2f3", "e7e5", "g2g4", "d8h4"]):
            board = chess.Board()
            for uci in moves:
                board.push(chess.Move.from_uci(uci))
            pgn = str(chess.pgn.Game.from_board(board))
            self.assertEqual(count_plies(pgn), len(moves), pgn)


class HealthFilterTest(unittest.TestCase):
    def test_a_game_where_a_side_never_searched_is_dropped(self) -> None:
        from round_robin import searched

        played = {"health": {"a": [4.0, 0.30], "b": [61.0, 0.28]}}
        idle = {"health": {"a": [4.0, 0.30], "b": [61.0, 0.00]}}
        self.assertTrue(searched(played, 0.05))
        self.assertFalse(searched(idle, 0.05))
        # with no floor asked for, every game counts, health data or not
        self.assertTrue(searched(idle, 0.0))
        self.assertTrue(searched({}, 0.0))
        self.assertFalse(searched({}, 0.05))

    def test_the_filter_shrinks_the_table(self) -> None:
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        import round_robin

        rows = []
        for game in range(4):
            median = 0.0 if game == 3 else 0.3
            rows.append(
                {
                    "left": "a",
                    "right": "b",
                    "game": game,
                    "white": "a" if game % 2 == 0 else "b",
                    "result": "white",
                    "termination": "checkmate",
                    "health": {"a": [1.0, 0.3], "b": [1.0, median]},
                }
            )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for floor, expected in ((0.0, "4 games played"), (0.05, "3 games played")):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    round_robin.report(path, ["a", "b"], floor)
                self.assertIn(expected, buffer.getvalue())
