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


class EloFitTest(unittest.TestCase):
    def test_recovers_spread_ratings(self) -> None:
        true_elo = {f"a{i}": value for i, value in enumerate([-600, -300, -100, 0, 150, 400, 700])}
        centred = sum(true_elo.values()) / len(true_elo)
        records = simulate(true_elo, 100, seed=7)
        fitted = bradley_terry(records, list(true_elo))
        for name, value in true_elo.items():
            self.assertAlmostEqual(fitted[name], value - centred, delta=90, msg=f"{name} {fitted}")

    def test_order_is_preserved(self) -> None:
        true_elo = {f"a{i}": i * 120.0 for i in range(10)}
        records = simulate(true_elo, 60, seed=11)
        fitted = bradley_terry(records, list(true_elo))
        order = sorted(fitted, key=lambda n: fitted[n])
        self.assertEqual(order, list(true_elo))

    def test_stable_with_two_games(self) -> None:
        records = simulate({"a": 0.0, "b": 0.0, "c": 0.0}, 2, seed=3)
        fitted = bradley_terry(records, ["a", "b", "c"])
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
        fitted = bradley_terry(records, ["a", "b"])
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
