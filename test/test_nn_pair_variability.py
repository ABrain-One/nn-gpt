"""A default pair request must return a new combination every time.

JoinConf.pair_seed=None orders the partner candidates at random; an integer
reproduces one shuffle. ab.nn.api.data is lru_cached, so the generation loops
clear it before asking again and so does this file.

Scope: LEMUR_TASK, LEMUR_DATASET, LEMUR_METRIC, LEMUR_PAIR_CALLS (default 5),
LEMUR_PAIR_ROWS (default 200).
"""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASK = os.getenv("LEMUR_TASK", "img-classification")
DATASET = os.getenv("LEMUR_DATASET", "cifar-10")
METRIC = os.getenv("LEMUR_METRIC", "acc")

PAIR_CALLS = int(os.getenv("LEMUR_PAIR_CALLS", "3"))
PAIR_ROWS = int(os.getenv("LEMUR_PAIR_ROWS", "200"))

# As ab/gpt/conf/prompt/train/NN_gen.json states it.
PAIR_CONFIG = {
    "num_joint_nns": 2,
    "keep_same": ["epoch", "metric", "dataset", "task"],
    "no_repeat": ["nn"],
    "improve": True,
}

# Loops that re-query LEMUR, so each has to drop the lru_cache first.
GENERATION_LOOPS = (
    "ab/gpt/util/tune/Tune.py",
    "ab/gpt/util/tune/Tune_Curriculum.py",
    "ab/gpt/util/tune/Tune_Onnx.py",
)


def _join_conf(**overrides):
    """A JoinConf shaped exactly like the one the pair prompts produce."""
    from ab.nn.api import JoinConf

    fields = dict(
        num_joint_nns=PAIR_CONFIG["num_joint_nns"],
        same_columns=tuple(PAIR_CONFIG["keep_same"]),
        diff_columns=tuple(PAIR_CONFIG["no_repeat"]),
        enhance_nn=PAIR_CONFIG["improve"],
    )
    fields.update(overrides)
    return JoinConf(**fields)


def _pairs(frame) -> list[tuple]:
    """The (first network, second network) combination of every row."""
    return list(zip(frame["nn"], frame["nn_2"]))


def _partner_of(frame) -> dict:
    """Partner per baseline row, keyed by what identifies that row."""
    keys = [c for c in ("nn", "epoch", "prm_id") if c in frame.columns]
    return {tuple(row[c] for c in keys): row["nn_2"]
            for _, row in frame.iterrows()}


class TestNNPairRequestIsNotFixed(unittest.TestCase):
    """What the shipped configs and the API guarantee, without a database."""

    def test_the_shipped_pair_prompts_ask_for_pairs_without_a_seed(self):
        """No prompt config pins pair_seed, so every one of them shuffles."""
        configs = sorted((ROOT / "ab/gpt/conf/prompt").rglob("*.json"))
        self.assertTrue(configs, "no prompt configs found")

        pair_keys, seeded = 0, []
        for path in configs:
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(content, dict):
                continue
            for key, cfg in content.items():
                if not isinstance(cfg, dict) or cfg.get("num_joint_nns") != 2:
                    continue
                pair_keys += 1
                if cfg.get("pair_seed") is not None:
                    seeded.append(f"{path.relative_to(ROOT)}::{key}")

        self.assertTrue(pair_keys, "no config asks for NN pairs any more")
        self.assertEqual(
            [], seeded,
            "a fixed pair_seed makes every run reuse one pairing: " +
            ", ".join(seeded))

    def test_a_pair_request_defaults_to_an_unseeded_shuffle(self):
        """JoinConf itself must not acquire a default seed."""
        try:
            conf = _join_conf()
        except ImportError as exc:
            self.skipTest(f"ab.nn (nn-dataset) is unavailable: {exc}")
        self.assertIsNone(
            conf.pair_seed,
            "JoinConf now defaults to a fixed pair_seed, so every call would "
            "pair the same networks")

    def test_the_generation_loops_drop_the_query_cache_before_asking_again(self):
        """ab.nn.api.data is lru_cached: without a clear, nothing can change."""
        missing = [
            path for path in GENERATION_LOOPS
            if not re.search(r"\bdata\.cache_clear\(\)",
                             (ROOT / path).read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            [], missing,
            "these loops re-query LEMUR without clearing the lru_cache, so "
            "every pass is served the first pass's pairing: " +
            ", ".join(missing))


class TestNNPairVariability(unittest.TestCase):
    """Repeated default pair requests must return new combinations.

    PAIR_CALLS queries are made once and shared by the tests below: the join is
    the expensive part, so it is not repeated per test.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import ab.nn.api as lemur
        except ImportError as exc:
            raise unittest.SkipTest(f"ab.nn (nn-dataset) is unavailable: {exc}")

        cls.lemur = lemur
        try:
            cls.frames = [cls._query(_join_conf()) for _ in range(PAIR_CALLS)]
        except Exception as exc:
            raise unittest.SkipTest(f"the LEMUR database is unavailable: {exc}")

        probe = cls.frames[0]
        if "nn_2" not in getattr(probe, "columns", ()):
            raise unittest.SkipTest(
                "the pair request did not return a paired (wide) frame; "
                f"columns={list(getattr(probe, 'columns', []))}")
        if len(probe) < 2 or len(set(probe["nn"])) < 3:
            raise unittest.SkipTest(
                f"too little paired data to shuffle: {len(probe)} row(s), "
                f"{len(set(probe['nn']))} distinct network(s) for "
                f"task={TASK} dataset={DATASET} metric={METRIC}")

    @classmethod
    def _query(cls, conf):
        """One pair request, made the way the generation loops make it."""
        cls.lemur.data.cache_clear()
        return cls.lemur.data(
            sql=conf,
            include_nn_stats=False,
            task=TASK,
            dataset=DATASET,
            metric=METRIC,
            max_rows=PAIR_ROWS,
        )

    def test_each_pair_request_returns_a_new_combination(self):
        """The same default request, repeated, must not repeat its pairing."""
        seen: dict[tuple, int] = {}
        for call, frame in enumerate(self.frames):
            combination = tuple(_pairs(frame))
            self.assertTrue(
                combination, f"call {call + 1} returned no paired rows")
            if combination in seen:
                self.fail(
                    f"call {call + 1} returned the combination of call "
                    f"{seen[combination] + 1} verbatim: a default pair request "
                    f"is expected to pair the rows afresh every time "
                    f"({len(combination)} rows, e.g. {combination[:3]})")
            seen[combination] = call

        self.assertEqual(
            PAIR_CALLS, len(seen),
            f"expected {PAIR_CALLS} distinct combinations, got {len(seen)}")

    def test_the_partner_of_a_network_changes_between_calls(self):
        """Not just the row order: the partner itself has to move."""
        partners: dict[str, set] = {}
        for frame in self.frames:
            for first, second in _pairs(frame):
                partners.setdefault(first, set()).add(second)

        eligible = {s for seconds in partners.values() for s in seconds}
        if len(eligible) < 2:
            self.skipTest("only one network is ever eligible as a partner, so "
                          "the pairing is forced by the data, not by the code")

        moved = [nn for nn, seconds in partners.items() if len(seconds) > 1]
        self.assertTrue(
            moved,
            f"over {PAIR_CALLS} calls no network was ever paired with a "
            f"different partner, so the pairing is fixed rather than shuffled "
            f"({len(partners)} networks seen)")

    def test_the_two_networks_of_a_pair_differ(self):
        """no_repeat: ["nn"] must still hold while the pairing shuffles."""
        same = [pair for pair in _pairs(self.frames[0]) if pair[0] == pair[1]]
        self.assertEqual(
            [], same[:5],
            f"{len(same)} pair(s) joined a network with itself")

    def test_a_repeat_without_a_clear_is_served_from_the_cache(self):
        """Why the loops clear: an identical call is answered, not re-run."""
        conf = _join_conf()
        first = self._query(conf)
        second = self.lemur.data(  # same arguments, no clear
            sql=conf,
            include_nn_stats=False,
            task=TASK,
            dataset=DATASET,
            metric=METRIC,
            max_rows=PAIR_ROWS,
        )
        self.assertEqual(
            _pairs(first), _pairs(second),
            "ab.nn.api.data no longer caches identical pair requests; if that "
            "is intended, the cache_clear() calls in the generation loops are "
            "now redundant and this test should be dropped")

    def test_a_seed_fixes_the_partner_each_baseline_gets(self):
        """A seed reproduces the choice of partner, baseline by baseline.

        Not the frame: the rows are drawn from a randomly ordered temp table,
        so which baselines come back, and in which order, differs anyway.
        """
        first = _partner_of(self._query(_join_conf(pair_seed=12345)))
        second = _partner_of(self._query(_join_conf(pair_seed=12345)))

        shared = set(first) & set(second)
        if not shared:
            self.skipTest("the two seeded calls returned no baseline in common")

        differing = {key: (first[key], second[key])
                     for key in shared if first[key] != second[key]}
        self.assertEqual(
            {}, dict(list(differing.items())[:5]),
            f"pair_seed=12345 gave {len(differing)} of {len(shared)} shared "
            f"baselines a different partner, so a run that has to repeat "
            f"exactly cannot")


if __name__ == "__main__":
    unittest.main()
