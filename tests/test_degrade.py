"""R15.3 degrade ladder and the SAFE-only baseline switch (main.py).

    python3 -m unittest tests.test_degrade -v
"""
from __future__ import annotations

import os
import sys
import time
import types
import unittest
from collections import deque
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import main  # noqa: E402
from test_foundation import load_stream  # noqa: E402


class Ladder:
    """main.Bot._degrade_update on a bare stand-in: only the ladder state, no pipeline."""

    def __init__(self, C):
        self.C = C
        self.runner_info = None
        self.tlm = None
        self.degrade_level = 0
        self._win = deque()
        self._hard_seen = 0
        self._l2_hard = 0
        self._prev_late = False
        self.applied = []

    def _apply_degrade(self):
        self.applied.append(self.degrade_level)

    def feed(self, el_ms, n=1):
        for _ in range(n):
            main.Bot._degrade_update(self, el_ms)
        return self.degrade_level


class DegradeLadderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hello, _, _ = load_stream()
        cls.C = main.config.bind(hello["config"])
        cls.fast = 5.0
        cls.hard = cls.C.HARD_DEADLINE_MS + 5.0
        cls.N = int(cls.C.DEGRADE_WINDOW_N)

    def test_three_hard_ticks_in_a_row_stop_at_l2(self):
        """The review's case: three slow ticks in a row used to put the legs on SAFE for >= 10 s."""
        lad = Ladder(self.C)
        self.assertEqual(lad.feed(self.hard), 1)
        self.assertEqual(lad.feed(self.hard), 2)
        self.assertEqual(lad.feed(self.hard), 2)                # first hard-late tick at L2: still L2
        self.assertEqual(lad.feed(self.hard), 3)                # DEGRADE_L3_HARD_N of them: L3

    def test_isolated_spikes_never_reach_l3(self):
        lad = Ladder(self.C)
        lad.feed(self.hard)
        lad.feed(self.hard)
        self.assertEqual(lad.degrade_level, 2)
        for _ in range(6):                                       # one spike per window, forever
            lad.feed(self.hard)
            lad.feed(self.fast, self.N)
            self.assertLess(lad.degrade_level, 3)

    def test_soft_overrun_stops_at_l2(self):
        lad = Ladder(self.C)
        slow = self.C.SOFT_DEADLINE_MS + 2.0                     # p95 over soft, never hard-late
        lad.feed(slow, self.N * 6)
        self.assertEqual(lad.degrade_level, 2)

    def test_l3_is_left_after_the_short_window(self):
        lad = Ladder(self.C)
        lad.feed(self.hard, 4)
        self.assertEqual(lad.degrade_level, 3)
        k = int(self.C.DEGRADE_L3_EXIT_N)
        self.assertLess(k, self.N)
        self.assertEqual(lad.feed(self.fast, k - 1), 3)
        self.assertEqual(lad.feed(self.fast), 2)                 # L3 -> L2 after DEGRADE_L3_EXIT_N clean ticks
        self.assertEqual(lad.feed(self.fast, self.N - 1), 2)
        self.assertEqual(lad.feed(self.fast), 1)                 # the cheaper levels keep the full window
        self.assertEqual(lad.feed(self.fast, self.N), 0)
        self.assertEqual(lad.applied, [1, 2, 3, 2, 1, 0])

    def test_soft_overrun_does_not_hold_l3(self):
        """L3 costs more CPU than L2 (the pipeline plus SAFE), so a p95 between soft and hard would have
        kept the legs on SAFE for the rest of the match."""
        lad = Ladder(self.C)
        lad.feed(self.hard, 4)
        self.assertEqual(lad.degrade_level, 3)
        slow = self.C.SOFT_DEADLINE_MS + 2.0
        self.assertEqual(lad.feed(slow, int(self.C.DEGRADE_L3_EXIT_N)), 2)
        lad.feed(slow, self.N * 3)
        self.assertEqual(lad.degrade_level, 2)

    def test_one_late_tick_is_counted_once(self):
        """The runner reports a replaced late answer one act() after that act() counted itself."""
        lad = Ladder(self.C)
        lad.runner_info = types.SimpleNamespace(hard_overruns=0)
        lad.feed(self.hard)                                      # L1 (counted by its own time)
        lad.runner_info.hard_overruns = 1
        self.assertEqual(lad.feed(self.fast), 1)                 # the runner's echo of the same tick
        lad.feed(self.hard)                                      # L2
        lad.runner_info.hard_overruns = 2
        self.assertEqual(lad.feed(self.fast), 2)
        lad.feed(self.hard)                                      # first hard-late tick at L2
        lad.runner_info.hard_overruns = 3
        self.assertEqual(lad.feed(self.fast), 2)                 # its echo must not be the second
        lad.runner_info.hard_overruns = 4                        # a late answer act() did not see itself
        self.assertEqual(lad.feed(self.fast), 3)


class SafeOnlyBaselineTest(unittest.TestCase):
    def test_env_switch_keeps_the_bot_on_safe(self):
        hello, obs, _ = load_stream()
        with mock.patch.dict(os.environ, {"BOT_SAFE_ONLY": "1"}):
            bot = main.Bot(hello, time.perf_counter() + 15)
        self.assertFalse(bot.ready)
        self.assertIsNone(bot.world)
        o = obs[0]
        act = bot.act(o, time.perf_counter())
        self.assertEqual(act, main.safe.act(o, bot.C))
        self.assertIn("move", act)


if __name__ == "__main__":
    unittest.main()
