"""M7 DROP: CAPTURE entry, the landing-time timeout (found by tracing minisim seed 501).

    python3 -m unittest tests.test_drop -v
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import Harness, goal_exits  # noqa: E402

import tactics_drop  # noqa: E402
import test_world as TW  # noqa: E402
import world as W  # noqa: E402
from contracts import Clock, DropState, ExitMemory, Steer  # noqa: E402


def _go(ctx, d):
    return True, False


class CaptureEntryTest(unittest.TestCase):
    def _landed(self, dist):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        h.world.drops[20000] = DropState(20000, (dist, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 70.0, 99.0)
        return h

    def test_landed_drop_far_away_is_committed_and_follows_the_path(self):
        """CAPTURE used to walk a straight line at the drop, into whatever stood in the way; it now takes
        its direction from the nav path (the fake nav answers with a detour heading)."""
        h = self._landed(235.0)
        h.nav.follow = lambda me, key, goal: Steer((0.0, 1.0))      # the path leads north around a wall
        with mock.patch.object(tactics_drop, "go_test", _go):
            for _ in range(3):
                intent, action = h.step()
            self.assertEqual((h.modes.goal, h.modes.sub), ("M7", "CAPTURE"))
            self.assertEqual(intent.rule, "R8.5")
            self.assertGreater(action["move"][1], 0.9)               # north along the path, not east at the drop

    def test_inside_the_ring_moves_to_the_center(self):
        h = self._landed(7.0)
        with mock.patch.object(tactics_drop, "go_test", _go):
            for _ in range(3):
                intent, action = h.step()
            self.assertEqual((h.modes.goal, h.modes.sub), ("M7", "CAPTURE"))
            self.assertGreater(action["move"][0], 0.9)

    def test_timeout_counts_from_the_landing(self):
        h = self._landed(235.0)
        C = h.C
        with mock.patch.object(tactics_drop, "go_test", _go):
            h.step()
            self.assertEqual(h.modes.goal, "M7")
            h.world.drops[20000].lands_at = h.world.tick.t - C.DROP_TIMEOUT_S - 1.0
            for _ in range(3):
                h.step()
        self.assertNotEqual(h.modes.goal, "M7")
        self.assertIn("timeout", [e[-1] for e in goal_exits(h.modes, "M7")])


class TimeoutDuringCaptureTest(unittest.TestCase):
    def test_running_capture_is_not_timed_out(self):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        C = h.C
        dr = DropState(20000, (3.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 70.0, 99.0)
        h.world.drops[20000] = dr
        with mock.patch.object(tactics_drop, "go_test", _go):
            for _ in range(3):
                h.step()
            self.assertEqual((h.modes.goal, h.modes.sub), ("M7", "CAPTURE"))
            dr.lands_at = h.world.tick.t - C.DROP_TIMEOUT_S - 1.0      # late, but our clock is running
            dr.holder, dr.progress = h.world.me.id, 2.0
            for _ in range(3):
                h.step()
            self.assertEqual(h.modes.goal, "M7")
            dr.holder, dr.progress = None, 0.0                         # pushed out of the ring: now it ends
            for _ in range(3):
                h.step()
        self.assertNotEqual(h.modes.goal, "M7")

    def test_frozen_progress_does_not_protect_from_timeout(self):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        C = h.C
        dr = DropState(20000, (3.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 70.0, 99.0)
        h.world.drops[20000] = dr
        with mock.patch.object(tactics_drop, "go_test", _go):
            for _ in range(3):
                h.step()
            self.assertEqual((h.modes.goal, h.modes.sub), ("M7", "CAPTURE"))
            dr.lands_at = h.world.tick.t - C.DROP_TIMEOUT_S - 1.0
            dr.holder, dr.progress = h.world.me.id, 2.0
            h.step()
            self.assertEqual(h.modes.goal, "M7")                       # progress just rose: protected
            for _ in range(200):                                       # holder stays ours, progress frozen
                h.step()
                if h.modes.goal != "M7":
                    break
        self.assertNotEqual(h.modes.goal, "M7")
        self.assertEqual(goal_exits(h.modes, "M7")[-1][-1], "timeout")
        self.assertEqual(h.modes.exit_memory.until("DROP", 20000), ExitMemory.PERMANENT)
        self.assertGreaterEqual(h.world.tick.t - h.modes.gs.data.get("prog_t", 0.0), max(C.DROP_STALL_S, 4.0))

    def test_progress_climbing_again_after_a_reset_is_not_stalled(self):
        """The capture clock restarts at 0 when the ring empties; prog_t used to move only past the old
        maximum, so a running capture after a reset looked frozen and timed out."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        C = h.C
        dr = DropState(20000, (3.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 70.0, 99.0)
        h.world.drops[20000] = dr
        with mock.patch.object(tactics_drop, "go_test", _go):
            for _ in range(3):
                h.step()
            dr.lands_at = h.world.tick.t - C.DROP_TIMEOUT_S - 1.0
            dr.holder, dr.progress = h.world.me.id, 3.9
            h.step()
            dr.progress = 0.1                                          # ring emptied in between: clock restarted at 0
            h.step()
            self.assertEqual(h.modes.goal, "M7")
            for _ in range(int((max(C.DROP_STALL_S, 4.0) + 3.0) / 0.1)):
                dr.progress += 0.02                                    # climbs again, never reaching the old 3.9
                h.step()
                self.assertEqual(h.modes.goal, "M7", dr.progress)
        self.assertLess(dr.progress, 3.9)


class LandingTimeTest(unittest.TestCase):
    def test_lands_at_stays_fixed_once_down(self):
        """world rewrote lands_at = t + lands_in every tick, so after landing it was always `now` and
        DROP_TIMEOUT_S could never run out."""
        C = TW.bound_config()
        w = W.WorldModel(C, TW.FakeMap(), 3)
        drop = {"id": 20000, "pos": [50.0, 0.0, 0.0], "radius": 8, "capture_s": 4, "progress": 0, "holder": None}
        seen = []
        for i, lands_in in enumerate([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]):
            o = TW.base_obs(tick=20 * i, t=100.0 + i)
            o["drops"] = [dict(drop, lands_in=lands_in)]
            w.update(o, Clock(time.perf_counter(), 1e6, 1e6))
            seen.append(w.drops[20000].lands_at)
        self.assertEqual(seen, [102.0] * 6)
        self.assertAlmostEqual(w.tick.t - w.drops[20000].lands_at, 3.0)


if __name__ == "__main__":
    unittest.main()
