"""X4 final point (R11.2): the bot holds the true center of the last circle (review bug 1).

    python3 -m unittest tests.test_endgame -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import Harness  # noqa: E402

import tactics_endgame  # noqa: E402


def _endgame(center=(30.0, -20.0), pos=(60.0, -20.0)):
    h = Harness(t=380.0, weapons=("gun", "laser"))
    h.world.me.pos = (pos[0], pos[1], 0.0)
    z = h.world.zone
    z.next_center, z.next_radius, z.next_changed = center, 0.0, True
    return h


class FinalPointTest(unittest.TestCase):
    def test_mech_on_center_does_not_move_the_point(self):
        """A mech standing on the center when the point is first computed used to push it 5.5 m out for the
        rest of the match: 5.5 s of the last shrink at 260 HP/s."""
        h = _endgame()
        tr = h.world.add_mech(5, (30.0, -20.0, 0.0), weapons=("gun",), visible=True)
        intent, action = h.step()
        self.assertEqual(intent.macro, "X4")
        self.assertEqual(h.modes.final_point, (30.0, -20.0))
        self.assertLess(action["move"][0], -0.9)                    # walking at it (we are due east)
        tr.pos = (200.0, 200.0, 0.0)                                # it leaves
        for _ in range(5):
            h.step()
        self.assertEqual(h.modes.final_point, (30.0, -20.0))

    def test_collapse_presses_into_an_occupied_center(self):
        h = _endgame(pos=(35.5, -20.0))
        h.world.tick.t = 395.0
        h.world.tick.tick = 7900
        h.world.add_mech(5, (30.0, -20.0, 0.0), weapons=("gun",), visible=True)
        intent, action = h.step()
        self.assertEqual(intent.sub, "COLLAPSE")
        self.assertEqual(h.modes.final_point, (30.0, -20.0))
        self.assertLess(action["move"][0], -0.9)                    # full push toward the center, not a stop

    def test_unwalkable_center_takes_the_nearest_standable_cell(self):
        h = _endgame()
        cx, cy = 30.0, -20.0
        wall_r = 3.2                                                 # blocked disc around the center
        h.map.clear_fn = lambda x, y: max(0.0, math.hypot(x - cx, y - cy) - wall_r)
        h.step()
        pt = h.modes.final_point
        d = math.hypot(pt[0] - cx, pt[1] - cy)
        stand = wall_r + h.world.me.radius                           # our body cannot be centred nearer
        self.assertGreaterEqual(d, stand - 1e-6)
        self.assertLessEqual(d, stand + tactics_endgame._WALK_STEP_M)  # the 2 m rings gave 6.0
        self.assertGreater(pt[0], cx)                                # of that ring, the side we come from

    def test_touching_an_occupant_counts_as_arrived_and_does_not_boost(self):
        """Pushed apart from a mech on the point we sit about 5 m out: that used to keep FINAL_APPROACH
        and boost into it from FINAL_ARRIVE_T, burning the energy COLLAPSE needs."""
        h = _endgame(pos=(35.3, -20.0))
        h.world.tick.t = 386.0
        h.world.tick.tick = 7720
        h.world.add_mech(5, (30.0, -20.0, 0.0), weapons=("gun",), visible=True)
        for _ in range(5):
            intent, action = h.step()
            self.assertEqual(intent.sub, "FINAL_HOLD")
            self.assertFalse(action["boost"])
            self.assertLess(action["move"][0], -0.9)                # still pressing toward the center
        self.assertEqual(h.modes.final_point, (30.0, -20.0))

    def test_far_from_an_occupied_point_still_approaches(self):
        h = _endgame(pos=(60.0, -20.0))
        h.world.add_mech(5, (30.0, -20.0, 0.0), weapons=("gun",), visible=True)
        intent, _ = h.step()
        self.assertEqual(intent.sub, "FINAL_APPROACH")

    def test_point_follows_a_new_circle(self):
        h = _endgame()
        h.step()
        self.assertEqual(h.modes.final_point, (30.0, -20.0))
        h.world.zone.next_center, h.world.zone.next_changed = (31.0, -18.0), True
        h.step()
        self.assertEqual(h.modes.final_point, (31.0, -18.0))


if __name__ == "__main__":
    unittest.main()
