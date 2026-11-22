"""X4 ENDGAME against the stuck detector (O5), with the real navigation layer in the loop.

    cd bot && python3.13 -m unittest tests.test_endgame_stuck -v

tests/test_endgame.py drives X4 on fakes.FakeNav, which has no stuck logic and "arrives" 1 m from its goal.
The real follower stops WAYPOINT_REACH_M (2.5 m) out and the real detector fires on any commanded move
that goes nowhere, so three things went unseen there:
  (a) pressing into a mech that holds the final point (R11.2 does that on purpose) ran the R13.1 ladder:
      jump after STUCK_T, sidesteps of 6 m, a relocation 10-20 m away on the third episode;
  (b) a centre with some clearance, but less than our radius, stayed the final point: FINAL_APPROACH
      never ended, the pacing trigger fired and the ladder's jet put the bot on the wall;
  (c) on open ground FINAL_APPROACH stood 1.5 m from the point until COLLAPSE (the follower was done,
      FINAL_POINT_TOL_M was not reached).
X4Loop runs the decision pipeline in the order of main.Bot.act (strategic refresh, overlays with the last
commanded move, modes, arbiter, nav.anytime) on the real MapData, Nav, Modes, Overlays and Arbiter; the
world model and the assessment are the fakes; the physics is tools/navsim.NavSim plus protocol 7's
"overlapping mechs are pushed apart", with our reported velocity set to zero on contact (the worst case:
what the server reports there is not known). Nav's work per tick is counted, not timed.
Needs ../mechfight-kit/arena; skipped when the kit is absent.
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

import arbiter  # noqa: E402
import mapdata as MD  # noqa: E402
import modes  # noqa: E402
import nav as NAV  # noqa: E402
import navsim as NS  # noqa: E402
import overlays  # noqa: E402
from contracts import Clock, Intent, SelfState  # noqa: E402
from fakes import FakeAim, FakeWorld, RuleAssessor, build_A, load_C  # noqa: E402
from test_nav import HAVE_MAP, setup_map  # noqa: E402

BASE_R = 2.5


class _Ticks:
    """Time source for Clock: every look at it costs 0.05 ms, so a slice does the same work on any machine."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 5e-5
        return self.t


def work_clock():
    return Clock(0.0, 30.0, 38.0, now=_Ticks())


class X4Loop:
    """One decision per step(), in main.Bot.act's order. The last shrink runs FINAL_SHRINK_T .. +25 s from a
    25 m circle to radius 0 on `center` (protocol phase 6)."""

    def __init__(self, md, start, center, t=376.0, radius=None, seed=1):
        C = self.C = load_C()
        self.md = md
        self.world = FakeWorld(C, t, ("gun", "laser"))
        self.nav = NAV.Nav(C, md, random.Random(seed))
        self.rng = random.Random(seed)
        aim = FakeAim()
        self.modes = modes.Modes(C, md, self.nav, self.rng, aim_mod=aim)
        self.ov = overlays.Overlays(C, md, self.nav, self.rng)
        self.arb = arbiter.Arbiter(C, md, rng=self.rng, aim_mod=aim)
        self.assessor = RuleAssessor()
        self.sim = NS.NavSim(md, {"radius": radius} if radius else None)
        self.sim.reset(start)
        self.sim.t = t
        z = self.world.zone
        z.center = z.from_center = z.next_center = center
        z.radius = z.from_radius = 25.0
        z.next_radius, z.next_changed = 0.0, True
        z.shrink_start, z.shrink_end = C.FINAL_SHRINK_T, C.FINAL_SHRINK_T + 25.0
        self.last_action = {}
        self.zero_vel_on_contact = True
        self.held = False                 # True: something the map does not know holds us where we are
        self.episodes = []                # (t, sub, distance to the final point, first ladder step, pacing?)
        self.jumps = self.unsticks = self.air_boosts = 0
        self.max_dist = 0.0
        self.max_rise = 0.0
        self._z0 = self.sim.z
        self._was_active = False
        self._sync()

    # ------------------------------------------------------------------ world
    def _sync(self):
        s, me, w = self.sim, self.world.me, self.world
        me.pos, me.vel = (s.x, s.y, s.z), (s.vx, s.vy, s.vz)
        me.speed_h = math.hypot(s.vx, s.vy)
        me.grounded, me.energy, me.max_energy, me.radius = s.grounded, s.energy, s.st["energy"], s.R
        st = dict(me.stats)
        st.update(s.st)
        me.stats = st
        me.s = s.R / BASE_R
        me.muzzle_z, me.eye_z = s.z + 7.0, s.z + 8.0
        c = w.zone.center_at(w.tick.t)
        me.in_zone = math.hypot(s.x - c[0], s.y - c[1]) <= w.zone.radius_at(w.tick.t)

    def occupant(self, pos, cubes=0, tid=5):
        return self.world.add_mech(tid, (pos[0], pos[1], self.md.surface_h(pos[0], pos[1])), weapons=("gun",),
                                   visible=True, cubes=cubes)

    def dist(self):
        pt = self.modes.final_point
        return math.hypot(self.sim.x - pt[0], self.sim.y - pt[1])

    def _push_apart(self):
        """Protocol 7: overlapping mechs are pushed apart. The others do not give way."""
        s = self.sim
        for tr in self.world.tracks.values():
            need = s.R + BASE_R * (1.0 + 0.05 * (tr.cubes or 0))
            dx, dy = s.x - tr.pos[0], s.y - tr.pos[1]
            d = math.hypot(dx, dy)
            if tr.kind == "mech" and d < need and abs(s.z - tr.pos[2]) < 9.0:
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                s.x, s.y = tr.pos[0] + dx / d * need, tr.pos[1] + dy / d * need
                if self.zero_vel_on_contact:
                    s.vx = s.vy = 0.0

    # ------------------------------------------------------------------ one decision
    def step(self):
        w = self.world
        self._sync()
        me = w.me
        A = build_A(w, slack=self.nav.slack(me.pos, w.tick.t, w.zone)[0])
        clock = work_clock()
        intent = Intent(tick=w.tick.tick, t=w.tick.t)
        self.modes.strategic_refresh(w, self.assessor, self.nav, clock)
        last_move = self.arb.blocked_desire or self.last_action.get("move")       # main.py: O5's commanded vector
        ostate = self.ov.update(w, A, self.nav, intent, clock, macro=self.modes.macro, sub=self.modes.sub,
                                last_move=last_move)
        self.modes.decide(w, A, self.nav, ostate, intent, clock)
        action = self.arb.resolve(w, A, self.nav, intent, clock)
        self.nav.anytime(w, A, work_clock())
        self.last_action = action
        # bookkeeping for the assertions
        st = self.nav._stuck
        if st["active"] and not self._was_active:
            self.episodes.append((round(w.tick.t, 1), intent.sub, round(self.dist(), 2), st["steps"][0][0],
                                  bool(st["pace"])))
        self._was_active = bool(st["active"])
        self.jumps += bool(action["jump"])
        self.unsticks += intent.unstick is not None
        self.air_boosts += bool(action["boost"] and not me.grounded)
        self.max_dist = max(self.max_dist, self.dist())
        # physics
        x0, y0 = self.sim.x, self.sim.y
        self.sim.decide(action["move"], action["boost"], action["jump"])
        if self.held:
            self.sim.x, self.sim.y, self.sim.vx, self.sim.vy = x0, y0, 0.0, 0.0
        self._push_apart()
        self.max_rise = max(self.max_rise, self.sim.z - self._z0)
        w.advance()
        self.sim.t = w.tick.t
        return intent, action

    def run_until(self, t_end):
        intent = action = None
        while self.world.tick.t < t_end - 1e-9:
            intent, action = self.step()
        return intent, action


def open_ground(md, need=20.0, edge=80.0):
    """A walkable ground cell with `need` m of clearance, `edge` m inside the map."""
    b = md.bounds
    for c in range(md.nx4 * md.ny4):
        x, y = md.cell4_xy(c)
        if not (b[0] + edge < x < b[2] - edge and b[1] + edge < y < b[3] - edge):
            continue
        if md.nav_flags(c) & MD.NAV_WALKABLE and not md.nav_flags(c) & MD.NAV_ROOF and md.clearance(x, y) > need \
                and not md.roof_id(x, y):
            return (x, y)
    raise AssertionError("no open ground on this map")


# Centres on the arena with some clearance but less than a mech's radius, each with a start on open ground
# 15-18 m away: (centre, start, clearance at the centre). setUp checks the map still has them.
TIGHT_PACE = ((458.31, 67.68), (473.28, 77.68), 0.5)      # FINAL_APPROACH paced, laddered, then jetted in COLLAPSE
TIGHT_JET = ((62.54, -243.57), (47.83, -240.65), 1.5)     # reached FINAL_HOLD 0.9 m out, then jetted onto the wall


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, _ = setup_map()
        cls.open = open_ground(cls.md)

    def assert_no_ladder(self, lp, msg=""):
        self.assertEqual(lp.episodes, [], "O5 began an episode " + msg)
        self.assertEqual((lp.jumps, lp.unsticks, lp.air_boosts), (0, 0, 0), "jumps, unstick legs, jets " + msg)

    def assert_detector_idle(self, lp):
        st = lp.nav._stuck
        self.assertFalse(st["active"])
        self.assertEqual((st["episodes"], st["wedge_n"], st["wedge_p0"]), ([], 0, None))


class TestPressingIntoAnOccupant(_Base):
    """(a): X4 walks into a mech that stands on the final point. That is not being stuck."""

    def press(self, t0, t_end, zero_vel=True, cubes=0):
        p = self.open
        lp = X4Loop(self.md, (p[0] + 5.3, p[1]), p, t=t0)
        lp.zero_vel_on_contact = zero_vel
        lp.occupant(p, cubes=cubes)
        return lp, lp.run_until(t_end)

    def test_collapse_press_runs_no_ladder(self):
        """t = 391 s, pressing west into the occupant, reported speed 0. The ladder used to jump at 392.0,
        sidestep until 393.4, repeat at 394.7 and relocate at 397.5; the bot was up to 8.7 m from the point."""
        lp, (intent, action) = self.press(391.0, 403.0)
        self.assertEqual(intent.sub, "COLLAPSE")
        self.assertEqual(lp.modes.final_point, self.open)
        self.assert_no_ladder(lp, "while X4 pressed on purpose")
        self.assert_detector_idle(lp)
        self.assertLessEqual(lp.max_dist, 2 * BASE_R + 0.35)            # never further out than first contact
        self.assertAlmostEqual(lp.dist(), 2 * BASE_R, delta=0.05)       # as close as the server allows
        self.assertLess(action["move"][0], -0.9)                        # and still pressing at full speed

    def test_the_whole_last_shrink(self):
        """Through 415 s with the real zone slack: O1 takes the legs (L4, full speed at the centre) when the
        circle gets within ZONE_SLACK_HARD_S of us. Its press is the same press."""
        lp, _ = self.press(388.0, 414.9)
        self.assert_no_ladder(lp, "during the last shrink")
        self.assertLessEqual(lp.max_dist, 2 * BASE_R + 0.35)
        self.assertIn("L4:O1", lp.arb.last_owners["legs"])              # the scenario did reach O1's press

    def test_final_hold_press_runs_no_ladder(self):
        lp, (intent, _) = self.press(380.0, 389.9)
        self.assertEqual(intent.sub, "FINAL_HOLD")
        self.assert_no_ladder(lp, "in FINAL_HOLD")
        self.assert_detector_idle(lp)

    def test_same_when_the_velocity_is_reported_as_commanded(self):
        """minisim's push-apart leaves the reported velocity alone: the result must not depend on it."""
        lp, _ = self.press(391.0, 403.0, zero_vel=False)
        self.assert_no_ladder(lp)
        self.assertAlmostEqual(lp.dist(), 2 * BASE_R, delta=0.05)

    def test_a_bigger_occupant_keeps_us_further_out(self):
        """10 cubes: radius 3.75, so contact is 6.25 m from the point, beyond HOLD_ARRIVE_R + 2 base radii."""
        lp, _ = self.press(391.0, 400.0, cubes=10)
        self.assert_no_ladder(lp)
        self.assertAlmostEqual(lp.dist(), BASE_R + 3.75, delta=0.05)

    def test_the_point_is_taken_once_the_occupant_leaves(self):
        lp, _ = self.press(380.0, 384.0)
        lp.world.tracks.clear()
        intent, _ = lp.run_until(387.0)
        self.assertEqual(intent.sub, "FINAL_HOLD")
        self.assertLessEqual(lp.dist(), 0.2)
        self.assert_no_ladder(lp)

    def test_no_stale_exemption_after_the_press(self):
        """The occupant leaves and something the map does not know holds us 5 m out. Not held off and beyond
        HOLD_ARRIVE_R is FINAL_APPROACH again: O5 has to act, STUCK_T after the press ended and not before
        (nothing was counted during the press), and with the first rung of the ladder."""
        lp, _ = self.press(380.0, 385.0)
        self.assert_no_ladder(lp, "during the press")
        lp.world.tracks.clear()
        lp.held = True
        t_free = lp.world.tick.t
        intent, _ = lp.run_until(t_free + 2.0)
        self.assertEqual(len(lp.episodes), 1, lp.episodes)
        t_ep, sub, _, first, pace = lp.episodes[0]
        self.assertEqual((sub, first, pace), ("FINAL_APPROACH", "jump", False))
        self.assertGreaterEqual(t_ep - t_free, lp.C.STUCK_T)
        self.assertLessEqual(t_ep - t_free, lp.C.STUCK_T + 0.45)
        self.assertGreaterEqual(lp.jumps, 1)


class TestCentreWithoutRoomForTheBody(_Base):
    """(b): 0 < clearance(centre) < our radius."""

    def approach(self, case, t_end, radius=None):
        centre, start, cl = case
        md = self.md
        self.assertAlmostEqual(md.clearance(*centre), cl, delta=0.05, msg="the arena changed under this fixture")
        self.assertFalse(md.roof_id(*centre))
        lp = X4Loop(md, start, centre, t=376.0, radius=radius)
        return lp, lp.run_until(t_end)

    def check(self, case):
        """The point is moved to where the body fits, FINAL_HOLD is reached and kept, and nothing of the
        ladder runs from 376 s to the end of the last shrink; the bot stays on the ground."""
        lp, (intent, _) = self.approach(case, 389.9)
        pt = lp.modes.final_point
        self.assertNotEqual(pt, case[0])
        self.assertGreaterEqual(self.md.clearance(*pt), BASE_R)
        self.assertLessEqual(math.hypot(pt[0] - case[0][0], pt[1] - case[0][1]), 4.0)
        self.assertEqual(intent.sub, "FINAL_HOLD")
        self.assertLessEqual(lp.dist(), lp.C.FINAL_POINT_TOL_M)
        hist = [h for h in lp.modes.history if h[1] == "sub"]
        self.assertEqual([h[3] for h in hist], ["FINAL_HOLD"], hist)
        self.assertLess(hist[0][0], 380.0)                              # 15-18 m at walking speed
        lp.run_until(414.9)
        self.assert_no_ladder(lp)
        self.assertLess(lp.max_rise, 2.0)
        self.assertLessEqual(lp.dist(), lp.C.FINAL_POINT_TOL_M)
        return lp

    def test_final_approach_ends_and_nothing_paces(self):
        """Was: FINAL_APPROACH until 390 s, the pacing trigger at 384.1 s, a sidestep to 13 m from the point,
        and in COLLAPSE the ladder's jet to 12 m above the ground."""
        self.check(TIGHT_PACE)

    def test_final_hold_does_not_jet_onto_the_wall(self):
        """Was: FINAL_HOLD 0.9 m from a centre the body could not stand on, the ladder at 378.5 s, its jet
        to 12 m up the wall, and the same again three times in COLLAPSE."""
        self.check(TIGHT_JET)

    def test_grown_body(self):
        """8 cubes (radius 3.5): the open-ground point of the other tests has room for it; one with 2.5 m
        of clearance, standable for a base mech, is moved."""
        md = self.md
        b = md.bounds
        rng = random.Random(5)
        centre = None
        while centre is None:
            x, y = rng.uniform(b[0] + 80, b[2] - 80), rng.uniform(b[1] + 80, b[3] - 80)
            if 2.5 <= md.clearance(x, y) <= 2.6 and not md.roof_id(x, y) and \
                    md.clearance(x + 15.0, y) >= 8.0 and not md.roof_id(x + 15.0, y) and \
                    abs(md.surface_h(x + 15.0, y) - md.surface_h(x, y)) < 1.0:
                centre = (x, y)
        lp = X4Loop(md, (centre[0] + 15.0, centre[1]), centre, t=376.0, radius=3.5)
        intent, _ = lp.run_until(389.9)
        pt = lp.modes.final_point
        self.assertNotEqual(pt, centre)
        self.assertGreaterEqual(md.clearance(*pt), 3.5)
        self.assertEqual(intent.sub, "FINAL_HOLD")
        self.assertLessEqual(lp.dist(), lp.C.FINAL_POINT_TOL_M)
        self.assert_no_ladder(lp)


class TestFinalApproach(_Base):
    def test_ends_on_open_ground(self):
        """(c): the follower is done 2.5 m from its goal. FINAL_APPROACH used to stand 1.5 m from the point
        with no move commanded until COLLAPSE; now the servo of FINAL_HOLD takes the last metres."""
        p = self.open
        ends = []
        for k in range(8):                                              # 8 bearings, 12 .. 17.6 m out
            a, r = k * math.pi / 4, 12.0 + 0.8 * k
            lp = X4Loop(self.md, (p[0] + r * math.cos(a), p[1] + r * math.sin(a)), p, t=376.0)
            intent, _ = lp.run_until(380.0)
            ends.append((intent.sub, round(lp.dist(), 2)))
            self.assert_no_ladder(lp)
        self.assertEqual(ends, [("FINAL_HOLD", 0.0)] * 8)

    def test_stuck_en_route_still_recovers(self):
        """FINAL_APPROACH 15 m from the point, held by something the map does not know, another mech in
        sight but not in the way: O5 acts STUCK_T later, as anywhere else. Let go, the bot arrives."""
        p = self.open
        lp = X4Loop(self.md, (p[0] + 15.0, p[1]), p, t=376.0)
        lp.occupant((p[0] + 15.0, p[1] + 12.0))
        lp.held = True
        intent, _ = lp.run_until(378.0)
        self.assertEqual(intent.sub, "FINAL_APPROACH")
        self.assertEqual([e[1:] for e in lp.episodes], [("FINAL_APPROACH", round(lp.dist(), 2), "jump", False)])
        self.assertAlmostEqual(lp.episodes[0][0] - 376.0, lp.C.STUCK_T, delta=0.25)
        self.assertGreaterEqual(lp.jumps, 1)
        self.assertGreaterEqual(lp.unsticks, 5)
        lp.held = False
        intent, _ = lp.run_until(386.0)
        self.assertEqual(intent.sub, "FINAL_HOLD")
        self.assertLessEqual(lp.dist(), 0.2)

    def test_stuck_in_collapse_away_from_the_point_still_recovers(self):
        """COLLAPSE has no exemption of its own: 15 m out is beyond HOLD_ARRIVE_R plus a contact distance."""
        p = self.open
        lp = X4Loop(self.md, (p[0] + 15.0, p[1]), p, t=391.0)
        lp.held = True
        intent, _ = lp.run_until(393.0)
        self.assertEqual(intent.sub, "COLLAPSE")
        self.assertEqual(len(lp.episodes), 1, lp.episodes)
        self.assertGreaterEqual(lp.jumps, 1)


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestStuckExempt(unittest.TestCase):
    """nav.stuck_exempt(): what the mode tells the detector, on the real ladder."""

    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()
        p = open_ground(cls.md)
        cls.pos = (p[0], p[1], cls.md.surface_h(*p))

    def world(self, t, vel=(0.0, 0.0, 0.0)):
        me = SelfState(pos=self.pos, vel=vel, grounded=True, energy=100.0, max_energy=100.0, stats={})
        return NS.FakeWorld(me, t)

    def test_one_call_covers_one_update(self):
        """Commanded east, speed 0. Exempt before every update: nothing, however long. The calls stop: the
        episode begins STUCK_T after the last exempt update, not on the next one."""
        nav = NAV.Nav(self.C, self.md)
        t = 0.0
        for _ in range(60):
            nav.stuck_exempt()
            self.assertEqual(nav.stuck_update(self.world(t), (1.0, 0.0)), (None, None, None))
            self.assertFalse(nav.stuck_active())
            self.assertIsNone(nav._stuck["slow_since"])
            self.assertIsNone(nav._stuck["pace_t0"])
            t += 0.1
        t_end, began = t, None
        while began is None and t < t_end + 3.0:
            nav.stuck_update(self.world(t), (1.0, 0.0))
            if nav.stuck_active():
                began = t
            t += 0.1
        self.assertIsNotNone(began)
        self.assertAlmostEqual(began - t_end, nav._STUCK_T, delta=0.11)
        self.assertEqual(nav._stuck["steps"][0][0], "jump")

    def test_pacing_does_not_count_during_the_press(self):
        """A nav direction that goes nowhere at speed for STUCK_PACE_S paces. Not while exempt, and the 4 s
        start again after it."""
        nav = NAV.Nav(self.C, self.md)
        fast = (8.0, 0.0, 0.0)
        t = 0.0
        while t < 2 * nav._STUCK_PACE_S:
            nav.stuck_exempt()
            nav.stuck_update(self.world(t, fast), (1.0, 0.0))
            self.assertFalse(nav.stuck_active())
            t += 0.1
        t_end = t
        while t < t_end + nav._STUCK_PACE_S - 0.25:
            nav.stuck_update(self.world(t, fast), (1.0, 0.0))
            self.assertFalse(nav.stuck_active(), "paced %.1f s after the press" % (t - t_end))
            t += 0.1
        for _ in range(5):
            nav.stuck_update(self.world(t, fast), (1.0, 0.0))
            t += 0.1
        self.assertTrue(nav.stuck_active())

    def test_it_ends_an_episode_and_forgets_the_counts(self):
        """Two episodes on one spot arm the wedge lift, and a third inside STUCK_REPEAT_WINDOW_S would
        relocate. A press in between: the episode in progress ends at once and the next one is the first
        rung again."""
        nav = NAV.Nav(self.C, self.md, random.Random(5))
        t, ended, was = 0.0, 0, False
        while ended < 2:
            nav.stuck_update(self.world(t), (1.0, 0.0))
            ended += was and not nav.stuck_active()
            was = nav.stuck_active()
            t += 0.1
        while not nav.stuck_active():                                   # the third: the lift
            nav.stuck_update(self.world(t), (1.0, 0.0))
            t += 0.1
        self.assertEqual(nav._stuck["steps"][0][0], "lift")
        nav.stuck_exempt()
        self.assertEqual(nav.stuck_update(self.world(t), (1.0, 0.0)), (None, None, None))
        st = nav._stuck
        self.assertFalse(nav.stuck_active())
        self.assertEqual((st["episodes"], st["wedge_n"], st["wedge_p0"], st["slow_since"]), ([], 0, None, None))
        t += 0.1
        while not nav.stuck_active():
            nav.stuck_update(self.world(t), (1.0, 0.0))
            t += 0.1
        self.assertEqual([s[0] for s in st["steps"]], ["jump", "side1", "side2", "block"])


if __name__ == "__main__":
    unittest.main()
