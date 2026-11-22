"""Leg directions from the navigation layer through the real Arbiter._legs (closed loop in tools/navsim.py).

    cd bot && python3.13 -m unittest tests.test_legs_closed_loop -v

The navsim gate in test_nav drives nav.follow straight into the simulator. The bot does not: every leg
direction goes through arbiter._legs, whose clearance probe used to drop the direction toward a wall at every
altitude (these 14 climb routes: 1 reached the goal with the flat probe and no nav flag), the ladder's jump and
jet pointed at the obstacle and were dropped too, and the zone field's jump edges were never issued. These
tests put the real arbiter in the loop: its leg filter, jump and boost resolution, and the O5 stuck update.
The slices of work nav gets per tick are counted, not timed, so a loaded machine gives the same result.
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
import nav as NAV  # noqa: E402
import navsim as NS  # noqa: E402
from contracts import (BOOST_ORDER, BoostRequest, Clock, Intent, JumpRequest, LegRequest, Steer,  # noqa: E402
                       ZoneState)
from fakes import FakeAim  # noqa: E402
from test_nav import HAVE_MAP, find_wall_spot, setup_map  # noqa: E402


class _World(NS.FakeWorld):
    def visible_mechs(self):
        return []


class _Ticks:
    """Time source for Clock: every look at it costs 0.05 ms, so a slice does the same work on any machine."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 5e-5
        return self.t


def work_clock():
    return Clock(0.0, 30.0, 38.0, now=_Ticks())


class Loop:
    """NavSim + Nav + Arbiter: one decision per call to act()."""

    def __init__(self, md, C, start, zone=None, stats=None, energy=None, nav_flag=True, flat_probe=False,
                 o1_sprint=False):
        self.md, self.C, self.zone = md, C, zone
        self.nav = NAV.Nav(C, md, random.Random(1))
        self.arb = arbiter.Arbiter(C, md, rng=random.Random(0), aim_mod=FakeAim())
        self.sim = NS.NavSim(md, stats)
        self.sim.reset(start, energy)
        self.nav_flag, self.o1_sprint = nav_flag, o1_sprint
        if flat_probe:
            # the probe as it was: ground heights only, whatever our altitude
            probe = self.nav.clear_dir
            self.nav.clear_dir = lambda pos, d, dist: probe(tuple(pos)[:2], d, dist)
        self.cmd = None
        self.unstick = 0

    def world(self):
        return _World(self.sim.self_state(), self.sim.t, self.zone)

    def act(self, steer):
        """stuck_update, then the intent Modes.go / zone_go and overlays._o5 build, through Arbiter._legs, _jump and
        _boost; advances the sim."""
        world = self.world()
        leg, jmp, bst = self.nav.stuck_update(world, self.cmd)
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M1", steer.direction, nav=self.nav_flag))
        if steer.jump:
            intent.jump.append(JumpRequest("CLIMB", steer.rule))
        if steer.climb_boost:
            intent.boost.append(BoostRequest("CLIMB", steer.rule, True, 0.0))
        if self.o1_sprint:                         # O1's ground sprint, off while a climb is on (overlays._o1_update)
            intent.boost.append(BoostRequest("O1", "O1", not steer.rule.startswith("R13.2"), 0.0))
        if leg is not None:
            intent.unstick = leg
            self.unstick += 1
        if jmp is not None:
            intent.jump.append(JumpRequest("O5", jmp.rule or "R13.1", jmp.want))
        if bst is not None:
            owner = bst.owner if bst.owner in BOOST_ORDER else "CLIMB"
            intent.boost.append(BoostRequest(owner, bst.rule or "R13.1", bst.want, bst.floor))
        move, _ = self.arb._legs(world, NS._A, self.nav, intent)
        jump, _ = self.arb._jump(intent)
        boost, _ = self.arb._boost(world, intent)
        self.cmd = self.arb.blocked_desire or move
        self.sim.decide(move, boost, jump)


def drive_path(md, C, start, goal, max_t, **kw):
    """Follow the path to goal with the arbiter in the loop; True when we got within 4 m."""
    lp = Loop(md, C, start, **kw)
    nav, sim = lp.nav, lp.sim
    t_end = sim.t + max_t
    while sim.t < t_end:
        me = sim.self_state()
        if math.hypot(goal[0] - sim.x, goal[1] - sim.y) < 4.0:
            return True
        world = lp.world()
        nav.path_request("goal", me.pos, goal, {}, sim.t)
        nav.anytime(world, NS._A, work_clock())
        lp.act(nav.follow(me, "goal", goal))
    return False


def route_kind(md, C, start, goal):
    nav = NAV.Nav(C, md, random.Random(1))
    nav.path_request("probe", start, goal, {}, 0.0)
    nav.run_until_done(4000)
    job = nav._paths.get("probe")
    if job is None or job.path is None or not job.path.complete:
        return None
    ks = set(job.kinds)
    return ("climb" if (2 in ks or 3 in ks) else "jump" if 1 in ks else "walk"), job.path.eta_s


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestPathsThroughArbiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    @classmethod
    def routes(cls):
        """Seeded routes classified by the path's edge kinds: up to 14 climbs, 12 walks, 12 jumps."""
        planner = NAV.Nav(cls.C, cls.md, random.Random(1))
        pairs = NS.random_pairs(cls.md, planner, 70, random.Random(11), allow_roof=True)
        res = {"walk": [], "jump": [], "climb": []}
        for a, b, _ in pairs:
            k = route_kind(cls.md, cls.C, a, b)
            if k is None:
                continue
            kind, eta = k
            if len(res[kind]) < (14 if kind == "climb" else 12):
                res[kind].append((a, b, eta * 2.0 + 20.0))
        return res

    def test_climb_and_walk_routes_reach_the_goal(self):
        md, C = self.md, self.C
        res = self.routes()
        climbs = [drive_path(md, C, *r) for r in res["climb"]]
        walks = [drive_path(md, C, *r) for r in res["walk"] + res["jump"]]
        print("\n[legs closed loop] climb %d/%d, walk+jump %d/%d reached the goal" % (
            sum(climbs), len(climbs), sum(walks), len(walks)), file=sys.stderr)
        self.assertGreaterEqual(len(climbs), 12)
        self.assertEqual(sum(climbs), len(climbs))
        self.assertGreaterEqual(len(walks), 10)
        self.assertEqual(sum(walks), len(walks))

    def test_nav_flag_alone_gets_the_climbs(self):
        """With the probe as it was (ground heights only) the nav flag is what lets a direction toward the wall
        through: every climb still arrives. Without the flag, and the flat probe, 1 of these 14 did."""
        md, C = self.md, self.C
        climbs = [drive_path(md, C, *r, flat_probe=True) for r in self.routes()["climb"]]
        print("\n[legs closed loop] flat probe, nav flag: climb %d/%d" % (sum(climbs), len(climbs)), file=sys.stderr)
        self.assertGreaterEqual(len(climbs), 12)
        self.assertEqual(sum(climbs), len(climbs))


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestZoneSteerThroughArbiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def leave_basin(self, priced_for_cube):
        """Seconds until x > -230 or the safe set, driving nav.zone_steer + stuck_update + Arbiter._legs (None: 40 s)."""
        md, C = self.md, self.C
        z = ZoneState(center=(41.3, -72.8), radius=560.0, next_center=(196.0, -43.0), next_radius=380.0,
                      shrink_start=60.0, shrink_end=110.0, from_center=(41.3, -72.8), from_radius=560.0)
        start = (-276.0, 60.0, md.surface_h(-276.0, 60.0))
        lp = Loop(md, C, start, z, stats={"radius": 2.625})        # 1 cube
        if priced_for_cube:
            lp.nav._sync_world(lp.world())
        lp.nav.zone_field_start(z, outside_current=False)           # else the field is priced for the size before the cube
        lp.nav.run_until_done(10000)
        while lp.sim.t < 40.0:
            world = lp.world()
            lp.nav.anytime(world, NS._A, work_clock())
            lp.act(lp.nav.zone_steer(world.me))
            if lp.sim.x > -230.0 or lp.nav.in_safe_set((lp.sim.x, lp.sim.y, lp.sim.z)):
                return lp.sim.t
        return None

    def climb_to_the_safe_set(self, energy, steer=None):
        """Seconds until the safe set (a tiny circle on a big roof) is reached from 10 m out at the foot of a climb
        face, with O1's ground sprint in the loop (None: 40 s). `steer` replaces nav.zone_steer."""
        md, C = self.md, self.C
        f, roof = find_wall_spot(md, NAV.Nav(C, md))
        cx, cy = roof.centroid
        z = ZoneState(center=(cx, cy), radius=200.0, next_center=(cx, cy), next_radius=8.0,
                      shrink_start=1e6, shrink_end=1e6 + 1, from_center=(cx, cy), from_radius=200.0)
        start = (f["x"] + f["nx"] * 10.0, f["y"] + f["ny"] * 10.0)
        lp = Loop(md, C, start, z, energy=energy, o1_sprint=True)
        lp.nav.zone_field_start(z, outside_current=False)
        lp.nav.run_until_done(10000)
        while lp.sim.t < 40.0:
            world = lp.world()
            lp.nav.anytime(world, NS._A, work_clock())
            lp.act(steer(lp.nav, world.me) if steer else lp.nav.zone_steer(world.me, world.tick.t))
            if lp.nav.in_safe_set((lp.sim.x, lp.sim.y, lp.sim.z)):
                return lp.sim.t
        return None

    def test_zone_field_climb_with_little_energy(self):
        """The field's step is a climb and the energy is below its cost plus the reserve (a long zone run leaves
        it there): the climb starts anyway and waits at the foot, instead of pushing into the face with a
        sprint on until the stuck ladder cycles at the wall (never got up in 30 s)."""
        for energy in (100.0, 35.0):
            out = self.climb_to_the_safe_set(energy)
            print("\n[legs closed loop] zone climb from %d energy: on the roof after %s s" % (
                energy, "%.1f" % out if out is not None else "never"), file=sys.stderr)
            self.assertIsNotNone(out, "energy %d" % energy)
            self.assertLessEqual(out, 25.0, "energy %d" % energy)

    def test_zone_descent_alone_does_not_climb(self):
        """The control for the test above: the plain field direction (what O1 used before zone_steer) never gets
        onto the roof, so the closed loop does exercise the climb start."""
        out = self.climb_to_the_safe_set(100.0, steer=lambda nav, me: Steer(direction=nav.zone_descent(tuple(me.pos))))
        self.assertIsNone(out)

    def test_leaves_the_basin(self):
        """A low basin at (-276, 60), 1 cube. The zone field steers over ledges; the arbiter used to drop the
        direction toward each one and the bot paced until the zone killed it (90 s measured). A field priced for
        the size we have gets us out at once; one priced before the cube picks up (cells now too tight for the
        steepest-descent step) leaves dead ends, and the ladder's pacing trigger has to get us out."""
        for priced, limit in ((True, 20.0), (False, 40.0)):
            out = self.leave_basin(priced)
            print("\n[legs closed loop] basin, field priced for the cube %s: out after %s s" % (
                priced, "%.1f" % out if out is not None else "never"), file=sys.stderr)
            self.assertIsNotNone(out, "still in the basin after 40 s (priced for the cube: %s)" % priced)
            self.assertLessEqual(out, limit, "priced for the cube: %s" % priced)


if __name__ == "__main__":
    unittest.main()
