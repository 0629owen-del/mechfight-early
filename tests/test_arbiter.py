"""Arbiter / actuator resolution (plan 6.2-6.7, 14.1 "Arbiter") and overlays -> arbiter interplay.

    python3 -m unittest tests.test_arbiter -v
"""
from __future__ import annotations

import math
import os
import random
import statistics
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import FakeAim, FakeMap, FakeNav, FakeWorld, Harness, build_A, load_C  # noqa: E402

import arbiter  # noqa: E402
import overlays  # noqa: E402
from contracts import (AimRequest, BoostRequest, Clock, FightEval, FireRequest, Intent, JukeStyle,  # noqa: E402
                       JumpRequest, LegFilter, LegRequest, PickupMem, RunnerInfo, ShellPred, Steer, TakeRequest,
                       Threats)


def mk(t=100.0, weapons=("gun", "laser")):
    C = load_C()
    w = FakeWorld(C, t, weapons)
    nav = FakeNav()
    arb = arbiter.Arbiter(C, FakeMap(), rng=random.Random(0), aim_mod=FakeAim())
    return C, w, nav, arb


def clock(C):
    return Clock(time.perf_counter(), C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS)


def resolve(C, w, nav, arb, intent, A=None):
    return arb.resolve(w, A or build_A(w), nav, intent, clock(C))



def _boost_intent(*reqs):
    intent = Intent()
    intent.boost += list(reqs)
    return intent

class TestLegs(unittest.TestCase):
    def test_level_order(self):
        C, w, nav, arb = mk()
        order = [("L8", (0, 1)), ("L5", (1, 0)), ("L4", (-1, 0)), ("L3b", (0, -1)), ("L3", (0.7071, 0.7071))]
        intent = Intent()
        for lv, d in order:
            intent.legs.append(LegRequest(lv, lv, d, juke_ok=False))
        for lv, d in reversed(order):                               # L3 > L3b > L4 > L5 > L8
            a = resolve(C, w, nav, arb, intent)
            self.assertTrue(arb.last_owners["legs"].startswith(lv + ":"), arb.last_owners["legs"])
            self.assertAlmostEqual(a["move"][0], d[0], places=3)
            intent.legs = [lr for lr in intent.legs if lr.level != lv]

    def test_nav_direction_survives_the_clearance_probe(self):
        """A direction from the navigation layer goes over or into the obstacle on purpose (the probe only knows
        ground heights); the same direction from any other owner is dropped for a clear one."""
        C, w, nav, arb = mk()
        nav.blocked = lambda pos, d: d[0] > 0.5                    # a wall to the east
        for flag in (True, False):
            intent = Intent()
            intent.legs.append(LegRequest("L8", "M1", (1.0, 0.0), juke_ok=False, nav=flag))
            a = resolve(C, w, nav, arb, intent)
            if flag:
                self.assertAlmostEqual(a["move"][0], 1.0, places=6)
                self.assertIsNone(arb.blocked_desire)
            else:
                self.assertLess(a["move"][0], 0.5)
        # the unstick request in force decides, not the owner's flag
        for flag, kept in ((True, True), (False, False)):
            intent = Intent()
            intent.legs.append(LegRequest("L8", "M1", (0.0, 1.0), juke_ok=False, nav=not flag))
            intent.unstick = LegRequest("O5", "R13.1.1", (1.0, 0.0), nav=flag)
            a = resolve(C, w, nav, arb, intent)
            self.assertEqual(a["move"][0] > 0.99, kept, (flag, a["move"]))

    def test_slow_nav_direction_keeps_the_probe(self):
        """A silent walk (speed 0.41) is under O5's STUCK_MOVE_MIN: the ladder could never see it press into a
        wall, so its nav direction is probed like any other. The arbiter reports for O5 whether the winner was nav."""
        C, w, nav, arb = mk()
        nav._STUCK_MOVE_MIN = 0.5
        nav.blocked = lambda pos, d: d[0] > 0.5
        for speed, kept in ((1.0, True), (0.41, False)):
            intent = Intent()
            intent.legs.append(LegRequest("L8", "M1", (1.0, 0.0), speed, juke_ok=False, nav=True))
            a = resolve(C, w, nav, arb, intent)
            self.assertEqual(a["move"][0] > 0.99 * speed, kept, (speed, a["move"]))
            self.assertEqual(nav.leg_nav, kept)
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M1", (1.0, 0.0), 1.0, juke_ok=False))
        resolve(C, w, nav, arb, intent)
        self.assertFalse(nav.leg_nav)

    def test_climb_jet_is_not_juked(self):
        """O7 turns a winning direction 50-80 degrees; the jet of a climb is aimed at the ledge, so a nav direction
        is left alone while a climb runs (and juked as before otherwise)."""
        C, w, nav, arb = mk()
        for climbing in (True, False):
            nav._climb = {"face": None} if climbing else None
            intent = Intent()
            intent.legs.append(LegRequest("L8", "M1", (1.0, 0.0), juke_ok=True, nav=True))
            intent.juke = JukeStyle(1, math.radians(60))
            a = resolve(C, w, nav, arb, intent)
            self.assertEqual(a["move"][1] < 0.01, climbing, (climbing, a["move"]))

    def test_nav_direction_still_meets_the_hard_filters(self):
        C, w, nav, arb = mk()
        nav.blocked = lambda pos, d: d[0] > 0.5
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M1", (1.0, 0.0), juke_ok=False, nav=True))
        intent.leg_filters.append(LegFilter("L2", "x", lambda d, s: d[0] <= 0.0))
        a = resolve(C, w, nav, arb, intent)
        self.assertLessEqual(a["move"][0], 0.0)

    def test_zone_escape_carries_the_field_steer(self):
        """O1 steers by nav.zone_steer: its direction is a nav direction and the jump / climb boost come along
        (the sprint boost stays O1's)."""
        C, w, nav, arb = mk()
        A = build_A(w, slack=2.0)
        nav.steer = Steer((1.0, 0.0), jump=True, climb_boost=True, rule="R13.2")
        intent = Intent()
        ov = overlays.Overlays(C, FakeMap(), nav, random.Random(0))
        ov.update(w, A, nav, intent, clock(C))
        l4 = [lr for lr in intent.legs if lr.level == "L4"]
        self.assertEqual([(lr.rule, lr.direction, lr.nav) for lr in l4], [("O1", (1.0, 0.0), True)])
        self.assertIn(("CLIMB", "R13.2"), [(j.owner, j.rule) for j in intent.jump])
        self.assertIn(("CLIMB", True), [(b.owner, b.want) for b in intent.boost])
        self.assertIn("O1", [b.owner for b in intent.boost])
        # no field direction: the straight line to the circle is an ordinary direction
        nav.steer = Steer(None)
        intent = Intent()
        ov.update(w, A, nav, intent, clock(C))
        self.assertEqual([lr.nav for lr in intent.legs if lr.level == "L4"], [False])
        self.assertEqual(intent.jump, [])

    def test_zone_escape_waiting_at_a_climb_foot(self):
        """Field step is a climb we lack the energy for: the climb waits (R13.2w, no direction). O1 holds (no
        straight line into the face) and does not sprint on the ground, which would eat the energy it waits for."""
        C, w, nav, arb = mk()
        A = build_A(w, slack=2.0)
        nav.steer = Steer(None, rule="R13.2w")
        intent = Intent()
        ov = overlays.Overlays(C, FakeMap(), nav, random.Random(0))
        ov.update(w, A, nav, intent, clock(C))
        self.assertEqual([lr.direction for lr in intent.legs if lr.level == "L4"], [None])
        self.assertEqual([b.want for b in intent.boost if b.owner == "O1"], [False])

    def test_zone_escape_under_a_dodge_starts_no_climb(self):
        """A mortar dodge (L2) outranks O1: O1 asks the field for a direction only, so no climb begins and no
        CLIMB jump / jet is issued for legs that go elsewhere."""
        C, w, nav, arb = mk()
        shell = ShellPred(1, (0.0, 0.0, 0.0), w.tick.t + 0.8, 7.0)
        A = build_A(w, slack=2.0, threats=Threats(shells=[shell]))
        nav.steer = Steer((1.0, 0.0), jump=True, climb_boost=True, rule="R13.2")
        nav.descent = (0.0, 1.0)
        intent = Intent()
        ov = overlays.Overlays(C, FakeMap(), nav, random.Random(0))
        ov.update(w, A, nav, intent, clock(C))
        self.assertTrue(any(lr.level == "L4" for lr in intent.legs))
        self.assertEqual(intent.jump, [])
        self.assertNotIn("CLIMB", [b.owner for b in intent.boost])

    def test_zone_is_l2_tiebreak(self):
        """dodge picks the inward / lower-field side among equal candidates."""
        C, w, nav, arb = mk()
        nav.zone_eta = lambda pos: (max(0.0, pos[0]) * 0.1, True)   # lower ETA to the west (x < 0)
        shell = ShellPred(1, (0.0, 0.0, 0.0), w.tick.t + 0.8, 7.0)
        A = build_A(w, threats=Threats(shells=[shell]))
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M3", (0.0, 1.0), juke_ok=False))
        ov = overlays.Overlays(C, FakeMap(), nav, random.Random(0))
        ov.update(w, A, nav, intent, clock(C))
        self.assertIn("L2", [f.level for f in intent.leg_filters])
        a = resolve(C, w, nav, arb, intent, A)
        self.assertLessEqual(a["move"][0], 1e-9)                     # never the eastern (higher ETA) twin
        self.assertTrue(a["boost"])                                  # only boost speed clears: L2 boost
        self.assertEqual(arb.last_owners["boost"], "L2")
        v = 19.5 * 0.8
        self.assertGreaterEqual(math.hypot(a["move"][0] * v, a["move"][1] * v), C.MORTAR_DODGE_R - 1e-6)

    def test_l3_cover_wins_and_l2_still_filters(self):
        h = Harness(t=100.0)
        h.world.add_mech(5, (100.0, 0.0, 0.0), charging=True, charging_since=h.world.tick.t)
        h.nav.cover = (0.0, 20.0)
        h.A_fn = lambda hh: build_A(hh.world, over={5: {"charging_at_us": True}},
                                    threats=Threats(laser_chargers=[5]))
        intent, action = h.step(advance=False)
        self.assertTrue(h.arb.last_owners["legs"].startswith("L3:O3"))
        self.assertAlmostEqual(action["move"][0], 0.0, places=6)
        self.assertAlmostEqual(action["move"][1], 1.0, places=6)
        self.assertIn("O3", [b.owner for b in intent.boost])
        shell = ShellPred(1, (0.0, 9.0, 0.0), h.world.tick.t + 0.8, 7.0)
        h.A_fn = lambda hh: build_A(hh.world, over={5: {"charging_at_us": True}},
                                    threats=Threats(laser_chargers=[5], shells=[shell]))
        intent, action = h.step(advance=False)
        self.assertTrue(h.arb.last_owners["legs"].startswith("L3:O3"))
        m = action["move"]
        end = (m[0] * 11.7 * 0.8, m[1] * 11.7 * 0.8)
        self.assertGreaterEqual(math.hypot(end[0] - 0.0, end[1] - 9.0), h.C.MORTAR_DODGE_R - 1e-6)

    def test_low_scope_filters_only_under_l5_l8(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.leg_filters.append(LegFilter("O8", "O8", lambda d, s: d[0] <= 0, None, "low"))
        intent.legs.append(LegRequest("L3", "O3", (1.0, 0.0), juke_ok=False))
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(a["move"][0], 1.0, places=6)          # O8 ignored under L3
        intent.legs = [LegRequest("L8", "M2", (1.0, 0.0), juke_ok=False)]
        a = resolve(C, w, nav, arb, intent)
        self.assertLessEqual(a["move"][0], 1e-9)                     # O8 applied under L8

    def test_filter_removing_all_is_ignored(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M2", (1.0, 0.0), juke_ok=False))
        intent.leg_filters.append(LegFilter("O9", "R15.2b", lambda d, s: False, None, "all"))
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(a["move"][0], 1.0, places=6)

    def test_l2_fallback_max_min(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M2", (1.0, 0.0), juke_ok=False))
        intent.leg_filters.append(LegFilter("L2", "O2", lambda d, s: False, None, "all",
                                            lambda d, s: -d[0] if (d[0] or d[1]) else -2.0))
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(a["move"][0], -1.0, places=6)

    def test_unstick_replaces_owner_direction(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M2", (1.0, 0.0)))
        intent.unstick = LegRequest("L8", "R13.1", (0.0, -1.0))
        intent.leg_filters.append(LegFilter("O8", "O8", lambda d, s: d[1] >= 0, None, "low"))   # skipped
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(a["move"][1], -1.0, places=6)
        self.assertIn("unstick", arb.last_owners["legs"])

    def test_juke_rotation_rechecked(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.legs.append(LegRequest("L5", "RZ1", (1.0, 0.0)))
        intent.juke = JukeStyle(1, math.radians(60))
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(math.atan2(a["move"][1], a["move"][0]), math.radians(60), places=6)
        intent.leg_filters.append(LegFilter("O9", "R15.2b", lambda d, s: d[1] <= 0.01, None, "all"))
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(a["move"][0], 1.0, places=6)          # rotated vector fails -> no juke
        intent.legs = [LegRequest("L3", "O3", (1.0, 0.0), juke_ok=False, is_cover_dash=True)]
        intent.leg_filters = []
        a = resolve(C, w, nav, arb, intent)
        self.assertAlmostEqual(a["move"][0], 1.0, places=6)          # straight cover dash: no juke

    def test_blocked_directions_dropped(self):
        C, w, nav, arb = mk()
        nav.blocked = lambda pos, d: d[0] > 0.5
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M2", (1.0, 0.0), juke_ok=False))
        a = resolve(C, w, nav, arb, intent)
        self.assertLessEqual(a["move"][0], 0.5 + 1e-9)
        self.assertGreater(math.hypot(*a["move"]), 0.9)              # slides along the wall


class TestOwners(unittest.TestCase):
    def test_aim_priority_and_tiebreak(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.aim += [AimRequest(6, "AIM6", 1.0, 0.0), AimRequest(2, "AIM2b", 2.0, 0.0, None, 0.5),
                       AimRequest(2, "AIM2a", 3.0, 0.0, None, 0.1), AimRequest(9, "AIM9", 4.0, 0.0)]
        a = resolve(C, w, nav, arb, intent)
        self.assertEqual(a["aim"][0], 3.0)
        self.assertEqual(arb.last_owners["aim"], "AIM2a")
        a = resolve(C, w, nav, arb, Intent())
        self.assertEqual(arb.last_owners["aim"], "AIM10")

    def test_boost_order_and_o9_veto(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.boost += [BoostRequest("MODE", "RL1", True, 30.0), BoostRequest("O3", "O3", False, 10.0)]
        a = resolve(C, w, nav, arb, intent)
        self.assertFalse(a["boost"])
        self.assertEqual(arb.last_owners["boost"], "O3")
        w.me.energy = C.COAST_BOOST_MIN_ENERGY - 1.0
        intent.boost = [BoostRequest("MODE", "M2", True, 0.0)]
        a = resolve(C, w, nav, arb, intent)
        self.assertFalse(a["boost"])                                 # O9 veto
        intent.boost = [BoostRequest("O1", "O1", True, 0.0)]
        a = resolve(C, w, nav, arb, intent)
        self.assertTrue(a["boost"])                                  # O1 exempt

    def test_sprint_boost_is_void_in_the_air(self):
        """Protocol 6: boost in the air is a jet. A sprint request while airborne lifted the bot 45 m."""
        C, w, nav, arb = mk()
        w.me.grounded = False
        for owner in ("O1", "L2", "O3", "O4", "MODE"):
            a = resolve(C, w, nav, arb, _boost_intent(BoostRequest(owner, "x", True, 0.0)))
            self.assertFalse(a["boost"], owner)
        for owner in ("CLIMB", "O6"):
            a = resolve(C, w, nav, arb, _boost_intent(BoostRequest(owner, "x", True, 0.0)))
            self.assertTrue(a["boost"], owner)
        # an airborne sprint request does not hide the climb behind it in BOOST_ORDER
        a = resolve(C, w, nav, arb, _boost_intent(BoostRequest("O1", "O1", True, 0.0),
                                                  BoostRequest("CLIMB", "R13.2", True, 0.0)))
        self.assertTrue(a["boost"])
        self.assertEqual(arb.last_owners["boost"], "CLIMB")
        # a sprint veto (want False) is void in the air too: it must not cut out the CLIMB jet below it
        a = resolve(C, w, nav, arb, _boost_intent(BoostRequest("O1", "O1", False, 0.0),
                                                  BoostRequest("CLIMB", "R13.2", True, 0.0)))
        self.assertTrue(a["boost"])
        self.assertEqual(arb.last_owners["boost"], "CLIMB")
        # O6 is the fall brake and is in AIR_BOOST_OWNERS on purpose: its want=False vetoes a CLIMB jet below it
        a = resolve(C, w, nav, arb, _boost_intent(BoostRequest("O6", "O6", False, 0.0),
                                                  BoostRequest("CLIMB", "R13.2", True, 0.0)))
        self.assertFalse(a["boost"])
        self.assertEqual(arb.last_owners["boost"], "O6")
        w.me.grounded = True
        # on the ground a want=False from a higher owner still vetoes
        a = resolve(C, w, nav, arb, _boost_intent(BoostRequest("O1", "O1", False, 0.0),
                                                  BoostRequest("MODE", "M2", True, 0.0)))
        self.assertFalse(a["boost"])
        self.assertEqual(arb.last_owners["boost"], "O1")
        a = resolve(C, w, nav, arb, _boost_intent(BoostRequest("MODE", "M2", True, 0.0)))
        self.assertTrue(a["boost"])

    def test_jump_order(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.jump += [JumpRequest("O5", "R13.1"), JumpRequest("CLIMB", "R13.2", True)]
        resolve(C, w, nav, arb, intent)
        self.assertEqual(arb.last_owners["jump"], "CLIMB")

    def test_take_writer_order_and_gates(self):
        C, w, nav, arb = mk(weapons=("gun", "laser"))
        w.pickups[8001] = PickupMem(8001, "lightning", (2.0, 0.0, 0.0), w.tick.t)
        w.pickups[8002] = PickupMem(8002, "mortar", (1.0, 0.0, 0.0), w.tick.t)
        intent = Intent()
        intent.take += [TakeRequest("L7", "6.7", 8002, None), TakeRequest("R0.4", "R0.4", 8001, 0)]
        a = resolve(C, w, nav, arb, intent)
        self.assertEqual((a.get("take"), a.get("drop")), (8001, 0))
        self.assertEqual(arb.last_owners["take"], "R0.4")
        w.pickups[8001].pos = (10.0, 0.0, 0.0)                       # out of pickup_radius -> no take
        a = resolve(C, w, nav, arb, intent)
        self.assertNotIn("take", a)
        self.assertNotIn("drop", a)                                  # never drop alone with a failed take
        w.pickups[8001].pos = (1.0, 0.0, 0.0)
        w.me.weapons[1].charge = 0.3                                 # replacing a charging laser: busy
        intent.take = [TakeRequest("R0.4", "R0.4", 8001, 1)]
        a = resolve(C, w, nav, arb, intent)
        self.assertNotIn("take", a)

    def test_pick_copied(self):
        C, w, nav, arb = mk()
        intent = Intent()
        intent.pick, intent.pick_rule = 2, "R14.1"
        a = resolve(C, w, nav, arb, intent)
        self.assertEqual(a["pick"], 2)


class TestIntegrationFixes(unittest.TestCase):
    def test_blocked_owner_reports_desire_for_o5(self):
        """6.3: the owner's direction fails the probe and every clear candidate points away -> stop wins and
        the wanted vector goes to O5 (blocked_desire); a clear owner direction reports nothing."""
        C, w, nav, arb = mk()
        nav.blocked = lambda pos, d: d[0] > -0.5                     # only westward is clear
        intent = Intent()
        intent.legs.append(LegRequest("L8", "M2", (1.0, 0.0), juke_ok=False))
        a = resolve(C, w, nav, arb, intent)
        self.assertEqual(a["move"], [0.0, 0.0])
        self.assertEqual(arb.blocked_desire, (1.0, 0.0))
        nav.blocked = lambda pos, d: False
        resolve(C, w, nav, arb, intent)
        self.assertIsNone(arb.blocked_desire)

    def test_implicit_fire_follows_aim_owner(self):
        """6.6 F1: a slot without a FireRequest fires at the AIM1-AIM6 owner's mech target; F5 keeps it to
        self-defense; AIM9+ (goal aim) never fires implicitly."""
        C, w, nav, arb = mk(weapons=("gun",))
        w.add_mech(5, (120.0, 0.0, 0.0))
        intent = Intent()
        intent.aim.append(AimRequest(6, "AIM6", 0.0, 0.0, 5))
        self.assertTrue(resolve(C, w, nav, arb, intent)["fire"][0])
        self.assertEqual(arb.last_owners["fire"][0], "F1")
        intent.fire_self_defense_only = True                         # loot window: not damaged -> no fire
        self.assertFalse(resolve(C, w, nav, arb, intent)["fire"][0])
        intent = Intent()
        intent.aim.append(AimRequest(9, "AIM9", 0.0, 0.0, 5))
        self.assertFalse(resolve(C, w, nav, arb, intent)["fire"][0])


class TestFire(unittest.TestCase):
    def _x2(self, ri=None, missed=False):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_sub="DUEL")
        if ri is not None:
            h.arb.runner_info = ri
        if missed:
            h.world.tick.missed_ticks = 1
        return h

    def test_soft_overrun_does_not_block_laser_start(self):
        h = self._x2()
        h.arb.runner_info = RunnerInfo(soft_overruns=1, last_soft_overrun_tick=h.world.tick.tick)
        intent, action = h.step()
        self.assertTrue(action["fire"][1], h.arb.last_owners)

    def test_missed_tick_blocks_laser_start(self):
        h = self._x2(missed=True)
        intent, action = h.step()
        self.assertFalse(action["fire"][1])
        self.assertTrue(action["fire"][0])                           # gun F1 unaffected

    def test_hard_overrun_and_degrade_block_laser_start(self):
        h = self._x2()
        h.arb.runner_info = RunnerInfo(hard_overruns=1, last_hard_overrun_tick=h.world.tick.tick)
        _, action = h.step()
        self.assertFalse(action["fire"][1])
        h = self._x2()
        h.arb.degrade_level = 1
        _, action = h.step()
        self.assertFalse(action["fire"][1])

    def test_laser_charging_release_and_r96(self):
        C, w, nav, arb = mk()
        w.add_mech(5, (150.0, 0.0, 0.0))
        w.me.weapons[1].charge = 0.4
        intent = Intent()
        intent.fire.append(FireRequest(1, "F2", 5))
        a = resolve(C, w, nav, arb, intent)
        self.assertTrue(a["fire"][1])                                # keep charging
        w.me.weapons[1].charge = 0.75
        a = resolve(C, w, nav, arb, intent)
        self.assertFalse(a["fire"][1])                               # full -> release
        w.me.weapons[1].charge = 0.4
        intent.fire.append(FireRequest(1, "R9.6", None, hold=False, release=True))
        a = resolve(C, w, nav, arb, intent)
        self.assertFalse(a["fire"][1])
        self.assertEqual(arb.last_owners["fire"][1], "R9.6")

    def test_f5_and_clock_guard(self):
        C, w, nav, arb = mk()
        w.add_mech(5, (150.0, 0.0, 0.0))
        intent = Intent(fire_self_defense_only=True)
        intent.fire.append(FireRequest(0, "F1", 5))
        a = resolve(C, w, nav, arb, intent)
        self.assertFalse(a["fire"][0])
        w.damage(10.0, 5)
        a = resolve(C, w, nav, arb, intent)
        self.assertTrue(a["fire"][0])
        w.tick.t = C.CLOCK_HP_GUARD_T + 1.0                           # R11.5: no firing at all (2+ alive)
        w.damage(10.0, 5)
        a = resolve(C, w, nav, arb, intent)
        self.assertEqual(a["fire"], [False, False])

    def test_f6_reload_when_quiet(self):
        C, w, nav, arb = mk()
        w.me.weapons[0].ammo = 3
        a = resolve(C, w, nav, arb, Intent())
        self.assertTrue(a["reload"])
        self.assertEqual(arb.last_owners["reload"], "F6")
        w.add_mech(5, (150.0, 0.0, 0.0))
        a = resolve(C, w, nav, arb, Intent())
        self.assertFalse(a["reload"])


class TestOverlaysD1(unittest.TestCase):
    def _collapse(self, t):
        C = load_C()
        w = FakeWorld(C, t)
        z = w.zone
        z.from_center = z.center = z.next_center = (0.0, 0.0)
        z.from_radius, z.next_radius = 8.0, 0.0
        z.shrink_start, z.shrink_end = t - 5.0, t + 20.0
        z.radius = z.radius_at(t)
        nav = FakeNav()
        shell = ShellPred(1, (0.0, 0.0, 0.0), t + 0.8, 7.0)
        A = build_A(w, threats=Threats(shells=[shell]))
        intent = Intent()
        intent.legs.append(LegRequest("X4", "R11.5", None, juke_ok=False))
        ov = overlays.Overlays(C, FakeMap(), nav, random.Random(0))
        ov.update(w, A, nav, intent, clock(C), macro="X4")
        arb = arbiter.Arbiter(C, FakeMap(), rng=random.Random(0), aim_mod=FakeAim())
        a = arb.resolve(w, A, nav, intent, clock(C))
        return C, intent, a

    def test_d1_collapse_zone_wins(self):
        C, intent, a = self._collapse(load_C().FINAL_SHRINK_T + 5.0)
        levels = [f.level for f in intent.leg_filters]
        self.assertNotIn("L2", levels)
        self.assertIn("D1", [f.rule for f in intent.leg_filters])
        self.assertIn("D1", intent.trace)
        self.assertLess(math.hypot(*a["move"]) * 11.7 * 0.8, 8.0)    # stays inside the closing circle

    def test_d1_not_before_collapse(self):
        C, intent, a = self._collapse(300.0)
        self.assertIn("L2", [f.level for f in intent.leg_filters])
        self.assertNotIn("D1", intent.trace)
        v = 19.5 if a["boost"] else 11.7
        self.assertGreaterEqual(math.hypot(*a["move"]) * v * 0.8, C.MORTAR_DODGE_R - 1e-6)


class TestOneWriterAndBudget(unittest.TestCase):
    def test_one_writer_per_actuator_and_time_budget(self):
        """Every decision has exactly one modes leg owner and single arbiter owners (fakes.check_one_writer
        runs in every Harness.step); arbiter + goal + actuator resolution + aim/fire <= 4 ms (p99)."""
        rng = random.Random(3)
        times = []
        for k in range(6):
            h = Harness(t=[5.0, 100.0, 240.0, 330.0, 380.0, 395.0][k], weapons=[("gun", "laser"), ("gun", "lightning"),
                                                                            ("gun",), ("laser", "lightning"),
                                                                            ("gun", "mortar"), ("gun", "laser")][k])
            for i in range(16):
                h.world.add_mech(i + 1, (rng.uniform(-250, 250), rng.uniform(-250, 250), 0.0),
                                 weapons=rng.choice([("gun", "laser"), ("gun", "lightning"), ()]),
                                 vel=(rng.uniform(-8, 8), rng.uniform(-8, 8), 0.0), visible=rng.random() < 0.7)
            for i in range(12):
                h.world.add_mech(100 + i, (rng.uniform(-250, 250), rng.uniform(-250, 250), 0.0), weapons=(),
                                 kind="npc")
            for g in range(6):
                h.world.add_ghost(-(g + 1), (rng.uniform(-150, 150), rng.uniform(-150, 150)), rng.choice(["near", "mid"]))
            h.pre = lambda hh: [hh.world.hear(rng.choice(["gun", "step", "laser", "mortar"]), -(g + 1),
                                              hh.world.ghosts[-(g + 1)].band) for g in range(6)]
            shells = [ShellPred(s, (rng.uniform(-20, 20), rng.uniform(-20, 20), 0.0), h.world.tick.t + 1.0, 7.0)
                      for s in range(6)]
            h.A_fn = lambda hh, sh=shells: build_A(hh.world, threats=Threats(shells=sh), slack=rng.uniform(0, 30))
            h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(
                target_id=A.target_id, win=rng.random() < 0.3, win_sub="DUEL",
                lose_clauses=frozenset(rng.choice(["", "f", "g"])), lose_tracks={})
            for _ in range(100):
                h.step()
            times.append(h.max_ms)
            self.assertLess(h.sum_ms / h.n, 4.0, f"mean {h.sum_ms / h.n:.2f} ms")
        self.assertLess(statistics.median(times), 4.0, f"max per run {times}")


if __name__ == "__main__":
    unittest.main()
