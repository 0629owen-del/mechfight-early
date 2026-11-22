"""Scenario fixtures for the decision layer (plan 14.2) on fakes: each asserts the deciding rule id in the
trace/label and the actuator outcome.

    python3 -m unittest tests.test_modes -v
"""
from __future__ import annotations

import math
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import (Harness, build_A, goal_entries, goal_exits, macro_changes)  # noqa: E402

from contracts import DamageRec, DropState, FightEval, FireRequest, Intent, Roof, Steer, Threats  # noqa: E402
import tactics_drop  # noqa: E402


def e_rule(h):
    """Naive LOSE (e) (R6.1 trigger) from geometry; also flags never-seen loadouts (prior-only) so the test
    proves modes filters them."""
    C = h.C

    def fn(w, A, macro, target, ctx):
        for tr in w.tracks.values():
            ta = A.per_track[tr.id]
            m = tr.lightning_range_mult or 1.0
            maybe = "lightning" in tr.weapons or tr.known_weapons.get("lightning", 0) > 0 or not tr.weapons
            if maybe and ((ta.d <= C.LIGHTNING_THREAT_R * m and ta.closing_speed > C.RUSH_CLOSING_MIN)
                          or ta.d <= C.LIGHTNING_KEEPOUT_R * m):
                return FightEval(target_id=tr.id, lose_clauses=frozenset("e"), lose_tracks={"e": tr.id})
        return FightEval()
    return fn


class TestLightningChase(unittest.TestCase):
    def test_lightning_80m_closing_kite(self):
        """lightning track at 80 m closing -> X3 KITE (LOSE (e) before it is in range)."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        h.world.add_mech(5, (80.0, 0.0, 0.0), weapons=("gun", "lightning"), vel=(-10.0, 0.0, 0.0))
        h.assessor.fn = e_rule(h)
        intent, action = h.step()
        self.assertEqual(intent.macro, "X3")
        self.assertEqual(intent.sub, "KITE")
        self.assertIn("T4:e", intent.trace)
        self.assertEqual(intent.rule, "R6.1")
        self.assertTrue(intent.label().startswith("X3.KITE"))
        self.assertLess(action["move"][0], 0.0)                     # backpedal away (east threat)

    def test_wall_48m_one_entry_per_cooldown_then_corner_bypass(self):
        """lightning track held at 48 m behind a wall for 600 decisions -> <= 1 X3 entry per
        DISENGAGE_COOLDOWN_S; it rounds the corner with LOS and closes -> immediate X3 (bypass)."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        tr = h.world.add_mech(5, (48.0, 0.0, 0.0), weapons=("gun", "lightning"))
        tr.visible = False                                          # memory-listed: fresh, no LOS
        h.assessor.fn = e_rule(h)
        los = {5: False}
        h.A_fn = lambda hh: build_A(hh.world, los=los)
        h.run(600)
        C = h.C
        enters = [r[0] for r in macro_changes(h.modes) if r[3] == "X3"]
        self.assertTrue(enters)
        for a, b in zip(enters, enters[1:]):
            self.assertGreaterEqual(b - a, C.DISENGAGE_COOLDOWN_S - 1e-3)
        # run until we are in X1 with (FLEE, 5) active, then the corner rush
        for _ in range(200):
            if h.modes.macro == "X1" and not h.modes.exit_memory.clear("FLEE", 5, h.world.tick.t):
                break
            h.step()
        self.assertEqual(h.modes.macro, "X1")
        self.assertFalse(h.modes.exit_memory.clear("FLEE", 5, h.world.tick.t))
        tr.visible, los[5] = True, True
        tr.vel = (-10.0, 0.0, 0.0)
        intent, _ = h.step()
        self.assertEqual(intent.macro, "X3", "T4 (e) bypass: LOS and closing")

    def test_stale_track_no_flee(self):
        """a 20 s stale lightning track -> no flee (LIGHTNING_TRACK_MAX_AGE_S)."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        tr = h.world.add_mech(5, (40.0, 0.0, 0.0), weapons=("gun", "lightning"), visible=False, fresh=False)
        tr.last_seen_t = h.world.tick.t - 20.0
        h.assessor.fn = e_rule(h)
        h.A_fn = lambda hh: build_A(hh.world, los={5: False})
        for _ in range(20):
            intent, _ = h.step()
            self.assertEqual(intent.macro, "X1")

    def test_prior_only_o8_only(self):
        """a prior-only lightning contact -> O8 only, no X3."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        h.world.add_mech(5, (40.0, 0.0, 0.0), weapons=(), known={})
        h.assessor.fn = e_rule(h)
        intent, action = h.step()
        self.assertEqual(intent.macro, "X1")
        self.assertIn("O8", h.ostate.active)
        self.assertIn("O8", [f.level for f in intent.leg_filters])
        self.assertEqual(h.ostate.keep_away_contacts, [5])
        # the O8 filter never lets the legs close on the contact
        self.assertLessEqual(action["move"][0], 1e-9)

    def test_mixed_clause_x3_exit(self):
        """(e) and (f) at X3 entry, (f) clears, the track stays at 48 m behind a wall -> one X3 exit,
        (FLEE, track) written, no re-entry within DISENGAGE_COOLDOWN_S."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        tr = h.world.add_mech(5, (48.0, 0.0, 0.0), weapons=("gun", "lightning"))
        tr.visible = False
        h.A_fn = lambda hh: build_A(hh.world, los={5: False})

        def fn(w, A, macro, target, ctx):
            cl = "ef" if w.tick.t < 101.0 else "e"
            return FightEval(target_id=5, lose_clauses=frozenset(cl), lose_tracks={"e": 5})
        h.assessor.fn = fn
        h.pre = lambda hh: hh.world.damage(5.0, 5) if hh.world.tick.t < 101.0 else None
        h.run(100)
        C = h.C
        mc = macro_changes(h.modes)
        self.assertEqual(mc[0][5], "T4:ef")
        exits = [r for r in mc if r[2] == "X3"]
        self.assertGreaterEqual(len(exits), 1)
        self.assertEqual(exits[0][5], "T6:R6.1")
        t_exit = exits[0][0]
        later = [r[0] for r in mc if r[3] == "X3" and r[0] > t_exit]
        self.assertTrue(not later or later[0] - t_exit >= C.DISENGAGE_COOLDOWN_S - 1e-3)
        self.assertEqual(len([r for r in exits if r[0] < t_exit + C.DISENGAGE_COOLDOWN_S]), 1)


class TestDropCapture(unittest.TestCase):
    def test_committed_capture_with_intruder_stays(self):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        dr = DropState(20000, (0.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 90.0, 99.0)
        h.world.drops[20000] = dr
        with mock.patch.object(tactics_drop, "go_test", lambda ctx, d: (True, False)):
            for _ in range(3):
                h.step()
            self.assertEqual((h.modes.goal, h.modes.sub), ("M7", "CAPTURE"))
            dr.progress, dr.holder = 3.0, 0
            h.world.add_mech(7, (60.0, 0.0, 0.0), weapons=("gun", "lightning"), vel=(-5.0, 0.0, 0.0))
            h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(
                target_id=7, win=True, win_sub="DUEL", lose_clauses=frozenset("eg"), lose_tracks={"e": 7, "g": 7})
            for _ in range(10):
                intent, action = h.step()
                self.assertEqual((intent.macro, intent.goal, h.modes.sub), ("X1", "M7", "CAPTURE"))
                self.assertEqual(intent.rule, "R8.5")
        self.assertTrue(any(f.target_id == 7 for f in intent.fire))
        self.assertTrue(action["fire"][0])                           # gun at the intruder
        self.assertTrue(h.assessor.calls[-1][3]["committed_capture"])


class TestHideAmbush(unittest.TestCase):
    def test_hide_no_flap(self):
        """HIDE does not flap when reaching cover breaks LOS (tracks seen within HIDE_CLEAR_S)."""
        h = Harness(t=100.0)
        a = h.world.add_mech(5, (150.0, 0.0, 0.0))
        b = h.world.add_mech(6, (0.0, 150.0, 0.0))
        h.nav.cover = (0.0, -10.0)
        h.step()
        self.assertEqual(h.modes.goal, "M3H")
        a.visible = b.visible = False                               # behind cover: memory-listed only
        h.A_fn = lambda hh: build_A(hh.world, los={5: False, 6: False})
        h.run(140)
        self.assertEqual(len(goal_entries(h.modes, "M3H")), 1)
        self.assertEqual(goal_exits(h.modes, "M3H"), [])
        self.assertEqual(h.modes.goal, "M3H")

    def test_m3a_keeps_ambush_bearing(self):
        """M3A keeps the ambush bearing when a hostile is visible but out of its reach (AIM9 > AIM6)."""
        h = Harness(t=100.0, weapons=("gun", "lightning"))
        h.world.add_ghost(-1, (40.0, 0.0), "near")
        h.pre = lambda hh: hh.world.hear("step", -1, "near")
        h.step()
        self.assertEqual(h.modes.goal, "M3A")
        h.world.add_mech(5, (0.0, 150.0, 0.0))
        h.A_fn = lambda hh: build_A(hh.world, over={5: {"in_its_reach": False}})
        intent, action = h.step()
        self.assertEqual(intent.goal, "M3A")
        self.assertEqual(h.arb.last_owners["aim"], "R4.3")
        self.assertAlmostEqual(action["aim"][0], 0.0, places=6)
        # inside its reach -> AIM6 owns the aim
        h.A_fn = lambda hh: build_A(hh.world, over={5: {"in_its_reach": True}})
        intent, action = h.step()
        self.assertEqual(h.arb.last_owners["aim"], "AIM6")

    def test_no_first_fire_in_m3h_m3a_m6(self):
        cases = []
        h = Harness(t=100.0)
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.world.add_mech(6, (0.0, 150.0, 0.0))
        cases.append(("M3H", h))
        h = Harness(t=100.0, weapons=("gun", "lightning"))
        h.world.add_ghost(-1, (40.0, 0.0), "near")
        h.pre = lambda hh: hh.world.hear("step", -1, "near")
        h.world.add_mech(5, (0.0, 150.0, 0.0))
        h.A_fn = lambda hh: build_A(hh.world, over={5: {"in_its_reach": False}})
        cases.append(("M3A", h))
        h = Harness(t=100.0)
        h.world.me.hp = 300.0
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.A_fn = lambda hh: build_A(hh.world, los={5: False})
        cases.append(("M6", h))
        for goal, h in cases:
            with self.subTest(goal=goal):
                h.inject = lambda hh, it: it.fire.append(FireRequest(0, "F1", 5))
                intent, action = h.step()
                self.assertEqual(intent.goal, goal)
                self.assertTrue(intent.fire_self_defense_only)
                self.assertEqual(action["fire"], [False, False])
                h.world.damage(50.0, 5)                             # now it is self-defense
                intent, action = h.step()
                self.assertTrue(action["fire"][0])


class TestTimeoutsAndLoot(unittest.TestCase):
    def test_x3_timeout_both_branches(self):
        for fire in (True, False):
            with self.subTest(under_fire=fire):
                h = Harness(t=100.0)
                h.world.add_mech(5, (150.0, 0.0, 0.0))
                h.A_fn = lambda hh: build_A(hh.world, A_ratio=0.5)
                h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, lose_clauses=frozenset("f"))
                if fire:
                    h.pre = lambda hh: hh.world.damage(5.0, 5)
                h.run(int(h.C.DISENGAGE_MAX_S * 10) + 2)
                mc = macro_changes(h.modes)
                if fire:
                    self.assertEqual((h.modes.macro, h.modes.sub), ("X2", "CORNERED"))
                    self.assertEqual(mc[-1][5], "T6:timeout_fire")
                    self.assertTrue(h.modes.exit_memory.clear("FLEE", "*", h.world.tick.t))
                else:
                    self.assertEqual(h.modes.macro, "X1")
                    self.assertEqual(mc[-1][5], "T6:timeout")
                    self.assertAlmostEqual(h.modes.exit_memory.until("FLEE", "*"),
                                           mc[-1][0] + h.C.DISENGAGE_COOLDOWN_S, places=6)

    def test_loot_window_exploration(self):
        h = Harness(t=5.0, weapons=("gun",))
        intent, action = h.step()
        self.assertEqual((intent.macro, intent.goal, intent.rule), ("X1", "M1", "8.3x"))
        self.assertEqual(h.arb.last_owners["legs"].split(":")[0], "L8")
        self.assertGreater(math.hypot(*action["move"]), 0.5)


class TestEarlyGameAndNpc(unittest.TestCase):
    def test_r12_neighbor_set_passed_and_contested_boost(self):
        from contracts import PickupMem
        h = Harness(t=5.0, weapons=("gun",))
        h.world.pickups[8001] = PickupMem(8001, "laser", (50.0, 0.0, 0.0), 5.0)
        h.world.add_mech(5, (0.0, 120.0, 0.0), weapons=("gun",))
        intent, action = h.step()
        self.assertEqual(h.modes.neighbor, 5)
        self.assertEqual(h.assessor.calls[-1][3]["neighbor"], 5)
        self.assertEqual(intent.goal, "M1")
        self.assertIn("R1.2", [b.rule for b in intent.boost])   # our ETA <= its ETA + margin: contested
        self.assertTrue(action["boost"])
        h.world.tick.t = h.C.LOOT_END_T
        h.step()
        self.assertIsNone(h.modes.neighbor)                      # cleared at LOOT_END_T

    def test_r72_npc_farming(self):
        h = Harness(t=100.0, weapons=("gun", "lightning"))
        h.world.me.pos = (0.0, 0.0, 0.0)
        h.world.add_mech(101, (20.0, 0.0, 0.0), weapons=(), kind="npc", hp=350.0, max_hp=350.0)
        h.modes.last_threat_sound_t = -1e9
        h.world.tick.t = 100.0
        # opportunistic M1 needs the NPC within HOLD_LOOT_R
        intent, action = h.step()
        self.assertEqual((intent.goal, intent.rule), ("M1", "R7.2"))
        self.assertTrue(any(f.rule == "R7.2" and f.target_id == 101 for f in intent.fire))
        # a mech appears -> the R7.2 gate fails -> abort + blacklist
        h.world.add_mech(5, (300.0, 0.0, 0.0))
        h.step()
        self.assertFalse(h.modes.exit_memory.clear("LOOT", ("n", 101), h.world.tick.t))

    def test_r72_self_defense_and_magazine_gate(self):
        """Integration gap-fill: an NPC that damaged us is an R7.2 target although a mech is in view; once
        farming, the gun gate counts the magazine, so low ammo mid-farm does not abort."""
        h = Harness(t=100.0, weapons=("gun",))
        h.world.add_mech(101, (20.0, 0.0, 0.0), weapons=(), kind="npc", hp=350.0, max_hp=350.0)
        h.world.add_mech(5, (300.0, 0.0, 0.0))
        h.modes.last_threat_sound_t = -1e9
        intent, _ = h.step()
        self.assertNotEqual(intent.rule, "R7.2")                  # mech in view, not attacked: no farming
        h.world.damage_log.append(DamageRec(h.world.tick.t, 25.0, "npc", 0.0, 101))
        intent, _ = h.step()
        self.assertEqual((intent.goal, intent.rule), ("M1", "R7.2"))
        h2 = Harness(t=100.0, weapons=("gun",))                    # quiet farm, ammo runs low mid-farm
        h2.world.add_mech(101, (20.0, 0.0, 0.0), weapons=(), kind="npc", hp=350.0, max_hp=350.0)
        h2.modes.last_threat_sound_t = -1e9
        intent, _ = h2.step()
        self.assertEqual(intent.rule, "R7.2")
        h2.world.me.weapons[0].ammo = 2
        intent, _ = h2.step()
        self.assertEqual(intent.rule, "R7.2")
        self.assertTrue(h2.modes.exit_memory.clear("LOOT", ("n", 101), h2.world.tick.t))

    def test_h4_compares_current_score(self):
        """H4: switching compares against the current item's score of this tick, not its selection-time score
        (a far item chosen alone, walked up to; a mediocre new item must not steal it)."""
        from contracts import CubeObs, PickupMem
        h = Harness(t=5.0, weapons=("gun",))
        h.world.pickups[8001] = PickupMem(8001, "laser", (300.0, 0.0, 0.0), 5.0)
        h.step()
        self.assertEqual(h.modes.loot_item[0], ("p", 8001))
        h.world.me.pos = (90.0, 0.0, 0.0)                          # walked 90 m: the laser's score rose
        h.world.pickups[8001].last_seen = h.world.tick.t
        # a cube a bit better than the laser now (< LOOT_SWITCH_MARGIN x its current score) but well above
        # 1.25 x the laser's selection-time score
        h.world.cubes[5001] = CubeObs(5001, (90.0, 54.0, 0.0))
        for _ in range(int(h.C.LOOT_MIN_HOLD_S / 0.1) + 5):
            h.step()
            self.assertEqual(h.modes.loot_item[0], ("p", 8001))


class TestOverlayScopes(unittest.TestCase):
    def test_x4_overlay_scope(self):
        """X4: O4 does not move the legs, O1 does not boost at the point, O8 off."""
        h = Harness(t=380.0, weapons=("gun", "laser"))
        h.world.add_mech(5, (40.0, 0.0, 0.0), weapons=("gun", "lightning"), visible=True)
        hit = DamageRec(380.0, 50.0, "gun", 1.0, 9, False, True)
        h.A_fn = lambda hh: build_A(hh.world, slack=2.0, threats=Threats(unseen_hits=[hit]))
        h.step()
        intent, action = h.step()
        self.assertEqual(intent.macro, "X4")
        self.assertNotIn("L3b", [lr.level for lr in intent.legs])
        self.assertIn("R3.1", [a.rule for a in intent.aim])        # AIM7 kept (aim only)
        o1 = [b for b in intent.boost if b.owner == "O1"]
        self.assertTrue(o1 and not o1[0].want)
        self.assertFalse(action["boost"])
        self.assertNotIn("O8", [f.level for f in intent.leg_filters])
        self.assertNotIn("O8", h.ostate.active)

    def test_lose_h_neighbor(self):
        h = Harness(t=10.0, weapons=("gun",))
        h.world.add_mech(5, (100.0, 0.0, 0.0), weapons=("gun", "laser"))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, lose_clauses=frozenset("h"),
                                                                    lose_tracks={"h": 5})
        intent, _ = h.step()
        self.assertEqual(intent.macro, "X3")
        self.assertIn("T4:h", intent.trace)

    def test_r63_laser_in_view(self):
        """GL bot with a seen laser holder in LOS and A < HOLD_A_MIN -> X3 BREAK_LOS (R6.3), then one entry
        per cooldown while the holder stays put."""
        h = Harness(t=100.0, weapons=("gun", "lightning"))
        tr = h.world.add_mech(5, (150.0, 0.0, 0.0), weapons=("gun", "laser"))
        state = {"los": True}

        def a_fn(hh):
            state["los"] = hh.modes.macro != "X3"                   # in X3 we reach cover: LOS breaks
            tr.visible = state["los"]
            return build_A(hh.world, los={5: state["los"]})
        h.A_fn = a_fn
        h.assessor.fn = lambda w, A, macro, target, ctx: (
            FightEval(target_id=5, lose_clauses=frozenset("i"), lose_tracks={"i": 5}) if A.per_track[5].los
            else FightEval(target_id=5))
        intent, _ = h.step()
        intent, _ = h.step()
        self.assertEqual((intent.macro, intent.sub, intent.rule), ("X3", "BREAK_LOS", "R6.3"))
        h.run(600)
        C = h.C
        enters = [r[0] for r in macro_changes(h.modes) if r[3] == "X3"]
        for a, b in zip(enters, enters[1:]):
            self.assertGreaterEqual(b - a, C.DISENGAGE_COOLDOWN_S - 1e-3)
        # LASER_VIEW_RESPONSE gl_only: a GZ bot ignores (i)
        h2 = Harness(t=100.0, weapons=("gun", "laser"))
        h2.world.add_mech(5, (150.0, 0.0, 0.0), weapons=("gun", "laser"))
        h2.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, lose_clauses=frozenset("i"),
                                                                     lose_tracks={"i": 5})
        intent, _ = h2.step()
        self.assertEqual(intent.macro, "X1")


class TestHoldSpots(unittest.TestCase):
    def test_roof_strafe_no_m2_and_cover_shuffle(self):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        roof = Roof(3, 20.0, 0.0, 20.0, 900.0, 30.0, (50.0, 50.0), [],
                    [((35.0, 65.0), (65.0, 65.0), (0.0, 1.0)), ((65.0, 35.0), (35.0, 35.0), (0.0, -1.0)),
                     ((65.0, 65.0), (65.0, 35.0), (1.0, 0.0)), ((35.0, 35.0), (35.0, 65.0), (-1.0, 0.0))])
        h.map.roofs = [h.map.roofs[0], None, None, roof]
        h.map.roof_fn = lambda x, y: 3 if (35 <= x <= 65 and 35 <= y <= 65) else 0
        h.nav.spots["roof"] = ((50.0, 50.0), 1.0, 3)
        h.world.me.pos = (50.0, 50.0, 20.0)
        k = {"i": 0}

        def pre(hh):                                                # strafe around on the roof
            k["i"] += 1
            hh.world.me.pos = (50.0 + 12.0 * math.sin(k["i"] / 7.0), 50.0 + 10.0 * math.cos(k["i"] / 5.0), 20.0)
        h.pre = pre
        h.run(300)
        self.assertEqual(goal_entries(h.modes, "M2"), [])
        self.assertEqual(h.modes.goal, "M3")
        self.assertEqual(h.modes.sub, "ROOF")
        # cover spot: shuffle within HOLD_LEAVE_R does not enter M2, beyond it does
        h = Harness(t=100.0)
        h.nav.spots["hold"] = ((0.0, 0.0), 1.0, 0)
        h.step()
        self.assertEqual(h.modes.goal, "M3")
        h.world.me.pos = (h.C.HOLD_LEAVE_R - 1.0, 0.0, 0.0)
        h.run(30)
        self.assertEqual(goal_entries(h.modes, "M2"), [])
        h.world.me.pos = (h.C.HOLD_LEAVE_R + 1.0, 0.0, 0.0)
        h.step()
        self.assertEqual(h.modes.goal, "M2")


class TestUserDecisions(unittest.TestCase):
    def test_d14_alive3_no_initiation(self):
        h = Harness(t=200.0)
        h.world.tick.alive = 3
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_sub="DUEL")
        for _ in range(5):
            intent, _ = h.step()
            self.assertEqual(intent.macro, "X1")
        h.world.damage(40.0, 5)                                      # self-defense -> WIN allowed
        intent, _ = h.step()
        self.assertEqual(intent.macro, "X2")
        # cornered also allowed
        h = Harness(t=200.0)
        h.world.tick.alive = 3
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, cornered=True)
        intent, _ = h.step()
        self.assertEqual(intent.macro, "X2")

    def test_d15_alive2_behind_fights(self):
        h = Harness(t=340.0)
        h.world.tick.alive = 2
        h.world.add_mech(5, (150.0, 0.0, 0.0), max_hp=2000.0)
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, lose_clauses=frozenset("f"))
        intent, _ = h.step()
        self.assertEqual(intent.macro, "X2")
        self.assertIn("T3", intent.trace)
        for _ in range(20):                                          # LOSE suppressed (except d)
            intent, _ = h.step()
            self.assertEqual(intent.macro, "X2")
        h = Harness(t=380.0)
        h.world.tick.alive = 2
        h.world.add_mech(5, (150.0, 0.0, 0.0), max_hp=2000.0)
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, lose_clauses=frozenset("f"))
        intent, _ = h.step()
        self.assertEqual(intent.macro, "X4")


class TestZoneOnlyDisengage(unittest.TestCase):
    """LOSE (d) alone (we left the fight for the zone): X3 runs along the zone field, not to cover."""

    def _x3(self, clauses):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        h.world.add_mech(5, (0.0, 150.0, 0.0))                     # threat to the north
        h.nav.cover = (-40.0, 0.0)                                   # cover to the west
        h.nav.descent = (1.0, 0.0)                                   # safety to the east
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_sub="DUEL")
        h.step()
        self.assertEqual(h.modes.macro, "X2")
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(
            target_id=5, lose_clauses=frozenset(clauses), lose_tracks={})
        for _ in range(int(h.C.ENGAGE_MIN_HOLD_S / 0.1) + 3):
            intent, action = h.step()
        self.assertEqual((intent.macro, intent.sub), ("X3", "BREAK_LOS"))
        return h, intent, action

    def test_zone_clause_alone_heads_for_the_zone(self):
        h, intent, action = self._x3("d")
        self.assertEqual(intent.rule, "X3:d")
        self.assertGreater(action["move"][0], 0.9)                  # east, along the zone field
        self.assertEqual(h.nav.cover_calls, 0)
        h.nav.descent = None                                         # inside the safe set: cover as before
        intent, action = h.step()
        self.assertLess(action["move"][0], -0.9)

    def test_other_clauses_still_break_los(self):
        h, intent, action = self._x3("ad")
        self.assertLess(action["move"][0], -0.9)                    # west, to the cover point

    def test_flee_with_no_threat_left_walks_the_zone_field(self):
        """X3 with its threat track gone stood still until its exit (49 s once, then a zone death)."""
        h, intent, action = self._x3("a")
        h.world.tracks.clear()
        h.A_fn = lambda hh: build_A(hh.world)
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval()
        intent, action = h.step()
        self.assertEqual((intent.macro, intent.rule), ("X3", "X3"))
        self.assertGreater(action["move"][0], 0.9)                  # east, along the zone field
        h2, _, _ = self._x3("a")
        h2.world.tracks.clear()
        h2.nav.descent = None                                        # inside the safe set: nothing to do
        h2.A_fn = lambda hh: build_A(hh.world)
        h2.assessor.fn = lambda w, A, macro, target, ctx: FightEval()
        intent, action = h2.step()
        self.assertEqual((intent.macro, intent.rule), ("X3", "X3"))
        self.assertEqual(list(action["move"]), [0.0, 0.0])

    # the zone field's first step can be a ledge or a wall: the legs must carry the jump / climb with them
    def _assert_zone_nav(self, intent):
        self.assertIn(("CLIMB", "R13.2"), [(j.owner, j.rule) for j in intent.jump])
        self.assertIn(("CLIMB", True), [(b.owner, b.want) for b in intent.boost])
        owners = [lr for lr in intent.legs if lr.direction == (1.0, 0.0)]
        self.assertTrue(owners and all(lr.nav for lr in owners), "the zone direction is a nav direction")

    def test_zone_clause_alone_forwards_jump_and_climb(self):
        h, intent, action = self._x3("d")
        h.nav.steer = Steer((1.0, 0.0), jump=True, climb_boost=True, rule="R13.2")
        intent, action = h.step()
        self.assertEqual(intent.rule, "X3:d")
        self._assert_zone_nav(intent)

    def test_flee_with_no_threat_forwards_jump_and_climb(self):
        h, intent, action = self._x3("a")
        h.world.tracks.clear()
        h.A_fn = lambda hh: build_A(hh.world)
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval()
        h.nav.steer = Steer((1.0, 0.0), jump=True, climb_boost=True, rule="R13.2")
        intent, action = h.step()
        self.assertEqual((intent.macro, intent.rule), ("X3", "X3"))
        self._assert_zone_nav(intent)

    def test_urgent_rotation_forwards_jump_and_climb(self):
        h = Harness(t=100.0)
        h.modes._owner_level = "L8"
        ctx = types.SimpleNamespace(world=h.world, nav=h.nav, intent=Intent(), A=build_A(h.world))
        h.nav.steer = Steer((1.0, 0.0), jump=True, climb_boost=True, rule="R13.2")
        self.assertEqual(h.modes._r_M2U(ctx, None), "M2U")
        self._assert_zone_nav(ctx.intent)
        # no field direction: the straight line to the next centre is an ordinary direction
        ctx = types.SimpleNamespace(world=h.world, nav=h.nav, intent=Intent(), A=build_A(h.world))
        h.nav.steer = Steer(None)
        h.modes._r_M2U(ctx, None)
        self.assertEqual([lr.nav for lr in ctx.intent.legs], [False])
        self.assertEqual(ctx.intent.jump, [])

    def test_walk_to_is_a_nav_direction_and_own_vectors_are_not(self):
        h = Harness(t=100.0)
        h.modes._owner_level = "L8"
        ctx = types.SimpleNamespace(world=h.world, nav=h.nav, intent=Intent(), A=build_A(h.world))
        h.modes.walk_to(ctx, "k", "M1", (200.0, 0.0))
        h.modes.legs(ctx, "M1", (0.0, 1.0))
        self.assertEqual([lr.nav for lr in ctx.intent.legs], [True, False])

    def test_a_raw_line_fallback_is_not_a_nav_direction(self):
        """follow() failing (or go() falling back) leaves a straight line to the goal: the arbiter's probe still
        has to vet it, or the bot walks into the wall between."""
        h = Harness(t=100.0)
        h.modes._owner_level = "L8"
        ctx = types.SimpleNamespace(world=h.world, nav=h.nav, intent=Intent(), A=build_A(h.world))

        def boom(*a):
            raise RuntimeError("follow failed")
        h.nav.follow = boom
        h.modes.walk_to(ctx, "k", "M1", (200.0, 0.0))
        h.nav.follow = lambda me, key, goal: Steer((1.0, 0.0), rule="8.9f")
        h.modes.walk_to(ctx, "k", "M1", (200.0, 0.0))
        self.assertEqual([lr.nav for lr in ctx.intent.legs], [False, False])

    def test_approach_legs_are_nav_directions(self):
        """The approach to a duel target (laser RZ1, gun RG1) follows a path: nav=True."""
        for weapons, rule, far, sub in ((("gun", "laser"), "RZ1", "LASER_HOLD_MAX", "DUEL"),
                                        (("gun",), "RG1", "GUN_HOLD_MAX", "DUEL_GUN")):
            h = Harness(t=100.0, weapons=weapons)
            h.world.add_mech(5, (getattr(h.C, far) + 40.0, 0.0, 0.0))
            h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_sub=sub)
            intent, _ = h.step()
            owners = [lr for lr in intent.legs if lr.rule == rule]
            self.assertTrue(owners and all(lr.nav for lr in owners), (rule, [(lr.rule, lr.nav) for lr in intent.legs]))

    def test_urgent_rotation_waits_at_a_climb_foot(self):
        """R13.2w from the zone steer: no straight line into the face and no ground sprint."""
        h = Harness(t=100.0)
        h.modes._owner_level = "L8"
        ctx = types.SimpleNamespace(world=h.world, nav=h.nav, intent=Intent(), A=build_A(h.world))
        h.nav.steer = Steer(None, rule="R13.2w")
        h.modes._r_M2U(ctx, None)
        self.assertEqual([lr.direction for lr in ctx.intent.legs], [None])
        self.assertEqual([b.want for b in ctx.intent.boost], [False])

    def test_final_approach_follows_the_zone_steer(self):
        """X4 FINAL_APPROACH with the zone closing: the field's direction, jump and climb replace the path's."""
        h = Harness(t=380.0, weapons=("gun", "laser"))
        h.world.me.pos = (60.0, -20.0, 0.0)
        z = h.world.zone
        z.next_center, z.next_radius, z.next_changed = (30.0, -20.0), 0.0, True
        h.A_fn = lambda hh: build_A(hh.world, slack=10.0)
        h.nav.eta = (5.0, True)
        h.nav.steer = Steer((0.0, 1.0), jump=True, climb_boost=True, rule="R13.2")
        intent, _ = h.step(advance=False)
        self.assertEqual((intent.macro, intent.sub), ("X4", "FINAL_APPROACH"))
        owners = [lr for lr in intent.legs if lr.rule == "R11.1"]
        self.assertEqual([(lr.direction, lr.nav) for lr in owners], [((0.0, 1.0), True)])
        self.assertIn(("CLIMB", "R13.2"), [(j.owner, j.rule) for j in intent.jump])
        self.assertIn(("CLIMB", True), [(b.owner, b.want) for b in intent.boost])


class TestRuleIdsInLabels(unittest.TestCase):
    def test_label_carries_rule(self):
        h = Harness(t=100.0, weapons=("gun", "laser"))
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_sub="DUEL")
        intent, _ = h.step()
        self.assertEqual(intent.label(), "X2.DUEL RZ1")
        self.assertLessEqual(len(intent.label()), 80)


class TestCoverRunJukeToggle(unittest.TestCase):
    """COVER_RUN_JUKE: "straight" (shipped, E3) runs to the X3 BREAK_LOS cover point in a straight line;
    "juke" (E3 off) lets O7 rotate that leg, as before the fix."""

    def _x3(self, value, clauses="a"):
        """A gun holder 150 m north (inside the juke band: O7 asks for a juke), cover 40 m west."""
        h = Harness(t=100.0, weapons=("gun", "laser"))
        self.assertEqual(h.C.COVER_RUN_JUKE, "straight")            # the shipped value
        h.C.COVER_RUN_JUKE = value
        h.world.add_mech(5, (0.0, 150.0, 0.0))
        h.nav.cover = (-40.0, 0.0)
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_sub="DUEL")
        h.step()
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(
            target_id=5, lose_clauses=frozenset(clauses), lose_tracks={})
        for _ in range(int(h.C.ENGAGE_MIN_HOLD_S / 0.1) + 3):
            intent, action = h.step()
        self.assertEqual((intent.macro, intent.sub, intent.rule), ("X3", "BREAK_LOS", "X3:" + clauses))
        self.assertIsNotNone(intent.juke)                            # O7 is on whatever the toggle says
        return h, intent, action

    def _owner(self, intent):
        own = [lr for lr in intent.legs if lr.rule == intent.rule]
        self.assertEqual(len(own), 1)
        return own[0]

    def test_both_values_on_the_same_cover_run(self):
        h, intent, action = self._x3("straight")
        leg, style = self._owner(intent), intent.juke
        self.assertEqual((leg.direction, leg.goal, leg.juke_ok), ((-1.0, 0.0), (-40.0, 0.0), False))
        self.assertEqual(h.arb.last_owners["legs"].split(":", 1)[1], "X3:a")
        self.assertEqual(list(action["move"]), [-1.0, 0.0])          # straight at the cover point
        h, intent, action = self._x3("juke")
        leg = self._owner(intent)
        self.assertEqual(intent.juke, style)                         # same seed: the same O7 request
        self.assertEqual((leg.direction, leg.goal, leg.juke_ok), ((-1.0, 0.0), (-40.0, 0.0), True))
        self.assertEqual(h.arb.last_owners["legs"].split(":", 1)[1], "X3:a+R12.1")
        a = style.side * style.angle                                 # the arbiter turned the leg by the juke
        self.assertAlmostEqual(action["move"][0], -math.cos(a), places=9)
        self.assertAlmostEqual(action["move"][1], -math.sin(a), places=9)
        # at most cos(JUKE_ANGLE_MIN_DEG) of the speed is left toward the cover point
        self.assertLessEqual(-action["move"][0], math.cos(math.radians(h.C.JUKE_ANGLE_MIN_DEG)) + 1e-9)

    def test_juke_value_leaves_the_zone_run_straight(self):
        """LOSE (d) alone walks the zone field, not to a cover point (a later fix): that leg is not E3's and
        stays straight under "juke"; the cover run it falls back to inside the safe set is juked."""
        h, intent, action = self._x3("juke", clauses="d")
        h.nav.descent = (1.0, 0.0)                                   # safety to the east
        intent, action = h.step()
        self.assertFalse(self._owner(intent).juke_ok)
        self.assertEqual(list(action["move"]), [1.0, 0.0])
        h.nav.descent = None                                         # inside the safe set: the cover point
        intent, action = h.step()
        self.assertTrue(self._owner(intent).juke_ok)
        self.assertTrue(h.arb.last_owners["legs"].endswith("X3:d+R12.1"))


class TestVisibleThreatFirstToggle(unittest.TestCase):
    """VISIBLE_THREAT_FIRST: True (shipped, E4) makes CORNERED fight the mech we can see; False (E4 off) keeps
    the X3 threat track, seen or not, as before the fix (we aimed at an unseen attacker and held fire)."""

    def _cornered_by_unseen_attacker(self, value, chaser_visible):
        """Mech 9, never seen, is a damage-inferred estimate 150 m north and the X3 trigger (LOSE (g));
        mech 5 is the chaser 150 m east. Returns the harness in X3 with the next evaluation cornered."""
        h = Harness(t=100.0, weapons=("gun",))
        self.assertIs(h.C.VISIBLE_THREAT_FIRST, True)               # the shipped value
        h.C.VISIBLE_THREAT_FIRST = value
        chaser = h.world.add_mech(5, (150.0, 0.0, 0.0), weapons=("gun",), visible=chaser_visible)
        h.world.add_mech(9, (0.0, 150.0, 0.0), weapons=("gun",), visible=False, est=True)
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(
            target_id=9, lose_clauses=frozenset("g"), lose_tracks={"g": 9})
        intent, _ = h.step()
        self.assertEqual((intent.macro, h.modes.x3_threat), ("X3", 9))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(
            target_id=9, win=True, win_rule="CORNERED", cornered=True)
        return h, chaser

    def _assert_fights(self, h, intent, action, target, yaw, fires):
        self.assertEqual((intent.macro, intent.sub, intent.rule), ("X2", "CORNERED", "R6.2"))
        self.assertEqual(h.modes.target, target)
        self.assertEqual([f.target_id for f in intent.fire], [target])
        self.assertAlmostEqual(action["aim"][0], yaw, places=6)
        self.assertEqual(action["fire"][0], fires)

    def test_cornered_entry_for_both_values(self):
        """R6.2 entry with the chaser in view: True takes the chaser and shoots; False takes the unseen X3
        threat, turns north and holds fire."""
        for value, target, yaw, fires in ((True, 5, 0.0, True), (False, 9, math.pi / 2, False)):
            with self.subTest(VISIBLE_THREAT_FIRST=value):
                h, _ = self._cornered_by_unseen_attacker(value, chaser_visible=True)
                self.assertEqual(h.last[2].target_id, 5)             # assess's target is the mech in view
                intent, action = h.step()
                self.assertEqual(macro_changes(h.modes)[-1][2:], ("X3", "X2", target, "R6.2"))
                self._assert_fights(h, intent, action, target, yaw, fires)

    def test_cornered_swap_for_both_values(self):
        """CORNERED entered with nothing in view fights the X3 threat either way; when the chaser comes into
        view True swaps to it, False keeps the unseen track (pre-E4 CORNERED never changed its target)."""
        for value, target, yaw, fires in ((True, 5, 0.0, True), (False, 9, math.pi / 2, False)):
            with self.subTest(VISIBLE_THREAT_FIRST=value):
                h, chaser = self._cornered_by_unseen_attacker(value, chaser_visible=False)
                intent, action = h.step()
                self._assert_fights(h, intent, action, 9, math.pi / 2, False)
                chaser.visible = chaser.in_obs = True
                intent, action = h.step()
                self.assertEqual(h.last[2].target_id, 5)
                self._assert_fights(h, intent, action, target, yaw, fires)


if __name__ == "__main__":
    unittest.main()
