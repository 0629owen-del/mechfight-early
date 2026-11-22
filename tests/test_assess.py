"""Assessment tests (plan 8.1, 8.2, 8.5, 8.6, 8.8 slack, R0.1; 14.1 "DPS / A", "A_rush / RUSH_A_MIN",
"Terrain call", "Loadout", "Target priority"; 3.3 budget).

    cd bot && python3 -m unittest tests.test_assess -v
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import assess as AS  # noqa: E402
import config  # noqa: E402
import test_world as TW  # noqa: E402
import world as W  # noqa: E402
from contracts import TERRAIN_CALLS, Clock, ExitMemory, ZoneState  # noqa: E402

GUN = {"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0,
       "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12,
                 "reload": 2.4}}
LASER = {"kind": "laser", "cooldown": 0, "charge": 0,
         "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "cooldown": 1.65, "range": 280}}
LIGHTNING = {"kind": "lightning", "cooldown": 0,
             "stats": {"damage": 60, "cooldown": 0.25, "range": 45, "cone_deg": 30, "energy": 3}}


def calibrated():
    C = TW.bound_config()
    AS.calibrate_rush_a_min(C)
    return C


class Scene:
    """WorldModel + Assessor on a flat fake map, driven by synthetic observations."""

    def __init__(self, C, md=None):
        self.C = C
        self.map = md or TW.FakeMap()
        self.w = W.WorldModel(C, self.map, 3)
        self.a = AS.Assessor(C, self.map)
        self.tick = 0

    def run(self, mechs=(), weapons=(GUN, None), nav=None, **kw):
        o = TW.base_obs(weapons=list(weapons), **_obs_kw(kw))
        o["mechs"] = list(mechs)
        for k in ("events", "sounds", "beams", "projectiles", "drops"):
            if k + "_" in kw:
                o[k] = kw[k + "_"]
        o["tick"], o["t"] = self.tick, round(kw.get("t", self.tick * 0.05), 3)
        self.tick += 2
        config.update_live(self.C, o["self"])
        clk = Clock(time.perf_counter(), 1e6, 1e6)
        self.w.update(o, clk)
        A = self.a.update(self.w, nav, clk)
        return A


def _obs_kw(kw):
    return {k: v for k, v in kw.items() if not k.endswith("_")}


# ====================================================================== 8.1 DPS
class DpsTest(unittest.TestCase):
    def setUp(self):
        self.C = TW.bound_config()

    def test_gl_burst(self):
        g, lt = AS.enemy_wstats(self.C, "gun"), AS.enemy_wstats(self.C, "lightning")
        self.assertAlmostEqual(AS.raw_dps("gun", g, 0, self.C) + AS.raw_dps("lightning", lt, 0, self.C), 427.5)

    def test_gun_formula_rear_armor_size(self):
        C = self.C
        g = AS.enemy_wstats(C, "gun")
        sustain = 12 * 0.24 / (12 * 0.24 + 2.4)
        base = 45 / 0.24 * sustain * 0.95
        d = AS.dps("gun", g, 30, 0, 0, rear=False, armor=0.0, energy=None, C=C, enemy=False)
        self.assertAlmostEqual(d, base)
        self.assertAlmostEqual(AS.dps("gun", g, 30, 0, 0, rear=True, armor=0.0, energy=None, C=C, enemy=False), base * 1.5)
        self.assertAlmostEqual(AS.dps("gun", g, 30, 0, 0, rear=False, armor=0.08, energy=None, C=C, enemy=False), base * 0.92)
        # ammo below the horizon -> no sustain factor; reloading -> 0; out of range -> 0
        self.assertAlmostEqual(AS.dps("gun", g, 30, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False, ammo=12)
                               , base)
        self.assertEqual(AS.dps("gun", g, 30, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False, reloading=True), 0)
        self.assertEqual(AS.dps("gun", g, 231, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False), 0)
        # hit table interpolation (87.5 m -> 0.8) and size factor capped at HIT_P_CAP
        p = AS.dps("gun", g, 87.5, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False) / (45 / 0.24 * sustain)
        self.assertAlmostEqual(p, 0.8)
        p10 = AS.dps("gun", g, 150, 0, 10, rear=False, armor=0, energy=None, C=C, enemy=False) / (45 / 0.24 * sustain)
        self.assertAlmostEqual(p10, 0.3 * 1.5)
        # cubes scale damage
        self.assertAlmostEqual(AS.dps("gun", g, 30, 10, 0, rear=False, armor=0, energy=None, C=C, enemy=False), base * 1.4)

    def test_lightning_energy_range_los(self):
        C = self.C
        lt = AS.enemy_wstats(C, "lightning")
        full = AS.dps("lightning", lt, 30, 0, 0, rear=False, armor=0, energy=100, C=C, enemy=False)
        self.assertAlmostEqual(full, 240)
        self.assertAlmostEqual(AS.dps("lightning", lt, 30, 0, 0, rear=False, armor=0, energy=36, C=C, enemy=False), 120)
        self.assertEqual(AS.dps("lightning", lt, 30, 0, 0, rear=False, armor=0, energy=100, C=C, enemy=False, d3=46), 0)
        self.assertEqual(AS.dps("lightning", lt, 30, 0, 0, rear=False, armor=0, energy=100, C=C, enemy=False, los=False), 0)
        self.assertAlmostEqual(AS.dps("lightning", lt, 60, 0, 0, rear=False, armor=0, energy=100, C=C, enemy=True,
                                      range_mult=1.44), 240)

    def test_laser_mortar_npc(self):
        C = self.C
        lz = AS.enemy_wstats(C, "laser")
        self.assertAlmostEqual(AS.dps("laser", lz, 100, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False),
                               315 / 2.4 * C.HIT_LASER_OPEN)
        self.assertAlmostEqual(AS.dps("laser", lz, 100, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False,
                                      cover=True), 315 / 2.4 * C.HIT_LASER_COVER)
        self.assertEqual(AS.dps("laser", lz, 100, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False,
                                shorted=True), 0)
        mo = AS.enemy_wstats(C, "mortar")
        ours = AS.dps("mortar", mo, 100, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False)
        theirs = AS.dps("mortar", mo, 100, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=True)
        self.assertAlmostEqual(theirs / ours, (200 + 160) / 200)     # A-2: enemy direct + splash
        self.assertEqual(AS.dps("mortar", mo, 5, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False), 0)
        self.assertEqual(AS.dps("mortar", mo, 240, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=False), 0)
        npc = AS.enemy_wstats(C, "npc")
        self.assertAlmostEqual(AS.dps("npc", npc, 40, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=True), 25 * C.HIT_NPC)
        self.assertEqual(AS.dps("npc", npc, 50, 0, 0, rear=False, armor=0, energy=None, C=C, enemy=True), 0)

    def test_cone_halves(self):
        self.assertAlmostEqual(AS.lightning_cone_half(self.C, True), math.radians(30))
        self.assertAlmostEqual(AS.lightning_cone_half(self.C, False), math.radians(15) * self.C.LIGHTNING_CONE_MARGIN)


# ====================================================================== 8.2.1 A
class FightAdvantageTest(unittest.TestCase):
    def setUp(self):
        self.C = calibrated()

    def A_for(self, my_hp=1400, their_hp=1400, my_cubes=0, their_cubes=0, extra=(), d=120):
        sc = Scene(self.C)
        ms = [TW.mech(7, (d, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"), hp=their_hp, cubes=their_cubes)]
        ms += list(extra)
        A = sc.run(ms, weapons=(GUN, LASER), aim=(0.0, 0.0), hp=my_hp, cubes=my_cubes)
        return A, sc

    def test_symmetry(self):
        A, sc = self.A_for()
        self.assertEqual(A.target_id, 7)
        ta = A.per_track[7]
        self.assertAlmostEqual(ta.dps_to_us, ta.dps_from_us, places=9)
        self.assertAlmostEqual(A.A, 1.0, places=9)
        self.assertAlmostEqual(ta.A, 1.0, places=9)
        self.assertTrue(ta.aiming_at_us and ta.in_its_reach and ta.los)
        self.assertEqual(A.hostiles, [7])
        self.assertEqual(A.main_threat_id, 7)

    def test_monotonicity(self):
        base = self.A_for()[0].A
        self.assertGreater(self.A_for(my_hp=2000)[0].A, base)
        self.assertLess(self.A_for(their_hp=2000)[0].A, base)
        self.assertGreater(self.A_for(my_cubes=5)[0].A, base)
        self.assertLess(self.A_for(their_cubes=5)[0].A, base)
        two = self.A_for(extra=[TW.mech(8, (0, 100, 0), aim=(-math.pi / 2, 0), weapons=("gun", "laser"))])
        self.assertLess(two[0].A, base)
        self.assertEqual(sorted(two[0].hostiles), [7, 8])
        # a_vs alone ignores the second hostile
        self.assertAlmostEqual(two[1].a.a_vs(two[1].w, 7), 1.0, places=9)

    def test_not_aiming_counts_fraction(self):
        sc = Scene(self.C)
        A = sc.run([TW.mech(7, (120, 0, 0), aim=(math.pi / 2, 0), weapons=("gun", "laser"))],
                   weapons=(GUN, LASER), aim=(0.0, 0.0))
        ta = A.per_track[7]
        self.assertFalse(ta.aiming_at_us)
        # it faces north, we hit its side from the east: inside its front arc -> no rear bonus
        self.assertAlmostEqual(ta.dps_to_us / ta.dps_from_us, self.C.NOT_AIMING_DPS_FRAC)

    def test_unknown_loadout_worst_case(self):
        sc = Scene(self.C)
        # damage-inferred lightning user: its DPS at 30 m counts lightning
        A = sc.run([], weapons=(GUN, None), aim=(0.0, 0.0),
                   events_=[{"type": "damaged", "amount": 60, "weapon": "lightning", "bearing": 0.0, "by": 9}])
        ta = A.per_track[9]
        self.assertIn(9, A.hostiles)
        self.assertGreater(ta.dps_to_us, 0)


class RushTest(unittest.TestCase):
    def setUp(self):
        self.C = calibrated()

    def test_a_rush_strictly_decreasing(self):
        C = self.C
        ds = [C.BRAWL_DIST + 1 + 2.5 * i for i in range(60)]
        st = [AS._canonical_rush(C, d, 0.0) for d in ds]
        for a, b in zip(st, st[1:]):
            self.assertGreater(a, b)
        walk = C.path("mech.walk_speed")
        rt = [AS._canonical_rush(C, d, walk) for d in ds]
        for a, b in zip(rt, rt[1:]):
            if b > 0:
                self.assertGreater(a, b)
        self.assertEqual(AS._canonical_rush(C, 80, C.path("mech.boost_speed") - C.RUSH_CLOSING_MIN), 0.0)

    def test_calibration_reproduces_gp_table(self):
        C = self.C
        self.assertIsNotNone(C.RUSH_A_MIN)
        walk = C.path("mech.walk_speed")
        self.assertGreaterEqual(AS._canonical_rush(C, C.LIGHTNING_RUSH_R, walk), C.RUSH_A_MIN)
        self.assertGreaterEqual(AS._canonical_rush(C, C.LIGHTNING_RUSH_STILL_R, 0.0), C.RUSH_A_MIN)
        self.assertAlmostEqual(C.RUSH_A_MIN, min(C.rush_calibration.values()))
        print(f"\n  RUSH_A_MIN = {C.RUSH_A_MIN:.5f}  {C.rush_calibration}")

    def test_d7_toggle_margin(self):
        C = TW.bound_config()
        C.RUSH_GATE = "d7"
        AS.calibrate_rush_a_min(C)
        self.assertGreaterEqual(C.RUSH_A_MIN, C.ENGAGE_A_MIN_SAFE)

    def _rl1(self, d, extra=(), energy=70.0, vel=(0, 0, 0)):
        sc = Scene(self.C)
        ms = [TW.mech(7, (d, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"), vel=vel)] + list(extra)
        sc.run(ms, weapons=(GUN, LIGHTNING), aim=(0.0, 0.0), energy=energy, t=100.0)
        # still >= STILL_MIN_S, LOS dwell < LASER_VIEW_DWELL_S (else R6.3 LOSE (i) fires for a GL bot)
        A = sc.run(ms, weapons=(GUN, LIGHTNING), aim=(0.0, 0.0), energy=energy, t=100.6)
        fe = sc.a.evaluate_fight(sc.w, A, "X1", None, {})
        return fe, A

    def test_rl1_gp_range_rule_in_evaluate_fight(self):
        C = self.C
        fe, A = self._rl1(C.LIGHTNING_RUSH_R)
        self.assertTrue(fe.win, fe)
        self.assertEqual((fe.win_rule, fe.win_sub), ("RL1", "RUSH"))
        self.assertGreaterEqual(A.A_rush, C.RUSH_A_MIN)
        fe, _ = self._rl1(C.LIGHTNING_RUSH_STILL_R)          # standing target
        self.assertTrue(fe.win)
        fe, _ = self._rl1(C.LIGHTNING_RUSH_STILL_R + C.IMPL_RUSH_STEP_M)
        self.assertFalse(fe.win)
        fe, _ = self._rl1(C.LIGHTNING_RUSH_STILL_R, energy=30.0)   # still target: energy rule only for <= 75 m
        self.assertTrue(fe.win)
        fe, _ = self._rl1(60, vel=(C.RETREAT_SPEED_MIN + 1, 0, 0))   # RL2: backing away
        self.assertFalse(fe.win)

    def _r63(self, d, macro="X1", ctx=None, vel=(0, 0, 0), energy=100.0):
        sc = Scene(self.C)
        ms = [TW.mech(7, (d, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"), vel=vel)]
        sc.run(ms, weapons=(GUN, LIGHTNING), aim=(0.0, 0.0), energy=energy, t=100.0)
        A = sc.run(ms, weapons=(GUN, LIGHTNING), aim=(0.0, 0.0), energy=energy, t=100.0 + self.C.LASER_VIEW_DWELL_S)
        return sc.a.evaluate_fight(sc.w, A, macro, 7 if macro == "X2" else None, ctx or {})

    def test_r63_lose_i_after_dwell(self):
        # beyond RL1's rush trigger (UQ-2 rush exemption does not apply)
        fe = self._r63(self.C.LIGHTNING_RUSH_STILL_R + 20)
        self.assertIn("i", fe.lose_clauses)
        self.assertEqual(fe.lose_tracks["i"], 7)
        self.assertFalse(fe.win)

    def test_uq2_does_not_cancel_lightning_rush(self):
        """Integration decision (UQ-2 KEEP): LOSE (i) / R6.3 is exempt for a laser track that satisfies
        RL1's rush trigger, or that is our X2 target in sub-state RUSH."""
        C = self.C
        fe = self._r63(C.LIGHTNING_RUSH_R)                                   # RL1 range rule with energy
        self.assertNotIn("i", fe.lose_clauses)
        self.assertTrue(fe.win, fe)
        self.assertEqual((fe.win_rule, fe.win_sub), ("RL1", "RUSH"))
        fe = self._r63(C.LIGHTNING_RUSH_STILL_R)                             # standing target <= 110 m
        self.assertNotIn("i", fe.lose_clauses)
        fe = self._r63(C.LIGHTNING_RUSH_R, energy=C.RUSH_ENERGY_MIN - 20,    # energy gate fails, moving target
                       vel=(0.0, 6.0, 0.0))
        self.assertIn("i", fe.lose_clauses)
        far = C.LIGHTNING_RUSH_STILL_R + 20
        fe = self._r63(far, macro="X2", ctx={"sub": "RUSH"})                 # X2 RUSH on this target
        self.assertNotIn("i", fe.lose_clauses)
        fe = self._r63(far, macro="X2", ctx={"sub": "BRAWL"})
        self.assertIn("i", fe.lose_clauses)

    def test_second_hostile_can_veto_rush(self):
        C = self.C
        fe, A = self._rl1(C.LIGHTNING_RUSH_R)
        self.assertTrue(fe.win)
        crowd = [TW.mech(20 + i, (C.LIGHTNING_RUSH_R * math.cos(a), C.LIGHTNING_RUSH_R * math.sin(a), 0),
                         aim=(a + math.pi, 0), weapons=("gun", "laser")) for i, a in enumerate((0.6, -0.6, 1.2))]
        fe2, A2 = self._rl1(C.LIGHTNING_RUSH_R, extra=crowd)
        self.assertLess(A2.A_rush, A.A_rush)
        self.assertLess(A2.A_rush, C.RUSH_A_MIN)
        self.assertFalse(fe2.win and fe2.win_rule == "RL1")


# ====================================================================== target priority / threats
class TargetThreatTest(unittest.TestCase):
    def setUp(self):
        self.C = calibrated()

    def test_priority_and_hysteresis(self):
        sc = Scene(self.C)
        far = TW.mech(7, (180, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"))
        near = TW.mech(8, (60, 30, 0), aim=(math.pi, 0), weapons=("gun", "laser"), hp=300)
        A = sc.run([far], weapons=(GUN, LASER), aim=(0.0, 0.0))
        self.assertEqual(A.target_id, 7)
        A = sc.run([far, near], weapons=(GUN, LASER), aim=(0.0, 0.0))
        self.assertGreater(A.per_track[8].prio, A.per_track[7].prio * self.C.TARGET_SWITCH_MARGIN)
        self.assertEqual(A.target_id, 7)                       # held < TARGET_MIN_HOLD_S
        for _ in range(int(self.C.TARGET_MIN_HOLD_S / 0.1) + 1):
            A = sc.run([far, near], weapons=(GUN, LASER), aim=(0.0, 0.0))
        self.assertEqual(A.target_id, 8)

    def test_laser_charger_and_aim1(self):
        sc = Scene(self.C)
        ch = TW.mech(7, (40, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"), charging=True)
        A = sc.run([ch], weapons=(GUN, LIGHTNING), aim=(0.2, 0.0))
        self.assertTrue(A.per_track[7].charging_at_us)
        self.assertEqual(A.threats.laser_chargers, [7])
        self.assertEqual(sc.a.aim1, [7])
        sc.map.blockers.append(lambda a, b: True)
        ch2 = TW.mech(7, (41.5, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"), charging=True)
        A = sc.run([ch2], weapons=(GUN, LIGHTNING), aim=(0.2, 0.0))
        self.assertFalse(A.per_track[7].charging_at_us)        # no LOS from its muzzle

    def test_unseen_hits_and_shells(self):
        sc = Scene(self.C)
        A = sc.run([], aim=(0.0, 0.0),
                   events_=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.0, "by": 9}],
                   projectiles_=[{"id": 77, "kind": "mortar", "pos": [0, 0, 30], "vel": [0.5, 0, 5]}])
        self.assertEqual(len(A.threats.unseen_hits), 1)
        self.assertEqual([s.id for s in A.threats.shells], [77])
        self.assertEqual(sc.a.shell_hits_us, [77])


# ====================================================================== 8.8 slack
class SlackTest(unittest.TestCase):
    def setUp(self):
        self.C = TW.bound_config()

    def zone(self):
        z = ZoneState(center=(0, 0), radius=300, next_center=(100, 0), next_radius=100, shrink_start=60,
                      shrink_end=110, dps=20, from_center=(0, 0), from_radius=300)
        return z

    def test_tau_edge(self):
        z = self.zone()
        self.assertEqual(AS.tau_edge(z, (100, 0), 10.0, 20), math.inf)
        self.assertEqual(AS.tau_edge(z, (0, 310), 10.0, 20), 10.0)
        te = AS.tau_edge(z, (-150, 0), 10.0, 20)
        # |p - c(f)| = 150 + 100 f, r(f) = 300 - 200 f -> crossing at f = 0.5 -> tau = 85
        self.assertAlmostEqual(te, 85.0, delta=60 / 2 ** 19)

    def test_fallback_and_nav(self):
        C = self.C
        sc = Scene(C)
        zone = {"center": [0, 0], "radius": 300, "next_center": [100, 0], "next_radius": 100,
                "shrink_start": 60, "shrink_end": 110, "dps": 20}
        A = sc.run([], pos=(-150, 0, 0), zone=zone, t=10.0)
        dist = 250 - (100 - C.ZONE_EDGE_MARGIN_M)
        eta = dist / 11.7 * C.ZONE_FALLBACK_DETOUR_K * C.ZONE_ETA_SAFETY
        self.assertAlmostEqual(A.zone_eta, eta)
        self.assertFalse(A.zone_eta_settled)
        self.assertAlmostEqual(A.slack, A.tau_edge - 10.0 - eta)

        class Nav:
            def zone_eta(self, pos):
                return 12.5, True
        A = sc.run([], pos=(-150, 0, 0), zone=zone, t=10.0, nav=Nav())
        self.assertEqual((A.zone_eta, A.zone_eta_settled), (12.5, True))

        class NanNav:
            def zone_eta(self, pos):
                return float("nan"), False
        A = sc.run([], pos=(-150, 0, 0), zone=zone, t=10.0, nav=NanNav())
        self.assertAlmostEqual(A.zone_eta, eta)
        A = sc.run([], pos=(100, 0, 0), zone=zone, t=10.0)
        self.assertEqual(A.slack, math.inf)


# ====================================================================== R0.1 terrain call
class TerrainCallTest(unittest.TestCase):
    def setUp(self):
        self.C = TW.bound_config()

    def call(self, sight, wall, radius=200, t=100.0):
        md = TW.FakeMap(sight=sight, wall=wall)
        S, Cs, n = AS.terrain_stats(md, (0, 0), radius, self.C)
        return AS.terrain_call_from_stats(S, Cs, radius, t, self.C), S, Cs, n

    def test_branches(self):
        C = self.C
        self.assertEqual(self.call(120, 50, radius=C.CALL_R_LIGHTNING)[0], "LIGHTNING")
        self.assertEqual(self.call(C.CALL_S_LIGHTNING, 50)[0], "LIGHTNING")
        self.assertEqual(self.call(120, C.COVER_WALL_R)[0], "LIGHTNING")             # C = 1
        self.assertEqual(self.call(C.CALL_S_LASER, 50)[0], "LASER")
        self.assertEqual(self.call(120, 50)[0], "MIXED")
        self.assertEqual(self.call(C.CALL_S_LASER, 50, t=C.PHASE_T4_SHRINK)[0], "LIGHTNING")
        call, S, Cs, n = self.call(float("nan"), 50)                                  # degraded: no sight
        self.assertEqual(call, "MIXED")
        self.assertTrue(S != S)
        self.assertLessEqual(n, AS.CALL_MAX_SAMPLES * 1.3)

    def test_wall_share_threshold(self):
        C = self.C
        S, Cs = 120.0, C.CALL_C_LIGHTNING
        self.assertEqual(AS.terrain_call_from_stats(S, Cs, 200, 0, C), "LIGHTNING")
        self.assertEqual(AS.terrain_call_from_stats(200, C.CALL_C_LASER, 200, 0, C), "MIXED")
        self.assertEqual(AS.terrain_call_from_stats(200, C.CALL_C_LASER - 0.01, 200, 0, C), "LASER")

    def test_assessor_terrain_call_caches_per_next_circle(self):
        sc = Scene(self.C, TW.FakeMap(sight=200, wall=50))
        A = sc.run([], t=10.0)
        self.assertEqual((A.terrain_call, A.desired_loadout), ("LASER", "GZ"))
        n = sc.a.call_n
        sc.map._sight = 80
        A = sc.run([], t=10.1)                       # same next circle: cached
        self.assertEqual(A.terrain_call, "LASER")
        zone = {"center": [0, 0], "radius": 560, "next_center": [10, 0], "next_radius": 240,
                "shrink_start": 60, "shrink_end": 110, "dps": 20}
        A = sc.run([], t=10.2, zone=zone)
        self.assertEqual((A.terrain_call, A.desired_loadout), ("LIGHTNING", "GL"))
        self.assertGreater(n, 0)
        A = sc.run([], t=self.C.PHASE_T4_SHRINK, zone=zone)
        self.assertEqual(A.terrain_call, "LIGHTNING")

    @unittest.skipUnless(TW.HAVE_ARENA, "arena missing")
    def test_real_map(self):
        C = self.C
        md = TW.real_map(C)
        t0 = time.perf_counter()
        S, Cs, n = AS.terrain_stats(md, (-42.12, 7.11), 380, C)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  R0.1 on the recorded next circle: S {S:.1f} m, C {Cs:.3f}, n {n}, {ms:.2f} ms")
        self.assertTrue(0 < S < 400 and 0 <= Cs <= 1 and n > 500)
        self.assertIn(AS.terrain_call_from_stats(S, Cs, 380, 0, C), TERRAIN_CALLS)
        self.assertLess(ms, 15.0)


# ====================================================================== 8.5 loadout
class LoadoutTest(unittest.TestCase):
    def setUp(self):
        self.C = TW.bound_config()

    def test_rank_table(self):
        self.assertEqual(AS.loadout_rank(("gun", "lightning"), "LIGHTNING"), 3)
        self.assertEqual(AS.loadout_rank(("lightning", "gun"), "LASER"), 1)
        self.assertEqual(AS.loadout_rank(("laser",), "MIXED"), 0.8)
        self.assertEqual(AS.loadout_rank((), "MIXED"), 0)

    def test_d8_rank_rule(self):
        C = self.C
        self.assertEqual(C.SWAP_RULE, "rank")
        self.assertEqual(AS.slot_to_replace(["gun", "laser"], "lightning", "LIGHTNING", C), (1, 2.0))   # -> GL
        self.assertEqual(AS.slot_to_replace(["gun", "laser"], "lightning", "MIXED", C), (0, 1.0))       # -> ZL
        slot, gain = AS.slot_to_replace(["gun", "laser"], "lightning", "LASER", C)
        self.assertEqual(slot, 0)
        self.assertLess(gain, 0)                                                                         # keep GZ

    def test_mortar_first_empty_fill_same_kind(self):
        C = self.C
        self.assertEqual(AS.slot_to_replace(["gun", "mortar"], "laser", "LASER", C), (1, 3 - 0.6))
        self.assertEqual(AS.slot_to_replace(["mortar", "gun"], "lightning", "MIXED", C)[0], 0)
        self.assertEqual(AS.slot_to_replace(["gun", None], "mortar", "MIXED", C), (1, 0.6 - 0.3))
        self.assertEqual(AS.slot_to_replace([None, None], "laser", "MIXED", C), (0, 0.8))
        self.assertEqual(AS.slot_to_replace(["gun", "laser"], "laser", "MIXED", C), (None, 0.0))

    def test_gp_rz5(self):
        C = TW.bound_config()
        C.SWAP_RULE = "gp"
        slot, gain = AS.slot_to_replace(["laser", "gun"], "lightning", "LIGHTNING", C)
        self.assertEqual(slot, 1)
        self.assertGreater(gain, 0)


# ====================================================================== 8.6 drop go/no-go
class DropTest(unittest.TestCase):
    def setUp(self):
        self.C = TW.bound_config()

    def drop(self, **kw):
        d = {"id": 20000, "pos": [60, 0, 0], "radius": 8, "lands_in": 10, "capture_s": 4, "progress": 0,
             "holder": None}
        d.update(kw)
        return d

    def test_go_and_failures(self):
        C = self.C
        sc = Scene(C, TW.FakeMap(wall=10))
        sc.run([], t=80.0, aim=(math.pi, 0), drops_=[self.drop()])
        go, info = sc.a.drop_go(sc.w, 20000)
        self.assertTrue(go, info)
        self.assertFalse(info["open"])
        # rival seen near the ring
        sc.run([TW.mech(7, (90, 10, 0))], t=80.1, aim=(0.0, 0), drops_=[self.drop()])
        go, info = sc.a.drop_go(sc.w, 20000)
        self.assertFalse(go)
        self.assertIn("3:rivals", info["fail"])
        # low HP
        sc2 = Scene(C, TW.FakeMap(wall=10))
        sc2.run([], t=80.0, hp=500, drops_=[self.drop()])
        self.assertIn("2:hp", sc2.a.drop_go(sc2.w, 20000)[1]["fail"])
        # endgame and exit memory
        sc3 = Scene(C, TW.FakeMap(wall=10))
        sc3.run([], t=C.ENDGAME_T, drops_=[self.drop()])
        self.assertIn("5:endgame", sc3.a.drop_go(sc3.w, 20000)[1]["fail"])
        sc4 = Scene(C, TW.FakeMap(wall=10))
        sc4.run([], t=80.0, drops_=[self.drop()])
        em = ExitMemory()
        em.write("DROP", 20000, 200.0)
        self.assertIn("6:memory", sc4.a.drop_go(sc4.w, 20000, exit_memory=em)[1]["fail"])
        # far away -> ETA bound
        sc5 = Scene(C, TW.FakeMap(wall=10))
        sc5.run([], t=80.0, drops_=[self.drop(pos=[600, 0, 0])])
        self.assertIn("1:eta_slack", sc5.a.drop_go(sc5.w, 20000)[1]["fail"])

    def test_open_landing_with_ghost_contact(self):
        C = self.C
        sc = Scene(C, TW.FakeMap(wall=100))
        sc.run([], t=80.0, aim=(math.pi, 0), drops_=[self.drop()],
               sounds_=[{"kind": "gun", "bearing": 0.0, "dist": "far", "age": 0.1}])
        go, info = sc.a.drop_go(sc.w, 20000)
        self.assertTrue(info["open"])
        self.assertIn("4:open", info["fail"])


# ====================================================================== 8.2.3-8.2.6
class EvaluateFightTest(unittest.TestCase):
    def setUp(self):
        self.C = calibrated()

    def test_lose_e_kite_for_pb_z(self):
        sc = Scene(self.C)
        lt = TW.mech(7, (40, 0, 0), aim=(math.pi, 0), weapons=("gun", "lightning"))
        A = sc.run([lt], weapons=(GUN, LASER), aim=(0.0, 0.0), t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X1", None, {})
        self.assertIn("e", fe.lose_clauses)
        self.assertEqual(fe.lose_tracks["e"], 7)
        self.assertFalse(fe.win)

    def test_lose_f_low_hp_and_mutual_exclusion(self):
        sc = Scene(self.C)
        en = TW.mech(7, (120, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"), hp=200)
        A = sc.run([en], weapons=(GUN, LASER), aim=(0.0, 0.0), hp=300, t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X1", None, {})
        self.assertIn("f", fe.lose_clauses)
        self.assertFalse(fe.win)
        rng = random.Random(3)
        for _ in range(40):
            sc = Scene(self.C)
            ms = [TW.mech(7 + i, (rng.uniform(-200, 200), rng.uniform(-200, 200), 0),
                          aim=(rng.uniform(-3, 3), 0), weapons=("gun", rng.choice(("laser", "lightning", "mortar"))),
                          hp=rng.uniform(100, 1400)) for i in range(rng.randint(1, 3))]
            A = sc.run(ms, weapons=(GUN, rng.choice((LASER, LIGHTNING, None))), aim=(rng.uniform(-3, 3), 0),
                       hp=rng.uniform(100, 1400), energy=rng.uniform(0, 100), t=rng.uniform(61, 370))
            fe = sc.a.evaluate_fight(sc.w, A, rng.choice(("X1", "X2", "X3")), None, {})
            self.assertFalse(fe.win and fe.lose_clauses and not fe.cornered)

    def test_rz1_band_and_loot_only(self):
        C = self.C
        sc = Scene(C)
        en = TW.mech(7, (120, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",), hp=600)
        A = sc.run([en], weapons=(GUN, LASER), aim=(0.0, 0.0), t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X1", None, {})
        self.assertTrue(fe.win, (fe, A.A))
        self.assertEqual((fe.win_rule, fe.win_sub), ("RZ1", "DUEL"))
        sc = Scene(C)
        en = TW.mech(7, (50, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",), hp=600)
        A = sc.run([en], weapons=(GUN, LASER), aim=(0.0, 0.0), t=100.0)
        self.assertFalse(sc.a.evaluate_fight(sc.w, A, "X1", None, {}).win)     # inside LASER_HOLD_MIN
        self.assertTrue(sc.a.evaluate_fight(sc.w, A, "X2", 7, {"x2_since": 99.0}).win)
        sc = Scene(C)
        en = TW.mech(7, (120, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",), hp=1300)
        A = sc.run([en], weapons=(GUN, LASER), aim=(0.0, 0.0), t=20.0)
        if A.A < C.EARLY_FIGHT_A_MIN:
            self.assertFalse(sc.a.evaluate_fight(sc.w, A, "X1", None, {}).win)   # R1.2/R0.5 loot only

    def test_cornered(self):
        sc = Scene(self.C, TW.FakeMap(wall=100))
        lt = TW.mech(7, (30, 0, 0), aim=(math.pi, 0), weapons=("gun", "lightning"), vel=(-15, 0, 0))
        A = sc.run([lt], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X3", None, {"x3_since": 99.0})
        self.assertTrue(fe.cornered)
        self.assertTrue(fe.win)
        self.assertEqual(fe.win_rule, "CORNERED")


class EarlyDeathFixTest(unittest.TestCase):
    """E1 (engage ratio) and E2 (cornered vs ranged threats), from the minisim early-death analysis."""

    def setUp(self):
        self.C = calibrated()

    def test_e1_entry_ratio_ignores_transient_not_aiming(self):
        sc = Scene(self.C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",))     # looking north, not at us
        A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        self.assertGreater(A.A, 1.5)                        # live A doubled by NOT_AIMING_DPS_FRAC
        self.assertAlmostEqual(A.A_entry, 1.0, places=6)    # symmetric gun duel when it turns on us
        fe = sc.a.evaluate_fight(sc.w, A, "X1", None, {})
        self.assertFalse(fe.win, (fe, A.A, A.A_entry))      # RG1 needs ENGAGE_A_MIN + PB_G_A_EXTRA

    def test_e1_busy_mech_keeps_discount(self):
        sc = Scene(self.C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",))
        sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        sc.w.tracks[7].fighting_with = 9                    # busy with another mech: a third-party chance
        A = sc.a.update(sc.w, None, Clock(time.perf_counter(), 1e6, 1e6))
        self.assertAlmostEqual(A.A_entry, A.A, places=6)

    def test_e2_gun_duel_in_open_is_not_cornered(self):
        sc = Scene(self.C)                                  # flat open map, no cover within reach
        en = TW.mech(7, (150, 0, 0), aim=(math.pi, 0), weapons=("gun",))
        A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X3", None, {"x3_since": 99.0})
        self.assertFalse(fe.cornered, (fe.t_close, fe.cover_eta))
        self.assertLess(fe.cover_eta, fe.t_close)           # walk out of its reach before half our ttk

    def test_e2_laser_duelist_in_open_is_cornered_unless_cover(self):
        sc = Scene(self.C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi, 0), weapons=("gun", "laser"))
        A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X3", None, {"x3_since": 99.0})
        self.assertTrue(fe.cornered, (fe.t_close, fe.cover_eta))
        sc2 = Scene(self.C)
        A = sc2.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        fe = sc2.a.evaluate_fight(sc2.w, A, "X3", None, {"x3_since": 99.0, "cover_eta": 1.5})
        self.assertFalse(fe.cornered, (fe.t_close, fe.cover_eta))


class EngageModelToggleTest(unittest.TestCase):
    """ENGAGE_MODEL = "live" switches E1 and E2 off together (the A/B the review asked for)."""

    def setUp(self):
        self.C = calibrated()
        self.C.ENGAGE_MODEL = "live"

    def test_live_entry_ratio_is_the_live_ratio(self):
        sc = Scene(self.C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",))     # not aiming at us
        A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        self.assertGreater(A.A, 1.5)
        self.assertAlmostEqual(A.A_entry, A.A, places=6)                         # E1 off: no engage ratio
        self.assertAlmostEqual(A.per_track[7].dps_to_us_entry, A.per_track[7].dps_to_us, places=6)

    def _looking_away_attacker(self, model, extra=()):
        """Mech 7 hit us a second ago and now looks away. With NOT_AIMING_DPS_FRAC = 0 its live DPS on us is 0,
        so it is in H (SELF_DEFENSE_S) with its raw DPS, while under "live" its entry DPS is the live one (0)."""
        C = calibrated()
        C.ENGAGE_MODEL = model
        C.NOT_AIMING_DPS_FRAC = 0.0
        sc = Scene(C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",))
        hit = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 7}]
        sc.run([en, *extra], weapons=(GUN, None), aim=(0.0, 0.0), t=99.0, events_=hit)
        return sc, sc.run([en, *extra], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)

    def test_live_entry_ratio_equals_live_ratio_for_recent_attacker_looking_away(self):
        sc, A = self._looking_away_attacker("live")
        self.assertEqual(A.target_id, 7)
        self.assertEqual(A.per_track[7].dps_to_us, 0.0)
        self.assertGreater(sc.a.hostile_dps[7], 0.0)
        self.assertEqual(A.per_track[7].dps_to_us_entry, 0.0)
        self.assertLess(A.A, math.inf)
        self.assertEqual(A.A_entry, A.A)

    def test_live_a_vs_entry_is_the_live_ratio(self):
        # H = {7, 8}: 7 counts with its raw DPS in the live ratio and with its entry DPS (0 under "live") in the
        # entry ratio, so a_vs(entry=True) only matches a_vs() when "live" turns the entry branch off.
        full = TW.mech(8, (100, 10, 0), aim=(math.pi, 0), weapons=("gun",), hp=900)
        sc, A = self._looking_away_attacker("live", extra=(full,))
        self.assertIn(7, A.hostiles)
        self.assertEqual(sc.a.a_vs(sc.w, 8, [7, 8], entry=True), sc.a.a_vs(sc.w, 8, [7, 8]))
        self.assertEqual(sc.a.a_vs(sc.w, 7, entry=True), sc.a.a_vs(sc.w, 7))

    def test_live_early_fight_veto_uses_a_of_the_target_for_any_mech(self):
        # 7 is the target (weak, looking away: A = 1.88); 8 is not (full duel, alone it is 1.56 < 1.6).
        # Before E1 the veto read A.A for either, so the fight with 8 was allowed.
        def fight(model):
            C = calibrated()
            C.ENGAGE_MODEL = model
            sc = Scene(C)
            weak = TW.mech(7, (200, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",), hp=200)
            full = TW.mech(8, (100, 10, 0), aim=(math.pi, 0), weapons=("gun",), hp=900)
            sc.run([weak, full], weapons=(GUN, None), aim=(0.0, 0.0), t=49.0)
            A = sc.run([weak, full], weapons=(GUN, None), aim=(0.0, 0.0), t=50.0)
            fe = sc.a.evaluate_fight(sc.w, A, "X2", 8, {"x2_since": 49.5})
            return C, sc, A, fe

        C, sc, A, fe = fight("live")
        self.assertEqual(A.target_id, 7)
        self.assertGreaterEqual(A.A, C.EARLY_FIGHT_A_MIN)
        self.assertLess(sc.a.a_vs(sc.w, 8), C.EARLY_FIGHT_A_MIN)
        self.assertTrue(fe.win, (fe.win_rule, fe.lose_clauses))
        self.assertEqual(fe.win_rule, "RG1")
        _, _, A, fe = fight("entry")                  # control: the single-track ratio vetoes it
        self.assertFalse(fe.win)

    def test_live_cornered_cover_eta_uses_assessor_walk_speed(self):
        # a gun mech 300 m out closing at 14 m/s (70 m to its reach, 5 s) and a wall 20 m away: the cover
        # fallback is wall / walk. Pre-E2 it used the assessor's walk speed, not the live stat.
        def fight(model):
            C = calibrated()
            C.ENGAGE_MODEL = model
            sc = Scene(C, TW.FakeMap(wall=20))
            en = TW.mech(7, (300, 0, 0), aim=(math.pi, 0), weapons=("gun",), vel=(-14, 0, 0))
            A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
            sc.w.me.stats["walk_speed"] = 1.0                                    # live stat != sc.a.walk
            return sc, sc.a.evaluate_fight(sc.w, A, "X3", 7, {"x3_since": 99.0})

        sc, fe = fight("live")
        self.assertAlmostEqual(fe.t_close, 5.0, places=6)
        self.assertGreater(20.0 / 1.0, fe.t_close)
        self.assertLess(20.0 / sc.a.walk, fe.t_close)
        self.assertFalse(fe.cornered)                 # cover in 1.7 s beats the 5 s: a way out, not cornered
        self.assertIsNone(fe.cover_eta)
        sc, fe = fight("entry")                       # control: the live stat (1 m/s) puts cover 20 s away
        self.assertAlmostEqual(fe.cover_eta, 20.0, places=6)
        self.assertTrue(fe.cornered)
    def test_default_entry_model_keeps_engage_ratio(self):
        C = calibrated()
        self.assertEqual(C.ENGAGE_MODEL, "entry")
        sc = Scene(C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi / 2, 0), weapons=("gun",))
        A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        self.assertGreater(A.A, 1.5)
        self.assertAlmostEqual(A.A_entry, 1.0, places=6)

    def test_live_ranged_threat_in_reach_is_cornered_as_before(self):
        sc = Scene(self.C)
        en = TW.mech(7, (150, 0, 0), aim=(math.pi, 0), weapons=("gun",))
        A = sc.run([en], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        fe = sc.a.evaluate_fight(sc.w, A, "X3", None, {"x3_since": 99.0, "cover_eta": 1.0})
        self.assertTrue(fe.cornered)                 # time to close is 0 inside its reach (pre-E2 rule)
        self.assertEqual(fe.t_close, 0.0)
        self.assertIsNone(fe.cover_eta)


class VisibleThreatFirstToggleTest(unittest.TestCase):
    """VISIBLE_THREAT_FIRST = False switches E4 off in assess: the main threat is the H mech with the highest
    DPS on us (nearest on ties), seen or not, as before E4. True (shipped) puts a mech we can see first."""

    def _seen_gun_and_unseen_laser(self, value):
        """Mech 7 stands in view 150 m east with its gun on us. Mech 9 is never seen: its laser hit from the
        north makes it a damage-inferred estimate with the higher DPS on us. Same input for both values."""
        C = calibrated()
        self.assertIs(C.VISIBLE_THREAT_FIRST, True)                  # the shipped value
        C.VISIBLE_THREAT_FIRST = value
        sc = Scene(C)
        seen = TW.mech(7, (150, 0, 0), aim=(math.pi, 0), weapons=("gun",))
        hit = [{"type": "damaged", "amount": 315, "weapon": "laser", "bearing": math.pi / 2, "by": 9}]
        sc.run([seen], weapons=(GUN, None), aim=(0.0, 0.0), t=99.0, events_=hit)
        A = sc.run([seen], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        self.assertEqual(A.hostiles, [7, 9])
        self.assertFalse(sc.w.tracks[7].est)
        self.assertTrue(sc.w.tracks[9].est and not sc.w.tracks[9].visible)
        self.assertGreater(sc.a.hostile_dps[9], sc.a.hostile_dps[7])
        self.assertEqual(A.target_id, 7)                             # the toggle does not touch the target
        return sc, A

    def test_main_threat_for_both_values_on_the_same_input(self):
        _, A = self._seen_gun_and_unseen_laser(True)
        self.assertEqual(A.main_threat_id, 7)                        # E4: the mech we can see
        sc, A = self._seen_gun_and_unseen_laser(False)
        self.assertEqual(A.main_threat_id, 9)                        # before E4: the highest DPS on us
        self.assertEqual(A.main_threat_id,
                         max(A.hostiles, key=lambda i: (sc.a.hostile_dps[i], -A.per_track[i].d)))

    def test_false_changes_nothing_when_every_hostile_is_seen(self):
        def main_threat(value):
            C = calibrated()
            C.VISIBLE_THREAT_FIRST = value
            sc = Scene(C)
            near = TW.mech(7, (150, 0, 0), aim=(math.pi, 0), weapons=("gun",))
            far = TW.mech(8, (0, 200, 0), aim=(-math.pi / 2, 0), weapons=("gun",))
            A = sc.run([near, far], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
            self.assertEqual(A.hostiles, [7, 8])
            return A.main_threat_id

        self.assertEqual(main_threat(True), 7)
        self.assertEqual(main_threat(False), 7)


class ArmorCountedOnceTest(unittest.TestCase):
    """Review bug 2: own armor used to enter every time-to-kill-us twice (HP / (1 - armor), and incoming
    DPS x (1 - armor)), which read our odds 1 / (1 - armor) too high."""

    def setUp(self):
        self.C = calibrated()

    def _duel(self, armor):
        sc = Scene(self.C)
        o = TW.base_obs(weapons=[GUN, None], aim=(0.0, 0.0), t=100.0)
        o["self"]["stats"]["armor"] = armor
        o["mechs"] = [TW.mech(7, (150, 0, 0), aim=(math.pi, 0), weapons=("gun",))]
        config.update_live(self.C, o["self"])
        clk = Clock(time.perf_counter(), 1e6, 1e6)
        sc.w.update(o, clk)
        return sc, sc.a.update(sc.w, None, clk)

    def test_symmetric_gun_duel_scales_once(self):
        _, a0 = self._duel(0.0)
        sc, a2 = self._duel(0.2)
        self.assertAlmostEqual(a0.A, 1.0, places=6)
        self.assertAlmostEqual(a2.A, 1.0 / 0.8, places=6)            # was 1 / 0.8**2 = 1.5625
        self.assertAlmostEqual(a2.A_entry, 1.0 / 0.8, places=6)
        t0, t2 = a0.per_track[7], a2.per_track[7]
        self.assertAlmostEqual(t2.dps_to_us, t0.dps_to_us * 0.8, places=6)
        self.assertAlmostEqual(t2.ttk_them_on_us, t0.ttk_them_on_us / 0.8, places=6)
        self.assertAlmostEqual(sc.a.a_vs(sc.w, 7), 1.0 / 0.8, places=6)
        self.assertAlmostEqual(sc.a.a_vs(sc.w, 7, entry=True), 1.0 / 0.8, places=6)

    def test_rush_ratio_gets_raw_hp(self):
        sc, A = self._duel(0.2)
        seen = {}
        real = AS.a_rush

        def spy(C, d, v, hp_us, *a, **k):
            seen["hp_us"] = hp_us
            return real(C, d, v, hp_us, *a, **k)

        AS.a_rush = spy
        try:
            sc.a.a_rush_for(sc.w, A, 7)
        finally:
            AS.a_rush = real
        self.assertAlmostEqual(seen["hp_us"], 1400.0)               # not 1400 / 0.8

    def test_rl4_keeps_armor_adjusted_hp(self):
        sc, _ = self._duel(0.2)
        self.assertAlmostEqual(AS.hp_eff_self(sc.w.me), 1400 / 0.8)
        self.assertAlmostEqual(AS.hp_self(sc.w.me), 1400.0)


class ThirdPartyRuleTest(unittest.TestCase):
    """LOSE (c) (R5.2) and its D12 toggle THIRD_PARTY_RULE: "gp" ends the fight for any newly visible mech,
    "summed" only when A with that mech's DPS added falls under HOLD_A_MIN."""

    def _fight(self, rule, third):
        C = calibrated()
        C.THIRD_PARTY_RULE = rule
        sc = Scene(C)
        tgt = TW.mech(7, (100, 0, 0), aim=(math.pi, 0), weapons=("gun",), hp=700)
        sc.run([tgt], weapons=(GUN, None), aim=(0.0, 0.0), t=99.0)
        A = sc.run([tgt, third], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)   # the third appears now
        return sc.a.evaluate_fight(sc.w, A, "X2", 7, {"x2_since": 99.5})

    def test_far_third_mech_not_aiming(self):
        far = TW.mech(9, (160, 100, 0), aim=(math.pi / 2, 0), weapons=("gun",))  # 189 m off, looking away
        self.assertIn("c", self._fight("gp", far).lose_clauses)
        self.assertNotIn("c", self._fight("summed", far).lose_clauses)

    def test_close_third_mech_aiming_still_ends_the_fight(self):
        near = TW.mech(9, (50, 30, 0), aim=(math.atan2(-30, -50), 0), weapons=("gun", "laser"))   # 58 m, on us
        self.assertIn("c", self._fight("gp", near).lose_clauses)
        fe = self._fight("summed", near)
        self.assertIn("c", fe.lose_clauses)
        self.assertEqual(fe.lose_tracks.get("c"), 9)

    def test_close_third_mech_looking_away_counts_as_aiming(self):
        away = TW.mech(9, (50, 30, 0), aim=(math.atan2(30, 50), 0), weapons=("gun", "laser"))   # back to us
        self.assertIn("c", self._fight("summed", away).lose_clauses)

    def test_weak_arrivals_add_up(self):
        C = calibrated()
        C.THIRD_PARTY_RULE = "summed"
        sc = Scene(C)
        tgt = TW.mech(7, (100, 0, 0), aim=(math.pi, 0), weapons=("gun",), hp=700)
        sc.run([tgt], weapons=(GUN, None), aim=(0.0, 0.0), t=99.0)
        crowd = [TW.mech(20 + i, (150, -60 + 30 * i, 0), aim=(math.pi / 2, 0), weapons=("gun",)) for i in range(5)]
        one = sc.run([tgt, crowd[0]], weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        self.assertNotIn("c", sc.a.evaluate_fight(sc.w, one, "X2", 7, {"x2_since": 99.5}).lose_clauses)
        sc2 = Scene(C)
        sc2.run([tgt], weapons=(GUN, None), aim=(0.0, 0.0), t=99.0)
        many = sc2.run([tgt] + crowd, weapons=(GUN, None), aim=(0.0, 0.0), t=100.0)
        self.assertIn("c", sc2.a.evaluate_fight(sc2.w, many, "X2", 7, {"x2_since": 99.5}).lose_clauses)


class RobustnessTest(unittest.TestCase):
    def test_update_falls_back_and_counts(self):
        C = calibrated()
        sc = Scene(C)
        A0 = sc.run([TW.mech(7, (100, 0, 0), aim=(math.pi, 0))], aim=(0.0, 0.0))

        class BadZone:
            next_center, next_radius = (0, 0), 380

        sc.w.zone = BadZone()          # broken input: no center_at / radius_at
        A1 = sc.a.update(sc.w, None, None)
        self.assertIs(A1, A0)
        self.assertEqual(sc.a.take_exceptions(), 1)
        self.assertEqual(sc.a.exceptions, 0)

    def test_world_ignores_garbage(self):
        C = TW.bound_config()
        w = W.WorldModel(C, TW.FakeMap(), 3)
        w.update({"tick": 0, "t": 0, "self": {"pos": "x"}, "mechs": [None, {"id": "a"}, {"kind": "mech"}],
                  "sounds": [{"kind": "gun"}], "events": [{"type": "damaged", "by": 4}], "zone": None}, None)
        self.assertEqual(w.take_exceptions(), 0)
        self.assertIn(4, w.tracks)
        w.update("not a dict", None)
        self.assertEqual(w.take_exceptions(), 1)


# ====================================================================== CPU budget (3.3)
@unittest.skipUnless(TW.HAVE_ARENA, "arena missing")
class WorstCaseTimingTest(unittest.TestCase):
    def test_assess_update_p99(self):
        C = calibrated()
        md = TW.real_map(C)
        w = W.WorldModel(C, md, 3)
        a = AS.Assessor(C, md)
        rng = random.Random(11)
        wms, ams, ems = [], [], []
        for k in range(300):
            o = TW.worst_case_obs(rng, 2 * k, md)
            o["self"]["weapons"] = [GUN, LIGHTNING]
            config.update_live(C, o["self"])
            md.new_tick()
            clk = Clock(time.perf_counter(), C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS)
            t0 = time.perf_counter()
            w.update(o, clk)
            t1 = time.perf_counter()
            A = a.update(w, None, clk)
            t2 = time.perf_counter()
            a.evaluate_fight(w, A, "X2" if k % 2 else "X1", None, {"x2_since": 0.0})
            t3 = time.perf_counter()
            wms.append((t1 - t0) * 1000)
            ams.append((t2 - t1) * 1000)
            ems.append((t3 - t2) * 1000)
        self.assertEqual(w.exceptions + a.exceptions, 0, w.last_error + a.last_error)
        tot = [x + y + z for x, y, z in zip(wms, ams, ems)]
        p = TW.pct
        print(f"\n  worst-case (16 mechs, 12 NPCs, 20 bullets, 6 new shells, 40 sounds):"
              f"\n    world.update    p50 {p(wms, .5):.3f}  p99 {p(wms[5:], .99):.3f}  max {max(wms[5:]):.3f} ms"
              f"\n    assess.update   p50 {p(ams, .5):.3f}  p99 {p(ams[5:], .99):.3f}  max {max(ams[5:]):.3f} ms"
              f"\n    evaluate_fight  p50 {p(ems, .5):.3f}  p99 {p(ems[5:], .99):.3f}  max {max(ems[5:]):.3f} ms"
              f"\n    total           p99 {p(tot[5:], .99):.3f} ms")
        self.assertLess(p(wms[5:], 0.99), 4.0)
        self.assertLess(p(ams[5:], 0.99) + p(ems[5:], 0.99), 4.0)


if __name__ == "__main__":
    unittest.main()
