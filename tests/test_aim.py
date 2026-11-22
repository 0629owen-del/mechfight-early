"""Aim tests (plan 8.7, 6.6): gun lead, mortar solver, lightning gate, laser release, scanner.

    cd bot && python3 -m unittest tests.test_aim -v
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
import unittest
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

import aim  # noqa: E402
import config  # noqa: E402
from contracts import CrateMem, Ghost, SelfState, TickInfo, Track, WeaponState  # noqa: E402

STREAM = os.path.join(os.path.dirname(REPO), "mechfight-kit", "check", "stream.jsonl.gz")
HAVE_KIT = os.path.exists(STREAM)


def hello_config():
    with gzip.open(STREAM, "rt") as f:
        return json.loads(f.readline())["config"]


class FlatMap:
    """Stub mapdata: flat ground at z=0 plus optional boxes (x0, x1, top)."""

    def __init__(self, walls=(), los=True):
        self.walls = walls
        self._los = los

    def surface_h(self, x, y):
        for x0, x1, top in self.walls:
            if x0 <= x <= x1:
                return top
        return 0.0

    def los(self, a, b):
        return self._los

    def sight5_dir(self, x, y, k):
        return 100.0 if k == 2 else 50.0


def make_me(C, aim_=(0.0, 0.0), weapons=None, **kw):
    me = SelfState(id=1, pos=(0.0, 0.0, 0.0), aim=aim_, hp=1400, max_hp=1400, energy=100, max_energy=100)
    me.stats = {"muzzle_height": 7.0, "eye_height": 8.0, "min_pitch": -0.7, "max_pitch": 1.3}
    me.muzzle_z, me.eye_z, me.radius, me.s = 7.0, 8.0, 2.5, 1.0
    me.weapons = weapons or [None, None]
    for k, v in kw.items():
        setattr(me, k, v)
    return me


def gun_ws(C):
    return WeaponState(kind="gun", ammo=12, stats=dict(C.cfg["weapons"]["gun"]))


def mortar_ws(C):
    return WeaponState(kind="mortar", stats=dict(C.cfg["weapons"]["mortar"]))


def laser_ws(C, charge=0.0, cooldown=0.0):
    return WeaponState(kind="laser", charge=charge, cooldown=cooldown, stats=dict(C.cfg["weapons"]["laser"]))


def lightning_ws(C):
    return WeaponState(kind="lightning", stats=dict(C.cfg["weapons"]["lightning"]))


def sim_gun(C, me, tr, yaw, pitch, fire_delay):
    """Fire one bullet along (yaw, pitch) `fire_delay` s after the observation; return the closest
    approach of the bullet to the target's center."""
    speed = C.cfg["weapons"]["gun"]["speed"]
    off = me.radius + C.SHOT_OFFSET_PAD
    d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    o = (me.pos[0] + off * d[0], me.pos[1] + off * d[1], me.muzzle_z + off * d[2])
    best = 1e9
    t = 0.0
    while t < 3.0:
        b = (o[0] + d[0] * speed * t, o[1] + d[1] * speed * t, o[2] + d[2] * speed * t)
        T = fire_delay + t
        c = (tr.pos[0] + tr.vel[0] * T, tr.pos[1] + tr.vel[1] * T, tr.pos[2] + 4.5 + tr.vel[2] * T)
        best = min(best, math.dist(b, c))
        t += 0.0005
    return best


@unittest.skipUnless(HAVE_KIT, "recorded hello not available")
class TestGun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = config.bind(hello_config())

    def test_lead_hits_constant_velocity_targets(self):
        C = self.C
        gun = gun_ws(C)
        cases = [((120, 40, 0), (8, -6, 0)), ((200, 0, 0), (0, 10, 0)), ((-60, 90, 3), (-9, -9, 0)),
                 ((30, 10, 0), (15, 0, 0)), ((150, -80, 0), (-5, 5, 0))]
        for pos, vel in cases:
            me = make_me(C)
            tr = Track(id=5, kind="mech", pos=pos, vel=vel)
            yaw, pitch, tof = aim.gun_lead(me, tr, gun, C)
            miss = sim_gun(C, me, tr, yaw, pitch, C.AIM_LATENCY_S)
            self.assertLess(miss, 0.5, (pos, vel, miss))
            self.assertGreater(tof, 0)

    def test_lead_beats_naive_aim(self):
        C = self.C
        me = make_me(C)
        tr = Track(id=5, kind="mech", pos=(150.0, 0.0, 0.0), vel=(0.0, 12.0, 0.0))
        yaw, pitch, _ = aim.gun_lead(me, tr, gun_ws(C), C)
        self.assertGreater(yaw, 0.03)           # leads north
        naive = sim_gun(C, me, tr, 0.0, 0.0, C.AIM_LATENCY_S)
        self.assertGreater(naive, 3.0)

    def test_gate_tolerance_scales_with_size(self):
        C = self.C
        gun = gun_ws(C)
        d = 100.0
        small = Track(id=5, kind="mech", pos=(d, 0.0, 0.0), cubes=0)
        big = Track(id=6, kind="mech", pos=(d, 0.0, 0.0), cubes=10)
        me = make_me(C)
        tgt = aim.gun_lead(me, small, gun, C)[:2]
        # error between the small and big tolerance
        tol_small = math.atan(C.GUN_FIRE_TOL * (2.5 + 0.3) / d)
        err = (tol_small + math.atan(C.GUN_FIRE_TOL * (2.5 * 1.5 + 0.3) / d)) / 2
        me2 = make_me(C, aim_=(tgt[0] + err, tgt[1]))
        self.assertFalse(aim.gun_gate(me2, small, gun, tgt, C))
        tgt_b = aim.gun_lead(me, big, gun, C)[:2]
        me3 = make_me(C, aim_=(tgt_b[0] + err, tgt_b[1]))
        self.assertTrue(aim.gun_gate(me3, big, gun, tgt_b, C))
        self.assertTrue(aim.gun_gate(make_me(C, aim_=tgt), small, gun, tgt, C))

    def test_gate_range_and_ammo(self):
        C = self.C
        gun = gun_ws(C)
        far = Track(id=5, kind="mech", pos=(250.0, 0.0, 0.0))
        me = make_me(C)
        tgt = aim.gun_lead(me, far, gun, C)[:2]
        self.assertFalse(aim.gun_gate(make_me(C, aim_=tgt), far, gun, tgt, C))
        near = Track(id=5, kind="mech", pos=(100.0, 0.0, 0.0))
        tgt = aim.gun_lead(me, near, gun, C)[:2]
        gun.ammo = 0
        self.assertFalse(aim.gun_gate(make_me(C, aim_=tgt), near, gun, tgt, C))


def sim_shell(C, me, yaw, pitch, z_target):
    """Independent integration of a mortar shell; returns (x, y, t) when it descends through z_target."""
    v, g = C.cfg["weapons"]["mortar"]["speed"], C.cfg["weapons"]["mortar"]["gravity"]
    off = me.radius + C.SHOT_OFFSET_PAD
    d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    x = me.pos[0] + off * d[0]
    y = me.pos[1] + off * d[1]
    z = me.pos[2] + 9.0 * me.s + C.MORTAR_TUBE_PAD + off * d[2]
    vx, vy, vz = v * d[0], v * d[1], v * d[2]
    dt = 0.001
    t = 0.0
    while t < 12:
        vz2 = vz - g * dt
        nx, ny, nz = x + vx * dt, y + vy * dt, z + vz * dt - 0.5 * g * dt * dt
        if vz2 < 0 and nz <= z_target:
            return nx, ny, t
        x, y, z, vz, t = nx, ny, nz, vz2, t + dt
    return None


@unittest.skipUnless(HAVE_KIT, "recorded hello not available")
class TestMortar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = config.bind(hello_config())

    def test_lands_within_1m_on_flat_ground(self):
        C = self.C
        mort = mortar_ws(C)
        me = make_me(C)
        for D in (25.0, 40.0, 70.0, 110.0, 160.0, 210.0):
            for ang in (0.0, 1.1, -2.4):
                tgt = (D * math.cos(ang), D * math.sin(ang), 0.0)
                sol = aim.mortar_solution(me, FlatMap(), tgt, (0.0, 0.0, 0.0), mort, C)
                self.assertIsNotNone(sol, (D, ang))
                yaw, pitch, impact = sol
                lo, hi = me.stats["min_pitch"], me.stats["max_pitch"]
                self.assertTrue(lo <= pitch <= hi)
                x, y, _ = sim_shell(C, me, yaw, pitch, 0.0)
                self.assertLess(math.hypot(x - tgt[0], y - tgt[1]), 1.0, (D, ang))
                self.assertLess(math.hypot(impact[0] - tgt[0], impact[1] - tgt[1]), 1.0)

    def test_moving_target_prediction(self):
        C = self.C
        mort = mortar_ws(C)
        me = make_me(C)
        tgt, vel = (120.0, 30.0, 0.0), (-6.0, 8.0, 0.0)
        sol = aim.mortar_solution(me, FlatMap(), tgt, vel, mort, C)
        self.assertIsNotNone(sol)
        x, y, t = sim_shell(C, me, sol[0], sol[1], 0.0)
        T = t + C.AIM_LATENCY_S
        self.assertLess(math.hypot(x - (tgt[0] + vel[0] * T), y - (tgt[1] + vel[1] * T)), 1.5)

    def test_height_difference(self):
        C = self.C
        me = make_me(C)
        tgt = (80.0, 0.0, 12.0)   # target on a roof 12 m up
        sol = aim.mortar_solution(me, FlatMap(), tgt, (0.0, 0.0, 0.0), mortar_ws(C), C)
        self.assertIsNotNone(sol)
        x, y, _ = sim_shell(C, me, sol[0], sol[1], 12.0)
        self.assertLess(math.hypot(x - 80.0, y), 1.0)

    def test_out_of_range_is_none(self):
        C = self.C
        me = make_me(C)
        self.assertIsNone(aim.mortar_solution(me, FlatMap(), (260.0, 0.0, 0.0), (0, 0, 0), mortar_ws(C), C))

    def test_low_arc_falls_to_high_arc_and_high_is_clamped(self):
        C = self.C
        me = make_me(C)
        wall = FlatMap(walls=[(60.0, 64.0, 40.0)])
        # 150 m: low arc (~20 deg) is blocked by the wall, the high arc (~70 deg) is inside max_pitch
        sol = aim.mortar_solution(me, wall, (150.0, 0.0, 0.0), (0, 0, 0), mortar_ws(C), C)
        self.assertIsNotNone(sol)
        self.assertGreater(sol[1], 0.9)
        self.assertLessEqual(sol[1], me.stats["max_pitch"])
        x, y, _ = sim_shell(C, me, sol[0], sol[1], 0.0)
        self.assertLess(math.hypot(x - 150.0, y), 1.0)
        # 30 m with a wall in front: the high arc would need ~83 deg > max_pitch: rejected, not clamped
        wall2 = FlatMap(walls=[(10.0, 20.0, 50.0)])
        self.assertIsNone(aim.mortar_solution(me, wall2, (30.0, 0.0, 0.0), (0, 0, 0), mortar_ws(C), C))
        # same wall, no obstruction: low arc is fine
        self.assertIsNotNone(aim.mortar_solution(me, FlatMap(), (30.0, 0.0, 0.0), (0, 0, 0), mortar_ws(C), C))

    def test_gate_splash_safe_and_aim_tolerance(self):
        C = self.C
        mort = mortar_ws(C)
        me0 = make_me(C)
        # target inside MORTAR_SELF_SAFE_M: never fires even with a perfect aim
        near = (C.MORTAR_SELF_SAFE_M - 3.0, 0.0, 0.0)
        sol = aim.mortar_solution(me0, FlatMap(), near, (0, 0, 0), mort, C)
        if sol is not None:
            me = make_me(C, aim_=(sol[0], sol[1]))
            self.assertFalse(aim.mortar_gate(me, sol, sol[2], mort, C))
        # target at 90 m: perfect aim fires, a pitch error of 0.04 rad (~6 m at this range) does not
        tgt = (90.0, 0.0, 0.0)
        sol = aim.mortar_solution(me0, FlatMap(), tgt, (0, 0, 0), mort, C)
        self.assertTrue(aim.mortar_gate(make_me(C, aim_=(sol[0], sol[1])), sol, sol[2], mort, C))
        self.assertFalse(aim.mortar_gate(make_me(C, aim_=(sol[0], sol[1] + 0.04)), sol, sol[2], mort, C))
        self.assertFalse(aim.mortar_gate(make_me(C, aim_=(sol[0] + 0.1, sol[1])), sol, sol[2], mort, C))
        self.assertFalse(aim.mortar_gate(me0, None, tgt, mort, C))

    def test_shell_predictor(self):
        C = self.C
        me = make_me(C)
        sol = aim.mortar_solution(me, FlatMap(), (100.0, 0.0, 0.0), (0, 0, 0), mortar_ws(C), C)
        v = C.cfg["weapons"]["mortar"]["speed"]
        d = (math.cos(sol[1]) * math.cos(sol[0]), math.cos(sol[1]) * math.sin(sol[0]), math.sin(sol[1]))
        off = 3.0
        pos = (off * d[0], off * d[1], 9.5 + off * d[2])
        sp = aim.predict_shell(FlatMap(), 7, pos, (v * d[0], v * d[1], v * d[2]), 10.0, C)
        self.assertIsNotNone(sp)
        self.assertLess(math.hypot(sp.impact[0] - 100.0, sp.impact[1]), 1.0)
        self.assertGreater(sp.t_impact, 10.0)
        self.assertAlmostEqual(sp.splash_r, 7.0 * C.MORTAR_ENEMY_SPLASH_MULT)


def lw_world(C, me, tracks=(), crates=(), t=10.0):
    return SimpleNamespace(me=me, tick=TickInfo(tick=1, t=t), tracks={tr.id: tr for tr in tracks},
                           crates={c.id: c for c in crates}, ghosts={})


def at(d, deg, z=0.0):
    a = math.radians(deg)
    return (d * math.cos(a), d * math.sin(a), z)


@unittest.skipUnless(HAVE_KIT, "recorded hello not available")
class TestLightning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = config.bind(hello_config())

    def me(self):
        return make_me(self.C, weapons=[gun_ws(self.C), lightning_ws(self.C)])

    def mech(self, deg, d=30.0, id_=5, kind="mech"):
        return Track(id=id_, kind=kind, pos=at(d, deg), visible=True, in_obs=True)

    def test_inside_margin_fires_outside_does_not(self):
        C, mp = self.C, FlatMap()
        # cone_deg 30: narrow reading half-width 15 deg, margin 0.8 -> 12 deg
        w = lw_world(C, self.me(), [self.mech(10)])
        self.assertEqual(aim.lightning_gate(w, mp, 1, C), (True, 5))
        w = lw_world(C, self.me(), [self.mech(13)])
        self.assertEqual(aim.lightning_gate(w, mp, 1, C), (False, None))
        # inside the wide reading (30 deg) only: not eligible
        w = lw_world(C, self.me(), [self.mech(25)])
        self.assertEqual(aim.lightning_gate(w, mp, 1, C), (False, None))

    def test_range_energy_los_slot(self):
        C = self.C
        w = lw_world(C, self.me(), [self.mech(0, d=60.0)])
        self.assertEqual(aim.lightning_gate(w, FlatMap(), 1, C), (False, None))        # out of range
        m = self.me()
        m.energy = 2.0
        self.assertEqual(aim.lightning_gate(lw_world(C, m, [self.mech(0)]), FlatMap(), 1, C), (False, None))
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [self.mech(0)]), FlatMap(los=False), 1, C),
                         (False, None))
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [self.mech(0)]), FlatMap(), 0, C),
                         (False, None))                                                 # slot 0 is the gun
        # 3D range: 44 m horizontal but 20 m above
        hi = Track(id=5, kind="mech", pos=(44.0, 0.0, 30.0), visible=True, in_obs=True)
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [hi]), FlatMap(), 1, C), (False, None))

    def test_npc_nearest_blocks_in_wide_cone(self):
        C, mp = self.C, FlatMap()
        mech = self.mech(5, d=35.0)
        npc = self.mech(25, d=20.0, id_=104, kind="npc")           # outside narrow, inside wide
        w = lw_world(C, self.me(), [mech, npc])
        self.assertEqual(aim.lightning_gate(w, mp, 1, C), (False, None))
        self.assertEqual(aim.lightning_gate(w, mp, 1, C, allow_npc=True), (True, 5))
        # NPC outside the wide cone does not block
        npc2 = self.mech(45, d=20.0, id_=104, kind="npc")
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [mech, npc2]), mp, 1, C), (True, 5))
        # NPC behind the mech does not block
        npc3 = self.mech(5, d=40.0, id_=105, kind="npc")
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [mech, npc3]), mp, 1, C), (True, 5))

    def test_crate_nearest_needs_chain(self):
        C, mp = self.C, FlatMap()
        mech = self.mech(0, d=35.0)
        near_crate = CrateMem(id=1007, pos=(30.0, 2.0, 0.0), hp=300, last_seen=10.0)     # within 14 m of mech
        far_crate = CrateMem(id=1008, pos=at(12.0, 20.0), hp=300, last_seen=10.0)         # far from mech
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [mech], [near_crate]), mp, 1, C), (True, 5))
        self.assertEqual(aim.lightning_gate(lw_world(C, self.me(), [mech], [far_crate]), mp, 1, C), (False, None))

    def test_allow_npc_targets_npc_when_no_mech(self):
        C, mp = self.C, FlatMap()
        npc = self.mech(3, d=25.0, id_=104, kind="npc")
        w = lw_world(C, self.me(), [npc])
        self.assertEqual(aim.lightning_gate(w, mp, 1, C), (False, None))
        self.assertEqual(aim.lightning_gate(w, mp, 1, C, allow_npc=True), (True, 104))


@unittest.skipUnless(HAVE_KIT, "recorded hello not available")
class TestLaser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = config.bind(hello_config())

    def setup(self, charge, d=100.0, aim_err=0.0, cooldown=0.0, cubes=0):
        C = self.C
        tr = Track(id=5, kind="mech", pos=(d, 0.0, 0.0), cubes=cubes, visible=True)
        laser = laser_ws(C, charge=charge, cooldown=cooldown)
        me = make_me(C, weapons=[laser, None])
        yaw, pitch = aim.laser_point(me, tr, C)
        me.aim = (yaw + aim_err, pitch)
        return me, tr, laser

    def test_start_conditions(self):
        C = self.C
        me, tr, laser = self.setup(0.0)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C), (True, False))
        self.assertEqual(aim.laser_decision(me, tr, laser, False, False, True, C), (False, False))
        self.assertEqual(aim.laser_decision(me, tr, laser, True, True, True, C), (False, False))
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, False, C), (False, False))
        self.assertEqual(aim.laser_decision(me, None, laser, True, False, True, C), (False, False))
        me.short_until = 20.0
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C, t=15.0), (False, False))
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C, t=25.0), (True, False))
        me2, tr2, cd = self.setup(0.0, cooldown=1.0)
        self.assertEqual(aim.laser_decision(me2, tr2, cd, True, False, True, C), (False, False))

    def test_release_at_full_charge_when_aimed(self):
        C = self.C
        full = C.cfg["weapons"]["laser"]["charge"]
        me, tr, laser = self.setup(full)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C), (False, True))
        # mid-charge: keep holding
        me, tr, laser = self.setup(0.4)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C), (True, False))

    def test_full_charge_waits_for_aim(self):
        C = self.C
        full = C.cfg["weapons"]["laser"]["charge"]
        tol = math.atan(C.LASER_FIRE_TOL * 2.5 / 100.0)
        me, tr, laser = self.setup(full, aim_err=tol * 1.5)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C), (True, False))
        me, tr, laser = self.setup(full, aim_err=tol * 0.5)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, True, C), (False, True))
        # unsafe window at full charge: let go anyway
        me, tr, laser = self.setup(full, aim_err=tol * 1.5)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, False, False, C), (False, True))

    def test_early_release_when_los_breaks(self):
        C = self.C
        full = C.cfg["weapons"]["laser"]["charge"]
        me, tr, laser = self.setup(C.LASER_RELEASE_MIN_FRAC * full + 0.01, aim_err=0.2)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, True, True, C), (False, True))
        me, tr, laser = self.setup(C.LASER_RELEASE_MIN_FRAC * full - 0.05)
        self.assertEqual(aim.laser_decision(me, tr, laser, True, True, True, C), (True, False))

    def test_target_lost_releases(self):
        C = self.C
        me, tr, laser = self.setup(0.3)
        self.assertEqual(aim.laser_decision(me, tr, laser, False, False, True, C), (False, True))
        self.assertEqual(aim.laser_decision(me, None, laser, True, False, True, C), (False, True))


@unittest.skipUnless(HAVE_KIT, "recorded hello not available")
class TestScanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = config.bind(hello_config())

    def world(self, t, ghosts=()):
        return SimpleNamespace(me=make_me(self.C), tick=TickInfo(tick=int(t * 10), t=t),
                               ghosts={g.id: g for g in ghosts}, tracks={}, crates={})

    def test_sweep_stays_inside_arc_around_open_sector(self):
        C = self.C
        sc = aim.Scanner(C, random.Random(3))
        centre = (2 + 0.5) * math.pi / 4          # stub: sector 2 is the most open
        half = math.radians(C.SCAN_HALF_ARC_DEG)
        ys = []
        for i in range(120):
            yaw, pitch = sc.scan(self.world(i * 0.1), FlatMap())
            ys.append(yaw)
            self.assertLessEqual(abs(aim.wrap(yaw - centre)), half + 1e-6)
            self.assertEqual(pitch, 0.0)
        self.assertGreater(max(ys) - min(ys), half)       # it actually sweeps

    def test_deterministic_per_seed(self):
        C = self.C
        a = aim.Scanner(C, random.Random(7)).scan(self.world(1.0), FlatMap())
        b = aim.Scanner(C, random.Random(7)).scan(self.world(1.0), FlatMap())
        self.assertEqual(a, b)

    def test_look_order_and_window(self):
        C = self.C
        sc = aim.Scanner(C, random.Random(1))
        g_gun = Ghost(id=-1, pos=(0, 0), bearing=0.5, band="near", first_t=9.9, last_t=9.9, last_kind="gun")
        g_mortar = Ghost(id=-2, pos=(0, 0), bearing=-1.0, band="mid", first_t=9.8, last_t=9.8, last_kind="mortar")
        g_far = Ghost(id=-3, pos=(0, 0), bearing=2.0, band="far", first_t=9.9, last_t=9.9, last_kind="mortar")
        req = sc.look(self.world(10.0, [g_gun, g_mortar, g_far]))
        self.assertIsNotNone(req)
        self.assertEqual((req.rule, req.prio, req.yaw), ("AIM8", 8, -1.0))
        # window holds the same bearing for SCAN_LOOK_S, then moves on to the next ghost
        req2 = sc.look(self.world(10.0 + C.SCAN_LOOK_S / 2, [g_gun, g_mortar, g_far]))
        self.assertEqual(req2.yaw, -1.0)
        req3 = sc.look(self.world(10.0 + C.SCAN_LOOK_S + 0.01, [g_gun, g_mortar, g_far]))
        self.assertEqual(req3.yaw, 0.5)          # the gun sound is next (mortar already looked at)
        req4 = sc.look(self.world(10.0 + 2 * C.SCAN_LOOK_S + 0.05, [g_gun, g_mortar, g_far]))
        self.assertIsNone(req4)                  # nothing new: no repeat look at the same sounds

    def test_look_ignores_old_and_far(self):
        C = self.C
        sc = aim.Scanner(C, random.Random(1))
        old = Ghost(id=-1, pos=(0, 0), bearing=0.5, band="near", first_t=1.0, last_t=1.0, last_kind="gun")
        far = Ghost(id=-2, pos=(0, 0), bearing=0.5, band="far", first_t=9.9, last_t=9.9, last_kind="gun")
        self.assertIsNone(sc.look(self.world(10.0, [old, far])))


class TestWrap(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(aim.wrap(3 * math.pi), math.pi)
        self.assertAlmostEqual(aim.wrap(-math.pi), math.pi)
        self.assertAlmostEqual(aim.wrap(0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
