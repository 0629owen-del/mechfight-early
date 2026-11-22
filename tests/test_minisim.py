"""Rules sanity tests for tools/minisim.py (stdlib unittest).

    python3 -m unittest tests.test_minisim -v
"""
import contextlib
import copy
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import minisim as M  # noqa: E402

KIT = M.KIT_DEFAULT
HELLO = M.load_hello(KIT)


def cfg(**mods):
    c = copy.deepcopy(HELLO["config"])
    for path, v in mods.items():
        d = c
        keys = path.split("__")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return c


def flat_world(c=None, **opts):
    return M.World(c or cfg(), M.Geo(None), seed=3, n=0, opts=opts, populate=False)


def fire(w, mk, slot, yaw=None, pitch=0.0, ticks=None):
    a = {"tick": w.tick, "fire": [slot == 0, slot == 1], "aim": [mk.yaw if yaw is None else yaw, pitch]}
    w.apply_action(mk, a)


def dmg_events(mk, weapon=None):
    return [e for e in mk.events if e["type"] == "damaged" and (weapon is None or e["weapon"] == weapon)]


class TestZone(unittest.TestCase):
    def test_zone_damage_and_regen_block(self):
        w = flat_world()
        out = w.add_mech(0, 480, 480)          # ~705 m from the start centre (r 560)
        inside = w.add_mech(1, 41, -72)
        inside.hp = 1000
        for _ in range(10):                    # 1 s
            w.step_decision()
        self.assertAlmostEqual(out.hp, 1400 - 20, places=4)
        self.assertFalse(out.in_zone)
        self.assertTrue(inside.in_zone)
        self.assertAlmostEqual(inside.hp, 1000 + 45, places=4)   # inside: regen_per_s (no combat yet)
        ob = w.build_obs(out)
        z = [e for e in ob["events"] if e["weapon"] == "zone"]
        self.assertEqual(len(z), 1)
        self.assertIsNone(z[0]["bearing"])
        self.assertEqual(ob["zone"]["radius"], 560)
        self.assertEqual(ob["zone"]["shrink_start"], 60)
        self.assertEqual(ob["zone"]["shrink_end"], 110)

    def test_zone_phases_linear(self):
        w = flat_world()
        ph = w.phases[0]
        w.t = (ph["ss"] + ph["se"]) / 2
        w._update_zone()
        self.assertAlmostEqual(w.zone_live[2], (560 + 380) / 2, places=6)
        self.assertEqual(w.phases[-1]["se"], 415)
        w.t = 420
        w._update_zone()
        self.assertEqual(w.zone_live[2], 0)
        self.assertEqual(w.zone_live[3], 260)


class TestGun(unittest.TestCase):
    def test_rear_multiplier(self):
        w = flat_world(cfg(weapons__gun__spread_deg=0))
        sh = w.add_mech(0, 0, 0, yaw=0.0)
        tg = w.add_mech(1, 50, 0, yaw=0.0)      # facing away from the shooter
        fire(w, sh, 0, yaw=0.0)
        for _ in range(4):
            w.step_tick()
        ev = dmg_events(tg, "gun")
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0]["amount"], 67.5)
        self.assertTrue(ev[0].get("rear"))
        self.assertEqual(ev[0]["by"], 0)
        hit = [e for e in sh.events if e["type"] == "hit"]
        self.assertTrue(hit and hit[0]["rear"] and hit[0]["target"] == 1)
        # facing the shooter: x1
        w2 = flat_world(cfg(weapons__gun__spread_deg=0))
        sh2 = w2.add_mech(0, 0, 0, yaw=0.0)
        tg2 = w2.add_mech(1, 50, 0, yaw=math.pi)
        fire(w2, sh2, 0, yaw=0.0)
        for _ in range(4):
            w2.step_tick()
        ev = dmg_events(tg2, "gun")
        self.assertAlmostEqual(ev[0]["amount"], 45.0)
        self.assertNotIn("rear", ev[0])

    def test_mag_rate_and_reload(self):
        w = flat_world(cfg(weapons__gun__spread_deg=0))
        mk = w.add_mech(0, 0, 0)
        g = mk.weapons[0]
        fire(w, mk, 0)
        shots = []
        reload_done = None
        last = g.ammo
        for k in range(1, 140):
            w.step_tick()
            if g.ammo < last:
                shots.append(w.t)
            if g.ammo > last and reload_done is None:
                reload_done = w.t           # refills to 12 and may fire again on the same tick
            last = g.ammo
        self.assertGreaterEqual(len(shots), 13)
        first12 = shots[:12]
        self.assertAlmostEqual(first12[-1] - first12[0], 11 * 0.24, delta=0.051)
        self.assertIsNotNone(reload_done)
        self.assertAlmostEqual(reload_done - first12[-1], 2.4, delta=0.051)
        self.assertFalse(any(first12[-1] < s < reload_done for s in shots))

    def test_dropped_gun_keeps_ammo(self):
        w = flat_world()
        mk = w.add_mech(0, 0, 0)
        mk.weapons[0].ammo = 5
        w.resolve_loot([], [(mk, 0)])
        ev = [e for e in mk.events if e["type"] == "dropped"]
        self.assertEqual(ev[0]["kind"], "gun")
        self.assertIsNone(mk.weapons[0])
        w.resolve_loot([(mk, ev[0]["id"], None)], [])
        self.assertEqual(mk.weapons[0].ammo, 5)
        self.assertAlmostEqual(mk.weapons[0].equip, 1.0)


class TestLaser(unittest.TestCase):
    def setUp(self):
        self.w = flat_world()
        self.sh = self.w.add_mech(0, 0, 0, yaw=0.0)
        self.sh.weapons[1] = M.Weapon("laser")
        self.tg = self.w.add_mech(1, 100, 0, yaw=math.pi)

    def hold(self, n):
        for _ in range(n):
            fire(self.w, self.sh, 1, yaw=0.0)
            self.w.step_decision()

    def release(self):
        self.w.apply_action(self.sh, {"tick": self.w.tick, "aim": [0.0, 0.0]})
        self.w.step_decision()

    def test_full_charge(self):
        self.hold(7)
        self.assertAlmostEqual(self.w.self_obs(self.sh)["weapons"][1]["charge"], 0.7)
        self.assertTrue(self.sh.laser_charging())
        self.hold(1)
        self.assertAlmostEqual(self.w.self_obs(self.sh)["weapons"][1]["charge"], 0.75)
        self.release()
        ev = dmg_events(self.tg, "laser")
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0]["amount"], 315.0)
        self.assertAlmostEqual(self.sh.weapons[1].cd, 1.65 - 0.05, places=6)

    def test_partial_and_min_charge(self):
        self.hold(3)                       # 0.30 s
        self.release()
        self.assertAlmostEqual(dmg_events(self.tg, "laser")[0]["amount"], round(315 * (0.35 + 0.65 * 0.4), 2))
        w = flat_world()
        sh = w.add_mech(0, 0, 0, yaw=0.0)
        sh.weapons[1] = M.Weapon("laser")
        tg = w.add_mech(1, 100, 0, yaw=math.pi)
        fire(w, sh, 1, yaw=0.0)
        w.step_decision()                  # 0.10 s, below min_charge 0.15
        w.apply_action(sh, {"tick": w.tick})
        w.step_tick()                      # keeps charging to 0.15 and fires
        self.assertAlmostEqual(dmg_events(tg, "laser")[0]["amount"], round(315 * (0.35 + 0.65 * 0.2), 2))

    def test_hold_autofire_and_half_speed(self):
        self.w.apply_action(self.sh, {"tick": 0, "move": [1, 0], "fire": [False, True], "aim": [0, 0]})
        for _ in range(20):
            self.w.step_tick()
        self.assertLessEqual(math.hypot(self.sh.vx, self.sh.vy), 11.7 * 0.5 + 1e-6)
        n = 0
        while not dmg_events(self.tg, "laser") and n < 200:
            self.w.step_tick()
            n += 1
        # full at tick 15, then hold_s 2 s = 40 more ticks -> fires on tick 55
        self.assertEqual(20 + n, 55)

    def test_cookoff(self):
        self.tg.weapons[1] = M.Weapon("mortar")
        self.hold(8)
        self.release()
        ev = dmg_events(self.tg, "laser")
        self.assertAlmostEqual(ev[0]["amount"], 315 * 1.6, places=2)
        self.assertTrue(any(e["type"] == "effect" and e["effect"] == "cookoff" for e in self.tg.events))


class TestLightning(unittest.TestCase):
    def build(self, cone="full"):
        w = flat_world(cone=cone)
        sh = w.add_mech(0, 0, 0, yaw=0.0)
        sh.weapons[1] = M.Weapon("lightning")
        return w, sh

    def test_nearest_in_cone_and_chain(self):
        w, sh = self.build()
        a = w.add_mech(1, 30, 0, yaw=math.pi)
        b = w.add_mech(2, 30, 10, yaw=-math.pi / 2)
        c = w.add_mech(3, 30, 20, yaw=-math.pi / 2)
        d = w.add_mech(4, 30, 30, yaw=-math.pi / 2)
        e = w.add_mech(5, 20, 15.5, yaw=math.pi)     # nearer (25.4 m) but at 37.8 deg: outside the cone
        fire(w, sh, 1, yaw=0.0)
        w.step_tick()
        self.assertAlmostEqual(dmg_events(a)[0]["amount"], 60.0)
        self.assertAlmostEqual(dmg_events(b)[0]["amount"], 42.0)
        self.assertAlmostEqual(dmg_events(c)[0]["amount"], 29.4)
        self.assertEqual(dmg_events(d), [])
        self.assertEqual(dmg_events(e), [])
        self.assertAlmostEqual(sh.energy, 100 - 3)

    def test_cone_reading_and_short(self):
        w, sh = self.build("full")
        t = w.add_mech(1, 30 * math.cos(math.radians(22)), 30 * math.sin(math.radians(22)), yaw=math.pi)
        t.weapons[1] = M.Weapon("laser")
        t.weapons[1].charging, t.weapons[1].charge = True, 0.5
        fire(w, sh, 1, yaw=0.0)
        w.step_tick()
        self.assertEqual(len(dmg_events(t, "lightning")), 1)
        self.assertFalse(t.laser_charging())
        self.assertTrue(any(e["type"] == "effect" and e["effect"] == "short" for e in t.events))
        self.assertGreater(t.short_until, w.t + 1.9)
        w2, sh2 = self.build("half")
        t2 = w2.add_mech(1, 30 * math.cos(math.radians(22)), 30 * math.sin(math.radians(22)), yaw=math.pi)
        fire(w2, sh2, 1, yaw=0.0)
        w2.step_tick()
        self.assertEqual(dmg_events(t2), [])
        self.assertAlmostEqual(sh2.energy, 97)     # fires (and costs) without a target


class TestMortar(unittest.TestCase):
    def test_splash_knock_drain_self(self):
        w = flat_world()
        owner = w.add_mech(0, -5, 0, yaw=0.0)
        tg = w.add_mech(1, 3.5, 0, yaw=math.pi)
        owner.weapons[1] = M.Weapon("mortar")
        p = M.Proj(1, "mortar", 0, 0.0, 0.0, 0.3, 0.0, 0.0, -20.0, 1e9, 7, 200.0, None, 0.3)
        w._impact(p, None, 0.0, 0.0, 0.2)
        f_t = 1 - 0.7 * (1.0 / 7)
        f_o = 1 - 0.7 * (2.5 / 7)
        self.assertAlmostEqual(dmg_events(tg, "mortar")[0]["amount"], round(160 * f_t, 2), places=2)
        self.assertAlmostEqual(dmg_events(owner, "mortar")[0]["amount"], round(160 * f_o * 0.5, 2), places=2)
        self.assertAlmostEqual(tg.vx, 16 * f_t, places=6)
        self.assertAlmostEqual(tg.vz, 9 * f_t, places=6)
        self.assertAlmostEqual(tg.energy, 100 - 45 * f_t, places=6)
        self.assertFalse(tg.grounded)
        self.assertTrue(any(e["type"] == "effect" and e["effect"] == "concuss" for e in tg.events))
        self.assertTrue(any(e["type"] == "hit" for e in owner.events))
        self.assertFalse(any(e["type"] == "hit" and e["target"] == 0 for e in owner.events))

    def test_direct_hit_stacking_option(self):
        for stack, expect in ((False, 200.0), (True, 360.0)):
            w = flat_world(mortar_stack=stack)
            owner = w.add_mech(0, -60, 0)
            owner.weapons[1] = M.Weapon("mortar")
            tg = w.add_mech(1, 0, 0, yaw=math.pi)
            p = M.Proj(1, "mortar", 0, -10.0, 0.0, 4.0, 30.0, 0.0, -5.0, 1e9, 7, 200.0, None, 0.3)
            w._impact(p, tg, -2.5, 0.0, 4.0)
            self.assertAlmostEqual(sum(e["amount"] for e in dmg_events(tg, "mortar")), expect, places=2)


class TestLoot(unittest.TestCase):
    def test_take_drop_semantics(self):
        w = flat_world()
        mk = w.add_mech(0, 0, 0)
        las = w.add_pickup("laser", 2, 0, 0)
        las2 = w.add_pickup("laser", -2, 0, 0)
        mor = w.add_pickup("mortar", 0, 2, 0)
        far = w.add_pickup("lightning", 10, 0, 0)
        w.resolve_loot([(mk, las.id, None)], [])
        self.assertEqual(mk.events[-1], {"type": "took", "id": las.id, "kind": "laser", "slot": 1})
        self.assertAlmostEqual(mk.weapons[1].equip, 1.65)
        w.resolve_loot([(mk, las2.id, None)], [])
        self.assertEqual(mk.events[-1]["reason"], "have")
        w.resolve_loot([(mk, mor.id, None)], [])
        self.assertEqual(mk.events[-1]["reason"], "full")
        w.resolve_loot([(mk, far.id, None)], [])
        self.assertEqual(mk.events[-1]["reason"], "range")
        w.resolve_loot([(mk, 8999, None)], [])
        self.assertEqual(mk.events[-1]["reason"], "gone")
        mk.weapons[1].charging = True
        w.resolve_loot([(mk, mor.id, 1)], [])
        self.assertEqual(mk.events[-1]["reason"], "busy")
        w.resolve_loot([], [(mk, 1)])          # can't drop a charging laser
        self.assertEqual(mk.weapons[1].kind, "laser")
        mk.weapons[1].charging = False
        w.resolve_loot([(mk, mor.id, 1)], [])
        kinds = [e["type"] for e in mk.events[-2:]]
        self.assertEqual(kinds, ["dropped", "took"])
        self.assertEqual(mk.weapons[1].kind, "mortar")
        self.assertTrue(any(it.kind == "laser" and it.id not in (las.id, las2.id) for it in w.items.values()))

    def test_contested_take_closer_wins(self):
        w = flat_world()
        a = w.add_mech(0, 0, 0)
        b = w.add_mech(1, 6, 0)
        p = w.add_pickup("laser", 3.5, 0, 0)
        w.resolve_loot([(a, p.id, None), (b, p.id, None)], [])
        self.assertEqual(b.weapons[1].kind, "laser")
        self.assertEqual(a.events[-1]["reason"], "gone")

    def test_cube_pickup_and_scaling(self):
        w = flat_world()
        a = w.add_mech(0, 0, 0)
        w.add_item("cube", 2, 0, 0)
        w.step_tick()
        self.assertEqual(a.cubes, 1)
        self.assertAlmostEqual(w.max_hp(a), 1400 * 1.04)
        self.assertAlmostEqual(a.hp, 1400 * 1.04)
        self.assertAlmostEqual(w.radius(a), 2.5 * 1.05)
        self.assertAlmostEqual(w.self_obs(a)["stats"]["eye_height"], 8.4)


class TestOffers(unittest.TestCase):
    def test_timeout_applies_choice_zero_and_queue(self):
        w = flat_world()
        mk = w.add_mech(0, 41, -72)
        w.give_cubes(mk, 3)                    # thresholds 1 and 3 -> two queued offers
        w.step_tick()
        o = mk.offer
        self.assertIsNotNone(o)
        self.assertEqual(o["id"], 1)
        self.assertEqual(len(o["choices"]), 3)
        self.assertEqual(len({c["id"] for c in o["choices"]}), 3)
        self.assertAlmostEqual(o["expires"], round(w.t + 5, 2))
        first = o["choices"][0]["id"]
        while mk.offer is not None and mk.offer["id"] == 1:
            w.step_tick()
        self.assertEqual(mk.upg_list, [first])
        self.assertTrue(any(e == {"type": "upgrade", "id": first} for e in mk.events))
        self.assertAlmostEqual(w.t - (o["expires"] - 5), 5.0, delta=0.051)
        self.assertEqual(mk.offer["id"], 2)
        w.apply_action(mk, {"tick": w.tick, "pick": 2})
        self.assertEqual(mk.upg_list[-1], w.mechs[0].upg_list[-1])
        self.assertEqual(len(mk.upg_list), 2)

    def test_weapon_upgrades_need_the_weapon(self):
        w = flat_world()
        mk = w.add_mech(0, 0, 0)
        el = {u["id"] for u in w._eligible(mk)}
        self.assertIn("gun_damage", el)
        self.assertNotIn("laser_damage", el)


class TestPlacement(unittest.TestCase):
    def test_same_tick_deaths_share_and_clock_ranking(self):
        w = flat_world()
        ms = [w.add_mech(i, i * 50.0, 0) for i in range(4)]
        ms[1].hp = ms[2].hp = -1
        w._deaths()
        self.assertEqual((ms[1].place, ms[2].place), (4, 4))
        ms[3].hp = -5
        w._deaths()
        self.assertEqual(ms[3].place, 2)
        self.assertEqual(ms[0].place, 1)
        self.assertTrue(w.over)
        w2 = flat_world()
        ms = [w2.add_mech(i, i * 50.0, 0) for i in range(3)]
        ms[0].hp, ms[1].hp, ms[2].hp = 500, 500, 300
        w2.finish_clock()
        self.assertEqual([m.place for m in ms], [1, 1, 3])


class TestVisionSound(unittest.TestCase):
    def test_wedge_near_memory(self):
        w = flat_world()
        o = w.add_mech(0, 0, 0, yaw=0.0)
        front = w.add_mech(1, 150, 0)
        behind = w.add_mech(2, -100, 0)
        near = w.add_mech(3, -40, 0)
        ids = {m["id"] for m in w.build_obs(o)["mechs"]}
        self.assertEqual(ids, {1, 3})
        front.x = -150                         # leaves view: remembered for memory_s
        for _ in range(10):
            w.step_decision()
            ids = {m["id"] for m in w.build_obs(o)["mechs"]}
        self.assertIn(1, ids)
        for _ in range(45):
            w.step_decision()
            ids = {m["id"] for m in w.build_obs(o)["mechs"]}
        self.assertNotIn(1, ids)

    def test_gun_sound_ranges(self):
        w = flat_world()
        sh = w.add_mech(0, 0, 0)
        mid = w.add_mech(1, 100, 0)
        far = w.add_mech(2, 0, 400)
        fire(w, sh, 0)
        w.step_tick()
        s_mid = [s for s in w.build_obs(mid)["sounds"] if s["kind"] == "gun"]
        s_far = [s for s in w.build_obs(far)["sounds"] if s["kind"] == "gun"]
        self.assertEqual(len(s_mid), 1)
        self.assertEqual(s_mid[0]["dist"], "mid")
        self.assertLess(abs(M.wrap(s_mid[0]["bearing"] - math.pi)), math.radians(40))
        self.assertEqual(s_far, [])
        self.assertEqual([s for s in w.build_obs(sh)["sounds"] if s["kind"] == "gun"], [])


class TestRealMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = M.get_geo(os.path.join(KIT, "arena"))

    def test_schema_valid_obs(self):
        sys.path.insert(0, os.path.join(KIT, "check"))
        import check
        schema = check.Schema(os.path.join(KIT, "mechfight.schema.json"))
        w = M.World(cfg(), self.geo, seed=11, n=16)
        for k in range(150):
            for mk in w.mechs:
                if not mk.alive:
                    continue
                ob = w.build_obs(mk)
                self.assertEqual(schema.errors(ob, "Observation"), [])
                w.apply_action(mk, {"tick": ob["tick"], "move": [math.cos(k / 9 + mk.id), math.sin(k / 9 + mk.id)],
                                    "fire": [True, True], "jump": k % 20 == 0})
            w.end_decision()
            w.step_decision()
        dists = [math.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(w.mechs) for b in w.mechs[i + 1:]]
        self.assertTrue(dists)

    def test_spawn_separation(self):
        w = M.World(cfg(), self.geo, seed=5, n=16)
        c = cfg()
        for i, a in enumerate(w.mechs):
            self.assertLessEqual(math.hypot(a.x - 41.3, a.y + 72.8), 0.8 * 560 + 1)
            for b in w.mechs[i + 1:]:
                self.assertGreaterEqual(math.hypot(a.x - b.x, a.y - b.y), c["match"]["spawn_min_separation"] * 0.95 ** 5)

    def test_walls_stop_and_roof_landing(self):
        g = self.geo
        w = M.World(cfg(), g, seed=1, n=0, populate=False)
        tested = 0
        for area, p, bb in g.roofs[::7]:
            cy = (bb[1] + bb[3]) / 2
            sx = next((bb[0] - d for d in range(4, 60) if all(g.open_ground(bb[0] - d + ox, cy + oy)
                                                             for ox, oy in ((0, 0), (3, 0), (0, 3), (0, -3)))), None)
            if sx is None:
                continue
            w.mechs = []
            mk = w.add_mech(0, sx, cy)
            for _ in range(40):
                w.apply_action(mk, {"tick": 0, "move": [1, 0]})
                w.step_decision()
                self.assertLessEqual(g.ctop[g.idx(mk.x, mk.y)], mk.z + 1.2 + 1e-6)
            tested += 1
            if tested >= 6:
                break
        self.assertGreaterEqual(tested, 3)
        area, p, bb = g.roofs[0]
        x, y, z = g.random_roof(M.random.Random(4))
        w.mechs = []
        mk = w.add_mech(0, x, y, z + 6)
        mk.grounded = False
        for _ in range(20):
            w.step_decision()
        self.assertTrue(mk.grounded)
        self.assertAlmostEqual(mk.z, z, delta=0.6)


class TestCoast(unittest.TestCase):
    def test_coast_then_idle(self):
        w = flat_world()
        mk = w.add_mech(0, 0, 0)
        w.apply_action(mk, {"tick": 0, "move": [1, 0], "boost": True, "aim": [1.0, 0.1], "fire": [True, False],
                            "jump": True})
        for k in range(10):                    # 1.0 s of missed answers: keep move/boost/aim, drop one-shots
            w.apply_action(mk, None)
            self.assertEqual(mk.move, (1.0, 0.0))
            self.assertTrue(mk.boost)
            self.assertEqual(mk.fire, [False, False])
            self.assertFalse(mk.jump)
        w.apply_action(mk, None)               # first decision past coast_s: idle, aim held
        self.assertEqual(mk.move, (0.0, 0.0))
        self.assertFalse(mk.boost)
        self.assertAlmostEqual(mk.cmd_yaw, 1.0)


def block_geo(x0, y0, x1, y1, top):
    """Flat ground with one solid box (walls, flat roof at `top`) on the grid cells x0..x1 by y0..y1."""
    g = M.Geo(None)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            k = g.idx(x, y)
            g.ctop[k] = g.surf[k] = top
    return g


def hspeed(vel):
    return math.hypot(vel[0], vel[1])


def without_vel_xy(ob):
    """An observation minus the only fields --push-vel may change: x/y of self.vel and of each seen mech's vel."""
    ob = copy.deepcopy(ob)
    ob["self"]["vel"] = ob["self"]["vel"][2:]
    for e in ob["mechs"]:
        if e["kind"] == "mech":
            e["vel"] = e["vel"][2:]
    return ob


class TestPushVel(unittest.TestCase):
    """--push-vel: what `vel` reports on a tick in which the mech-mech push-apart (or the eave slide) moved a
    mech. actual (default) = the displacement really made / dt; keep = the velocity state (the old reading).

    What differs in POSITIONS between the two options: nothing. The option only changes what is reported, so
    with the same actions every mech's position, velocity state, energy and HP is bit-identical on every tick,
    and observations differ only in the x/y of self.vel and mechs[].vel, only when the last tick pushed or
    slid one of those mechs (test_same_actions_same_positions_only_vel_differs, test_real_map_*)."""

    WALK = HELLO["config"]["mech"]["walk_speed"]
    STUCK_SPEED = 1.5       # the bot's stuck trigger: commanded to move but slower than this (nav._STUCK_SPEED)

    # ---- helpers
    @staticmethod
    def state(w):
        return [(m.x, m.y, m.z, m.vx, m.vy, m.vz, m.grounded, m.energy, m.hp, m.alive) for m in w.mechs]

    def run_both(self, build, act, decisions):
        """Drive one world per option with the same scripted actions `act(k, mech id)`. Per option:
        states[tick], obs[decision][id], and per tick the ids of mechs that
          extra[tick]: moved differently from what their velocity state says (a push-apart or an eave slide),
          bad[tick]:   moved differently from what their REPORTED velocity says."""
        out = {}
        for mode in ("actual", "keep"):
            w = build(mode)
            rec = {"states": [], "obs": [], "extra": [], "bad": [], "world": w}
            for k in range(decisions):
                obs = {}
                for mk in w.mechs:
                    if mk.alive:
                        obs[mk.id] = w.build_obs(mk)
                        w.apply_action(mk, dict(act(k, mk.id), tick=w.tick))
                w.end_decision()
                rec["obs"].append(obs)
                for _ in range(w.decide_every):
                    before = [(m.x, m.y) for m in w.mechs]
                    w.step_tick()
                    rec["states"].append(self.state(w))
                    extra, bad = set(), set()
                    for m, (x0, y0) in zip(w.mechs, before):
                        if not m.alive:
                            continue
                        dx, dy = m.x - x0, m.y - y0
                        if math.hypot(m.vx * w.dt - dx, m.vy * w.dt - dy) > 1e-9:
                            extra.add(m.id)
                        rvx, rvy = w.vel_h(m)
                        if math.hypot(rvx * w.dt - dx, rvy * w.dt - dy) > 1e-9:
                            bad.add(m.id)
                    rec["extra"].append(extra)
                    rec["bad"].append(bad)
            out[mode] = rec
        return out

    def check_pair(self, res, min_extra):
        """The claims that hold for ANY scenario run through run_both."""
        ac, ke = res["actual"], res["keep"]
        # 1. same physics: positions, velocity state, energy, HP of every mech, every tick, bit for bit
        self.assertEqual(ac["states"], ke["states"])
        self.assertEqual(ac["extra"], ke["extra"])
        n_extra = sum(len(s) for s in ac["extra"])
        self.assertGreaterEqual(n_extra, min_extra)              # the scenario really pushes or slides mechs
        # 2. actual: the reported velocity is the displacement on every mech-tick; keep: wrong on exactly the
        #    mech-ticks where something other than the velocity state moved the mech
        self.assertEqual(sum(len(s) for s in ac["bad"]), 0)
        self.assertEqual(ke["bad"], ke["extra"])
        # 3. observations: equal apart from vel x/y, and fully equal unless the last tick pushed or slid the
        #    observer or a mech it reports
        differing = 0
        for k, (oa, ok) in enumerate(zip(ac["obs"], ke["obs"])):
            self.assertEqual(set(oa), set(ok))
            for mid in oa:
                self.assertEqual(without_vel_xy(oa[mid]), without_vel_xy(ok[mid]))
                if oa[mid] != ok[mid]:
                    differing += 1
                    shown = {mid} | {e["id"] for e in oa[mid]["mechs"] if e["kind"] == "mech"}
                    last_tick = k * ac["world"].decide_every - 1         # the tick this observation follows
                    self.assertTrue(k > 0 and shown & ac["extra"][last_tick], (k, mid))
        self.assertGreater(differing, 0)
        return n_extra, differing

    def head_on(self, **opts):
        """Two mechs walk straight into each other for 3 s. Rows: (obs of 0, obs of 1) after each decision."""
        w = flat_world(**opts)
        a = w.add_mech(0, -10, 0, yaw=0.0)
        b = w.add_mech(1, 10, 0, yaw=math.pi)
        rows = []
        for _ in range(30):
            w.apply_action(a, {"tick": w.tick, "move": [1, 0]})
            w.apply_action(b, {"tick": w.tick, "move": [-1, 0]})
            w.step_decision()
            rows.append((w.build_obs(a), w.build_obs(b)))
        return w, a, b, rows

    # ---- the reading
    def test_head_on_actual_reads_slow_keep_reads_commanded(self):
        for opts, pinned in (({}, True), ({"push_vel": "actual"}, True), ({"push_vel": "keep"}, False)):
            w, a, b, rows = self.head_on(**opts)
            # free approach (0.2 .. 0.7 s): the commanded walking speed under both options
            for oa, ob in rows[1:7]:
                self.assertAlmostEqual(hspeed(oa["self"]["vel"]), self.WALK, places=2)
                self.assertAlmostEqual(hspeed(ob["self"]["vel"]), self.WALK, places=2)
            # they touch at 0.75 s; from 0.9 s on (rows[8:]) both stand rim to rim at +-radius and neither moves
            for oa, ob in rows[8:]:
                self.assertEqual((oa["self"]["pos"][:2], ob["self"]["pos"][:2]), ([-2.5, 0.0], [2.5, 0.0]), opts)
                for me in (oa, ob):
                    other = me["mechs"][0]
                    self.assertEqual(other["kind"], "mech")
                    for sp in (hspeed(me["self"]["vel"]), hspeed(other["vel"])):      # to itself, to the observer
                        if pinned:
                            self.assertLess(sp, self.STUCK_SPEED, opts)
                        else:
                            self.assertAlmostEqual(sp, self.WALK, places=2, msg=opts)
            # the velocity state is not rewritten (unlike at a wall): both still push at walking speed
            self.assertAlmostEqual(a.vx, self.WALK, places=9)
            self.assertAlmostEqual(b.vx, -self.WALK, places=9)

    def test_free_walk_reports_the_same_under_both(self):
        def build(mode):
            w = flat_world(push_vel=mode)
            w.add_mech(0, 0, 0, yaw=0.0)
            w.add_mech(1, 30, 6, yaw=math.pi)       # in each other's view the whole time, never in contact
            return w

        def act(k, mid):                             # both do the same, so they keep their distance
            if k < 12:
                return {"move": [1, 0]}              # walk
            if k < 20:
                return {"move": [1, 0], "boost": True}
            if k < 30:
                return {"move": [0, 1], "jump": k == 24}
            return {"move": [math.cos(k / 3.0), math.sin(k / 3.0)]}     # weave

        res = self.run_both(build, act, 45)
        ac, ke = res["actual"], res["keep"]
        self.assertEqual(sum(len(s) for s in ac["extra"]), 0)                   # nothing pushed or slid
        self.assertEqual(ac["states"], ke["states"])
        speeds = {0: [], 1: []}
        for oa, ok in zip(ac["obs"], ke["obs"]):
            for mid in (0, 1):
                self.assertEqual(json.dumps(oa[mid], sort_keys=True), json.dumps(ok[mid], sort_keys=True))
                self.assertEqual(len(oa[mid]["mechs"]), 1)                      # the other one is reported too
                speeds[mid].append(hspeed(oa[mid]["self"]["vel"]))
        for mid in (0, 1):
            self.assertAlmostEqual(speeds[mid][11], self.WALK, places=2)
            self.assertAlmostEqual(speeds[mid][19], HELLO["config"]["mech"]["boost_speed"], places=2)
        self.assertEqual((sum(len(s) for s in ac["bad"]), sum(len(s) for s in ke["bad"])), (0, 0))

    # ---- positions
    def test_same_actions_same_positions_only_vel_differs(self):
        def build(mode):
            w = M.World(cfg(), block_geo(60, 20, 70, 40, 20.0), seed=3, n=0, opts={"push_vel": mode}, populate=False)
            w.add_mech(0, -10, 0, yaw=0.0)          # 0 and 1: head-on
            w.add_mech(1, 10, 0, yaw=math.pi)
            w.add_mech(2, 40, -12)                  # 2 walks +y into the idle 3, off-centre (glancing shove)
            w.add_mech(3, 41.5, 0)
            w.add_mech(4, -30, 30)                  # 4 boosts into the back of the walking 5
            w.add_mech(5, -22, 30)
            w.add_mech(6, 52, 30)                   # 6 walks into the box wall, 7 shoves it from behind
            w.add_mech(7, 44, 30)
            return w

        def act(k, mid):
            if mid in (0, 1):
                if k >= 30:                          # break off sideways: the pinned pair separates
                    return {"move": [0, 1 if mid == 0 else -1]}
                return {"move": [1, 0] if mid == 0 else [-1, 0], "jump": mid == 1 and k == 20}
            if mid == 2:
                return {"move": [0, 1] if k < 35 else [0, 0]}
            if mid == 4:
                return {"move": [1, 0], "boost": True}
            if mid in (5, 6, 7):
                return {"move": [1, 0]}
            return {}

        res = self.run_both(build, act, 50)
        n_extra, differing = self.check_pair(res, min_extra=150)
        w = res["actual"]["world"]
        self.assertTrue(all(m.alive for m in w.mechs))
        # the wall-held mech 6 was not moved by the shove (walls win), mech 7 behind it took its half
        st = res["actual"]["states"][-1]
        self.assertAlmostEqual(st[6][0], 60 - 0.5 - 2.5 * 0.95, delta=0.6)
        self.assertLess(st[7][0], st[6][0] - 2.5)

    def test_eave_slide_counts_as_displacement(self):
        """The other after-integration move: a mech set down on a roof edge is slid off it at 0.6 m per tick."""
        seen = {}
        for mode in ("actual", "keep"):
            w = M.World(cfg(), block_geo(0, -10, 20, 10, 10.0), seed=3, n=0, opts={"push_vel": mode}, populate=False)
            mk = w.add_mech(0, -1.5, 0.0, 10.3)     # centre 1 m outside the roof, rim over it
            mk.grounded = False
            rows = []
            for _ in range(8):
                x0 = mk.x
                w.step_tick()
                rows.append((round(mk.x - x0, 9), mk.x, mk.z, mk.vx, mk.vy, w.self_obs(mk)["vel"][:2]))
            seen[mode] = rows
        slid = [k for k, r in enumerate(seen["actual"]) if r[0] != 0.0]
        self.assertEqual(slid, [2, 3, 4])                                        # lands on the rim, slides 3 ticks
        for k in range(8):
            ra, rk = seen["actual"][k], seen["keep"][k]
            self.assertEqual(ra[:5], rk[:5])                                     # same motion, same state
            self.assertEqual((ra[3], ra[4]), (0.0, 0.0))                         # the state never carries the slide
            self.assertEqual(rk[5], [0.0, 0.0])
            self.assertEqual(ra[5], [-12.0, 0.0] if k in slid else [0.0, 0.0])   # 0.6 m / 0.05 s

    def test_mortar_knock_is_an_impulse_under_both(self):
        """A knock moves nobody: it adds to the velocity state, which reports at once and moves the mech from
        the next tick on, the same under both options (the one case where vel is not last tick's displacement)."""
        for mode in ("actual", "keep"):
            w = flat_world(push_vel=mode)
            w.add_mech(0, -60, 0).weapons[1] = M.Weapon("mortar")       # the shell's owner, out of the blast
            tg = w.add_mech(1, 3.5, 0, yaw=math.pi)
            p = M.Proj(1, "mortar", 0, 0.0, 0.0, 0.3, 0.0, 0.0, -20.0, 1e9, 7, 200.0, None, 0.3)
            w._impact(p, None, 0.0, 0.0, 0.2)
            knock = 16 * (1 - 0.7 * (1.0 / 7))
            self.assertEqual((tg.x, tg.y), (3.5, 0.0))
            self.assertEqual(w.self_obs(tg)["vel"][:2], [round(knock, 2), 0.0], mode)
            x0 = tg.x
            w.step_tick()
            self.assertAlmostEqual(w.self_obs(tg)["vel"][0], (tg.x - x0) / w.dt, delta=0.006, msg=mode)
            self.assertGreater(tg.x - x0, 0.5)

    def test_real_map_reported_velocity_is_the_displacement(self):
        """16 mechs converge on each other on the real map (walls, slopes, roofs, jumps, boost, gunfire)."""
        geo = M.get_geo(os.path.join(KIT, "arena"))

        def build(mode):
            return M.World(cfg(), geo, seed=11, n=16, opts={"push_vel": mode})

        centre = {}

        def act(k, mid):
            w = centre["w"]
            if centre.get("k") != k:
                live = [m for m in w.mechs if m.alive]
                centre["k"], centre["c"] = k, (sum(m.x for m in live) / len(live), sum(m.y for m in live) / len(live))
            m = w.mechs[mid]
            a = math.atan2(centre["c"][1] - m.y, centre["c"][0] - m.x) + 0.6 * math.sin(k / 7 + mid)
            return {"move": [math.cos(a), math.sin(a)], "boost": (k // 15 + mid) % 3 == 0,
                    "jump": (k + mid) % 23 == 0, "fire": [True, True]}

        def build_tracked(mode):
            centre.clear()
            centre["w"] = build(mode)
            return centre["w"]

        res = self.run_both(build_tracked, act, 300)
        self.check_pair(res, min_extra=100)

    # ---- the option
    def test_option_values_and_default(self):
        self.assertEqual(flat_world().opts["push_vel"], "actual")
        with self.assertRaises(ValueError):
            flat_world(push_vel="bogus")
        ap = M.build_argparser()
        self.assertEqual(ap.parse_args(["--out", "x"]).push_vel, "actual")
        self.assertEqual(ap.parse_args(["--out", "x", "--push-vel", "keep"]).push_vel, "keep")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ap.parse_args(["--out", "x", "--push-vel", "bogus"])

    def test_option_recorded_in_meta_and_passed_through_run_matches(self):
        def meta(path):
            with open(path) as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec["type"], "meta")
            return rec

        tiny = ["--duration", "2", "--mechs", "2"]
        with tempfile.TemporaryDirectory() as d:
            cmd = [sys.executable, os.path.join(REPO, "tools", "minisim.py"), "--out", os.path.join(d, "direct"),
                   "--bot", "scripted:duelist", "--fill", "scripted:duelist", "--quiet"] + tiny
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertEqual(meta(os.path.join(d, "direct", "match.jsonl"))["opts"]["push_vel"], "actual")
            # run_matches.py hands everything after "--" to minisim.py unchanged
            cmd = [sys.executable, os.path.join(REPO, "tools", "run_matches.py"), "-k", "1", "-j", "1",
                   "--out", os.path.join(d, "batch"), "--bot", "scripted:duelist", "--fill", "scripted:duelist",
                   "--", "--push-vel", "keep"] + tiny
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            m = meta(os.path.join(d, "batch", "m000", "match.jsonl"))
            self.assertEqual((m["opts"]["push_vel"], m["duration"], m["n"]), ("keep", 2.0, 2))


SLOW_BOT = r"""
import sys, json, time
n = 0
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "hello":
        print(json.dumps({"type": "ready", "name": "slow"}), flush=True)
    elif m["type"] == "obs":
        n += 1
        if n % 5 == 0:
            time.sleep(0.12)
        print(json.dumps({"tick": m["tick"], "move": [1, 0]}), flush=True)
    elif m["type"] == "end":
        break
"""


class TestServerTiming(unittest.TestCase):
    def test_budget_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            bot = os.path.join(d, "slow.py")
            with open(bot, "w") as f:
                f.write(SLOW_BOT)
            cmd = [sys.executable, os.path.join(REPO, "tools", "minisim.py"), "--out", os.path.join(d, "o"),
                   "--duration", "6", "--mechs", "2", "--bot", f"{sys.executable} -u {bot}", "--fill",
                   "scripted:duelist", "--quiet", "--no-shuffle"]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            with open(os.path.join(d, "o", "summary.json")) as f:
                s = json.load(f)
            slow = [r for r in s["places"] if r["name"] == "slow"][0]
            self.assertGreater(slow["missed"], 5)
            self.assertGreater(slow["answered"], 30)
            self.assertGreater(slow["late"] + slow["stale"], 0)


class TestScriptedBots(unittest.TestCase):
    """Calibrated engagement knobs of tools/minisim_bots.py (profile v2) and the v1 fallback."""

    @staticmethod
    def duel(B, seconds=30.0):
        """A scripted duelist vs an idle mech 120 m ahead on flat ground; returns (bot, gun shot times)."""
        w = flat_world()
        me = w.add_mech(0, 0, 0, yaw=0.0)
        w.add_mech(1, 120, 0, yaw=math.pi)
        bot = B.ScriptBot({"config": w.cfg, "slot": 0, "seed": 1}, "duelist")
        shots = []
        while w.t < seconds:
            ob = w.build_obs(me)
            w.end_decision()
            ammo = me.weapons[0].ammo
            w.apply_action(me, bot.act(ob))
            w.step_decision()
            if me.weapons[0].ammo < ammo:
                shots.append(w.t)
        return bot, shots

    def test_v1_specs(self):
        argv, env, label = M.expand_spec("scripted:mix_v1", KIT, 0)
        self.assertEqual((env, label, argv[-1]), ({"MINISIM_BOTS_PROFILE": "v1"}, f"scripted:{M.MIX[0]}_v1", M.MIX[0]))
        argv, env, label = M.expand_spec("scripted:rusher", KIT, 3)
        self.assertEqual((env, label, argv[-1]), ({}, "scripted:rusher", "rusher"))

    def test_calibrated_gates(self):
        if any(k.startswith("MINISIM_BOTS_") for k in os.environ):
            self.skipTest("bot knobs overridden by the environment")
        import minisim_bots as B
        bot, shots = self.duel(B)
        self.assertTrue(shots)
        self.assertGreaterEqual(shots[0], bot.peace_end + B.REACT_S)            # peace window, then reaction
        gaps = [b - a for a, b in zip(shots, shots[1:])]
        self.assertGreaterEqual(min(gaps), 0.3 - 1e-6)                           # cadence (cooldown alone: 0.25)
        self.assertGreaterEqual(max(gaps), B.PAUSE_S - 0.15)                     # a burst/pause (reload is 2.4 s)

    def test_v1_profile_fires_on_sight(self):
        import importlib
        import minisim_bots as B
        old = os.environ.get("MINISIM_BOTS_PROFILE")
        os.environ["MINISIM_BOTS_PROFILE"] = "v1"
        try:
            importlib.reload(B)
            _, shots = self.duel(B, seconds=6.0)
        finally:
            if old is None:
                os.environ.pop("MINISIM_BOTS_PROFILE", None)
            else:
                os.environ["MINISIM_BOTS_PROFILE"] = old
            importlib.reload(B)
        self.assertLess(shots[0], 1.0)
        self.assertLessEqual(min(b - a for a, b in zip(shots, shots[1:])), 0.25 + 1e-6)


class TestSmokeMatch(unittest.TestCase):
    def test_example_and_scripted_60s(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = [sys.executable, os.path.join(REPO, "tools", "minisim.py"), "--out", d, "--seed", "2",
                   "--duration", "60", "--bot", "example", "--fill", "scripted:mix", "--quiet"]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            with open(os.path.join(d, "summary.json")) as f:
                s = json.load(f)
            self.assertEqual(len(s["places"]), 16)
            self.assertGreaterEqual(s["t_end"], 20)
            ex = [r for r in s["places"] if r["label"] == "example"][0]
            self.assertEqual(ex["name"], "example-py")
            alive_decisions = int(((ex["t_death"] or s["t_end"])) * 10)
            self.assertGreaterEqual(ex["answered"], 0.8 * alive_decisions)
            self.assertTrue(all(1 <= r["place"] <= 16 for r in s["places"]))
            with open(os.path.join(d, "match.jsonl")) as f:
                types = {json.loads(l)["type"] for l in f}
            self.assertTrue({"meta", "shot", "sight", "place", "summary"} <= types)
            self.assertEqual(len([n for n in os.listdir(d) if n.endswith(".stderr")]), 16)


if __name__ == "__main__":
    unittest.main()
