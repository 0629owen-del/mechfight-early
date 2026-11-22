"""WorldModel tests (plan section 4, 14.1 "Sounds" / "Projectiles", 14.2 stream replay, 3.3 budget).

    cd bot && python3 -m unittest tests.test_world -v

Needs ../mechfight-kit (recorded hello + stream, arena) for the replay and timing tests; the injected
scenarios run on a flat fake map and only need the recorded hello (else config defaults).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

import config  # noqa: E402
import world as W  # noqa: E402
from contracts import Clock  # noqa: E402

KIT = os.path.join(os.path.dirname(REPO), "mechfight-kit")
STREAM = os.path.join(KIT, "check", "stream.jsonl.gz")
ARENA = os.environ.get("MECHFIGHT_MAP_DIR", os.path.join(KIT, "arena"))
HAVE_STREAM = os.path.exists(STREAM)
HAVE_ARENA = os.path.exists(os.path.join(ARENA, "v2", "world.json"))

_LINES = None


def stream_lines():
    global _LINES
    if _LINES is None:
        with gzip.open(STREAM, "rt") as f:
            _LINES = [json.loads(line) for line in f]
    return _LINES


def recorded_hello_config():
    return stream_lines()[0]["config"] if HAVE_STREAM else {}


def bound_config():
    C = config.bind(recorded_hello_config())
    return C


_MAP = None


def real_map(C):
    global _MAP
    if _MAP is None:
        import mapdata
        _MAP = mapdata.MapData.load(ARENA, C, time.perf_counter() + 15.0)
    return _MAP


class FakeMap:
    """Flat open map: ground at z=0, no walls, LOS always clear unless `blockers` says otherwise."""

    def __init__(self, sight=120.0, wall=100.0, ground=0.0):
        self.bounds = (-500.0, -330.0, 550.0, 380.0)
        self.gx0, self.gy0 = -500.5, -330.5
        self.has_sight = sight == sight
        self.has_cover = True
        self._sight, self._wall, self._g = sight, wall, ground
        self.blockers = []          # list of predicates (a, b) -> True if blocked
        self.los_calls = 0

    def terrain_h(self, x, y):
        return self._g

    surface_h = terrain_h

    def los(self, a, b, **kw):
        self.los_calls += 1
        return not any(f(a, b) for f in self.blockers)

    shot_clear = los

    def sight5(self, x, y):
        return self._sight

    def wall5(self, x, y):
        return self._wall

    def roof_id(self, x, y):
        return 0

    def in_bounds(self, x, y):
        b = self.bounds
        return b[0] <= x <= b[2] and b[1] <= y <= b[3]

    def new_tick(self):
        pass


def base_obs(tick=0, t=0.0, pos=(0.0, 0.0, 0.0), aim=(0.0, 0.0), weapons=None, hp=1400.0, energy=100.0,
             cubes=0, zone=None):
    weapons = weapons if weapons is not None else [
        {"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0,
         "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12,
                   "reload": 2.4}}, None]
    return {
        "type": "obs", "tick": tick, "t": t, "alive": 16,
        "self": {"id": 3, "pos": list(pos), "vel": [0, 0, 0], "aim": list(aim), "hp": hp, "max_hp": 1400,
                 "energy": energy, "max_energy": 100, "grounded": True, "cubes": cubes, "upgrades": [],
                 "weapons": weapons, "in_zone": True,
                 "stats": {"radius": 2.5, "height": 9, "eye_height": 8, "muzzle_height": 7, "walk_speed": 11.7,
                           "boost_speed": 19.5, "turn_rate": 4, "armor": 0, "hearing": 1, "regen_delay": 5,
                           "boost_drain": 17, "jet_drain": 15, "max_rise_speed": 10}},
        "mechs": [], "projectiles": [], "crates": [], "cubes": [], "pickups": [], "sounds": [], "beams": [],
        "events": [], "drops": [],
        "zone": zone or {"center": [0, 0], "radius": 560, "next_center": [0, 0], "next_radius": 380,
                         "shrink_start": 60, "shrink_end": 110, "dps": 20},
        "offer": None,
    }


def mech(mid, pos, aim=(0.0, 0.0), weapons=("gun",), vel=(0, 0, 0), charging=False, cubes=0, hp=1400):
    return {"id": mid, "kind": "mech", "pos": list(pos), "vel": list(vel), "aim": list(aim), "hp": hp,
            "max_hp": 1400, "cubes": cubes, "weapons": list(weapons), "charging": charging}


def npc(nid, pos):
    return {"id": nid, "kind": "npc", "pos": list(pos), "vel": [0, 0, 0], "aim": [0, 0], "hp": 350,
            "max_hp": 350}


class _Base(unittest.TestCase):
    def setUp(self):
        self.C = bound_config()
        self.map = FakeMap()
        self.w = W.WorldModel(self.C, self.map, 3)
        self.tick = 0

    def step(self, obs=None, dt=0.1, **kw):
        t = self.tick * 0.05
        o = obs if obs is not None else base_obs(**kw)
        o["tick"], o["t"] = self.tick, round(t, 3)
        config.update_live(self.C, o["self"])
        self.w.update(o, Clock(time.perf_counter(), 1e6, 1e6))
        self.tick += int(round(dt / 0.05))
        self.assertEqual(self.w.exceptions, 0, self.w.last_error)
        return o


# ====================================================================== replay
@unittest.skipUnless(HAVE_STREAM and HAVE_ARENA, "kit stream / arena missing")
class StreamReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = bound_config()
        cls.md = real_map(cls.C)
        cls.w = W.WorldModel(cls.C, cls.md, stream_lines()[0]["slot"])
        cls.ms = []
        cls.first_next_changed = None
        cls.next_changed_count = 0
        cls.bullet_ticks = 0
        cls.missed = []
        for o in stream_lines()[1:]:
            if o.get("type") != "obs":
                continue
            config.update_live(cls.C, o["self"])
            cls.md.new_tick()
            t0 = time.perf_counter()
            cls.w.update(o, Clock(t0, cls.C.SOFT_DEADLINE_MS, cls.C.HARD_DEADLINE_MS))
            cls.ms.append((time.perf_counter() - t0) * 1000)
            if cls.first_next_changed is None:
                cls.first_next_changed = cls.w.zone.next_changed
            cls.next_changed_count += cls.w.zone.next_changed
            cls.bullet_ticks += bool(cls.w.bullets)
            if cls.w.tick.missed_ticks:
                cls.missed.append(cls.w.tick.missed_ticks)
        cls.last = [o for o in stream_lines() if o.get("type") == "obs"][-1]

    def test_no_exceptions(self):
        self.assertEqual(self.w.exceptions, 0, self.w.last_error)

    def test_self_state(self):
        me = self.w.me
        s = self.last["self"]
        self.assertEqual(me.id, s["id"])
        self.assertEqual(me.hp, s["hp"])
        self.assertAlmostEqual(me.eye_z, s["pos"][2] + 8.0)
        self.assertEqual(me.playbook, "PB_G")
        self.assertEqual(self.w.held_kinds(), ("gun",))

    def test_track_10_loadout_and_laser_shots(self):
        tr = self.w.tracks[10]
        self.assertEqual(tr.kind, "mech")
        self.assertEqual(set(tr.weapons), {"gun", "laser"})
        self.assertEqual(tr.known_weapons, {"gun": 1.0, "laser": 1.0})
        self.assertFalse(tr.est)
        self.assertAlmostEqual(tr.last_laser_shot_t, 35.8, places=6)     # beam at t=35.9, age 0.1
        self.assertGreater(tr.damage_to_us, 0)                               # hit at 36.8 within 3 s
        self.assertEqual(tr.fighting_with, self.w.me.id)
        self.assertEqual(tr.lightning_range_mult, 1.0)                       # 0 cubes, no beams

    def test_damage_log_and_hits(self):
        obs = [o for o in stream_lines() if o.get("type") == "obs"]
        t_end = obs[-1]["t"]
        dmg = [e for o in obs for e in o["events"] if e["type"] == "damaged" and o["t"] >= t_end - W.LOG_KEEP_S]
        hits = [e for o in obs for e in o["events"] if e["type"] == "hit" and o["t"] >= t_end - W.LOG_KEEP_S]
        self.assertEqual(len(self.w.damage_log), len(dmg))
        self.assertEqual(len(self.w.hits), len(hits))
        self.assertAlmostEqual(self.w.damage_since(0.0), sum(e["amount"] for e in dmg))
        self.assertAlmostEqual(self.w.damage_since(0.0, by=10), sum(e["amount"] for e in dmg))
        self.assertTrue(self.w.damaged_us_recently(10, 1.0))
        self.assertGreater(self.w.me.regen_block_until, t_end)

    def test_zone_and_ticks(self):
        self.assertTrue(self.first_next_changed)
        self.assertEqual(self.next_changed_count, 1)
        self.assertEqual(self.w.zone.next_radius, 380)
        self.assertEqual(self.w.zone.from_radius, 560)
        self.assertIn(63, self.missed)          # the recorded 128-tick gap = 63 missed decisions
        self.assertEqual(self.w.phase_idx, 0)

    def test_bullets_seen_inbound(self):
        self.assertGreater(self.bullet_ticks, 3)

    def test_memory_and_seen_cells(self):
        me = self.w.me
        self.assertTrue(self.w.seen_at(me.pos[0], me.pos[1]))
        self.assertGreater(self.w.seen_count, 100)
        self.assertIn(8019, self.w.pickups)       # seen at 36.1 s, never in near range
        self.assertGreater(len(self.w.heat), 0)
        g0x, g0y, n = self.w.g0x, self.w.g0y, self.w.gnx
        hot = max(self.w.heat_at(g0x + (c % n + 0.5) * W.SEEN_STEP, g0y + (c // n + 0.5) * W.SEEN_STEP)
                  for c in self.w.heat)
        self.assertGreater(hot, 0.0)

    def test_world_update_budget(self):
        ms = sorted(self.ms)
        p99 = ms[int(0.99 * (len(ms) - 1))]
        print(f"\n  replay world.update: p50 {ms[len(ms) // 2]:.3f} ms, p99 {p99:.3f} ms, max {ms[-1]:.3f} ms")
        self.assertLess(p99, 4.0)


# ====================================================================== sounds / ghosts (4.3)
class SoundInferenceTest(_Base):
    def snd(self, kind, bearing, dist="mid", age=0.05):
        return {"kind": kind, "bearing": bearing, "dist": dist, "age": age}

    def test_ghost_create_merge_band_expire(self):
        o = base_obs()
        o["sounds"] = [self.snd("lightning", 1.0, "near")]
        self.step(o)
        self.assertEqual(len(self.w.ghosts), 1)
        g = next(iter(self.w.ghosts.values()))
        self.assertLess(g.id, 0)
        self.assertEqual(g.kinds, {"lightning": self.C.CONF_SOUND})
        rep = self.C.SOUND_BAND_REP["near"]
        self.assertAlmostEqual(g.pos[0], rep * math.cos(1.0), places=6)
        self.assertEqual(self.w.sounds[0].ghost_id, g.id)
        # same band within GHOST_MERGE_DEG -> merged
        o = base_obs()
        o["sounds"] = [self.snd("gun", 1.0 + math.radians(self.C.GHOST_MERGE_DEG) * 0.8, "near")]
        self.step(o)
        self.assertEqual(len(self.w.ghosts), 1)
        self.assertEqual(set(g.kinds), {"lightning", "gun"})
        # other band -> new ghost
        o = base_obs()
        o["sounds"] = [self.snd("gun", 1.0, "far")]
        self.step(o)
        self.assertEqual(len(self.w.ghosts), 2)
        # expiry after GHOST_TTL_S (ghost is outside our view: aim points away)
        for _ in range(int(self.C.GHOST_TTL_S / 0.1) + 3):
            self.step(aim=(1.0 + math.pi, 0.0))
        self.assertEqual(len(self.w.ghosts), 0)

    def test_ghost_deleted_when_spot_in_view_and_empty(self):
        o = base_obs(aim=(1.0, 0.0))
        o["sounds"] = [self.snd("gun", 1.0, "near")]
        self.step(o)
        self.assertEqual(len(self.w.ghosts), 1)
        for _ in range(int(self.C.SOUND_REACT_AGE_S / 0.1) + 2):
            self.step(aim=(1.0, 0.0))
        self.assertEqual(len(self.w.ghosts), 0)

    def test_association_with_track_and_inference(self):
        # track 7 seen holding gun+laser to the north at 100 m, then hidden (memory over)
        o = base_obs(aim=(math.pi / 2, 0))
        o["mechs"] = [mech(7, (0, 100, 0), weapons=("gun", "laser"))]
        self.step(o)
        self.step()   # not listed any more
        tr = self.w.tracks[7]
        self.assertFalse(tr.in_obs)
        # reload sound, mid band, bearing within 3 sigma -> associates, sets reloading_until
        o = base_obs()
        o["sounds"] = [self.snd("reload", math.pi / 2 + math.radians(15), "mid", age=0.1)]
        self.step(o)
        self.assertEqual(self.w.sounds[0].track_id, 7)
        self.assertEqual(len(self.w.ghosts), 0)
        self.assertAlmostEqual(tr.reloading_until, self.w.tick.t - 0.1 + self.C.path("weapons.gun.reload"))
        # bearing outside ASSOC_SIGMA_K sigma -> ghost instead
        o = base_obs()
        o["sounds"] = [self.snd("gun", math.pi / 2 + math.radians(40), "mid")]
        self.step(o)
        self.assertIsNone(self.w.sounds[0].track_id)
        self.assertEqual(len(self.w.ghosts), 1)
        # wrong band (near) -> not the 100 m track
        o = base_obs()
        o["sounds"] = [self.snd("gun", math.pi / 2, "near")]
        self.step(o)
        self.assertIsNone(self.w.sounds[0].track_id)

    def test_sound_infers_weapon_on_unsighted_track(self):
        # damage-inferred track (never sighted) gets lightning from a sound at CONF_SOUND
        o = base_obs()
        o["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 9}]
        self.step(o)
        tr = self.w.tracks[9]
        self.assertTrue(tr.est)
        o = base_obs()
        o["sounds"] = [self.snd("lightning", 0.05, "mid")]
        # INFER_RANGE_GUN 150 m is in the mid band (60-160)
        self.step(o)
        self.assertEqual(self.w.sounds[0].track_id, 9)
        self.assertEqual(tr.known_weapons["lightning"], self.C.CONF_SOUND)
        self.assertEqual(tr.known_weapons["gun"], self.C.CONF_HARD)
        probs = self.w.weapon_probs(9)
        self.assertEqual(probs["lightning"], self.C.CONF_SOUND)
        self.assertEqual(probs["gun"], 1.0)
        self.assertEqual(probs["laser"], 0.0)       # second slot taken by a confident lightning

    def test_sighted_loadout_excludes_source(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(7, (40, 0, 0), weapons=("gun", "laser"))]
        o["sounds"] = [self.snd("lightning", 0.0, "near")]
        self.step(o)
        self.assertIsNone(self.w.sounds[0].track_id)
        self.assertEqual(len(self.w.ghosts), 1)

    def test_ghost_merges_into_visible_track(self):
        o = base_obs(aim=(math.pi, 0))
        o["sounds"] = [self.snd("laser", 0.0, "mid")]
        self.step(o)
        self.assertEqual(len(self.w.ghosts), 1)
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(11, (100, 5, 0), weapons=("gun", "laser"))]
        self.step(o)
        self.assertEqual(len(self.w.ghosts), 0)

    def test_priors_for_unknown(self):
        o = base_obs()
        o["sounds"] = [self.snd("step", 0.3, "near")]
        self.step(o)
        gid = next(iter(self.w.ghosts))
        probs = self.w.weapon_probs(gid)
        self.assertEqual(probs["laser"], self.C.PRIOR_SECOND["laser"])
        self.assertEqual(probs["lightning"], self.C.PRIOR_SECOND["lightning"])
        kinds = self.w.possible_kinds(gid)
        self.assertEqual(set(kinds), {"gun", "laser", "lightning", "mortar"})   # mortar 0.15 >= PRIOR_MIN_P

    def test_explosion_and_mortar_wary(self):
        o = base_obs()
        o["sounds"] = [self.snd("mortar", 2.0, "mid")]
        self.step(o)
        self.assertGreater(self.w.mortar_active_until, self.w.tick.t)
        o = base_obs()
        o["sounds"] = [self.snd("explosion", 2.0, "mid")]
        self.step(o)
        self.assertEqual(len(self.w.explosions), 1)
        self.assertNotIn("explosion", [g.last_kind for g in self.w.ghosts.values()])
        self.assertAlmostEqual(self.w.mortar_active_until, self.w.tick.t + self.C.MORTAR_WARY_S)

    def test_death_sound_loot_burst(self):
        o = base_obs()
        o["sounds"] = [self.snd("death", 0.0, "near")]
        self.step(o)
        self.assertEqual(len(self.w.loot_bursts), 1)
        self.assertGreater(self.w.heat_at(self.C.SOUND_BAND_REP["near"], 0.0), 0.9)


# ====================================================================== damage inference (4.3.4)
class DamageInferenceTest(_Base):
    def test_unseen_laser_creates_est_track(self):
        o = base_obs()
        o["events"] = [{"type": "damaged", "amount": 315, "weapon": "laser", "bearing": 0.5, "by": 7, "rear": True}]
        self.step(o)
        tr = self.w.tracks[7]
        self.assertTrue(tr.est)
        self.assertEqual(tr.kind, "mech")
        r = self.C.INFER_RANGE_LASER
        self.assertAlmostEqual(tr.pos[0], r * math.cos(0.5), places=6)
        self.assertAlmostEqual(tr.pos[1], r * math.sin(0.5), places=6)
        self.assertEqual(tr.known_weapons, {"laser": self.C.CONF_HARD})
        self.assertEqual(tr.damage_to_us, 315)
        rec = self.w.damage_log[-1]
        self.assertTrue(rec.unseen and rec.rear)
        self.assertEqual(tr.lightning_range_mult, self.w.ghost_lightning_mult)
        # a second hit along a nearby bearing keeps the estimated range, refreshes evidence
        o = base_obs()
        o["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.52, "by": 7}]
        self.step(o)
        self.assertAlmostEqual(math.hypot(tr.pos[0], tr.pos[1]), r, places=6)
        self.assertEqual(set(tr.known_weapons), {"laser", "gun"})
        self.assertEqual(tr.damage_to_us, 360)

    def test_lightning_range_uses_tesla_worst_case(self):
        o = base_obs()
        o["events"] = [{"type": "damaged", "amount": 60, "weapon": "lightning", "bearing": 0.0, "by": 4}]
        self.step(o)
        tr = self.w.tracks[4]
        expect = self.C.INFER_RANGE_LIGHTNING * self.C.TESLA_MUL ** self.C.TESLA_MAX_STACKS
        self.assertAlmostEqual(tr.pos[0], expect, places=6)

    def test_npc_damage_creates_npc_track(self):
        o = base_obs()
        o["events"] = [{"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 1.0, "by": 104}]
        self.step(o)
        tr = self.w.tracks[104]
        self.assertEqual(tr.kind, "npc")
        self.assertTrue(tr.aggro_on_us)
        self.assertEqual(tr.known_weapons, {})

    def test_visible_attacker_not_unseen_and_seen_overrides(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(5, (50, 0, 0), weapons=("gun", "laser"))]
        o["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]
        self.step(o)
        self.assertFalse(self.w.damage_log[-1].unseen)
        self.assertFalse(self.w.tracks[5].est)

    def test_zone_damage_no_track(self):
        o = base_obs()
        o["events"] = [{"type": "damaged", "amount": 20, "weapon": "zone", "bearing": None, "by": None}]
        self.step(o)
        self.assertEqual(self.w.tracks, {})
        self.assertEqual(len(self.w.damage_log), 1)


# ====================================================================== track memory (4.2)
class TrackMemoryTest(_Base):
    def test_mech_forgotten_npc_kept(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (120, 0, 0)), npc(105, (0, 150, 0))]
        o["mechs"][1]["pos"] = [0, 150, 0]
        o["self"]["aim"] = [math.pi / 2 - 0.3, 0]
        self.step(o)
        self.assertIn(6, self.w.tracks)
        self.assertIn(105, self.w.tracks)
        self.assertEqual(self.w.tracks[105].home, (0.0, 150.0, 0.0))
        n = int(self.C.TRACK_FORGET_S / 0.1)
        for _ in range(n - 2):
            self.step(aim=(math.pi, 0))
        self.assertIn(6, self.w.tracks)          # still remembered just before TRACK_FORGET_S
        self.assertFalse(self.w.tracks[6].in_obs)
        for _ in range(5):
            self.step(aim=(math.pi, 0))
        self.assertNotIn(6, self.w.tracks)
        self.assertIn(105, self.w.tracks)        # NPC beyond near range: kept

    def test_npc_dropped_when_home_in_near_view_and_absent(self):
        o = base_obs()
        o["mechs"] = [npc(106, (20, 0, 0))]
        self.step(o)
        for _ in range(int(self.C.path("vision.memory_s", 5) / 0.1) + 15):
            self.step()
        self.assertNotIn(106, self.w.tracks)

    def test_visibility_requires_view_and_los(self):
        o = base_obs(aim=(math.pi, 0))
        o["mechs"] = [mech(6, (120, 0, 0))]            # behind us, beyond near range: memory listing
        self.step(o)
        tr = self.w.tracks[6]
        self.assertTrue(tr.in_obs)
        self.assertFalse(tr.visible)
        self.assertTrue(self.w.tx[6].los_geo)
        self.map.blockers.append(lambda a, b: True)
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (122, 0, 0))]       # new 1 m cell: the LOS cache key changes
        self.step(o)
        self.assertFalse(tr.visible)
        self.map.blockers.clear()
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (121, 0, 0))]
        self.step(o)
        self.assertTrue(tr.visible)
        self.assertEqual(len(self.w.visible_mechs()), 1)

    def test_charging_still_energy(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (100, 0, 0), weapons=("gun", "laser"), charging=True)]
        self.step(o)
        tr = self.w.tracks[6]
        t0 = self.w.tick.t
        self.assertEqual(tr.charging_since, t0)
        self.assertEqual(tr.still_since, t0)
        for _ in range(5):
            o = base_obs(aim=(0.0, 0))
            o["mechs"] = [mech(6, (100, 0, 0), weapons=("gun", "laser"), vel=(19.5, 0, 0))]
            self.step(o)
        self.assertIsNone(tr.charging_since)
        self.assertIsNone(tr.still_since)
        self.assertLess(tr.energy_est, 100.0 - 17 * 0.3)      # boosting drains energy_est

    def test_tesla_multiplier_from_cubes_and_beams(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (100, 0, 0), weapons=("gun", "lightning"), cubes=3),
                      mech(8, (100, 20, 0), weapons=("gun", "lightning"), cubes=1),
                      mech(9, (100, -20, 0), weapons=("gun", "lightning"), cubes=0)]
        o["beams"] = [{"by": 9, "kind": "lightning", "path": [[100, -20, 7], [100, -74, 5]], "age": 0.1}]
        self.step(o)
        self.assertAlmostEqual(self.w.tracks[6].lightning_range_mult, 1.2 ** 2)
        self.assertAlmostEqual(self.w.tracks[8].lightning_range_mult, 1.2)
        self.assertAlmostEqual(self.w.tracks[9].lightning_range_mult, math.dist((100, -20, 7), (100, -74, 5)) / 45)
        self.assertLess(self.w.tracks[9].energy_est, 100.0)

    def test_kill_event_removes_track(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (100, 0, 0))]
        self.step(o)
        o = base_obs()
        o["events"] = [{"type": "kill", "target": 6}]
        self.step(o)
        self.assertNotIn(6, self.w.tracks)
        self.assertEqual(self.w.kills, 1)
        self.assertEqual(self.w.loot_bursts[-1][0], (100.0, 0.0))

    def test_fighting_with_from_aim(self):
        o = base_obs(aim=(0.0, 0))
        o["mechs"] = [mech(6, (100, 0, 0), aim=(math.pi / 2, 0)), mech(8, (100, 60, 0), aim=(-math.pi / 2, 0))]
        self.step(o)
        self.assertEqual(self.w.tracks[6].fighting_with, 8)
        self.assertEqual(self.w.tracks[8].fighting_with, 6)


# ====================================================================== projectiles / items / zone
class ProjectileItemZoneTest(_Base):
    def test_mortar_impact_matches_closed_form(self):
        z0, vx, vz = 30.0, 40.0, 20.0
        o = base_obs()
        o["projectiles"] = [{"id": 900, "kind": "mortar", "pos": [100, 0, z0], "vel": [vx, 0, vz]}]
        self.step(o)
        g = self.C.path("weapons.mortar.gravity")
        tau = (vz + math.sqrt(vz * vz + 2 * g * z0)) / g
        s = self.w.shells[0]
        self.assertLessEqual(s.t_impact - self.w.tick.t - tau, W.SHELL_STEP_S + 1e-9)
        self.assertGreaterEqual(s.t_impact - self.w.tick.t - tau, -1e-9)
        self.assertLess(abs(s.impact[0] - (100 + vx * tau)), vx * W.SHELL_STEP_S + 1e-6)
        self.assertEqual(s.impact[2], 0.0)
        self.assertEqual(s.splash_r, self.C.path("weapons.mortar.splash_radius") * self.C.MORTAR_ENEMY_SPLASH_MULT)
        self.assertGreaterEqual(self.w.mortar_active_until, s.t_impact)

    def test_bullet_closest_approach(self):
        o = base_obs()
        o["projectiles"] = [{"id": 1, "kind": "gun", "pos": [100, 1.0, 4], "vel": [-252, 0, 0]},
                            {"id": 2, "kind": "gun", "pos": [100, 10.0, 4], "vel": [-252, 0, 0]},
                            {"id": 3, "kind": "gun", "pos": [-100, 0.5, 4], "vel": [-252, 0, 0]},
                            {"id": 4, "kind": "npc", "pos": [30, 0, 3], "vel": [-90, 0, 0]}]
        self.step(o)
        self.assertEqual([b.id for b in self.w.bullets], [1])
        b = self.w.bullets[0]
        self.assertAlmostEqual(b.miss_dist, 1.0, places=6)
        self.assertAlmostEqual(b.t_closest - self.w.tick.t, 100 / 252, places=6)
        self.assertEqual([b.id for b in self.w.npc_bullets], [4])

    def test_pickup_memory(self):
        o = base_obs(aim=(0.0, 0))
        o["pickups"] = [{"id": 8001, "kind": "laser", "pos": [30, 0, 0]},
                        {"id": 8002, "kind": "lightning", "pos": [150, 0, 0]}]
        self.step(o)
        self.assertEqual(set(self.w.pickups), {8001, 8002})
        for _ in range(int(W.ITEM_GONE_GRACE_S / 0.1) + 2):
            self.step(aim=(0.0, 0))
        self.assertNotIn(8001, self.w.pickups)    # near range, LOS, unlisted -> gone
        self.assertIn(8002, self.w.pickups)       # wedge beyond near range: listings flicker -> kept
        o = base_obs()
        o["events"] = [{"type": "take_failed", "id": 8002, "reason": "gone"}]
        self.step(o)
        self.assertNotIn(8002, self.w.pickups)
        o = base_obs(pos=(5, 5, 0))
        o["events"] = [{"type": "dropped", "id": 8040, "kind": "mortar", "slot": 1}]
        self.step(o)
        self.assertTrue(self.w.pickups[8040].ours)

    def test_took_short_and_offer(self):
        laser = {"kind": "laser", "cooldown": 1.65, "charge": 0,
                 "stats": {"damage": 315, "charge": 0.75, "cooldown": 1.65, "range": 280}}
        gun = base_obs()["self"]["weapons"][0]
        o = base_obs(weapons=[gun, laser])
        o["events"] = [{"type": "took", "id": 8003, "kind": "laser", "slot": 1},
                       {"type": "effect", "effect": "short", "by": 4}]
        o["offer"] = {"id": 2, "expires": 9.0, "choices": [{"id": "plating"}, {"id": "armor"}, {"id": "sensors"}]}
        self.step(o)
        t = self.w.tick.t
        self.assertAlmostEqual(self.w.me.equip_until[1], t + 1.65)
        self.assertAlmostEqual(self.w.me.short_until, t + self.C.path("weapons.lightning.short_s"))
        self.assertEqual(self.w.me.playbook, "PB_Z")
        self.assertEqual(self.w.me.laser_slot, 1)
        self.assertEqual(self.w.offer.choices, ["plating", "armor", "sensors"])
        self.w.offer.answered_tick = 5
        self.step(o)
        self.assertEqual(self.w.offer.answered_tick, 5)    # same offer id keeps upgrades' bookkeeping

    def test_zone_mid_shrink_back_compute(self):
        z = {"center": [10, 0], "radius": 300, "next_center": [20, 0], "next_radius": 200,
             "shrink_start": 0.0, "shrink_end": 0.2, "dps": 35}
        self.tick = 2            # t = 0.1 -> f = 0.5
        o = base_obs(zone=z)
        self.step(o)
        zs = self.w.zone
        self.assertAlmostEqual(zs.from_center[0], 0.0)
        self.assertAlmostEqual(zs.from_radius, 400.0)
        self.assertAlmostEqual(zs.radius_at(0.1), 300.0)
        self.assertTrue(zs.next_changed)

    def _zone_obs(self, t, center, radius):
        # recorded config: phase 0 holds 0-60 s at 560 m, shrinks 60-110 s to 380 m
        z = {"center": list(center), "radius": radius, "next_center": [40, 0], "next_radius": 380,
             "shrink_start": 60.0, "shrink_end": 110.0, "dps": 20}
        self.tick = int(round(t / 0.05))
        self.step(base_obs(zone=z))
        return self.w.zone

    def test_zone_live_circle_reading(self):
        """4.7 / integration decision: obs.zone during a shrink is the live circle."""
        self._zone_obs(50.0, (0, 0), 560)
        zs = self._zone_obs(60.05, (0.04, 0), 559.82)        # just after shrink start: within tolerance
        self.assertAlmostEqual(zs.radius, 559.82, places=2)
        zs = self._zone_obs(85.0, (20, 0), 470)             # f = 0.5, live circle
        self.assertAlmostEqual(zs.radius, 470.0)
        self.assertAlmostEqual(zs.center[0], 20.0)
        self.assertAlmostEqual(zs.radius_at(110.0), 380.0)
        self.assertAlmostEqual(zs.from_radius, 560.0)
        zs = self._zone_obs(97.5, (30, 0), 425)              # first look mid-shrink, live reading
        self.assertAlmostEqual(zs.radius, 425.0)
        self.assertAlmostEqual(zs.radius_at(110.0), 380.0)

    def test_zone_phase_start_reading(self):
        """4.7 / integration decision: obs.zone keeps reporting the phase-start circle -> interpolate."""
        self._zone_obs(50.0, (0, 0), 560)
        zs = self._zone_obs(85.0, (0, 0), 560)
        self.assertAlmostEqual(zs.radius, 470.0)
        self.assertAlmostEqual(zs.center[0], 20.0)
        self.assertAlmostEqual(zs.radius_at(97.5), 425.0)
        w2 = W.WorldModel(self.C, self.map, 3)             # first look mid-shrink, phase-start reading
        self.w = w2
        zs = self._zone_obs(97.5, (0, 0), 560)
        self.assertAlmostEqual(zs.from_radius, 560.0)
        self.assertAlmostEqual(zs.radius, 425.0)
        self.assertAlmostEqual(zs.center[0], 30.0)

    def test_missed_ticks(self):
        self.step()
        self.tick += 8           # gap of 10 ticks = 4 missed decisions
        self.step()
        self.assertEqual(self.w.tick.missed_ticks, 4)

    def test_seen_cells_fov(self):
        for _ in range(60):
            self.step(aim=(0.0, 0))
        self.assertTrue(self.w.seen_at(60, 0))     # in the wedge, within SEEN_FOV_R
        self.assertTrue(self.w.seen_at(-30, 0))    # near range, behind
        self.assertFalse(self.w.seen_at(-90, 0))   # behind and beyond near range


# ====================================================================== canopy / trunks in sight (protocol 8)
def _item_spans():
    """{(list key, id): [pos, first_t, last_t]} for every pickup / crate the recorded stream lists."""
    span = {}
    for o in stream_lines()[1:]:
        if o.get("type") != "obs":
            continue
        for key in ("pickups", "crates"):
            for it in o.get(key) or []:
                e = span.setdefault((key, it["id"]), [tuple(it["pos"]), o["t"], o["t"]])
                e[2] = o["t"]
    return span


@unittest.skipUnless(HAVE_STREAM and HAVE_ARENA, "kit stream / arena missing")
class StreamSightTest(unittest.TestCase):
    """The recorded match is the only server ground truth for protocol 8 sight: items are listed only
    while seen, mechs for vision.memory_s after they were last seen."""

    @classmethod
    def setUpClass(cls):
        cls.C = bound_config()
        cls.md = real_map(cls.C)
        cls.w = W.WorldModel(cls.C, cls.md, stream_lines()[0]["slot"])
        span = _item_spans()
        mem = float(cls.C.path("vision.memory_s", 5))
        cls.items = {"n": 0, "agree": 0, "seen_unlisted": 0, "listed_hidden": 0}
        cls.mech = {"listed": 0, "visible": 0, "behind_trees": 0, "behind_trees_visible": 0, "max_unseen_s": 0.0}
        last_sight = {}
        for o in stream_lines()[1:]:
            if o.get("type") != "obs":
                continue
            config.update_live(cls.C, o["self"])
            cls.md.new_tick()
            cls.w.update(o, Clock(time.perf_counter(), 1e6, 1e6))
            me = cls.w.me
            eye = (me.pos[0], me.pos[1], me.eye_z)
            listed = {(k, it["id"]) for k in ("pickups", "crates") for it in o.get(k) or []}
            for key, (pos, t0, t1) in span.items():
                if not t0 <= o["t"] <= t1:          # the item certainly exists between its listings
                    continue
                pred = cls.w._point_in_view(pos, 0.5 if key[0] == "pickups" else 1.0, None)
                cls.items["n"] += 1
                cls.items["agree"] += pred == (key in listed)
                cls.items["seen_unlisted"] += pred and key not in listed
                cls.items["listed_hidden"] += (not pred) and key in listed
            for m in o.get("mechs") or []:
                tr, x = cls.w.tracks[m["id"]], cls.w.tx[m["id"]]
                h = cls.w._height_of(tr)
                d = math.hypot(tr.pos[0] - eye[0], tr.pos[1] - eye[1])
                near = d <= cls.w.vis_near
                pts = [(tr.pos[0], tr.pos[1], tr.pos[2] + 0.5 * h), (tr.pos[0], tr.pos[1], tr.pos[2] + h)]
                shot = any(cls.md.los(eye, q) for q in pts)
                sight = x.in_view and any(cls.md.visible(eye, q, near=near) for q in pts)
                if sight:
                    last_sight[m["id"]] = o["t"]
                cls.mech["listed"] += 1
                cls.mech["visible"] += tr.visible
                cls.mech["behind_trees"] += x.in_view and shot and not sight
                cls.mech["behind_trees_visible"] += x.in_view and shot and not sight and tr.visible
                cls.mech["max_unseen_s"] = max(cls.mech["max_unseen_s"], o["t"] - last_sight.get(m["id"], -1e9))
        cls.mem = mem

    def test_sight_estimate_matches_item_listings(self):
        """4.4 / protocol 8: "would the server show us this spot" (the question behind item-gone, ghost-empty
        and seen cells) must follow the listings. The shot line alone says "seen" for 186 of 652 item-ticks
        the server does not list (trees); the sight line gets 649 right and never says "seen" wrongly."""
        it = self.items
        print(f"\n  recorded items: {it['n']} item-ticks, sight estimate agrees on {it['agree']}, "
              f"seen-but-unlisted {it['seen_unlisted']}, listed-but-hidden {it['listed_hidden']}")
        self.assertGreater(it["n"], 600)
        self.assertGreaterEqual(it["agree"], 0.98 * it["n"])
        self.assertLessEqual(it["seen_unlisted"], 0.01 * it["n"])

    def test_listed_mech_behind_trees_stays_visible(self):
        """A-14: obs.mechs keeps a mech for memory_s after the server last saw it (here: trees between us,
        shot line clear). Such an entry has its real position, so Track.visible (fire / aim) keeps it."""
        m = self.mech
        print(f"\n  recorded mech entries: {m['listed']} listed, {m['visible']} Track.visible, "
              f"{m['behind_trees']} in view with a clear shot line but hidden by trees, "
              f"longest listing after the last protocol 8 sighting {m['max_unseen_s']:.1f} s")
        self.assertGreater(m["behind_trees"], 50)
        self.assertEqual(m["behind_trees_visible"], m["behind_trees"])
        self.assertLessEqual(m["max_unseen_s"], self.mem + 0.051)     # never listed past the memory window


_SCENE = {}


def _cell_center(md, x, y):
    """Centre of the 5 m seen cell that holds (x, y)."""
    return (md.gx0 + (math.floor((x - md.gx0) / W.SEEN_STEP) + 0.5) * W.SEEN_STEP,
            md.gy0 + (math.floor((y - md.gy0) / W.SEEN_STEP) + 0.5) * W.SEEN_STEP)


def _open_ground(md, x, y):
    return md.in_bounds(x, y) and md.clearance(x, y) > 0 and not md.roof_id(x, y)


def canopy_scene(md, C):
    """First spot on the real map (trees in file order, 16 headings) where we stand on open ground, on a
    5 m cell centre, right behind a big crown and look through it along `yaw`:
      G   the "mid" ghost spot SOUND_BAND_REP m ahead. A mech standing there has a clear shot line to
          its centre, while the sight lines to its centre and top are blocked by canopy (not a trunk);
      b2  the bearing of a second mid ghost spot inside our wedge with a clear sight line;
      P   a pickup spot 30 m ahead (near range): canopy alone would hide it, a trunk does not.
    Checked with mapdata only, so the scene does not depend on the code under test."""
    if "canopy" in _SCENE:
        return _SCENE["canopy"]
    rep = float(C.SOUND_BAND_REP["mid"])
    for t in md.trees:
        cx, cy, cz, R, Rz = t[:5]
        if R < 5 or Rz < 3:
            continue
        for a in range(16):
            yaw = W.wrap(a * math.pi / 8)
            ux, uy = math.cos(yaw), math.sin(yaw)
            ex, ey = _cell_center(md, cx - (R + 6) * ux, cy - (R + 6) * uy)
            if not _open_ground(md, ex, ey):
                continue
            ez = md.terrain_h(ex, ey)
            eye = (ex, ey, ez + 8.0)
            gx, gy = ex + rep * ux, ey + rep * uy
            px, py = ex + 30.0 * ux, ey + 30.0 * uy
            if not (md.in_bounds(gx, gy) and _open_ground(md, px, py)):
                continue
            gz = md.terrain_h(gx, gy)
            c, top = (gx, gy, gz + 4.5), (gx, gy, gz + 9.0)
            if not md.los(eye, c) or md.los(eye, c, canopy=True) or md.los(eye, top, canopy=True) \
                    or not md.los(eye, c, trunks=True):
                continue
            p = (px, py, md.terrain_h(px, py))
            pc = (px, py, p[2] + 0.5)
            if not md.los(eye, pc) or md.los(eye, pc, canopy=True) or not md.visible(eye, pc, near=True):
                continue
            b2 = None
            for k in range(-45, 46):
                b = yaw + math.radians(k)
                qx, qy = ex + rep * math.cos(b), ey + rep * math.sin(b)
                if md.in_bounds(qx, qy) and md.visible(eye, (qx, qy, md.terrain_h(qx, qy) + 4.5)):
                    b2 = b
                    break
            if b2 is None:
                continue
            _SCENE["canopy"] = {"pos": (ex, ey, ez), "yaw": yaw, "G": (gx, gy, gz), "b2": b2, "P": p}
            return _SCENE["canopy"]
    raise AssertionError("no canopy scene on this map")


def trunk_scene(md):
    """First tree whose trunk stands exactly between an open-ground eye spot and a pickup spot 24 m away
    (near range): shot line clear, sight line cut by the trunk; 5 m to the side the sight line is clear."""
    if "trunk" in _SCENE:
        return _SCENE["trunk"]
    for t in md.trees:
        tx, ty = t[5], t[6]
        for a in range(16):
            ux, uy = math.cos(a * math.pi / 8), math.sin(a * math.pi / 8)
            ex, ey = tx - 12.0 * ux, ty - 12.0 * uy
            px, py = tx + 12.0 * ux, ty + 12.0 * uy
            sx, sy = ex - 5.0 * uy, ey + 5.0 * ux
            if not (_open_ground(md, ex, ey) and _open_ground(md, px, py) and _open_ground(md, sx, sy)):
                continue
            ez, pz, sz = md.terrain_h(ex, ey), md.terrain_h(px, py), md.terrain_h(sx, sy)
            ok = True
            for h in (0.5, 1.75):                     # pickup centre, NPC centre (as world.py tests them)
                q = (px, py, pz + h)
                ok = ok and md.los((ex, ey, ez + 8.0), q) and not md.visible((ex, ey, ez + 8.0), q, near=True) \
                    and md.visible((sx, sy, sz + 8.0), q, near=True)
            if ok:
                _SCENE["trunk"] = {"pos": (ex, ey, ez), "side": (sx, sy, sz), "P": (px, py, pz),
                                   "yaw": math.atan2(uy, ux)}
                return _SCENE["trunk"]
    raise AssertionError("no trunk scene on this map")


class _CountingMap:
    """mapdata proxy that counts rays: the cost of a tick without a clock (sight rays test trees too)."""

    def __init__(self, md):
        self._md = md
        self.n_sight = self.n_shot = 0

    def __getattr__(self, name):
        return getattr(self._md, name)

    def visible(self, eye, target, *, near=False):
        self.n_sight += 1
        return self._md.visible(eye, target, near=near)

    def los(self, a, b, **kw):
        self.n_shot += 1
        return self._md.los(a, b, **kw)


@unittest.skipUnless(HAVE_ARENA, "arena missing")
class CanopySightTest(unittest.TestCase):
    """Protocol 8 on the real map: far vision is blocked by dense canopy and by trunks, proximity
    sensing inside vision.near_range by trunks only; shots are never blocked by trees."""

    def setUp(self):
        self.C = bound_config()
        self.md = real_map(self.C)
        self.k = 0

    def world(self, C=None):
        self.k = 0
        return W.WorldModel(C or self.C, self.md, 3)

    def feed(self, w, o):
        o["tick"], o["t"] = 2 * self.k, round(0.1 * self.k, 3)
        self.k += 1
        config.update_live(w.C, o["self"])
        self.md.new_tick()
        w.update(o, Clock(time.perf_counter(), 1e6, 1e6))      # no time limit: counts are deterministic
        self.assertEqual(w.exceptions, 0, w.last_error)
        return o

    def test_seen_cells_follow_sight_lines(self):
        """4.4: after looking long enough the seen bitmap is exactly the set of cells the server would
        show us: far cells behind a dense crown stay unseen, near-range cells behind it are seen."""
        sc, md, C = canopy_scene(self.md, self.C), self.md, self.C
        w = self.world()
        for _ in range(120):
            self.feed(w, base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0)))
        ex, ey, ez = sc["pos"]
        eye = (ex, ey, ez + 8.0)
        near, far, half = w.vis_near, float(C.SEEN_FOV_R), w.vis_half_fov
        n = {"far_seen": 0, "far_canopy": 0, "near_through_crown": 0, "wrong": 0, "out": 0}
        R = int(far // W.SEEN_STEP) + 2
        for dj in range(-R, R + 1):
            for di in range(-R, R + 1):
                x, y = ex + di * W.SEEN_STEP, ey + dj * W.SEEN_STEP
                d = math.hypot(di, dj) * W.SEEN_STEP
                if d == 0.0 or not md.in_bounds(x, y):
                    continue
                off = abs(W.wrap(math.atan2(dj, di) - sc["yaw"]))
                if abs(d - near) < 0.01 or abs(d - far) < 0.01 or abs(off - half) < 0.01:
                    continue                                    # on a boundary of the view
                tgt = (x, y, md.surface_h(x, y) + 1.0)
                if d < near:
                    want = md.visible(eye, tgt, near=True)
                    n["near_through_crown"] += want and not md.los(eye, tgt, canopy=True)
                elif d < far and off < half:
                    want = md.visible(eye, tgt)
                    n["far_seen"] += want
                    n["far_canopy"] += md.los(eye, tgt) and not md.los(eye, tgt, canopy=True)
                else:
                    want = False
                    n["out"] += 1
                n["wrong"] += w.seen_at(x, y) != want
        self.assertGreater(n["far_canopy"], 5, n)             # the crown does hide far cells here
        self.assertGreater(n["near_through_crown"], 0, n)     # and near-range sensing looks through it
        self.assertGreater(n["far_seen"], 20, n)
        self.assertEqual(n["wrong"], 0, n)

    def _ghost(self, w, bearing):
        sc = canopy_scene(self.md, self.C)
        o = base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0))
        o["sounds"] = [{"kind": "gun", "bearing": bearing, "dist": "mid", "age": 0.05}]
        self.feed(w, o)
        self.assertEqual(len(w.ghosts), 1)
        for _ in range(int(self.C.SOUND_REACT_AGE_S / 0.1) + 5):
            self.feed(w, base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0)))

    def test_ghost_behind_dense_crown_is_not_cleared_by_looking(self):
        """4.3 step 6: "its spot is in view and no mech is there" needs a sight line. The server would not
        show us a mech behind the crown, so an unlisted spot there says nothing; an open spot does."""
        sc = canopy_scene(self.md, self.C)
        w = self.world()
        self._ghost(w, sc["yaw"])
        self.assertEqual(len(w.ghosts), 1)                     # behind the crown: kept until GHOST_TTL_S
        g = next(iter(w.ghosts.values()))
        self.assertAlmostEqual(g.pos[0], sc["G"][0], places=6)
        self.assertAlmostEqual(g.pos[1], sc["G"][1], places=6)
        for _ in range(int(self.C.GHOST_TTL_S / 0.1)):
            self.feed(w, base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0)))
        self.assertEqual(len(w.ghosts), 0)                     # still expires
        w = self.world()
        self._ghost(w, sc["b2"])
        self.assertEqual(len(w.ghosts), 0)                     # clear sight line, nobody there: deleted

    def test_canopy_blocks_false_turns_canopy_off(self):
        """hello.config vision.canopy_blocks = false: the crown no longer hides the ghost spot."""
        cfg = json.loads(json.dumps(recorded_hello_config() or {"vision": {}}))
        cfg.setdefault("vision", {})["canopy_blocks"] = False
        w = self.world(config.bind(cfg))
        self.assertFalse(w.vis_canopy)
        self._ghost(w, canopy_scene(self.md, self.C)["yaw"])
        self.assertEqual(len(w.ghosts), 0)

    def test_listed_mech_behind_dense_crown_fire_and_aim_inputs_unchanged(self):
        """A-13 / A-14, protocol 7: a mech listed behind a dense crown (a remembered entry with its real
        position) is still visible / targetable for the fire and aim code: the shot line ignores trees."""
        sc, md = canopy_scene(self.md, self.C), self.md
        gx, gy, gz = sc["G"]
        eye = (sc["pos"][0], sc["pos"][1], sc["pos"][2] + 8.0)
        self.assertFalse(md.visible(eye, (gx, gy, gz + 4.5)) or md.visible(eye, (gx, gy, gz + 9.0)))
        w = self.world()
        for k in range(3):
            o = base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0))
            o["mechs"] = [mech(6, (gx, gy, gz), weapons=("gun", "laser"))]
            self.feed(w, o)
            tr, x = w.tracks[6], w.tx[6]
            self.assertTrue(tr.in_obs and tr.visible and not tr.est)
            self.assertTrue(x.in_view and x.los_geo)
            self.assertEqual(tr.last_visible_t, w.tick.t)
            self.assertEqual((x.los_since, x.vis_since), (0.0, 0.0))
            self.assertEqual(tr.pos, (gx, gy, gz))
            self.assertEqual([m.id for m in w.visible_mechs()], [6])
        # ... and a hit from it is not an "unseen" hit (O4 does not treat it as a blind hit)
        o = base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0))
        o["mechs"] = [mech(6, (gx, gy, gz), weapons=("gun", "laser"))]
        o["events"] = [{"type": "damaged", "amount": 45, "by": 6, "weapon": "gun", "bearing": sc["yaw"]}]
        self.feed(w, o)
        self.assertFalse(w.damage_log[-1].unseen)

    def test_sight_rays_per_tick_stay_capped(self):
        """3.3: sight rays cost more than shot rays, so their number per tick stays bounded: the seen grid
        takes at most SEEN_LOS_PER_TICK, plus one per remembered item / NPC / ghost that is checked. Cells a
        crown hides stay unmarked and are looked at again, but only inside the same per-tick cap."""
        sc = canopy_scene(self.md, self.C)
        cm = _CountingMap(self.md)
        self.k = 0
        w = W.WorldModel(self.C, cm, 3)
        per_tick = []
        for _ in range(120):                                   # standing still behind the crown
            cm.n_sight = cm.n_shot = 0
            self.feed(w, base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0)))
            self.assertEqual(w.los_calls, cm.n_sight + cm.n_shot)
            self.assertEqual(cm.n_shot, 0)                     # nothing listed: no shot line needed
            per_tick.append(cm.n_sight)
        self.assertLessEqual(max(per_tick), W.SEEN_LOS_PER_TICK)
        self.assertGreater(min(per_tick[60:]), 0)              # hidden cells are still being re-checked
        rng = random.Random(7)                                 # the crowded stream of the timing test
        self.k = 0
        w = W.WorldModel(self.C, cm, 3)
        rays = []
        for k in range(150):
            o = worst_case_obs(rng, 2 * k, self.md)
            # everything that can be asked "is its spot in view" once this tick: remembered + newly listed
            asked = (len(w.pickups) + len(o["pickups"]) + len(w.crates) + len(o["crates"])
                     + len(w.ghosts) + len(o["sounds"])
                     + sum(1 for tr in w.tracks.values() if tr.kind == "npc")
                     + sum(1 for m in o["mechs"] if m["kind"] == "npc"))
            cm.n_sight = cm.n_shot = 0
            config.update_live(self.C, o["self"])
            w.update(o, Clock(time.perf_counter(), 1e6, 1e6))
            self.assertEqual(w.exceptions, 0, w.last_error)
            self.assertLessEqual(cm.n_sight, W.SEEN_LOS_PER_TICK + asked)
            self.assertLessEqual(cm.n_shot, 2 * len(o["mechs"]))      # centre and top of each listed mech
            rays.append((cm.n_sight + cm.n_shot, cm.n_sight))
        print(f"\n  crowded stream rays per tick: max {max(r[0] for r in rays)} "
              f"(sight rays max {max(r[1] for r in rays)}, mean {sum(r[1] for r in rays) / len(rays):.1f})")

    def test_near_range_canopy_does_not_hide_an_item(self):
        """Protocol 8: proximity sensing is not blocked by canopy. A remembered pickup 30 m away behind the
        crown that is no longer listed is gone."""
        sc = canopy_scene(self.md, self.C)
        w = self.world()
        o = base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0))
        o["pickups"] = [{"id": 8001, "kind": "laser", "pos": list(sc["P"])}]
        self.feed(w, o)
        self.assertIn(8001, w.pickups)
        for _ in range(int(W.ITEM_GONE_GRACE_S / 0.1) + 2):
            self.feed(w, base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0)))
        self.assertNotIn(8001, w.pickups)

    def test_trunk_hides_near_item_and_npc(self):
        """Protocol 8: trunks block sight. A remembered pickup / NPC right behind a trunk is not listed
        although it is still there, so it is kept; from 5 m to the side the spot is seen empty -> gone."""
        sc = trunk_scene(self.md)
        w = self.world()
        o = base_obs(pos=sc["side"], aim=(sc["yaw"], 0.0))
        o["pickups"] = [{"id": 8001, "kind": "laser", "pos": list(sc["P"])}]
        o["mechs"] = [npc(106, sc["P"])]
        self.feed(w, o)
        n = int(float(self.C.path("vision.memory_s", 5)) / 0.1) + 15
        for _ in range(n):
            self.feed(w, base_obs(pos=sc["pos"], aim=(sc["yaw"], 0.0)))
        self.assertIn(8001, w.pickups)
        self.assertIn(106, w.tracks)
        for _ in range(3):
            self.feed(w, base_obs(pos=sc["side"], aim=(sc["yaw"], 0.0)))
        self.assertNotIn(8001, w.pickups)
        self.assertNotIn(106, w.tracks)


# ====================================================================== CPU budget (3.3)
def worst_case_obs(rng, tick, md, base_pos=(-60.0, 20.0)):
    """16 mechs (15 others), 12 NPCs, 20 bullets, 6 shells, 40 sounds, beams, events, items."""
    bx, by = base_pos
    o = base_obs(tick=tick, t=tick * 0.05, pos=(bx, by, md.terrain_h(bx, by)), aim=(rng.uniform(-3, 3), 0))
    kinds = ("gun", "laser", "lightning", "mortar")
    ms = []
    for i in range(15):
        mid = i if i < 3 else i + 1
        x, y = bx + rng.uniform(-200, 200), by + rng.uniform(-200, 200)
        ms.append(mech(mid, (x, y, md.terrain_h(x, y)), aim=(rng.uniform(-3, 3), 0),
                       weapons=("gun", kinds[1 + i % 3]), vel=(rng.uniform(-12, 12), rng.uniform(-12, 12), 0),
                       charging=(i % 5 == 0), cubes=i % 7))
    for i in range(12):
        x, y = bx + rng.uniform(-150, 150), by + rng.uniform(-150, 150)
        ms.append(npc(100 + i, (x, y, md.terrain_h(x, y))))
    o["mechs"] = ms
    pr = []
    for i in range(20):
        pr.append({"id": 10000 + tick * 30 + i, "kind": "gun" if i < 16 else "npc",
                   "pos": [bx + rng.uniform(-60, 60), by + rng.uniform(-60, 60), 5],
                   "vel": [rng.uniform(-252, 252), rng.uniform(-252, 252), 0]})
    for i in range(6):
        pr.append({"id": 50000 + tick * 10 + i, "kind": "mortar",      # new ids every tick: no cache hits
                   "pos": [bx + rng.uniform(-100, 100), by + rng.uniform(-100, 100), 40],
                   "vel": [rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(0, 30)]})
    o["projectiles"] = pr
    snd = ("gun", "laser", "lightning", "mortar", "step", "boost", "reload", "jet", "explosion", "npc_gun",
           "laser_charge", "crate", "death")
    o["sounds"] = [{"kind": snd[i % len(snd)], "bearing": rng.uniform(-3.1, 3.1),
                    "dist": ("near", "mid", "far")[i % 3], "age": 0.05} for i in range(40)]
    o["beams"] = [{"by": ms[i]["id"], "kind": "lightning" if i % 2 else "laser",
                   "path": [[ms[i]["pos"][0], ms[i]["pos"][1], 7], [bx, by, 4]], "age": 0.05} for i in range(6)]
    o["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": rng.uniform(-3, 3), "by": 14},
                   {"type": "damaged", "amount": 60, "weapon": "lightning", "bearing": rng.uniform(-3, 3), "by": 15},
                   {"type": "hit", "target": 1, "amount": 45, "weapon": "gun"}]
    o["pickups"] = [{"id": 8000 + i, "kind": kinds[1 + i % 3],
                     "pos": [bx + rng.uniform(-150, 150), by + rng.uniform(-150, 150), 0]} for i in range(40)]
    o["crates"] = [{"id": 1000 + i, "pos": [bx + rng.uniform(-150, 150), by + rng.uniform(-150, 150), 0],
                    "hp": 300} for i in range(20)]
    o["cubes"] = [{"id": 5000 + i, "pos": [bx + rng.uniform(-50, 50), by + rng.uniform(-50, 50), 0]}
                  for i in range(10)]
    o["drops"] = [{"id": 20000, "pos": [bx + 50, by, 0], "radius": 8, "lands_in": 5, "capture_s": 4,
                   "progress": 0, "holder": None}]
    o["offer"] = {"id": 1, "expires": 99, "choices": [{"id": "plating"}, {"id": "armor"}, {"id": "servos"}]}
    return o


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]


@unittest.skipUnless(HAVE_ARENA, "arena missing")
class WorstCaseTimingTest(unittest.TestCase):
    def test_world_update_p99(self):
        C = bound_config()
        md = real_map(C)
        w = W.WorldModel(C, md, 3)
        rng = random.Random(7)
        ms = []
        for k in range(300):
            o = worst_case_obs(rng, 2 * k, md)
            config.update_live(C, o["self"])
            md.new_tick()
            t0 = time.perf_counter()
            w.update(o, Clock(t0, C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS))
            ms.append((time.perf_counter() - t0) * 1000)
        self.assertEqual(w.exceptions, 0, w.last_error)
        p99, mx = pct(ms[5:], 0.99), max(ms[5:])
        print(f"\n  worst-case world.update: p50 {pct(ms, 0.5):.3f} ms, p99 {p99:.3f} ms, max {mx:.3f} ms "
              f"(tracks {len(w.tracks)}, ghosts {len(w.ghosts)})")
        self.assertLess(p99, 4.0)


if __name__ == "__main__":
    unittest.main()
