"""Minimal fakes of world / assess / nav / mapdata / aim matching CONTRACTS.md, for the decision-layer
tests (test_modes, test_arbiter, test_loopfree). No real world.py/assess.py/nav.py/aim.py is imported.

    h = Harness()                       # bound config (recorded hello if the kit is present)
    h.world.add_mech(5, (80, 0, 0), weapons=("gun", "lightning"), vel=(-10, 0, 0))
    h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(...)   # or RuleAssessor rules
    intent, action = h.step()           # strategic_refresh, overlays, decide, resolve, advance 0.1 s
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
import time
from collections import deque

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

import config  # noqa: E402
from contracts import (Assessment, Clock, DamageRec, FightEval, Ghost, Intent, Roof,  # noqa: E402
                       SelfState, SoundRec, Steer, TickInfo, Track, TrackAssess, Threats, WeaponState,
                       ZoneState)

KIT = os.path.join(os.path.dirname(REPO), "mechfight-kit")
STREAM = os.path.join(KIT, "check", "stream.jsonl.gz")
INF = float("inf")
_C_CACHE = {}
_FLAGS = {}          # id(track) -> (fresh: last_seen_t follows t, move: integrate vel) (Track is slotted)


def load_C():
    """config bound to the recorded hello (fresh copy each call); falls back to the table defaults."""
    if "hello" not in _C_CACHE:
        hello_cfg = {}
        if os.path.exists(STREAM):
            with gzip.open(STREAM, "rt") as f:
                hello_cfg = json.loads(f.readline()).get("config", {})
        _C_CACHE["hello"] = hello_cfg
    C = config.bind(_C_CACHE["hello"])
    C.RUSH_A_MIN = 1.0
    return C


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


GUN_STATS = {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12, "reload": 2.4}
LASER_STATS = {"damage": 315, "charge": 0.75, "cooldown": 1.65, "range": 280, "min_charge": 0.15, "hold_s": 2}
LIGHTNING_STATS = {"damage": 60, "range": 45, "cone_deg": 30, "energy": 3, "cooldown": 0.25}
MORTAR_STATS = {"damage": 200, "splash": 160, "splash_radius": 7, "speed": 72, "gravity": 22, "cooldown": 3}
STATS = {"gun": GUN_STATS, "laser": LASER_STATS, "lightning": LIGHTNING_STATS, "mortar": MORTAR_STATS}
PB = {"lightning": "PB_L", "laser": "PB_Z"}


def weapon(kind):
    w = WeaponState(kind, stats=dict(STATS[kind]))
    if kind == "gun":
        w.ammo = 12
    return w


class FakeWorld:
    def __init__(self, C, t=100.0, weapons=("gun", "laser"), pos=(0.0, 0.0, 0.0)):
        self.C = C
        self.tick = TickInfo(tick=int(round(t * 20)), t=t, dt=0.1, alive=16)
        ws = [weapon(k) for k in weapons] + [None] * (2 - len(weapons))
        self.me = SelfState(id=0, pos=pos, hp=1400.0, max_hp=1400.0, energy=100.0, max_energy=100.0,
                            weapons=ws, stats={"walk_speed": 11.7, "boost_speed": 19.5, "turn_rate": 4.0,
                                               "min_pitch": -0.7, "max_pitch": 1.3, "radius": 2.5},
                            muzzle_z=pos[2] + 7.0, eye_z=pos[2] + 8.0)
        self.set_playbook()
        self.tracks = {}
        self.ghosts = {}
        self.sounds = []
        self.beams = []
        self.events = []
        self.pickups = {}
        self.crates = {}
        self.cubes = {}
        self.drops = {}
        self.zone = ZoneState(center=(0.0, 0.0), radius=600.0, next_center=(0.0, 0.0), next_radius=400.0,
                              shrink_start=1e9, shrink_end=1e9 + 10, dps=20.0, from_center=(0.0, 0.0),
                              from_radius=600.0)
        self.offer = None
        self.damage_log = deque()
        self.hits = deque()
        self.shells = []
        self.bullets = []
        self.npc_bullets = []
        self.mortar_active_until = 0.0
        self.kills = 0
        self.loot_bursts = []
        self.seen = set()
        self.heat = {}
        self.persist_events = []      # events re-emitted every tick (e.g. a hit every tick)

    def set_playbook(self):
        kinds = [w.kind for w in self.me.weapons if w is not None]
        self.me.playbook = "PB_L" if "lightning" in kinds else ("PB_Z" if "laser" in kinds else "PB_G")

    def advance(self, dt=0.1):
        self.tick.t = round(self.tick.t + dt, 6)
        self.tick.tick += 2
        self.tick.dt = dt
        self.tick.missed_ticks = 0
        self.events = list(self.persist_events)
        self.sounds = []
        self.zone.next_changed = False
        for tr in self.tracks.values():
            fresh, move = _FLAGS.get(id(tr), (True, False))
            if move:
                tr.pos = (tr.pos[0] + tr.vel[0] * dt, tr.pos[1] + tr.vel[1] * dt, tr.pos[2])
            if fresh:
                tr.last_seen_t = self.tick.t
                if tr.visible:
                    tr.last_visible_t = self.tick.t

    # ---------------------------------------------------------------- builders
    def add_mech(self, tid, pos, weapons=("gun", "laser"), visible=True, vel=(0.0, 0.0, 0.0), fresh=True,
                 move=False, known=None, hp=1400.0, max_hp=1400.0, kind="mech", **kw):
        tr = Track(tid, kind, pos=tuple(pos), vel=tuple(vel), hp=hp, max_hp=max_hp, weapons=tuple(weapons),
                   in_obs=visible, visible=visible, last_seen_t=self.tick.t,
                   last_visible_t=self.tick.t if visible else -1e9,
                   known_weapons=dict(known or {k: 1.0 for k in weapons}))
        for k, v in kw.items():
            setattr(tr, k, v)
        self.tracks[tid] = tr
        _FLAGS[id(tr)] = (fresh, move)
        return tr

    def add_ghost(self, gid, pos, band="near", kinds=None, last_kind="gun"):
        gh = Ghost(gid, (pos[0], pos[1]), math.atan2(pos[1] - self.me.pos[1], pos[0] - self.me.pos[0]), band,
                   self.tick.t, self.tick.t, last_kind, dict(kinds or {}))
        self.ghosts[gid] = gh
        return gh

    def hear(self, kind, gid, band="near", age=0.05):
        gh = self.ghosts.get(gid)
        b = math.atan2(gh.pos[1] - self.me.pos[1], gh.pos[0] - self.me.pos[0]) if gh else 0.0
        if gh:
            gh.last_t = self.tick.t
        self.sounds.append(SoundRec(kind, b, band, age, None, gid))

    def damage(self, amount, by, weapon="gun", bearing=None, unseen=False):
        self.damage_log.append(DamageRec(self.tick.t, amount, weapon, bearing, by, False, unseen))
        tr = self.tracks.get(by)
        if tr is not None:
            tr.last_damaged_us_t = self.tick.t
            tr.damage_to_us += amount

    # ---------------------------------------------------------------- contract queries
    def held_kinds(self):
        return tuple(w.kind for w in self.me.weapons if w is not None)

    def visible_mechs(self):
        return [tr for tr in self.tracks.values() if tr.kind == "mech" and tr.visible]

    def hostile_tracks(self, max_age_s):
        return [tr for tr in self.tracks.values() if self.tick.t - tr.last_seen_t <= max_age_s]

    def damage_since(self, t, by=None):
        return sum(r.amount for r in self.damage_log if r.t >= t and (by is None or r.by == by))

    def damaged_us_recently(self, track_id, window_s):
        return any(r.by == track_id and r.t >= self.tick.t - window_s for r in self.damage_log)

    def heat_at(self, x, y):
        return 0.0

    def seen_at(self, x, y):
        return (int(x // 5), int(y // 5)) in self.seen


def set_flags(tr, fresh=None, move=None):
    f, m = _FLAGS.get(id(tr), (True, False))
    _FLAGS[id(tr)] = (f if fresh is None else fresh, m if move is None else move)


class FakeMap:
    degraded = False

    def __init__(self):
        self.roof_fn = lambda x, y: 0
        self.los_fn = lambda a, b: True
        self.roofs = [Roof(0, 0, 0, 0, 0, 0, (0, 0), [])]
        self.climbs = []
        self.cells = []
        self.clear_fn = lambda x, y: 1.0

    def new_tick(self):
        pass

    def terrain_h(self, x, y):
        return 0.0

    def surface_h(self, x, y):
        return 0.0

    def ground_under(self, x, y, z):
        return 0.0

    def los(self, a, b):
        return self.los_fn(a, b)

    def los_cached(self, key, a, b):
        return self.los_fn(a, b)

    def clearance(self, x, y):
        return self.clear_fn(x, y)

    def roof_id(self, x, y):
        return self.roof_fn(x, y)

    def wall5(self, x, y):
        return 10.0

    def sight5(self, x, y):
        return float("nan")

    def sight5_dir(self, x, y, s):
        return float("nan")

    def cover2(self, x, y):
        return 0

    def cover2_cells(self, x, y, r):
        return list(self.cells)


class FakeNav:
    def __init__(self):
        self.cover = None                 # point, or callable(me, threats, budget) -> point
        self.spots = {}                   # kind -> (point, score, roof_id)
        self.spot_ok = True
        self.blocked = lambda pos, d: False
        self.descent = None
        self.steer = None                 # zone_steer result; None = Steer(self.descent)
        self.eta = (0.0, True)
        self.travel = None                # callable(x, y) -> float | None
        self.stuck = (None, None, None)
        self.exempt_calls = 0             # nav.stuck_exempt() calls (X4 pressing on purpose)
        self.cover_calls = 0

    def zone_field_start(self, zone, outside_current):
        pass

    def zone_eta(self, pos):
        return self.eta

    def zone_descent(self, pos):
        return self.descent

    def zone_steer(self, me, t=None):
        return self.steer if self.steer is not None else Steer(self.descent)

    def zone_field_done(self):
        return True

    def flood_request(self, origin, radius, cost):
        pass

    def travel_time(self, x, y):
        return self.travel(x, y) if self.travel else None

    def path_request(self, key, start, goal, cost, t):
        pass

    def path(self, key):
        return None

    def follow(self, me, key, goal):
        if goal is None:
            return Steer(None, arrived=True)
        d = math.hypot(goal[0] - me.pos[0], goal[1] - me.pos[1])
        if d < 1.0:
            return Steer(None, arrived=True)
        return Steer(unit(goal[0] - me.pos[0], goal[1] - me.pos[1]))

    def clear_dir(self, pos, direction, dist):
        return not self.blocked(pos, direction)

    def stuck_update(self, world, commanded):
        return self.stuck

    def stuck_exempt(self, t):
        self.exempt_calls += 1

    def block_edge(self, a, b, ttl_s, t):
        pass

    def cover_query(self, me, threats, budget_s, sl):
        self.cover_calls += 1
        return self.cover(me, threats, budget_s) if callable(self.cover) else self.cover

    def best_spot(self, kind, world, A):
        return self.spots.get(kind)

    def spot_valid(self, kind, point, world, A):
        return self.spot_ok

    def anytime(self, world, A, clock):
        pass


class FakeAim:
    def _direct(self, me, tr):
        d = max(0.1, math.hypot(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]))
        return math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0]), math.atan2(tr.pos[2] + 4.5 - me.muzzle_z, d)

    def gun_lead(self, me, tr, gun, C):
        y, p = self._direct(me, tr)
        return y, p, 0.1

    def gun_gate(self, me, tr, gun, aim_target, C):
        return tr.visible

    def laser_point(self, me, tr, C):
        return self._direct(me, tr)

    def laser_decision(self, me, tr, laser, los_now, los_breaks_next, safe_window_ok, C, t=None):
        if laser.charge > 0:
            return True, laser.charge >= laser.stats.get("charge", 0.75)
        return bool(tr is not None and los_now and safe_window_ok), False

    def lightning_gate(self, world, mapdata, slot, C, allow_npc=False):
        best = None
        for tr in world.visible_mechs():
            d = math.hypot(tr.pos[0] - world.me.pos[0], tr.pos[1] - world.me.pos[1])
            if d <= C.LIGHTNING_RANGE and (best is None or d < best[0]):
                best = (d, tr.id)
        return (best is not None), (best[1] if best else None)

    def mortar_solution(self, me, mapdata, target, target_vel, mortar, C):
        return None

    def mortar_gate(self, me, solution, target_pred, mortar, C):
        return False

    class Scanner:
        def __init__(self, C, rng):
            pass

        def scan(self, world, mapdata):
            return (0.0, 0.0)


class FightEvalX(FightEval):
    """FightEval plus the optional `cover_eta` extension modes reads with getattr (8.2.6 hysteresis)."""


class RuleAssessor:
    """Fake assessor. `fn(world, A, macro, target, ctx) -> FightEval` scripts the predicates; the default
    returns nothing (no WIN, no LOSE)."""

    def __init__(self, fn=None):
        self.fn = fn
        self.calls = []

    def terrain_call(self, world):
        return ("MIXED", "ZL")

    def evaluate_fight(self, world, A, macro, target_id, ctx):
        self.calls.append((world.tick.t, macro, target_id, dict(ctx)))
        if self.fn is None:
            return FightEval(target_id=target_id)
        return self.fn(world, A, macro, target_id, ctx)


def build_A(world, los=None, slack=100.0, over=None, A_ratio=1.0, **kw):
    """Assessment from fake world geometry. los: {track_id: bool} (default visible -> True)."""
    me = world.me
    per = {}
    los = los or {}
    over = over or {}
    hostiles = []
    for tr in world.tracks.values():
        dx, dy = tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]
        d = math.hypot(dx, dy)
        u = unit(dx, dy) or (1.0, 0.0)
        closing = -(tr.vel[0] * u[0] + tr.vel[1] * u[1])
        lv = los.get(tr.id, tr.visible)
        ta = TrackAssess(tr.id, d, d, math.atan2(dy, dx), lv, 50.0, 50.0, 10.0, 20.0, 2.0, 1.0,
                         False, lv and d < 230, False, closing)
        for k, v in over.get(tr.id, {}).items():
            setattr(ta, k, v)
        per[tr.id] = ta
        if tr.kind == "mech" and lv and ta.in_its_reach:
            hostiles.append(tr.id)
    vis = [tr.id for tr in world.visible_mechs()]
    A = Assessment(t=world.tick.t, per_track=per, target_id=vis[0] if vis else None,
                   main_threat_id=vis[0] if vis else None, hostiles=hostiles, A=A_ratio,
                   threats=Threats(), zone_eta=0.0, slack=slack, terrain_call="MIXED", playbook=me.playbook)
    for k, v in kw.items():
        setattr(A, k, v)
    return A


class Harness:
    """Runs the decision layer (modes + overlays + arbiter) on the fakes, one decision per step()."""

    def __init__(self, t=100.0, weapons=("gun", "laser"), seed=1, C=None):
        import arbiter
        import modes
        import overlays
        self.C = C or load_C()
        self.world = FakeWorld(self.C, t, weapons)
        self.map = FakeMap()
        self.nav = FakeNav()
        self.aim = FakeAim()
        self.assessor = RuleAssessor()
        self.rng = random.Random(seed)
        self.modes = modes.Modes(self.C, self.map, self.nav, self.rng, aim_mod=self.aim)
        self.ov = overlays.Overlays(self.C, self.map, self.nav, self.rng)
        self.arb = arbiter.Arbiter(self.C, self.map, rng=self.rng, aim_mod=self.aim)
        self.A_fn = lambda h: build_A(h.world)
        self.pre = None                   # callable(h) before each decision (scenario script)
        self.inject = None                # callable(h, intent) after modes.decide, before arbiter.resolve
        self.last = None
        self.ostate = None
        self.max_ms = 0.0
        self.sum_ms = 0.0
        self.n = 0

    def step(self, advance=True):
        if self.pre is not None:
            self.pre(self)
        w = self.world
        A = self.A_fn(self)
        clock = Clock(time.perf_counter(), self.C.SOFT_DEADLINE_MS, self.C.HARD_DEADLINE_MS)
        intent = Intent(tick=w.tick.tick, t=w.tick.t)
        self.modes.strategic_refresh(w, self.assessor, self.nav, clock)
        self.ostate = self.ov.update(w, A, self.nav, intent, clock, macro=self.modes.macro, sub=self.modes.sub)
        t1 = time.perf_counter()
        self.modes.decide(w, A, self.nav, self.ostate, intent, clock)
        if self.inject is not None:
            self.inject(self, intent)
        action = self.arb.resolve(w, A, self.nav, intent, clock)
        ms = (time.perf_counter() - t1) * 1000.0
        self.max_ms = max(self.max_ms, ms)
        self.sum_ms += ms
        self.n += 1
        check_one_writer(intent, self.arb)
        self.last = (intent, action, A)
        if advance:
            w.advance()
        return intent, action

    def run(self, n):
        out = []
        for _ in range(n):
            out.append(self.step())
        return out


OWNER_LEVELS = ("X4", "L5", "L6", "L7", "L8")


def check_one_writer(intent, arb):
    """Exactly one writer per actuator: one modes owner leg level; arbiter owners are single values."""
    lv = [lr.level for lr in intent.legs if lr.level in OWNER_LEVELS]
    if len(lv) != 1:
        raise AssertionError(f"modes wrote {len(lv)} owner LegRequests: {lv} ({intent.label()})")
    o = arb.last_owners
    for k in ("legs", "aim", "boost", "jump", "take", "reload"):
        if k not in o or isinstance(o[k], (list, tuple, set)):
            raise AssertionError(f"actuator {k} owner not single: {o.get(k)!r}")
    if not isinstance(o.get("fire"), list) or len(o["fire"]) != 2:
        raise AssertionError(f"fire owners not per slot: {o.get('fire')!r}")


def macro_changes(modes):
    return [h for h in modes.history if h[1] == "macro"]


def goal_entries(modes, goal=None):
    return [h for h in modes.history if h[1] == "goal" and h[5] == "enter" and (goal is None or h[3] == goal)]


def goal_exits(modes, goal=None):
    """Own exits / timeouts / preemptions of a goal (records with no destination)."""
    return [h for h in modes.history if h[1] == "goal" and h[3] is None and (goal is None or h[2] == goal)]
