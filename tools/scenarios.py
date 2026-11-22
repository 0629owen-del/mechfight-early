#!/usr/bin/env python3
"""Scenario fixtures: recorded observations mutated with injected enemies, shells, damage, offers, drops.

Usable from tests (`tests/test_worstcase.py`) and from the shell:

    sc = Scenario.quiet(site=find_open_site(), warm=10)         # still self on a real walkable cell
    sc.weapons("gun", "laser")
    sc.laser_at_us(dist=70, bearing=0.5)                         # add_* / pickup / crate / drop return the entity dict
    action = sc.run()                                            # real main.Bot, last action
    move_dir(action)                                             # unit vector or None

    sc = Scenario.from_recorded(index=120, warm=10)              # keep the recorded entities/positions
    sc.offer(["armor", "laser_damage", "plating"])

Everything is stdlib; only the `Bot(hello, deadline)` / `act(obs, t0)` interface of main.py is used.
Frames: world frame, yaw = atan2(dy, dx) (protocol 5), z up.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import worstcase as W  # noqa: E402  (shared loaders: stream, map, schema)

OBS_DT = W.OBS_DT


# ---------------------------------------------------------------------------- geometry helpers
def yaw_to(dx, dy):
    return math.atan2(dy, dx)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def move_dir(action):
    """Unit world-frame move direction of an action, or None when it stands still."""
    m = (action or {}).get("move")
    if not m:
        return None
    n = math.hypot(m[0], m[1])
    return None if n < 1e-6 else (m[0] / n, m[1] / n)


def angle_between(u, v):
    """Unsigned angle in radians between 2-vectors."""
    d = (u[0] * v[0] + u[1] * v[1]) / ((math.hypot(*u) * math.hypot(*v)) or 1.0)
    return math.acos(max(-1.0, min(1.0, d)))


def _g():
    md = W.load_map()
    return W.Ground(md, __import__("random").Random(0))


def find_open_site(seed=0, zone_frac=0.35, min_clear=0.0):
    """A real walkable ground (x, y, z) well inside both circles (recorded zone: c=(41,-73) r=560, next r=380)."""
    import random
    hello, rec, _ = W.load_stream()
    z = hello["config"]["zone"]
    g = W.Ground(W.load_map(), random.Random(seed))
    for _ in range(3000):
        p = g.near(z["start_center"][0], z["start_center"][1], 0, z["start_radius"] * zone_frac)
        if p and p[2] < 10 and all(g.ok(p[0] + dx, p[1] + dy) for dx in (-12, 0, 12) for dy in (-12, 0, 12)):
            return p
    raise RuntimeError("no open site")


def clear_bearing(site, dist, start=0.0, eye=4.5):
    """First bearing (scanned from `start` in 15 degree steps) along which the real-map LOS (canopy included) from our eye to a
    walkable point `dist` away on similar ground is open. None when every direction is blocked."""
    md = W.load_map()
    g = _g()
    for k in range(24):
        b = wrap(start + math.radians(15 * k))
        x, y = site[0] + dist * math.cos(b), site[1] + dist * math.sin(b)
        if g.ok(x, y) and abs(g.z(x, y) - site[2]) < 3.0 and md.los((site[0], site[1], site[2] + 8.0), (x, y, g.z(x, y) + eye), canopy=True):
            return b
    return None


def find_cover_site(seed=0, charger_dist=70.0, cover_r=11.0):
    """A walkable site with a laser charger placed `charger_dist` away such that the site has LOS to the charger's
    muzzle but a cell within `cover_r` does not (real map LOS). Returns dict(site, bearing, charger, cover_cells)."""
    import random
    md = W.load_map()
    hello, rec, _ = W.load_stream()
    z = hello["config"]["zone"]
    rng = random.Random(seed)
    g = W.Ground(md, rng)
    for _ in range(4000):
        p = g.near(z["start_center"][0], z["start_center"][1], 0, 200)
        if not p or p[2] > 10:
            continue
        for k in range(12):
            b = rng.uniform(-math.pi, math.pi)
            cx, cy = p[0] + charger_dist * math.cos(b), p[1] + charger_dist * math.sin(b)
            if not g.ok(cx, cy):
                continue
            cz = g.z(cx, cy)
            muz = (cx, cy, cz + 7.0)
            if not md.los(muz, (p[0], p[1], p[2] + 4.5)):
                continue  # site must be exposed
            cover = []
            for a in range(0, 360, 20):
                for r in (4.0, 7.0, 10.0):
                    qx, qy = p[0] + r * math.cos(math.radians(a)), p[1] + r * math.sin(math.radians(a))
                    if g.ok(qx, qy) and abs(g.z(qx, qy) - p[2]) < 2.5 and not md.los(muz, (qx, qy, g.z(qx, qy) + 4.5)):
                        cover.append((qx, qy))
            # require cover on one side only (unambiguous direction), nearest cover within cover_r
            if cover:
                d = [(math.hypot(q[0] - p[0], q[1] - p[1]), q) for q in cover]
                d.sort()
                if d[0][0] <= cover_r:
                    return {"site": p, "bearing": b, "charger": (cx, cy, cz), "cover_cells": [q for _, q in d]}
    raise RuntimeError("no cover site found")


# ---------------------------------------------------------------------------- the builder
class Scenario:
    """A short observation sequence ending in the injected tick(s). Methods mutate the final observation."""

    def __init__(self, hello, base_obs, ground):
        self.hello = hello
        self.cfg = hello["config"]
        self.g = ground
        self.seq = base_obs            # list of obs; the last one is the injected tick
        self.cur = base_obs[-1]
        self._next_id = 50000
        self._shells = []              # (id, p0, v0, eta, spawn_t) for follow-up ticks
        self.follow = 0

    # -- constructors
    @classmethod
    def quiet(cls, site=None, t=20.0, warm=10, seed=0, hello=None):
        """Nothing in view, self standing still on `site`, zone from the recorded stream."""
        h, rec, _ = W.load_stream()
        hello = hello or h
        site = site or find_open_site(seed)
        tpl = copy.deepcopy(rec[0])
        seq = []
        tick0 = int(round(t * 20)) - 2 * warm
        for i in range(warm + 1):
            o = copy.deepcopy(tpl)
            o["tick"] = tick0 + 2 * i
            o["t"] = round(o["tick"] / 20, 3)
            me = o["self"]
            me["pos"] = [round(site[0], 2), round(site[1], 2), round(site[2], 2)]
            me["vel"] = [0, 0, 0]
            me["grounded"] = True
            me["in_zone"] = True
            for k in ("mechs", "projectiles", "crates", "cubes", "pickups", "sounds", "beams", "events", "drops"):
                o[k] = []
            o["offer"] = None
            seq.append(o)
        return cls(hello, seq, W.Ground(W.load_map(), __import__("random").Random(seed)))

    @classmethod
    def from_recorded(cls, index=120, warm=10, hello=None):
        """Keep the recorded observations [index-warm, index] verbatim; injections go into the last one."""
        h, rec, _ = W.load_stream()
        index = min(max(index, warm), len(rec) - 1)
        seq = copy.deepcopy(rec[index - warm:index + 1])
        return cls(hello or h, seq, W.Ground(W.load_map(), __import__("random").Random(0)))

    # -- accessors
    @property
    def me(self):
        return self.cur["self"]

    @property
    def pos(self):
        return tuple(self.me["pos"])

    def nid(self):
        self._next_id += 1
        return self._next_id

    def at(self, dist, bearing):
        """World (x, y, z) at `dist` from us along `bearing`; z from the map when walkable there."""
        x, y, z = self.pos
        px, py = x + dist * math.cos(bearing), y + dist * math.sin(bearing)
        pz = self.g.z(px, py) if self.g.ok(px, py) else z
        return [round(px, 2), round(py, 2), round(pz, 2)]

    # -- self
    def weapons(self, *kinds):
        """Held loadout, e.g. weapons("gun", "laser"). Stats come from the recorded hello."""
        wc = self.cfg["weapons"]
        slots = []
        for k in kinds[:2]:
            st = {kk: vv for kk, vv in wc[k].items() if kk in ("damage", "cooldown", "speed", "range", "spread_deg", "mag", "reload", "charge", "pierce")}
            w = {"kind": k, "cooldown": 0, "stats": st}
            if k == "gun":
                w["ammo"] = wc["gun"]["mag"]
                w["reload"] = 0
            if k == "laser":
                w["charge"] = 0
            slots.append(w)
        while len(slots) < 2:
            slots.append(None)
        for o in self.seq:
            o["self"]["weapons"] = copy.deepcopy(slots)
        return self

    def hp(self, hp):
        self.me["hp"] = hp
        return self

    def outside_zone(self, margin=60.0):
        """Teleport self (all warm ticks) `margin` m outside the current circle on a real walkable cell."""
        z = self.cur["zone"]
        for _ in range(500):
            p = self.g.near(z["center"][0], z["center"][1], z["radius"] + margin * 0.5, z["radius"] + margin * 2)
            if p:
                break
        else:
            raise RuntimeError("no walkable cell outside the zone")
        for o in self.seq:
            o["self"]["pos"] = [round(p[0], 2), round(p[1], 2), round(p[2], 2)]
            o["self"]["in_zone"] = False
        return self

    def set_zone(self, center=None, radius=None, next_center=None, next_radius=None, shrink_start=None, shrink_end=None):
        for o in self.seq:
            z = o["zone"]
            if center is not None:
                z["center"] = list(center)
            if radius is not None:
                z["radius"] = radius
            if next_center is not None:
                z["next_center"] = list(next_center)
            if next_radius is not None:
                z["next_radius"] = next_radius
            if shrink_start is not None:
                z["shrink_start"] = shrink_start
            if shrink_end is not None:
                z["shrink_end"] = shrink_end
        return self

    # -- entities
    def add_mech(self, dist, bearing, weapons=("gun", "laser"), charging=False, face_us=True, hp=1400, cubes=0,
                 vel=(0.0, 0.0), id=None, kind="mech", pos=None):
        p = list(pos) if pos else self.at(dist, bearing)
        aim_yaw = yaw_to(self.pos[0] - p[0], self.pos[1] - p[1]) if face_us else bearing
        a = {"id": id if id is not None else self.nid(), "kind": kind, "pos": p, "vel": [vel[0], vel[1], 0],
             "aim": [round(aim_yaw, 3), 0], "hp": hp, "max_hp": hp if kind == "npc" else 1400}
        if kind == "mech":
            a["cubes"] = cubes
            a["weapons"] = list(weapons)
            a["charging"] = bool(charging)
        self.cur["mechs"].append(a)
        return a

    def add_npc(self, dist, bearing, **kw):
        return self.add_mech(dist, bearing, kind="npc", hp=350, **kw)

    def laser_at_us(self, dist=70.0, bearing=0.0, pos=None, id=None, sound=True):
        """An enemy mech with a laser, `charging: true`, aimed at us (urgent-cover trigger)."""
        a = self.add_mech(dist, bearing, weapons=("gun", "laser"), charging=True, pos=pos, id=id)
        if sound:
            ex = a["pos"]
            self.add_sound("laser_charge", yaw_to(ex[0] - self.pos[0], ex[1] - self.pos[1]), "mid" if dist > 60 else "near")
        return a

    def add_sound(self, kind, bearing, dist="mid", age=0.05):
        self.cur["sounds"].append({"kind": kind, "bearing": round(bearing, 3), "dist": dist, "age": age})
        return self

    def add_bullet(self, dist=60.0, bearing=0.0, speed=252.0, aimed_at_us=True, kind="gun"):
        p = self.at(dist, bearing)
        p[2] = self.pos[2] + 7.0
        d = math.hypot(self.pos[0] - p[0], self.pos[1] - p[1]) or 1.0
        ux, uy = (self.pos[0] - p[0]) / d, (self.pos[1] - p[1]) / d
        if not aimed_at_us:
            ux, uy = -uy, ux
        self.cur["projectiles"].append({"id": self.nid(), "kind": kind, "pos": p, "vel": [round(ux * speed, 2), round(uy * speed, 2), 0]})
        return self

    def add_shell(self, land_dx=0.0, land_dy=0.0, eta=1.2, shooter_dist=120.0, shooter_bearing=1.0):
        """A mortar shell that lands at (our pos + (land_dx, land_dy)) in `eta` s (real ballistics, gravity from config)."""
        gz = float(self.cfg["weapons"]["mortar"]["gravity"])
        lx, ly = self.pos[0] + land_dx, self.pos[1] + land_dy
        lz = self.g.z(lx, ly) if self.g.ok(lx, ly) else self.pos[2]
        sx, sy = self.pos[0] + shooter_dist * math.cos(shooter_bearing), self.pos[1] + shooter_dist * math.sin(shooter_bearing)
        p0 = [sx, sy, self.g.z(sx, sy) + 7.0]
        # the shell is in flight: place it so that `eta` seconds remain
        T = eta + 0.6
        v0 = [(lx - p0[0]) / T, (ly - p0[1]) / T, (lz - p0[2] + 0.5 * gz * T * T) / T]
        sid = self.nid()
        self._shells.append((sid, p0, v0, T, self.cur["t"] - 0.6))
        self.cur["projectiles"].append(self._shell_at(sid, p0, v0, 0.6, gz))
        return self

    @staticmethod
    def _shell_at(sid, p0, v0, age, gz):
        p = [p0[0] + v0[0] * age, p0[1] + v0[1] * age, p0[2] + v0[2] * age - 0.5 * gz * age * age]
        v = [v0[0], v0[1], v0[2] - gz * age]
        return {"id": sid, "kind": "mortar", "pos": [round(c, 2) for c in p], "vel": [round(c, 2) for c in v]}

    def damage(self, amount=90.0, weapon="gun", bearing=0.0, by=None, rear=False):
        self.cur["events"].append({"type": "damaged", "amount": amount, "weapon": weapon, "bearing": round(bearing, 3),
                                   "by": by, "rear": rear})
        return self

    def offer(self, ids=("armor", "laser_damage", "plating"), expires_in=5.0, offer_id=3):
        ups = {u["id"]: u for u in self.cfg["upgrades"]}
        self.cur["offer"] = {"id": offer_id, "expires": round(self.cur["t"] + expires_in, 2),
                             "choices": [{"id": i, "name": ups[i].get("name", i), "desc": ups[i].get("desc", i)} for i in ids]}
        return self

    def pickup(self, kind="laser", dist=2.5, bearing=0.0):
        p = {"id": self.nid(), "kind": kind, "pos": self.at(dist, bearing)}
        self.cur["pickups"].append(p)
        return p

    def crate(self, dist=60.0, bearing=0.0, hp=300):
        c = {"id": self.nid(), "pos": self.at(dist, bearing), "hp": hp}
        self.cur["crates"].append(c)
        return c

    def cube(self, dist=10.0, bearing=0.0):
        c = {"id": self.nid(), "pos": self.at(dist, bearing)}
        self.cur["cubes"].append(c)
        return c

    def drop(self, dist=120.0, bearing=0.0, lands_in=0, holder=None, progress=0, radius=8, capture_s=4):
        d = {"id": self.nid(), "pos": self.at(dist, bearing), "radius": radius, "lands_in": lands_in,
             "capture_s": capture_s, "progress": progress, "holder": holder}
        self.cur["drops"].append(d)
        return d

    # -- build / run
    def persist(self, ticks):
        """Repeat the injected tick `ticks` more times: entities keep their velocity, shells fly on, charging stays on,
        one-shot events/sounds are dropped."""
        self.follow = ticks
        return self

    def build(self):
        seq = list(self.seq)
        last = seq[-1]
        gz = float(self.cfg["weapons"]["mortar"]["gravity"])
        for k in range(1, self.follow + 1):
            o = copy.deepcopy(last)
            o["tick"] = last["tick"] + 2 * k
            o["t"] = round(o["tick"] / 20, 3)
            o["events"], o["sounds"], o["beams"] = [], [], []
            dt = OBS_DT * k
            for a in o["mechs"]:
                a["pos"] = [round(a["pos"][0] + a["vel"][0] * dt, 2), round(a["pos"][1] + a["vel"][1] * dt, 2), a["pos"][2]]
            keep = []
            for p in o["projectiles"]:
                shell = next((s for s in self._shells if s[0] == p["id"]), None)
                if shell:
                    age = o["t"] - shell[4]
                    if age < shell[3]:
                        keep.append(self._shell_at(shell[0], shell[1], shell[2], age, gz))
                else:
                    p["pos"] = [round(p["pos"][i] + p["vel"][i] * dt, 2) for i in range(3)]
                    keep.append(p)
            o["projectiles"] = keep
            seq.append(o)
        return seq

    def validate(self):
        sch = W.schema_validator() or W.MiniSchema()
        errs = []
        for o in self.build():
            errs += sch.errors(o, "Observation")
        return errs

    def make_bot(self):
        return make_bot(self.hello)

    def run_all(self, bot=None):
        """Feed the sequence to the real Bot; returns [action, ...] (one per observation) and the bot."""
        bot = bot or self.make_bot()
        acts = []
        for o in self.build():
            t0 = time.perf_counter()
            acts.append(bot.act(copy.deepcopy(o), t0))
        return acts, bot

    def run(self, bot=None):
        """Action for the last observation."""
        acts, _ = self.run_all(bot)
        return acts[-1]


def make_bot(hello=None, budget_s=15.0):
    """main.Bot(hello, deadline) with hello.map.dir pointing at the kit's arena."""
    import main
    if hello is None:
        hello = W.load_stream()[0]
    hello = copy.deepcopy(hello)
    hello["map"]["dir"] = W.ARENA
    return main.Bot(hello, time.perf_counter() + budget_s)


# ---------------------------------------------------------------------------- expected values from the plan tables
def gp_pick_index(choices_ids, held_kinds):
    """Index the GP order (config.UPGRADE_PRIORITY via config.loadout_key) says to pick; fallback list, then 0.
    Mirrors CONTRACTS.md safe.pick_upgrade so a bot deviating from the GP list is caught."""
    import config
    key = config.loadout_key(list(held_kinds))
    lists = getattr(config, "UPGRADE_PRIORITY", {})
    if key in ("GL", "GZ", "ZL"):
        order = list(lists.get(key, []))
    elif key == "LM":
        order = [u for u in lists.get("ZL", []) if u != "laser_damage"]
    else:
        order = list(lists.get("MECH", []))
    for uid in order:
        if uid in choices_ids:
            return list(choices_ids).index(uid)
    return 0


if __name__ == "__main__":
    # smoke: print the action for each canned scenario
    s = Scenario.quiet().weapons("gun", "laser")
    s.laser_at_us(70, 0.5)
    print("laser_at_us:", s.run())
