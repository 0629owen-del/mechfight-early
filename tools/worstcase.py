#!/usr/bin/env python3
"""Synthetic worst-case observation stream for the CPU gate (game-plan-v2 14.3 item 4).

    python3 tools/worstcase.py --n 600 --seed 1 --out /tmp/worst.jsonl --validate

Stdlib only. Every observation is schema-valid (checked with the kit's check.py `Schema`) and built
by mutating a real recorded observation (check/stream.jsonl.gz): same `self` block, same zone/config,
entities placed on real walkable map cells (mapdata nav grid) and moved along it tick by tick.

Per observation the stream carries, all the time:
  16 mechs + 12 NPCs visible, 20 bullets (about a third aimed at us), 6 mortar shells (a new one
  roughly every 0.7 s, landing close to us), sounds, beams, an upgrade offer (a new one every 5 s),
  loot (pickups / crates / cubes) in view and a supply drop far away.
It forces the expensive paths on a schedule:
  zone field rebuild   a new `zone.next_center/next_radius` every `zone_every` obs (self outside it)
  flood                a new pickup / crate / cube id enters view every `loot_every` obs
  A*                   a drop (and loot) 250-350 m away is always the best far goal; a new one every 150 obs
  urgent cover         every `laser_every` obs a mech 50-100 m away charges a laser at us for 0.8 s
                       (`charging: true`, aim at us, `laser_charge` sound, then a beam + `damaged` event)
  blind hit            `damaged` events from ids that are not in `mechs`
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
DOGMA = os.path.dirname(BOT)
KIT = os.path.join(DOGMA, "mechfight-kit")
STREAM = os.path.join(KIT, "check", "stream.jsonl.gz")
SCHEMA = os.path.join(KIT, "mechfight.schema.json")
ARENA = os.path.join(KIT, "arena")
for _p in (BOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OBS_DT = 0.1  # decide_every 2 at 20 Hz


# ---------------------------------------------------------------------------- shared loaders
def load_stream(path=STREAM):
    """(hello, [obs...], end) from the recorded stream. hello.map.dir is made absolute."""
    with gzip.open(path, "rt") as f:
        msgs = [json.loads(l) for l in f if l.strip()]
    hello = msgs[0]
    hello["map"]["dir"] = ARENA
    obs = [m for m in msgs if m.get("type") == "obs"]
    return hello, obs, msgs[-1]


_MAP = None


def load_map(map_dir=ARENA):
    """mapdata.MapData (cached). Only used for walkable cells and heights."""
    global _MAP
    if _MAP is None:
        import mapdata
        _MAP = mapdata.MapData.load(map_dir)
    return _MAP


def schema_validator(schema_path=SCHEMA):
    """check.py's Schema if importable, else None (callers fall back to `MiniSchema`)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mf_check", os.path.join(KIT, "check", "check.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Schema(schema_path)
    except Exception:
        return None


class MiniSchema:
    """Minimal fallback (types / required / enums for Observation) when check.py is not importable."""

    def __init__(self, path=SCHEMA):
        with open(path) as f:
            self.root = json.load(f)

    def errors(self, value, name):
        out = []
        self._c(value, {"$ref": "#/definitions/" + name}, "$", out)
        return out[:6]

    def _c(self, v, s, at, out):
        if "$ref" in s:
            return self._c(v, self.root["definitions"][s["$ref"].split("/")[-1]], at, out)
        if "anyOf" in s:
            if not any(not self._sub(v, a, at) for a in s["anyOf"]):
                out.append(f"{at}: matches none of the allowed shapes")
            return
        t = s.get("type")
        if t is not None:
            ts = t if isinstance(t, list) else [t]
            ok = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
            good = False
            for x in ts:
                if x in ("number", "integer"):
                    good |= isinstance(v, (int, float)) and not isinstance(v, bool)
                else:
                    good |= isinstance(v, ok[x])
            if not good:
                out.append(f"{at}: bad type {type(v).__name__}")
                return
        if "const" in s and v != s["const"]:
            out.append(f"{at}: expected {s['const']!r}")
        if "enum" in s and v not in s["enum"]:
            out.append(f"{at}: not in enum")
        if isinstance(v, dict):
            for k in s.get("required", []):
                if k not in v:
                    out.append(f"{at}: missing {k!r}")
            for k, x in v.items():
                if k in s.get("properties", {}):
                    self._c(x, s["properties"][k], f"{at}.{k}", out)
        if isinstance(v, list) and isinstance(s.get("items"), dict):
            for i, x in enumerate(v):
                self._c(x, s["items"], f"{at}[{i}]", out)

    def _sub(self, v, s, at):
        out = []
        self._c(v, s, at, out)
        return out


# ---------------------------------------------------------------------------- map helpers
class Ground:
    """Random walkable positions near a point, on the real nav grid."""

    def __init__(self, md, rng):
        self.md, self.rng = md, rng
        b = md.bounds
        self.bounds = (b[0], b[1], b[2], b[3])  # minX, minY, maxX, maxY

    def ok(self, x, y):
        c = self.md.cell4(x, y)
        return c >= 0 and self.md.walkable(c)

    def z(self, x, y):
        c = self.md.cell4(x, y)
        return self.md.nav_h(c) if c >= 0 else 0.0

    def near(self, cx, cy, rmin, rmax, tries=200, within=None):
        """Walkable (x, y, z) with rmin <= dist <= rmax from (cx, cy); `within`=(zx, zy, zr) keeps it in a circle."""
        r = self.rng
        for _ in range(tries):
            a = r.uniform(-math.pi, math.pi)
            d = r.uniform(rmin, rmax)
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            if within and math.hypot(x - within[0], y - within[1]) > within[2]:
                continue
            if self.ok(x, y):
                return x, y, self.z(x, y)
        return None

    def anywhere(self, zone_c, zone_r):
        return self.near(zone_c[0], zone_c[1], 0, zone_r * 0.8, tries=2000)

    def step(self, x, y, heading, dist):
        """Move `dist` along `heading`, turning away from blocked cells. Returns (x, y, z, heading)."""
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 6):
            h = heading + k * 0.4
            nx, ny = x + dist * math.cos(h), y + dist * math.sin(h)
            if self.ok(nx, ny) and abs(self.z(nx, ny) - self.z(x, y)) < 3.0:
                return nx, ny, self.z(nx, ny), h
        return x, y, self.z(x, y), heading + math.pi / 2


def _r2(v):
    return round(v, 2)


def _vec3(x, y, z):
    return [_r2(x), _r2(y), _r2(z)]


# ---------------------------------------------------------------------------- generator
def laser_weapon_slot(hello):
    st = dict(hello["config"]["weapons"]["laser"])
    st = {k: st[k] for k in ("damage", "cooldown", "range", "pierce", "charge") if k in st}
    return {"kind": "laser", "cooldown": 0, "stats": st, "charge": 0}


class WorstCase:
    def __init__(self, n=600, seed=1, zone_every=25, loot_every=20, laser_every=30, hello=None,
                 n_mechs=16, n_npcs=12, n_bullets=20, n_shells=6, start_t=65.0):
        self.hello_stream, self.rec, _ = load_stream()
        self.hello = hello or self.hello_stream
        self.cfg = self.hello["config"]
        self.rng = random.Random(seed)
        self.g = Ground(load_map(), self.rng)
        self.n, self.zone_every, self.loot_every, self.laser_every = n, zone_every, loot_every, laser_every
        self.n_mechs, self.n_npcs, self.n_bullets, self.n_shells = n_mechs, n_npcs, n_bullets, n_shells
        self.start_t = start_t
        self.template = json.loads(json.dumps(self.rec[min(len(self.rec) - 1, 60)]))
        self.upgrade_ids = [u["id"] for u in self.cfg["upgrades"]]
        self.upgrades = {u["id"]: u for u in self.cfg["upgrades"]}
        z = self.cfg["zone"]
        self.zone_c = list(z["start_center"])
        self.zone_r = float(z["start_radius"])
        self._id = 10000

    # -- ids
    def nid(self):
        self._id += 1
        return self._id

    def generate(self):
        rng, g = self.rng, self.g
        # self: start well inside the circle, walk toward a far walkable goal
        sp = g.near(self.zone_c[0], self.zone_c[1], 0, self.zone_r * 0.5, tries=5000)
        sx, sy, sz = sp
        goal = g.near(sx, sy, 250, 350, tries=5000, within=(self.zone_c[0], self.zone_c[1], self.zone_r * 0.8)) or (sx + 200, sy, sz)
        heading = math.atan2(goal[1] - sy, goal[0] - sx)
        last_p = (sx, sy)
        me = self.template["self"]
        me["weapons"] = [me["weapons"][0], laser_weapon_slot(self.hello)]
        me["cubes"] = 4
        me["upgrades"] = []
        me["hp"] = me["max_hp"] = 1400

        def around(rmin, rmax, within_zone=True):
            p = g.near(sx, sy, rmin, rmax, within=(self.zone_c[0], self.zone_c[1], self.zone_r * 0.9) if within_zone else None)
            return p or (sx + rmin, sy, g.z(sx + rmin, sy))

        mechs = []
        for i in range(self.n_mechs):
            x, y, z = around(40, 190)
            wp = [["gun", "laser"], ["gun", "mortar"], ["gun", "lightning"], ["laser", "lightning"]][i % 4]
            mechs.append({"kind": "mech", "id": 100 + i, "x": x, "y": y, "z": z, "h": rng.uniform(-math.pi, math.pi),
                          "hp": rng.uniform(300, 1400), "cubes": rng.randint(0, 6), "weapons": wp, "charge_until": -1.0})
        npcs = []
        for i in range(self.n_npcs):
            x, y, z = around(30, 150)
            npcs.append({"kind": "npc", "id": 2000 + i, "x": x, "y": y, "z": z, "h": rng.uniform(-math.pi, math.pi),
                         "hp": 350.0})
        bullets, shells = [], []
        loot = {"pickups": [], "crates": [], "cubes": []}
        drops = []
        offer, offer_until = None, -1
        zone = {"center": list(self.zone_c), "radius": self.zone_r}
        next_zone = None
        recent_events = []
        out = []
        tick = 0
        speed = 11.7
        for i in range(self.n):
            t = round(self.start_t + i * OBS_DT, 3)
            tick = int(round(t * 20)) if i == 0 else tick + 2
            # ---- self walks toward the far goal
            gh = math.atan2(goal[1] - sy, goal[0] - sx)
            if math.hypot(goal[0] - sx, goal[1] - sy) < 40:
                goal = g.near(sx, sy, 250, 350, tries=5000, within=(self.zone_c[0], self.zone_c[1], self.zone_r * 0.8)) or goal
                gh = math.atan2(goal[1] - sy, goal[0] - sx)
            if i % 20 == 0:
                if math.hypot(sx - last_p[0], sy - last_p[1]) < 8.0 and i:
                    goal = g.near(sx, sy, 250, 350, tries=5000, within=(self.zone_c[0], self.zone_c[1], self.zone_r * 0.8)) or goal
                    gh = math.atan2(goal[1] - sy, goal[0] - sx)
                last_p = (sx, sy)
            sx, sy, sz, heading = g.step(sx, sy, gh, speed * OBS_DT)
            svx, svy = speed * math.cos(heading), speed * math.sin(heading)
            # ---- zone: a new next circle every zone_every obs, self outside it
            if i % self.zone_every == 0:
                nr = rng.uniform(150, 380)
                ang = rng.uniform(-math.pi, math.pi)
                off = min(self.zone_r - nr - 5, rng.uniform(100, 300))
                ncx = self.zone_c[0] + off * 0.6 * math.cos(ang)
                ncy = self.zone_c[1] + off * 0.6 * math.sin(ang)
                next_zone = {"next_center": [_r2(ncx), _r2(ncy)], "next_radius": _r2(nr)}
            zone = {"center": [_r2(self.zone_c[0]), _r2(self.zone_c[1])], "radius": _r2(self.zone_r),
                    **next_zone, "shrink_start": _r2(t + 12), "shrink_end": _r2(t + 62), "dps": 20}
            # ---- mechs and npcs wander on walkable cells, recycled into view when they leave it
            for a in mechs + npcs:
                spd = 10.0 if a["kind"] == "mech" else 5.0
                if math.hypot(a["x"] - sx, a["y"] - sy) > 195 or rng.random() < 0.02:
                    if math.hypot(a["x"] - sx, a["y"] - sy) > 195:
                        a["x"], a["y"], a["z"] = around(60, 170)
                    a["h"] = rng.uniform(-math.pi, math.pi)
                a["x"], a["y"], a["z"], a["h"] = g.step(a["x"], a["y"], a["h"], spd * OBS_DT)
                a["vx"], a["vy"] = spd * math.cos(a["h"]), spd * math.sin(a["h"])
            # ---- laser charging at us
            if i % self.laser_every == 0:
                m = mechs[(i // self.laser_every) % len(mechs)]
                m["x"], m["y"], m["z"] = around(50, 100)
                m["charge_until"] = t + 0.8
                m["weapons"] = ["gun", "laser"]
                recent_events.append(("laser_charge", m["id"], t))
            events, sounds, beams = [], [], []
            for m in mechs:
                if m["charge_until"] > t:
                    sounds.append({"kind": "laser_charge", "bearing": _r2(math.atan2(m["y"] - sy, m["x"] - sx) + rng.gauss(0, 0.12)),
                                   "dist": "near", "age": 0.0})
                elif 0 < t - m["charge_until"] < OBS_DT * 0.99 and m["charge_until"] > 0:
                    beams.append({"by": m["id"], "kind": "laser", "age": 0.05,
                                  "path": [_vec3(m["x"], m["y"], m["z"] + 7), _vec3(sx, sy, sz + 4.5)]})
                    events.append({"type": "damaged", "amount": 315, "weapon": "laser",
                                   "bearing": _r2(math.atan2(m["y"] - sy, m["x"] - sx)), "by": m["id"], "rear": False})
            if i % 37 == 11:  # blind hit: attacker id not visible
                events.append({"type": "damaged", "amount": 45, "weapon": "gun", "bearing": _r2(rng.uniform(-3, 3)),
                               "by": 900 + i, "rear": True})
            if i % 17 == 5:
                events.append({"type": "hit", "target": mechs[i % len(mechs)]["id"], "amount": 45.0, "weapon": "gun", "rear": False})
            # ---- bullets (kept at n_bullets, spawned mid-flight, ~40% aimed at us)
            alive = []
            for b in bullets:
                b["life"] -= OBS_DT
                if b["life"] > 0:
                    for k in range(3):
                        b["p"][k] += b["v"][k] * OBS_DT
                    alive.append(b)
            bullets = alive
            while len(bullets) < self.n_bullets:
                a = rng.choice(mechs if rng.random() < 0.6 else npcs)
                kind, spd = ("gun", 252.0) if a["kind"] == "mech" else ("npc", 90.0)
                tx, ty = sx, sy
                if rng.random() > 0.4:
                    tx, ty = sx + rng.uniform(-30, 30), sy + rng.uniform(-30, 30)
                d = max(10.0, math.hypot(tx - a["x"], ty - a["y"]))
                ux, uy = (tx - a["x"]) / d, (ty - a["y"]) / d
                frac = rng.uniform(0.0, 0.9)
                bullets.append({"id": self.nid(), "kind": kind, "life": max(0.1, min(d, 230) / spd * (1 - frac)),
                                "p": [a["x"] + ux * d * frac, a["y"] + uy * d * frac, a["z"] + 7 - 2 * frac],
                                "v": [ux * spd, uy * spd, -3.0]})
            # ---- mortar shells: a new one lands near us roughly every 0.7 s
            g_ = float(self.cfg["weapons"]["mortar"]["gravity"])
            alive = []
            for s in shells:
                s["age"] += OBS_DT
                if s["age"] < s["T"]:
                    alive.append(s)
                else:
                    sounds.append({"kind": "explosion", "bearing": _r2(math.atan2(s["land"][1] - sy, s["land"][0] - sx)),
                                   "dist": "near", "age": 0.0})
            shells = alive
            while len(shells) < self.n_shells:
                a = rng.choice(mechs)
                lx, ly = sx + rng.uniform(-14, 14) + svx * 1.5, sy + rng.uniform(-14, 14) + svy * 1.5
                lz = g.z(lx, ly) if g.ok(lx, ly) else sz
                T = rng.uniform(1.5, 4.5)
                age = rng.uniform(0.0, 0.7) if len(out) == 0 else 0.0
                p0 = [a["x"], a["y"], a["z"] + 7]
                v0 = [(lx - p0[0]) / T, (ly - p0[1]) / T, (lz - p0[2] + 0.5 * g_ * T * T) / T]
                shells.append({"id": self.nid(), "T": T, "age": age, "p0": p0, "v0": v0, "land": (lx, ly, lz)})
                if age == 0.0:
                    sounds.append({"kind": "mortar", "bearing": _r2(math.atan2(a["y"] - sy, a["x"] - sx)), "dist": "mid", "age": 0.0})
            # ---- loot: a new id enters view every loot_every obs (flood restart), old ones leave
            if i % self.loot_every == 0:
                kind = rng.choice(["laser", "mortar", "lightning", "gun"])
                x, y, z = around(40, 150)
                loot["pickups"].append({"id": self.nid(), "kind": kind, "x": x, "y": y, "z": z, "born": i})
                x, y, z = around(40, 150)
                loot["crates"].append({"id": self.nid(), "x": x, "y": y, "z": z, "born": i})
                x, y, z = around(20, 120)
                loot["cubes"].append({"id": self.nid(), "x": x, "y": y, "z": z, "born": i})
            for k, cap in (("pickups", 8), ("crates", 5), ("cubes", 6)):
                loot[k] = [e for e in loot[k] if math.hypot(e["x"] - sx, e["y"] - sy) < 200][-cap:]
            # ---- supply drop far away (A* goal); a new one every 150 obs
            if i % 150 == 0:
                x, y, z = g.near(sx, sy, 250, 350, tries=5000, within=(self.zone_c[0], self.zone_c[1], self.zone_r * 0.8)) or (sx + 280, sy, sz)
                drops = [{"id": self.nid(), "pos": _vec3(x, y, z), "radius": 8, "lands_in": 3.0 if i == 0 else 0,
                          "capture_s": 4, "progress": 0, "holder": None}]
            elif drops and drops[0]["lands_in"] > 0:
                drops[0]["lands_in"] = max(0, round(drops[0]["lands_in"] - OBS_DT, 2))
            # ---- offer: new id every 5 s
            if offer is None or t >= offer_until:
                ids = rng.sample(self.upgrade_ids, 3)
                offer = {"id": len(out) // 50 + 1, "expires": _r2(t + 5.0),
                         "choices": [{"id": u, "name": self.upgrades[u].get("name", u), "desc": self.upgrades[u].get("desc", u)} for u in ids]}
                offer_until = t + 5.0
            # ---- sounds: a few random far/mid ones
            for _ in range(rng.randint(1, 3)):
                kind = rng.choice(["gun", "step", "reload", "boost", "jet", "npc_gun", "laser", "lightning", "crate"])
                sounds.append({"kind": kind, "bearing": _r2(rng.uniform(-math.pi, math.pi)),
                               "dist": rng.choice(["near", "mid", "far"]), "age": _r2(rng.uniform(0, 0.1))})
            # ---- self block
            me["pos"] = _vec3(sx, sy, sz)
            me["vel"] = [_r2(svx), _r2(svy), 0]
            me["aim"] = [_r2(heading), 0]
            me["hp"] = max(200, 1400 - 20 * (i % 60))
            me["grounded"] = True
            me["in_zone"] = math.hypot(sx - zone["center"][0], sy - zone["center"][1]) < zone["radius"]
            obs = {
                "type": "obs", "tick": tick, "t": t, "alive": 16, "self": json.loads(json.dumps(me)),
                "mechs": [self._actor(a, t, sx, sy) for a in mechs + npcs],
                "projectiles": [{"id": b["id"], "kind": b["kind"], "pos": _vec3(*b["p"]), "vel": _vec3(*b["v"])} for b in bullets]
                + [self._shell(s) for s in shells],
                "crates": [{"id": c["id"], "pos": _vec3(c["x"], c["y"], c["z"]), "hp": 300} for c in loot["crates"]],
                "cubes": [{"id": c["id"], "pos": _vec3(c["x"], c["y"], c["z"])} for c in loot["cubes"]],
                "pickups": [{"id": p["id"], "kind": p["kind"], "pos": _vec3(p["x"], p["y"], p["z"])} for p in loot["pickups"]],
                "sounds": sounds, "beams": beams, "events": events, "zone": zone,
                "offer": json.loads(json.dumps(offer)), "drops": json.loads(json.dumps(drops)),
            }
            out.append(obs)
        return out

    @staticmethod
    def _actor(a, t, sx, sy):
        d = {"id": a["id"], "kind": a["kind"], "pos": _vec3(a["x"], a["y"], a["z"]), "vel": [_r2(a["vx"]), _r2(a["vy"]), 0],
             "aim": [_r2(a["h"]), 0], "hp": _r2(a["hp"]), "max_hp": 1400 if a["kind"] == "mech" else 350}
        if a["kind"] == "mech":
            d["cubes"] = a["cubes"]
            d["weapons"] = list(a["weapons"])
            d["charging"] = a["charge_until"] > t
            if d["charging"]:  # telegraph: aimed at us
                d["aim"] = [_r2(math.atan2(sy - a["y"], sx - a["x"])), 0]
        return d

    def _shell(self, s):
        g_ = float(self.cfg["weapons"]["mortar"]["gravity"])
        t = s["age"]
        p = [s["p0"][0] + s["v0"][0] * t, s["p0"][1] + s["v0"][1] * t, s["p0"][2] + s["v0"][2] * t - 0.5 * g_ * t * t]
        v = [s["v0"][0], s["v0"][1], s["v0"][2] - g_ * t]
        return {"id": s["id"], "kind": "mortar", "pos": _vec3(*p), "vel": _vec3(*v)}


def generate(n=600, seed=1, **kw):
    """List of `n` schema-valid worst-case observations (deterministic for a seed)."""
    wc = WorstCase(n=n, seed=seed, **kw)
    return wc.generate()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=None, help="observations (default WORSTCASE_TICKS)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write hello + obs as JSONL (gz if it ends with .gz)")
    ap.add_argument("--validate", action="store_true", help="check every observation against mechfight.schema.json")
    a = ap.parse_args()
    n = a.n
    if n is None:
        import config
        n = config.DEFAULTS.get("WORSTCASE_TICKS", 600)
    t0 = time.perf_counter()
    obs = generate(n, a.seed)
    print(f"generated {len(obs)} observations in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    if a.validate:
        sch = schema_validator() or MiniSchema()
        bad = 0
        for o in obs:
            e = sch.errors(o, "Observation")
            if e:
                bad += 1
                if bad <= 5:
                    print(f"tick {o['tick']}: {e[:3]}", file=sys.stderr)
        print(f"schema: {len(obs) - bad}/{len(obs)} valid", file=sys.stderr)
        if bad:
            return 1
    if a.out:
        hello, _, end = load_stream()
        op = gzip.open if a.out.endswith(".gz") else open
        with op(a.out, "wt") as f:
            for m in [hello] + obs + [end]:
                f.write(json.dumps(m, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
