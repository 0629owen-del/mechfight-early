#!/usr/bin/env python3
"""minisim: a local, APPROXIMATE match server for mechfight/1 (stdlib only).

    python3 tools/minisim.py --out runs/m1 --seed 7 --bot "python3 -u main.py" --fill scripted:mix
    python3 tools/minisim.py --out runs/m2 --bot example --fill scripted:duelist --duration 60

It launches one subprocess per slot, speaks the exact protocol (hello / ready / lockstep obs and
actions at 10 decisions per second, 2 ticks each, the 50 ms budget, coast/idle, newest-observation
tick echo, end with place and kills), simulates the rules of protocol.md sections 5-10 as closely as
is cheap in pure Python, and writes:

    <out>/match.jsonl        ground-truth event log (shots, sightings, fights, deaths, placements)
    <out>/slotNN.stderr      every stderr line of the bot in slot NN, verbatim (bot telemetry)
    <out>/summary.json       placements and per-slot counters

Bot specs (for --bot / --fill): a shell-style command line, or
    example              the kit's python/example_bot.py with the kit's python/ on PYTHONPATH
    scripted:<arch>      tools/minisim_bots.py <arch>; arch in duelist, rusher, lobber, looter, mix
                         (`mix` cycles a built-in-like blend, weighted towards gun+laser duelists);
                         calibrated engagement by default, `scripted:<arch>_v1` = original fire-on-sight

APPROXIMATIONS (also in tools/MINISIM.md). "exact" = implemented as protocol.md states it.

| Area | Model here | Real server (protocol) |
| --- | --- | --- |
| Timing | lockstep, 2 ticks/decision, 50 ms budget per decision (wall clock, all bots in parallel), coast 1 s then idle; late/stale answers dropped | exact rules; server-side timing details unknown |
| Geometry | prisms rasterised on the 1 m terrain grid (cell centres); per cell: a solid column ground..top plus overhang intervals (bridges); thin features < 1 m may leak | exact polygons |
| Terrain | bilinear int16 heights for feet; nearest-vertex height for shots/LOS | bilinear |
| Mech collision | 8 (12 when big) perimeter samples + centre vs the cell grid; wall slide = axis split then +-30/60 deg deflection; step-up 1.2 m; slope limit on terrain rise at the centre | exact cylinder sweep |
| Eaves / roof edges | perimeter samples can support you when landing (edge landing); slid off at 0.6 m/tick unless moving onto the roof; ceilings stop jets | as protocol, exact |
| Mech overlap | pairwise horizontal push-apart, half each (a mech a wall holds is not moved); positions only: the velocity state used for integration is untouched. Reported `vel` (self and seen mechs, x/y): `--push-vel actual` (default) = the displacement the mech really made in the last tick / dt, i.e. state + push-apart + eave slide; `--push-vel keep` = the state alone (reads full speed while pinned against a mech). Same actions give identical positions under both; only `vel` differs | "overlapping mechs are pushed apart"; whether `vel` shows it is unknown (hence the option) |
| Fall damage | none (A-1) | unstated |
| Gun | projectile, Gaussian spread sigma = spread_deg on yaw and pitch, exact fire-rate accumulator, mag, auto reload at 0 | exact |
| Laser | charge/min_charge/hold_s/cooldown/half speed/cook-off/pierce, hitscan vs cylinders and the grid | exact |
| Lightning | nearest target (mech, NPC, crate) by 3D distance from the muzzle within the cone (default +-cone_deg, `--cone half` for +-cone_deg/2: A-19), LOS through buildings only, chain to nearest unhit within chain_range | exact (cone reading unknown) |
| Mortar | ballistic shell, direct hit = damage (splash too only with `--mortar-stack`, A-2), splash linear falloff to splash_edge, distance to the target's cylinder, blocked by walls (LOS from blast), knock/knock_up/drain scaled, self 50% | exact (stacking unknown) |
| Rear armor | source direction (bullet origin / beam or arc origin / blast point) outside +-front_arc/2 of the target's aim yaw -> x rear_mult; NPC bullets too | exact |
| Vision | 45 m near (buildings block, canopy not) / 200 m 120 deg wedge with canopy; eye -> centre or top; LOS march every 1.5 m on the cell grid; canopy = crowns as cylinders, transmittance exp(-canopyDensity*metres) < canopyBlockT blocks; trunks ignored; item LOS cached 0.5 s per observer (or until it moves 4 m); projectiles need range/wedge + LOS | exact LOS, crown shapes unknown |
| Beams | shown when the start, end or a segment midpoint passes the vision test | same rule |
| Sounds | kinds, ranges, near/mid/far, age, Gaussian bearing noise x mech.hearing; steps every step_every while grounded > step_speed; jet every 0.3 s | exact |
| Zone | phases from config; obs reports the phase-start circle + next + window (live circle lerped); next centres uniform inside the allowed disc, kept inside bounds | centre picking unknown |
| NPCs | straight-line wander/chase/return, no pathfinding; aggro on attack or LOS within aggro_range while within leash of home; bullets aimed at centre (no lead) | pathing/aim unknown |
| Drops | landing point uniform within spread x next_radius of the next centre on open ground; ring = horizontal radius and feet within -3..+10 m; NPCs don't count (A-6, `--drop-npc-blocks`) | unknown details |
| Loot spawns | uniform on open ground inside bounds and start circle; roofs = random building prism tops <= 55 m high | unknown |
| Placement | death order, same-tick deaths share `alive` before the tick; clock end ranks survivors by HP (ties shared); `end` sent at death | exact |
| Zone damage events | one aggregated `damaged` (weapon zone) per observation | unknown granularity |
| Kill credit | the mech whose damage took HP to 0 (own mortar: none) | unknown |
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import queue
import random
import shlex
import subprocess
import sys
import threading
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
KIT_DEFAULT = os.path.join(os.path.dirname(REPO), "mechfight-kit")

NONE_H = -1.0e9
TAU = 2.0 * math.pi
WEAPON_KINDS = ("gun", "mortar", "laser", "lightning")
PUSH_VEL_MODES = ("actual", "keep")     # --push-vel, see World.vel_h
MIX = ["duelist", "duelist", "lobber", "duelist", "rusher", "looter", "duelist", "rusher",
       "duelist", "lobber", "duelist", "looter", "duelist", "duelist", "rusher", "duelist"]


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def r2(v):
    return round(v, 2)


def load_hello(kit):
    """The recorded hello (real config defaults) from the kit's check stream."""
    path = os.path.join(kit, "check", "stream.jsonl.gz")
    with gzip.open(path, "rt") as f:
        return json.loads(f.readline())


# ============================================================================ geometry
class Geo:
    """Map model on the 1 m terrain vertex grid (cells centred on vertices)."""

    LOS_STEP = 1.5

    def __init__(self, arena_dir=None, flat_bounds=500.0):
        if arena_dir is None:
            self._flat(flat_bounds)
            return
        with open(os.path.join(arena_dir, "v2", "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        self.nx, self.ny = int(t["nx"]), int(t["ny"])
        self.minX, self.minY = float(t["minX"]), float(t["minY"])
        b = w["bounds"]
        self.bounds = (b["minX"], b["minY"], b["maxX"], b["maxY"])
        raw = array("h")
        with open(os.path.join(arena_dir, "v2", t.get("file", "terrain.i16")), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        sc = float(t.get("scale", 0.01))
        self.th = [v * sc for v in raw]
        n = self.nx * self.ny
        self.ctop = [NONE_H] * n
        self.hasov = bytearray(n)
        self.over = {}
        self.hascan = bytearray(n)
        self.canopy = {}
        rules = w.get("rules", {})
        self.canopy_density = float(rules.get("canopyDensity", 0.35))
        self.canopy_block_t = float(rules.get("canopyBlockT", 0.25))
        self.prisms = w["prisms"]
        self._raster_prisms(self.prisms)
        tr = w.get("trees", {})
        if tr.get("data"):
            self._raster_trees(tr["fields"], tr["data"])
        self.surf = [max(a, c) for a, c in zip(self.th, self.ctop)]
        self.roofs = self._roof_candidates()

    def _flat(self, half):
        self.nx = self.ny = int(2 * half) + 1
        self.minX = self.minY = -half
        self.bounds = (-half, -half, half, half)
        n = self.nx * self.ny
        self.th = [0.0] * n
        self.ctop = [NONE_H] * n
        self.surf = [0.0] * n
        self.hasov = bytearray(n)
        self.over = {}
        self.hascan = bytearray(n)
        self.canopy = {}
        self.canopy_density, self.canopy_block_t = 0.35, 0.25
        self.prisms = []
        self.roofs = []

    # -- rasterisation
    def _spans(self, ring, holes):
        """Yield (j, i0, i1) cell spans inside ring and outside holes (even-odd scanline)."""
        polys = [ring] + list(holes or [])
        edges = []
        ys = []
        for p in polys:
            m = len(p) // 2
            for k in range(m):
                x0, y0 = p[2 * k], p[2 * k + 1]
                x1, y1 = p[2 * ((k + 1) % m)], p[2 * ((k + 1) % m) + 1]
                if y0 != y1:
                    edges.append((x0, y0, x1, y1))
            ys.extend(p[1::2])
        if not edges:
            return
        j0 = max(0, int(math.ceil(min(ys) - self.minY)))
        j1 = min(self.ny - 1, int(math.floor(max(ys) - self.minY)))
        for j in range(j0, j1 + 1):
            y = self.minY + j
            xs = []
            for x0, y0, x1, y1 in edges:
                if (y0 <= y < y1) or (y1 <= y < y0):
                    xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil(xs[k] - self.minX)))
                i1 = min(self.nx - 1, int(math.floor(xs[k + 1] - self.minX)))
                if i0 <= i1:
                    yield j, i0, i1

    def _raster_prisms(self, prisms):
        cells = {}
        nx = self.nx
        for p in prisms:
            iv = (float(p["base"]), float(p["top"]))
            for j, i0, i1 in self._spans(p["ring"], p.get("holes")):
                row = j * nx
                for i in range(i0, i1 + 1):
                    idx = row + i
                    c = cells.get(idx)
                    if c is None:
                        cells[idx] = [iv]
                    else:
                        c.append(iv)
        th, ctop = self.th, self.ctop
        for idx, ivs in cells.items():
            g = th[idx]
            ivs.sort()
            merged = []
            for b, t in ivs:
                if merged and b <= merged[-1][1] + 0.5:
                    if t > merged[-1][1]:
                        merged[-1][1] = t
                else:
                    merged.append([b, t])
            rest = []
            col = NONE_H
            for b, t in merged:      # sorted by base: anything resting on the column extends it
                if t <= g:
                    continue
                if b <= g + 1.0 or (col > NONE_H and b <= col + 0.5):
                    col = max(col, t)
                else:
                    rest.append((b, t))
            ctop[idx] = col
            if rest:
                self.hasov[idx] = 1
                self.over[idx] = tuple(rest)

    def _raster_trees(self, fields, data):
        f = {k: i for i, k in enumerate(fields)}
        n = len(fields)
        nx, ny = self.nx, self.ny
        for k in range(0, len(data) - n + 1, n):
            row = data[k:k + n]
            z = row[f["z"]]
            cx = row[f["crownX"]] if "crownX" in f else row[f["x"]]
            cy = row[f["crownY"]] if "crownY" in f else row[f["y"]]
            cr = row[f["crownR"]]
            cb = z + row[f["crownBase"]]
            ct = z + row[f["height"]]
            i0 = max(0, int(math.ceil(cx - cr - self.minX)))
            i1 = min(nx - 1, int(math.floor(cx + cr - self.minX)))
            j0 = max(0, int(math.ceil(cy - cr - self.minY)))
            j1 = min(ny - 1, int(math.floor(cy + cr - self.minY)))
            r2_ = cr * cr
            for j in range(j0, j1 + 1):
                dy = self.minY + j - cy
                for i in range(i0, i1 + 1):
                    dx = self.minX + i - cx
                    if dx * dx + dy * dy <= r2_:
                        idx = j * nx + i
                        old = self.canopy.get(idx)
                        self.canopy[idx] = (min(cb, old[0]), max(ct, old[1])) if old else (cb, ct)
                        self.hascan[idx] = 1

    def _roof_candidates(self):
        out = []
        for p in self.prisms:
            if p["kind"] != "building":
                continue
            ring = p["ring"]
            xs, ys = ring[0::2], ring[1::2]
            area = 0.0
            m = len(xs)
            for k in range(m):
                area += xs[k] * ys[(k + 1) % m] - xs[(k + 1) % m] * ys[k]
            area = abs(area) / 2
            g = self.ground(sum(xs) / m, sum(ys) / m)
            if area >= 120 and p["top"] - g <= 55 and p["top"] - g >= 4:
                out.append((area, p, (min(xs), min(ys), max(xs), max(ys))))
        return out

    # -- queries
    def idx(self, x, y):
        i = int(x - self.minX + 0.5)
        j = int(y - self.minY + 0.5)
        if i < 0:
            i = 0
        elif i >= self.nx:
            i = self.nx - 1
        if j < 0:
            j = 0
        elif j >= self.ny:
            j = self.ny - 1
        return j * self.nx + i

    def ground(self, x, y):
        fx = x - self.minX
        fy = y - self.minY
        i = int(fx)
        j = int(fy)
        if i < 0:
            i, fx = 0, 0.0
        elif i >= self.nx - 1:
            i, fx = self.nx - 2, float(self.nx - 1)
        if j < 0:
            j, fy = 0, 0.0
        elif j >= self.ny - 1:
            j, fy = self.ny - 2, float(self.ny - 1)
        ax, ay = fx - i, fy - j
        th = self.th
        k = j * self.nx + i
        h0 = th[k] + (th[k + 1] - th[k]) * ax
        h1 = th[k + self.nx] + (th[k + self.nx + 1] - th[k + self.nx]) * ax
        return h0 + (h1 - h0) * ay

    def floor_at(self, x, y, zmax):
        """Highest walkable surface under (x, y) whose top is <= zmax. Returns (height, is_terrain)."""
        f = self.ground(x, y)
        terr = True
        k = self.idx(x, y)
        c = self.ctop[k]
        if f < c <= zmax:
            f, terr = c, False
        if self.hasov[k]:
            for b, t in self.over[k]:
                if f < t <= zmax:
                    f, terr = t, False
        return f, terr

    def roof_top(self, x, y, zmax):
        """Highest prism top (not terrain) under (x, y) that is <= zmax, or NONE_H."""
        k = self.idx(x, y)
        best = NONE_H
        c = self.ctop[k]
        if c <= zmax:
            best = c
        if self.hasov[k]:
            for b, t in self.over[k]:
                if best < t <= zmax:
                    best = t
        return best

    def wall_hit(self, x, y, zlo, zhi):
        """Any solid in cell (x, y) overlapping the open interval (zlo, zhi)?"""
        k = self.idx(x, y)
        if self.ctop[k] > zlo:
            return True
        if self.hasov[k]:
            for b, t in self.over[k]:
                if b < zhi and t > zlo:
                    return True
        return False

    def ceiling(self, x, y, zlo, zhi):
        """Lowest overhang base in (zlo, zhi], or None."""
        k = self.idx(x, y)
        best = None
        if self.hasov[k]:
            for b, t in self.over[k]:
                if zlo < b <= zhi and (best is None or b < best):
                    best = b
        return best

    def solid(self, x, y, z):
        k = self.idx(x, y)
        if z < self.surf[k]:
            return True
        if self.hasov[k]:
            for b, t in self.over[k]:
                if b <= z <= t:
                    return True
        return False

    def open_ground(self, x, y):
        k = self.idx(x, y)
        return self.ctop[k] == NONE_H and not self.hasov[k]

    def in_bounds(self, x, y, margin=0.0):
        b = self.bounds
        return b[0] + margin <= x <= b[2] - margin and b[1] + margin <= y <= b[3] - margin

    def seg_hit(self, x0, y0, z0, x1, y1, z1, step=1.0):
        """First parameter in (0, 1] where the segment enters solid geometry, or None."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(L / step) + 1
        surf, hasov, over = self.surf, self.hasov, self.over
        nx, ny, mnx, mny = self.nx, self.ny, self.minX, self.minY
        inv = 1.0 / n
        for k in range(1, n + 1):
            s = k * inv
            x = x0 + dx * s
            y = y0 + dy * s
            z = z0 + dz * s
            i = int(x - mnx + 0.5)
            j = int(y - mny + 0.5)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                continue
            idx = j * nx + i
            if z < surf[idx]:
                return s
            if hasov[idx]:
                for b, t in over[idx]:
                    if b <= z <= t:
                        return s
        return None

    def los(self, x0, y0, z0, x1, y1, z1, canopy=False):
        """Line of sight: buildings, bridges, terrain; optional canopy transmittance."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(L / self.LOS_STEP) + 1
        if n < 2:
            return True
        surf, hasov, over, hascan, can = self.surf, self.hasov, self.over, self.hascan, self.canopy
        nx, ny, mnx, mny = self.nx, self.ny, self.minX, self.minY
        sx, sy, sz = dx / n, dy / n, dz / n
        seg = L / n
        fol = 0.0
        lim = -math.log(self.canopy_block_t) / self.canopy_density if canopy else 1e18
        x, y, z = x0, y0, z0
        for _ in range(n - 1):
            x += sx
            y += sy
            z += sz
            i = int(x - mnx + 0.5)
            j = int(y - mny + 0.5)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                continue
            idx = j * nx + i
            if z < surf[idx]:
                return False
            if hasov[idx]:
                for b, t in over[idx]:
                    if b <= z <= t:
                        return False
            if canopy and hascan[idx]:
                cb, ct = can[idx]
                if cb <= z <= ct:
                    fol += seg
                    if fol > lim:
                        return False
        return True

    def random_open(self, rng, cx, cy, rad, margin=30.0, tries=400, max_slope=0.5):
        b = self.bounds
        for _ in range(tries):
            a = rng.random() * TAU
            d = rad * math.sqrt(rng.random())
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            if not self.in_bounds(x, y, margin):
                continue
            ok = True
            for ox, oy in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)):
                if not self.open_ground(x + ox, y + oy):
                    ok = False
                    break
            if not ok:
                continue
            g = self.ground(x, y)
            if abs(self.ground(x + 2, y) - g) > 2 * max_slope or abs(self.ground(x, y + 2) - g) > 2 * max_slope:
                continue
            return x, y, g
        x = min(max(cx, b[0] + margin), b[2] - margin)
        y = min(max(cy, b[1] + margin), b[3] - margin)
        return x, y, self.ground(x, y)

    def random_roof(self, rng, tries=200):
        if not self.roofs:
            return None
        tot = sum(a for a, _, _ in self.roofs)
        for _ in range(tries):
            r = rng.random() * tot
            for a, p, bb in self.roofs:
                r -= a
                if r <= 0:
                    break
            x = bb[0] + rng.random() * (bb[2] - bb[0])
            y = bb[1] + rng.random() * (bb[3] - bb[1])
            k = self.idx(x, y)
            if abs(self.ctop[k] - p["top"]) < 0.6 and not self.hasov[k]:
                return x, y, self.ctop[k]
        return None


_GEO_CACHE = {}


def get_geo(arena_dir):
    if arena_dir not in _GEO_CACHE:
        _GEO_CACHE[arena_dir] = Geo(arena_dir)
    return _GEO_CACHE[arena_dir]


def ray_cyl(ox, oy, oz, dx, dy, dz, cx, cy, z0, z1, R, tmax):
    """Entry parameter t in [0, tmax] of the ray o + d t into a vertical cylinder, or None."""
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy
    c = fx * fx + fy * fy - R * R
    if a < 1e-12:
        if c > 0:
            return None
        t0, t1 = -1e18, 1e18
    else:
        b = 2.0 * (fx * dx + fy * dy)
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        t0 = (-b - sq) / (2 * a)
        t1 = (-b + sq) / (2 * a)
    if t1 < 0 or t0 > tmax:
        return None
    if abs(dz) < 1e-12:
        if oz < z0 or oz > z1:
            return None
        tz0, tz1 = -1e18, 1e18
    else:
        ta, tb = (z0 - oz) / dz, (z1 - oz) / dz
        tz0, tz1 = (ta, tb) if ta < tb else (tb, ta)
    lo = max(t0, tz0, 0.0)
    hi = min(t1, tz1, tmax)
    return lo if lo <= hi else None


# ============================================================================ entities
class Weapon:
    __slots__ = ("kind", "cd", "equip", "ammo", "reload", "charging", "charge", "release", "hold")

    def __init__(self, kind, mag=0):
        self.kind = kind
        self.cd = 0.0
        self.equip = 0.0
        self.ammo = mag
        self.reload = 0.0
        self.charging = False
        self.charge = 0.0
        self.release = False
        self.hold = 0.0


class Mech:
    def __init__(self, mid, x, y, z, yaw):
        self.id = mid
        self.kind = "mech"
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.pvx = self.pvy = 0.0   # this tick's after-integration displacement / dt (see World.vel_h)
        self.yaw, self.pitch = yaw, 0.0
        self.cmd_yaw, self.cmd_pitch = yaw, 0.0
        self.move = (0.0, 0.0)
        self.boost = False
        self.jump = False
        self.fire = [False, False]
        self.grounded = True
        self.alive = True
        self.hp = 0.0
        self.energy = 0.0
        self.last_spend = -99.0
        self.last_combat = -99.0
        self.cubes = 0
        self.upg = {}
        self.upg_list = []
        self.weapons = [None, None]
        self.kills = 0
        self.events = []
        self.zone_pending = 0.0
        self.in_zone = True
        self.offer = None
        self.offer_queue = 0
        self.offer_seq = 0
        self.short_until = -99.0
        self.step_t = 0.0
        self.jet_t = 0.0
        self.jetting = False
        self.boosting = False
        self.seen = {}          # id -> last time seen (mechs and NPCs)
        self.ever_seen = set()
        self.vis_ids = set()
        self.vis_known = False
        self.item_cache = {}
        self.last_action = None
        self.missed = 0
        self.killing_blow = None
        self.death_t = None
        self.place = None
        self.st = None
        self.wst = {}
        self.dmg_dealt = 0.0
        self.dmg_taken = 0.0
        self.zone_taken = 0.0
        self.edge_dir = None

    def holds(self, kind):
        return any(w is not None and w.kind == kind for w in self.weapons)

    def laser_charging(self):
        return any(w is not None and w.kind == "laser" and w.charging for w in self.weapons)

    def loadout(self):
        return [w.kind for w in self.weapons if w is not None]


class NPC:
    def __init__(self, nid, x, y, z, hp):
        self.id = nid
        self.kind = "npc"
        self.hx, self.hy = x, y
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = 0.0
        self.yaw = 0.0
        self.hp = self.max_hp = hp
        self.alive = True
        self.target = None
        self.state = "wander"
        self.goal = None
        self.cd = 0.0


class Item:
    __slots__ = ("id", "kind", "x", "y", "z", "hp", "weapon", "alive")

    def __init__(self, iid, kind, x, y, z, hp=0.0, weapon=None):
        self.id = iid
        self.kind = kind          # crate | cube | gun | mortar | laser | lightning
        self.x, self.y, self.z = x, y, z
        self.hp = hp
        self.weapon = weapon
        self.alive = True


class Proj:
    __slots__ = ("id", "kind", "owner", "x", "y", "z", "vx", "vy", "vz", "dist", "rng", "ttl", "dmg", "shot", "radius")

    def __init__(self, pid, kind, owner, x, y, z, vx, vy, vz, rng, ttl, dmg, shot, radius):
        self.id = pid
        self.kind = kind
        self.owner = owner
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.dist = 0.0
        self.rng = rng
        self.ttl = ttl
        self.dmg = dmg
        self.shot = shot
        self.radius = radius


class Drop:
    def __init__(self, did, x, y, z, t_call, t_land):
        self.id = did
        self.x, self.y, self.z = x, y, z
        self.t_call, self.t_land = t_call, t_land
        self.holder = None
        self.progress = 0.0
        self.landed = False
        self.taken = False


# ============================================================================ recorder
class Recorder:
    def __init__(self, path=None):
        self.f = open(path, "w") if path else None
        self.records = []
        self.keep = path is None

    def write(self, rec):
        if self.f:
            self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if self.keep:
            self.records.append(rec)

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


# ============================================================================ world (rules)
class World:
    def __init__(self, cfg, geo, seed=1, n=16, opts=None, rec=None, populate=True):
        self.cfg = cfg
        self.geo = geo
        self.rng = random.Random(seed)
        self.opts = dict(cone="full", mortar_stack=False, drop_npc_blocks=False, peace=False, push_vel="actual")
        self.opts.update(opts or {})
        if self.opts["push_vel"] not in PUSH_VEL_MODES:
            raise ValueError(f"push_vel must be one of {PUSH_VEL_MODES}, got {self.opts['push_vel']!r}")
        self.rec = rec or Recorder(None)
        m = cfg["match"]
        self.dt = 1.0 / m["tick_hz"]
        self.decide_every = m["decide_every"]
        self.duration = m["duration_s"]
        self.tick = 0
        self.t = 0.0
        self.mechs = []
        self.npcs = []
        self.items = {}
        self.projs = []
        self.drops = []
        self.sounds = []        # (t, kind, x, y, z, src)
        self.beams = []         # (t, by, kind, path)
        self.next_item = {"crate": 1000, "cube": 5000, "weapon": 8000}
        self.next_proj = 1
        self.next_drop = 20000
        self.pending_shots = {}
        self.fights = {}
        self.deaths_this_tick = []
        self.order = []          # (tick, [ids])
        self.over = False
        self.n = n
        self.upgrades = {u["id"]: u for u in cfg["upgrades"]}
        self._build_zone()
        if populate:
            self._spawn_mechs(n)
            self._spawn_world()

    # ------------------------------------------------------------------ setup
    def _pick_center(self, c, r, r1):
        g = self.geo
        for _ in range(200):
            a = self.rng.random() * TAU
            d = (r - r1) * math.sqrt(self.rng.random())
            x, y = c[0] + d * math.cos(a), c[1] + d * math.sin(a)
            if g.in_bounds(x, y, min(30.0, max(5.0, r1 * 0.3))):
                if r1 > 0 or g.open_ground(x, y):
                    return (x, y)
        return (c[0], c[1])

    def _build_zone(self):
        z = self.cfg["zone"]
        c = tuple(z["start_center"])
        r = float(z["start_radius"])
        t = 0.0
        self.phases = []
        for ph in z["phases"]:
            r1 = float(ph["radius"])
            c1 = self._pick_center(c, r, r1)
            ss = t + ph["hold_s"]
            se = ss + ph["shrink_s"]
            self.phases.append(dict(t0=t, ss=ss, se=se, c0=c, r0=r, c1=c1, r1=r1, dps=ph["dps"]))
            c, r, t = c1, r1, se
        self.zone_live = (c[0], c[1], r, self.phases[-1]["dps"] if self.phases else 0)
        self._update_zone()

    def zone_phase(self, t=None):
        t = self.t if t is None else t
        for ph in self.phases:
            if t < ph["se"]:
                return ph
        return None

    def _update_zone(self):
        ph = self.zone_phase()
        if ph is None:
            last = self.phases[-1]
            self.zone_live = (last["c1"][0], last["c1"][1], last["r1"], last["dps"])
            return
        if self.t <= ph["ss"]:
            self.zone_live = (ph["c0"][0], ph["c0"][1], ph["r0"], ph["dps"])
        else:
            f = (self.t - ph["ss"]) / max(1e-9, ph["se"] - ph["ss"])
            self.zone_live = (ph["c0"][0] + (ph["c1"][0] - ph["c0"][0]) * f,
                              ph["c0"][1] + (ph["c1"][1] - ph["c0"][1]) * f,
                              ph["r0"] + (ph["r1"] - ph["r0"]) * f, ph["dps"])

    def zone_obs(self):
        ph = self.zone_phase()
        if ph is None:
            last = self.phases[-1]
            c = [r2(last["c1"][0]), r2(last["c1"][1])]
            return {"center": c, "radius": last["r1"], "next_center": c, "next_radius": last["r1"],
                    "shrink_start": last["se"], "shrink_end": last["se"], "dps": last["dps"]}
        return {"center": [r2(ph["c0"][0]), r2(ph["c0"][1])], "radius": r2(ph["r0"]),
                "next_center": [r2(ph["c1"][0]), r2(ph["c1"][1])], "next_radius": r2(ph["r1"]),
                "shrink_start": r2(ph["ss"]), "shrink_end": r2(ph["se"]), "dps": ph["dps"]}

    def _spawn_mechs(self, n):
        m = self.cfg["match"]
        z = self.cfg["zone"]
        sep = float(m["spawn_min_separation"])
        rad = float(m["spawn_zone_fraction"]) * float(z["start_radius"])
        c = z["start_center"]
        margin = float(m.get("arena_margin", 30))
        pts = []
        for attempt in range(60):
            pts = []
            for _ in range(4000):
                if len(pts) == n:
                    break
                x, y, g = self.geo.random_open(self.rng, c[0], c[1], rad, margin, tries=50)
                if all((x - p[0]) ** 2 + (y - p[1]) ** 2 >= sep * sep for p in pts):
                    pts.append((x, y, g))
            if len(pts) == n:
                break
            if attempt % 10 == 9:
                sep *= 0.95
        while len(pts) < n:
            pts.append(self.geo.random_open(self.rng, c[0], c[1], rad, margin))
        for i, (x, y, g) in enumerate(pts):
            self.add_mech(i, x, y, g, yaw=math.atan2(c[1] - y, c[0] - x))

    def add_mech(self, mid, x, y, z=None, yaw=0.0):
        if z is None:
            z = self.geo.floor_at(x, y, 1e9)[0]
        mk = Mech(mid, x, y, z, yaw)
        self._restat(mk)
        mk.hp = self.max_hp(mk)
        mk.energy = mk.st["energy"]
        mk.weapons[0] = Weapon("gun", int(self.cfg["weapons"]["gun"]["mag"]))
        while len(self.mechs) <= mid:
            self.mechs.append(None)
        self.mechs[mid] = mk
        return mk

    def new_id(self, kind):
        k = "weapon" if kind in WEAPON_KINDS else kind
        i = self.next_item[k]
        self.next_item[k] += 1
        return i

    def add_item(self, kind, x, y, z, hp=0.0, weapon=None):
        it = Item(self.new_id(kind), kind, x, y, z, hp, weapon)
        self.items[it.id] = it
        return it

    def add_pickup(self, kind, x, y, z, weapon=None):
        if weapon is None:
            weapon = Weapon(kind, int(self.cfg["weapons"]["gun"]["mag"]) if kind == "gun" else 0)
        return self.add_item(kind, x, y, z, weapon=weapon)

    def add_npc(self, x, y, z=None):
        if z is None:
            z = self.geo.floor_at(x, y, 1e9)[0]
        npc = NPC(100 + len(self.npcs), x, y, z, float(self.cfg["npc"]["hp"]))
        self.npcs.append(npc)
        return npc

    def _rand_weapon(self):
        w = self.cfg["loot"]["weights"]
        kinds = [k for k in WEAPON_KINDS if w.get(k, 0) > 0]
        tot = sum(w[k] for k in kinds)
        r = self.rng.random() * tot
        for k in kinds:
            r -= w[k]
            if r <= 0:
                return k
        return kinds[-1]

    def _spawn_world(self):
        cfg, g, rng = self.cfg, self.geo, self.rng
        zc = cfg["zone"]
        c, rad = zc["start_center"], zc["start_radius"]
        margin = cfg["match"].get("arena_margin", 30)

        def spot(roof_frac):
            if rng.random() < roof_frac:
                p = g.random_roof(rng)
                if p:
                    return p
            return g.random_open(rng, c[0], c[1], rad, margin)

        for _ in range(int(cfg["crates"]["count"])):
            x, y, z = spot(cfg["crates"]["roof_fraction"])
            self.add_item("crate", x, y, z, hp=float(cfg["crates"]["hp"]))
        for _ in range(int(cfg["loot"]["floor_count"])):
            x, y, z = spot(cfg["loot"]["floor_roof_fraction"])
            self.add_pickup(self._rand_weapon(), x, y, z)
        for _ in range(int(cfg["npc"]["count"])):
            x, y, z = g.random_open(rng, c[0], c[1], rad * 0.85, margin)
            self.add_npc(x, y, z)

    # ------------------------------------------------------------------ stats
    def _restat(self, mk):
        st = dict(self.cfg["mech"])
        ws = {k: dict(v) for k, v in self.cfg["weapons"].items()}
        for uid, cnt in mk.upg.items():
            u = self.upgrades.get(uid)
            if not u:
                continue
            for e in u["effects"]:
                parts = e["path"].split(".")
                if parts[0] == "mech":
                    d, key = st, parts[1]
                elif parts[0] == "weapons" and parts[1] in ws:
                    d, key = ws[parts[1]], parts[2]
                else:
                    continue
                if key not in d:
                    d[key] = 0.0
                if "mul" in e:
                    d[key] = d[key] * (e["mul"] ** cnt)
                if "add" in e:
                    d[key] = d[key] + e["add"] * cnt
        mk.st = st
        mk.wst = ws

    def scale(self, mk):
        return 1.0 + self.cfg["cubes"]["size_per_cube"] * mk.cubes

    def max_hp(self, mk):
        c = self.cfg["cubes"]
        return mk.st["hp"] * (1.0 + c["hp_per_cube"] * min(mk.cubes, c["stat_cap"]))

    def dmg_mult(self, mk):
        c = self.cfg["cubes"]
        return 1.0 + c["damage_per_cube"] * min(mk.cubes, c["stat_cap"])

    def radius(self, mk):
        return mk.st["radius"] * self.scale(mk)

    def height(self, mk):
        return mk.st["height"] * self.scale(mk)

    def eye(self, mk):
        return mk.z + mk.st["eye_height"] * self.scale(mk)

    def muzzle(self, mk):
        return mk.z + mk.st["muzzle_height"] * self.scale(mk)

    # ------------------------------------------------------------------ actions
    def apply_action(self, mk, act):
        """Decision-time control update. act is a parsed action dict or None (no valid answer)."""
        cb = self.cfg["bot"]
        if act is None:
            mk.missed += 1
            mk.fire = [False, False]
            mk.jump = False
            if mk.missed * self.dt * self.decide_every > cb["coast_s"] + 1e-9:
                mk.move = (0.0, 0.0)
                mk.boost = False
            return
        mk.missed = 0
        mv = act.get("move", [0, 0])
        mx, my = _num(mv, 0), _num(mv, 1)
        L = math.hypot(mx, my)
        if L > 1.0:
            mx, my = mx / L, my / L
        mk.move = (mx, my)
        mk.boost = bool(act.get("boost", False))
        mk.jump = bool(act.get("jump", False))
        aim = act.get("aim")
        if isinstance(aim, list) and len(aim) >= 2:
            mk.cmd_yaw = _num(aim, 0, mk.cmd_yaw)
            mk.cmd_pitch = min(max(_num(aim, 1, mk.cmd_pitch), mk.st["min_pitch"]), mk.st["max_pitch"])
        fire = act.get("fire", [False, False])
        mk.fire = [bool(fire[0]) if isinstance(fire, list) and len(fire) > 0 else False,
                   bool(fire[1]) if isinstance(fire, list) and len(fire) > 1 else False]
        if act.get("reload"):
            g = self._weapon(mk, "gun")
            if g and g.reload <= 0 and g.ammo < int(mk.wst["gun"]["mag"]):
                self._start_reload(mk, g)
        pick = act.get("pick")
        if isinstance(pick, (int, float)) and not isinstance(pick, bool) and mk.offer is not None:
            i = int(pick)
            if 0 <= i < len(mk.offer["choices"]):
                self._apply_upgrade(mk, mk.offer["choices"][i]["id"])
                mk.offer = None

    def _weapon(self, mk, kind):
        for w in mk.weapons:
            if w is not None and w.kind == kind:
                return w
        return None

    def resolve_loot(self, takes, drops):
        """takes: list of (mech, pickup_id, drop_slot or None); drops: list of (mech, slot) alone."""
        for mk, slot in drops:
            if slot not in (0, 1):
                continue
            w = mk.weapons[slot]
            if w is None or (w.kind == "laser" and w.charging):
                continue
            self._drop_weapon(mk, slot)
        order = []
        for mk, pid, dslot in takes:
            it = self.items.get(pid) if isinstance(pid, int) else None
            d = math.hypot(it.x - mk.x, it.y - mk.y) if it else 1e9
            order.append((d, mk.id, mk, pid, dslot))
        order.sort(key=lambda e: (e[0], e[1]))
        rad = self.cfg["loot"]["pickup_radius"]
        for d, _, mk, pid, dslot in order:
            it = self.items.get(pid)
            if it is None or not it.alive or it.kind not in WEAPON_KINDS:
                mk.events.append({"type": "take_failed", "id": pid, "reason": "gone"})
                continue
            if d > rad or abs(it.z - mk.z) > 4.0:
                mk.events.append({"type": "take_failed", "id": pid, "reason": "range"})
                continue
            if dslot not in (0, 1):
                dslot = None
            others = [w for i, w in enumerate(mk.weapons) if w is not None and i != dslot]
            if any(w.kind == it.kind for w in others):
                mk.events.append({"type": "take_failed", "id": pid, "reason": "have"})
                continue
            if dslot is None:
                slot = 0 if mk.weapons[0] is None else (1 if mk.weapons[1] is None else None)
                if slot is None:
                    mk.events.append({"type": "take_failed", "id": pid, "reason": "full"})
                    continue
            else:
                slot = dslot
                old = mk.weapons[slot]
                if old is not None and old.kind == "laser" and old.charging:
                    mk.events.append({"type": "take_failed", "id": pid, "reason": "busy"})
                    continue
                if old is not None:
                    self._drop_weapon(mk, slot)
            w = it.weapon
            it.alive = False
            del self.items[it.id]
            w.charging, w.charge, w.release, w.hold = False, 0.0, False, 0.0
            w.equip = max(self.cfg["loot"]["equip_s"], mk.wst[w.kind]["cooldown"])
            mk.weapons[slot] = w
            mk.events.append({"type": "took", "id": it.id, "kind": w.kind, "slot": slot})
            self.rec.write({"type": "take", "t": r2(self.t), "id": mk.id, "kind": w.kind, "slot": slot})

    def _drop_weapon(self, mk, slot):
        w = mk.weapons[slot]
        mk.weapons[slot] = None
        w.charging, w.charge, w.release, w.hold = False, 0.0, False, 0.0
        it = self.add_pickup(w.kind, mk.x, mk.y, mk.z, weapon=w)
        mk.events.append({"type": "dropped", "id": it.id, "kind": w.kind, "slot": slot})
        return it

    # ------------------------------------------------------------------ upgrades / cubes
    def _eligible(self, mk):
        out = []
        for u in self.cfg["upgrades"]:
            if mk.upg.get(u["id"], 0) >= u.get("max", 1):
                continue
            req = u.get("requires")
            if req and not mk.holds(req):
                continue
            out.append(u)
        return out

    def _offer_tick(self, mk):
        if mk.offer is not None and self.t >= mk.offer["expires"] - 1e-9:
            self._apply_upgrade(mk, mk.offer["choices"][0]["id"])
            mk.offer = None
        if mk.offer is None and mk.offer_queue > 0:
            mk.offer_queue -= 1
            el = self._eligible(mk)
            if el:
                ch = self.rng.sample(el, min(3, len(el)))
                mk.offer_seq += 1
                mk.offer = {"id": mk.offer_seq, "expires": r2(self.t + self.cfg["bot"]["offer_timeout_s"]),
                            "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in ch]}

    def _apply_upgrade(self, mk, uid):
        old_max = self.max_hp(mk)
        mk.upg[uid] = mk.upg.get(uid, 0) + 1
        mk.upg_list.append(uid)
        self._restat(mk)
        new_max = self.max_hp(mk)
        if new_max > old_max:
            mk.hp += new_max - old_max
        mk.hp = min(mk.hp, new_max)
        mk.energy = min(mk.energy, mk.st["energy"])
        mk.events.append({"type": "upgrade", "id": uid})
        self.rec.write({"type": "upgrade", "t": r2(self.t), "id": mk.id, "upgrade": uid})

    def give_cubes(self, mk, n):
        th = self.cfg["cubes"]["upgrade_thresholds"]
        for _ in range(n):
            old_max = self.max_hp(mk)
            mk.cubes += 1
            mk.hp += self.max_hp(mk) - old_max
            if mk.cubes in th:
                mk.offer_queue += 1

    # ------------------------------------------------------------------ sounds / damage
    def sound(self, kind, x, y, z, src):
        self.sounds.append((self.t, kind, x, y, z, src))

    def damage_mech(self, tgt, amount, weapon, att, src_xy, effect=None, shot=None, self_hit=False):
        if self.opts.get("peace"):          # --peace: weapons and drones cannot hurt mechs (zone still does)
            return 0.0
        rear = False
        bearing = None
        if src_xy is not None:
            bearing = math.atan2(src_xy[1] - tgt.y, src_xy[0] - tgt.x)
            if not self_hit and abs(wrap(bearing - tgt.yaw)) > math.radians(tgt.st["front_arc_deg"]) / 2:
                rear = True
                amount *= tgt.st["rear_mult"]
        amount *= max(0.0, 1.0 - tgt.st["armor"])
        was_alive = tgt.hp > 0
        tgt.hp -= amount
        tgt.dmg_taken += amount
        tgt.last_combat = self.t
        by = att.id if att is not None else None
        ev = {"type": "damaged", "amount": r2(amount), "weapon": weapon,
              "bearing": round(bearing, 3) if bearing is not None else None, "by": by}
        if rear:
            ev["rear"] = True
        tgt.events.append(ev)
        if att is not None and att.kind == "mech":
            att.last_combat = self.t
            if not self_hit:
                att.dmg_dealt += amount
                hv = {"type": "hit", "target": tgt.id, "amount": r2(amount), "weapon": weapon}
                if effect:
                    hv["effect"] = effect
                if rear:
                    hv["rear"] = True
                att.events.append(hv)
                if att.alive and was_alive:
                    self._fight_damage(att, tgt, amount)
            if shot is not None and not self_hit:
                shot["hit"] = True
                shot.setdefault("hit_ids", []).append(tgt.id)
                shot["dmg"] = r2(shot.get("dmg", 0) + amount)
        if was_alive and tgt.hp <= 0:
            tgt.killing_blow = (att.id if (att is not None and att.kind == "mech" and not self_hit) else None,
                                ("self_" + weapon) if self_hit else weapon, att.id if att is not None else None)
        return amount

    def damage_other(self, tgt, amount, weapon, att, effect=None, shot=None):
        """NPC or crate."""
        if not tgt.alive:
            return 0
        tgt.hp -= amount
        if att is not None and att.kind == "mech":
            att.last_combat = self.t
            hv = {"type": "hit", "target": tgt.id, "amount": r2(amount), "weapon": weapon}
            if effect:
                hv["effect"] = effect
            att.events.append(hv)
            if shot is not None:
                shot.setdefault("other_hits", []).append(tgt.id)
            if isinstance(tgt, NPC):
                tgt.target = att
                tgt.state = "chase"
        if tgt.hp <= 0:
            tgt.alive = False
            if isinstance(tgt, NPC):
                for k in range(int(self.cfg["npc"]["cubes"])):
                    a = self.rng.random() * TAU
                    self.add_item("cube", tgt.x + 1.2 * math.cos(a), tgt.y + 1.2 * math.sin(a), tgt.z)
                self.rec.write({"type": "npc_kill", "t": r2(self.t), "id": tgt.id,
                                "by": att.id if att is not None else None})
            else:
                del self.items[tgt.id]
                self.add_item("cube", tgt.x, tgt.y, tgt.z)
                if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                    a = self.rng.random() * TAU
                    self.add_pickup(self._rand_weapon(), tgt.x + 1.5 * math.cos(a), tgt.y + 1.5 * math.sin(a), tgt.z)
                self.sound("crate", tgt.x, tgt.y, tgt.z, None)
        return amount

    def _fight_damage(self, a, b, amount):
        key = (min(a.id, b.id), max(a.id, b.id))
        f = self.fights.get(key)
        if f is None:
            f = {"start": self.t, "a": a.id, "b": b.id, "initiator": a.id,
                 "range": r2(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))),
                 "loadout": {str(a.id): a.loadout(), str(b.id): b.loadout()},
                 "hp_start": {str(a.id): r2(a.hp), str(b.id): r2(b.hp + amount)},
                 "cubes": {str(a.id): a.cubes, str(b.id): b.cubes},
                 "dmg": {str(a.id): 0.0, str(b.id): 0.0}, "last": self.t}
            self.fights[key] = f
        f["dmg"][str(a.id)] += amount
        f["last"] = self.t

    def _close_fight(self, key, outcome, winner):
        f = self.fights.pop(key)
        a, b = self.mechs[f["a"]], self.mechs[f["b"]]
        self.rec.write({"type": "fight", "t0": r2(f["start"]), "t1": r2(self.t), "a": f["a"], "b": f["b"],
                        "initiator": f["initiator"], "range": f["range"], "loadout": f["loadout"],
                        "hp_start": f["hp_start"], "cubes": f["cubes"],
                        "hp_end": {str(a.id): r2(max(0, a.hp)), str(b.id): r2(max(0, b.hp))},
                        "dmg": {k: r2(v) for k, v in f["dmg"].items()}, "outcome": outcome, "winner": winner})

    # ------------------------------------------------------------------ shots
    def _intended(self, mk, yaw, rng_max, ang=math.radians(10)):
        best, bd = None, None
        for o in self.mechs:
            if o is None or o is mk or not o.alive:
                continue
            if mk.vis_known and o.id not in mk.vis_ids:
                continue
            dx, dy = o.x - mk.x, o.y - mk.y
            d = math.hypot(dx, dy)
            if d > rng_max:
                continue
            err = abs(wrap(math.atan2(dy, dx) - yaw))
            if err <= max(ang, math.atan2(self.radius(o) * 1.5, max(d, 1.0))):
                if best is None or err < bd:
                    best, bd = o, err
        return best

    def _new_shot(self, mk, weapon, tgt):
        rec = {"type": "shot", "t": r2(self.t), "by": mk.id, "weapon": weapon,
               "target": tgt.id if tgt is not None else None,
               "range": r2(math.dist((mk.x, mk.y, mk.z), (tgt.x, tgt.y, tgt.z))) if tgt is not None else None,
               "moving": (math.hypot(tgt.vx, tgt.vy) > 3.0) if tgt is not None else None,
               "self_moving": math.hypot(mk.vx, mk.vy) > 3.0, "hit": False}
        return rec

    def _shot_origin(self, mk, yaw, pitch, h):
        """Muzzle point radius+0.5 out along the aim, pulled in if a wall is closer."""
        off = (mk.st["radius"] + 0.5) * self.scale(mk)
        cp = math.cos(pitch)
        dx, dy, dz = cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)
        x0, y0, z0 = mk.x, mk.y, h
        s = self.geo.seg_hit(x0, y0, z0, x0 + dx * off, y0 + dy * off, z0 + dz * off, step=0.5)
        if s is not None:
            off = max(0.0, off * s - 0.3)
        return x0 + dx * off, y0 + dy * off, z0 + dz * off, dx, dy, dz

    def _targets(self, exclude=None):
        out = []
        for o in self.mechs:
            if o is not None and o.alive and o is not exclude:
                out.append((o, o.x, o.y, o.z, o.z + self.height(o), self.radius(o)))
        ncfg = self.cfg["npc"]
        for n in self.npcs:
            if n.alive:
                out.append((n, n.x, n.y, n.z, n.z + ncfg["height"], ncfg["radius"]))
        cc = self.cfg["crates"]
        for it in self.items.values():
            if it.kind == "crate":
                out.append((it, it.x, it.y, it.z, it.z + cc["height"], cc["radius"]))
        return out

    def _fire(self, mk, slot, w):
        kind = w.kind
        ws = mk.wst[kind]
        mult = self.dmg_mult(mk)
        if kind == "gun":
            sp = math.radians(ws["spread_deg"])
            yaw = mk.yaw + self.rng.gauss(0, sp)
            pitch = mk.pitch + self.rng.gauss(0, sp)
            x, y, z, dx, dy, dz = self._shot_origin(mk, yaw, pitch, self.muzzle(mk))
            shot = self._new_shot(mk, "gun", self._intended(mk, mk.yaw, ws["range"] * 1.1))
            self.projs.append(Proj(self.next_proj, "gun", mk.id, x, y, z, dx * ws["speed"], dy * ws["speed"],
                                   dz * ws["speed"], ws["range"], 1e9, ws["damage"] * mult, shot,
                                   self.cfg["projectile"]["gun_radius"]))
            self.next_proj += 1
            self.sound("gun", mk.x, mk.y, mk.z, mk.id)
        elif kind == "mortar":
            x, y, z, dx, dy, dz = self._shot_origin(mk, mk.yaw, mk.pitch, mk.z + self.height(mk) + 0.5 * self.scale(mk))
            v = ws["speed"]
            shot = self._new_shot(mk, "mortar", self._intended(mk, mk.yaw, 260, math.radians(12)))
            self.projs.append(Proj(self.next_proj, "mortar", mk.id, x, y, z, dx * v, dy * v, dz * v, 1e9,
                                   ws["ttl"], ws["damage"] * mult, shot, 0.3))
            self.next_proj += 1
            self.sound("mortar", mk.x, mk.y, mk.z, mk.id)
        elif kind == "laser":
            self._fire_laser(mk, w, ws, mult)
        elif kind == "lightning":
            self._fire_lightning(mk, ws, mult)

    def _fire_laser(self, mk, w, ws, mult):
        frac = ws["min_frac"] + (1 - ws["min_frac"]) * min(1.0, w.charge / max(1e-9, ws["charge"]))
        dmg = ws["damage"] * frac * mult
        x, y, z, dx, dy, dz = self._shot_origin(mk, mk.yaw, mk.pitch, self.muzzle(mk))
        R = ws["range"]
        s = self.geo.seg_hit(x, y, z, x + dx * R, y + dy * R, z + dz * R, step=1.0)
        tmax = R * s if s is not None else R
        hits = []
        for o, cx, cy, z0, z1, rad in self._targets(exclude=mk):
            if abs(cx - x) > R + 10 or abs(cy - y) > R + 10:
                continue
            tt = ray_cyl(x, y, z, dx, dy, dz, cx, cy, z0, z1, rad, tmax)
            if tt is not None:
                hits.append((tt, o))
        hits.sort(key=lambda e: e[0])
        n = 1 + int(ws.get("pierce", 0))
        hits = hits[:n]
        end_t = hits[-1][0] if (hits and len(hits) == n) else tmax
        shot = self._new_shot(mk, "laser", self._intended(mk, mk.yaw, R * 1.05))
        shot["charge"] = r2(w.charge)
        for tt, o in hits:
            if isinstance(o, Mech):
                eff = None
                d = dmg
                if o.holds("mortar"):
                    d *= 1 + ws["cookoff"]
                    eff = "cookoff"
                    o.events.append({"type": "effect", "effect": "cookoff", "by": mk.id})
                self.damage_mech(o, d, "laser", mk, (x, y), effect=eff, shot=shot)
            else:
                self.damage_other(o, dmg, "laser", mk, shot=shot)
        self.rec.write(shot)
        self.beams.append((self.t, mk.id, "laser", [[r2(x), r2(y), r2(z)],
                                                   [r2(x + dx * end_t), r2(y + dy * end_t), r2(z + dz * end_t)]]))
        self.sound("laser", mk.x, mk.y, mk.z, mk.id)
        w.charging, w.charge, w.release, w.hold = False, 0.0, False, 0.0
        w.cd = ws["cooldown"]

    def _fire_lightning(self, mk, ws, mult):
        cost = ws["energy"]
        mk.energy -= cost
        mk.last_spend = self.t
        mz = self.muzzle(mk)
        ox, oy = mk.x, mk.y
        half = math.radians(ws["cone_deg"]) * (0.5 if self.opts["cone"] == "half" else 1.0)
        R = ws["range"]
        best, bd = None, None
        cands = self._targets(exclude=mk)
        for o, cx, cy, z0, z1, rad in cands:
            cz = (z0 + z1) / 2
            d = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2 + (cz - mz) ** 2)
            if d > R or (bd is not None and d >= bd):
                continue
            if abs(wrap(math.atan2(cy - oy, cx - ox) - mk.yaw)) > half:
                continue
            if not self.geo.los(ox, oy, mz, cx, cy, cz):
                continue
            best, bd = (o, cx, cy, cz), d
        path = [[r2(ox), r2(oy), r2(mz)]]
        shot = None
        if best is None:
            x, y, z, dx, dy, dz = self._shot_origin(mk, mk.yaw, mk.pitch, mz)
            s = self.geo.seg_hit(x, y, z, x + dx * R, y + dy * R, z + dz * R)
            L = R * s if s is not None else R
            path.append([r2(x + dx * L), r2(y + dy * L), r2(z + dz * L)])
            tg = self._intended(mk, mk.yaw, R * 1.3, half)
            shot = self._new_shot(mk, "lightning", tg)
        else:
            hit = [best]
            hit_ids = {id(best[0])}
            for k in range(int(ws["chain"])):
                po, px, py, pz = hit[-1]
                nb, nd = None, None
                for o, cx, cy, z0, z1, rad in cands:
                    if id(o) in hit_ids:
                        continue
                    cz = (z0 + z1) / 2
                    d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
                    if d <= ws["chain_range"] and (nd is None or d < nd) and self.geo.los(px, py, pz, cx, cy, cz):
                        nb, nd = (o, cx, cy, cz), d
                if nb is None:
                    break
                hit.append(nb)
                hit_ids.add(id(nb[0]))
            first = hit[0][0]
            shot = self._new_shot(mk, "lightning", first if isinstance(first, Mech) else
                                  self._intended(mk, mk.yaw, R * 1.3, half))
            prev = (ox, oy)
            for k, (o, cx, cy, cz) in enumerate(hit):
                dmg = ws["damage"] * (ws["chain_falloff"] ** k) * mult
                path.append([r2(cx), r2(cy), r2(cz)])
                if isinstance(o, Mech):
                    eff = None
                    if o.holds("laser"):
                        eff = "short"
                        o.short_until = self.t + ws["short_s"]
                        for lw in o.weapons:
                            if lw is not None and lw.kind == "laser":
                                lw.charging, lw.charge, lw.release, lw.hold = False, 0.0, False, 0.0
                        o.events.append({"type": "effect", "effect": "short", "by": mk.id})
                    self.damage_mech(o, dmg, "lightning", mk, prev, effect=eff, shot=shot)
                else:
                    self.damage_other(o, dmg, "lightning", mk, shot=shot)
                prev = (cx, cy)
        self.rec.write(shot)
        self.beams.append((self.t, mk.id, "lightning", path))
        self.sound("lightning", mk.x, mk.y, mk.z, mk.id)

    def _start_reload(self, mk, g):
        g.reload = mk.wst["gun"]["reload"]
        self.sound("reload", mk.x, mk.y, mk.z, mk.id)

    def _weapons_tick(self, mk):
        dt = self.dt
        for slot in (0, 1):
            w = mk.weapons[slot]
            if w is None:
                continue
            ws = mk.wst[w.kind]
            trig = mk.fire[slot]
            if w.equip > 0:
                w.equip = max(0.0, w.equip - dt)
            if w.kind == "gun":
                if w.reload > 0:
                    w.reload -= dt
                    if w.reload <= 1e-9:
                        w.reload = 0.0
                        w.ammo = int(ws["mag"])
                w.cd = max(w.cd - dt, -dt)
                if trig and w.cd <= 1e-9 and w.equip <= 0 and w.reload <= 0 and w.ammo > 0:
                    self._fire(mk, slot, w)
                    w.ammo -= 1
                    w.cd += ws["cooldown"]
                    if w.ammo <= 0:
                        self._start_reload(mk, w)
            elif w.kind == "lightning":
                w.cd = max(w.cd - dt, -dt)
                if trig and w.cd <= 1e-9 and w.equip <= 0 and mk.energy >= ws["energy"]:
                    self._fire(mk, slot, w)
                    w.cd += ws["cooldown"]
            elif w.kind == "mortar":
                w.cd = max(w.cd - dt, 0.0)
                if trig and w.cd <= 1e-9 and w.equip <= 0:
                    self._fire(mk, slot, w)
                    w.cd = ws["cooldown"]
            elif w.kind == "laser":
                w.cd = max(w.cd - dt, 0.0)
                if not w.charging:
                    if trig and w.cd <= 1e-9 and w.equip <= 0 and self.t >= mk.short_until:
                        w.charging, w.charge, w.release, w.hold = True, 0.0, False, 0.0
                        self.sound("laser_charge", mk.x, mk.y, mk.z, mk.id)
                    else:
                        continue
                full = ws["charge"]
                was_full = w.charge >= full - 1e-9
                if not trig:
                    w.release = True
                if not w.release or w.charge < ws["min_charge"] - 1e-9:
                    w.charge = min(full, w.charge + dt)
                if w.release and w.charge >= ws["min_charge"] - 1e-9:
                    self._fire(mk, slot, w)
                elif was_full:
                    w.hold += dt
                    if w.hold >= ws["hold_s"] - 1e-9:
                        self._fire(mk, slot, w)

    # ------------------------------------------------------------------ movement
    def _offsets(self, r):
        n = 8 if r < 3.5 else 12
        rr = r * 0.95
        return [(0.0, 0.0)] + [(rr * math.cos(k * TAU / n), rr * math.sin(k * TAU / n)) for k in range(n)]

    def _blocked(self, x, y, z, r, h, step, offs):
        g = self.geo
        if not g.in_bounds(x, y, r):
            return 99
        cnt = 0
        for ox, oy in offs:
            if g.wall_hit(x + ox, y + oy, z + step, z + h):
                cnt += 1
        return cnt

    def _move_mech(self, mk):
        dt, st, g = self.dt, mk.st, self.geo
        r, h = self.radius(mk), self.height(mk)
        step = st["step_up"]
        offs = self._offsets(r)
        mx, my = mk.move
        moving = mx * mx + my * my > 0.0025
        mk.boosting = mk.jetting = False
        mk.pvx = mk.pvy = 0.0       # nothing has moved this mech after integration yet this tick (World.vel_h)
        speed = st["walk_speed"]
        if mk.boost and mk.energy > 0:
            if mk.grounded and moving:
                speed = st["boost_speed"]
                mk.energy -= st["boost_drain"] * dt
                mk.last_spend = self.t
                mk.boosting = True
            elif not mk.grounded:
                if mk.vz < st["max_rise_speed"]:
                    mk.vz = min(st["max_rise_speed"], mk.vz + st["jet_accel"] * dt)
                mk.energy -= st["jet_drain"] * dt
                mk.last_spend = self.t
                mk.jetting = True
        if mk.energy < 0:
            mk.energy = 0.0
        if mk.laser_charging():
            speed *= st["charge_move_mult"]
        acc = st["ground_accel"] if mk.grounded else st["air_accel"]
        dvx, dvy = mx * speed - mk.vx, my * speed - mk.vy
        dl = math.hypot(dvx, dvy)
        if dl > acc * dt:
            dvx, dvy = dvx * acc * dt / dl, dvy * acc * dt / dl
        mk.vx += dvx
        mk.vy += dvy
        if mk.jump and mk.grounded:
            mk.vz = st["jump_speed"]
            mk.grounded = False
            mk.jump = False
        if not mk.grounded:
            mk.vz -= st["gravity"] * dt
        # horizontal
        x0, y0, z0 = mk.x, mk.y, mk.z
        sx, sy = mk.vx * dt, mk.vy * dt
        if sx or sy:
            base = self._blocked(x0, y0, z0, r, h, step, offs)
            placed = False
            cands = [(sx, sy, None)]
            cands.append((sx, 0.0, "x"))
            cands.append((0.0, sy, "y"))
            for ang in (0.5236, -0.5236, 1.0472, -1.0472):
                ca, sa = math.cos(ang), math.sin(ang)
                c = math.cos(ang)
                cands.append(((sx * ca - sy * sa) * c, (sx * sa + sy * ca) * c, "r"))
            for cx_, cy_, mode in cands:
                nx_, ny_ = x0 + cx_, y0 + cy_
                b = self._blocked(nx_, ny_, z0, r, h, step, offs)
                if b == 0 or (base > 0 and b < base):
                    if mk.grounded:
                        f, terr = g.floor_at(nx_, ny_, z0 + step)
                        run = math.hypot(cx_, cy_)
                        if terr and f - z0 > 0.02 and f - z0 > math.tan(math.radians(st["max_slope_deg"])) * run:
                            continue
                    mk.x, mk.y = nx_, ny_
                    if mode == "x":
                        mk.vy = 0.0
                    elif mode == "y":
                        mk.vx = 0.0
                    elif mode == "r":
                        mk.vx, mk.vy = cx_ / dt, cy_ / dt
                    placed = True
                    break
            if not placed:
                mk.vx = mk.vy = 0.0
        # vertical
        x, y = mk.x, mk.y
        znew = mk.z + mk.vz * dt
        if mk.vz > 0:
            lo = mk.z + h
            best = None
            for ox, oy in offs:
                c = g.ceiling(x + ox, y + oy, lo - 0.01, znew + h)
                if c is not None and (best is None or c < best):
                    best = c
            if best is not None:
                znew = min(znew, best - h)
                mk.vz = 0.0
        thr = mk.z + (step if mk.grounded else 0.05)
        fc, _ = g.floor_at(x, y, thr)
        fp = NONE_H
        edge_vec = [0.0, 0.0]
        for ox, oy in offs[1:]:
            rt = g.roof_top(x + ox, y + oy, mk.z + 0.05)
            if rt > fp + 0.05:
                fp = rt
                edge_vec = [ox, oy]
            elif abs(rt - fp) <= 0.05:
                edge_vec[0] += ox
                edge_vec[1] += oy
        sup = max(fc, fp)
        mk.edge_dir = None
        if znew <= sup + 1e-6 or (mk.grounded and mk.vz <= 0 and mk.z - sup <= step):
            mk.z = sup
            if mk.vz < 0:
                mk.vz = 0.0
            mk.grounded = True
            if fp > fc + 0.05:
                L = math.hypot(*edge_vec) or 1.0
                ex, ey = edge_vec[0] / L, edge_vec[1] / L
                mk.edge_dir = (ex, ey)
                if mx * ex + my * ey < 0.3:
                    px, py = x - ex * 0.6, y - ey * 0.6
                    if self._blocked(px, py, mk.z, r, h, step, offs) == 0:
                        mk.x, mk.y = px, py
                        mk.pvx += (px - x) / dt     # a move the velocity state does not carry (World.vel_h)
                        mk.pvy += (py - y) / dt
        else:
            mk.z = znew
            mk.grounded = False

    # ------------------------------------------------------------------ npcs
    def _npc_tick(self, n, decide):
        nc = self.cfg["npc"]
        dt = self.dt
        g = self.geo
        if decide:
            if n.target is not None:
                t = n.target
                if (not t.alive or math.hypot(t.x - n.x, t.y - n.y) > nc["deaggro_range"]
                        or math.hypot(n.x - n.hx, n.y - n.hy) > nc["chase_leash"]):
                    n.target = None
                    n.state = "return"
            if n.target is None and n.state != "return" and math.hypot(n.x - n.hx, n.y - n.hy) <= nc["leash"]:
                best, bd = None, None
                for m in self.mechs:
                    if m is None or not m.alive:
                        continue
                    d = math.hypot(m.x - n.x, m.y - n.y)
                    if d <= nc["aggro_range"] and (bd is None or d < bd):
                        if g.los(n.x, n.y, n.z + nc["height"] * 0.85, m.x, m.y, m.z + self.height(m) / 2):
                            best, bd = m, d
                if best is not None:
                    n.target, n.state = best, "chase"
        # movement
        tx = ty = None
        sp = nc["speed"]
        if n.state == "chase" and n.target is not None:
            if math.hypot(n.target.x - n.x, n.target.y - n.y) > 20:
                tx, ty = n.target.x, n.target.y
        elif n.state == "return":
            tx, ty = n.hx, n.hy
            if math.hypot(n.x - n.hx, n.y - n.hy) < 3:
                n.state = "wander"
                n.goal = None
        else:
            if n.goal is None or math.hypot(n.goal[0] - n.x, n.goal[1] - n.y) < 2 or self.rng.random() < 0.002:
                a = self.rng.random() * TAU
                d = nc["leash"] * math.sqrt(self.rng.random())
                n.goal = (n.hx + d * math.cos(a), n.hy + d * math.sin(a))
            tx, ty = n.goal
            sp *= 0.5
        n.vx = n.vy = 0.0
        if tx is not None:
            dx, dy = tx - n.x, ty - n.y
            d = math.hypot(dx, dy)
            if d > 0.1:
                vx, vy = dx / d * sp, dy / d * sp
                nx_, ny_ = n.x + vx * dt, n.y + vy * dt
                f, _ = g.floor_at(nx_, ny_, n.z + 1.2)
                ok = g.in_bounds(nx_, ny_, 3) and abs(f - n.z) <= 1.2
                if ok:
                    for ox, oy in ((nc["radius"], 0), (-nc["radius"], 0), (0, nc["radius"]), (0, -nc["radius"])):
                        if g.wall_hit(nx_ + ox, ny_ + oy, n.z + 1.2, n.z + nc["height"]):
                            ok = False
                            break
                if ok:
                    n.x, n.y, n.z = nx_, ny_, f
                    n.vx, n.vy = vx, vy
                    n.yaw = math.atan2(vy, vx)
                elif n.state == "wander":
                    n.goal = None
        # fire
        n.cd = max(0.0, n.cd - dt)
        t = n.target
        if t is not None and t.alive and n.cd <= 0:
            ez = n.z + nc["height"] * 0.7
            tz = t.z + self.height(t) / 2
            dx, dy, dz = t.x - n.x, t.y - n.y, tz - ez
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= nc["deaggro_range"] and g.los(n.x, n.y, ez, t.x, t.y, tz):
                n.yaw = math.atan2(dy, dx)
                sp_ = math.radians(nc["spread_deg"])
                yaw = n.yaw + self.rng.gauss(0, sp_)
                pitch = math.atan2(dz, math.hypot(dx, dy)) + self.rng.gauss(0, sp_)
                v = nc["projectile_speed"]
                cp = math.cos(pitch)
                off = nc["radius"] + 0.5
                ux, uy, uz = cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)
                self.projs.append(Proj(self.next_proj, "npc", n.id, n.x + ux * off, n.y + uy * off, ez + uz * off,
                                       ux * v, uy * v, uz * v, nc["deaggro_range"] * 1.6, 1e9, nc["damage"], None,
                                       self.cfg["projectile"]["npc_radius"]))
                self.next_proj += 1
                self.sound("npc_gun", n.x, n.y, n.z, n.id)
                n.cd = nc["cooldown"]

    # ------------------------------------------------------------------ projectiles
    def _proj_tick(self):
        if not self.projs:
            return
        dt = self.dt
        g = self.geo
        tg = self._targets()
        keep = []
        for p in self.projs:
            x0, y0, z0 = p.x, p.y, p.z
            if p.kind == "mortar":
                gr = self.mechs[p.owner].wst["mortar"]["gravity"] if self.mechs[p.owner] else 22.0
                x1 = x0 + p.vx * dt
                y1 = y0 + p.vy * dt
                z1 = z0 + p.vz * dt - 0.5 * gr * dt * dt
                p.vz -= gr * dt
            else:
                x1, y1, z1 = x0 + p.vx * dt, y0 + p.vy * dt, z0 + p.vz * dt
            seglen = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            tlim = 1.0
            expire = False
            if p.kind != "mortar" and p.dist + seglen > p.rng:
                tlim = max(0.0, (p.rng - p.dist) / max(seglen, 1e-9))
                expire = True
            p.ttl -= dt
            if p.ttl <= 0:
                expire = True
            sg = g.seg_hit(x0, y0, z0, x0 + (x1 - x0) * tlim, y0 + (y1 - y0) * tlim, z0 + (z1 - z0) * tlim)
            if sg is not None:
                sg *= tlim
            best_t = sg if sg is not None else tlim
            best_o = None
            bx0, bx1 = min(x0, x1) - 6, max(x0, x1) + 6
            by0, by1 = min(y0, y1) - 6, max(y0, y1) + 6
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            pr = p.radius
            for o, cx, cy, oz0, oz1, rad in tg:
                if cx < bx0 or cx > bx1 or cy < by0 or cy > by1:
                    continue
                if (o.kind == "mech" and o.id == p.owner and p.kind != "npc") or (p.kind == "npc" and o.kind == "npc"):
                    continue
                if p.kind == "npc" and isinstance(o, Item):
                    continue
                tt = ray_cyl(x0, y0, z0, dx, dy, dz, cx, cy, oz0 - pr, oz1 + pr, rad + pr, best_t)
                if tt is not None and tt <= best_t:
                    best_t, best_o = tt, o
            hx, hy, hz = x0 + dx * best_t, y0 + dy * best_t, z0 + dz * best_t
            if best_o is not None or sg is not None or (p.kind == "mortar" and expire):
                self._impact(p, best_o, hx, hy, hz)
                continue
            if expire or not g.in_bounds(x1, y1, -50):
                if p.shot is not None:
                    self.rec.write(p.shot)
                continue
            p.x, p.y, p.z = x1, y1, z1
            p.dist += seglen
            keep.append(p)
        self.projs = keep

    def _impact(self, p, o, hx, hy, hz):
        att = self.mechs[p.owner] if p.kind != "npc" else next((n for n in self.npcs if n.id == p.owner), None)
        if p.kind == "gun" or p.kind == "npc":
            if o is not None:
                src = (hx - p.vx * 0.05, hy - p.vy * 0.05)
                wname = "gun" if p.kind == "gun" else "npc"
                if isinstance(o, Mech):
                    self.damage_mech(o, p.dmg, wname, att, src, shot=p.shot)
                elif p.kind == "gun":
                    self.damage_other(o, p.dmg, "gun", att, shot=p.shot)
            if p.shot is not None:
                self.rec.write(p.shot)
            return
        # mortar
        mc = self.cfg["weapons"]["mortar"]
        ws = att.wst["mortar"] if att else mc
        mult = p.dmg / ws["damage"] if ws["damage"] else 1.0
        sp = math.sqrt(p.vx ** 2 + p.vy ** 2 + p.vz ** 2) or 1.0
        bx, by, bz = hx - p.vx / sp * 0.4, hy - p.vy / sp * 0.4, hz - p.vz / sp * 0.4
        self.sound("explosion", hx, hy, hz, p.owner)
        if o is not None:
            if isinstance(o, Mech):
                self.damage_mech(o, p.dmg, "mortar", att, (bx, by), shot=p.shot)
            else:
                self.damage_other(o, p.dmg, "mortar", att, shot=p.shot)
        R = ws["splash_radius"]
        edge = ws.get("splash_edge", 0.3)
        g = self.geo
        for t, cx, cy, z0, z1, rad in self._targets():
            if t is o and not self.opts["mortar_stack"] and not isinstance(t, Mech):
                continue
            hd = max(0.0, math.hypot(cx - hx, cy - hy) - rad)
            vz = 0.0 if z0 <= hz <= z1 else (z0 - hz if hz < z0 else hz - z1)
            d = math.hypot(hd, vz)
            if d > R:
                continue
            cz = min(max(bz, z0 + 0.5), z1 - 0.5)
            if not g.los(bx, by, bz, cx, cy, cz):
                if not g.los(bx, by, bz, cx, cy, z1 - 0.2):
                    continue
            f = 1.0 - (1.0 - edge) * (d / R) if R > 0 else 1.0
            if isinstance(t, Mech):
                self_hit = att is t
                if t is not o or self.opts["mortar_stack"]:
                    amt = ws["splash"] * f * mult * (t.st["self_damage_mult"] if self_hit else 1.0)
                    self.damage_mech(t, amt, "mortar", att, (bx, by), effect="concuss", shot=p.shot, self_hit=self_hit)
                dx, dy = cx - hx, cy - hy
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    a = self.rng.random() * TAU
                    dx, dy, L = math.cos(a), math.sin(a), 1.0
                t.vx += dx / L * ws["knock"] * f
                t.vy += dy / L * ws["knock"] * f
                t.vz = max(t.vz, 0.0) + ws["knock_up"] * f
                t.grounded = False
                t.energy = max(0.0, t.energy - ws["drain"] * f)
                t.last_spend = self.t
                t.events.append({"type": "effect", "effect": "concuss", "by": p.owner})
            elif t is not o:
                self.damage_other(t, ws["splash"] * f * mult, "mortar", att, shot=p.shot)
        if p.shot is not None:
            self.rec.write(p.shot)

    # ------------------------------------------------------------------ the tick
    def step_tick(self):
        dt = self.dt
        decide = self.tick % self.decide_every == 0
        self.t = round((self.tick + 1) * dt, 6)
        self.tick += 1
        self._update_zone()
        zx, zy, zr, zdps = self.zone_live
        alive = [m for m in self.mechs if m is not None and m.alive]
        for mk in alive:
            self._offer_tick(mk)
            st = mk.st
            # aim slew
            dy_ = wrap(mk.cmd_yaw - mk.yaw)
            lim = st["turn_rate"] * dt
            mk.yaw = wrap(mk.yaw + max(-lim, min(lim, dy_)))
            dp = mk.cmd_pitch - mk.pitch
            lim = st["pitch_rate"] * dt
            mk.pitch += max(-lim, min(lim, dp))
            self._weapons_tick(mk)
        for mk in alive:
            self._move_mech(mk)
            mk.jump = False if mk.grounded is False else mk.jump
            st = mk.st
            if self.t - mk.last_spend >= st["energy_regen_delay"]:
                mk.energy = min(st["energy"], mk.energy + st["energy_regen"] * dt)
            mk.in_zone = math.hypot(mk.x - zx, mk.y - zy) < zr
            if not mk.in_zone:
                mk.hp -= zdps * dt
                mk.zone_pending += zdps * dt
                mk.zone_taken += zdps * dt
                mk.last_combat = self.t
                if mk.hp <= 0 and mk.killing_blow is None:
                    mk.killing_blow = (None, "zone", None)
            elif self.t - mk.last_combat >= st["regen_delay"]:
                mh = self.max_hp(mk)
                if mk.hp < mh:
                    mk.hp = min(mh, mk.hp + st["regen_per_s"] * dt)
            sp = math.hypot(mk.vx, mk.vy)
            if mk.grounded and sp > self.cfg["sound"]["step_speed"]:
                mk.step_t += dt
                if mk.step_t >= self.cfg["sound"]["step_every"] - 1e-9:
                    mk.step_t = 0.0
                    self.sound("boost" if mk.boosting else "step", mk.x, mk.y, mk.z, mk.id)
            if mk.jetting:
                mk.jet_t += dt
                if mk.jet_t >= 0.3 - 1e-9:
                    mk.jet_t = 0.0
                    self.sound("jet", mk.x, mk.y, mk.z, mk.id)
        self._push_apart(alive)
        for n in self.npcs:
            if n.alive:
                self._npc_tick(n, decide)
        self._proj_tick()
        self._cubes_tick(alive)
        self._drops_tick(alive)
        self._deaths()
        if decide:
            for key in [k for k, f in self.fights.items() if self.t - f["last"] > 10.0]:
                self._close_fight(key, "disengage", None)

    def _push_apart(self, alive):
        n = len(alive)
        for i in range(n):
            a = alive[i]
            ra = self.radius(a)
            for j in range(i + 1, n):
                b = alive[j]
                dx, dy = b.x - a.x, b.y - a.y
                if abs(dx) > 12 or abs(dy) > 12:
                    continue
                rb = self.radius(b)
                d = math.hypot(dx, dy)
                if d >= ra + rb or abs(a.z - b.z) > max(self.height(a), self.height(b)):
                    continue
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                push = (ra + rb - d) / 2
                ux, uy = dx / d, dy / d
                for m, s in ((a, -1), (b, 1)):
                    nx_, ny_ = m.x + s * ux * push, m.y + s * uy * push
                    if self._blocked(nx_, ny_, m.z, self.radius(m), self.height(m), m.st["step_up"],
                                     self._offsets(self.radius(m))) == 0:
                        m.pvx += (nx_ - m.x) / self.dt      # positions only; vx/vy stay (World.vel_h)
                        m.pvy += (ny_ - m.y) / self.dt
                        m.x, m.y = nx_, ny_

    def vel_h(self, mk):
        """The horizontal velocity `mk` reports in observations (`self.vel` and `mechs[].vel`, x and y).

        Two things move a mech AFTER `_move_mech` integrated its velocity state (vx, vy), and neither writes
        that state: the mech-mech push-apart and the eave slide (0.6 m per tick off a roof edge). The tick
        sums what they moved, per second, in mk.pvx/pvy (reset at the start of `_move_mech`).

        `--push-vel actual` (default): state + (pvx, pvy) = the displacement the mech really made this tick
        / dt. Walls are the model: `_move_mech` leaves vx * dt, vy * dt equal to the step that was really
        taken (slide modes zero or rewrite the blocked part), so on a tick with no push and no slide the
        state already IS the displacement and this returns it unchanged, bit for bit.
        `--push-vel keep`: the state alone, as before the option existed: two mechs walking into each other
        both report full walking speed while neither moves. Which one the real server does is unknown.

        Only the REPORTED value changes; the state (vx, vy) is deliberately not rewritten, unlike at walls:
        - a wall clips the integration step itself, so its rewrite can only shrink the speed. A push is a
          position correction that is not bounded by the mech's own speed: a mech that lands on another is
          moved up to a radius (2.5 m) in one tick, which is 50 m/s. Written into the state that would be an
          impulse the next ticks integrate (a launch of tens of metres at air_accel 12): a third physics
          hypothesis that protocol.md ("Overlapping mechs are pushed apart") gives no ground for;
        - with the state untouched both options run the same physics: the same actions give bit-identical
          positions, and the options differ in the `vel` fields alone (and only on ticks with a push or a
          slide), so a keep-vs-actual comparison tests the velocity reading and nothing else.
        Everything the simulator itself derives from speed (step sounds, the shot records' `moving` flags)
        keeps reading the state, so it is the same under both options as well.

        Not covered, on purpose: vz is the vertical state under both (0 while grounded, also on a slope or a
        step-up; the recorded real stream, check/stream.jsonl.gz, does the same), and a mortar knock is an
        impulse on the state that shows in `vel` at once and is integrated from the next tick on (it is not
        a displacement). NPCs set their velocity only on a tick they really moved and are never pushed.
        """
        if self.opts["push_vel"] == "keep" or not (mk.pvx or mk.pvy):
            return mk.vx, mk.vy
        return mk.vx + mk.pvx, mk.vy + mk.pvy

    def _cubes_tick(self, alive):
        cc = self.cfg["cubes"]
        rad = cc["pickup_radius"]
        for it in [i for i in self.items.values() if i.kind == "cube"]:
            best, bd = None, None
            for mk in alive:
                dx, dy = mk.x - it.x, mk.y - it.y
                if abs(dx) > rad or abs(dy) > rad:
                    continue
                d = math.hypot(dx, dy)
                if d <= rad and abs(mk.z - it.z) <= 4.0 and (bd is None or d < bd - 1e-9 or (abs(d - bd) <= 1e-9 and mk.id < best.id)):
                    best, bd = mk, d
            if best is not None:
                del self.items[it.id]
                self.give_cubes(best, 1)
                best.events.append({"type": "pickup", "cubes": best.cubes})

    def _drops_tick(self, alive):
        dc = self.cfg["drops"]
        times = dc["times"]
        for k, tc in enumerate(times):
            if abs(self.t - tc) < self.dt / 2 and not any(d.t_call == tc for d in self.drops):
                ph = self.zone_phase()
                c, r = (ph["c1"], ph["r1"]) if ph else (self.phases[-1]["c1"], 0.0)
                x, y, z = self.geo.random_open(self.rng, c[0], c[1], max(1.0, r * dc.get("spread", 0.7)), 20)
                self.drops.append(Drop(self.next_drop, x, y, z, tc, tc + dc["fall_s"]))
                self.next_drop += 1
                self.rec.write({"type": "drop_call", "t": r2(self.t), "id": self.next_drop - 1, "pos": [r2(x), r2(y)]})
        for d in self.drops:
            if d.taken:
                continue
            if not d.landed:
                if self.t >= d.t_land - 1e-9:
                    d.landed = True
                    self.sound("drop", d.x, d.y, d.z, None)
                continue
            inside = [m for m in alive if math.hypot(m.x - d.x, m.y - d.y) <= dc["radius"] and -3 <= m.z - d.z <= 10]
            n_in = len(inside)
            if self.opts["drop_npc_blocks"]:
                n_in += sum(1 for n in self.npcs if n.alive and math.hypot(n.x - d.x, n.y - d.y) <= dc["radius"])
            if n_in == 0:
                d.holder, d.progress = None, 0.0
            elif n_in == 1 and len(inside) == 1:
                m = inside[0]
                if d.holder != m.id:
                    d.holder, d.progress = m.id, 0.0
                d.progress += self.dt
                if d.progress >= dc["capture_s"] - 1e-9:
                    d.taken = True
                    self.give_cubes(m, int(dc["cubes"]))
                    if dc.get("repair", 1):
                        m.hp = self.max_hp(m)
                        m.energy = m.st["energy"]
                    m.events.append({"type": "drop", "id": d.id, "cubes": m.cubes})
                    self.rec.write({"type": "drop_capture", "t": r2(self.t), "id": d.id, "by": m.id})
        self.drops = [d for d in self.drops if not d.taken]

    def _deaths(self):
        dead = [m for m in self.mechs if m is not None and m.alive and m.hp <= 0]
        if not dead:
            return
        n_alive = sum(1 for m in self.mechs if m is not None and m.alive)
        cc = self.cfg["cubes"]
        for m in dead:
            m.alive = False
            m.place = n_alive
            m.death_t = self.t
            kb = m.killing_blow or (None, "unknown", None)
            killer = self.mechs[kb[0]] if kb[0] is not None else None
            if killer is not None:
                killer.kills += 1
                killer.events.append({"type": "kill", "target": m.id})
            for k in range(m.cubes + int(cc["death_drop_extra"])):
                a = self.rng.random() * TAU
                rr = 1.0 + self.rng.random() * 2.0
                self.add_item("cube", m.x + rr * math.cos(a), m.y + rr * math.sin(a), m.z)
            for slot, w in enumerate(m.weapons):
                if w is not None and w.kind != "gun":
                    w.charging, w.charge, w.release, w.hold = False, 0.0, False, 0.0
                    self.add_pickup(w.kind, m.x + (slot - 0.5) * 2, m.y, m.z, weapon=w)
            m.weapons = [None, None]
            self.sound("death", m.x, m.y, m.z, m.id)
            self.rec.write({"type": "death", "t": r2(self.t), "id": m.id, "cause": kb[1], "by": kb[2],
                            "killer": kb[0], "place": n_alive, "pos": [r2(m.x), r2(m.y), r2(m.z)],
                            "killer_loadout": killer.loadout() if killer else None,
                            "zone_taken": r2(m.zone_taken)})
            for key in [k for k in self.fights if m.id in k]:
                other = key[0] if key[1] == m.id else key[1]
                self._close_fight(key, "kill", other)
        self.deaths_this_tick = dead
        self.order.append((self.tick, [m.id for m in dead]))
        alive = [m for m in self.mechs if m is not None and m.alive]
        if len(alive) <= 1:
            for m in alive:
                m.place = 1
            self.over = True

    def finish_clock(self):
        """Clock ran out: rank survivors by HP (ties shared)."""
        alive = [m for m in self.mechs if m is not None and m.alive]
        for m in alive:
            m.place = 1 + sum(1 for o in alive if o.hp > m.hp + 1e-9)
        for key in list(self.fights):
            self._close_fight(key, "timeout", None)
        self.over = True

    def step_decision(self):
        for _ in range(self.decide_every):
            self.step_tick()
            if self.over:
                break
        if not self.over and self.t >= self.duration - 1e-9:
            self.finish_clock()

    # ------------------------------------------------------------------ vision / obs
    def _sees_point(self, o, ex, ey, ez, px, py, pzc, pzt, need_los=True):
        st = o.st
        vc = self.cfg["vision"]
        dx, dy, dz = px - ex, py - ey, pzc - ez
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d <= vc["near_range"] + st.get("near_range_bonus", 0):
            mode, can = "near", False
        elif d <= vc["range"] + st.get("vision_bonus", 0):
            if abs(wrap(math.atan2(dy, dx) - o.yaw)) > math.radians(vc["fov_deg"]) / 2:
                return None
            mode, can = "wedge", bool(vc.get("canopy_blocks", True))
        else:
            return None
        if not need_los:
            return mode
        g = self.geo
        if g.los(ex, ey, ez, px, py, pzt, can) or g.los(ex, ey, ez, px, py, pzc, can):
            return mode
        return None

    def build_obs(self, mk):
        t = self.t
        vc = self.cfg["vision"]
        ex, ey, ez = mk.x, mk.y, self.eye(mk)
        mechs_out = []
        vis_ids = set()
        for o in self.mechs:
            if o is None or o is mk or not o.alive:
                continue
            h = self.height(o)
            mode = self._sees_point(mk, ex, ey, ez, o.x, o.y, o.z + h / 2, o.z + h)
            if mode:
                mk.seen[o.id] = t
                if o.id not in mk.ever_seen:
                    mk.ever_seen.add(o.id)
                    self.rec.write({"type": "sight", "t": r2(t), "by": mk.id, "target": o.id, "mode": mode,
                                    "dist": r2(math.dist((ex, ey, ez), (o.x, o.y, o.z + h / 2))),
                                    "hdist": r2(math.hypot(o.x - mk.x, o.y - mk.y)),
                                    "target_loadout": o.loadout()})
            if t - mk.seen.get(o.id, -99) <= vc["memory_s"] + 1e-9:
                vis_ids.add(o.id)
                ovx, ovy = self.vel_h(o)
                ent = {"id": o.id, "kind": "mech", "pos": [r2(o.x), r2(o.y), r2(o.z)],
                       "vel": [r2(ovx), r2(ovy), r2(o.vz)], "aim": [round(o.yaw, 3), round(o.pitch, 3)]}
                if vc.get("show_enemy_hp", True):
                    ent["hp"] = r2(max(0.0, o.hp))
                    ent["max_hp"] = r2(self.max_hp(o))
                ent["cubes"] = o.cubes
                ent["weapons"] = o.loadout()
                ent["charging"] = o.laser_charging()
                mechs_out.append(ent)
        nh = self.cfg["npc"]["height"]
        for n in self.npcs:
            if not n.alive:
                continue
            if self._sees_point(mk, ex, ey, ez, n.x, n.y, n.z + nh / 2, n.z + nh):
                mk.seen[n.id] = t
            if t - mk.seen.get(n.id, -99) <= vc["memory_s"] + 1e-9:
                ent = {"id": n.id, "kind": "npc", "pos": [r2(n.x), r2(n.y), r2(n.z)], "vel": [r2(n.vx), r2(n.vy), 0],
                       "aim": [round(n.yaw, 3), 0]}
                if vc.get("show_enemy_hp", True):
                    ent["hp"] = r2(max(0.0, n.hp))
                    ent["max_hp"] = n.max_hp
                mechs_out.append(ent)
        mk.vis_ids = vis_ids
        mk.vis_known = True
        crates, cubes, pickups = [], [], []
        cache = mk.item_cache
        for it in self.items.values():
            top = 2.4 if it.kind == "crate" else 0.8
            mode = self._sees_point(mk, ex, ey, ez, it.x, it.y, it.z + top / 2, it.z + top, need_los=False)
            if not mode:
                continue
            c = cache.get(it.id)
            if c is None or t > c[0] or abs(c[2] - ex) + abs(c[3] - ey) > 4.0 or c[4] != (mode == "wedge"):
                can = mode == "wedge" and bool(vc.get("canopy_blocks", True))
                ok = self.geo.los(ex, ey, ez, it.x, it.y, it.z + top, can) or \
                    self.geo.los(ex, ey, ez, it.x, it.y, it.z + top / 2, can)
                c = (t + 0.5, ok, ex, ey, mode == "wedge")
                cache[it.id] = c
            if not c[1]:
                continue
            if it.kind == "crate":
                crates.append({"id": it.id, "pos": [r2(it.x), r2(it.y), r2(it.z)], "hp": r2(it.hp)})
            elif it.kind == "cube":
                cubes.append({"id": it.id, "pos": [r2(it.x), r2(it.y), r2(it.z)]})
            else:
                pickups.append({"id": it.id, "kind": it.kind, "pos": [r2(it.x), r2(it.y), r2(it.z)]})
        projs = []
        for p in self.projs:
            if p.owner == mk.id:
                continue
            if self._sees_point(mk, ex, ey, ez, p.x, p.y, p.z, p.z):
                projs.append({"id": p.id, "kind": p.kind, "pos": [r2(p.x), r2(p.y), r2(p.z)],
                              "vel": [r2(p.vx), r2(p.vy), r2(p.vz)]})
        sounds = []
        sc = self.cfg["sound"]
        noise = math.radians(sc["bearing_noise_deg"]) * mk.st.get("hearing", 1.0)
        for (ts, kind, sx, sy, sz, src) in self.sounds:
            if src == mk.id:
                continue
            d = math.hypot(sx - mk.x, sy - mk.y)
            if d > sc["range"].get(kind, 0):
                continue
            b = math.atan2(sy - mk.y, sx - mk.x) + self.rng.gauss(0, noise)
            band = "near" if d < sc["near"] else ("mid" if d < sc["mid"] else "far")
            sounds.append({"kind": kind, "bearing": round(wrap(b), 3), "dist": band, "age": r2(t - ts + self.dt)})
        beams = []
        for (tb, by, kind, path) in self.beams:
            if by == mk.id:
                continue
            pts = list(path)
            for a, b in zip(path, path[1:]):
                pts.append([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2])
            if any(self._sees_point(mk, ex, ey, ez, p[0], p[1], p[2], p[2]) for p in pts):
                beams.append({"by": by, "kind": kind, "path": path, "age": r2(t - tb + self.dt)})
        events = mk.events
        if mk.zone_pending > 0:
            events.append({"type": "damaged", "amount": r2(mk.zone_pending), "weapon": "zone", "bearing": None, "by": None})
            mk.zone_pending = 0.0
        mk.events = []
        drops = [{"id": d.id, "pos": [r2(d.x), r2(d.y), r2(d.z)], "radius": self.cfg["drops"]["radius"],
                  "lands_in": r2(max(0.0, d.t_land - t)), "capture_s": self.cfg["drops"]["capture_s"],
                  "progress": r2(d.progress), "holder": d.holder} for d in self.drops]
        return {"type": "obs", "tick": self.tick, "t": r2(t),
                "alive": sum(1 for m in self.mechs if m is not None and m.alive),
                "self": self.self_obs(mk), "mechs": mechs_out, "projectiles": projs, "crates": crates,
                "cubes": cubes, "pickups": pickups, "sounds": sounds, "beams": beams, "events": events,
                "zone": self.zone_obs(), "offer": mk.offer, "drops": drops}

    def self_obs(self, mk):
        s = self.scale(mk)
        st = mk.st
        stats = {"radius": r2(st["radius"] * s), "height": r2(st["height"] * s),
                 "eye_height": r2(st["eye_height"] * s), "muzzle_height": r2(st["muzzle_height"] * s)}
        for k, v in st.items():
            if k not in stats:
                stats[k] = round(v, 4) if isinstance(v, float) else v
        weps = []
        for w in mk.weapons:
            if w is None:
                weps.append(None)
                continue
            ws = mk.wst[w.kind]
            e = {"kind": w.kind, "cooldown": r2(max(0.0, w.cd, w.equip)),
                 "stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ws.items()}}
            if w.kind == "gun":
                e["ammo"] = w.ammo
                e["reload"] = r2(max(0.0, w.reload))
            elif w.kind == "laser":
                e["charge"] = r2(w.charge if w.charging else 0.0)
            weps.append(e)
        svx, svy = self.vel_h(mk)
        return {"id": mk.id, "pos": [r2(mk.x), r2(mk.y), r2(mk.z)], "vel": [r2(svx), r2(svy), r2(mk.vz)],
                "aim": [round(mk.yaw, 3), round(mk.pitch, 3)], "hp": r2(max(0.0, mk.hp)), "max_hp": r2(self.max_hp(mk)),
                "energy": r2(mk.energy), "max_energy": r2(st["energy"]), "grounded": mk.grounded,
                "cubes": mk.cubes, "upgrades": list(mk.upg_list), "weapons": weps, "in_zone": mk.in_zone,
                "stats": stats}

    def end_decision(self):
        """Clear per-decision broadcast buffers (sounds, beams) after every bot got its obs."""
        self.sounds = []
        self.beams = []


def _num(v, i, default=0.0):
    try:
        x = float(v[i])
    except (TypeError, ValueError, IndexError):
        return default
    return x if math.isfinite(x) else default


# ============================================================================ processes
class BotProc:
    def __init__(self, slot, argv, env_extra, stderr_path, inbox):
        self.slot = slot
        self.argv = argv
        env = dict(os.environ)
        env.update(env_extra or {})
        self.err = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
        self.alive = True
        self.inbox = inbox
        self.out = queue.Queue()
        try:
            self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                      bufsize=0, env=env)
        except OSError as e:
            sys.stderr.write(f"slot {slot}: cannot start {argv}: {e}\n")
            self.p = None
            self.alive = False
            return
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._write, daemon=True).start()

    def _read(self):
        try:
            for raw in self.p.stdout:
                self.inbox.put((self.slot, time.perf_counter(), raw))
        except (OSError, ValueError):
            pass
        self.inbox.put((self.slot, time.perf_counter(), None))

    def _write(self):
        while True:
            data = self.out.get()
            if data is None:
                try:
                    self.p.stdin.close()
                except OSError:
                    pass
                return
            try:
                self.p.stdin.write(data)
                self.p.stdin.flush()
            except (OSError, ValueError):
                self.alive = False

    def send(self, msg):
        if self.p is not None:
            self.out.put((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    def close(self):
        if self.p is not None:
            self.out.put(None)

    def kill(self):
        if self.p is not None and self.p.poll() is None:
            try:
                self.p.kill()
            except OSError:
                pass
        if self.err not in (None, subprocess.DEVNULL):
            try:
                self.err.close()
            except OSError:
                pass


def expand_spec(spec, kit, idx=0):
    """Bot spec -> (argv, env_extra, label)."""
    spec = spec.strip()
    if spec == "example":
        return ([sys.executable, "-u", os.path.join(kit, "python", "example_bot.py")],
                {"PYTHONPATH": os.path.join(kit, "python")}, "example")
    if spec.startswith("scripted:"):
        arch = spec.split(":", 1)[1] or "mix"
        env, suffix = {}, ""
        if arch.endswith("_v1"):  # original (pre-calibration) fire-on-sight behaviour
            arch, env, suffix = arch[:-3] or "mix", {"MINISIM_BOTS_PROFILE": "v1"}, "_v1"
        if arch == "mix":
            arch = MIX[idx % len(MIX)]
        return ([sys.executable, "-u", os.path.join(HERE, "minisim_bots.py"), arch], env, "scripted:" + arch + suffix)
    env = {}
    argv = shlex.split(spec)
    while argv and "=" in argv[0] and not argv[0].startswith(("/", ".")) and argv[0].split("=")[0].isidentifier():
        k, v = argv.pop(0).split("=", 1)
        env[k] = v
    return argv, env, spec


def run_match(a):
    kit = os.path.abspath(a.kit)
    hello0 = load_hello(kit)
    cfg = hello0["config"]
    if a.duration:
        cfg["match"]["duration_s"] = a.duration
    if a.budget_ms:
        cfg["bot"]["tick_budget_ms"] = a.budget_ms
    specs = list(a.bot or [])
    n = a.mechs or int(cfg["match"]["mechs"])
    fill = a.fill or "scripted:mix"
    while len(specs) < n:
        specs.append(fill)
    specs = specs[:n]
    cfg["match"]["mechs"] = n
    rng = random.Random(a.seed)
    order = list(range(n))
    if not a.no_shuffle:
        rng.shuffle(order)
    os.makedirs(a.out, exist_ok=True)
    arena = os.path.join(kit, "arena")
    t_geo = time.perf_counter()
    geo = get_geo(arena)
    t_geo = time.perf_counter() - t_geo
    rec = Recorder(os.path.join(a.out, "match.jsonl"))
    world = World(cfg, geo, seed=a.seed, n=n,
                  opts=dict(cone=a.cone, mortar_stack=a.mortar_stack, drop_npc_blocks=a.drop_npc_blocks,
                            peace=a.peace, push_vel=a.push_vel), rec=rec)
    inbox = queue.Queue()
    procs = {}
    slot_specs = {}
    mix_i = 0
    for i, spec in enumerate(specs):
        slot = order[i]
        is_mix = spec.strip().endswith((":mix", ":mix_v1"))
        argv, env, label = expand_spec(spec, kit, mix_i if is_mix else 0)
        if is_mix:
            mix_i += 1
        slot_specs[slot] = (spec, label, argv)
        procs[slot] = BotProc(slot, argv, env, os.path.join(a.out, f"slot{slot:02d}.stderr"), inbox)
    rec.write({"type": "meta", "seed": a.seed, "n": n, "duration": cfg["match"]["duration_s"],
               "opts": world.opts, "geo_load_s": round(t_geo, 2),
               "slots": [{"slot": s, "spec": slot_specs[s][0], "label": slot_specs[s][1]} for s in range(n)],
               "zone": [{"t0": p["t0"], "ss": p["ss"], "se": p["se"], "c1": [r2(p["c1"][0]), r2(p["c1"][1])],
                         "r1": p["r1"]} for p in world.phases]})
    cb = cfg["bot"]
    rng_seeds = random.Random(a.seed * 7919 + 1)
    for s in range(n):
        procs[s].send({"type": "hello", "protocol": "mechfight/1", "slot": s, "seed": rng_seeds.getrandbits(31),
                       "map": {"dir": arena, "files": ["v2/world.json", "v2/terrain.i16"]}, "config": cfg})
    names = {s: None for s in range(n)}
    stats = {s: {"answered": 0, "missed": 0, "late": 0, "stale": 0, "bad": 0, "times": [], "exited": False}
             for s in range(n)}
    deadline = time.perf_counter() + cb["hello_budget_ms"] / 1000.0
    while any(v is None for s, v in names.items() if procs[s].alive) and time.perf_counter() < deadline:
        try:
            slot, at, raw = inbox.get(timeout=max(0.0, deadline - time.perf_counter()))
        except queue.Empty:
            break
        if raw is None:
            procs[slot].alive = False
            stats[slot]["exited"] = True
            continue
        try:
            m = json.loads(raw)
        except ValueError:
            stats[slot]["bad"] += 1
            continue
        if isinstance(m, dict) and m.get("type") == "ready":
            nm = "".join(ch for ch in str(m.get("name", "")) if ch.isprintable())[:32]
            names[slot] = nm or "bot"
    t_start = time.perf_counter()
    budget = cb["tick_budget_ms"] / 1000.0
    ended = set()
    if not a.quiet:
        print(f"[minisim] seed {a.seed} n {n} geo {t_geo:.1f}s; ready: "
              + ", ".join(f"{s}:{names[s]}" for s in range(n)), file=sys.stderr)
    last_print = time.perf_counter()
    while not world.over:
        sent_at = {}
        expect = {}
        for mk in world.mechs:
            if mk is None or not mk.alive:
                continue
            obs = world.build_obs(mk)
            pr = procs[mk.id]
            if pr.alive:
                pr.send(obs)
                sent_at[mk.id] = time.perf_counter()
                expect[mk.id] = obs["tick"]
        world.end_decision()
        actions = {}
        dl = time.perf_counter() + budget
        while expect:
            if a.no_timeout:
                timeout = 30.0
            else:
                timeout = dl - time.perf_counter()
                if timeout <= 0:
                    break
            try:
                slot, at, raw = inbox.get(timeout=timeout)
            except queue.Empty:
                if a.no_timeout:
                    continue
                break
            if raw is None:
                procs[slot].alive = False
                stats[slot]["exited"] = True
                expect.pop(slot, None)
                continue
            if len(raw.rstrip(b"\n")) > cb["max_line_bytes"]:
                stats[slot]["bad"] += 1
                continue
            try:
                m = json.loads(raw)
            except ValueError:
                stats[slot]["bad"] += 1
                continue
            if not isinstance(m, dict) or m.get("type") == "ready":
                continue
            if slot in expect and m.get("tick") == expect[slot]:
                if a.no_timeout or at - sent_at[slot] <= budget:
                    actions[slot] = m
                    stats[slot]["answered"] += 1
                    stats[slot]["times"].append(at - sent_at[slot])
                    del expect[slot]
                else:
                    stats[slot]["late"] += 1
            else:
                stats[slot]["stale"] += 1
        for slot in expect:
            stats[slot]["missed"] += 1
        takes, drops = [], []
        for mk in world.mechs:
            if mk is None or not mk.alive:
                continue
            act = actions.get(mk.id)
            world.apply_action(mk, act)
            if act is not None:
                tk = act.get("take")
                dr = act.get("drop")
                dr = int(dr) if isinstance(dr, (int, float)) and not isinstance(dr, bool) else None
                if isinstance(tk, (int, float)) and not isinstance(tk, bool):
                    takes.append((mk, int(tk), dr))
                elif dr is not None:
                    drops.append((mk, dr))
        world.resolve_loot(takes, drops)
        world.step_decision()
        for mk in world.mechs:
            if mk is not None and not mk.alive and mk.id not in ended:
                ended.add(mk.id)
                procs[mk.id].send({"type": "end", "place": mk.place, "kills": mk.kills})
                procs[mk.id].close()
        if not a.quiet and time.perf_counter() - last_print > 15:
            last_print = time.perf_counter()
            print(f"[minisim] t={world.t:.0f}s alive={sum(1 for m in world.mechs if m.alive)} "
                  f"wall={time.perf_counter() - t_start:.0f}s", file=sys.stderr)
    for mk in world.mechs:
        if mk.id not in ended:
            ended.add(mk.id)
            procs[mk.id].send({"type": "end", "place": mk.place, "kills": mk.kills})
            procs[mk.id].close()
    wall = time.perf_counter() - t_start
    places = []
    for mk in world.mechs:
        s = stats[mk.id]
        ts = sorted(s["times"])
        pct = (lambda q: round(ts[min(len(ts) - 1, int(q * len(ts)))] * 1000, 2) if ts else None)
        rec_ = {"type": "place", "id": mk.id, "place": mk.place, "score": 16 - mk.place if n == 16 else n - mk.place,
                "kills": mk.kills, "spec": slot_specs[mk.id][0], "label": slot_specs[mk.id][1],
                "name": names[mk.id], "hp": r2(max(0.0, mk.hp)), "cubes": mk.cubes, "loadout": mk.loadout(),
                "upgrades": mk.upg_list, "dmg_dealt": r2(mk.dmg_dealt), "dmg_taken": r2(mk.dmg_taken),
                "zone_taken": r2(mk.zone_taken), "t_death": mk.death_t, "answered": s["answered"],
                "missed": s["missed"], "late": s["late"], "stale": s["stale"], "bad_lines": s["bad"],
                "exited_early": s["exited"] and mk.id not in ended, "resp_ms_p50": pct(0.5), "resp_ms_p99": pct(0.99),
                "resp_ms_max": pct(1.0)}
        rec.write(rec_)
        places.append(rec_)
    rec.write({"type": "summary", "wall_s": round(wall, 2), "t_end": r2(world.t), "ticks": world.tick,
               "realtime_x": round(world.t / wall, 2) if wall > 0 else None})
    rec.close()
    deadline = time.time() + 3
    for p in procs.values():
        if p.p is not None:
            try:
                p.p.wait(timeout=max(0.05, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pass
        p.kill()
    summ = {"seed": a.seed, "wall_s": round(wall, 2), "t_end": r2(world.t), "places": places}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    if not a.quiet:
        print(f"[minisim] done: match t={world.t:.1f}s wall={wall:.1f}s", file=sys.stderr)
        for p in sorted(places, key=lambda r: r["place"]):
            print(f"  place {p['place']:2d}  slot {p['id']:2d}  {p['label'][:40]:40s} kills {p['kills']} "
                  f"hp {p['hp']:.0f} missed {p['missed']}", file=sys.stderr)
    return summ


def build_argparser():
    ap = argparse.ArgumentParser(description="Local approximate mechfight/1 match server.")
    ap.add_argument("--out", required=True, help="output directory for match.jsonl, slotNN.stderr, summary.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bot", action="append", help="bot spec (repeatable): a command line, 'example' or 'scripted:<arch>'")
    ap.add_argument("--fill", default="scripted:mix", help="spec used for the remaining slots (default scripted:mix)")
    ap.add_argument("--mechs", type=int, default=0, help="number of mechs (default config match.mechs = 16)")
    ap.add_argument("--duration", type=float, default=0, help="override match.duration_s")
    ap.add_argument("--budget-ms", type=float, default=0, help="override bot.tick_budget_ms")
    ap.add_argument("--no-timeout", action="store_true", help="wait for every answer (debugging)")
    ap.add_argument("--no-shuffle", action="store_true", help="bots take slots in the order given")
    ap.add_argument("--cone", choices=["full", "half"], default="full",
                    help="lightning cone reading: +-cone_deg (full, default) or +-cone_deg/2 (A-19)")
    ap.add_argument("--mortar-stack", action="store_true", help="direct mortar hits also take splash (A-2)")
    ap.add_argument("--drop-npc-blocks", action="store_true", help="NPCs count as drop-ring occupants (A-6)")
    ap.add_argument("--peace", action="store_true",
                    help="no weapon or drone damage to mechs: every mech lives until the zone takes it, so "
                         "the match is a pure test of rotation and endgame positioning")
    ap.add_argument("--push-vel", choices=list(PUSH_VEL_MODES), default="actual",
                    help="what `vel` reports on a tick with a mech-mech push-apart or an eave slide: the "
                         "displacement really made in the tick / dt (actual, default) or the untouched velocity "
                         "state (keep, the behaviour before this option). Positions are the same either way; "
                         "which one the real server does is unknown")
    ap.add_argument("--kit", default=KIT_DEFAULT, help="mechfight-kit directory")
    ap.add_argument("--quiet", action="store_true")
    return ap


if __name__ == "__main__":
    run_match(build_argparser().parse_args())
