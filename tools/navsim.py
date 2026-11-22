#!/usr/bin/env python3
"""Single-mech kinematic micro-sim on the real map (plan 3.2 / 14.2) to validate nav.py.

    python3 tools/navsim.py [--pairs 200] [--seed 1] [--map-dir ../mechfight-kit/arena] [-v]

Model (protocol section 7, defaults from hello.config.mech; 20 Hz ticks, an action holds 2 ticks):
- horizontal velocity accelerates toward move * speed (walk_speed, boost_speed on the ground while
  boosting) at ground_accel / air_accel;
- the mech is a cylinder (radius R) standing on the 1 m nav surface under its centre (surface_h +
  low bridge decks / overhangs, nav.nav_surface_1m); terrain heights are bilinear;
- walls: a building/deck column sampled on the body rim (16 points at R, 8 at R/2) higher than
  feet + step_up (grounded) or feet (airborne) stops the move; the move slides along the wall (the
  component into the wall normal is removed), map bounds are walls;
- ledges up to step_up are walked up; terrain slopes steeper than max_slope_deg cannot be walked up;
- jump (grounded) sets vz = jump_speed; boost in the air jets (jet_accel against gravity, rise capped
  at max_rise_speed, jet_drain energy/s); boost on the ground drains boost_drain/s; energy regenerates
  energy_regen/s after energy_regen_delay s without spending;
- roof edges: rim columns at or below the feet support the mech (it lands on / stands on an edge while
  any part overhangs it, falls when the whole body is off);
- the underside of a bridge / overhang you can walk under stops a jet.
Not modelled: other mechs, eave slide-out, the exact server collision shapes.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mapdata as M  # noqa: E402
import nav as NAV  # noqa: E402
from contracts import Clock, SelfState, TickInfo, ZoneState  # noqa: E402

DEF_MECH = {"radius": 2.5, "height": 9.0, "eye_height": 8.0, "walk_speed": 11.7, "boost_speed": 19.5,
            "ground_accel": 58.5, "air_accel": 12.0, "gravity": 22.0, "jump_speed": 11.0,
            "jet_accel": 34.0, "max_rise_speed": 10.0, "step_up": 1.2, "max_slope_deg": 50.0,
            "energy": 100.0, "energy_regen": 18.0, "energy_regen_delay": 0.6, "boost_drain": 17.0,
            "jet_drain": 15.0}
TICK = 0.05
COLUMN_CM = 30           # 1 m nav cell more than this above terrain = building / deck column


class NavSim:
    def __init__(self, md, stats=None):
        self.md = md
        self.st = dict(DEF_MECH)
        if stats:
            self.st.update({k: float(v) for k, v in stats.items() if k in DEF_MECH})
        self.nav1, self.ceil = NAV.nav_surface_1m(md)
        self.R = self.st["radius"]
        self.rim = []
        for k in range(16):
            a = k * math.pi / 8
            self.rim.append((self.R * math.cos(a), self.R * math.sin(a)))
        for k in range(8):
            a = k * math.pi / 4 + math.pi / 8
            self.rim.append((0.55 * self.R * math.cos(a), 0.55 * self.R * math.sin(a)))
        self.tan_slope = math.tan(math.radians(self.st["max_slope_deg"]))
        self.reset((0.0, 0.0))

    # ------------------------------------------------------------------ state
    def reset(self, xy, energy=None):
        x, y = xy[0], xy[1]
        self.x, self.y = x, y
        self.z = self.support(x, y, 1e9 if len(xy) < 3 else xy[2] + 3.0)
        self.vx = self.vy = self.vz = 0.0
        self.grounded = True
        self.energy = self.st["energy"] if energy is None else energy
        self.t = 0.0
        self.last_spend = -10.0
        self.jet_time = 0.0
        self.energy_used = 0.0

    def self_state(self):
        st = self.st
        me = SelfState(id=1, pos=(self.x, self.y, self.z), vel=(self.vx, self.vy, self.vz), hp=1400,
                       max_hp=1400, energy=self.energy, max_energy=st["energy"], grounded=self.grounded,
                       stats=dict(st), radius=self.R)
        me.speed_h = math.hypot(self.vx, self.vy)
        return me

    # ------------------------------------------------------------------ geometry
    def column(self, x, y):
        """(height m, is_column) of the 1 m nav cell; None outside the grid / bounds."""
        md = self.md
        b = md.bounds
        if not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
            return None
        i = int(x - md.gx0)
        j = int(y - md.gy0)
        if not (0 <= i < md.nx and 0 <= j < md.ny):
            return None
        c = j * md.nx + i
        h = self.nav1[c]
        if h - md.terr[c] > COLUMN_CM:
            return h * 0.01, True
        return md.terrain_h(x, y), False

    def support(self, x, y, zlim):
        """Highest surface under the body whose top is <= zlim (centre: any; rim: columns only)."""
        col = self.column(x, y)
        best = col[0] if col is not None else -1e9
        if col is not None and best > zlim:
            best = -1e9
            # centre over a wall top we are below: stand on what the rim offers
        for ox, oy in self.rim:
            c = self.column(x + ox, y + oy)
            if c is None or not c[1]:
                continue
            if best < c[0] <= zlim:
                best = c[0]
        if best < -1e8:
            best = self.md.terrain_h(x, y)
        return best

    def blocked(self, x, y, z, grounded):
        """Wall test for the body at (x, y) with feet z: (blocked, normal or None, n_hits)."""
        lim = z + (self.st["step_up"] if grounded else 0.05)
        nx = ny = 0.0
        hits = 0
        for ox, oy in self.rim:
            c = self.column(x + ox, y + oy)
            if c is None or (c[1] and c[0] > lim):
                hits += 1
                nx -= ox
                ny -= oy
        c = self.column(x, y)
        if c is None:
            return True, None, 99
        if c[0] > lim:
            hits += 4
        if not hits:
            return False, None, 0
        ln = math.hypot(nx, ny)
        return True, ((nx / ln, ny / ln) if ln > 1e-9 else None), hits

    def slope_blocked(self, x0, y0, x1, y1, z0):
        """Terrain steeper than max_slope_deg going up (centre on terrain at both ends)."""
        c0 = self.column(x0, y0)
        c1 = self.column(x1, y1)
        if c0 is None or c1 is None or c0[1] or c1[1]:
            return False
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-6:
            return False
        rise = c1[0] - max(c0[0], z0 - 0.05)
        return rise > 0.05 and rise / d > self.tan_slope

    def _ok(self, x, y, z, grounded, x0, y0):
        b, n, hits = self.blocked(x, y, z, grounded)
        if b:
            # already overlapping a wall (spawned / pushed there): moves that do not dig deeper are
            # allowed, so the body works itself free ("you never end up inside a wall")
            b0, _, h0 = self.blocked(x0, y0, z, grounded)
            if not (b0 and hits <= h0 and h0 < 99):
                return False, n
        if grounded and self.slope_blocked(x0, y0, x, y, z):
            return False, None
        return True, None

    # ------------------------------------------------------------------ step
    def step(self, move, boost=False, jump=False):
        st = self.st
        dt = TICK
        mx, my = move if move is not None else (0.0, 0.0)
        ml = math.hypot(mx, my)
        if ml > 1.0:
            mx, my = mx / ml, my / ml
            ml = 1.0
        spent = False
        # vertical intent
        if self.grounded and jump:
            self.vz = st["jump_speed"]
            self.grounded = False
        if not self.grounded:
            g = st["gravity"]
            if boost and self.energy > 0.0:
                cap = st["max_rise_speed"]
                v_new = self.vz + (st["jet_accel"] - g) * dt
                self.vz = min(v_new, max(self.vz - g * dt, cap))
                use = st["jet_drain"] * dt
                self.energy = max(0.0, self.energy - use)
                self.energy_used += use
                self.jet_time += dt
                spent = True
            else:
                self.vz -= g * dt
        # horizontal velocity
        spd = st["walk_speed"]
        if self.grounded and boost and ml > 0.05 and self.energy > 0.0:
            spd = st["boost_speed"]
            use = st["boost_drain"] * dt
            self.energy = max(0.0, self.energy - use)
            self.energy_used += use
            spent = True
        tx, ty = mx * spd, my * spd
        acc = st["ground_accel"] if self.grounded else st["air_accel"]
        dvx, dvy = tx - self.vx, ty - self.vy
        dl = math.hypot(dvx, dvy)
        if dl > acc * dt:
            dvx, dvy = dvx / dl * acc * dt, dvy / dl * acc * dt
        self.vx += dvx
        self.vy += dvy
        # horizontal move with wall slide
        x0, y0 = self.x, self.y
        nx_, ny_ = x0 + self.vx * dt, y0 + self.vy * dt
        if nx_ != x0 or ny_ != y0:
            ok, n = self._ok(nx_, ny_, self.z, self.grounded, x0, y0)
            if ok:
                self.x, self.y = nx_, ny_
            else:
                moved = False
                cands = []
                if n is not None:
                    dot = self.vx * n[0] + self.vy * n[1]
                    if dot < 0:
                        cands.append((self.vx - dot * n[0], self.vy - dot * n[1]))
                    # grazing a corner / post: deflect away from it while keeping the tangent motion
                    sp = math.hypot(self.vx, self.vy)
                    tvx, tvy = self.vx - min(dot, 0.0) * n[0], self.vy - min(dot, 0.0) * n[1]
                    cands.append((tvx + n[0] * 0.35 * sp, tvy + n[1] * 0.35 * sp))
                cands.append((self.vx, 0.0))
                cands.append((0.0, self.vy))
                for svx, svy in cands:
                    if abs(svx) + abs(svy) < 1e-6:
                        continue
                    sx, sy = x0 + svx * dt, y0 + svy * dt
                    ok2, _ = self._ok(sx, sy, self.z, self.grounded, x0, y0)
                    if ok2:
                        self.x, self.y = sx, sy
                        self.vx, self.vy = svx, svy
                        moved = True
                        break
                if not moved:
                    self.vx = self.vy = 0.0
        # vertical
        if self.grounded:
            s = self.support(self.x, self.y, self.z + st["step_up"])
            if s < self.z - 0.6:
                self.grounded = False
                self.vz = 0.0
            else:
                self.z = s
        if not self.grounded:
            zprev = self.z
            self.z += self.vz * dt
            # underside of a walk-under bridge / overhang stops the jet
            md = self.md
            i = int(self.x - md.gx0)
            j = int(self.y - md.gy0)
            u = self.ceil.get(j * md.nx + i) if self.ceil else None
            if u is not None and zprev + st["height"] <= u * 0.01 + 0.01 and self.z + st["height"] > u * 0.01:
                self.z = u * 0.01 - st["height"]
                self.vz = min(self.vz, 0.0)
            s = self.support(self.x, self.y, max(zprev, self.z) + 0.05)
            if self.vz <= 0.0 and self.z <= s:
                self.z = s
                self.vz = 0.0
                self.grounded = True
        # energy regen
        if spent:
            self.last_spend = self.t
        elif self.t - self.last_spend >= st["energy_regen_delay"]:
            self.energy = min(st["energy"], self.energy + st["energy_regen"] * dt)
        self.t += dt

    def decide(self, move, boost=False, jump=False):
        """One decision = 2 ticks with the same action."""
        self.step(move, boost, jump)
        self.step(move, boost, False if not self.grounded else jump)


# ======================================================================= driver
class FakeWorld:
    """The subset of world.WorldModel that nav reads."""

    def __init__(self, me, t, zone=None):
        self.me = me
        self.tick = TickInfo(t=t)
        self.zone = zone or ZoneState(center=(0.0, 0.0), radius=1e6, next_center=(0.0, 0.0),
                                      next_radius=1e6, shrink_start=1e9, shrink_end=1e9,
                                      from_center=(0.0, 0.0), from_radius=1e6)
        self.tracks = {}
        self.ghosts = {}
        self.damage_log = []

    def heat_at(self, x, y):
        return 0.0


class _A:
    slack = float("inf")
    hostiles = []
    playbook = "PB_G"


def drive(nav, sim, goal, max_t, key="goal", cost=None, budget_ms=30.0, on_tick=None, arrive_r=4.0,
          zone=None):
    """Follow nav's path to `goal` in the sim. Returns dict(success, t, reason, energy_used, ...)."""
    t = sim.t
    cmd = None
    info = {"success": False, "t": 0.0, "reason": "timeout", "unstick": 0, "jumps": 0, "climbs": 0}
    t_end = sim.t + max_t
    was_climbing = False
    while sim.t < t_end:
        me = sim.self_state()
        world = FakeWorld(me, sim.t, zone)
        if math.hypot(goal[0] - sim.x, goal[1] - sim.y) < arrive_r:
            info.update(success=True, reason="arrived")
            break
        nav.path_request(key, me.pos, goal, cost or {}, sim.t)
        t0 = time.perf_counter()
        clock = Clock(t0, budget_ms, budget_ms + 8.0)
        nav.anytime(world, _A, clock)
        steer = nav.follow(me, key, goal)
        leg, jmp, bst = nav.stuck_update(world, cmd)
        move = steer.direction
        jump = steer.jump
        boost = steer.climb_boost
        if leg is not None:
            move = leg.direction
            info["unstick"] += 1
        if jmp is not None:
            jump = True
        if bst is not None:
            boost = bst.want
        if steer.arrived and math.hypot(goal[0] - sim.x, goal[1] - sim.y) < arrive_r:
            info.update(success=True, reason="arrived")
            break
        if jump:
            info["jumps"] += 1
        if nav.climbing() and not was_climbing:
            info["climbs"] += 1
        was_climbing = nav.climbing()
        cmd = move
        sim.decide(move, boost, jump)
        if on_tick:
            on_tick(sim, steer)
    info["t"] = sim.t - t
    info["energy_used"] = sim.energy_used
    info["pos"] = (sim.x, sim.y, sim.z)
    return info


def random_pairs(md, nav, n, rng, dmin=40.0, dmax=250.0, allow_roof=True):
    """n (start, goal) pairs of walkable nav4 cell centres, goal reachable by a complete A*."""
    N = md.nx4 * md.ny4
    cells = [c for c in range(N) if md.nav_flags(c) & M.NAV_WALKABLE
             and (allow_roof or not md.nav_flags(c) & M.NAV_ROOF)
             and not md.nav_flags(c) & (M.NAV_DECK | M.NAV_UNDERPASS)]
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        a = rng.choice(cells)
        b = rng.choice(cells)
        pa, pb = md.cell4_xy(a), md.cell4_xy(b)
        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        if not (dmin <= d <= dmax):
            continue
        # start point must actually be standable for the body (not wedged against a wall)
        if md.clearance(pa[0], pa[1]) < 2.0 or md.clearance(pb[0], pb[1]) < 1.0:
            continue
        nav.path_request("_probe", (pa[0], pa[1], md.nav_h(a)), pb, {}, 0.0)
        nav.run_until_done(2000)
        p = nav.path("_probe")
        nav._paths.pop("_probe", None)
        if p is None or not p.complete:
            continue
        out.append(((pa[0], pa[1], md.nav_h(a)), pb, p.eta_s))
    return out


def run_pairs(md, C, n=200, seed=1, verbose=False, eta_k=1.5, eta_add=3.0):
    rng = random.Random(seed)
    stats = (C.cfg.get("mech") if C is not None and isinstance(getattr(C, "cfg", None), dict) else None)
    planner = NAV.Nav(C, md)
    pairs = random_pairs(md, planner, n, rng)
    ok = 0
    fails = []
    for k, (a, b, eta) in enumerate(pairs):
        nav = NAV.Nav(C, md)
        sim = NavSim(md, stats)
        sim.reset(a)
        res = drive(nav, sim, b, eta * eta_k + eta_add)
        if res["success"]:
            ok += 1
        else:
            fails.append((k, a, b, eta, res))
        if verbose:
            print("%3d %s eta %.1f t %.1f %s unstick %d jumps %d climbs %d" % (
                k, "OK  " if res["success"] else "FAIL", eta, res["t"], res["reason"], res["unstick"],
                res["jumps"], res["climbs"]), file=sys.stderr)
    return ok, len(pairs), fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--map-dir", default=os.path.join(os.path.dirname(ROOT), "mechfight-kit", "arena"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    import config
    C = config.bind(None)
    md = M.MapData.load(a.map_dir, C)
    if md.degraded:
        print("map degraded:", md.degraded_reason, file=sys.stderr)
    t0 = time.perf_counter()
    ok, n, fails = run_pairs(md, C, a.pairs, a.seed, a.verbose)
    print("navsim: %d / %d = %.1f%% in %.1f s" % (ok, n, 100.0 * ok / max(1, n), time.perf_counter() - t0),
          file=sys.stderr)
    for k, pa, pb, eta, res in fails[:20]:
        print("  fail %d %s -> %s eta %.1f: %s at %s" % (k, pa, pb, eta, res["reason"], res["pos"]),
              file=sys.stderr)
    return 0 if n and ok / n >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
