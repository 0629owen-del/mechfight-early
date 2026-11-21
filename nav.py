"""Navigation: flood, A*, path follower, stuck ladder, climbs, spots, urgent cover, zone field.

Plan refs (mechfight-kit/docs/game-plan-v2.md): 7.13 (R13.1-R13.6), 8.3 (loot flood), 8.4 (spot
ranking, urgent cover), 8.8 (zone time-to-safety field, slack), 8.9 (costs, A*, follower),
R10.2 / R13.2 / RZ4 (climbs), 3.3 (budgets). API: CONTRACTS.md 6.4.

Grid: the 4 m nav4 layer of mapdata (one level = lowest walkable surface; walk/jump edge bits in
NAV_DIRS order E, NE, N, NW, W, SW, S, SE) plus climb edges ground_cell -> roof_cell from the
precomputed climb faces (eave faces excluded). Costs are seconds:
    walk edge  = (len + SLOPE_COST_K * max(0, rise)) / speed          (len 4 or 5.66 m)
    jump edge  = walk + JUMP_EDGE_COST_S
    climb edge = dist / speed + h / max_rise_speed + CLIMB_EDGE_PAD_S, only when
                 energy >= CLIMB_COST(h) + reserve (CLIMB_COST(h) = CLIMB_COST_PER_M * h + CLIMB_COST_PAD)
    extras (cost dict): exposure (EXPOSURE_COST_S per cell in LOS of a laser/gun track within reach,
                 <= EXPOSURE_LOS_CHECKS LOS per search), heat (HEAT_COST_S * heat), zone
                 (ZONE_CELL_COST_S if outside the circle at predicted arrival), silent (SILENT_MOVE
                 speed, no climbs).

Work that can exceed a tick is resumable and runs only inside anytime() (zone field slice on its
own budget, then flood / A* on NAV_MS_CAP, then spot ranking on RANK_MS_CAP); the urgent cover
query is the only search run inside the caller's tick (its own Slice).

Choices where the contract / plan left room (see also the report in the hand-off):
- `cover_query` threats are FEET positions (Track.pos); the eye is z + 8 m (stats.eye_height of a
  0-cube mech). Passing eye points is merely stricter (eye + 8).
- The unstick LegRequest uses level "O5" (intent.unstick is its own field); the unstick jet uses
  BoostRequest owner "CLIMB" (BOOST_ORDER has no O5) and JumpRequest owner "O5".
- A slice that finishes its flood / A* job continues with the next queued job inside the same
  NAV_MS_CAP slice (same CPU cap, more throughput).
- A* uses a weighted octile heuristic (IMPL_ASTAR_W, default 1.0 = admissible).
- The zone field uses climb edges only when the safe set contains roof cells (R11.2 roof point).
- Juke (8.10) is not here: CONTRACTS 6.8 gives O7 / intent.juke to overlays.

mapdata (CONTRACTS 6.1) as actually used:
- hot loops read mapdata's raw nav4 arrays (_nh, _nf, _nw, _nj, _nu) when present, else the 6.1
  methods (degraded maps: lazy walk_edges / jump_edges through the methods);
- climb faces come from mapdata.climb_info (dicts or CLIMB_FIELDS tuples: stand x, y and landing
  rx, ry are needed; ClimbFace has neither). A face whose roof_cell / ground_cell level differs from
  the landing / stand height (a 4 m cell straddling two roofs) is re-anchored on the neighbour cell
  at that level, else dropped;
- there is no 1 m nav-surface blob: nav_surface_1m() rebuilds it from surface_h + overhangs (2 ms).
"""
from __future__ import annotations

import array
import heapq
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import mapdata as M
from contracts import (BoostRequest, JumpRequest, LegRequest, Path, Slice, Steer, Vec2, Vec3,
                       ZoneState)

INF = float("inf")
_now = time.perf_counter
SQ2 = math.sqrt(2.0)
NAV_STEP = 4.0
DIRS8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
DLEN = tuple(math.hypot(di, dj) * NAV_STEP for di, dj in DIRS8)
UNIT8 = tuple((di / math.hypot(di, dj), dj / math.hypot(di, dj)) for di, dj in DIRS8)
BITS8 = tuple(tuple(k for k in range(8) if (b >> k) & 1) for b in range(256))
F_WALK, F_ROOF, F_DECK, F_UNDER, F_EDGE = 1, 2, 4, 8, 16
COVER_WALK = 0x80
COVER_DIST = 0x3F
EYE_H = 8.0
CENTER_H = 4.5
MECH_R = 2.5
MECH_H = 9.0
SIDE_WALL = 2.4          # side sample this much above the centre line = wall beside (precompute rule)
CLIMB_START_R = 1.6      # start the climb sequence within this distance of the face stand point
CLIMB_TIMEOUT_S = 7.0
ZONE_SEED_BAND = 6.0     # S cells this close to the S boundary seed the reverse Dijkstra (> 4*sqrt2)

_DEF = {
    "NAV_MS_CAP": 10, "RANK_MS_CAP": 3, "ZONE_FIELD_MS": 8, "ZONE_FIELD_MIN_MS": 4, "ANYTIME_RESERVE_MS": 2,
    "DEGRADE_SCALE": 0.6,
    "ZONE_SOFT_MULT": 2, "ZONE_SLACK_SOFT_S": 15, "ZONE_SLACK_HARD_S": 4,
    "STEP_UP": 1.2, "JUMP_H": 2.75, "NAV_FLOOD_R": 150, "NAV_FLOOD_REFRESH_S": 2,
    "ASTAR_MAX_EXPANSIONS": 20000, "LOOKAHEAD_M": 6, "WAYPOINT_REACH_M": 2.5, "NAV_CLEAR_PAD": 0.3,
    "NAV_HEAD_PAD": 0.3, "SLOPE_COST_K": 1.0, "JUMP_EDGE_COST_S": 0.5, "CLIMB_EDGE_PAD_S": 0.8,
    "EXPOSURE_COST_S": 0.5, "EXPOSURE_LOS_CHECKS": 120, "IMPL_EXPOSURE_CACHE_M": 8,
    "HEAT_COST_S": 2, "ZONE_CELL_COST_S": 5, "SILENT_MOVE": 0.41,
    "STUCK_SPEED": 1.5, "STUCK_T": 0.8, "STUCK_MOVE_MIN": 0.5, "UNSTUCK_OK_SPEED": 3,
    "UNSTUCK_OK_S": 0.5, "UNSTUCK_STEP_S": 0.6, "UNSTUCK_JET_MAX_S": 1.5, "UNSTUCK_JET_MIN_ENERGY": 20,
    "NAV_BLOCK_TTL_S": 30, "NAV_RECOVER_MAX_S": 4, "STUCK_REPEAT_N": 3, "STUCK_REPEAT_WINDOW_S": 10,
    "ESCAPE_MIN_M": 10, "ESCAPE_MAX_M": 20,
    "CLIMB_COST_PER_M": 1.5, "CLIMB_COST_PAD": 3, "CLIMB_NAV_RESERVE": 20, "CLIMB_MAX_H": 51,
    "CLIMB_CLEAR_M": 0.5, "CLIMB_ABORT_ENERGY": 5, "JUMP_TRIGGER_DIST_M": 1.5,
    "ROOF_MIN_H": 10, "ROOF_MAX_H": 30, "ROOF_MIN_W": 20, "ROOF_ESCAPE_RESERVE": 30,
    "ROOF_INSIDE_M": 15, "ROOF_NEIGHBOR_R": 15, "ROOF_W_NEIGHBOR": 3, "ROOF_W_CENTER": 2,
    "ROOF_SIGHT_NORM": 200, "ROOF_W_CONTACT": 2, "ROOF_CONTACT_MEMORY_S": 60,
    "ZONE_FIELD_MARGIN_M": 20, "ZONE_TINY_R": 6, "ZONE_FALLBACK_DETOUR_K": 1.4,
    "ZONE_EDGE_MARGIN_M": 10, "ZONE_ETA_SAFETY": 1.15, "IMPL_BISECT_STEPS": 20,
    "COVER_SEARCH_R": 25, "COVER_URGENT_MS": 8, "COVER_URGENT_CANDIDATES": 64,
    "COVER_W_OUTSIDE": 0.5, "COVER_W_HEAT": 0.3, "COVER_W_FRESH": 1.5,
    "SPOT_CANDIDATES": 200, "ETA_NORM_S": 30, "HOLD_WALL_R": 6, "HOLD_W_COVER": 2.0,
    "HOLD_W_SIGHT": 1.5, "HOLD_W_CENTER": 1.0, "HOLD_W_TRAVEL": 1.0, "HOLD_W_DANGER": 0.5,
    "SIGHT_NORM_Z": 150, "SIGHT_TARGET_L": 60, "SIGHT_NORM_G": 120, "NOOK_SIGHT_NORM": 100,
    "NOOK_W_HEAT": 0.5, "SPOT_CROWD_R": 20, "LEG_PROBE_M": 3, "PATH_DEVIATION_M": 6,
    "IMPL_LEG_DIRS": 16, "LOOT_TIME_PAD_S": 1, "INFER_RANGE_GUN": 150, "NPC_CELL_COST_S": 3,
    "IMPL_ASTAR_W": 1.0, "FALL_BRAKE_ENABLED": False, "FALL_SAFE_DROP_M": 6, "FALL_COST_S": 1.0,
}
_RANGE_DEF = {"gun": 230.0, "laser": 200.0, "lightning": 40.0, "mortar": 235.0}


# ======================================================================= module helpers
def nav_surface_1m(md) -> Tuple[array.array, Dict[int, int]]:
    """1 m nav surface (cm) = surface_h + bridge decks / overhangs too low to walk under (solid up
    to their top), exactly as precompute builds clear_dm / nav4. Also returns {1 m cell: underside
    cm} for overhang cells you can walk under (their underside stops a jet). Shared with navsim."""
    nav1 = getattr(md, "_nav1", None)
    nav = array.array("h", nav1 if nav1 is not None else md.surf)
    ceil: Dict[int, int] = {}
    nx, ny = md.nx, md.ny
    for o in sorted(getattr(md, "overhangs", None) or (), key=lambda o: o.get("base", 0.0)):
        try:
            base = int(round(o["base"] * 100))
            top = int(round(o["top"] * 100))
            spans = M.scan_spans(o["ring"], o.get("holes"), md.tminx, md.tminy, nx, ny)
            for j, i0, i1 in spans:
                row = j * nx
                for c in range(row + i0, row + i1):
                    if base - nav[c] < M.HEAD_CM:
                        if top > nav[c]:
                            nav[c] = top
                    else:
                        u = ceil.get(c)
                        if u is None or base < u:
                            ceil[c] = base
        except (KeyError, TypeError, ValueError):
            continue
    return nav, ceil


def tau_edge(zone: ZoneState, p, t: float, steps: int = 20) -> float:
    """8.8: earliest tau >= t with |p - c(tau)| > r(tau) (t if already outside, +inf if never)."""
    def outside(tau):
        cx, cy = zone.center_at(tau)
        return math.hypot(p[0] - cx, p[1] - cy) > zone.radius_at(tau)
    if outside(t):
        return t
    end = zone.shrink_end
    if end <= t or not outside(end):
        return INF
    lo, hi = t, end
    for _ in range(max(1, int(steps))):
        mid = 0.5 * (lo + hi)
        if outside(mid):
            hi = mid
        else:
            lo = mid
    return hi


class _LazyEdges:
    """Degraded maps compute edges lazily through mapdata.walk_edges / jump_edges."""
    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, c):
        return self.fn(c)


class _Search:
    """One resumable forward Dijkstra / A* over nav4 (flood, path or local cover search)."""
    __slots__ = ("kind", "key", "dist", "prev", "closed", "etype", "face", "heap", "goal_cell",
                 "gi", "gj", "hs", "oi", "oj", "rad2", "inv", "jcost", "pen", "climbs", "energy",
                 "reserve", "max_climb_h", "done", "found", "exp", "max_exp", "best", "best_h",
                 "origin", "origin_cell", "goal", "start", "radius", "created_t", "cost")

    def __init__(self, N):
        self.dist = array.array("d", [INF]) * N
        self.prev = array.array("i", [-1]) * N
        self.closed = bytearray(N)
        self.etype = bytearray(N)
        self.face = {}
        self.heap = []
        self.goal_cell = -1
        self.gi = self.gj = 0
        self.hs = 0.0
        self.oi = self.oj = 0
        self.rad2 = 0
        self.pen = None
        self.climbs = None
        self.done = False
        self.found = False
        self.exp = 0
        self.max_exp = 1 << 30
        self.best = -1
        self.best_h = INF


class _PathJob:
    __slots__ = ("key", "goal", "goal_cell", "start", "cost", "search", "path", "cells", "idx",
                 "req_t", "created_t", "climb_at", "jump_at", "stale", "fail", "kinds", "raw",
                 "raw_stop", "out_stop")

    def __init__(self, key):
        self.key = key
        self.search = None
        self.path = None
        self.cells = []
        self.idx = 0
        self.climb_at = {}
        self.jump_at = set()
        self.stale = False
        self.fail = False
        self.kinds = []
        self.raw = ((), (), {})
        self.raw_stop = 0
        self.out_stop = 0


class _ZoneField:
    __slots__ = ("sx", "sy", "sr", "tiny", "rx", "ry", "rr", "d", "closed", "inr", "heap", "phase",
                 "row", "rows", "climbs", "scells", "inv", "pops")

    def __init__(self, N):
        self.d = array.array("d", [INF]) * N
        self.closed = bytearray(N)
        self.inr = bytearray(N)
        self.heap = []
        self.phase = "mark"
        self.row = 0
        self.rows = (0, -1)
        self.climbs = False
        self.scells = None
        self.pops = 0


# ======================================================================= Nav
class Nav:
    def __init__(self, C, mapdata, rng: Optional[random.Random] = None) -> None:
        self.C = C
        self.md = md = mapdata
        self.rng = rng if rng is not None else random.Random(0)
        for name, v in _DEF.items():
            setattr(self, "_" + name, v)
        self._load_constants()
        self.NX, self.NY = md.nx4, md.ny4
        self.N = self.NX * self.NY
        self.gx0, self.gy0 = md.min_x4, md.min_y4
        self._H = getattr(md, "_nh", None) or array.array("h", [int(round(md.nav_h(c) * 100)) for c in range(self.N)])
        self._F = getattr(md, "_nf", None) or bytes(md.nav_flags(c) for c in range(self.N))
        self._U = getattr(md, "_nu", None) or bytes(int(md.underpass_clear(c) * 10) for c in range(self.N))
        if getattr(md, "_nav_lazy", False) or getattr(md, "_nw", None) is None:
            self._W = _LazyEdges(md.walk_edges)
            self._J = _LazyEdges(md.jump_edges)
        else:
            self._W, self._J = md._nw, md._nj
        self._OFF = tuple(di + dj * self.NX for di, dj in DIRS8)
        # climb edges: ground cell -> [(roof cell, h, face dict)], best (lowest) face per pair
        self._climb_from: Dict[int, list] = {}
        self._climb_to: Dict[int, list] = {}
        self._faces_by_roof: Dict[int, list] = {}
        # 1 m nav surface for probes / smoothing (~2 ms, built on first use)
        self._nav1 = None
        self._ceil1 = None
        self._build_climbs()
        self._clear1 = getattr(md, "clear", None) if not md.degraded else None
        # live mech numbers
        st = (C.cfg.get("mech") if isinstance(getattr(C, "cfg", None), dict) else None) or {}
        self.walk = float(st.get("walk_speed", 11.7))
        self.rise = float(st.get("max_rise_speed", 10.0))
        self.max_energy = float(st.get("energy", 100.0))
        self.radius = float(st.get("radius", MECH_R))
        self.height = float(st.get("height", MECH_H))
        self.eye_h = float(st.get("eye_height", EYE_H))
        self.energy = self.max_energy
        self.s = 1.0
        self._pos = (0.0, 0.0, 0.0)
        self._t = 0.0
        self._world = None
        self._zone: Optional[ZoneState] = None
        self._size_check = False
        self._cst = None
        # blocked edges (c * 8 + k) -> until t
        self._blocked: Dict[int, float] = {}
        # flood
        self._flood: Optional[_Search] = None        # running
        self._flood_done: Optional[_Search] = None   # latest complete
        # paths
        self._paths: Dict[str, _PathJob] = {}
        self._nav_turn = 0
        self._exp_cache: Dict[tuple, bool] = {}
        # zone fields: [current-circle field (only while outside), main field]
        self._zfields: List[_ZoneField] = []
        self._zmain: Optional[_ZoneField] = None
        self._zcur: Optional[_ZoneField] = None
        # climb in progress
        self._climb = None
        self._hop = None
        # stuck ladder
        self._stuck = {"active": False, "slow_since": None, "ok_since": None, "start": 0.0,
                       "dir": (1.0, 0.0), "steps": [], "episodes": [], "pos0": None,
                       "escape": None, "blocked": False, "jet_top": 0.0, "side_first": 1,
                       "step_name": None}
        self._last_follow_key = None
        # spots
        self._spot_jobs: Dict[str, object] = {}
        self._spots: Dict[str, tuple] = {}
        self._spot_wanted: Dict[str, float] = {}
        self._spot_turn = 0
        self.exceptions = 0
        self.degrade_level = 0           # R15.3, kept current by main (slice caps x DEGRADE_SCALE^level)
        self.stats = {"flood_slices": 0, "astar_slices": 0, "zone_slices": 0, "rank_slices": 0}

    # ------------------------------------------------------------------ setup
    def _load_constants(self):
        C = self.C
        for name in _DEF:
            v = getattr(C, name, None) if C is not None else None
            if v is not None:
                setattr(self, "_" + name, v)

    def _build_climbs(self):
        md = self.md
        info = getattr(md, "climb_info", None)
        best: Dict[Tuple[int, int], dict] = {}
        fields = getattr(M, "CLIMB_FIELDS", None)
        if isinstance(info, dict):
            for rid, faces in info.items():
                for f in faces:
                    if not isinstance(f, dict):
                        if not fields:
                            continue
                        f = dict(zip(fields, f))     # tuple faces in mapdata.CLIMB_FIELDS order
                    if f.get("eave"):
                        continue
                    g, r = int(f["ground_cell"]), int(f["roof_cell"])
                    if not (0 <= g < self.N and 0 <= r < self.N) or g == r:
                        continue
                    rec = {"roof": int(rid), "g": g, "r": r, "h": float(f["h"]),
                           "x": float(f["x"]), "y": float(f["y"]),
                           "rx": float(f.get("rx", f["x"])), "ry": float(f.get("ry", f["y"])),
                           "wx": float(f.get("wx", f["x"])), "wy": float(f.get("wy", f["y"])),
                           "nx": float(f.get("nx", 0.0)), "ny": float(f.get("ny", 0.0))}
                    # the precomputed roof_cell / ground_cell is the nav4 cell of the landing / stand
                    # point; when that cell's level is another surface (cell straddles two roofs),
                    # re-anchor on the neighbour whose level matches, else drop the face
                    zl = self._h1(rec["rx"], rec["ry"])
                    zs = self._h1(rec["x"], rec["y"])
                    if zl is None or zs is None:
                        continue
                    g = self._level_cell(rec["x"], rec["y"], zs)
                    r = self._level_cell(rec["rx"], rec["ry"], zl)
                    if g < 0 or r < 0 or g == r:
                        continue
                    rec["g"], rec["r"] = g, r
                    rec["top"] = zl
                    rec["h"] = max(rec["h"], zl - zs)
                    self._faces_by_roof.setdefault(int(rid), []).append(rec)
                    k = (g, r)
                    if k not in best or rec["h"] < best[k]["h"]:
                        best[k] = rec
        for (g, r), rec in best.items():
            gx, gy = md.cell4_xy(g)
            rx, ry = md.cell4_xy(r)
            rec["d"] = math.hypot(rx - gx, ry - gy)
            self._climb_from.setdefault(g, []).append(rec)
            self._climb_to.setdefault(r, []).append(rec)

    def _level_cell(self, x, y, z):
        """Walkable nav4 cell at or next to (x, y) whose level is within STEP_UP of z (nearest), -1."""
        c0 = self._cell_raw(x, y)
        if c0 < 0:
            return -1
        F, H = self._F, self._H
        best, bd = -1, INF
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                c = c0 + di + dj * self.NX
                if not (0 <= c < self.N) or abs((c % self.NX) - (c0 % self.NX)) > 1:
                    continue
                if not F[c] & F_WALK or abs(H[c] * 0.01 - z) > self._STEP_UP:
                    continue
                cx, cy = self.cell_xy(c)
                d = math.hypot(cx - x, cy - y)
                if d < bd:
                    bd, best = d, c
        return best

    def _ensure_nav1(self):
        if self._nav1 is None:
            self._nav1, self._ceil1 = nav_surface_1m(self.md)
        return self._nav1

    def _sync_me(self, me):
        if me is None:
            return
        try:
            self._pos = tuple(me.pos)
            stt = me.stats or {}
            self.walk = float(stt.get("walk_speed", self.walk))
            self.rise = float(stt.get("max_rise_speed", self.rise))
            self.max_energy = float(me.max_energy or stt.get("energy", self.max_energy))
            self.energy = float(me.energy)
            r = float(me.radius or stt.get("radius", self.radius))
            h = float(stt.get("height", self.height))
            self.eye_h = float(stt.get("eye_height", self.eye_h))
            self.s = float(me.s or 1.0)
            if abs(r - self.radius) > 1e-6 or abs(h - self.height) > 1e-6:
                self.radius, self.height = r, h
                self._size_check = r > MECH_R + 0.05 or h + self._NAV_HEAD_PAD > MECH_H + 0.3 + 1e-6
                self._cst = bytearray(self.N) if self._size_check else None
        except (TypeError, ValueError, AttributeError):
            self.exceptions += 1

    def _sync_world(self, world):
        if world is None:
            return
        self._world = world
        try:
            self._t = float(world.tick.t)
        except AttributeError:
            pass
        z = getattr(world, "zone", None)
        if z is not None:
            self._zone = z
        self._sync_me(getattr(world, "me", None))

    def _cell_status(self, c):
        """R12.3: 1 ok / 2 blocked for the current size (clearance and underpass head room)."""
        md = self.md
        ok = True
        if self._F[c] & F_UNDER and self._U[c] * 0.1 < self.height + self._NAV_HEAD_PAD:
            ok = False
        if ok and self.radius > MECH_R + 0.05:
            x, y = md.cell4_xy(c)
            # the nav4 edge test assumes a 2.5 m body: need the extra radius around the centre
            if md.clearance(x, y) < (self.radius - MECH_R) + 1.0:
                ok = False
        self._cst[c] = 1 if ok else 2
        return self._cst[c]

    # ------------------------------------------------------------------ small geometry helpers
    def cell_xy(self, c) -> Vec2:
        return (self.gx0 + (c % self.NX + 0.5) * NAV_STEP, self.gy0 + (c // self.NX + 0.5) * NAV_STEP)

    def _cell_raw(self, x, y):
        I = int((x - self.gx0) // NAV_STEP)
        J = int((y - self.gy0) // NAV_STEP)
        if 0 <= I < self.NX and 0 <= J < self.NY:
            return J * self.NX + I
        return -1

    def walkable_cell(self, x, y, z=None, rings=2):
        """The nav4 cell under (x, y), or the nearest walkable one within `rings` (prefers matching
        height when z is given). -1 when none."""
        c = self._cell_raw(x, y)
        F, H = self._F, self._H
        if c >= 0 and F[c] & F_WALK and (z is None or abs(H[c] * 0.01 - z) < 3.0):
            return c
        I = int((x - self.gx0) // NAV_STEP)
        J = int((y - self.gy0) // NAV_STEP)
        best, bd = -1, INF
        for r in range(1, rings + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i2, j2 = I + di, J + dj
                    if not (0 <= i2 < self.NX and 0 <= j2 < self.NY):
                        continue
                    c2 = j2 * self.NX + i2
                    if not F[c2] & F_WALK:
                        continue
                    cx, cy = self.cell_xy(c2)
                    d = math.hypot(cx - x, cy - y)
                    if z is not None:
                        d += 2.0 * abs(H[c2] * 0.01 - z)
                    if d < bd:
                        bd, best = d, c2
            if best >= 0:
                return best
        if c >= 0 and F[c] & F_WALK:
            return c
        return -1

    def _h1(self, x, y):
        """1 m nav surface height (m) at (x, y); None outside the grid."""
        nav = self._nav1 if self._nav1 is not None else self._ensure_nav1()
        md = self.md
        i = int(x - md.gx0)
        j = int(y - md.gy0)
        if 0 <= i < md.nx and 0 <= j < md.ny:
            return nav[j * md.nx + i] * 0.01
        return None

    def _rise_ok(self, x0, y0, x1, y1, half_w, max_side=SIDE_WALL):
        """Walkable straight line: consecutive 1 m rises <= STEP_UP on the centre line and on two
        lines offset +-half_w, no side sample > SIDE_WALL above the centre sample, in bounds.
        Drops of any height are allowed (falling is free)."""
        nav = self._nav1 if self._nav1 is not None else self._ensure_nav1()
        md = self.md
        nx, ny, gx0, gy0 = md.nx, md.ny, md.gx0, md.gy0
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        if ln < 1e-6:
            return True
        n = int(ln) + 1
        ux, uy = dx / ln, dy / ln
        px, py = -uy * half_w, ux * half_w
        su = int(self._STEP_UP * 100) + 5
        sw = int(max_side * 100)
        prev = [None, None, None]
        b = md.bounds
        for s in range(n + 1):
            t = s / n
            x = x0 + dx * t
            y = y0 + dy * t
            if not (b[0] + 0.5 <= x <= b[2] - 0.5 and b[1] + 0.5 <= y <= b[3] - 0.5):
                return False
            cur = None
            for li, (ox, oy) in enumerate(((0.0, 0.0), (px, py), (-px, -py))):
                i = int(x + ox - gx0)
                j = int(y + oy - gy0)
                if not (0 <= i < nx and 0 <= j < ny):
                    return False
                h = nav[j * nx + i]
                if li == 0:
                    cur = h
                elif h - cur > sw:
                    return False
                p = prev[li]
                if p is not None and h - p > su:
                    return False
                prev[li] = h
        return True

    def _clear_ok(self, x0, y0, x1, y1, thr):
        """Fast path: every 1 m sample on the line has clear_dm >= thr."""
        cl = self._clear1
        if cl is None:
            return False
        md = self.md
        nx, ny, gx0, gy0 = md.nx, md.ny, md.gx0, md.gy0
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        n = int(ln) + 1
        t10 = int(thr * 10 + 0.999)
        for s in range(n + 1):
            t = s / n
            i = int(x0 + dx * t - gx0)
            j = int(y0 + dy * t - gy0)
            if not (0 <= i < nx and 0 <= j < ny) or cl[j * nx + i] < t10:
                return False
        return True

    def line_ok(self, a: Vec2, b: Vec2) -> bool:
        """Straight line a -> b walkable for our body: clearance grid (>= R + NAV_CLEAR_PAD) or the
        exact rise test (lets drops through)."""
        thr = self.radius + self._NAV_CLEAR_PAD
        if self._clear_ok(a[0], a[1], b[0], b[1], thr):
            return True
        return self._rise_ok(a[0], a[1], b[0], b[1], min(thr, self.radius))

    def clear_dir(self, pos: Vec3, direction: Vec2, dist: float) -> bool:
        """Leg probe (6.3): moving `dist` m along `direction` (plus the body radius) meets no wall
        (rise > STEP_UP between 1 m samples on the centre line or +-0.7 R), no map edge."""
        try:
            dx, dy = direction
            ln = math.hypot(dx, dy)
            if ln < 1e-9:
                return True
            dx, dy = dx / ln, dy / ln
            x0, y0 = pos[0], pos[1]
            L = dist + self.radius
            return self._rise_ok(x0, y0, x0 + dx * L, y0 + dy * L, 0.7 * self.radius, max_side=1e9)
        except (TypeError, ValueError):
            self.exceptions += 1
            return True

    def _ledge_ahead(self, pos, d):
        """Rise (m) of the first ledge within R + JUMP_TRIGGER_DIST_M along d (0 if none)."""
        z = self._h1(pos[0], pos[1])
        if z is None:
            return 0.0
        z = max(z, pos[2]) if len(pos) > 2 else z
        reach = self.radius + self._JUMP_TRIGGER_DIST_M
        s = 0.5
        best = 0.0
        while s <= reach + 1e-9:
            h = self._h1(pos[0] + d[0] * s, pos[1] + d[1] * s)
            if h is None:
                break
            r = h - z
            if r > self._STEP_UP + 0.05:
                return r
            best = max(best, r)
            s += 0.5
        return 0.0

    def _probe_dir(self, pos, want: Vec2, dist=None):
        """Clearance-filtered direction closest to `want` among IMPL_LEG_DIRS compass dirs (wall
        slide); (dir, blocked)."""
        dist = self._LEG_PROBE_M if dist is None else dist
        ln = math.hypot(want[0], want[1])
        if ln < 1e-9:
            return None, False
        w = (want[0] / ln, want[1] / ln)
        if self.clear_dir(pos, w, dist):
            return w, False
        n = int(self._IMPL_LEG_DIRS) or 16
        a0 = math.atan2(w[1], w[0])
        best = None
        for k in range(1, n // 2 + 1):
            for sgn in (1, -1):
                a = a0 + sgn * k * 2 * math.pi / n
                d = (math.cos(a), math.sin(a))
                if self.clear_dir(pos, d, dist):
                    best = d
                    break
            if best is not None:
                break
        if best is None:
            return w, True
        return best, False

    # ------------------------------------------------------------------ cost helpers
    def climb_cost(self, h: float) -> float:
        """CLIMB_COST(h) (energy)."""
        return self._CLIMB_COST_PER_M * h + self._CLIMB_COST_PAD

    def _make_pen(self, cost: dict):
        """Per-cell extra seconds from the cost dict (exposure / heat / zone), or None."""
        if not cost:
            return None
        tracks = cost.get("exposure_tracks") or []
        heat = bool(cost.get("heat")) and self._world is not None and hasattr(self._world, "heat_at")
        zone = self._zone if cost.get("zone") else None
        shooters = []
        for tr in tracks:
            kinds = tuple(getattr(tr, "weapons", ()) or ()) or tuple(getattr(tr, "known_weapons", {}) or ())
            reach = 0.0
            for k in kinds:
                if k in ("laser", "gun"):
                    reach = max(reach, self._weapon_range(k))
            if reach <= 0.0 and not kinds:
                reach = self._weapon_range("gun")
            if reach > 0.0:
                p = tr.pos
                q = self._IMPL_EXPOSURE_CACHE_M
                key = (round(p[0] / q), round(p[1] / q), round(p[2] / q))
                shooters.append((p[0], p[1], p[2] + self.eye_h * (1 + 0.05 * getattr(tr, "cubes", 0)), reach * reach, key))
        if not shooters and not heat and zone is None:
            return None
        md = self.md
        H = self._H
        cache: Dict[int, float] = {}
        budget = [int(self._EXPOSURE_LOS_CHECKS)]
        exp_cache = self._exp_cache
        if len(exp_cache) > 20000:
            exp_cache.clear()
        ecost, hcost, zcost = self._EXPOSURE_COST_S, self._HEAT_COST_S, self._ZONE_CELL_COST_S
        world = self._world
        t0 = self._t
        center_h = CENTER_H * self.s
        if zone is not None:      # c(t), r(t) of ZoneState inlined (hoisted out of the per-cell call)
            z_ss, z_se = zone.shrink_start, zone.shrink_end
            z_inv = 1.0 / (z_se - z_ss) if z_se > z_ss else 0.0
            z_fx, z_fy = zone.from_center
            z_dx, z_dy = zone.next_center[0] - z_fx, zone.next_center[1] - z_fy
            z_fr, z_dr = zone.from_radius, zone.next_radius - zone.from_radius

        def pen(v, g):
            p = cache.get(v)
            if p is None:
                x, y = self.cell_xy(v)
                p = 0.0
                if shooters:
                    z = H[v] * 0.01 + center_h
                    for sx, sy, sz, r2, key in shooters:
                        if (x - sx) ** 2 + (y - sy) ** 2 > r2:
                            continue
                        k2 = (key, v)
                        vis = exp_cache.get(k2)
                        if vis is None:
                            if budget[0] <= 0:
                                continue
                            budget[0] -= 1
                            vis = exp_cache[k2] = md.los((sx, sy, sz), (x, y, z))
                        if vis:
                            p += ecost
                            break
                if heat:
                    try:
                        p += hcost * float(world.heat_at(x, y))
                    except Exception:
                        pass
                cache[v] = p
            if zone is not None:
                x, y = self.cell_xy(v)
                ta = t0 + g
                f = 0.0 if ta <= z_ss else (1.0 if ta >= z_se or z_inv == 0.0 else (ta - z_ss) * z_inv)
                cx, cy = z_fx + z_dx * f, z_fy + z_dy * f
                r = z_fr + z_dr * f
                if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
                    return p + zcost
            return p
        return pen

    def _weapon_range(self, kind):
        try:
            v = self.C.path("weapons.%s.range" % kind)
            if v:
                return float(v)
        except Exception:
            pass
        return _RANGE_DEF.get(kind, 200.0)

    def _new_search(self, kind, start_cell, cost, radius=0.0, goal_cell=-1, goal=None):
        s = _Search(self.N)
        s.kind = kind
        s.cost = cost or {}
        speed = self.walk
        silent = bool(s.cost.get("silent"))
        if silent:
            speed *= self._SILENT_MOVE
        s.inv = 1.0 / max(0.5, speed)
        s.jcost = self._JUMP_EDGE_COST_S
        s.energy = float(s.cost.get("energy", self.energy if self.energy > 0 else self.max_energy))
        s.reserve = float(s.cost.get("reserve", self._CLIMB_NAV_RESERVE))
        s.max_climb_h = self._ROOF_MAX_H if kind == "flood" else self._CLIMB_MAX_H
        s.climbs = None if (silent or kind == "cover") else self._climb_from
        s.pen = self._make_pen(s.cost) if kind != "cover" else None
        s.origin_cell = start_cell
        s.oi, s.oj = start_cell % self.NX, start_cell // self.NX
        if radius > 0:
            r = radius / NAV_STEP
            s.rad2 = r * r
        s.radius = radius
        s.goal_cell = goal_cell
        s.goal = goal
        if goal_cell >= 0:
            s.gi, s.gj = goal_cell % self.NX, goal_cell // self.NX
            s.hs = NAV_STEP * s.inv * float(self._IMPL_ASTAR_W)
        s.dist[start_cell] = 0.0
        s.heap.append((0.0, start_cell))
        s.created_t = self._t
        return s

    def _run(self, s: _Search, sl: Optional[Slice]):
        """Advance a forward search until done or the slice expires (checked every 64 pops)."""
        H, W, J, OFF = self._H, self._W, self._J, self._OFF
        NX = self.NX
        dist, prev, closed, heap, et = s.dist, s.prev, s.closed, s.heap, s.etype
        inv, jc = s.inv, s.jcost
        K = self._SLOPE_COST_K * 0.01
        hs, gi, gj, goal = s.hs, s.gi, s.gj, s.goal_cell
        rad2, oi, oj = s.rad2, s.oi, s.oj
        pen = s.pen
        climbs = s.climbs
        blocked = self._blocked if self._blocked else None
        cst = self._cst if self._size_check else None
        cell_status = self._cell_status
        pop, push = heapq.heappop, heapq.heappush
        expired = sl.expired if sl is not None else None
        bits8, dlen = BITS8, DLEN
        sq2m2 = SQ2 - 2.0
        cmax = s.max_climb_h
        efree = s.energy - s.reserve
        ccost_pm, ccost_pad = self._CLIMB_COST_PER_M, self._CLIMB_COST_PAD
        crise, cpad = 1.0 / max(0.5, self.rise), self._CLIMB_EDGE_PAD_S
        n = 0
        while heap:
            f, c = pop(heap)
            if closed[c]:
                continue
            closed[c] = 1
            s.exp += 1
            g = dist[c]
            if hs:
                if c == goal:
                    s.found = True
                    s.done = True
                    s.best = c
                    s.best_h = 0.0
                    return
                hh = f - g
                if hh < s.best_h:
                    s.best_h = hh
                    s.best = c
            w = W[c]
            bits = w | J[c]
            if bits:
                hc = H[c]
                for k in bits8[bits]:
                    v = c + OFF[k]
                    if closed[v]:
                        continue
                    if blocked is not None and (c * 8 + k) in blocked:
                        continue
                    if cst is not None:
                        st = cst[v] or cell_status(v)
                        if st == 2:
                            continue
                    if rad2:
                        a = v % NX - oi
                        b = v // NX - oj
                        if a * a + b * b > rad2:
                            continue
                    dh = H[v] - hc
                    cost = ((dlen[k] + K * dh) if dh > 0 else dlen[k]) * inv
                    walk = (w >> k) & 1
                    if not walk:
                        cost += jc
                    if pen is not None:
                        cost += pen(v, g)
                    ng = g + cost
                    if ng < dist[v]:
                        dist[v] = ng
                        prev[v] = c
                        et[v] = 0 if walk else 1
                        if hs:
                            a = v % NX - gi
                            b = v // NX - gj
                            if a < 0:
                                a = -a
                            if b < 0:
                                b = -b
                            push(heap, (ng + (a + b + sq2m2 * (a if a < b else b)) * hs, v))
                        else:
                            push(heap, (ng, v))
            if climbs is not None:
                lst = climbs.get(c)
                if lst:
                    for rec in lst:
                        v = rec["r"]
                        h = rec["h"]
                        if closed[v] or h > cmax or ccost_pm * h + ccost_pad > efree:
                            continue
                        if rad2:
                            a = v % NX - oi
                            b = v // NX - oj
                            if a * a + b * b > rad2:
                                continue
                        ng = g + rec["d"] * inv + h * crise + cpad
                        if pen is not None:
                            ng += pen(v, g)
                        if ng < dist[v]:
                            dist[v] = ng
                            prev[v] = c
                            et[v] = 2
                            s.face[v] = rec
                            if hs:
                                a = abs(v % NX - gi)
                                b = abs(v // NX - gj)
                                push(heap, (ng + (a + b + sq2m2 * min(a, b)) * hs, v))
                            else:
                                push(heap, (ng, v))
            n += 1
            if s.exp >= s.max_exp:
                s.done = True
                return
            if expired is not None and (n & 63) == 0 and expired():
                return
        s.done = True

    # ================================================================== zone field (8.8)
    def zone_field_start(self, zone: ZoneState, outside_current: bool) -> None:
        """Restart the field(s) for this zone (call on next_changed). Work happens in anytime()."""
        try:
            self._zone = zone
            fields = []
            self._zcur = None
            if outside_current:
                f = self._zone_field_new(zone.center[0], zone.center[1], zone.radius, tiny=False, cur=True)
                self._zcur = f
                fields.append(f)
            m = self._ZONE_EDGE_MARGIN_M
            sr = zone.next_radius - m
            if sr < self._ZONE_TINY_R:
                f = self._zone_field_new(zone.next_center[0], zone.next_center[1], self._ZONE_TINY_R, tiny=True)
            else:
                f = self._zone_field_new(zone.next_center[0], zone.next_center[1], sr, tiny=False)
            self._zmain = f
            fields.append(f)
            self._zfields = fields
        except Exception:
            self.exceptions += 1
            self._zfields = []
            self._zmain = self._zcur = None

    def _zone_field_new(self, sx, sy, sr, tiny, cur=False):
        f = _ZoneField(self.N)
        f.sx, f.sy, f.sr, f.tiny = sx, sy, sr, tiny
        z = self._zone
        if cur:
            # S = inside the current circle; restriction = whole map (we are outside it)
            f.rx, f.ry, f.rr = sx, sy, 1e9
        else:
            f.rx, f.ry = z.center
            f.rr = max(z.radius, z.from_radius if z.from_radius else 0.0) + self._ZONE_FIELD_MARGIN_M
            if math.hypot(sx - f.rx, sy - f.ry) + sr > f.rr:
                f.rr = math.hypot(sx - f.rx, sy - f.ry) + sr + self._ZONE_FIELD_MARGIN_M
        f.inv = 1.0 / max(0.5, self.walk)
        if tiny:
            # S = walkable cells within ZONE_TINY_R of the point (grow until one exists)
            r = sr
            cells = []
            while r <= 64.0:
                cells = self._cells_in_disc(sx, sy, r)
                if cells:
                    break
                r += NAV_STEP
            f.sr = r
            f.scells = cells
            f.climbs = any(self._F[c] & F_ROOF for c in cells)
        J0 = max(0, int((f.ry - f.rr - self.gy0) // NAV_STEP))
        J1 = min(self.NY - 1, int((f.ry + f.rr - self.gy0) // NAV_STEP))
        f.rows = (J0, J1)
        f.row = J0
        return f

    def _cells_in_disc(self, x, y, r):
        out = []
        I0 = max(0, int((x - r - self.gx0) // NAV_STEP))
        I1 = min(self.NX - 1, int((x + r - self.gx0) // NAV_STEP))
        J0 = max(0, int((y - r - self.gy0) // NAV_STEP))
        J1 = min(self.NY - 1, int((y + r - self.gy0) // NAV_STEP))
        F = self._F
        for J in range(J0, J1 + 1):
            cy = self.gy0 + (J + 0.5) * NAV_STEP
            for I in range(I0, I1 + 1):
                c = J * self.NX + I
                cx = self.gx0 + (I + 0.5) * NAV_STEP
                if F[c] & F_WALK and (cx - x) ** 2 + (cy - y) ** 2 <= r * r:
                    out.append(c)
        return out

    def _zone_step(self, f: _ZoneField, sl: Optional[Slice]):
        NX = self.NX
        gx0, gy0 = self.gx0, self.gy0
        F = self._F
        d, closed, inr, heap = f.d, f.closed, f.inr, f.heap
        if f.phase == "mark":
            J0, J1 = f.rows
            rx, ry, rr = f.rx, f.ry, f.rr
            sx, sy, sr = f.sx, f.sy, f.sr
            sr2 = sr * sr
            band2 = max(0.0, sr - ZONE_SEED_BAND) ** 2
            tiny = f.tiny
            while f.row <= J1:
                J = f.row
                f.row += 1
                cy = gy0 + (J + 0.5) * NAV_STEP
                dy = cy - ry
                if rr < 1e8:
                    if abs(dy) > rr:
                        continue
                    half = math.sqrt(rr * rr - dy * dy)
                    I0 = max(0, int(math.ceil((rx - half - gx0) / NAV_STEP - 0.5)))
                    I1 = min(NX - 1, int(math.floor((rx + half - gx0) / NAV_STEP - 0.5)))
                else:
                    I0, I1 = 0, NX - 1
                row = J * NX
                dsy = (cy - sy) ** 2
                for I in range(I0, I1 + 1):
                    c = row + I
                    inr[c] = 1
                    if tiny or not F[c] & F_WALK:
                        continue
                    cx = gx0 + (I + 0.5) * NAV_STEP
                    dd = (cx - sx) ** 2 + dsy
                    if dd <= sr2:
                        d[c] = 0.0
                        if dd >= band2:
                            heap.append((0.0, c))
                        else:
                            closed[c] = 1
                if sl is not None and sl.expired():
                    return
            if tiny:
                for c in f.scells:
                    inr[c] = 1
                    d[c] = 0.0
                    heap.append((0.0, c))
            heapq.heapify(heap)
            f.phase = "run"
        if f.phase != "run":
            return
        H, W, J, OFF = self._H, self._W, self._J, self._OFF
        N = self.N
        inv = f.inv
        jc = self._JUMP_EDGE_COST_S
        K = self._SLOPE_COST_K * 0.01
        blocked = self._blocked if self._blocked else None
        cst = self._cst if self._size_check else None
        cell_status = self._cell_status
        pop, push = heapq.heappop, heapq.heappush
        dlen = DLEN
        climbs = self._climb_to if f.climbs else None
        crise, cpad = 1.0 / max(0.5, self.rise), self._CLIMB_EDGE_PAD_S
        cmax = min(self._CLIMB_MAX_H, (self.max_energy - self._CLIMB_NAV_RESERVE - self._CLIMB_COST_PAD)
                   / max(0.1, self._CLIMB_COST_PER_M))
        n = 0
        expired = sl.expired if sl is not None else None
        while heap:
            g, c = pop(heap)
            if closed[c] or g > d[c]:
                continue
            closed[c] = 1
            f.pops += 1
            hc = H[c]
            for k in range(8):
                v = c + OFF[k]
                if v < 0 or v >= N or closed[v] or not inr[v]:
                    continue
                kk = (k + 4) & 7
                wv = W[v]
                if not ((wv | J[v]) >> kk) & 1:
                    continue
                if blocked is not None and (v * 8 + kk) in blocked:
                    continue
                if cst is not None:
                    st = cst[v] or cell_status(v)
                    if st == 2:
                        continue
                dh = hc - H[v]
                cost = ((dlen[kk] + K * dh) if dh > 0 else dlen[kk]) * inv
                if not (wv >> kk) & 1:
                    cost += jc
                ng = g + cost
                if ng < d[v]:
                    d[v] = ng
                    push(heap, (ng, v))
            if climbs is not None:
                for rec in climbs.get(c, ()):
                    v = rec["g"]
                    if closed[v] or not inr[v] or rec["h"] > cmax:
                        continue
                    ng = g + rec["d"] * inv + rec["h"] * crise + cpad
                    if ng < d[v]:
                        d[v] = ng
                        push(heap, (ng, v))
            n += 1
            if expired is not None and (n & 63) == 0 and expired():
                return
        f.phase = "done"

    def zone_field_done(self) -> bool:
        return self._zmain is not None and self._zmain.phase == "done"

    def _field_for(self, pos):
        """(field, cell): the current-circle field while we are outside its S disc, else main."""
        f = self._zcur
        if f is not None and math.hypot(pos[0] - f.sx, pos[1] - f.sy) > f.sr:
            return f
        return self._zmain

    def _field_cell(self, f, pos):
        c = self.walkable_cell(pos[0], pos[1], pos[2] if len(pos) > 2 else None, rings=1)
        if c < 0:
            return -1
        return c

    def zone_eta(self, pos: Vec3) -> Tuple[float, bool]:
        """(ETA to the safe set incl. ZONE_ETA_SAFETY, settled). Unsettled -> 8.8 fallback."""
        f = self._zmain
        k = self._ZONE_ETA_SAFETY
        try:
            if f is None:
                z = self._zone
                if z is None:
                    return 0.0, False
                sr = max(z.next_radius - self._ZONE_EDGE_MARGIN_M, self._ZONE_TINY_R)
                dd = max(0.0, math.hypot(pos[0] - z.next_center[0], pos[1] - z.next_center[1]) - sr)
                return dd / max(0.5, self.walk) * self._ZONE_FALLBACK_DETOUR_K * k, False
            c = self._field_cell(f, pos)
            if c >= 0 and f.closed[c]:
                return f.d[c] * k, True
            dd = max(0.0, math.hypot(pos[0] - f.sx, pos[1] - f.sy) - f.sr)
            return dd / max(0.5, self.walk) * self._ZONE_FALLBACK_DETOUR_K * k, False
        except Exception:
            self.exceptions += 1
            return 0.0, False

    def in_safe_set(self, pos) -> bool:
        f = self._zmain
        if f is None:
            return True
        if f.tiny:
            c = self._cell_raw(pos[0], pos[1])
            return c in (f.scells or ())
        return math.hypot(pos[0] - f.sx, pos[1] - f.sy) <= f.sr

    def zone_descent(self, pos: Vec3) -> Optional[Vec2]:
        """Steepest-descent direction on the time-to-safety field; None inside S. Unsettled cell:
        8.8 fallback (A* path to the nearest S point when available, else the clearance-filtered
        straight line)."""
        try:
            f = self._field_for(pos)
            if f is None:
                return None
            if f is self._zmain and self.in_safe_set(pos):
                return None
            c = self._field_cell(f, pos)
            if c >= 0 and f.closed[c] and f.d[c] < INF:
                if f.d[c] <= 0.0:
                    # S cell of the current-circle field: hand over to the main field next call
                    return self._dir_to((f.sx, f.sy), pos) if f is self._zcur else None
                chain = [c]
                cur = c
                for _ in range(4):
                    nxt = self._descend(f, cur)
                    if nxt < 0:
                        break
                    chain.append(nxt)
                    cur = nxt
                    if f.d[nxt] <= 0.0:
                        break
                tgt = None
                for cc in reversed(chain[1:]):
                    p = self.cell_xy(cc)
                    if self.line_ok((pos[0], pos[1]), p):
                        tgt = p
                        break
                if tgt is None and len(chain) > 1:
                    tgt = self.cell_xy(chain[1])
                if tgt is not None:
                    d = self._dir_to(tgt, pos)
                    if d is not None:
                        return d
            # fallback (8.8)
            if f.tiny:
                target = (f.sx, f.sy)
            else:
                ddx, ddy = pos[0] - f.sx, pos[1] - f.sy
                dl = math.hypot(ddx, ddy) or 1.0
                rin = max(0.0, f.sr - 2.0)
                target = (f.sx + ddx / dl * rin, f.sy + ddy / dl * rin)
            self.path_request("_zone", pos, target, {}, self._t)
            job = self._paths.get("_zone")
            if job is not None and job.path is not None:
                st = self._pursue(job, pos)
                if st is not None and st.direction is not None:
                    return st.direction
            d, _ = self._probe_dir(pos, (target[0] - pos[0], target[1] - pos[1]))
            return d
        except Exception:
            self.exceptions += 1
            return None

    def _descend(self, f, c):
        """Neighbour of c minimising edge cost + field (the Dijkstra successor toward S)."""
        H, W, J, OFF, N = self._H, self._W, self._J, self._OFF, self.N
        d = f.d
        inv = f.inv
        K = self._SLOPE_COST_K * 0.01
        w = W[c]
        best, bv = -1, d[c] + 1e-9
        for k in BITS8[w | J[c]]:
            v = c + OFF[k]
            if v < 0 or v >= N or not f.closed[v]:
                continue
            if self._blocked and (c * 8 + k) in self._blocked:
                continue
            dh = H[v] - H[c]
            cost = ((DLEN[k] + K * dh) if dh > 0 else DLEN[k]) * inv
            if not (w >> k) & 1:
                cost += self._JUMP_EDGE_COST_S
            val = cost + d[v]
            if val < bv:
                bv, best = val, v
        if f.climbs:
            for rec in self._climb_from.get(c, ()):
                v = rec["r"]
                if f.closed[v]:
                    val = d[v] + rec["d"] * inv + rec["h"] / max(0.5, self.rise) + self._CLIMB_EDGE_PAD_S
                    if val < bv:
                        bv, best = val, v
        return best

    @staticmethod
    def _dir_to(tgt, pos):
        dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
        ln = math.hypot(dx, dy)
        if ln < 1e-6:
            return None
        return (dx / ln, dy / ln)

    def tau_edge(self, pos, t: float, zone: Optional[ZoneState] = None) -> float:
        z = zone or self._zone
        if z is None:
            return INF
        return tau_edge(z, pos, t, int(self._IMPL_BISECT_STEPS))

    def slack(self, pos, t: float, zone: Optional[ZoneState] = None) -> Tuple[float, float, float, bool]:
        """8.8: (slack, tau_edge, eta, settled); slack = tau_edge(pos) - t - ETA."""
        te = self.tau_edge(pos, t, zone)
        eta, settled = self.zone_eta(pos)
        return te - t - eta, te, eta, settled

    # ================================================================== loot flood (8.3)
    def flood_request(self, origin: Vec3, radius: float, cost: dict) -> None:
        """Flood once when targets appear; reused for NAV_FLOOD_REFRESH_S unless the origin moved
        > 2 cells, the radius changed or cost["force"] is set."""
        try:
            c = self.walkable_cell(origin[0], origin[1], origin[2] if len(origin) > 2 else None)
            if c < 0:
                return
            radius = float(radius or self._NAV_FLOOD_R)
            force = bool(cost and cost.get("force"))
            for s in (self._flood, self._flood_done):
                if s is None or force:
                    continue
                moved = max(abs(s.oi - c % self.NX), abs(s.oj - c // self.NX))
                if s.radius == radius and moved <= 2 and (not s.done or self._t - s.created_t < self._NAV_FLOOD_REFRESH_S):
                    return
            self._flood = self._new_search("flood", c, cost or {}, radius=radius)
        except Exception:
            self.exceptions += 1

    def travel_time(self, x: float, y: float) -> Optional[float]:
        """Seconds from the flood origin (settled cells of the running flood, else the latest complete
        flood); None = unknown / unreachable / outside the radius."""
        c = self.walkable_cell(x, y, None, rings=1)
        if c < 0:
            return None
        s = self._flood
        if s is not None and s.closed[c]:
            return s.dist[c]
        s = self._flood_done
        if s is not None and s.dist[c] < INF:
            return s.dist[c]
        return None

    def flood_done(self) -> bool:
        return self._flood_done is not None and self._flood is None

    def loot_score(self, value: float, risk: float, handle_s: float, x: float, y: float,
                   extra_s: float = 0.0) -> Optional[float]:
        """8.3: V * Risk / (T_travel + T_handle + LOOT_TIME_PAD_S); None while travel is unknown."""
        tt = self.travel_time(x, y)
        if tt is None:
            return None
        return value * risk / (tt + extra_s + handle_s + self._LOOT_TIME_PAD_S)

    # ================================================================== A* (8.9)
    def path_request(self, key: str, start: Vec3, goal: Vec2, cost: dict, t: float) -> None:
        """Keyed resumable A*. Replans only when the goal moves to another cell, the job went stale
        (blocked edge, deviation > PATH_DEVIATION_M, partial path end reached) or failed."""
        try:
            self._t = max(self._t, float(t))
            job = self._paths.get(key)
            gc = self.walkable_cell(goal[0], goal[1], None, rings=2)
            if job is not None:
                job.req_t = self._t
                same = (gc == job.goal_cell) or math.hypot(goal[0] - job.goal[0], goal[1] - job.goal[1]) < self._WAYPOINT_REACH_M
                if same and not job.stale:
                    job.goal = (goal[0], goal[1]) if same and gc == job.goal_cell else job.goal
                    return
            else:
                job = self._paths[key] = _PathJob(key)
                job.req_t = self._t
            self._start_job(job, start, goal, gc, cost)
        except Exception:
            self.exceptions += 1

    def _start_job(self, job, start, goal, gc, cost):
        job.goal = (goal[0], goal[1])
        job.goal_cell = gc
        job.start = tuple(start)
        job.cost = dict(cost or {})
        job.stale = False
        job.fail = False
        job.created_t = self._t
        sc = self.walkable_cell(start[0], start[1], start[2] if len(start) > 2 else None)
        if sc < 0 or gc < 0:
            job.search = None
            job.fail = True
            return
        s = self._new_search("astar", sc, job.cost, goal_cell=gc, goal=job.goal)
        s.max_exp = int(self._ASTAR_MAX_EXPANSIONS)
        s.start = job.start
        s.key = job.key
        job.search = s
        if sc == gc:
            s.found = s.done = True
            s.best = sc
            self._finish_job(job)

    def _restart_job(self, job, pos):
        self._start_job(job, pos, job.goal, job.goal_cell, job.cost)

    def path(self, key: str) -> Optional[Path]:
        job = self._paths.get(key)
        return job.path if job is not None else None

    def path_pending(self, key: str) -> bool:
        job = self._paths.get(key)
        return job is not None and job.search is not None and not job.search.done

    def _finish_job(self, job, sl=None):
        s = job.search
        end = s.best if s.best >= 0 else s.origin_cell
        cells = []
        c = end
        guard = 0
        while c != -1 and guard < self.N:
            cells.append(c)
            if c == s.origin_cell:
                break
            c = s.prev[c]
            guard += 1
        cells.reverse()
        job.cells = cells
        complete = bool(s.found)
        pts: List[Vec2] = [(job.start[0], job.start[1])]
        kinds: List[int] = [0]          # 0 normal, 1 jump start, 2 climb stand, 3 climb landing
        faces: Dict[int, dict] = {}
        for i in range(1, len(cells)):
            v = cells[i]
            et = s.etype[v]
            if et == 2:
                rec = s.face.get(v)
                if rec is not None:
                    faces[len(pts)] = rec
                    pts.append((rec["x"], rec["y"]))
                    kinds.append(2)
                    pts.append((rec["rx"], rec["ry"]))
                    kinds.append(3)
                    continue
            if et == 1:
                kinds[-1] = 1 if kinds[-1] == 0 else kinds[-1]
            pts.append(self.cell_xy(v))
            kinds.append(0)
        if complete:
            g = job.goal
            if len(pts) >= 2 and math.hypot(pts[-1][0] - g[0], pts[-1][1] - g[1]) > 0.5 and self.line_ok(pts[-1], g):
                if kinds[-1] in (0,) and len(pts) >= 2 and kinds[-2] in (0,) and self.line_ok(pts[-2], g):
                    pts[-1] = g
                else:
                    pts.append(g)
                    kinds.append(0)
            elif len(pts) == 1:
                pts.append(g)
                kinds.append(0)
        dl = None
        if sl is not None:
            dl = max(sl.end, _now() + 0.0003)
        spts, skinds, sfaces, stop = self._smooth(pts, kinds, faces, 0, dl)
        eta = s.dist[end] if s.dist[end] < INF else 0.0
        job.path = Path(key=job.key, goal=job.goal, points=spts, complete=complete, eta_s=eta,
                        climbs=[i for i, k in enumerate(skinds) if k in (1, 2)], created_t=self._t)
        job.climb_at = sfaces
        job.jump_at = {i for i, k in enumerate(skinds) if k == 1}
        job.kinds = skinds
        job.raw = (pts, kinds, faces)
        job.raw_stop = stop
        job.out_stop = (len(spts) - (len(pts) - 1 - stop) - 1) if stop < len(pts) - 1 else len(spts) - 1
        job.idx = 0
        job.search = None

    def _refine(self, job, sl):
        """Continue smoothing the raw tail of a path whose smoothing ran out of slice time."""
        pts, kinds, faces = job.raw
        if job.raw_stop >= len(pts) - 1:
            return
        k0 = job.out_stop
        if job.idx >= k0 - 1:                      # already following the raw tail: leave it
            job.raw_stop = len(pts) - 1
            return
        out, okinds, ofaces, stop = self._smooth(pts, kinds, faces, job.raw_stop, max(sl.end, _now() + 0.0003))
        path = job.path
        path.points[k0:] = out
        job.kinds[k0:] = okinds
        job.climb_at = {i: v for i, v in job.climb_at.items() if i < k0}
        for i, v in ofaces.items():
            job.climb_at[k0 + i] = v
        job.jump_at = {i for i, k in enumerate(job.kinds) if k == 1}
        path.climbs = [i for i, k in enumerate(job.kinds) if k in (1, 2)]
        job.out_stop = (k0 + len(out) - (len(pts) - 1 - stop) - 1) if stop < len(pts) - 1 else len(path.points) - 1
        job.raw_stop = stop

    def _smooth(self, pts, kinds, faces, i0=0, deadline=None):
        """String-pulling from raw index i0 between anchors (jump starts, climb stand / landing points)
        with line_ok. Past `deadline` (perf_counter s) the remaining raw points are appended as they
        are. Returns (points, kinds, {out index: climb face}, raw index where smoothing stopped)."""
        n = len(pts)
        out = [pts[i0]]
        okinds = [kinds[i0]]
        ofaces = {}
        if kinds[i0] == 2:
            ofaces[0] = faces.get(i0)
        i = i0
        while i < n - 1:
            if deadline is not None and _now() > deadline:
                for k in range(i + 1, n):
                    if kinds[k] == 2:
                        ofaces[len(out)] = faces.get(k)
                    out.append(pts[k])
                    okinds.append(kinds[k])
                return out, okinds, ofaces, i
            if kinds[i] == 2:                     # climb edge itself is not smoothed
                out.append(pts[i + 1])
                okinds.append(kinds[i + 1])
                i += 1
                continue
            best = i + 1
            fails = 0
            j = i + 2
            while j < n and j <= i + 24:
                if kinds[j - 1] in (1, 2):         # never skip past a jump start / climb stand point
                    break
                if self.line_ok(pts[i], pts[j]):
                    best = j
                    fails = 0
                else:
                    fails += 1
                    if fails >= 3:
                        break
                j += 1
            if kinds[best] == 2:
                ofaces[len(out)] = faces.get(best)
            out.append(pts[best])
            okinds.append(kinds[best])
            i = best
        return out, okinds, ofaces, n - 1

    # ------------------------------------------------------------------ follower
    def follow(self, me, key: Optional[str], goal: Optional[Vec2]) -> Steer:
        """Pure pursuit on the keyed path (8.9) with jump probe and climb sequence (R13.2); interim
        steering (straight line + clearance probe, wall slide) while no path is available."""
        try:
            self._sync_me(me)
            pos = tuple(me.pos)
            self._last_follow_key = key
            if self._climb is not None:
                st = self._climb_step(me)
                if st is not None:
                    return st
            job = self._paths.get(key) if key else None
            if goal is None and job is not None:
                goal = job.goal
            if goal is not None and math.hypot(goal[0] - pos[0], goal[1] - pos[1]) < self._WAYPOINT_REACH_M:
                return Steer(direction=None, arrived=True, rule="8.9")
            if job is not None and job.path is not None:
                st = self._pursue(job, pos, me)
                if st is not None:
                    return st
            if goal is None:
                return Steer(direction=None, arrived=True, rule="8.9")
            d, blocked = self._probe_dir(pos, (goal[0] - pos[0], goal[1] - pos[1]))
            if d is None:
                return Steer(direction=None, arrived=True, rule="8.9")
            jump = bool(me.grounded) and self._STEP_UP < self._ledge_ahead(pos, d) <= self._JUMP_H
            return Steer(direction=d, jump=jump, blocked=blocked, rule="8.9i")
        except Exception:
            self.exceptions += 1
            if goal is not None and me is not None:
                d = self._dir_to(goal, me.pos)
                return Steer(direction=d, rule="8.9f")
            return Steer(direction=None, blocked=True, rule="8.9f")

    def _pursue(self, job, pos, me=None) -> Optional[Steer]:
        path = job.path
        pts = path.points
        n = len(pts)
        if n < 2:
            return None
        px, py = pos[0], pos[1]
        idx = min(job.idx, n - 2)
        best_d, best_i, best_t = INF, idx, 0.0
        stop = n - 1
        for i in range(idx, n - 1):
            if i > idx and (i in job.climb_at):
                stop = i
                break
        for i in range(idx, min(stop, idx + 10, n - 1)):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            tt = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
            qx, qy = ax + vx * tt, ay + vy * tt
            dd = math.hypot(px - qx, py - qy)
            if dd < best_d - 1e-6 or (dd < best_d + 0.5 and i > best_i and tt > 0.0):
                best_d, best_i, best_t = dd, i, tt
        if best_i >= idx:
            idx = best_i
        job.idx = idx
        last = pts[-1]
        dl = math.hypot(last[0] - px, last[1] - py)
        if dl < self._WAYPOINT_REACH_M:
            if path.complete:
                return Steer(direction=None, arrived=True, rule="8.9")
            self._restart_job(job, pos)               # partial path end reached: replan from here
            return None
        if best_d > self._PATH_DEVIATION_M and self._climb is None:
            self._restart_job(job, pos)
            return None
        # climb ahead?
        nxt = idx + 1
        slow = me is not None and math.hypot(me.vel[0], me.vel[1]) < 2.0
        s_idx = next((i for i in range(nxt, min(n, idx + 4)) if i in job.climb_at), None)
        if s_idx is not None:
            sp = pts[s_idx]
            dsp = math.hypot(sp[0] - px, sp[1] - py)
            if me is not None and (dsp < CLIMB_START_R or (slow and dsp < CLIMB_START_R + 2.0)):
                rec = job.climb_at[s_idx]
                if rec is not None:
                    self._climb_begin(rec, job, s_idx + 1)
                    st = self._climb_step(me)
                    if st is not None:
                        return st
        if idx in job.climb_at and best_t < 0.5 and me is not None:
            sp = pts[idx]
            if math.hypot(sp[0] - px, sp[1] - py) < CLIMB_START_R:
                rec = job.climb_at[idx]
                if rec is not None:
                    self._climb_begin(rec, job, idx + 1)
                    st = self._climb_step(me)
                    if st is not None:
                        return st
        # lookahead point along the polyline (stops at a climb stand point)
        ax, ay = pts[idx]
        bx, by = pts[idx + 1]
        cx, cy = ax + (bx - ax) * best_t, ay + (by - ay) * best_t
        rem = self._LOOKAHEAD_M
        i = idx
        tgt = None
        while i < n - 1:
            nx_, ny_ = pts[i + 1]
            seg = math.hypot(nx_ - cx, ny_ - cy)
            if (i + 1) in job.climb_at:
                tgt = (nx_, ny_)
                break
            if seg >= rem:
                f = rem / seg if seg > 1e-9 else 0.0
                tgt = (cx + (nx_ - cx) * f, cy + (ny_ - cy) * f)
                break
            rem -= seg
            cx, cy = nx_, ny_
            i += 1
        if tgt is None:
            tgt = pts[-1]
        if math.hypot(tgt[0] - px, tgt[1] - py) > 1.0 and not self.line_ok((px, py), tgt):
            tgt = pts[idx + 1]
            if math.hypot(tgt[0] - px, tgt[1] - py) < 1.0 and idx + 2 < n:
                tgt = pts[idx + 2]
        d = self._dir_to(tgt, pos)
        if d is None:
            d = self._dir_to(pts[min(n - 1, idx + 1)], pos)
            if d is None:
                return Steer(direction=None, arrived=True, rule="8.9")
        return self._hop_steer(d, pos, me, "8.9")

    def _hop_steer(self, d, pos, me, rule):
        """R13.2 jump edge: jump within JUMP_TRIGGER_DIST_M of a ledge in (STEP_UP, JUMP_H]; for a
        ledge within 0.6 m of JUMP_H a short jet assist (climb_boost) keeps the feet above it."""
        grounded = True if me is None else bool(me.grounded)
        jump = boost = False
        if grounded:
            r = self._ledge_ahead(pos, d)
            jump = self._STEP_UP < r <= self._JUMP_H + 0.25
            if jump:
                self._hop = (pos[2] + r, self._t + 1.2, r > self._JUMP_H - 0.6)
        if self._hop is not None and me is not None:
            top, until, high = self._hop
            if self._t > until or (grounded and not jump):
                self._hop = None
            elif high and pos[2] < top + 0.3 and me.energy > self._CLIMB_ABORT_ENERGY + 5:
                boost = True
        return Steer(direction=d, jump=jump, climb_boost=boost, rule="R13.2" if (jump or boost) else rule)

    # ------------------------------------------------------------------ climbs (R13.2, R10.2, RZ4)
    def climb_faces(self, roof_id: int) -> List[dict]:
        """Non-eave climb faces of a roof: dicts {roof, g, r, h, x, y (stand), rx, ry (landing), ...}."""
        return list(self._faces_by_roof.get(int(roof_id), ()))

    def climb_feasible(self, h: float, energy: float, reserve: float) -> bool:
        return h <= self._CLIMB_MAX_H and energy >= self.climb_cost(h) + reserve

    def best_face(self, roof_id: int, near: Vec2, energy: Optional[float] = None,
                  reserve: Optional[float] = None) -> Optional[dict]:
        """Climb face of the roof nearest `near` that our energy allows (CLIMB_COST(h) + reserve)."""
        e = self.energy if energy is None else energy
        r = self._CLIMB_NAV_RESERVE if reserve is None else reserve
        best, bd = None, INF
        for f in self._faces_by_roof.get(int(roof_id), ()):
            if not self.climb_feasible(f["h"], e, r):
                continue
            d = math.hypot(f["x"] - near[0], f["y"] - near[1])
            if d < bd:
                bd, best = d, f
        return best

    def climb_start(self, face: dict) -> None:
        """Begin the climb sequence on a face directly (tactics: roof assault / roof plan). follow()
        then drives it: approach the stand point, jump, jet until feet >= top + CLIMB_CLEAR_M, step on."""
        self._climb_begin(face, None, None)

    def climbing(self) -> bool:
        return self._climb is not None

    def climb_steer(self, me) -> Optional[Steer]:
        """Steer for a climb started with climb_start (None when no climb is in progress)."""
        self._sync_me(me)
        if self._climb is None:
            return None
        return self._climb_step(me)

    def _climb_begin(self, rec, job, after_idx):
        top = self._h1(rec["rx"], rec["ry"])
        if top is None:
            top = self.md.surface_h(rec["rx"], rec["ry"])
        self._climb = {"face": rec, "job": job, "after": after_idx, "phase": "approach",
                       "top": top, "t0": self._t, "tries": 0, "jet_t": None}

    def _climb_step(self, me) -> Optional[Steer]:
        cl = self._climb
        rec = cl["face"]
        pos = me.pos
        top = cl["top"]
        clear = self._CLIMB_CLEAR_M
        land = (rec["rx"], rec["ry"])
        stand = (rec["x"], rec["y"])
        need = self.climb_cost(max(0.0, top - pos[2]))
        if self._t - cl["t0"] > CLIMB_TIMEOUT_S:
            return self._climb_abort("timeout")
        ph = cl["phase"]
        if ph == "approach":
            if math.hypot(stand[0] - pos[0], stand[1] - pos[1]) > CLIMB_START_R + 2.5:
                d = self._dir_to(stand, pos)
                return Steer(direction=d, rule="R13.2a")
            if me.grounded and me.energy < need:
                cl["t0"] = self._t      # waiting for energy is not a timeout
                return Steer(direction=None, rule="R13.2w")
            # align: kill the speed along the wall before jumping (air_accel is low)
            into = self._dir_to(land, pos) or (1.0, 0.0)
            vx, vy = me.vel[0], me.vel[1]
            along = abs(vx * into[1] - vy * into[0])
            if me.grounded and along > 2.0 and cl.get("align_n", 0) < 6:
                cl["align_n"] = cl.get("align_n", 0) + 1
                return Steer(direction=into, rule="R13.2al")
            cl["phase"] = "jet"
            cl["jet_t"] = self._t
            return Steer(direction=self._dir_to(land, pos), jump=True, climb_boost=True, rule="R13.2j")
        if ph == "jet":
            if me.grounded and pos[2] >= top - 0.6:
                return self._climb_done()
            if me.grounded and self._t - cl["jet_t"] > 0.35:
                # fell back down (or never left): retry once, then give up
                cl["tries"] += 1
                if cl["tries"] >= 2 or me.energy < need:
                    return self._climb_abort("retry")
                cl["jet_t"] = self._t
                return Steer(direction=self._dir_to(land, pos), jump=True, climb_boost=True, rule="R13.2j")
            if me.energy < self._CLIMB_ABORT_ENERGY:
                return self._climb_abort("energy")
            d = self._dir_to(land, pos)
            boost = pos[2] < top + clear + 0.6
            return Steer(direction=d, climb_boost=boost, rule="R13.2c")
        return self._climb_abort("phase")

    def _climb_done(self):
        cl = self._climb
        self._climb = None
        job = cl.get("job")
        if job is not None and job.path is not None and cl.get("after") is not None:
            job.idx = min(cl["after"], len(job.path.points) - 2)
        return None

    def _climb_abort(self, why):
        cl = self._climb
        self._climb = None
        job = cl.get("job") if cl else None
        if job is not None:
            rec = cl["face"]
            # this face failed: block its edge for a while and replan
            self._blocked_climb(rec)
            job.stale = True
            self._restart_job(job, self._pos)
        return Steer(direction=None, rule="R13.2x:" + why)

    def _blocked_climb(self, rec):
        lst = self._climb_from.get(rec["g"])
        if lst and rec in lst:
            lst.remove(rec)
        lst = self._climb_to.get(rec["r"])
        if lst and rec in lst:
            lst.remove(rec)

    # ================================================================== stuck ladder (O5, R13.1)
    def stuck_update(self, world, commanded: Optional[Vec2]):
        """O5 trigger and the R13.1 recovery ladder. Returns (LegRequest | None, JumpRequest | None,
        BoostRequest | None); the LegRequest goes to intent.unstick."""
        try:
            self._sync_world(world)
            me = world.me
            t = self._t
            st = self._stuck
            vx, vy = me.vel[0], me.vel[1]
            spd = math.hypot(vx, vy)
            cmag = math.hypot(commanded[0], commanded[1]) if commanded else 0.0
            if self._climb is not None and t - self._climb["t0"] > CLIMB_TIMEOUT_S:
                # climb_step (which owns the timeout) only runs inside follow(); when another owner (O1, X2)
                # took the legs mid-climb the climb never ended and O5 stayed disabled for the match
                self._climb_abort("timeout")
            if not st["active"]:
                charging = (getattr(me, "laser_charge", 0.0) or 0.0) > 0.0
                # low horizontal speed accumulates in the air too (a jump loop in place is stuck);
                # the trigger itself needs ground contact unless it lasted 2 * STUCK_T
                trig = cmag >= self._STUCK_MOVE_MIN and spd < self._STUCK_SPEED and not charging
                if trig and self._climb is None:
                    if st["slow_since"] is None:
                        st["slow_since"] = t
                    el = t - st["slow_since"]
                    if el >= self._STUCK_T and (me.grounded or el >= 2 * self._STUCK_T):
                        self._stuck_begin(me, commanded, t)
                    else:
                        return None, None, None
                else:
                    st["slow_since"] = None
                    return None, None, None
            # active episode: exits
            if spd >= self._UNSTUCK_OK_SPEED:
                if st["ok_since"] is None:
                    st["ok_since"] = t
                if t - st["ok_since"] >= self._UNSTUCK_OK_S:
                    self._stuck_end("ok")
                    return None, None, None
            else:
                st["ok_since"] = None
            if t - st["start"] >= self._NAV_RECOVER_MAX_S:
                if not st["blocked"] and st["escape"] is None:
                    self._stuck_block(me, t)
                self._stuck_end("timeout")
                return None, None, None
            return self._ladder(me, t)
        except Exception:
            self.exceptions += 1
            return None, None, None

    def _stuck_begin(self, me, commanded, t):
        st = self._stuck
        ln = math.hypot(commanded[0], commanded[1]) or 1.0
        st.update(active=True, start=t, ok_since=None, slow_since=None, blocked=False, escape=None,
                  dir=(commanded[0] / ln, commanded[1] / ln), pos0=tuple(me.pos), step_name=None)
        eps = [e for e in st["episodes"] if t - e <= self._STUCK_REPEAT_WINDOW_S]
        eps.append(t)
        st["episodes"] = eps
        if len(eps) >= self._STUCK_REPEAT_N:
            st["episodes"] = []
            st["escape"] = self._escape_target(me)
            st["steps"] = [("escape", self._NAV_RECOVER_MAX_S)]
            return
        steps = [("jump", self._UNSTUCK_STEP_S)]
        top = self._obstacle_top(me, st["dir"])
        if top is not None and me.energy >= self._UNSTUCK_JET_MIN_ENERGY:
            st["jet_top"] = top
            steps.append(("jet", self._UNSTUCK_JET_MAX_S))
        a = st["dir"]
        left = (-a[1], a[0])
        first = 1 if self.clear_dir(me.pos, left, self._LEG_PROBE_M) else -1
        st["side_first"] = first
        steps.append(("side1", self._UNSTUCK_STEP_S))
        steps.append(("side2", self._UNSTUCK_STEP_S))
        steps.append(("block", 0.0))
        st["steps"] = steps

    def _obstacle_top(self, me, d):
        """Top (m) of a ledge >= STEP_UP within 3 m ahead, or the next path point if >= STEP_UP higher."""
        pos = me.pos
        z = pos[2]
        top = None
        s = 0.5
        while s <= self.radius + 3.0:
            h = self._h1(pos[0] + d[0] * s, pos[1] + d[1] * s)
            if h is not None and h - z >= self._STEP_UP:
                top = h if top is None else max(top, h)
            s += 0.5
        job = self._paths.get(self._last_follow_key) if self._last_follow_key else None
        if job is not None and job.path is not None:
            pts = job.path.points
            i = min(job.idx + 1, len(pts) - 1)
            h = self._h1(pts[i][0], pts[i][1])
            if h is not None and h - z >= self._STEP_UP:
                top = h if top is None else max(top, h)
        return top

    def _escape_target(self, me):
        pos = me.pos
        rng = self.rng
        for _ in range(24):
            a = rng.uniform(-math.pi, math.pi)
            r = rng.uniform(self._ESCAPE_MIN_M, self._ESCAPE_MAX_M)
            x, y = pos[0] + r * math.cos(a), pos[1] + r * math.sin(a)
            c = self._cell_raw(x, y)
            if c >= 0 and self._F[c] & F_WALK and abs(self._H[c] * 0.01 - pos[2]) < 3.0:
                return self.cell_xy(c)
        return (pos[0] - self._stuck["dir"][0] * self._ESCAPE_MIN_M,
                pos[1] - self._stuck["dir"][1] * self._ESCAPE_MIN_M)

    def _ladder(self, me, t):
        st = self._stuck
        el = t - st["start"]
        acc = 0.0
        name = None
        for nm, dur in st["steps"]:
            if el < acc + dur or nm in ("block", "escape"):
                name = nm
                break
            acc += dur
        st["step_name"] = name
        d = st["dir"]
        if name == "jump":
            return (LegRequest(level="O5", rule="R13.1.1", direction=d),
                    JumpRequest(owner="O5", rule="R13.1.1") if me.grounded else None, None)
        if name == "jet":
            want_boost = (not me.grounded or el - acc < 0.15) and me.pos[2] < st["jet_top"] + self._CLIMB_CLEAR_M + 0.3
            return (LegRequest(level="O5", rule="R13.1.2", direction=d),
                    JumpRequest(owner="O5", rule="R13.1.2") if me.grounded else None,
                    BoostRequest(owner="CLIMB", rule="R13.1.2", want=want_boost))
        if name in ("side1", "side2"):
            sgn = st["side_first"] if name == "side1" else -st["side_first"]
            sd = (-d[1] * sgn, d[0] * sgn)
            return LegRequest(level="O5", rule="R13.1.3", direction=sd), None, None
        if name == "block":
            if not st["blocked"]:
                self._stuck_block(me, t)
            return None, None, None
        if name == "escape":
            tgt = st["escape"]
            if tgt is None or math.hypot(tgt[0] - me.pos[0], tgt[1] - me.pos[1]) < 3.0:
                self._stuck_end("escaped")
                return None, None, None
            dd, _ = self._probe_dir(me.pos, (tgt[0] - me.pos[0], tgt[1] - me.pos[1]))
            jump = bool(me.grounded) and self._STEP_UP < self._ledge_ahead(me.pos, dd or d) <= self._JUMP_H
            return (LegRequest(level="O5", rule="R13.1.5", direction=dd, goal=tgt),
                    JumpRequest(owner="O5", rule="R13.1.5") if jump else None, None)
        return None, None, None

    def _stuck_block(self, me, t):
        st = self._stuck
        d = st["dir"]
        pos = me.pos
        self.block_edge(pos, (pos[0] + d[0] * NAV_STEP, pos[1] + d[1] * NAV_STEP, pos[2]),
                        self._NAV_BLOCK_TTL_S, t)
        st["blocked"] = True

    def _stuck_end(self, why):
        st = self._stuck
        if st["escape"] is not None or why == "escaped":
            for job in self._paths.values():
                job.stale = True
                self._restart_job(job, self._pos)
        st.update(active=False, ok_since=None, slow_since=None, escape=None, step_name=None,
                  last_end=why)

    def stuck_active(self) -> bool:
        return bool(self._stuck["active"])

    def block_edge(self, a: Vec3, b: Vec3, ttl_s: float, t: float) -> None:
        """Block the nav4 edge from a's cell toward b (first step if not adjacent) until t + ttl_s;
        paths using it go stale and replan."""
        try:
            ca = self.walkable_cell(a[0], a[1], a[2] if len(a) > 2 else None, rings=1)
            if ca < 0:
                return
            ax, ay = self.cell_xy(ca)
            dx, dy = b[0] - ax, b[1] - ay
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return
            ang = math.atan2(dy, dx)
            k = int(round(ang / (math.pi / 4))) % 8
            until = t + ttl_s
            self._blocked[ca * 8 + k] = until
            v = ca + self._OFF[k]
            if 0 <= v < self.N:
                kk = (k + 4) & 7
                self._blocked.setdefault(v * 8 + kk, until)
            for job in self._paths.values():
                cells = job.cells
                for i in range(len(cells) - 1):
                    if cells[i] == ca and cells[i + 1] == v:
                        job.stale = True
                        self._restart_job(job, self._pos)
                        break
        except Exception:
            self.exceptions += 1

    def _prune_blocked(self):
        if self._blocked:
            t = self._t
            for k in [k for k, u in self._blocked.items() if u <= t]:
                del self._blocked[k]

    # ================================================================== cover and spots (8.4)
    def cover_query(self, me, threats: List[Vec3], budget_s: float, sl: Slice) -> Optional[Vec2]:
        """Urgent cover (O3/O4, R3.1): one local Dijkstra within COVER_SEARCH_R gives all ETAs; up
        to COVER_URGENT_CANDIDATES walkable cover2 cells nearest first, one LOS check per threat.
        Hard: ETA < budget_s and LOS from every threat eye (feet + 8 m) to our centre at the cell
        blocked. Returns the best point found before the slice expires (None if none)."""
        try:
            self._sync_me(me)
            pos = me.pos
            md = self.md
            R = float(self._COVER_SEARCH_R)
            c0 = self.walkable_cell(pos[0], pos[1], pos[2])
            if c0 < 0:
                return None
            s = self._new_search("cover", c0, {}, radius=R + NAV_STEP * 1.5)
            self._run(s, sl)
            eyes = [(p[0], p[1], p[2] + EYE_H) for p in threats]
            cands = md.cover2_cells(pos[0], pos[1], R)
            zone = self._zone
            world = self._world
            heat_ok = world is not None and hasattr(world, "heat_at")
            inv = s.inv
            best, bscore, btie = None, -INF, INF
            nchk = 0
            cz = CENTER_H * self.s
            for cx, cy, bits in cands:
                if not bits & COVER_WALK:
                    continue
                c = self._cell_raw(cx, cy)
                if c < 0:
                    continue
                g = s.dist[c]
                if g == INF:
                    continue
                ccx, ccy = self.cell_xy(c)
                eta = g + math.hypot(cx - ccx, cy - ccy) * inv
                if eta >= budget_s:
                    continue
                nchk += 1
                if nchk > self._COVER_URGENT_CANDIDATES:
                    break
                z = self._h1(cx, cy)
                if z is None:
                    continue
                tgt = (cx, cy, z + cz)
                ok = True
                for e in eyes:
                    if md.los(e, tgt):
                        ok = False
                        break
                if ok:
                    sc = -eta / self._ETA_NORM_S
                    tie = 0.0
                    if zone is not None:
                        dz = math.hypot(cx - zone.center[0], cy - zone.center[1])
                        if dz > zone.radius:
                            sc -= self._COVER_W_OUTSIDE
                        tie = dz
                    if heat_ok:
                        try:
                            sc -= self._COVER_W_HEAT * float(world.heat_at(cx, cy))
                        except Exception:
                            pass
                    if sc > bscore + 1e-9 or (abs(sc - bscore) <= 1e-9 and tie < btie):
                        best, bscore, btie = (cx, cy), sc, tie
                if sl is not None and sl.expired():
                    break
            return best
        except Exception:
            self.exceptions += 1
            return None

    def best_spot(self, kind: str, world, A) -> Optional[Tuple[Vec2, float, int]]:
        """(point, score, roof_id or 0) from the latest finished ranking of `kind` (hold, cover, roof,
        nook); requests a (re)ranking that runs in anytime() on RANK_MS_CAP. None until one finished."""
        self._sync_world(world)
        self._spot_wanted[kind] = self._t
        r = self._spots.get(kind)
        if r is None:
            return None
        return r[0]

    def spot_valid(self, kind: str, point: Vec2, world, A) -> bool:
        """8.4 invalidation: outside next circle, LOS open from a known hostile within its reach, a mech
        within SPOT_CROWD_R, (roof) a hostile on the same roof. HOLD_MAX_S / blacklist: caller."""
        try:
            self._sync_world(world)
            z = getattr(world, "zone", None)
            if z is not None and math.hypot(point[0] - z.next_center[0], point[1] - z.next_center[1]) > z.next_radius:
                return False
            md = self.md
            zp = self._h1(point[0], point[1])
            zp = (zp if zp is not None else 0.0) + CENTER_H * self.s
            rid = md.roof_id(point[0], point[1]) if kind == "roof" else 0
            hostiles = set(getattr(A, "hostiles", None) or ())
            me_id = getattr(world.me, "id", None)
            for tr in (getattr(world, "tracks", None) or {}).values():
                if tr.kind != "mech" or tr.id == me_id:
                    continue
                if self._t - getattr(tr, "last_seen_t", self._t) > 10.0:
                    continue
                p = tr.pos
                d = math.hypot(p[0] - point[0], p[1] - point[1])
                if d < self._SPOT_CROWD_R:
                    return False
                if rid and md.roof_id(p[0], p[1]) == rid:
                    return False
                if tr.id in hostiles or not hostiles:
                    reach = max([self._weapon_range(k) for k in (tr.weapons or ("gun",))] or [0.0])
                    if d <= reach and md.los((p[0], p[1], p[2] + EYE_H), (point[0], point[1], zp)):
                        return False
            return True
        except Exception:
            self.exceptions += 1
            return True

    def _eta_est(self, x, y):
        tt = self.travel_time(x, y)
        if tt is not None:
            return tt
        p = self._pos
        return math.hypot(x - p[0], y - p[1]) / max(0.5, self.walk) * self._ZONE_FALLBACK_DETOUR_K

    def _heat(self, world, x, y):
        if world is None or not hasattr(world, "heat_at"):
            return 0.0
        try:
            return float(world.heat_at(x, y))
        except Exception:
            return 0.0

    def _threat_eyes(self, world, A, include_ghosts=False):
        eyes = []
        me_id = getattr(world.me, "id", None)
        hostiles = set(getattr(A, "hostiles", None) or ())
        for tr in (getattr(world, "tracks", None) or {}).values():
            if tr.id == me_id:
                continue
            if tr.kind != "mech" and tr.id not in hostiles:
                continue
            if self._t - getattr(tr, "last_seen_t", self._t) > 30.0:
                continue
            p = tr.pos
            eyes.append((p[0], p[1], p[2] + EYE_H * (1 + 0.05 * getattr(tr, "cubes", 0))))
        if include_ghosts:
            for gh in (getattr(world, "ghosts", None) or {}).values():
                x, y = gh.pos[0], gh.pos[1]
                z = self._h1(x, y) or 0.0
                eyes.append((x, y, z + EYE_H))
        return eyes

    def _rank_gen(self, kind, world, A):
        """Generator: yields to let anytime() check its slice; returns (point, score, roof_id) or None."""
        md = self.md
        z = getattr(world, "zone", None) or self._zone
        me = world.me
        pos = me.pos
        t = self._t
        if kind == "roof":
            best = None
            bsc = -INF
            s_self = self.s
            if z is None:
                return None
            n = 0
            contacts = []
            me_id = getattr(me, "id", None)
            for tr in (getattr(world, "tracks", None) or {}).values():
                if tr.kind == "mech" and tr.id != me_id and t - getattr(tr, "last_seen_t", t) <= self._ROOF_CONTACT_MEMORY_S:
                    contacts.append(md.roof_id(tr.pos[0], tr.pos[1]))
            for roof in md.roofs[1:]:
                n += 1
                if n % 16 == 0:
                    yield None
                h = roof.h
                if not (self._ROOF_MIN_H <= h <= self._ROOF_MAX_H):
                    continue
                if roof.min_width and roof.min_width < self._ROOF_MIN_W * s_self:
                    continue
                cx, cy = roof.centroid
                dn = math.hypot(cx - z.next_center[0], cy - z.next_center[1])
                if dn > z.next_radius - self._ROOF_INSIDE_M:
                    continue
                if math.hypot(cx - z.center[0], cy - z.center[1]) > z.radius - self._ROOF_INSIDE_M:
                    continue
                if self.climb_cost(h) + self._ROOF_ESCAPE_RESERVE > self.max_energy:
                    continue
                faces = self._faces_by_roof.get(roof.id)
                if not faces:
                    continue
                fbest, feta = None, INF
                for f in faces:
                    e = self._eta_est(f["x"], f["y"]) + f["h"] / max(0.5, self.rise) + self._CLIMB_EDGE_PAD_S
                    if e < feta:
                        feta, fbest = e, f
                sc = (self._ROOF_W_NEIGHBOR * (1.0 if roof.neighbors else 0.0)
                      + self._ROOF_W_CENTER * (1.0 - dn / max(1.0, z.next_radius))
                      + roof.sight / self._ROOF_SIGHT_NORM - feta / self._ETA_NORM_S
                      - self._ROOF_W_CONTACT * (1.0 if roof.id in contacts else 0.0))
                if sc > bsc:
                    pt = (cx, cy) if md.roof_id(cx, cy) == roof.id else (fbest["rx"], fbest["ry"])
                    bsc, best = sc, (pt, sc, roof.id)
            return best
        if z is None:
            return None
        if kind == "hold":
            m = self._ZONE_EDGE_MARGIN_M
            # region: next circle and current circle shrunk by the margin
            ncx, ncy = z.next_center
            nr = z.next_radius - m
            ccx, ccy = z.center
            cr = z.radius - m
            if nr <= 0 or cr <= 0:
                return None
            area = math.pi * nr * nr
            stride = max(3, int(math.ceil(math.sqrt(area / (4.0 * self._SPOT_CANDIDATES)) / 2.0)))
            gx0, gy0, nxc, nyc = md.gx0, md.gy0, md.C2_NX, md.C2_NY
            I0 = max(0, int((ncx - nr - gx0) // 2))
            I1 = min(nxc - 1, int((ncx + nr - gx0) // 2))
            J0 = max(0, int((ncy - nr - gy0) // 2))
            J1 = min(nyc - 1, int((ncy + nr - gy0) // 2))
            pb = getattr(A, "playbook", None) or getattr(me, "playbook", "PB_G")
            best, bsc = None, -INF
            n = 0
            for J in range(J0, J1 + 1, stride):
                y = gy0 + (J + 0.5) * 2
                for I in range(I0, I1 + 1, stride):
                    x = gx0 + (I + 0.5) * 2
                    n += 1
                    if n % 32 == 0:
                        yield None
                    sc = self._hold_score(x, y, z, nr, cr, pb, world)
                    if sc is not None and sc > bsc:
                        bsc, best = sc, (x, y)
            if best is None:
                return None
            # refine around the best lattice point
            bx, by = best
            for cx, cy, bits in md.cover2_cells(bx, by, stride * 2.0):
                sc = self._hold_score(cx, cy, z, nr, cr, pb, world)
                if sc is not None and sc > bsc:
                    bsc, best = sc, (cx, cy)
            return (best, bsc, 0)
        if kind in ("cover", "nook"):
            eyes = self._threat_eyes(world, A, include_ghosts=(kind == "nook"))
            fresh = None
            if kind == "cover":
                dl = getattr(world, "damage_log", None)
                if dl:
                    by = dl[-1].by
                    tr = (world.tracks or {}).get(by) if by is not None else None
                    if tr is not None:
                        fresh = (tr.pos[0], tr.pos[1], tr.pos[2] + EYE_H)
            radius = 60.0 if kind == "cover" else 80.0
            cands = md.cover2_cells(pos[0], pos[1], radius)
            yield None
            best, bsc, btie = None, -INF, INF
            n = 0
            cz = CENTER_H * self.s
            for cx, cy, bits in cands:
                if not bits & COVER_WALK:
                    continue
                n += 1
                if n > self._SPOT_CANDIDATES:
                    break
                if n % 8 == 0:
                    yield None
                if kind == "nook":
                    if math.hypot(cx - z.center[0], cy - z.center[1]) > z.radius:
                        continue
                    if math.hypot(cx - z.next_center[0], cy - z.next_center[1]) > z.next_radius:
                        continue
                zc = self._h1(cx, cy)
                if zc is None:
                    continue
                tgt = (cx, cy, zc + cz)
                if any(md.los(e, tgt) for e in eyes):
                    continue
                eta = self._eta_est(cx, cy)
                heat = self._heat(world, cx, cy)
                if kind == "cover":
                    sc = -eta / self._ETA_NORM_S - self._COVER_W_HEAT * heat
                    dz = math.hypot(cx - z.center[0], cy - z.center[1])
                    if dz > z.radius:
                        sc -= self._COVER_W_OUTSIDE
                    if fresh is not None and md.los(fresh, tgt):
                        sc -= self._COVER_W_FRESH
                    tie = dz
                else:
                    S = md.sight5(cx, cy)
                    if S != S:
                        S = self._NOOK_SIGHT_NORM
                    sc = -S / self._NOOK_SIGHT_NORM - eta / self._ETA_NORM_S - self._NOOK_W_HEAT * heat
                    tie = 0.0
                if sc > bsc + 1e-9 or (abs(sc - bsc) <= 1e-9 and tie < btie):
                    best, bsc, btie = (cx, cy), sc, tie
            if best is None:
                return None
            return (best, bsc, 0)
        return None

    def _hold_score(self, x, y, z, nr, cr, pb, world):
        md = self.md
        bits = md.cover2(x, y)
        if not bits & COVER_WALK:
            return None
        if math.hypot(x - z.next_center[0], y - z.next_center[1]) > nr:
            return None
        if math.hypot(x - z.center[0], y - z.center[1]) > cr:
            return None
        wall = bits & COVER_DIST
        if wall > self._HOLD_WALL_R or wall == 0:
            return None
        if pb == "PB_Z":
            best = 0.0
            for k in range(8):
                v = md.sight5_dir(x, y, k)
                if v == v and v > best:
                    best = v
            fs = min(1.0, best / self._SIGHT_NORM_Z)
        elif pb == "PB_L":
            S = md.sight5(x, y)
            fs = 0.5 if S != S else max(0.0, 1.0 - abs(S - self._SIGHT_TARGET_L) / self._SIGHT_TARGET_L)
        else:
            S = md.sight5(x, y)
            fs = 0.5 if S != S else min(1.0, S / self._SIGHT_NORM_G)
        eta = self._eta_est(x, y)
        return (self._HOLD_W_COVER + self._HOLD_W_SIGHT * fs
                - self._HOLD_W_CENTER * math.hypot(x - z.next_center[0], y - z.next_center[1]) / max(1.0, z.next_radius)
                - self._HOLD_W_TRAVEL * eta / self._ETA_NORM_S
                - self._HOLD_W_DANGER * self._heat(world, x, y))

    def _rank_step(self, world, A, sl: Slice):
        t = self._t
        kinds = [k for k, w in self._spot_wanted.items() if t - w <= 3.0]
        if not kinds:
            return
        for _ in range(len(kinds)):
            self._spot_turn = (self._spot_turn + 1) % len(kinds)
            kind = kinds[self._spot_turn]
            have = self._spots.get(kind)
            gen = self._spot_jobs.get(kind)
            if gen is None:
                if have is not None and t - have[1] < 2.0:
                    continue
                gen = self._spot_jobs[kind] = self._rank_gen(kind, world, A)
            while True:
                try:
                    next(gen)
                except StopIteration as e:
                    self._spots[kind] = (e.value, t)
                    del self._spot_jobs[kind]
                    break
                if sl.expired():
                    return
            if sl.expired():
                return

    # ================================================================== anytime (6.1 step 10)
    def anytime(self, world, A, clock=None) -> None:
        """Zone field slice (own budget, ZONE_FIELD_MIN_MS guaranteed while slack is short), then one
        nav slice (flood / A*, NAV_MS_CAP), then spot ranking (RANK_MS_CAP).

        Budget (3.3): every slice gets min(its cap x DEGRADE_SCALE^degrade_level, time left to the soft
        deadline - ANYTIME_RESERVE_MS), measured from t0; at most one heavy slice (zone field or nav) per
        tick unless the time left still holds a full nav cap after the zone slice."""
        if clock is None and hasattr(A, "slice"):     # anytime(world, clock) as in main.py's sketch
            A, clock = None, A
        guaranteed_used = False
        heavy = False
        scale = self._DEGRADE_SCALE ** max(0, int(self.degrade_level))
        reserve = self._ANYTIME_RESERVE_MS

        def left():
            return clock.soft_left_ms() - reserve
        try:
            self._sync_world(world)
            self._prune_blocked()
            if self._zfields:
                f = next((f for f in self._zfields if f.phase != "done"), None)
                if f is not None:
                    slack = getattr(A, "slack", INF) if A is not None else INF
                    g = self._ZONE_FIELD_MIN_MS if slack < self._ZONE_SOFT_MULT * self._ZONE_SLACK_SOFT_S else 0.0
                    cap = min(self._ZONE_FIELD_MS * scale, left())
                    if cap > 0.0 or g > 0.0:
                        guaranteed_used = g > 0 and cap < g
                        sl = clock.slice(max(cap, 0.0), g)
                        heavy = True
                        while f is not None and not sl.expired():
                            self._zone_step(f, sl)
                            f = next((f for f in self._zfields if f.phase != "done"), None)
                        self.stats["zone_slices"] += 1
        except Exception:
            self.exceptions += 1
        try:
            cap = min(self._NAV_MS_CAP * scale, left())
            if cap > 0.0 and (not heavy or cap >= self._NAV_MS_CAP * scale):
                self._nav_step(clock.slice(cap))
        except Exception:
            self.exceptions += 1
        try:
            cap = min(self._RANK_MS_CAP * scale, left())
            if not guaranteed_used and cap > 0.0:
                self._rank_step(world, A, clock.slice(cap))
                self.stats["rank_slices"] += 1
        except Exception:
            self.exceptions += 1
        self._prune_paths()

    def _pending_astar(self):
        return [j for j in self._paths.values() if j.search is not None and not j.search.done]

    def _nav_step(self, sl: Slice):
        """Flood or A*, alternating while both have work; leftover slice time goes to the next job."""
        while not sl.expired():
            jobs = self._pending_astar()
            flood = self._flood if (self._flood is not None and not self._flood.done) else None
            if flood is None and not jobs:
                for job in self._paths.values():
                    if job.path is not None and job.search is None and job.raw_stop < len(job.raw[0]) - 1:
                        self._refine(job, sl)
                        if sl.expired():
                            return
                return
            use_flood = flood is not None and (not jobs or self._nav_turn == 0)
            self._nav_turn ^= 1
            if use_flood:
                self._run(flood, sl)
                self.stats["flood_slices"] += 1
                if flood.done:
                    self._flood_done = flood
                    self._flood = None
            else:
                job = min(jobs, key=lambda j: j.created_t)
                self._run(job.search, sl)
                self.stats["astar_slices"] += 1
                if job.search.done:
                    self._finish_job(job, sl)

    def _prune_paths(self):
        t = self._t
        for k in [k for k, j in self._paths.items()
                  if t - j.req_t > 10.0 and k != self._last_follow_key]:
            del self._paths[k]

    # ------------------------------------------------------------------ synchronous helpers (tools/tests)
    def run_until_done(self, max_ms: float = 5000.0) -> None:
        """Run all pending zone / flood / A* work to completion (tests, boot-time warmup)."""
        import time as _t
        end = _t.perf_counter() + max_ms / 1000.0
        sl = Slice(end)
        for f in self._zfields:
            while f.phase != "done" and not sl.expired():
                self._zone_step(f, sl)
        self._nav_step(sl)
