"""nav.py tests (plan 14.1 / 14.2 navsim gates, 8.8 zone field, R13.1 ladder, 3.3 slice budgets).

    cd bot && python3.13 -m unittest tests.test_nav -v

Needs ../mechfight-kit/arena (map files) and precomp/map.bin; skipped when the kit is absent.
Timings print p50/p99 per slice to stderr (they are the numbers for the report).
"""
from __future__ import annotations

import heapq
import math
import os
import random
import sys
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.dont_write_bytecode = True

import config  # noqa: E402
import mapdata as M  # noqa: E402
import nav as NAV  # noqa: E402
import navsim as NS  # noqa: E402
from contracts import Clock, SelfState, Slice, ZoneState  # noqa: E402

KIT = os.path.join(os.path.dirname(REPO), "mechfight-kit")
ARENA = os.path.join(KIT, "arena")
HAVE_MAP = os.path.exists(os.path.join(ARENA, "v2", "world.json"))

_MD = None
_C = None


def setup_map():
    global _MD, _C
    if _MD is None:
        _C = config.bind(None)
        _MD = M.MapData.load(ARENA, _C)
    return _MD, _C


def zone(center, radius, next_center, next_radius, start=1e6, end=1e6 + 1):
    return ZoneState(center=center, radius=radius, next_center=next_center, next_radius=next_radius,
                     shrink_start=start, shrink_end=end, from_center=center, from_radius=radius)


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, max(0, int(math.ceil(p / 100.0 * len(xs))) - 1))
    return xs[k]


def find_wall_spot(md, nav, roof_min_area=4000.0):
    """(stand point, outward normal, roof) of a climb face on a big roof away from the map edge."""
    for roof in md.roofs[1:]:
        if roof.area < roof_min_area:
            continue
        cx, cy = roof.centroid
        b = md.bounds
        if not (b[0] + 120 < cx < b[2] - 120 and b[1] + 120 < cy < b[3] - 120):
            continue
        for f in nav.climb_faces(roof.id):
            if f["h"] < 8:
                continue
            ox, oy = f["x"] + f["nx"] * 10, f["y"] + f["ny"] * 10
            c = md.cell4(ox, oy)
            if c >= 0 and md.nav_flags(c) & M.NAV_WALKABLE and not md.nav_flags(c) & M.NAV_ROOF:
                return f, roof
    return None, None


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestNavsim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def test_navsim_success_rate(self):
        """14.2: >= NAVSIM_SUCCESS_MIN over NAVSIM_PAIRS random walkable pairs within NAVSIM_ETA_K x ETA."""
        n = int(self.C.NAVSIM_PAIRS)
        ok, total, fails = NS.run_pairs(self.md, self.C, n=n, seed=1, eta_k=float(self.C.NAVSIM_ETA_K))
        rate = ok / max(1, total)
        print("\n[navsim] %d / %d = %.1f%%" % (ok, total, 100 * rate), file=sys.stderr)
        self.assertGreaterEqual(total, n * 0.9)
        self.assertGreaterEqual(rate, float(self.C.NAVSIM_SUCCESS_MIN))

    def test_roof_climbs(self):
        """14.2: climbs to 10 roofs succeed within the planned energy (CLIMB_COST(h) + pad)."""
        md, C = self.md, self.C
        nav0 = NAV.Nav(C, md)
        done = 0
        tried = 0
        for roof in md.roofs[1:]:
            if done >= 10:
                break
            if not (C.ROOF_MIN_H <= roof.h <= C.ROOF_MAX_H):
                continue
            faces = [f for f in nav0.climb_faces(roof.id) if C.ROOF_MIN_H - 2 <= f["h"] <= C.ROOF_MAX_H]
            if not faces:
                continue
            f = faces[len(faces) // 2]
            sx, sy = f["x"] + f["nx"] * 15, f["y"] + f["ny"] * 15
            c = md.cell4(sx, sy)
            if c < 0 or not md.nav_flags(c) & M.NAV_WALKABLE or md.nav_flags(c) & M.NAV_ROOF:
                continue
            start = md.cell4_xy(c)
            if md.clearance(start[0], start[1]) < 2.0:
                continue
            tried += 1
            goal = (f["rx"] - f["nx"] * 4, f["ry"] - f["ny"] * 4)
            if md.roof_id(goal[0], goal[1]) != roof.id:
                goal = (f["rx"], f["ry"])
            nav = NAV.Nav(C, md)
            sim = NS.NavSim(md)
            sim.reset((start[0], start[1], md.nav_h(c)))
            res = NS.drive(nav, sim, goal, 30.0, cost={"energy": 100.0, "reserve": C.CLIMB_NAV_RESERVE},
                           arrive_r=3.0)
            on_roof = md.roof_id(sim.x, sim.y) == roof.id and sim.z > roof.top - 2.0
            planned = nav.climb_cost(f["h"]) + 10.0
            if res["success"] and on_roof:
                self.assertLessEqual(res["energy_used"], planned,
                                     "roof %d used %.1f energy > planned %.1f" % (roof.id, res["energy_used"], planned))
                done += 1
        print("\n[climbs] %d roofs climbed of %d tried" % (done, tried), file=sys.stderr)
        self.assertGreaterEqual(done, 10)
        self.assertGreaterEqual(done / max(1, tried), 0.8)


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestZoneField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def brute_eta(self, nav, f, c0):
        """Independent forward Dijkstra from c0 to S over mapdata's public edge API (no nav internals)."""
        md, C = self.md, self.C
        inv = 1.0 / nav.walk
        K = C.SLOPE_COST_K
        nx = md.nx4
        dist = {c0: 0.0}
        heap = [(0.0, c0)]
        done = set()

        def in_s(c):
            if f.tiny:
                return c in f.scells
            x, y = md.cell4_xy(c)
            return md.nav_flags(c) & M.NAV_WALKABLE and math.hypot(x - f.sx, y - f.sy) <= f.sr

        def in_r(c):
            x, y = md.cell4_xy(c)
            return math.hypot(x - f.rx, y - f.ry) <= f.rr + 1e-6

        while heap:
            g, c = heapq.heappop(heap)
            if c in done:
                continue
            done.add(c)
            if in_s(c):
                return g
            w, j = md.walk_edges(c), md.jump_edges(c)
            for k, (di, dj) in enumerate(M.DIRS8):
                if not ((w | j) >> k) & 1:
                    continue
                v = c + di + dj * nx
                if not in_r(v):
                    continue
                L = math.hypot(di, dj) * 4.0
                dh = md.nav_h(v) - md.nav_h(c)
                cost = (L + K * max(0.0, dh)) * inv + (0.0 if (w >> k) & 1 else C.JUMP_EDGE_COST_S)
                ng = g + cost
                if ng < dist.get(v, math.inf):
                    dist[v] = ng
                    heapq.heappush(heap, (ng, v))
        return math.inf

    def test_field_matches_brute_force(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        z = zone((-60.0, 90.0), 90.0, (-40.0, 110.0), 40.0)
        nav.zone_field_start(z, outside_current=False)
        nav.run_until_done(10000)
        self.assertTrue(nav.zone_field_done())
        f = nav._zmain
        rng = random.Random(3)
        cells = [c for c in range(md.nx4 * md.ny4) if f.inr[c] and md.nav_flags(c) & M.NAV_WALKABLE]
        sample = rng.sample(cells, min(150, len(cells)))
        checked = 0
        for c in sample:
            want = self.brute_eta(nav, f, c)
            got = f.d[c] if f.closed[c] else math.inf
            if want == math.inf:
                self.assertEqual(got, math.inf, "cell %d settled but unreachable by brute force" % c)
                continue
            self.assertAlmostEqual(got, want, places=6, msg="cell %d" % c)
            checked += 1
        self.assertGreater(checked, 50)

    def test_tiny_safe_set(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        nc = (20.0, 10.0)
        z = zone((0.0, 0.0), 30.0, nc, 3.0)            # next_radius - margin < ZONE_TINY_R
        nav.zone_field_start(z, False)
        f = nav._zmain
        self.assertTrue(f.tiny)
        self.assertTrue(f.scells)
        for c in f.scells:
            x, y = md.cell4_xy(c)
            self.assertLessEqual(math.hypot(x - nc[0], y - nc[1]), f.sr + 1e-6)
            self.assertTrue(md.nav_flags(c) & M.NAV_WALKABLE)
        self.assertLessEqual(f.sr, C.ZONE_TINY_R + 1e-6)
        nav.run_until_done(5000)
        self.assertTrue(nav.zone_field_done())
        sx, sy = md.cell4_xy(f.scells[0])
        eta, settled = nav.zone_eta((sx, sy, md.nav_h(f.scells[0])))
        self.assertTrue(settled)
        self.assertEqual(eta, 0.0)
        self.assertIsNone(nav.zone_descent((sx, sy, md.nav_h(f.scells[0]))))
        # a point 25 m away gets a finite settled ETA and a descent direction toward S
        p = (nc[0] + 25.0, nc[1], 0.0)
        eta, settled = nav.zone_eta(p)
        self.assertTrue(settled)
        self.assertGreater(eta, 0.0)
        self.assertIsNotNone(nav.zone_descent(p))
        # a centre with no walkable cell within ZONE_TINY_R: the set grows until one exists
        N = md.nx4 * md.ny4
        far = None
        for c in range(N):
            if md.nav_flags(c) & M.NAV_WALKABLE:
                continue
            x, y = md.cell4_xy(c)
            if not nav._cells_in_disc(x, y, C.ZONE_TINY_R) and nav._cells_in_disc(x, y, 30.0):
                far = (x, y)
                break
        if far is not None:
            nav.zone_field_start(zone(far, 80.0, far, 2.0), False)
            self.assertTrue(nav._zmain.scells)
            self.assertGreater(nav._zmain.sr, C.ZONE_TINY_R)

    def test_zone_escape_around_building(self):
        """Mech behind a building from S: the field routes around it and reaches S before tau_edge."""
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        self.assertIsNotNone(f, "no suitable building found")
        cx, cy = roof.centroid
        # S on the far side of the building, the mech just outside the near wall
        dx, dy = cx - f["x"], cy - f["y"]
        dl = math.hypot(dx, dy)
        ux, uy = dx / dl, dy / dl
        span = max(30.0, math.sqrt(roof.area))
        sc = (cx + ux * (span * 0.5 + 25.0), cy + uy * (span * 0.5 + 25.0))
        start = (f["x"] + f["nx"] * 4.0, f["y"] + f["ny"] * 4.0)
        sr = 12.0
        cur_r = math.hypot(start[0] - sc[0], start[1] - sc[1]) + 60.0
        z = zone(sc, cur_r, sc, sr + C.ZONE_EDGE_MARGIN_M)
        nav.zone_field_start(z, False)
        nav.run_until_done(10000)
        sim = NS.NavSim(md)
        c0 = md.cell4(start[0], start[1])
        sim.reset((start[0], start[1], md.nav_h(c0)))
        eta0, settled = nav.zone_eta((sim.x, sim.y, sim.z))
        self.assertTrue(settled)
        straight = math.hypot(sim.x - sc[0], sim.y - sc[1]) - sr
        self.assertGreater(eta0 / C.ZONE_ETA_SAFETY * nav.walk, straight * 1.05,
                           "field should route around the building")
        # a shrinking circle around S whose edge reaches our start at 1.3 x eta0: needs the detour
        # the field found, not more
        d0 = math.hypot(start[0] - sc[0], start[1] - sc[1])
        z2 = zone(sc, cur_r, sc, d0 - 0.01, start=0.0, end=eta0 * 1.3 * (cur_r - d0 + 0.01) / (cur_r - d0 + 0.01))
        tau = nav.tau_edge((sim.x, sim.y), 0.0, z2)
        t_edge = tau
        reached = None
        while sim.t < t_edge + 5.0:
            me = sim.self_state()
            if math.hypot(sim.x - sc[0], sim.y - sc[1]) <= sr:
                reached = sim.t
                break
            d = nav.zone_descent(me.pos)
            if d is None:
                reached = sim.t
                break
            world = NS.FakeWorld(me, sim.t)
            st = nav._hop_steer(d, me.pos, me, "O1")
            leg, jmp, bst = nav.stuck_update(world, d)
            mv = leg.direction if leg is not None else d
            sim.decide(mv, bool(st.climb_boost or (bst and bst.want)), bool(st.jump or jmp))
        self.assertIsNotNone(reached, "never reached S")
        print("\n[zone escape] eta %.1f s (straight %.1f m), reached in %.1f s, tau_edge %.1f s"
              % (eta0, straight, reached, tau), file=sys.stderr)
        self.assertLess(reached, tau)
        self.assertLess(reached, eta0 * 1.5 + 2.0)

    def test_outside_current_field_first(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        z = zone((0.0, 0.0), 60.0, (10.0, 0.0), 30.0)
        nav.zone_field_start(z, outside_current=True)
        self.assertEqual(len(nav._zfields), 2)
        nav.run_until_done(10000)
        p = (95.0, 0.0, md.surface_h(95.0, 0.0))
        d = nav.zone_descent(p)
        self.assertIsNotNone(d)
        self.assertLess(d[0], 0.3)          # heads back toward the circle (west-ish)

    def test_tau_edge_and_slack(self):
        z = zone((0.0, 0.0), 100.0, (0.0, 0.0), 50.0, start=10.0, end=20.0)
        self.assertEqual(NAV.tau_edge(z, (0.0, 0.0), 0.0), math.inf)
        self.assertEqual(NAV.tau_edge(z, (120.0, 0.0), 0.0), 0.0)
        te = NAV.tau_edge(z, (75.0, 0.0), 0.0, 30)
        self.assertAlmostEqual(te, 15.0, places=3)
        md, C = setup_map()
        nav = NAV.Nav(C, md)
        nav.zone_field_start(z, False)
        s, te2, eta, settled = nav.slack((75.0, 0.0, 0.0), 0.0, z)
        self.assertAlmostEqual(te2, te, places=3)
        self.assertAlmostEqual(s, te2 - eta, places=6)
        self.assertFalse(settled)          # field not run yet -> fallback ETA


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena missing")
class TestWideZoneField(unittest.TestCase):
    """The next circle is mostly one roof behind a long building (minisim seed 501, 350 s): the field
    restricted to the current circle never reached a mech north of the building, zone_eta fell back to
    the straight line (4 s) and the bot stayed outside until the zone had it."""

    POS = (-82.3, -19.7, -4.5)

    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def _zone(self):
        return ZoneState(center=(-82.38, -45.94), radius=70.0, next_center=(-53.56, -55.91), next_radius=25.0,
                         shrink_start=350.0, shrink_end=375.0, dps=160.0, from_center=(-82.38, -45.94),
                         from_radius=70.0)

    def test_restricted_field_misses_the_cell_and_the_wide_one_reaches_it(self):
        nv = NAV.Nav(self.C, self.md, random.Random(1))
        nv.zone_field_start(self._zone(), False)             # no position synced: no widening
        nv.run_until_done(8000)
        self.assertFalse(nv._zmain.wide)
        eta0, settled0 = nv.zone_eta(self.POS)
        self.assertFalse(settled0)
        nv._sync_me(SelfState(pos=self.POS, stats={}, energy=100, max_energy=100))
        self.assertIsNotNone(nv.zone_descent(self.POS))      # asks for the wide field
        nv.run_until_done(8000)
        f = nv._zmain
        self.assertTrue(f.wide and f.phase == "done")
        eta, settled = nv.zone_eta(self.POS)
        self.assertTrue(settled)
        self.assertGreater(eta, eta0 * 1.5)                  # the real route is a climb, not a straight line
        self.assertLess(eta, 60.0)
        self.assertLess(f.pops, 20000)                       # stops WIDE_FIELD_EXTRA_S past our cell
        self.assertEqual(nv.stats.get("zone_widen"), 1)
        nv.zone_descent(self.POS)
        self.assertEqual(nv.stats.get("zone_widen"), 1)      # once per circle

    def test_widening_happens_when_the_field_finishes_with_us_unreached(self):
        nv = NAV.Nav(self.C, self.md, random.Random(1))
        nv._sync_me(SelfState(pos=self.POS, stats={}, energy=100, max_energy=100))
        nv.zone_field_start(self._zone(), False)
        nv.run_until_done(8000)
        self.assertTrue(nv._zmain.wide)
        self.assertTrue(nv.zone_eta(self.POS)[1])


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena missing")
class TestGrownMechPaths(unittest.TestCase):
    """R12.3: a mech that has grown from cubes still gets a path out from a cliff foot (minisim seed 503:
    the nav graph closed every cell near the cliff, the search found no neighbour, and the bot paced
    along the wall until the zone killed it)."""

    CLIFF_FOOT = (-208.0, 140.0, 4.4)        # west foot of the ridge, 10 m ledge 2-4 m to the east
    SAFE_POINT = (-105.0, 128.0)             # the nearest safe point the bot was asked to reach

    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def _nav(self, cubes, pos=None):
        nv = NAV.Nav(self.C, self.md, random.Random(1))
        k = 1.0 + 0.05 * cubes
        me = SelfState(id=0, pos=pos or self.CLIFF_FOOT, hp=1400.0, max_hp=1400.0, energy=100.0, max_energy=100.0,
                       weapons=[None, None], radius=2.5 * k, s=k,
                       stats={"walk_speed": 11.7, "radius": 2.5 * k, "height": 9.0 * k, "eye_height": 8.0 * k})
        nv._sync_me(me)
        return nv

    def _path(self, nv):
        nv.path_request("t", self.CLIFF_FOOT, self.SAFE_POINT, {}, 150.0)
        nv.run_until_done(8000)
        return nv.path("t")

    def test_path_out_at_every_size(self):
        base = self._path(self._nav(0))
        self.assertTrue(base is not None and base.complete)
        for cubes in (1, 2, 5, 10, 22):
            p = self._path(self._nav(cubes))
            self.assertTrue(p is not None and p.complete, f"no path with {cubes} cubes")
            self.assertGreater(len(p.points), 2)
            # tight cells are closed unless nothing else gets out: a big mech may walk a longer way round
            self.assertLess(p.eta_s, base.eta_s * 3 + 30, f"{cubes} cubes")

    def test_cliff_foot_at_5_cubes_needs_the_open_retry(self):
        """At 5 cubes the cliff foot is connected to the rest only through tight cells: the closed search
        fails and the one open retry finds the path. A mech with no size penalty never retries."""
        nv = self._nav(5)
        p = self._path(nv)
        self.assertTrue(p is not None and p.complete)
        self.assertGreaterEqual(nv.stats.get("tight_retry", 0), 1)
        nv = self._nav(0)
        self._path(nv)
        self.assertEqual(nv.stats.get("tight_retry", 0), 0)

    def test_a_failed_retry_finishes_the_job(self):
        """The retry is made once: a goal no search reaches ends as an incomplete path after two searches."""
        nv = self._nav(5)
        nv.path_request("t", self.CLIFF_FOOT, self.SAFE_POINT, {}, 150.0)
        job = nv._paths["t"]
        job.search.max_exp = 50                     # both searches stop far from the goal
        nv._ASTAR_MAX_EXPANSIONS = 50
        nv.run_until_done(8000)
        self.assertIsNone(job.search)
        self.assertFalse(nv.path("t").complete)
        self.assertEqual(nv.stats.get("tight_retry", 0), 1)

    def test_grown_mech_on_open_ground_keeps_off_tight_cells(self):
        """Whenever the closed search reaches the goal, the path never walks into a tight cell, nor along
        tight cells (open mode is a retry only; tight -> tight is for a search that starts on a tight cell):
        40 seeded pairs of non-tight cells at 5 cubes. The paths are those of the committed code."""
        md, nv0 = self.md, NAV.Nav(self.C, self.md, random.Random(1))
        pairs = NS.random_pairs(md, nv0, 400, random.Random(11), allow_roof=False)
        nv = self._nav(5)
        done = 0
        for a, b, _eta in pairs:
            sc = nv.walkable_cell(a[0], a[1], a[2])
            gc = nv.walkable_cell(b[0], b[1], None, rings=2)
            if sc < 0 or gc < 0 or nv._cell_status(sc) != 1 or nv._cell_status(gc) != 1:
                continue
            probe = nv._new_search("astar", sc, {}, goal_cell=gc, goal=b)
            probe.max_exp = int(nv._ASTAR_MAX_EXPANSIONS)
            nv._run(probe, None)
            if not probe.found:
                continue
            nv.path_request("p", a, b, {}, 0.0)
            nv.run_until_done(8000)
            job = nv._paths.pop("p")
            self.assertTrue(job.path.complete)
            # the committed rule: a tight cell is only ever a climb landing (never walked into), and it
            # is left onto open ground; the job's path is the closed search's path
            cells, c = [], probe.best
            while c != -1:
                cells.append(c)
                c = probe.prev[c]
            self.assertEqual(job.cells, cells[::-1], (a, b))
            for p, c in zip(job.cells, job.cells[1:]):
                if nv._cell_status(c) == 3:
                    self.assertEqual(probe.etype[c], 2, (a, b, p, c))
                if nv._cell_status(p) == 3:
                    self.assertNotEqual(nv._cell_status(c), 3, (a, b, p, c))
            done += 1
            if done == 40:
                break
        self.assertEqual(done, 40)

    def test_closed_search_from_open_ground_never_walks_into_tight_cells(self):
        """A loot flood (it climbs) from open ground: a tight cell is reached only by a climb landing,
        never by a walk, so no wall-hugging route and no shortcut along a wall (travel_time as committed)."""
        nv = self._nav(5)
        rng = random.Random(3)
        done = 0
        for a, b, _eta in NS.random_pairs(self.md, NAV.Nav(self.C, self.md, random.Random(1)), 60, rng, allow_roof=False):
            sc = nv.walkable_cell(a[0], a[1], a[2])
            if sc < 0 or nv._cell_status(sc) != 1:
                continue
            s = nv._new_search("flood", sc, {})
            nv._run(s, None)
            self.assertGreater(sum(1 for c in range(nv.N) if s.dist[c] < NAV.INF), 1000)
            walked = [c for c in range(nv.N) if s.dist[c] < NAV.INF and nv._cell_status(c) == 3 and s.etype[c] != 2]
            self.assertEqual(walked, [], a)
            done += 1
            if done == 4:
                break
        self.assertEqual(done, 4)

    def test_closed_search_from_a_tight_cell_walks_out(self):
        nv = self._nav(5)
        sc = nv.walkable_cell(*self.CLIFF_FOOT)
        self.assertEqual(nv._cell_status(sc), 3)
        s = nv._new_search("flood", sc, {}, radius=40.0)
        nv._run(s, None)
        out = [c for c in range(nv.N) if s.dist[c] < NAV.INF and nv._cell_status(c) == 1]
        self.assertGreater(len(out), 20)

    def _cover_spots(self, nv):
        """(me position, threat position) beside big buildings, the threat on the far side."""
        out = []
        for roof in self.md.roofs[1:]:
            if roof.area < 4000.0:
                continue
            for f in nv.climb_faces(roof.id):
                if f["h"] < 8:
                    continue
                for dist in (6.0, 10.0, 14.0):
                    mx, my = f["x"] + f["nx"] * dist, f["y"] + f["ny"] * dist
                    tx, ty = f["x"] - f["nx"] * 60.0, f["y"] - f["ny"] * 60.0
                    mz, tz = nv._h1(mx, my), nv._h1(tx, ty)
                    if mz is not None and tz is not None:
                        out.append(((mx, my, mz), (tx, ty, tz)))
        random.Random(5).shuffle(out)
        return out

    def _cover_cells(self, nv, spots, open_search):
        """Status of the nav cell of the cover point of each spot whose own cell is not tight."""
        if open_search:                            # the old rule, emulated: tight cells open in every search
            new = nv._new_search
            nv._new_search = lambda *a, **k: new(*a, **dict(k, tight_open=True))
        out = []
        for me_pos, thr in spots:
            me = SelfState(id=0, pos=me_pos, hp=1400.0, max_hp=1400.0, energy=100.0, max_energy=100.0,
                           weapons=[None, None], radius=3.125, s=1.25,
                           stats={"walk_speed": 11.7, "radius": 3.125, "height": 11.25, "eye_height": 10.0})
            nv._sync_me(me)
            c0 = nv.walkable_cell(*me_pos)
            if c0 < 0 or nv._cell_status(c0) != 1:      # a start in a tight cell or a low underpass is no case
                continue
            pt = nv.cover_query(me, [thr], 15.0, Slice(time.perf_counter() + 5.0))
            if pt is not None:
                out.append(nv._cell_status(nv._cell_raw(pt[0], pt[1])))
        return out

    def test_cover_never_returns_a_tight_cell(self):
        nv = self._nav(5)
        spots = self._cover_spots(nv)       # ~2000 spots, 2 s: tight cover is rare (7 on this map)
        closed = self._cover_cells(nv, spots, False)
        self.assertGreater(len(closed), 100, "too few spots with cover: the test proves nothing")
        self.assertNotIn(3, closed)
        nv = self._nav(5)
        opened = self._cover_cells(nv, spots, True)
        self.assertIn(3, opened, "open-everywhere search never returned tight cover here: the test cannot fail")

    def test_low_underpasses_stay_closed_in_both_modes(self):
        nv = self._nav(5)
        nv0 = self._nav(0)
        low = {i for i in range(nv.N) if nv._F[i] & NAV.F_UNDER and nv._U[i] * 0.1 < nv.height + 0.3}
        self.assertTrue(low)
        reached = 0
        checked = 0
        for u in sorted(low):
            for k in range(8):
                v = u + nv._OFF[k]
                if 0 <= v < nv.N and nv._F[v] & NAV.F_WALK and not nv._F[v] & NAV.F_UNDER and nv._cell_status(v) == 1:
                    break
            else:
                continue
            for mode in (False, True):
                s = nv._new_search("astar", v, {}, radius=24.0, tight_open=mode)
                nv._run(s, None)
                self.assertEqual(s.dist[u], NAV.INF, (u, mode))
            s0 = nv0._new_search("astar", v, {}, radius=24.0)
            nv0._run(s0, None)
            reached += s0.dist[u] < NAV.INF           # the control: a mech that fits does get in
            checked += 1
            if checked == 12:
                break
        self.assertGreater(reached, 0)

    def test_zone_field_reaches_the_cliff_foot(self):
        z = zone((92.0, 21.0), 380.0, (120.0, 94.0), 240.0, start=155.0, end=195.0)
        for cubes in (0, 5):
            nv = self._nav(cubes)
            nv.zone_field_start(z, False)
            nv.run_until_done(8000)
            if cubes:
                self.assertTrue(nv._zmain.wide)               # only the widened (open) field gets there
            eta, settled = nv.zone_eta(self.CLIFF_FOOT)
            self.assertTrue(settled, f"{cubes} cubes")
            self.assertTrue(10.0 < eta < 120.0, (cubes, eta))
            self.assertIsNotNone(nv.zone_descent(self.CLIFF_FOOT))

    def test_closed_field_enters_a_tight_seed_cell_from_open_ground(self):
        """A seed of the safe set that is a tight cell is entered for free in an ordinary field, as in the
        committed code: otherwise whole areas whose way into the band crosses a tight seed lose their value
        (5 cubes, zone seen from far away, so the field stays closed)."""
        nv = self._nav(5, pos=(-427.5, 139.5, -5.77))
        nv._pos_seen = False
        nv.zone_field_start(zone((131.18, 103.92), 349.5, (88.60, 194.38), 186.15, start=155.0, end=195.0), False)
        nv.run_until_done(8000)
        f = nv._zmain
        self.assertFalse(f.wide)
        c = nv.walkable_cell(-27.5, 327.5, None, rings=2)
        self.assertTrue(f.closed[c])
        self.assertLess(f.d[c], 3.0)               # 0.34 s with tight cells closed; 12.6 s when seeds were not entered

    def test_closed_tiny_field_takes_no_climb_onto_a_tight_landing(self):
        """A tiny next circle on a roof: the ground gets its value by walking round (committed: 41.6 s at
        this cell), not through a climb onto a tight roof edge (12.8 s)."""
        nv = self._nav(5)
        nv._pos_seen = False
        nv.zone_field_start(zone((-31.5, 127.5), 380.0, (-31.5, 127.5), 1.0, start=155.0, end=195.0), False)
        nv.run_until_done(8000)
        f = nv._zmain
        self.assertTrue(f.tiny and f.climbs and not f.wide)
        c = nv.walkable_cell(-35.5, 71.5, None, rings=2)
        self.assertTrue(f.closed[c])
        self.assertGreater(f.d[c], 30.0)

    def test_descent_steps_add_up_to_the_field(self):
        """Following _descend from any settled cell must cost exactly the field value there: the step is
        priced like the field (tight-cell costs, climb limit). Before, the step ignored the tight costs and
        left the field's route for cells the field had priced out. Holds for an ordinary (closed) field,
        where a tight cell is entered only from a tight one, and for a widened (open) one."""
        z = zone((92.0, 21.0), 380.0, (120.0, 94.0), 240.0, start=155.0, end=195.0)
        nv = self._nav(5)                           # at the cliff foot: the ordinary field never reaches it
        nv.zone_field_start(z, False)
        nv.run_until_done(8000)
        self.assertTrue(nv._zmain.wide)
        self._check_descent(nv, nv._zmain)
        nv = self._nav(5, pos=(-60.0, 100.0, self.md.nav_h(self.md.cell4(-60.0, 100.0))))
        nv.zone_field_start(z, False)
        nv.run_until_done(8000)
        self.assertFalse(nv._zmain.wide)
        self._check_descent(nv, nv._zmain)

    def _check_descent(self, nv, f):
        K = self.C.SLOPE_COST_K * 0.01
        tight_cells = 0
        rng = random.Random(7)
        cells = [c for c in range(nv.N) if f.closed[c] and 0.0 < f.d[c] < 1e17]
        self.assertGreater(len(cells), 500)
        tight = [c for c in cells if nv._cell_status(c) == 3]
        self.assertTrue(tight, "no tight cell in the field")
        for c0 in rng.sample(cells, 300) + rng.sample(tight, min(100, len(tight))):
            c, total, steps = c0, 0.0, 0
            while f.d[c] > 0.0 and steps < 2000:
                v = nv._descend(f, c)
                self.assertGreaterEqual(v, 0, f"no descent step from cell {c} (d {f.d[c]:.2f})")
                face = nv._descend_face
                if face is not None:                                 # a widened field also has climb edges
                    self.assertTrue(f.wide)
                    total += face["d"] * f.inv + face["h"] / max(0.5, nv.rise) + nv._CLIMB_EDGE_PAD_S
                    c = v
                    steps += 1
                    continue
                k = next(k for k in range(8) if c + nv._OFF[k] == v)
                dh = nv._H[v] - nv._H[c]
                cost = ((NAV.DLEN[k] + K * dh) if dh > 0 else NAV.DLEN[k]) * f.inv
                if not (nv._W[c] >> k) & 1:
                    cost += self.C.JUMP_EDGE_COST_S
                if nv._cell_status(v) == 3:
                    tight_cells += 1
                    if nv._cell_status(c) == 3:
                        cost += self.C.SIZE_TIGHT_COST_S
                    elif f.wide:
                        cost += self.C.SIZE_TIGHT_ENTER_COST_S
                    else:                                            # a closed field: only into a seed, free
                        self.assertEqual(f.d[v], 0.0, "a closed field stepped into a tight cell from open ground")
                total += cost
                c = v
                steps += 1
            self.assertLess(steps, 2000)
            self.assertAlmostEqual(total, f.d[c0], delta=1e-6 * max(1.0, f.d[c0]) + 1e-6)
        self.assertGreater(tight_cells, 0, "the sample never crossed a tight cell: the test proves nothing")

    def test_tight_cells_cost_and_low_underpasses_stay_closed(self):
        nv = self._nav(5)
        c = nv.walkable_cell(self.CLIFF_FOOT[0], self.CLIFF_FOOT[1], self.CLIFF_FOOT[2])
        self.assertEqual(nv._cell_status(c), 3)                       # tight, not closed
        under = [i for i in range(nv.N) if nv._F[i] & NAV.F_UNDER and nv._U[i] * 0.1 < nv.height + 0.3]
        self.assertTrue(under, "the map has underpasses lower than an 11 m mech")
        self.assertEqual(nv._cell_status(under[0]), 2)


class TestStuckLadder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def open_spot(self):
        md = self.md
        for c in range(md.nx4 * md.ny4):
            x, y = md.cell4_xy(c)
            if md.nav_flags(c) & M.NAV_WALKABLE and md.clearance(x, y) > 15 and not md.nav_flags(c) & M.NAV_ROOF:
                return (x, y, md.nav_h(c))
        self.fail("no open spot")

    def me_at(self, pos, vel=(0.0, 0.0, 0.0), grounded=True, energy=100.0):
        me = SelfState(pos=pos, vel=vel, grounded=grounded, energy=energy, max_energy=100.0, stats={})
        return me

    def test_trigger_and_ladder(self):
        C = self.C
        nav = NAV.Nav(C, self.md)
        pos = self.open_spot()
        t = 0.0
        cmd = (1.0, 0.0)
        seen = []
        # not stuck while moving
        for _ in range(20):
            out = nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(5.0, 0.0, 0.0)), t), cmd)
            self.assertEqual(out, (None, None, None))
            t += 0.1
        # stuck: commanded but not moving for STUCK_T -> ladder
        t_stuck = t
        while t < t_stuck + C.NAV_RECOVER_MAX_S + 1.0:
            leg, jmp, bst = nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), cmd)
            name = nav._stuck["step_name"] if nav.stuck_active() else None
            if name and (not seen or seen[-1] != name):
                seen.append(name)
            if name == "jump":
                self.assertIsNotNone(leg)
                self.assertIsNotNone(jmp)
                self.assertEqual(leg.level, "O5")
            if name in ("side1", "side2"):
                self.assertIsNotNone(leg)
                self.assertAlmostEqual(leg.direction[0] * cmd[0] + leg.direction[1] * cmd[1], 0.0, places=6)
            t += 0.1
        self.assertGreaterEqual(t_stuck + C.STUCK_T, 0)
        self.assertEqual(seen[:3], ["jump", "side1", "side2"])
        self.assertIn("block", seen)
        self.assertTrue(nav._blocked, "the edge must be blocked (step 4)")
        self.assertFalse(nav.stuck_active(), "O5 must exit within NAV_RECOVER_MAX_S")

    def test_abandoned_climb_does_not_disable_o5(self):
        """R13.1 / R13.2 (integration fix): a climb whose owner gave up the legs (follow() no longer runs
        climb_step) times out in stuck_update, so the O5 trigger works again."""
        C = self.C
        nav = NAV.Nav(C, self.md)
        pos = self.open_spot()
        nav._climb = {"face": None, "job": None, "after": 0, "phase": "approach", "t0": 0.0}
        t = NAV.CLIMB_TIMEOUT_S + 0.1
        for _ in range(int(C.STUCK_T / 0.1) + 3):
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            t += 0.1
        self.assertIsNone(nav._climb)
        self.assertTrue(nav.stuck_active())

    def test_trigger_needs_stuck_t(self):
        C = self.C
        nav = NAV.Nav(C, self.md)
        pos = self.open_spot()
        t = 0.0
        while t < C.STUCK_T - 0.15:
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            self.assertFalse(nav.stuck_active())
            t += 0.1
        for _ in range(3):
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            t += 0.1
        self.assertTrue(nav.stuck_active())
        # small command never triggers
        nav2 = NAV.Nav(C, self.md)
        for k in range(30):
            nav2.stuck_update(NS.FakeWorld(self.me_at(pos), k * 0.1), (0.2, 0.0))
        self.assertFalse(nav2.stuck_active())

    def test_exit_on_motion(self):
        C = self.C
        nav = NAV.Nav(C, self.md)
        pos = self.open_spot()
        t = 0.0
        while not nav.stuck_active():
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            t += 0.1
        t_ok = t
        while t < t_ok + C.UNSTUCK_OK_S + 0.25:
            nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(C.UNSTUCK_OK_SPEED + 1, 0, 0)), t), (1.0, 0.0))
            t += 0.1
        self.assertFalse(nav.stuck_active())

    def test_jet_step_at_ledge(self):
        """Step 2 (jet) appears when a ledge >= STEP_UP is ahead and energy allows."""
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        pos = (f["x"], f["y"], md.surface_h(f["x"], f["y"]))
        cmd = (-f["nx"], -f["ny"])
        t = 0.0
        names = []
        while t < 3.0:
            leg, jmp, bst = nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), cmd)
            nm = nav._stuck["step_name"] if nav.stuck_active() else None
            if nm == "jet":
                self.assertIsNotNone(bst)
                self.assertEqual(bst.owner, "CLIMB")
            if nm and (not names or names[-1] != nm):
                names.append(nm)
            t += 0.1
        self.assertEqual(names[:2], ["jump", "jet"])

    def test_escape_after_repeats(self):
        C = self.C
        nav = NAV.Nav(C, self.md, rng=random.Random(5))
        pos = self.open_spot()
        t = 0.0
        episodes = 0
        while t < C.STUCK_REPEAT_WINDOW_S and episodes < C.STUCK_REPEAT_N:
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            if nav.stuck_active():
                episodes += 1
                if episodes >= C.STUCK_REPEAT_N:
                    break
                # end the episode by moving (a metre per episode: standing on one spot is a wedge, see the lift)
                for _ in range(8):
                    t += 0.1
                    pos = (pos[0] + 0.15, pos[1], pos[2])
                    nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(5, 0, 0)), t), (1.0, 0.0))
            t += 0.1
        self.assertTrue(nav.stuck_active())
        self.assertEqual(nav._stuck["steps"][0][0], "escape")
        tgt = nav._stuck["escape"]
        d = math.hypot(tgt[0] - pos[0], tgt[1] - pos[1])
        self.assertGreaterEqual(d, C.ESCAPE_MIN_M - 3.0)
        self.assertLessEqual(d, C.ESCAPE_MAX_M + 3.0)
        leg, _, _ = nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
        self.assertIsNotNone(leg)
        self.assertEqual(leg.rule, "R13.1.5")

    def test_jump_and_jet_legs_are_nav_directions(self):
        """The ladder's jump and jet point at the obstacle on purpose: the arbiter must not drop them."""
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        pos = (f["x"], f["y"], md.surface_h(f["x"], f["y"]))
        cmd = (-f["nx"], -f["ny"])
        t = 0.0
        seen = {}
        while t < 3.0:
            leg, jmp, bst = nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), cmd)
            nm = nav._stuck["step_name"] if nav.stuck_active() else None
            if nm in ("jump", "jet"):
                seen[nm] = leg.nav
            if nm in ("side1", "side2"):
                self.assertFalse(leg.nav, "a sideways step is an ordinary direction")
            t += 0.1
        self.assertEqual(seen, {"jump": True, "jet": True})

    def test_pacing_begins_an_episode(self):
        """Commanded to move, fast, but going nowhere for STUCK_PACE_S: an episode starts (the speed trigger
        never fires while pacing along a ledge). Real displacement keeps it off."""
        C = self.C
        pos = self.open_spot()
        nav = NAV.Nav(C, self.md)
        t = 0.0
        while t < nav._STUCK_PACE_S - 0.3:
            out = nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(8.0, 0.0, 0.0)), t), (1.0, 0.0))
            self.assertEqual(out, (None, None, None))
            t += 0.1
        self.assertFalse(nav.stuck_active())
        while t < nav._STUCK_PACE_S + 0.3:
            leg, _, _ = nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(8.0, 0.0, 0.0)), t), (1.0, 0.0))
            t += 0.1
        self.assertTrue(nav.stuck_active())
        self.assertIsNotNone(leg)
        # speed alone does not end a pacing episode
        for _ in range(8):
            nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(8.0, 0.0, 0.0)), t), (1.0, 0.0))
            t += 0.1
        self.assertTrue(nav.stuck_active())
        # walking away does
        for _ in range(12):
            pos = (pos[0] + 1.0, pos[1], pos[2])
            nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(8.0, 0.0, 0.0)), t), (1.0, 0.0))
            t += 0.1
        self.assertFalse(nav.stuck_active())
        # moving the whole time: never an episode
        nav2 = NAV.Nav(C, self.md)
        p = self.open_spot()
        for k in range(int(3 * nav._STUCK_PACE_S / 0.1)):
            p = (p[0] + 0.8, p[1], p[2])
            nav2.stuck_update(NS.FakeWorld(self.me_at(p, vel=(8.0, 0.0, 0.0)), k * 0.1), (1.0, 0.0))
            self.assertFalse(nav2.stuck_active())

    def test_wedge_lift(self):
        """WEDGE_REPEAT_N episodes that end where they began: the next is a lift. Jump and jet (CLIMB) until the
        feet are CLIMB_CLEAR_M above everything within radius + 3 m, toward the most open side, nav=True, and keep
        the direction for 1 s after the jet."""
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        pos = (f["x"], f["y"], md.surface_h(f["x"], f["y"]))
        cmd = (-f["nx"], -f["ny"])
        t = 0.0
        ended = 0
        was = False
        while not (nav.stuck_active() and nav._stuck["steps"][0][0] == "lift"):
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), cmd)
            if was and not nav.stuck_active():
                ended += 1
            was = nav.stuck_active()
            t += 0.1
            self.assertLess(t, 40.0, "no lift episode")
        self.assertEqual(ended, nav._WEDGE_REPEAT_N)
        st = nav._stuck
        top = st["lift_top"]
        self.assertGreater(top, pos[2] + 5.0, "the wall top is within radius + 3 m of the stand point")
        r = nav.radius + 4.0
        ring = [md.clearance(pos[0] + math.cos(k * math.pi / 8) * r, pos[1] + math.sin(k * math.pi / 8) * r)
                for k in range(16)]
        chosen = md.clearance(pos[0] + st["lift_dir"][0] * r, pos[1] + st["lift_dir"][1] * r)
        self.assertGreaterEqual(chosen, max(ring) - 1e-9)
        leg, jmp, bst = nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), cmd)
        self.assertTrue(leg.nav)
        self.assertIsNotNone(jmp)
        self.assertEqual((bst.owner, bst.want), ("CLIMB", True))
        # in the air below the top: still jetting, no jump request; the jet stops at the top and the episode
        # ends 1 s later with the direction kept until then
        air = lambda z: self.me_at((pos[0], pos[1], z), grounded=False)
        t += 0.1
        leg, jmp, bst = nav.stuck_update(NS.FakeWorld(air(top - 3.0), t), cmd)
        self.assertIsNone(jmp)
        self.assertTrue(bst.want)
        t += 0.1
        leg, jmp, bst = nav.stuck_update(NS.FakeWorld(air(top + 0.1), t), cmd)
        self.assertIsNone(bst)
        self.assertTrue(leg.nav)
        t += 0.8
        leg, _, _ = nav.stuck_update(NS.FakeWorld(air(top + 0.1), t), cmd)
        self.assertIsNotNone(leg)
        t += 0.3
        leg, _, _ = nav.stuck_update(NS.FakeWorld(air(top + 0.1), t), cmd)
        self.assertIsNone(leg)
        self.assertFalse(nav.stuck_active())

    def test_pacing_needs_a_nav_direction(self):
        """A fight owner strafing in place (random side every few seconds) commands a move and nets nothing:
        only a navigation direction (Nav.leg_nav, set by the arbiter) can count as pacing."""
        nav = NAV.Nav(self.C, self.md)
        nav.leg_nav = False
        pos = self.open_spot()
        t = 0.0
        while t < 3 * nav._STUCK_PACE_S:
            nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(8.0, 0.0, 0.0)), t), (1.0, 0.0))
            self.assertFalse(nav.stuck_active())
            t += 0.1
        nav.leg_nav = True
        while t < 4 * nav._STUCK_PACE_S + 0.5:
            nav.stuck_update(NS.FakeWorld(self.me_at(pos, vel=(8.0, 0.0, 0.0)), t), (1.0, 0.0))
            t += 0.1
        self.assertTrue(nav.stuck_active())

    def test_a_failed_lift_is_followed_by_the_escape(self):
        """Held on one spot, commanded into it: ladder, ladder, one lift, then the ladder again and its escape.
        The lift used to repeat for the rest of the match (22 in a row at a climb face) with no escape."""
        C = self.C
        nav = NAV.Nav(C, self.md, rng=random.Random(5))
        pos = self.open_spot()
        first_steps = []
        was = False
        t = 0.0
        while t < 90.0:
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            if nav.stuck_active() and not was:
                first_steps.append(nav._stuck["steps"][0][0])
            was = nav.stuck_active()
            if "escape" in first_steps:
                break
            t += 0.1
        self.assertEqual(first_steps.count("lift"), 1, first_steps)
        self.assertIn("escape", first_steps[first_steps.index("lift"):], first_steps)

    def test_wedge_lift_jet_is_capped(self):
        C = self.C
        nav = NAV.Nav(C, self.md)
        pos = self.open_spot()
        nav._stuck.update(wedge_p0=pos, wedge_n=nav._WEDGE_REPEAT_N)
        t = 0.0
        while not nav.stuck_active():
            nav.stuck_update(NS.FakeWorld(self.me_at(pos), t), (1.0, 0.0))
            t += 0.1
        nav._stuck["lift_top"] = pos[2] + 500.0                  # never reached
        t0 = nav._stuck["start"]
        last_boost = None
        while nav.stuck_active() and t < t0 + 10.0:
            leg, jmp, bst = nav.stuck_update(NS.FakeWorld(self.me_at(pos, grounded=False), t), (1.0, 0.0))
            if bst is not None:
                last_boost = t - t0
            t += 0.1
        self.assertLessEqual(last_boost, nav._WEDGE_JET_MAX_S + 1e-6)
        self.assertGreaterEqual(last_boost, nav._WEDGE_JET_MAX_S - 0.2)
        self.assertFalse(nav.stuck_active())


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestHeightAwareLegs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def me_at(self, pos, grounded=True, energy=100.0):
        return SelfState(pos=pos, vel=(0.0, 0.0, 0.0), grounded=grounded, energy=energy, max_energy=100.0, stats={})

    def test_clear_dir_knows_our_altitude(self):
        """Protocol 7: a wall is a wall unless the feet are above its top minus STEP_UP."""
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        top = nav._h1(f["rx"], f["ry"])
        d = (-f["nx"], -f["ny"])
        x, y = f["x"], f["y"]
        probe = lambda z: nav.clear_dir((x, y, z), d, C.LEG_PROBE_M)
        self.assertFalse(probe(md.surface_h(x, y)))
        self.assertFalse(probe(top - 2 * C.STEP_UP))
        self.assertTrue(probe(top - C.STEP_UP + 0.1))
        self.assertTrue(probe(top + 0.5))
        self.assertFalse(nav.clear_dir((x, y), d, C.LEG_PROBE_M), "a position without a z is on the ground")

    def settled(self, zn):
        nav = NAV.Nav(self.C, self.md)
        nav.zone_field_start(zn, False)
        nav.run_until_done(20000)
        return nav, nav._zmain

    def test_zone_steer_jumps_a_jump_edge(self):
        md, C = self.md, self.C
        zn = zone((41.3, -72.8), 560.0, (196.0, -43.0), 380.0)
        nav, f = self.settled(zn)
        cell = None
        for c in range(nav.N):
            if not f.closed[c] or f.d[c] <= 0.0 or f.d[c] >= NAV.INF:
                continue
            v = nav._descend(f, c)
            k = next((k for k in range(8) if v >= 0 and c + nav._OFF[k] == v), None)
            if k is None or nav._descend_face is not None or not (nav._J[c] >> k) & 1 or (nav._W[c] >> k) & 1:
                continue
            x, y = nav.cell_xy(c)
            r = nav._ledge_ahead((x, y, nav._H[c] * 0.01), nav._dir_to(nav.cell_xy(v), (x, y)))
            if C.STEP_UP + 0.1 < r <= C.JUMP_H:
                cell = (c, v, x, y)
                break
        self.assertIsNotNone(cell, "no jump-edge cell with a ledge in reach")
        c, v, x, y = cell
        pos = (x, y, nav._H[c] * 0.01)
        me = self.me_at(pos)
        nav._sync_world(NS.FakeWorld(me, 1.0, zn))
        st = nav.zone_steer(me)
        want = nav._dir_to(nav.cell_xy(v), pos)
        self.assertTrue(st.jump)
        self.assertAlmostEqual(st.direction[0] * want[0] + st.direction[1] * want[1], 1.0, places=6)
        self.assertIs(nav.zone_steer(me), st, "a second call in the same tick is the same answer")
        self.assertIs(nav.zone_steer(me, 1.0), st)
        # a caller that runs before the tick's first world sync (O1) passes the world time: no stale cache hit
        self.assertIsNot(nav.zone_steer(me, 1.1), st)
        self.assertAlmostEqual(nav._t, 1.1)
        nav._t = 1.0
        nav._zs_cache = None
        # in the air there is nothing to jump from
        nav2, _ = self.settled(zn)
        me2 = self.me_at(pos, grounded=False)
        nav2._sync_world(NS.FakeWorld(me2, 1.0, zn))
        self.assertFalse(nav2.zone_steer(me2).jump)
        # zone_descent stays a plain direction query
        self.assertIsNotNone(nav2.zone_descent(pos))
        self.assertFalse(nav2.climbing())

    def test_zone_steer_climbs_when_energy_allows(self):
        md, C = self.md, self.C
        nav0 = NAV.Nav(C, md)
        face, roof = find_wall_spot(md, nav0)
        cx, cy = roof.centroid
        zn = zone((cx, cy), 200.0, (cx, cy), 8.0)              # tiny safe set on the roof: the field has climbs
        g = face["g"]
        for energy, climbs in ((100.0, True), (5.0, False)):
            nav, f = self.settled(zn)
            self.assertTrue(f.climbs)
            pos = (*nav.cell_xy(g), nav._H[g] * 0.01)
            me = self.me_at(pos, energy=energy)
            nav._sync_world(NS.FakeWorld(me, 1.0, zn))
            st = nav.zone_steer(me)
            # the climb starts at any energy: short of it the approach waits at the foot (R13.2w) rather than
            # pushing into the face, whose energy a ground sprint would eat
            self.assertTrue(nav.climbing(), "energy %s" % energy)
            self.assertEqual(st.climb_boost, climbs)
            self.assertEqual(st.jump, climbs)
            self.assertEqual(st.rule, "R13.2j" if climbs else "R13.2w")
            self.assertEqual(st.direction is not None, climbs)
            if climbs:
                # the next ticks run the climb sequence, even where the field would say something else
                me.pos = (pos[0], pos[1], pos[2] + 3.0)
                me.grounded = False
                nav._t += 0.1
                self.assertEqual(nav.zone_steer(me).rule, "R13.2c")
            else:
                # energy comes back while we wait: the same climb goes on
                me.energy = 100.0
                nav._t += 0.1
                self.assertTrue(nav.zone_steer(me).jump)

    def test_zone_climb_nobody_steps_is_dropped_not_blocked(self):
        """A zone climb whose caller stopped asking (O1 ended, a dodge took the legs) is not run for another
        follower and does not cost the face: a timeout used to delete it for the match."""
        md, C = self.md, self.C
        nav0 = NAV.Nav(C, md)
        face, roof = find_wall_spot(md, nav0)
        cx, cy = roof.centroid
        zn = zone((cx, cy), 200.0, (cx, cy), 8.0)
        g = face["g"]
        nav, _ = self.settled(zn)
        pos = (*nav.cell_xy(g), nav._H[g] * 0.01)
        me = self.me_at(pos, energy=100.0)
        nav._sync_world(NS.FakeWorld(me, 1.0, zn))
        nav.zone_steer(me)
        self.assertTrue(nav.climbing())
        nav._t += 2.0 * nav._ZONE_CLIMB_IDLE_S
        nav.follow(me, None, (pos[0] + 30.0, pos[1]))
        self.assertFalse(nav.climbing())
        faces = [r for lst in nav._climb_from.values() for r in lst]
        self.assertIn(face["g"], [r["g"] for r in faces], "the face must still be in the climb graph")

    def test_failed_zone_climb_is_blocked(self):
        """A zone climb that fails on the face (retry / energy) is removed so the field's next step does not
        pick it again."""
        md, C = self.md, self.C
        nav0 = NAV.Nav(C, md)
        face, roof = find_wall_spot(md, nav0)
        cx, cy = roof.centroid
        zn = zone((cx, cy), 200.0, (cx, cy), 8.0)
        nav, _ = self.settled(zn)
        rec = next(r for lst in nav._climb_from.values() for r in lst if r["g"] == face["g"])
        nav._climb_begin(rec, None, None)
        nav._climb["zone"] = True
        nav._climb_abort("timeout")           # not the face's fault
        self.assertIn(rec, nav._climb_from.get(rec["g"], []))
        nav._climb_begin(rec, None, None)
        nav._climb["zone"] = True
        nav._climb_abort("retry")
        self.assertNotIn(rec, nav._climb_from.get(rec["g"], []))


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def test_clear_dir_wall(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        pos = (f["x"], f["y"], md.surface_h(f["x"], f["y"]))
        self.assertFalse(nav.clear_dir(pos, (-f["nx"], -f["ny"]), C.LEG_PROBE_M))   # into the wall
        self.assertTrue(nav.clear_dir(pos, (f["nx"], f["ny"]), C.LEG_PROBE_M))      # away from it

    def test_cover_query(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        f, roof = find_wall_spot(md, nav)
        cx, cy = roof.centroid
        # threat on the far side of the building, us 8 m outside the near wall
        tx, ty = cx + (cx - f["x"]) * 0.9, cy + (cy - f["y"]) * 0.9
        threat = (tx, ty, md.surface_h(tx, ty))
        p = (f["x"] + f["nx"] * 8, f["y"] + f["ny"] * 8)
        me = SelfState(pos=(p[0], p[1], md.surface_h(*p)), stats={}, energy=100, max_energy=100)
        pt = nav.cover_query(me, [threat], 10.0, Slice(time.perf_counter() + 0.05))
        self.assertIsNotNone(pt)
        z = md.surface_h(pt[0], pt[1])
        self.assertFalse(md.los((threat[0], threat[1], threat[2] + 8.0), (pt[0], pt[1], z + 4.5)))
        self.assertLessEqual(math.hypot(pt[0] - p[0], pt[1] - p[1]), C.COVER_SEARCH_R + 1.5)
        # budget too small -> nothing
        self.assertIsNone(nav.cover_query(me, [threat], 0.0, Slice(time.perf_counter() + 0.05)))

    def test_path_dedup_and_block(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        a, b = (0.5, -176.5), (-183.5, -308.5)
        nav.path_request("k", (a[0], a[1], md.surface_h(*a)), b, {}, 0.0)
        nav.run_until_done()
        p1 = nav.path("k")
        self.assertTrue(p1.complete)
        nav.path_request("k", (a[0] + 3, a[1], md.surface_h(*a)), (b[0] + 1, b[1]), {}, 0.1)
        self.assertIs(nav.path("k"), p1)              # same goal cell: no replan
        self.assertFalse(nav.path_pending("k"))
        # blocking an edge on the path makes it replan
        cells = nav._paths["k"].cells
        ca, cb = cells[3], cells[4]
        nav.block_edge(md.cell4_xy(ca) + (md.nav_h(ca),), md.cell4_xy(cb) + (md.nav_h(cb),), 30.0, 0.2)
        self.assertTrue(nav.path_pending("k"))
        nav.run_until_done()
        self.assertIsNot(nav.path("k"), p1)
        new_cells = nav._paths["k"].cells
        self.assertFalse(any(new_cells[i] == ca and new_cells[i + 1] == cb for i in range(len(new_cells) - 1)))

    def test_flood_travel_time(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        o = (0.5, -176.5, md.surface_h(0.5, -176.5))
        nav.flood_request(o, C.NAV_FLOOD_R, {})
        self.assertIsNone(nav.travel_time(60.0, -176.5))
        nav.run_until_done()
        tt = nav.travel_time(40.5, -176.5)
        self.assertIsNotNone(tt)
        self.assertGreaterEqual(tt, 40.0 / nav.walk - 1e-6)
        self.assertIsNone(nav.travel_time(o[0] + 400, o[1]))     # outside NAV_FLOOD_R
        # refresh rule: same origin within NAV_FLOOD_REFRESH_S is not re-flooded
        nav.flood_request(o, C.NAV_FLOOD_R, {})
        self.assertIsNone(nav._flood)
        sc = nav.loot_score(10.0, 1.0, 1.0, 40.5, -176.5)
        self.assertAlmostEqual(sc, 10.0 / (tt + 1.0 + C.LOOT_TIME_PAD_S), places=6)

    def test_spot_ranking(self):
        md, C = self.md, self.C
        nav = NAV.Nav(C, md)
        sim = NS.NavSim(md)
        sim.reset((-40.0, 20.0))
        me = sim.self_state()
        world = NS.FakeWorld(me, 100.0, zone((-42.0, 7.0), 380.0, (-42.0, 7.0), 240.0))
        A = NS._A()
        for kind in ("hold", "roof", "nook", "cover"):
            self.assertIsNone(nav.best_spot(kind, world, A))
        for _ in range(400):
            nav.anytime(world, A, Clock(time.perf_counter(), C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS))
            if all(nav.best_spot(k, world, A) is not None for k in ("hold", "roof", "nook")):
                break
        for kind in ("hold", "roof", "nook"):
            r = nav.best_spot(kind, world, A)
            self.assertIsNotNone(r, kind)
            (x, y), score, rid = r
            self.assertTrue(nav.spot_valid(kind, (x, y), world, A), kind)
            if kind == "roof":
                self.assertGreater(rid, 0)
                self.assertEqual(md.roof_id(x, y), rid)
            else:
                self.assertTrue(md.cover2(x, y) & M.COVER_WALKABLE)

    def test_degraded_map_still_navigates(self):
        md = M.MapData.load(ARENA, self.C, bin_path="/nonexistent/map.bin")
        self.assertTrue(md.degraded)
        nav = NAV.Nav(self.C, md)
        nav.path_request("k", (0.5, -176.5, 0.0), (40.5, -150.5), {}, 0.0)
        nav.run_until_done(5000)
        me = SelfState(pos=(0.5, -176.5, md.surface_h(0.5, -176.5)), stats={}, energy=100, max_energy=100)
        st = nav.follow(me, "k", (40.5, -150.5))
        self.assertIsNotNone(st.direction)


@unittest.skipUnless(HAVE_MAP, "mechfight-kit arena not found next to the repo")
class TestTimings(unittest.TestCase):
    """3.3: per-slice times on this interpreter (p50 / p99), each slice must respect its cap."""

    @classmethod
    def setUpClass(cls):
        cls.md, cls.C = setup_map()

    def report(self, name, xs, cap):
        p50, p99 = pct(xs, 50), pct(xs, 99)
        print("\n[timing %s] n=%d p50 %.2f ms p99 %.2f ms max %.2f ms (cap %.1f ms) py%d.%d" % (
            name, len(xs), p50, p99, max(xs), cap, sys.version_info[0], sys.version_info[1]), file=sys.stderr)
        self.assertLessEqual(p99, cap * 1.1 + 1.5, name)

    def test_slice_timings(self):
        md, C = self.md, self.C
        rng = random.Random(7)
        nav = NAV.Nav(C, md)
        N = md.nx4 * md.ny4
        walk = [c for c in range(N) if md.nav_flags(c) & M.NAV_WALKABLE and not md.nav_flags(c) & M.NAV_ROOF]
        # flood slices
        fl = []
        for _ in range(25):
            c = rng.choice(walk)
            x, y = md.cell4_xy(c)
            nav.flood_request((x, y, md.nav_h(c)), C.NAV_FLOOD_R, {"force": True})
            while nav._flood is not None:
                t0 = time.perf_counter()
                nav._nav_step(Clock(t0, 40, 50).slice(C.NAV_MS_CAP))
                fl.append((time.perf_counter() - t0) * 1000)
        self.report("flood slice", fl, C.NAV_MS_CAP)
        # A* slices (long paths)
        ast = []
        for k in range(25):
            a, b = rng.choice(walk), rng.choice(walk)
            pa, pb = md.cell4_xy(a), md.cell4_xy(b)
            nav.path_request("t%d" % k, (pa[0], pa[1], md.nav_h(a)), pb, {}, 0.0)
            while nav.path_pending("t%d" % k):
                t0 = time.perf_counter()
                nav._nav_step(Clock(t0, 40, 50).slice(C.NAV_MS_CAP))
                ast.append((time.perf_counter() - t0) * 1000)
            nav._paths.pop("t%d" % k, None)
        self.report("A* slice", ast, C.NAV_MS_CAP)
        # zone field slices (phase-1 circle: largest field)
        zs = []
        for k in range(4):
            z = zone((41.3, -72.8), 560.0, (-42.12 + 30 * k, 7.11), 380.0)
            nav.zone_field_start(z, outside_current=(k == 3))
            while not nav.zone_field_done():
                t0 = time.perf_counter()
                f = next(f for f in nav._zfields if f.phase != "done")
                nav._zone_step(f, Clock(t0, 40, 50).slice(C.ZONE_FIELD_MS))
                zs.append((time.perf_counter() - t0) * 1000)
        self.report("zone field slice", zs, C.ZONE_FIELD_MS)
        # urgent cover query
        cq = []
        for _ in range(60):
            c = rng.choice(walk)
            x, y = md.cell4_xy(c)
            me = SelfState(pos=(x, y, md.nav_h(c)), stats={}, energy=100, max_energy=100)
            thr = [(x + rng.uniform(-80, 80), y + rng.uniform(-80, 80), md.nav_h(c)) for _ in range(2)]
            t0 = time.perf_counter()
            nav.cover_query(me, thr, 6.0, Clock(t0, 40, 50).slice(C.COVER_URGENT_MS))
            cq.append((time.perf_counter() - t0) * 1000)
        self.report("urgent cover", cq, C.COVER_URGENT_MS)
        # follow() per tick (path following + probes)
        fo = []
        sim = NS.NavSim(md)
        for k in range(10):
            a, b = rng.choice(walk), rng.choice(walk)
            pa, pb = md.cell4_xy(a), md.cell4_xy(b)
            sim.reset((pa[0], pa[1], md.nav_h(a)))
            nav.path_request("f", (pa[0], pa[1], md.nav_h(a)), pb, {}, 0.0)
            nav.run_until_done()
            for _ in range(30):
                me = sim.self_state()
                t0 = time.perf_counter()
                st = nav.follow(me, "f", pb)
                fo.append((time.perf_counter() - t0) * 1000)
                sim.decide(st.direction, st.climb_boost, st.jump)
            nav._paths.pop("f", None)
        self.report("follow", fo, 2.0)


if __name__ == "__main__":
    unittest.main()
