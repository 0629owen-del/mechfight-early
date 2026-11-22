"""Unit tests for mapdata.py and precomp/map.bin (game-plan-v2.md sections 4.6, 11, 14.1).

    python3 -m unittest tests.test_mapdata -v

The map directory defaults to ../mechfight-kit/arena next to the bot repo; override with
MECHFIGHT_MAP_DIR. Tests that need the map are skipped when it is absent.
"""
from __future__ import annotations

import array
import hashlib
import math
import os
import random
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mapdata as M  # noqa: E402

MAP_DIR = os.environ.get("MECHFIGHT_MAP_DIR",
                         os.path.join(os.path.dirname(ROOT), "mechfight-kit", "arena"))
HAVE_MAP = os.path.exists(os.path.join(MAP_DIR, "v2", "world.json")) and os.path.exists(M.DEFAULT_BIN)


def _pct(ts, p):
    ts = sorted(ts)
    return ts[min(len(ts) - 1, int(p * len(ts)))]


@unittest.skipUnless(HAVE_MAP, "map files or precomp/map.bin missing")
class MapDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = M.MapData.load(MAP_DIR, None, time.perf_counter() + 10.0)

    # ---------------------------------------------------------------- loading
    def test_load_time_and_not_degraded(self):
        t = time.perf_counter()
        md = M.MapData.load(MAP_DIR, None, time.perf_counter() + 10.0)
        ms = (time.perf_counter() - t) * 1000
        print("\n  load: %.0f ms, map.bin %d bytes" % (ms, os.path.getsize(M.DEFAULT_BIN)))
        self.assertFalse(md.degraded, md.degraded_reason)
        self.assertLess(ms, 1500)
        self.assertTrue(md.has_sight and md.has_cover)

    def test_header_hashes(self):
        h = self.md.header
        with open(os.path.join(MAP_DIR, "v2", "world.json"), "rb") as f:
            self.assertEqual(h["world_sha256"], hashlib.sha256(f.read()).hexdigest())
        with open(os.path.join(MAP_DIR, "v2", "terrain.i16"), "rb") as f:
            self.assertEqual(h["terrain_sha256"], hashlib.sha256(f.read()).hexdigest())
        names = {b["name"] for b in h["blobs"]}
        for n in ("surface_h", "overhangs", "clear_dm", "foot_id", "nav4", "climbs", "sight5",
                  "sight5_dir", "wall5", "cover2", "roofs"):
            self.assertIn(n, names)

    def _copy_map(self, mutate_world):
        d = tempfile.mkdtemp(prefix="mf_map_")
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        shutil.copy(os.path.join(MAP_DIR, "v2", "terrain.i16"), os.path.join(d, "v2", "terrain.i16"))
        with open(os.path.join(MAP_DIR, "v2", "world.json"), "rb") as f:
            w = f.read()
        if mutate_world:
            w = w + b" \n"
        with open(os.path.join(d, "v2", "world.json"), "wb") as f:
            f.write(w)
        return d

    def test_degraded_on_hash_mismatch(self):
        d = self._copy_map(True)
        md = M.MapData.load(d, None, time.perf_counter() + 10.0)
        print("\n  degraded load (hash mismatch): %.0f ms, %s" % (md.load_ms, md.degraded_reason))
        self.assertTrue(md.degraded)
        self.assertIn("hash mismatch", md.degraded_reason)
        self.assertFalse(md.has_sight)
        self.assertTrue(math.isnan(md.sight5(0, 0)) and math.isnan(md.sight5_dir(0, 0, 2)))
        self.assertAlmostEqual(md.wall5(0, 0), self.md.wall5(0, 0), delta=4.0)   # coarse 4 m
        self.assertEqual(md.cover2(0, 0) & M.COVER_WALKABLE, self.md.cover2(0, 0) & M.COVER_WALKABLE)
        # same geometry as the precompute for LOS and ground
        rnd = random.Random(3)
        bx0, by0, bx1, by1 = md.bounds
        for _ in range(100):
            x, y = rnd.uniform(bx0, bx1), rnd.uniform(by0, by1)
            a = rnd.uniform(0, 2 * math.pi)
            q = (x + 120 * math.cos(a), y + 120 * math.sin(a))
            p3 = (x, y, self.md.surface(x, y) + 8)
            q3 = (q[0], q[1], self.md.surface(*q) + 4.5)
            self.assertEqual(md.los(p3, q3), self.md.los(p3, q3))
            self.assertEqual(md.visible(p3, q3), self.md.visible(p3, q3))
            self.assertAlmostEqual(md.ground(x, y), self.md.ground(x, y), places=6)
        # nav with lazily computed edges matches the precompute
        same = tot = 0
        for c in range(0, md.nx4 * md.ny4, 97):
            tot += 1
            same += (md.walk_edges(c), md.jump_edges(c), md.walkable(c)) == (
                self.md.walk_edges(c), self.md.jump_edges(c), self.md.walkable(c))
        self.assertEqual(same, tot)
        self.assertGreater(len(md.roofs), 50)
        self.assertEqual(md.climbs, [])

    def test_degraded_missing_bin_and_deadline(self):
        md = M.MapData.load(MAP_DIR, None, time.perf_counter() + 10.0, bin_path=os.path.join(ROOT, "nope.bin"))
        self.assertTrue(md.degraded)
        self.assertIn("missing", md.degraded_reason)
        md = M.MapData.load(MAP_DIR, None, time.perf_counter() - 1.0, bin_path=os.path.join(ROOT, "nope.bin"))
        self.assertTrue(md.degraded)
        self.assertIn("stopped", md.degraded_reason)
        # still answers every query without raising
        md.los((0, 0, 10), (50, 0, 10))
        md.walk_edges(md.cell4(0, 0))
        md.sight5(0, 0), md.wall5(0, 0), md.cover2(0, 0), md.cover2_cells(0, 0, 6), md.ground_under(0, 0, 0)
        md.clearance(0, 0)
        self.assertGreater(md.ground(0, 0), -50)

    def test_missing_map_dir_is_flat(self):
        md = M.MapData.load("/nonexistent_dir_xyz", None, time.perf_counter() + 5.0)
        self.assertTrue(md.degraded and md.empty)
        self.assertEqual(md.ground(10, 10), 0.0)
        self.assertTrue(md.los((0, 0, 8), (100, 0, 4.5)))

    # ---------------------------------------------------------------- terrain
    def test_ground_bilinear(self):
        md = self.md
        rnd = random.Random(1)
        for _ in range(200):
            i, j = rnd.randrange(md.nx - 1), rnd.randrange(md.ny - 1)
            c = j * md.nx + i
            x, y = md.tminx + i, md.tminy + j
            self.assertAlmostEqual(md.ground(x, y), md.terr[c] / 100.0, places=6)
            exp = (md.terr[c] + md.terr[c + 1] + md.terr[c + md.nx] + md.terr[c + md.nx + 1]) / 400.0
            self.assertAlmostEqual(md.ground(x + 0.5, y + 0.5), exp, places=6)
            ax, ay = rnd.random(), rnd.random()
            e2 = ((md.terr[c] * (1 - ax) + md.terr[c + 1] * ax) * (1 - ay)
                  + (md.terr[c + md.nx] * (1 - ax) + md.terr[c + md.nx + 1] * ax) * ay) / 100.0
            self.assertAlmostEqual(md.ground(x + ax, y + ay), e2, places=6)
        with open(os.path.join(MAP_DIR, "v2", "terrain.i16"), "rb") as f:
            raw = array.array("h")
            raw.frombytes(f.read())
        self.assertEqual(raw[12345] / 100.0, md.ground(md.tminx + 12345 % md.nx, md.tminy + 12345 // md.nx))

    # ---------------------------------------------------------------- LOS
    def _tall_roof_line(self):
        """Ground points on both sides of the tallest roof (west/east through one of its cells)."""
        md = self.md
        r = max((r for r in md.roof_info[1:] if r["kind"] == "building" and r["min_width"] >= 15), key=lambda r: r["h"])
        x0, y0, x1, y1 = r["bbox"]
        cy = None
        for k in range(1, 20):
            y = y0 + (y1 - y0) * k / 20
            for xx in range(int(x0), int(x1) + 1):
                if md.roof_id(xx, y) == r["id"]:
                    cy, cx = y, xx
                    break
            if cy is not None:
                break
        xa, xb = x0 - 10, x1 + 10
        return r, (xa, cy, md.ground(xa, cy) + 4.5), (xb, cy, md.ground(xb, cy) + 4.5), (cx, cy)

    def test_los_blocked_by_building(self):
        md = self.md
        r, p, q, c = self._tall_roof_line()
        self.assertGreater(md.surface(*c), max(p[2], q[2]))
        self.assertFalse(md.los(p, q))
        self.assertFalse(md.shot_clear(p, q))
        # well above the roof is clear
        hi = r["top_max"] + 30
        self.assertTrue(md.shot_clear((p[0], p[1], hi), (q[0], q[1], hi)))

    def test_open_field_clear(self):
        md = self.md
        # Ohio Stadium field (world.json stadium.center, fieldZ -9.23): open bowl floor
        g = md.ground(-276.3, 110)
        self.assertTrue(md.shot_clear((-276.3, 110, g + 8), (-276.3, 150, md.ground(-276.3, 150) + 4.5)))
        self.assertTrue(md.los((-276.3, 110, g + 8), (-276.3, 150, md.ground(-276.3, 150) + 4.5)))
        # the stands block a ground-level ray from the field to 200 m east
        self.assertFalse(md.shot_clear((-276.3, 128.7, -9.23 + 8), (-76.3, 128.7, md.ground(-76.3, 128.7) + 4.5)))
        # open-grid consistency: from cells with long sight, a mech 60 m along the best sector is visible
        hits = tot = 0
        for J in range(0, md.S5_NY, 3):
            for I in range(0, md.S5_NX, 3):
                x = md.gx0 + (I + 0.5) * 5
                y = md.gy0 + (J + 0.5) * 5
                if not md.in_bounds(x, y) or (md.sight(x, y) or 0) < 200 or md.wall_dist(x, y) < 40:
                    continue
                k = max(range(8), key=lambda s: md.sight_dir(x, y, s))
                a = math.radians(45 * k + 22.5)
                qx, qy = x + 60 * math.cos(a), y + 60 * math.sin(a)
                tot += 1
                hits += md.shot_clear((x, y, md.surface(x, y) + 8), (qx, qy, md.surface(qx, qy) + 4.5))
        self.assertGreater(tot, 20)
        self.assertGreater(hits / tot, 0.9)

    def test_under_bridge_shot_clear(self):
        md = self.md
        br = [o for o in md.overhangs if o["kind"] == "bridge" and o["base"] - md.ground(
            sum(o["ring"][0::2]) / (len(o["ring"]) // 2), sum(o["ring"][1::2]) / (len(o["ring"]) // 2)) > 9.3]
        self.assertTrue(br, "no walk-under bridge found")
        o = br[0]   # prism 4: deck 6.59-8.59 over terrain ~-4.8, long axis ~25 deg
        xs, ys = o["ring"][0::2], o["ring"][1::2]
        # long axis = direction between the two farthest ring vertices
        best = max(((a, b) for a in range(len(xs)) for b in range(len(xs))),
                   key=lambda ab: (xs[ab[0]] - xs[ab[1]]) ** 2 + (ys[ab[0]] - ys[ab[1]]) ** 2)
        ax = math.atan2(ys[best[1]] - ys[best[0]], xs[best[1]] - xs[best[0]])
        mx, my = (xs[best[0]] + xs[best[1]]) / 2, (ys[best[0]] + ys[best[1]]) / 2
        px, py = math.cos(ax + math.pi / 2), math.sin(ax + math.pi / 2)
        a = (mx - 20 * px, my - 20 * py)
        b = (mx + 20 * px, my + 20 * py)
        za, zb = md.ground(*a) + 3.0, md.ground(*b) + 3.0
        self.assertLess(max(za, zb), o["base"])
        self.assertTrue(md.shot_clear((a[0], a[1], za), (b[0], b[1], zb)), "shot under the bridge")
        zd = (o["base"] + o["top"]) / 2
        self.assertFalse(md.shot_clear((a[0], a[1], zd), (b[0], b[1], zd)), "shot through the deck")
        c = md.cell_of(mx, my)
        self.assertTrue(any(md.flags(cc) & M.NAV_F_UNDER for cc in [c] + [md.neighbor(c, k) for k in range(8)]))

    def test_trees_block_sight_not_shots(self):
        md = self.md
        found = 0
        for t in md.trees:
            cx, cy, cz, R, Rz, tx, ty, tr, zb, zt = t
            if R < 8 or Rz < 4:
                continue
            p, q = (cx - R - 3, cy, cz), (cx + R + 3, cy, cz)
            if not md.shot_clear(p, q):
                continue
            self.assertFalse(md.visible(p, q), "dense crown must block sight")
            self.assertFalse(md.los(p, q, canopy=True))
            self.assertTrue(md.los(p, q, canopy=False, trunks=False))
            found += 1
            if found >= 5:
                break
        self.assertGreater(found, 0)
        # trunk: a low ray through a trunk below the crown blocks sight even with canopy off
        for t in md.trees:
            cx, cy, cz, R, Rz, tx, ty, tr, zb, zt = t
            z = zb + 1.0
            if cz - Rz < z + 0.5:
                continue
            p, q = (tx - 6, ty, z), (tx + 6, ty, z)
            if md.shot_clear(p, q):
                self.assertFalse(md.los(p, q, canopy=False, trunks=True))
                break
        else:
            self.fail("no trunk test case")
        # a thin brush of foliage (short chord) does not block
        for t in md.trees:
            cx, cy, cz, R, Rz, tx, ty, tr, zb, zt = t
            if R < 4:
                continue
            off = math.sqrt(max(0.0, R * R - 1.0))   # chord 2 m < 3.96 m
            p, q = (cx - 30, cy + off, cz), (cx + 30, cy + off, cz)
            if md.shot_clear(p, q) and all(
                    math.hypot(o[0] - cx, o[1] - cy) > 30 + o[3] for o in md.trees if o is not t
                    if abs(o[1] - cy - off) < o[3] + 1):
                self.assertTrue(md.los(p, q, canopy=True, trunks=False))
                break

    def test_los_matches_bruteforce(self):
        md = self.md

        def pin(edges, x, y):
            ins = False
            for ex0, ey0, ex1, ey1 in edges:
                if (ey0 > y) != (ey1 > y) and x < ex0 + (y - ey0) * (ex1 - ex0) / (ey1 - ey0):
                    ins = not ins
            return ins

        def ref(p, q):
            dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            L = math.hypot(dx, dy)
            n = int(L / 0.05) + 1
            for s in range(n + 1):
                t = s / n
                x, y, z = p[0] + dx * t, p[1] + dy * t, p[2] + dz * t
                if M.LOS_END_SKIP < L * t < L - M.LOS_END_SKIP:
                    c = md._cell1(x, y)
                    if c is not None and z < md.surf[c] * 0.01:
                        return False
                for bx0, bx1, by0, by1, base, top, edges in md._ohp:
                    if base < z < top and bx0 <= x <= bx1 and by0 <= y <= by1 and pin(edges, x, y):
                        return False
            return True
        rnd = random.Random(7)
        bx0, by0, bx1, by1 = md.bounds
        for _ in range(80):
            x, y = rnd.uniform(bx0, bx1), rnd.uniform(by0, by1)
            a, r = rnd.uniform(0, 2 * math.pi), rnd.uniform(5, 200)
            x2, y2 = x + r * math.cos(a), y + r * math.sin(a)
            p = (x, y, md.surface(x, y) + rnd.choice([8, 20, 40]))
            q = (x2, y2, md.surface(x2, y2) + rnd.choice([1, 4.5, 9]))
            self.assertEqual(md.shot_clear(p, q), ref(p, q), (p, q))

    def test_los_benchmark(self):
        md = self.md
        rnd = random.Random(0)
        bx0, by0, bx1, by1 = md.bounds
        all_t, clear_t, shot_t = [], [], []
        while len(clear_t) < 1500:
            x, y = rnd.uniform(bx0, bx1), rnd.uniform(by0, by1)
            a = rnd.uniform(0, 2 * math.pi)
            x2, y2 = x + 200 * math.cos(a), y + 200 * math.sin(a)
            if not md.in_bounds(x2, y2):
                continue
            p = (x, y, md.surface(x, y) + 8)
            q = (x2, y2, md.surface(x2, y2) + 4.5)
            t0 = time.perf_counter()
            r = md.visible(p, q)
            dt = time.perf_counter() - t0
            all_t.append(dt)
            if r:
                clear_t.append(dt)
                t0 = time.perf_counter()
                md.los(p, q)
                shot_t.append(time.perf_counter() - t0)
        print("\n  visible() 200 m rays: all n=%d p50 %.1f us p99 %.1f us; clear-only n=%d p50 %.1f us "
              "p99 %.1f us max %.1f us" % (len(all_t), _pct(all_t, .5) * 1e6, _pct(all_t, .99) * 1e6, len(clear_t),
                                           _pct(clear_t, .5) * 1e6, _pct(clear_t, .99) * 1e6, max(clear_t) * 1e6))
        print("  los() (trees ignored) on the same clear rays: p50 %.1f us p99 %.1f us max %.1f us" % (
            _pct(shot_t, .5) * 1e6, _pct(shot_t, .99) * 1e6, max(shot_t) * 1e6))
        self.assertLess(_pct(clear_t, .99), 0.00015 * 3)   # 0.15 ms target, x3 slack for slow CI hosts

    # ---------------------------------------------------------------- grids / nav
    def test_nav_walkability(self):
        md = self.md
        c = md.cell_of(-276.3, 128.7)   # stadium field
        self.assertTrue(md.walkable(c))
        self.assertEqual(md.walk_edges(c), 0xFF)
        self.assertAlmostEqual(md.h(c), -9.23, delta=0.5)
        # never a walk edge straight up a wall; drops from roofs exist
        up = drops = 0
        for c in range(md.nx4 * md.ny4):
            w = md.walk_edges(c)
            for k in range(8):
                if w >> k & 1:
                    d = md.h(md.neighbor(c, k)) - md.h(c)
                    up = max(up, d)
                    drops += d < -5
        self.assertLess(up, 6 * M.STEP_UP)
        self.assertGreater(drops, 0)
        # map edge cells are not walkable
        self.assertFalse(md.walkable(md.cell4(md.bounds[0] + 1, 0)))
        # walkable cells only have edges to walkable cells
        for c in range(0, md.nx4 * md.ny4, 13):
            for k in range(8):
                if (md.walk_edges(c) | md.jump_edges(c)) >> k & 1:
                    self.assertTrue(md.walkable(md.neighbor(c, k)))
        # the tall building: its interior is a roof, ground next to it has no walk edge into it
        r, p, q, cc = self._tall_roof_line()
        rc = md.cell_of(*cc)
        self.assertTrue(md.flags(rc) & M.NAV_F_ROOF)
        self.assertGreaterEqual(md.roof_id(*cc), 1)

    def test_accessors(self):
        md = self.md
        for c in (0, 1234, md.nx4 * md.ny4 - 1):
            self.assertEqual(md.cell4(*md.cell4_xy(c)), c)
            self.assertEqual(md.cell_of(*md.center_of(c)), c)
        self.assertEqual(md.cell4(-10000, 0), -1)
        self.assertIsNone(md.cell_of(-10000, 0))
        self.assertEqual(md.neighbor4(0, 4), -1)
        self.assertEqual(md.roofs[0].id, 0)
        for rid in range(1, len(md.roofs)):
            self.assertEqual(md.roofs[rid].id, rid)
            self.assertEqual(md.roof_info[rid]["id"], rid)
        self.assertIs(md.roof(1), md.roofs[1])
        self.assertIsNone(md.roof(0))
        self.assertEqual(len(md.climbs), sum(len(v) for v in md.climb_info.values()))
        cf = md.climbs[0]
        self.assertGreaterEqual(md.roof_id(*cf.pos) if md.roof_id(*cf.pos) else cf.roof_id, 1)
        faces = [f for v in md.climb_info.values() for f in v]
        self.assertGreater(len(faces), 100)
        f = faces[0]
        self.assertGreater(md.roof_id(f[M.CF["rx"]], f[M.CF["ry"]]), 0)
        self.assertGreater(f[M.CF["h"]], M.STEP_UP)
        self.assertEqual(md.roof_id(f[M.CF["x"]], f[M.CF["y"]]) == md.roof_id(f[M.CF["rx"]], f[M.CF["ry"]]), False)
        self.assertTrue(0 <= md.sight(0, 0) <= 510)
        self.assertTrue(0 <= md.sight_dir(0, 0, 3) <= 510)
        self.assertTrue(0 <= md.wall_dist(0, 0) <= 127.5)
        self.assertTrue(0 <= md.clearance(0, 0) <= 25.5)
        self.assertIsInstance(md.cover_bits(0, 0), int)
        self.assertTrue(md.in_bounds(0, 0))
        cells = md.cover2_cells(0, 0, 7)
        ds = [math.hypot(cx, cy) for cx, cy, bits in cells]
        self.assertEqual(ds, sorted(ds))
        self.assertTrue(all(d <= 7 for d in ds) and len(cells) > 30)
        # cache and ground_under on the walk-under bridge (deck top 8.59 over terrain ~-4.8)
        a, b = (0, 0, 10), (60, 0, 10)
        md.new_tick()
        self.assertEqual(md.los_cached(("t", 1), a, b), md.los(a, b))
        self.assertIn(("t", 1), md._los_cache)
        md.new_tick()
        self.assertEqual(md._los_cache, {})
        self.assertAlmostEqual(md.ground_under(-116.2, -32.1, -4.8), md.terrain_h(-116.2, -32.1), places=3)
        self.assertAlmostEqual(md.ground_under(-116.2, -32.1, 8.0), 8.59, places=2)
        self.assertFalse(md.in_bounds(600, 0))
        # inside a tall building clearance is 0 at its edge cells and the surface is the roof
        r, p, q, cc = self._tall_roof_line()
        self.assertAlmostEqual(md.surface(*cc), md.surf[md._cell1(*cc)] / 100.0)


if __name__ == "__main__":
    unittest.main()
