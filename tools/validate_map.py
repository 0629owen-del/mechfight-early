#!/usr/bin/env python3
"""Precompute validation (game-plan-v2.md section 11.4). Stdlib only.

    python3 tools/validate_map.py [--map-dir ../mechfight-kit/arena] [--bin precomp/map.bin]

Checks / reports:
  1. Spot checks: 20 building footprints (and 200 random samples) against surface_h, with an
     independent point-in-polygon implementation (not mapdata.scan_spans).
  2. GP map facts, as measured (never forced):
       - three regions by median sight (GP: north/center grid 40-80 m, Oval + southeast 140-200 m,
         west stadium/fields/parking 200 m+);
       - share of open cells within 11 m / 20 m of a building wall (GP: 30% / 51%);
       - roofs 10-30 m tall and >= 20 m wide (GP: about 54).
  3. Nav sanity: walkable share, size of the largest strongly reachable set over walk + jump edges,
     reachability of 200 random walkable pairs (the navsim >= 95% path gate itself lives in
     tools/navsim.py).
Exit code 1 if a hard check fails (spot checks, header hash, nav connectivity < 90%).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mapdata as M  # noqa: E402


def pip(ring, holes, x, y):
    ins = False
    for poly in [ring] + list(holes or ()):
        n = len(poly) // 2
        for k in range(n):
            x0, y0 = poly[2 * k], poly[2 * k + 1]
            x1, y1 = poly[2 * ((k + 1) % n)], poly[2 * ((k + 1) % n) + 1]
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                ins = not ins
    return ins


def expected_surface(prisms, terr_cm, x, y):
    e = terr_cm
    for p in prisms:
        if p["kind"] == "bridge":
            continue
        if p["base"] * 100 <= terr_cm + M.OVERHANG_GAP * 100 and pip(p["ring"], p["holes"], x, y):
            e = max(e, int(round(p["top"] * 100)))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", default=os.path.join(os.path.dirname(ROOT), "mechfight-kit", "arena"))
    ap.add_argument("--bin", default=M.DEFAULT_BIN)
    args = ap.parse_args()
    fails = []
    md = M.MapData.load(args.map_dir, None, None, bin_path=args.bin)
    print("load: %.0f ms, degraded=%s %s, map.bin %d bytes" % (
        md.load_ms, md.degraded, md.degraded_reason, os.path.getsize(args.bin)))
    if md.degraded:
        fails.append("degraded load")
        print("FAIL")
        return 1
    world = json.load(open(os.path.join(args.map_dir, "v2", "world.json")))
    prisms = world["prisms"]
    nx = md.nx
    rnd = random.Random(11)

    # ---- 1. spot checks
    bad = 0
    checked = 0
    cands = [k for k, p in enumerate(prisms) if p["kind"] == "building"]
    rnd.shuffle(cands)
    done = 0
    for k in cands:
        if done >= 20:
            break
        p = prisms[k]
        xs, ys = p["ring"][0::2], p["ring"][1::2]
        pts = []
        for _ in range(200):
            x, y = rnd.uniform(min(xs), max(xs)), rnd.uniform(min(ys), max(ys))
            if pip(p["ring"], p["holes"], x, y):
                i, j = round(x - md.tminx), round(y - md.tminy)
                pts.append((md.tminx + i, md.tminy + j))
            if len(pts) >= 5:
                break
        pts = [q for q in pts if pip(p["ring"], p["holes"], q[0], q[1]) and md._cell1(q[0], q[1]) is not None]
        if not pts:
            continue
        done += 1
        for x, y in pts:
            c = md._cell1(x, y)
            e = expected_surface(prisms, md.terr[c], x, y)
            checked += 1
            if md.surf[c] != e:
                bad += 1
                print("  spot mismatch prism %d at (%.1f, %.1f): surface_h %d expected %d" % (k, x, y, md.surf[c], e))
    for _ in range(200):
        i, j = rnd.randrange(nx), rnd.randrange(md.ny)
        x, y = md.tminx + i, md.tminy + j
        c = j * nx + i
        checked += 1
        if md.surf[c] != expected_surface(prisms, md.terr[c], x, y):
            bad += 1
    print("1. spot checks: %d footprints, %d samples, %d mismatches" % (done, checked, bad))
    if bad:
        fails.append("surface_h spot checks")

    # ---- 2a. regions by sight
    open_cells = []   # (x, y, sight, wall)
    for J in range(md.S5_NY):
        for I in range(md.S5_NX):
            x = md.gx0 + (I + 0.5) * M.SIGHT_STEP
            y = md.gy0 + (J + 0.5) * M.SIGHT_STEP
            if not md.in_bounds(x, y):
                continue
            c = md._cell1(x, y)
            if md.surf[c] - md.terr[c] >= 50:
                continue
            open_cells.append((x, y, md.sight5(x, y), md.wall5(x, y)))

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")
    print("2a. median sight (m) on open 5 m cells, coarse 3 x 3 grid (rows north -> south, cols west -> east):")
    b = md._bd
    for rj in (2, 1, 0):
        row = []
        for ri in range(3):
            x0 = b["minX"] + (b["maxX"] - b["minX"]) * ri / 3
            x1 = b["minX"] + (b["maxX"] - b["minX"]) * (ri + 1) / 3
            y0 = b["minY"] + (b["maxY"] - b["minY"]) * rj / 3
            y1 = b["minY"] + (b["maxY"] - b["minY"]) * (rj + 1) / 3
            row.append(med([s for x, y, s, w in open_cells if x0 <= x < x1 and y0 <= y < y1]))
        print("     " + "  ".join("%6.0f" % v for v in row))
    bands = collections.Counter()
    for x, y, s, w in open_cells:
        bands["<=90" if s <= 90 else "90-140" if s < 140 else "140-200" if s <= 200 else ">200"] += 1
    n = len(open_cells)
    print("     sight bands over open cells: " + ", ".join(
        "%s m %.0f%%" % (k, 100.0 * bands[k] / n) for k in ("<=90", "90-140", "140-200", ">200")))
    # region proxies: west (stadium side, x < -200), and the rest split by building density
    west = [s for x, y, s, w in open_cells if x < -200]
    dense = [s for x, y, s, w in open_cells if x >= -200 and w <= 11]
    sparse = [s for x, y, s, w in open_cells if x >= -200 and w > 11]
    print("     region proxies: west x<-200: median %.0f m (n=%d); east, wall<=11 m: %.0f m (n=%d); "
          "east, wall>11 m: %.0f m (n=%d)" % (med(west), len(west), med(dense), len(dense), med(sparse), len(sparse)))

    # ---- 2b. wall distance shares
    w11 = sum(1 for c in open_cells if c[3] <= 11) / n
    w20 = sum(1 for c in open_cells if c[3] <= 20) / n
    print("2b. open cells with a wall within 11 m: %.1f%% (GP 30%%), within 20 m: %.1f%% (GP 51%%)" % (100 * w11, 100 * w20))
    c2 = [md.cover2(x, y) for x, y, s, w in open_cells]
    print("    (cover2 at the same cells: <=11 m %.1f%%, <=20 m %.1f%%; corner bit on %.1f%% of walkable 2 m cells)" % (
        100 * sum(1 for v in c2 if (v & 63) <= 11) / n, 100 * sum(1 for v in c2 if (v & 63) <= 20) / n,
        100 * sum(1 for v in md._c2 if v & M.COVER_CORNER and v & M.COVER_WALK)
        / max(1, sum(1 for v in md._c2 if v & M.COVER_WALK))))

    # ---- 2c. roofs
    roofs = md.roof_info[1:]
    q = [r for r in roofs if 10 <= r["h"] <= 30 and r["min_width"] >= 20]
    q_any = [r for r in roofs if 10 <= r["h"] <= 30]
    north = sum(1 for r in q if r["centroid"][1] > 0)
    center = sum(1 for r in q if -200 <= r["centroid"][0] <= 200)
    print("2c. roofs total %d (top-ground >= 5 m); 10-30 m tall: %d; 10-30 m and min width >= 20 m: %d (GP ~54); "
          "of those north (y>0): %d, central (|x|<=200): %d" % (len(roofs), len(q_any), len(q), north, center))
    nfaces = sum(len(v) for v in md.climb_info.values())
    neave = sum(1 for v in md.climb_info.values() for f in v if f[M.CF["eave"]])
    print("    climb faces %d (eave %d); roofs with >= 1 eave-free face: %d" % (
        nfaces, neave, sum(1 for v in md.climb_info.values() if any(not f[M.CF["eave"]] for f in v))))

    # ---- 3. nav
    NN = md.nx4 * md.ny4
    walk = [c for c in range(NN) if md.walkable(c)]
    inb = [c for c in range(NN) if md.in_bounds(*md.cell4_xy(c))]

    def reach(src):
        seen = {src}
        st = [src]
        while st:
            c = st.pop()
            bits = md.walk_edges(c) | md.jump_edges(c)
            for k in range(8):
                if bits >> k & 1:
                    c2 = md.neighbor4(c, k)
                    if c2 not in seen:
                        seen.add(c2)
                        st.append(c2)
        return seen
    # largest reachable set from 20 random ground starts
    best = set()
    for _ in range(20):
        s = reach(rnd.choice(walk))
        if len(s) > len(best):
            best = s
    ground_walk = [c for c in walk if not md.nav_flags(c) & M.NAV_F_ROOF]
    frac = sum(1 for c in ground_walk if c in best) / max(1, len(ground_walk))
    print("3. nav4: walkable %d / %d in-bounds cells (%.1f%%); largest reachable set %d cells, "
          "covers %.1f%% of walkable ground cells" % (len(walk), len(inb), 100 * len(walk) / len(inb), len(best), 100 * frac))
    ok = 0
    for _ in range(200):
        a, b2 = rnd.choice(ground_walk), rnd.choice(ground_walk)
        if a in best and b2 in best:
            ok += 1
    print("   200 random walkable ground pairs both in the main set: %d%% (walk + jump edges; climbs excluded)" % (ok // 2))
    edges = sum(bin(md.walk_edges(c)).count("1") for c in walk)
    jumps = sum(bin(md.jump_edges(c)).count("1") for c in walk)
    under = sum(1 for c in range(NN) if md.nav_flags(c) & M.NAV_F_UNDER)
    print("   walk edges %d, jump edges %d, underpass cells %d, roof cells %d" % (
        edges, jumps, under, sum(1 for c in walk if md.nav_flags(c) & M.NAV_F_ROOF)))
    if frac < 0.90:
        fails.append("nav connectivity")
    print("RESULT:", "OK" if not fails else "FAIL " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
