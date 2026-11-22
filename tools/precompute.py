#!/usr/bin/env python3
"""Offline map precompute -> precomp/map.bin (game-plan-v2.md section 11). Stdlib only.

    python3 tools/precompute.py [--map-dir ../mechfight-kit/arena] [--out precomp/map.bin] [--jobs N]

Container (11.1): line 1 = JSON header + "\\n" ({version, world_sha256, terrain_sha256, built,
blobs:[{name, dtype, nx, ny, min_x, min_y, step, offset, nbytes, raw_nbytes, zlib}]}); then the
zlib blobs back to back, offsets from the end of the header line. Grids are row-major
(j * nx + i), little-endian. For every grid blob, (min_x, min_y) is the LOWER-LEFT CORNER of
cell (0, 0), so cell = floor((x - min_x) / step) and its centre = min_x + (i + 0.5) * step.
All grids share min_x = terrain.minX - 0.5, min_y = terrain.minY - 0.5, so the 1 m cell i is
centred on terrain sample i.

Blobs (11.2) and the choices the plan leaves open
-------------------------------------------------
surface_h  int16 cm, 1 m. max(terrain sample, top of building/stadium prisms whose ring contains
           the sample (outside holes) and whose base <= terrain + 3 m AT THAT CELL). The 3 m test is
           per cell, not per prism, so a building on a slope is solid where it touches the ground
           and an overhang where it does not.
surface_max8  (extra) int16 cm, 8 m blocks (134 x 91): max surface_h per block, for the LOS
           coarse skip.
overhangs  JSON list: every bridge + every building/stadium prism with at least one overhang
           cell: {id (prism index), kind, base, top, ring, holes}. LOS tests them exactly.
clear_dm   uint8, 1 m, 0.1 m units, cap 25.5: Euclidean distance from the cell centre to the
           nearest non-walkable cell centre minus 0.5 m (~ distance to that cell's face).
           Non-walkable = outside world bounds, or a "wall top": a cell more than STEP_UP (1.2 m)
           above any of its 8 neighbours on the nav surface (building edges, steep steps; tan 50 deg
           x 1 m = 1.19 m so the slope rule and the ledge rule coincide at 1 m sampling). Roof
           cells near a roof edge therefore have low clearance too (edge = drop).
           The nav surface = surface_h plus bridges/overhangs whose underside is less than
           mech height 9 m + NAV_HEAD_PAD 0.3 m above the surface below (those are solid up to
           their top: decks you walk on, low overhangs you cannot pass under).
foot_id    uint16, 1 m: roof id (1-based index into roofs) or 0.
nav4       4 m, 268 x 181. PLANAR layout (not interleaved): int16 h[N] (cm) | uint8 flags[N] |
           uint8 walk[N] | uint8 jump[N] | uint8 under[N] (0.1 m). One layer = the lowest walkable
           level (the ground under high bridges, the deck of low bridges).
           h = median nav surface of the 16 sub-cells. flags: 1 walkable (>= 60% of sub-cells
           within 1.2 m of h, centre >= MAP_EDGE_MARGIN inside bounds), 2 roof (most of those
           sub-cells are on a roof), 4 deck (a bridge deck at or above this level), 8 underpass
           (head clearance under an overhang), 16 edge-of-map.
           Edge bit k (DIRS8 = E, NE, N, NW, W, SW, S, SE) from A to neighbour B (both walkable):
           sample the nav surface every <= 1 m along the centre line and along two lines offset
           +-2.0 m (mech body); WALK when every rise between consecutive samples <= STEP_UP on all
           three lines and no side sample is > 2.4 m above the centre sample (wall beside the
           path; this closes gaps narrower than ~4-5 m). Drops of any height are allowed (GP:
           falling is free). JUMP (only when not WALK) when at most one centre rise is in
           (STEP_UP, JUMP_H] and no side rise exceeds JUMP_H.
           Underpass clearance = min head clearance among the level's sub-cells.
climbs     JSON {fields: mapdata.CLIMB_FIELDS, roofs: [[roof_id, [[face values in field order], ...]], ...]}
           (columnar to keep the runtime footprint small). Faces sampled every <= 4 m along each simplified roof
           edge >= 2 m long: stand point 3.1 m out along the outward normal (mech radius + 0.6),
           landing point 2 m inside. Face: {ground_cell, roof_cell (nav4 indices), wx, wy (wall foot
           point on the roof edge = contracts.ClimbFace.pos), x, y (stand),
           rx, ry (landing), nx, ny (outward normal), h (landing surface - stand surface), clear
           (clearance at the stand point), eave}. eave = some cell within 2.8 m of the stand point
           has a bridge/overhang underside below landing height + 9 m (protocol 7: an eave over any
           part of you stops a jet). Faces with h <= STEP_UP are dropped (walkable).
sight5     uint8 (2 m units), 5 m, 214 x 145. Eye = surface at the cell centre sample + 8 m.
           32 rays (yaw (r + 0.5) * 11.25 deg), 2 m steps up to 300 m, stopping at world bounds.
           Along a ray, a target is visible at d when its centre (surface + 4.5 m) is above the
           running horizon of nearer samples; the ray's sight = the FARTHEST visible d (gaps
           allowed). Cell value = median over 32 rays. Buildings + terrain only (A-13: no trees,
           no bridges).
sight5_dir 8 bytes per cell (cell-major: (j*nx+i)*8 + k): sector k = yaw in [45k, 45k+45) deg,
           value = max of its 4 rays (best open line in that sector).
wall5      uint8 (0.5 m units, cap 127.5), 5 m: distance from the cell centre sample to the nearest
           wall cell (building/stadium column >= 3 m above terrain) minus 0.5 m.
cover2     uint8, 2 m, 535 x 362: bits 0-5 = min wall distance over the 4 sub-samples (m, floor,
           cap 63; 0 on roofs / inside buildings); bit 6 corner = walls within 4 m in two adjacent
           cardinal directions (N+E, E+S, S+W, W+N: a concave nook); bit 7 walkable = some
           sub-sample has clearance >= mech radius 2.5 m and lies >= MAP_EDGE_MARGIN inside bounds.
trees      (extra) float32 x 10 per tree: crown centre cx, cy, cz, crown semi-axes R, Rz, trunk x, y,
           radius, z_base, z_top (mapdata.world_meta), so runtime never parses world.json unless
           degraded. The header also carries world.json terrain/bounds/rules.
roofs      JSON list; roof = 4-connected region of building/stadium cells >= 0.5 m above terrain
           with steps <= 1.2 m between neighbours, >= 16 cells, median top - median terrain >= 5 m.
           {id, kind, prisms, top (median), top_min, top_max, ground (median terrain), h, area (m2),
           min_width (rotating calipers on the hull of the simplified ring), centroid, bbox,
           ring (simplified outer boundary, CCW, flat), edges [[x0, y0, x1, y1, nx, ny]] (outward
           normals), neighbors (roof ids within 15 m), sight (mean sight5 over its 5 m cells, m)}.
"""
from __future__ import annotations

import argparse
import array
import datetime
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mapdata as M  # noqa: E402

SIGHT_RAYS = 32
SIGHT_STEP_M = 2
SIGHT_MAX_M = 300
CLIMB_STAND_OUT = M.MECH_R + 0.6
CLIMB_LAND_IN = 2.0
CLIMB_SPACING = 4.0
EAVE_R = M.MECH_R + 0.3
NEIGHBOR_R = 15.0
RING_TOL = 1.0

_G = {}


def log(*a):
    print("[precompute %6.1fs]" % (time.monotonic() - T0), *a, flush=True)


edt = M.edt


# ------------------------------------------------------------------ workers
def _init_worker(shared):
    _G.update(shared)


def _nav_edges_row(J):
    g = _G
    NX = g["NX"]
    w = bytearray(NX)
    jb = bytearray(NX)
    for I in range(NX):
        a, b = M.nav4_edges_cell(J * NX + I, g["H"], g["F"], g["nav"], g["nx"], g["ny"], NX, g["NY"])
        w[I] = a
        jb[I] = b
    return J, bytes(w), bytes(jb)


def _sight_row(J):
    g = _G
    surf, nx, ny = g["surf"], g["nx"], g["ny"]
    SX = g["SX"]
    tminx, tminy, bounds = g["tminx"], g["tminy"], g["bounds"]
    hmax = g["hmax"]
    dirs = g["dirs"]
    bx0 = bounds["minX"] - tminx
    bx1 = bounds["maxX"] - tminx
    by0 = bounds["minY"] - tminy
    by1 = bounds["maxY"] - tminy
    med = bytearray(SX)
    sec = bytearray(SX * 8)
    steps = list(range(SIGHT_STEP_M, SIGHT_MAX_M + 1, SIGHT_STEP_M))
    for I in range(SX):
        ei, ej = I * M.SIGHT_STEP + 2, J * M.SIGHT_STEP + 2
        if ei >= nx or ej >= ny:
            continue
        ez = surf[ej * nx + ei] + 800
        reach = hmax + 450 - ez
        res = []
        for (cs, sn) in dirs:
            best = 0
            hor = -1e9
            for d in steps:
                x = ei + cs * d
                y = ej + sn * d
                if x < bx0 or x > bx1 or y < by0 or y > by1:
                    break
                h = surf[int(y + 0.5) * nx + int(x + 0.5)]
                if (h + 450 - ez) >= hor * d:
                    best = d
                s = (h - ez) / d
                if s > hor:
                    hor = s
                    if hor > 0 and reach < hor * (d + SIGHT_STEP_M):
                        break
            res.append(best)
        srt = sorted(res)
        m = (srt[15] + srt[16]) / 2
        med[I] = min(255, int(round(m / 2)))
        for k in range(8):
            sec[I * 8 + k] = min(255, int(round(max(res[4 * k:4 * k + 4]) / 2)))
    return J, bytes(med), bytes(sec)


# ------------------------------------------------------------------ roof geometry
def trace_ring(cells, nx):
    cs = set(cells)
    nxt = {}
    for c in cells:
        i, j = c % nx, c // nx
        if (c - nx) not in cs:
            nxt.setdefault((i, j), []).append((i + 1, j))
        if not (i < nx - 1 and (c + 1) in cs):
            nxt.setdefault((i + 1, j), []).append((i + 1, j + 1))
        if (c + nx) not in cs:
            nxt.setdefault((i + 1, j + 1), []).append((i, j + 1))
        if not (i > 0 and (c - 1) in cs):
            nxt.setdefault((i, j + 1), []).append((i, j))
    loops = []
    while nxt:
        start = next(iter(nxt))
        loop = [start]
        cur = start
        while True:
            lst = nxt.get(cur)
            if not lst:
                break
            n2 = lst.pop()
            if not lst:
                del nxt[cur]
            if n2 == start:
                break
            loop.append(n2)
            cur = n2
        if len(loop) >= 4:
            loops.append(loop)
    best, barea = None, -1
    for lp in loops:
        a = 0
        for k in range(len(lp)):
            x1, y1 = lp[k]
            x2, y2 = lp[(k + 1) % len(lp)]
            a += x1 * y2 - x2 * y1
        if a > barea:
            best, barea = lp, a
    return best


def _dp(pts, tol):
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    ln = math.hypot(dx, dy) or 1e-9
    dmax, idx = -1, 0
    for k in range(1, len(pts) - 1):
        x, y = pts[k]
        d = abs((x - x0) * dy - (y - y0) * dx) / ln
        if d > dmax:
            dmax, idx = d, k
    if dmax <= tol:
        return [pts[0], pts[-1]]
    a = _dp(pts[:idx + 1], tol)
    b = _dp(pts[idx:], tol)
    return a[:-1] + b


def simplify_ring(loop, tol):
    n = len(loop)
    far = max(range(n), key=lambda k: (loop[k][0] - loop[0][0]) ** 2 + (loop[k][1] - loop[0][1]) ** 2)
    a = _dp(loop[:far + 1], tol)
    b = _dp(loop[far:] + [loop[0]], tol)
    out = a[:-1] + b[:-1]
    return out if len(out) >= 3 else loop


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def min_width(pts):
    h = hull(pts)
    if len(h) < 3:
        return 0.0
    best = 1e18
    for k in range(len(h)):
        (x0, y0), (x1, y1) = h[k], h[(k + 1) % len(h)]
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        if ln == 0:
            continue
        w = max(abs((x - x0) * dy - (y - y0) * dx) / ln for x, y in h)
        best = min(best, w)
    return best


def seg_dist(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / l2))
    return math.hypot(px - x0 - t * dx, py - y0 - t * dy)


# ------------------------------------------------------------------ main
def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", default=os.path.join(os.path.dirname(ROOT), "mechfight-kit", "arena"))
    ap.add_argument("--out", default=os.path.join(ROOT, "precomp", "map.bin"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    wpath = os.path.join(args.map_dir, "v2", "world.json")
    tpath = os.path.join(args.map_dir, "v2", "terrain.i16")
    wbytes = open(wpath, "rb").read()
    tbytes = open(tpath, "rb").read()
    world = json.loads(wbytes)
    tm = world["terrain"]
    nx, ny = tm["nx"], tm["ny"]
    tminx, tminy = tm["minX"], tm["minY"]
    gx0, gy0 = tminx - 0.5, tminy - 0.5
    bounds = world["bounds"]
    N = nx * ny
    terr = array.array("h")
    terr.frombytes(tbytes)
    if sys.byteorder != "little":
        terr.byteswap()
    prisms = world["prisms"]
    kinds = [p["kind"] for p in prisms]
    log("map", nx, ny, "prisms", len(prisms))

    # ---- surfaces
    surf, nav, lab, under, over_ids = M.build_surfaces(terr, nx, ny, tminx, tminy, prisms)
    log("surfaces; overhang prisms", len(over_ids), "underpass cells", sum(1 for u in under if u))
    overhangs = [M._overhang_rec(k, prisms[k]) for k in over_ids]
    cmax, nbx, nby = M.coarse_max(surf, nx, ny)

    inb = bytearray(N)
    for j in range(ny):
        y = tminy + j
        if not (bounds["minY"] <= y <= bounds["maxY"]):
            continue
        for i in range(nx):
            x = tminx + i
            if bounds["minX"] <= x <= bounds["maxX"]:
                inb[j * nx + i] = 1

    # ---- clear_dm
    su = int(M.STEP_UP * 100)
    blocked = bytearray(N)
    for j in range(ny):
        r0 = j * nx
        for i in range(nx):
            c = r0 + i
            if not inb[c]:
                blocked[c] = 1
                continue
            h = nav[c] - su
            for di, dj in M.DIRS8:
                i2, j2 = i + di, j + dj
                if 0 <= i2 < nx and 0 <= j2 < ny and nav[j2 * nx + i2] < h:
                    blocked[c] = 1
                    break
    log("blocked cells", sum(blocked))
    dist = edt(blocked, nx, ny)
    clear = bytes(min(255, max(0, int((d - 0.5) * 10))) for d in dist)
    del dist
    log("clear_dm")

    # ---- roofs + foot_id
    comps = M.roof_components(nav, terr, lab, kinds, nx, ny)
    foot = array.array("H", [0]) * N
    roofs = []
    for rid, (cells, top, gnd) in enumerate(comps, 1):
        for c in cells:
            foot[c] = rid
        loop = trace_ring(cells, nx)
        ring = simplify_ring(loop, RING_TOL)
        wring = [(round(gx0 + a, 2), round(gy0 + b, 2)) for a, b in ring]
        edges = []
        for k in range(len(wring)):
            (x0, y0), (x1, y1) = wring[k], wring[(k + 1) % len(wring)]
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln == 0:
                continue
            edges.append([x0, y0, x1, y1, round((y1 - y0) / ln, 4), round(-(x1 - x0) / ln, 4)])
        sx = sum(c % nx for c in cells) / len(cells)
        sy = sum(c // nx for c in cells) / len(cells)
        tops = sorted(nav[c] for c in cells)
        pids = sorted({lab[c] for c in cells})
        ks = {kinds[p] for p in pids}
        xs = [p[0] for p in wring]
        ys = [p[1] for p in wring]
        roofs.append({
            "id": rid, "kind": "stadium" if ks == {"stadium"} else "building", "prisms": pids,
            "top": top / 100.0, "top_min": tops[0] / 100.0, "top_max": tops[-1] / 100.0,
            "ground": gnd / 100.0, "h": round((top - gnd) / 100.0, 2), "area": len(cells),
            "min_width": round(min_width(wring), 2),
            "centroid": [round(tminx + sx, 2), round(tminy + sy, 2)],
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "ring": [v for p in wring for v in p], "edges": edges, "neighbors": [], "sight": 0.0,
            "_cells": cells,
        })
    for a in roofs:
        ax0, ay0, ax1, ay1 = a["bbox"]
        for b in roofs:
            if b["id"] <= a["id"]:
                continue
            bx0, by0, bx1, by1 = b["bbox"]
            if bx0 > ax1 + NEIGHBOR_R or ax0 > bx1 + NEIGHBOR_R or by0 > ay1 + NEIGHBOR_R or ay0 > by1 + NEIGHBOR_R:
                continue
            dmin = 1e9
            for (p, q) in ((a, b), (b, a)):
                pr, qe = p["ring"], q["edges"]
                for k in range(0, len(pr), 2):
                    for e in qe:
                        dmin = min(dmin, seg_dist(pr[k], pr[k + 1], e[0], e[1], e[2], e[3]))
            if dmin <= NEIGHBOR_R:
                a["neighbors"].append(b["id"])
                b["neighbors"].append(a["id"])
    log("roofs", len(roofs))

    # ---- walls: wall5 + cover2
    wall = bytearray(N)
    wh = int(M.WALL_MIN_H * 100)
    for c in range(N):
        if lab[c] >= 0 and M.is_wall(nav, terr, lab, kinds, c):
            wall[c] = 1
    wdist = edt(wall, nx, ny)
    SX, SY = (nx + 4) // 5, (ny + 4) // 5
    wall5 = bytearray(SX * SY)
    for J in range(SY):
        for I in range(SX):
            i, j = min(nx - 1, 5 * I + 2), min(ny - 1, 5 * J + 2)
            wall5[J * SX + I] = min(255, max(0, int((wdist[j * nx + i] - 0.5) * 2)))
    CX, CY = (nx + 1) // 2, (ny + 1) // 2
    cover2 = bytearray(CX * CY)
    rmech = int(M.MECH_R * 10)
    card = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for J in range(CY):
        for I in range(CX):
            dmin = 1e9
            walk = False
            opensub = False
            for j in (2 * J, 2 * J + 1):
                for i in (2 * I, 2 * I + 1):
                    if i >= nx or j >= ny:
                        continue
                    c = j * nx + i
                    dmin = min(dmin, wdist[c] - 0.5)
                    if not wall[c]:
                        opensub = True
                    if clear[c] >= rmech and M.in_bounds_xy(bounds, tminx + i, tminy + j, M.MAP_EDGE_MARGIN):
                        walk = True
            v = min(63, max(0, int(dmin)))
            ci, cj = 2 * I + 0.5, 2 * J + 0.5
            hit = []
            for di, dj in card:
                h = False
                for d in (1, 2, 3, 4):
                    ii, jj = int(ci + di * d + 0.5), int(cj + dj * d + 0.5)
                    if 0 <= ii < nx and 0 <= jj < ny and wall[jj * nx + ii]:
                        h = True
                        break
                hit.append(h)
            if opensub and any(hit[k] and hit[(k + 1) % 4] for k in range(4)):
                v |= M.COVER_CORNER
            if walk:
                v |= M.COVER_WALK
            cover2[J * CX + I] = v
    del wdist
    log("wall5 / cover2")

    # ---- nav4
    H, F, U, NX, NY = M.nav4_cells(nav, under, lab, kinds, foot, nx, ny, gx0, gy0, bounds)
    shared = {"H": H, "F": bytes(F), "nav": nav, "nx": nx, "ny": ny, "NX": NX, "NY": NY}
    walkb = bytearray(NX * NY)
    jumpb = bytearray(NX * NY)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(shared,)) as pool:
        for J, w, jb in pool.imap_unordered(_nav_edges_row, range(NY), chunksize=4):
            walkb[J * NX:(J + 1) * NX] = w
            jumpb[J * NX:(J + 1) * NX] = jb
    log("nav4 edges; walkable", sum(1 for f in F if f & M.NAV_F_WALK), "of", NX * NY)

    # ---- sight5
    dirs = [(math.cos((r + 0.5) * 2 * math.pi / SIGHT_RAYS), math.sin((r + 0.5) * 2 * math.pi / SIGHT_RAYS))
            for r in range(SIGHT_RAYS)]
    shared = {"surf": surf, "nx": nx, "ny": ny, "SX": SX, "tminx": tminx, "tminy": tminy,
              "bounds": bounds, "hmax": max(surf), "dirs": dirs}
    sight5 = bytearray(SX * SY)
    sight5_dir = bytearray(SX * SY * 8)
    with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(shared,)) as pool:
        for J, med, sec in pool.imap_unordered(_sight_row, range(SY), chunksize=2):
            sight5[J * SX:(J + 1) * SX] = med
            sight5_dir[J * SX * 8:(J + 1) * SX * 8] = sec
    log("sight5")

    # ---- roof sight + climbs
    climbs = []
    for r in roofs:
        rid = r["id"]
        vals = []
        for J in range(SY):
            for I in range(SX):
                i, j = 5 * I + 2, 5 * J + 2
                if i < nx and j < ny and foot[j * nx + i] == rid:
                    vals.append(sight5[J * SX + I] * 2)
        if not vals:
            cx, cy = r["centroid"]
            I, J = int((cx - gx0) // 5), int((cy - gy0) // 5)
            vals = [sight5[J * SX + I] * 2]
        r["sight"] = round(sum(vals) / len(vals), 1)
        faces = []
        for x0, y0, x1, y1, enx, eny in r["edges"]:
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln < 2.0:
                continue
            n = max(1, int(round(ln / CLIMB_SPACING)))
            for s in range(n):
                t = (s + 0.5) / n
                px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                land = None
                for din in (CLIMB_LAND_IN, 1.0, 3.0):
                    rx, ry = px - enx * din, py - eny * din
                    c = int((ry - gy0) // 1) * nx + int((rx - gx0) // 1)
                    if 0 <= c < N and foot[c] == rid:
                        land = (rx, ry, c)
                        break
                if land is None:
                    continue
                sxp, syp = px + enx * CLIMB_STAND_OUT, py + eny * CLIMB_STAND_OUT
                si, sj = int((sxp - gx0) // 1), int((syp - gy0) // 1)
                if not (0 <= si < nx and 0 <= sj < ny):
                    continue
                cs = sj * nx + si
                if foot[cs] == rid or not M.in_bounds_xy(bounds, sxp, syp, M.MAP_EDGE_MARGIN):
                    continue
                hs = nav[cs]
                hl = nav[land[2]]
                hgt = (hl - hs) / 100.0
                if hgt <= M.STEP_UP:
                    continue
                eave = False
                rr = int(math.ceil(EAVE_R))
                for dj in range(-rr, rr + 1):
                    for di in range(-rr, rr + 1):
                        if di * di + dj * dj > EAVE_R * EAVE_R:
                            continue
                        i2, j2 = si + di, sj + dj
                        if 0 <= i2 < nx and 0 <= j2 < ny:
                            c2 = j2 * nx + i2
                            u = under[c2]
                            if u and nav[c2] + u * 10 < hl + M.HEAD_CM:
                                eave = True
                faces.append({
                    "ground_cell": int((syp - gy0) // 4) * NX + int((sxp - gx0) // 4),
                    "roof_cell": int((land[1] - gy0) // 4) * NX + int((land[0] - gx0) // 4),
                    "wx": round(px, 2), "wy": round(py, 2),
                    "x": round(sxp, 2), "y": round(syp, 2), "rx": round(land[0], 2), "ry": round(land[1], 2),
                    "nx": enx, "ny": eny, "h": round(hgt, 2), "clear": clear[cs] / 10.0, "eave": eave})
        climbs.append([rid, [[f[k] for k in M.CLIMB_FIELDS] for f in faces]])
        del r["_cells"]
    log("climbs", sum(len(c[1]) for c in climbs))

    # ---- pack
    def g(name, dtype, gnx, gny, step, raw, **extra):
        meta = {"name": name, "dtype": dtype, "nx": gnx, "ny": gny, "min_x": gx0, "min_y": gy0, "step": step}
        meta.update(extra)
        return (meta, raw)

    def le(a):
        if sys.byteorder != "little":
            a = array.array(a.typecode, a)
            a.byteswap()
        return a.tobytes()

    def js(name, obj):
        return ({"name": name, "dtype": "json"}, json.dumps(obj, separators=(",", ":")).encode())

    nav4raw = le(H) + bytes(F) + bytes(walkb) + bytes(jumpb) + bytes(U)
    blobs = [
        g("surface_h", "int16", nx, ny, 1, le(surf), units="cm"),
        g("surface_max8", "int16", nbx, nby, M.COARSE, le(array.array("h", cmax)), units="cm"),
        js("overhangs", overhangs),
        g("clear_dm", "uint8", nx, ny, 1, clear, units="0.1m"),
        g("foot_id", "uint16", nx, ny, 1, le(foot)),
        g("nav4", "struct_planar", NX, NY, M.NAV_STEP_I, nav4raw,
          fields=[["h", "int16", "cm"], ["flags", "uint8"], ["walk", "uint8"], ["jump", "uint8"],
                  ["under", "uint8", "0.1m"]],
          dirs="E,NE,N,NW,W,SW,S,SE", flag_bits={"walkable": 1, "roof": 2, "deck": 4, "underpass": 8, "edge": 16}),
        js("climbs", {"fields": list(M.CLIMB_FIELDS), "roofs": climbs}),
        g("sight5", "uint8", SX, SY, M.SIGHT_STEP, bytes(sight5), units="2m"),
        g("sight5_dir", "uint8x8", SX, SY, M.SIGHT_STEP, bytes(sight5_dir), units="2m",
          sectors="k: yaw in [45k, 45k+45) deg"),
        g("wall5", "uint8", SX, SY, M.SIGHT_STEP, bytes(wall5), units="0.5m"),
        g("cover2", "uint8", CX, CY, M.COVER_STEP, bytes(cover2),
          bits="0-5 wall dist m (cap 63), 6 corner, 7 walkable"),
        js("roofs", roofs),
    ]
    meta = M.world_meta(world)
    tflat = array.array("f", [v for t in meta["trees"] for v in t])
    blobs.append(({"name": "trees", "dtype": "float32x10",
                   "fields": "cx,cy,cz,R,Rz,trunk_x,trunk_y,trunk_r,z_base,z_top"}, le(tflat)))
    hdr = {"version": M.VERSION, "terrain": meta["terrain"], "bounds": meta["bounds"], "rules": meta["rules"], "world_sha256": M.sha256_bytes(wbytes), "terrain_sha256": M.sha256_bytes(tbytes),
           "built": datetime.date.today().isoformat(),
           "params": {"step_up": M.STEP_UP, "jump_h": M.JUMP_H, "head_cm": M.HEAD_CM, "overhang_gap": M.OVERHANG_GAP,
                      "mech_r": M.MECH_R, "side_off": M.NAV_SIDE_OFF, "wall_min_h": M.WALL_MIN_H}}
    data = M.pack_container(hdr, blobs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, args.out)
    log("wrote", args.out, len(data), "bytes")
    hdr2 = json.loads(data[:data.index(b"\n")])
    for b in hdr2["blobs"]:
        log("  %-13s raw %8d  zlib %8d" % (b["name"], b["raw_nbytes"], b["nbytes"]))


T0 = time.monotonic()

if __name__ == "__main__":
    main()
