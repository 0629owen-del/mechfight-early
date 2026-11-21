"""Static map model: loads precomp/map.bin, verifies the map hashes, answers geometry queries.

Plan refs: game-plan-v2.md section 11 (precompute, degraded mode), 4.6 (LOS), 8.4/8.9 (inputs).
The binding API is CONTRACTS.md section 6.1; everything listed there is implemented as written.

    md = MapData.load(hello["map"]["dir"], C, deadline)   # deadline: absolute time.perf_counter()
    md.degraded / md.degraded_reason                       # 11.3; never raises
    md.los(a, b)            # straight line: terrain + buildings + stadium + bridges, trees ignored
    md.visible(eye, tgt)    # protocol 8 vision line: + trunks + canopy (near=True: trunks only)

CONTRACTS 6.1 API (coordinates in world m; Vec2 = (x, y), Vec3 = (x, y, z))
    constants NAV_STEP (4.0), NAV_DIRS, NAV_WALKABLE/ROOF/DECK/UNDERPASS/EDGE, COVER_WALL_MASK/
              COVER_CORNER/COVER_WALKABLE
    attrs     degraded, degraded_reason, bounds (min_x, min_y, max_x, max_y), roofs (list[Roof],
              roofs[0] placeholder with h 0, roofs[id].id == id), climbs (list[ClimbFace]),
              nx4, ny4, min_x4, min_y4 (lower-left corner of nav cell 0)
    load(map_dir, C, deadline)   new_tick()   terrain_h(x, y)   surface_h(x, y)
    ground_under(x, y, z)        los(a, b)    los_cached(key, a, b)
    clearance(x, y)  roof_id(x, y)
    cell4(x, y) (-1 outside)  cell4_xy(idx)  nav_h(idx)  nav_flags(idx)  walk_edges(idx)
    jump_edges(idx)  underpass_clear(idx)
    sight5(x, y)  sight5_dir(x, y, sector)   (NaN when unknown = degraded)
    wall5(x, y)   cover2(x, y)   cover2_cells(x, y, r)   (degraded: coarse 4 m values)
Extras
    los(a, b, *, canopy=False, trunks=False)  shot_clear(a, b) (= los(a, b))  visible(eye, tgt, near=False)
    in_bounds(x, y)  walkable(idx)  neighbor4(idx, k) (-1 outside)  roof(rid) -> Roof | None
    roof_info[id]: rest of the precompute roof record (kind, prisms, top_min, top_max, bbox; ring and
        edges are in roofs[id])
    climb_info[roof_id]: face tuples in CLIMB_FIELDS order (index with CF["name"]): ground_cell,
        roof_cell, wx, wy (wall foot), x, y (stand point), rx, ry (landing), nx, ny (outward normal),
        h, clear (m at the stand point), eave
    overhangs: bridges + overhanging prisms {id, kind, base, top, ring, holes}
    trees: (cx, cy, cz, R, Rz, trunk_x, trunk_y, trunk_r, z_base, z_top)
    has_sight / has_cover (False in degraded), empty (no terrain), params_match (hello step_up/jump_h
    equal the values map.bin was built with), load_ms
    Task-brief aliases: ground = terrain_h, surface = surface_h, cell_of (None outside),
    center_of = cell4_xy, h = nav_h, flags = nav_flags, sight = sight5, sight_dir = sight5_dir,
    wall_dist = wall5, cover_bits = cover2, neighbor (None outside).

Grids: every grid's (min_x, min_y) is the lower-left corner of its cell 0 = terrain.minX/minY - 0.5,
so the 1 m cell i is centred on terrain sample i; cell = floor((x - min_x) / step).

Assumptions (see also tools/precompute.py header)
- los() ignores trees by default as CONTRACTS 6.1 and plan 4.6 / A-13 say, so callers that use it
  for shots (laser-at-us, lightning gate, cover) stay correct. Vision callers that want the
  protocol 8 rule use visible() (or los(..., canopy=True, trunks=True)).
- Canopy (protocol 8 is qualitative): a crown is an ellipsoid centred at (crownX, crownY,
  z + (crownBase + height) / 2), horizontal semi-axis crownR, vertical (height - crownBase) / 2
  (tree height / crownBase are relative to the tree's ground z: crownBase >= 2.0 always and
  height - z can be 0.03, so they cannot be absolute). Transmission = exp(-canopyDensity * total
  chord length through crowns); blocked when < canopyBlockT (0.35 / 0.25 -> 3.96 m of crown).
- Trunks: vertical cylinders (trunkR) from z to z + height at the tree's (x, y).
- LOS terrain is the 1 m sample as a flat column (error <= slope * 0.5 m); the first and last
  LOS_END_SKIP (0.75) m of a segment skip the grid test (an object on the ground is not hidden by
  its own cell). Buildings with base <= terrain + 3 m at a cell are solid from the ground there;
  other prism parts and all bridges are tested exactly.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import os
import time
import zlib

VERSION = 1

# ---- geometry constants (protocol section 5/7, plan section 10 defaults) ----
STEP_UP = 1.2            # mech.step_up (server); world.json rules.maxStep 1.5 is the viewer's
MAX_SLOPE_DEG = 50.0     # mech.max_slope_deg (tan 50 deg * 1 m = 1.19 m, ~= STEP_UP per 1 m sample)
JUMP_H = 2.75            # jump_speed^2 / (2 g)
MECH_R = 2.5
MECH_H = 9.0
EYE_H = 8.0
CENTER_H = 4.5
NAV_HEAD_PAD = 0.3
HEAD_CM = int(round((MECH_H + NAV_HEAD_PAD) * 100))
MAP_EDGE_MARGIN = 3.0
OVERHANG_GAP = 3.0       # building base > terrain + 3 m at a cell -> not in surface_h (overhang)
NAV_SIDE_OFF = 2.0       # nav4 edges are also checked on two lines offset +-2 m (mech body)
NAV_SIDE_WALL = 2.4      # a side sample this much above the centre sample = wall beside the path
WALL_MIN_H = 3.0         # building column >= 3 m above terrain counts as "wall" for wall5/cover2
ROOF_MIN_REL = 5.0       # roofs: top - ground >= 5 m
ROOF_MIN_CELLS = 16
LOS_END_SKIP = 0.75
COARSE = 8               # LOS coarse block (cells)
CANOPY_DENSITY = 0.35
CANOPY_BLOCK_T = 0.25

NAV_STEP_I, COVER_STEP, SIGHT_STEP = 4, 2, 5
NAV_STEP = 4.0                      # CONTRACTS 6.1 (m per nav4 cell)
DIRS8 = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
NAV_DIRS = DIRS8                    # edge-bit order E, NE, N, NW, W, SW, S, SE
NAV_F_WALK, NAV_F_ROOF, NAV_F_DECK, NAV_F_UNDER, NAV_F_EDGE = 1, 2, 4, 8, 16
NAV_WALKABLE, NAV_ROOF, NAV_DECK, NAV_UNDERPASS, NAV_EDGE = 1, 2, 4, 8, 16
COVER_WALL_MASK, COVER_CORNER, COVER_WALKABLE = 0x3F, 0x40, 0x80
COVER_WALK = COVER_WALKABLE
# climb face tuple layout (climb_info[roof_id] -> list of tuples in this order)
CLIMB_FIELDS = ("ground_cell", "roof_cell", "wx", "wy", "x", "y", "rx", "ry", "nx", "ny", "h", "clear", "eave")
CF = {name: k for k, name in enumerate(CLIMB_FIELDS)}

# default terrain grid (used only if world.json is unreadable)
_DEF_TERRAIN = {"minX": -513.0, "minY": -334.0, "step": 1.0, "nx": 1070, "ny": 724}
_DEF_BOUNDS = {"minX": -512.38, "maxX": 555.08, "minY": -333.09, "maxY": 388.64}

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(_HERE, "precomp", "map.bin")


class _Timeout(Exception):
    pass


# =====================================================================================
# Shared builders (used by tools/precompute.py and by degraded mode)
# =====================================================================================

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def scan_spans(ring, holes, minx, miny, nx, ny):
    """Yield (j, i0, i1): half-open runs of grid samples (minx + i, miny + j) inside ring minus holes.

    Even-odd rule over ring + hole edges; a sample on a left edge is inside, on a right edge outside.
    """
    edges = []
    ylo, yhi = 1e18, -1e18
    for poly in [ring] + list(holes or ()):
        n = len(poly) // 2
        for k in range(n):
            x1, y1 = poly[2 * k], poly[2 * k + 1]
            k2 = (k + 1) % n
            x2, y2 = poly[2 * k2], poly[2 * k2 + 1]
            if y1 < ylo:
                ylo = y1
            if y1 > yhi:
                yhi = y1
            if y1 != y2:
                edges.append((x1, y1, x2, y2))
    if not edges:
        return
    j0 = max(0, math.ceil(ylo - miny))
    j1 = min(ny - 1, math.floor(yhi - miny))
    for j in range(j0, j1 + 1):
        y = miny + j
        xs = [x1 + (y - y1) * (x2 - x1) / (y2 - y1)
              for (x1, y1, x2, y2) in edges if (y1 <= y < y2) or (y2 <= y < y1)]
        if len(xs) < 2:
            continue
        xs.sort()
        for a in range(0, len(xs) - 1, 2):
            i0 = max(0, math.ceil(xs[a] - minx))
            i1 = min(nx, math.ceil(xs[a + 1] - minx))
            if i1 > i0:
                yield j, i0, i1


def build_surfaces(terr, nx, ny, minx, miny, prisms, check=None):
    """Rasterize prisms on the 1 m grid.

    Returns (surf, nav, lab, under, over_ids):
      surf  int16 cm: max(terrain, top of building/stadium prisms with base <= terrain + 3 m)  (LOS)
      nav   int16 cm: surf, plus overhang/bridge columns too low to walk under (solid up to top)
      lab   int16: prism index giving the nav top (-1 = terrain)
      under uint8: head clearance (0.1 m) under a bridge/overhang that can be walked under, 0 = none
      over_ids: sorted prism indices that need the exact LOS test (bridges + overhang cells)
    """
    N = nx * ny
    surf = array.array("h", terr)
    lab = array.array("h", [-1]) * N
    over_cells = []
    for k, p in enumerate(prisms):
        if check:
            check()
        base = int(round(p["base"] * 100))
        top = int(round(p["top"] * 100))
        bridge = p.get("kind") == "bridge"
        gap = int(OVERHANG_GAP * 100)
        for j, i0, i1 in scan_spans(p["ring"], p.get("holes"), minx, miny, nx, ny):
            row = j * nx
            for c in range(row + i0, row + i1):
                if not bridge and base <= terr[c] + gap:
                    if top > surf[c]:
                        surf[c] = top
                        lab[c] = k
                else:
                    over_cells.append((base, top, k, c))
    nav = array.array("h", surf)
    under = bytearray(N)
    over_cells.sort()
    over_ids = set()
    for base, top, k, c in over_cells:
        over_ids.add(k)
        if base - nav[c] < HEAD_CM:
            if top > nav[c]:
                nav[c] = top
                lab[c] = k
        else:
            cl = min(255, (base - nav[c]) // 10)
            if under[c] == 0 or cl < under[c]:
                under[c] = cl
    for k, p in enumerate(prisms):
        if p.get("kind") == "bridge":
            over_ids.add(k)
    return surf, nav, lab, under, sorted(over_ids)


def coarse_max(surf, nx, ny, b=COARSE, check=None):
    """Max of surf over b x b blocks (list of ints, cm)."""
    nbx, nby = (nx + b - 1) // b, (ny + b - 1) // b
    out = [-32768] * (nbx * nby)
    for j in range(ny):
        if check and (j & 63) == 0:
            check()
        row = surf[j * nx:(j + 1) * nx]
        J = j // b
        base = J * nbx
        for I in range(nbx):
            m = max(row[I * b:(I + 1) * b])
            if m > out[base + I]:
                out[base + I] = m
    return out, nbx, nby


def in_bounds_xy(bounds, x, y, margin=0.0):
    return (bounds["minX"] + margin <= x <= bounds["maxX"] - margin
            and bounds["minY"] + margin <= y <= bounds["maxY"] - margin)


def nav4_cells(nav, under, lab, kinds, foot, nx, ny, gx0, gy0, bounds):
    """Per 4 m cell: h (cm), flags, underpass clearance. Edges are separate (nav4_edges_cell)."""
    NX, NY = (nx + NAV_STEP_I - 1) // NAV_STEP_I, (ny + NAV_STEP_I - 1) // NAV_STEP_I
    H = array.array("h", [0]) * (NX * NY)
    F = bytearray(NX * NY)
    U = bytearray(NX * NY)
    for J in range(NY):
        for I in range(NX):
            vals = []
            for j in range(J * NAV_STEP_I, min(ny, J * NAV_STEP_I + NAV_STEP_I)):
                for i in range(I * NAV_STEP_I, min(nx, I * NAV_STEP_I + NAV_STEP_I)):
                    vals.append((nav[j * nx + i], j * nx + i))
            vals.sort()
            h = vals[len(vals) // 2][0]
            lvl = [c for v, c in vals if abs(v - h) <= int(STEP_UP * 100)]
            c4 = J * NX + I
            H[c4] = h
            cx = gx0 + (I + 0.5) * NAV_STEP_I
            cy = gy0 + (J + 0.5) * NAV_STEP_I
            fl = 0
            if not in_bounds_xy(bounds, cx, cy, MAP_EDGE_MARGIN):
                fl |= NAV_F_EDGE
            elif len(lvl) * 10 >= len(vals) * 6:
                fl |= NAV_F_WALK
            nroof = 0
            ucl = 0
            deck = False
            for c in lvl:
                if foot is not None and foot[c]:
                    nroof += 1
                if lab[c] >= 0 and kinds[lab[c]] == "bridge":
                    deck = True
                u = under[c]
                if u:
                    deck = True
                    if ucl == 0 or u < ucl:
                        ucl = u
            if nroof * 2 > len(lvl):
                fl |= NAV_F_ROOF
            if deck:
                fl |= NAV_F_DECK
            if ucl:
                fl |= NAV_F_UNDER
            U[c4] = ucl
            F[c4] = fl
    return H, F, U, NX, NY


def nav4_edges_cell(c4, H, F, nav, nx, ny, NX, NY):
    """(walk_bits, jump_bits) for one 4 m cell (see precompute header for the rule)."""
    if not F[c4] & NAV_F_WALK:
        return 0, 0
    I, J = c4 % NX, c4 // NX
    ax = (I + 0.5) * NAV_STEP_I - 0.5      # in 1 m sample-index units (sample i at index i)
    ay = (J + 0.5) * NAV_STEP_I - 0.5
    ha = H[c4]
    su = int(STEP_UP * 100)
    jh = int(JUMP_H * 100)
    sw = int(NAV_SIDE_WALL * 100)
    walk = jump = 0
    for k, (di, dj) in enumerate(DIRS8):
        I2, J2 = I + di, J + dj
        if not (0 <= I2 < NX and 0 <= J2 < NY):
            continue
        c2 = J2 * NX + I2
        if not F[c2] & NAV_F_WALK:
            continue
        hb = H[c2]
        dx, dy = di * NAV_STEP_I, dj * NAV_STEP_I
        ln = math.hypot(dx, dy)
        n = int(math.ceil(ln))
        px, py = -dy / ln * NAV_SIDE_OFF, dx / ln * NAV_SIDE_OFF
        ok_walk = True
        ok_jump = True
        big = 0
        prev = ha
        prevs = [None, None]
        for s in range(0, n + 1):
            t = s / n
            x = ax + dx * t
            y = ay + dy * t
            if 0 < s < n:
                ii, jj = int(x + 0.5), int(y + 0.5)
                cur = nav[jj * nx + ii]
            elif s == 0:
                cur = ha
            else:
                cur = hb
            if s:
                r = cur - prev
                if r > su:
                    ok_walk = False
                    big += 1
                    if r > jh or big > 1:
                        ok_jump = False
                        break
            prev = cur
            for side in (0, 1):
                sx = x + (px if side else -px)
                sy = y + (py if side else -py)
                ii, jj = int(sx + 0.5), int(sy + 0.5)
                if not (0 <= ii < nx and 0 <= jj < ny):
                    ok_walk = ok_jump = False
                    break
                sv = nav[jj * nx + ii]
                if sv - cur > sw:
                    ok_walk = False
                    if sv - cur > jh + su:
                        ok_jump = False
                pv = prevs[side]
                if pv is not None:
                    r = sv - pv
                    if r > su:
                        ok_walk = False
                        if r > jh:
                            ok_jump = False
                prevs[side] = sv
            if not ok_jump:
                break
        if ok_walk:
            walk |= 1 << k
        elif ok_jump:
            jump |= 1 << k
    return walk, jump


_BIG = 1e12


def _edt1d(f, n):
    """Felzenszwalb 1-D squared distance transform of f (sites 0, others _BIG)."""
    v = [0] * n
    z = [0.0] * (n + 1)
    d = [0.0] * n
    k = 0
    z[0] = -_BIG
    z[1] = _BIG
    for q in range(1, n):
        fq = f[q] + q * q
        vk = v[k]
        s = (fq - (f[vk] + vk * vk)) / (2 * q - 2 * vk)
        while s <= z[k]:
            k -= 1
            vk = v[k]
            s = (fq - (f[vk] + vk * vk)) / (2 * q - 2 * vk)
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = _BIG
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        vk = v[k]
        d[q] = (q - vk) * (q - vk) + f[vk]
    return d


def edt(sites, nx, ny, check=None):
    """Euclidean distance (cells) from every cell centre to the nearest site centre (1e6 if none)."""
    rows = []
    for j in range(ny):
        if check and (j & 63) == 0:
            check()
        rows.append(_edt1d([0.0 if sites[j * nx + i] else _BIG for i in range(nx)], nx))
    out = [0.0] * (nx * ny)
    for i in range(nx):
        if check and (i & 63) == 0:
            check()
        col = _edt1d([rows[j][i] for j in range(ny)], ny)
        for j in range(ny):
            out[j * nx + i] = math.sqrt(col[j]) if col[j] < _BIG / 2 else 1e6
    return out


def is_wall(nav, terr, lab, kinds, c):
    """Wall cell for wall5/cover2: building/stadium column >= WALL_MIN_H above terrain."""
    k = lab[c]
    return k >= 0 and kinds[k] != "bridge" and nav[c] - terr[c] >= int(WALL_MIN_H * 100)


def wall_dist4(nav, terr, lab, kinds, nx, ny, NX, NY, check=None):
    """Degraded-mode wall distance on the 4 m grid (m, from cell centre to the nearest 4 m cell
    holding any wall cell, minus 2 m)."""
    m4 = bytearray(NX * NY)
    for c in range(nx * ny):
        if lab[c] >= 0 and is_wall(nav, terr, lab, kinds, c):
            m4[(c // nx // NAV_STEP_I) * NX + (c % nx) // NAV_STEP_I] = 1
        if check and (c & 0xFFFF) == 0:
            check()
    return [max(0.0, d * NAV_STEP_I - 2.0) for d in edt(m4, NX, NY, check)]


def roof_components(nav, terr, lab, kinds, nx, ny, check=None):
    """Connected roof regions: cells over building/stadium prisms, >= 0.5 m above terrain,
    4-connected with steps <= STEP_UP. Returns list of (cells, top_cm_median, ground_cm_median)
    for regions with top - ground >= ROOF_MIN_REL and >= ROOF_MIN_CELLS cells."""
    N = nx * ny
    seen = bytearray(N)
    su = int(STEP_UP * 100)
    out = []
    for c0 in range(N):
        if seen[c0]:
            continue
        k = lab[c0]
        if k < 0 or kinds[k] == "bridge" or nav[c0] - terr[c0] < 50:
            continue
        if check and (c0 & 0xFFF) == 0:
            check()
        seen[c0] = 1
        stack = [c0]
        cells = []
        while stack:
            c = stack.pop()
            cells.append(c)
            hc = nav[c]
            i = c % nx
            for c2 in (c - 1 if i > 0 else -1, c + 1 if i < nx - 1 else -1, c - nx, c + nx):
                if c2 < 0 or c2 >= N or seen[c2]:
                    continue
                k2 = lab[c2]
                if k2 < 0 or kinds[k2] == "bridge" or nav[c2] - terr[c2] < 50:
                    continue
                if abs(nav[c2] - hc) > su:
                    continue
                seen[c2] = 1
                stack.append(c2)
        if len(cells) < ROOF_MIN_CELLS:
            continue
        tops = sorted(nav[c] for c in cells)
        grounds = sorted(terr[c] for c in cells)
        top, gnd = tops[len(tops) // 2], grounds[len(grounds) // 2]
        if top - gnd >= ROOF_MIN_REL * 100:
            out.append((cells, top, gnd))
    return out


def pack_container(header_extra, blobs):
    """blobs: list of (meta dict without offset/nbytes, raw bytes). Returns file bytes."""
    metas, chunks, off = [], [], 0
    for meta, raw in blobs:
        z = zlib.compress(raw, 9)
        m = dict(meta)
        m.update({"offset": off, "nbytes": len(z), "raw_nbytes": len(raw), "zlib": True})
        metas.append(m)
        chunks.append(z)
        off += len(z)
    hdr = dict(header_extra)
    hdr["blobs"] = metas
    return (json.dumps(hdr, separators=(",", ":")) + "\n").encode() + b"".join(chunks)


def world_meta(world):
    """terrain/bounds/rules/prisms + tree tuples from a parsed world.json (defaults if unusable).

    Tree tuple: (cx, cy, cz, R, Rz, tx, ty, trunk_r, z_base, z_top) - see module docstring."""
    out = {"terrain": dict(_DEF_TERRAIN), "bounds": dict(_DEF_BOUNDS), "rules": {}, "prisms": [], "trees": []}
    if not world:
        return out
    try:
        out["terrain"].update({k: world["terrain"][k] for k in ("minX", "minY", "step", "nx", "ny")})
        out["bounds"] = dict(world["bounds"])
        out["prisms"] = world.get("prisms") or []
        out["rules"] = world.get("rules") or {}
    except (KeyError, TypeError):
        pass
    try:
        tr = world["trees"]
        f = tr["fields"]
        d = tr["data"]
        n = len(f)
        ix = {name: f.index(name) for name in ("x", "y", "z", "height", "crownR", "crownBase", "trunkR")}
        icx = f.index("crownX") if "crownX" in f else ix["x"]
        icy = f.index("crownY") if "crownY" in f else ix["y"]
        trees = []
        for o in range(0, len(d) - n + 1, n):
            x, y, z = d[o + ix["x"]], d[o + ix["y"]], d[o + ix["z"]]
            hgt, cr, cb = d[o + ix["height"]], d[o + ix["crownR"]], d[o + ix["crownBase"]]
            trees.append((d[o + icx], d[o + icy], z + (cb + hgt) * 0.5, max(0.25, cr),
                          max(0.25, (hgt - cb) * 0.5), x, y, d[o + ix["trunkR"]], z, z + hgt))
        out["trees"] = trees
    except (KeyError, ValueError, TypeError, IndexError):
        out["trees"] = []
    return out


# =====================================================================================
# Runtime
# =====================================================================================

try:  # shared plain-data types (CONTRACTS.md section 4); fallback keeps mapdata standalone
    from contracts import ClimbFace, Roof
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field

    @dataclass(slots=True)
    class Roof:  # type: ignore[no-redef]
        id: int
        top: float
        ground: float
        h: float
        area: float
        min_width: float
        centroid: tuple
        ring: list
        edges: list = field(default_factory=list)
        neighbors: list = field(default_factory=list)
        sight: float = 0.0

    @dataclass(slots=True)
    class ClimbFace:  # type: ignore[no-redef]
        roof_id: int
        ground_cell: int
        roof_cell: int
        pos: tuple
        h: float
        eave: bool

_NAN = float("nan")


class MapData:
    """See module docstring (CONTRACTS.md section 6.1 is the binding API)."""

    def __init__(self):
        self.degraded = True
        self.degraded_reason = "not loaded"
        self.empty = True
        self.has_sight = False
        self.has_cover = False
        self.params_match = True
        self.load_ms = 0.0
        self.header = None
        self.overhangs = []
        self.roofs = [_roof_placeholder()]
        self.roof_info = [None]
        self.climbs = []
        self.climb_info = {}
        self.trees = []
        self.prisms = []
        self._bd = dict(_DEF_BOUNDS)
        self.bounds = _btuple(self._bd)
        self.canopy_density = CANOPY_DENSITY
        self.canopy_block_t = CANOPY_BLOCK_T
        self._canopy_len = math.log(1.0 / CANOPY_BLOCK_T) / CANOPY_DENSITY
        self._deadline = None
        self._nav_lazy = False
        self._los_cache = {}
        self._dw4 = None
        self._ohp = []
        self._ob_ext = self._tr_ext = None
        self._cmax, self._nbx, self._nby = None, 1, 1

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, map_dir, C=None, deadline=None, *, bin_path=None, files=None):
        """Never raises. `deadline` = absolute time.perf_counter() seconds (None = unlimited).
        For convenience a number passed as the second positional argument is taken as the deadline."""
        if deadline is None and isinstance(C, (int, float)):
            C, deadline = None, C
        t0 = time.perf_counter()
        md = cls()
        md._deadline = deadline
        try:
            md._load(map_dir, C, bin_path, files)
        except Exception as e:  # last resort: flat empty map
            md.degraded = True
            md.degraded_reason = "load error: %s" % (e.__class__.__name__,)
            md._init_meta(_DEF_TERRAIN, _DEF_BOUNDS, {}, None)
            md._deadline = time.perf_counter() - 1.0
            md._build_degraded()
        md.load_ms = (time.perf_counter() - t0) * 1000.0
        return md

    def _load(self, map_dir, C, bin_path, files):
        files = files or ["v2/world.json", "v2/terrain.i16"]
        wbytes = _read(os.path.join(map_dir or "", files[0]))
        tbytes = _read(os.path.join(map_dir or "", files[1]))
        reason = None
        try:
            data = _read(bin_path or DEFAULT_BIN)
            if data is None:
                reason = "map.bin missing"
            elif wbytes is None or tbytes is None:
                reason = "map files missing"
            else:
                nl = data.index(b"\n")
                hdr = json.loads(data[:nl])
                self.header = hdr
                if hdr.get("version") != VERSION:
                    reason = "map.bin version"
                elif (hdr.get("world_sha256") != sha256_bytes(wbytes)
                      or hdr.get("terrain_sha256") != sha256_bytes(tbytes)):
                    reason = "hash mismatch"
                else:
                    self._init_meta(hdr["terrain"], hdr["bounds"], hdr.get("rules") or {}, tbytes)
                    self._load_blobs(hdr, memoryview(data)[nl + 1:])
                    self.degraded = False
                    self.degraded_reason = ""
                    self._compare_params(hdr.get("params") or {}, C)
        except _Timeout:
            reason = "deadline during load"
        except Exception as e:  # corrupt file: never crash the bot at hello
            reason = "map.bin error: %s" % (e.__class__.__name__,)
        if reason is not None:
            world = None
            if wbytes is not None:
                try:
                    world = json.loads(wbytes)
                except ValueError:
                    world = None
            meta = world_meta(world)
            self._init_meta(meta["terrain"], meta["bounds"], meta["rules"], tbytes)
            self.trees = meta["trees"]
            self.prisms = meta["prisms"]
            self.degraded = True
            self.degraded_reason = reason
            self._build_degraded()

    def _compare_params(self, params, C):
        """map.bin is baked with the recorded config; flag (do not degrade) when hello differs."""
        if C is None:
            return
        for key, name in (("step_up", "STEP_UP"), ("jump_h", "JUMP_H")):
            v = getattr(C, name, None)
            if isinstance(v, (int, float)) and key in params and abs(v - params[key]) > 0.05:
                self.params_match = False

    def _check(self):
        if self._deadline is not None and time.perf_counter() > self._deadline:
            raise _Timeout()

    def _init_meta(self, tm, bounds, rules, tbytes):
        self._bd = dict(bounds)
        self.bounds = _btuple(self._bd)
        self.tminx, self.tminy = float(tm["minX"]), float(tm["minY"])
        self.nx, self.ny = int(tm["nx"]), int(tm["ny"])
        self.gx0, self.gy0 = self.tminx - 0.5, self.tminy - 0.5   # 1 m cell (0,0) lower-left corner
        self.nx4, self.ny4 = (self.nx + NAV_STEP_I - 1) // NAV_STEP_I, (self.ny + NAV_STEP_I - 1) // NAV_STEP_I
        self.min_x4, self.min_y4 = self.gx0, self.gy0
        self.S5_NX, self.S5_NY = (self.nx + 4) // 5, (self.ny + 4) // 5
        self.C2_NX, self.C2_NY = (self.nx + 1) // 2, (self.ny + 1) // 2
        N = self.nx * self.ny
        self.terr = array.array("h")
        if tbytes is not None and len(tbytes) == 2 * N:
            self.terr.frombytes(tbytes)
            if array.array("h", [1]).tobytes() != b"\x01\x00":
                self.terr.byteswap()
            self.empty = False
        else:
            self.terr = array.array("h", [0]) * N
            self.empty = True
        self.canopy_density = float(rules.get("canopyDensity", CANOPY_DENSITY))
        self.canopy_block_t = float(rules.get("canopyBlockT", CANOPY_BLOCK_T))
        if self.canopy_density > 0 and 0 < self.canopy_block_t < 1:
            self._canopy_len = math.log(1.0 / self.canopy_block_t) / self.canopy_density
        else:
            self._canopy_len = 1e9

    def _load_blobs(self, hdr, body):
        metas = {m["name"]: m for m in hdr["blobs"]}
        little = array.array("h", [1]).tobytes() == b"\x01\x00"

        def blob(name):
            self._check()
            m = metas[name]
            b = body[m["offset"]:m["offset"] + m["nbytes"]]
            return zlib.decompress(b) if m.get("zlib", True) else bytes(b)

        def arr(name, tc):
            a = array.array(tc, blob(name))
            if not little and a.itemsize > 1:
                a.byteswap()
            return a

        self.surf = arr("surface_h", "h")
        self.clear = blob("clear_dm")
        self.foot = arr("foot_id", "H")
        m = metas["nav4"]
        self.nx4, self.ny4 = m["nx"], m["ny"]
        n4 = self.nx4 * self.ny4
        nb = blob("nav4")
        hb = array.array("h", nb[:2 * n4])
        if not little:
            hb.byteswap()
        self._nh = hb
        o = 2 * n4
        self._nf = nb[o:o + n4]
        self._nw = nb[o + n4:o + 2 * n4]
        self._nj = nb[o + 2 * n4:o + 3 * n4]
        self._nu = nb[o + 3 * n4:o + 4 * n4]
        m = metas["sight5"]
        self.S5_NX, self.S5_NY = m["nx"], m["ny"]
        self._s5 = blob("sight5")
        self._s5d = blob("sight5_dir")
        self._w5 = blob("wall5")
        m = metas["cover2"]
        self.C2_NX, self.C2_NY = m["nx"], m["ny"]
        self._c2 = blob("cover2")
        m = metas["surface_max8"]
        self._cmax = list(arr("surface_max8", "h"))
        self._nbx, self._nby = m["nx"], m["ny"]
        self.overhangs = json.loads(blob("overhangs"))
        self._set_roofs(json.loads(blob("roofs")))
        self._set_climbs(json.loads(blob("climbs")))
        tf = arr("trees", "f")
        self.trees = [tuple(tf[o:o + 10]) for o in range(0, len(tf) - 9, 10)]
        self.has_sight = True
        self.has_cover = True
        self._nav_lazy = False
        self._build_extras()

    def _set_roofs(self, recs):
        roofs = [_roof_placeholder()]
        for r in recs:
            ring = r.pop("ring", None) or []
            roofs.append(Roof(
                id=r["id"], top=r["top"], ground=r["ground"], h=r["h"], area=float(r["area"]),
                min_width=r.get("min_width", 0.0), centroid=tuple(r["centroid"]),
                ring=[(ring[k], ring[k + 1]) for k in range(0, len(ring) - 1, 2)],
                edges=[((e[0], e[1]), (e[2], e[3]), (e[4], e[5])) for e in r.pop("edges", None) or ()],
                neighbors=list(r.get("neighbors") or ()), sight=float(r.get("sight", 0.0))))
        self.roofs = roofs
        self.roof_info = [None] + recs     # ring/edges live only in self.roofs

    def _set_climbs(self, blob):
        ix = [blob["fields"].index(n) for n in CLIMB_FIELDS]
        self.climb_info = {int(rid): [tuple(f[k] for k in ix) for f in faces] for rid, faces in blob["roofs"]}
        g, r, wx, wy, h, ev = (CF[n] for n in ("ground_cell", "roof_cell", "wx", "wy", "h", "eave"))
        self.climbs = [ClimbFace(roof_id=rid, ground_cell=f[g], roof_cell=f[r], pos=(f[wx], f[wy]),
                                 h=f[h], eave=bool(f[ev]))
                       for rid, faces in self.climb_info.items() for f in faces]

    def _build_degraded(self):
        """Section 11.3: 1 m surfaces + nav4 (edges computed lazily on first use) from prisms and
        terrain; coarse 4 m wall distance for wall5/cover2; no sight grid; roofs without climbs."""
        nx, ny = self.nx, self.ny
        N = nx * ny
        self.has_sight = self.has_cover = False
        self.overhangs, self.climbs, self.climb_info = [], [], {}
        self.roofs, self.roof_info = [_roof_placeholder()], [None]
        self.foot = array.array("H", [0]) * N
        self.surf = array.array("h", self.terr)
        self._nav1 = self.surf
        self._cmax, self._nbx, self._nby = None, 1, 1
        self._ob_ext = self._tr_ext = None
        n4 = self.nx4 * self.ny4
        self._nh = array.array("h", [0]) * n4
        self._nf = bytearray([NAV_F_WALK]) * n4
        self._nu = bytearray(n4)
        self._nw = bytearray([255]) * n4
        self._nj = bytearray(n4)
        stage = "surfaces"
        try:
            if self.prisms and not self.empty:
                surf, nav, lab, under, over_ids = build_surfaces(
                    self.terr, nx, ny, self.tminx, self.tminy, self.prisms, self._check)
            else:
                nav, lab, under, over_ids = self.surf, array.array("h", [-1]) * N, bytearray(N), []
                surf = self.surf
            self.surf = surf
            self._nav1 = nav
            stage = "coarse"
            cmax, nbx, nby = coarse_max(surf, nx, ny, check=self._check)
            self.overhangs = [_overhang_rec(i, self.prisms[i]) for i in over_ids]
            self._cmax, self._nbx, self._nby = cmax, nbx, nby
            self._build_extras()
            stage = "nav4"
            self._check()
            kinds = [p.get("kind") for p in self.prisms]
            H, F, U, NX, NY = nav4_cells(nav, under, lab, kinds, None, nx, ny, self.gx0, self.gy0, self._bd)
            self._nh, self._nf, self._nu = H, F, U
            self._nw = bytearray(NX * NY)
            self._nj = bytearray(NX * NY)
            self._nav_done = bytearray(NX * NY)
            self._nav_lazy = True
            stage = "walls"
            self._check()
            self._dw4 = wall_dist4(nav, self.terr, lab, kinds, nx, ny, NX, NY, self._check)
            stage = "roofs"
            comps = roof_components(nav, self.terr, lab, kinds, nx, ny, self._check)
            recs = []
            for rid, (cells, top, gnd) in enumerate(comps, 1):
                sx = sy = 0
                for c in cells:
                    self.foot[c] = rid
                    sx += c % nx
                    sy += c // nx
                recs.append({"id": rid, "top": top / 100.0, "ground": gnd / 100.0,
                             "h": round((top - gnd) / 100.0, 2), "area": len(cells), "min_width": 0.0,
                             "centroid": [round(self.tminx + sx / len(cells), 2),
                                          round(self.tminy + sy / len(cells), 2)]})
            self._set_roofs(recs)
            stage = "done"
        except _Timeout:
            # out of time: what is missing stays flat terrain / all-walkable (straight-line steering + O5)
            self.degraded_reason += "; degraded build stopped at " + stage

    def _build_extras(self):
        """Per coarse-block lists of overhang prisms and trees (for LOS candidate collection)."""
        nbx, nby = self._nbx, self._nby
        ob = [None] * (nbx * nby)
        tr = [None] * (nbx * nby)
        gx0, gy0 = self.gx0, self.gy0
        B = float(COARSE)
        ohp = []
        for k, o in enumerate(self.overhangs):
            ring = o["ring"]
            xs, ys = ring[0::2], ring[1::2]
            edges = []
            for poly in [ring] + list(o.get("holes") or ()):
                n = len(poly) // 2
                for e in range(n):
                    e2 = (e + 1) % n
                    edges.append((poly[2 * e], poly[2 * e + 1], poly[2 * e2], poly[2 * e2 + 1]))
            ohp.append((min(xs), max(xs), min(ys), max(ys), o["base"], o["top"], tuple(edges)))
            self._register(ob, k, min(xs), max(xs), min(ys), max(ys), nbx, nby, gx0, gy0, B)
        for k, t in enumerate(self.trees):
            cx, cy, cz, R, Rz, tx, ty, trr, zb, zt = t
            x0, x1 = min(cx - R, tx - trr), max(cx + R, tx + trr)
            y0, y1 = min(cy - R, ty - trr), max(cy + R, ty + trr)
            self._register(tr, k, x0, x1, y0, y1, nbx, nby, gx0, gy0, B)
        self._ohp = ohp
        self._ob_ext = [tuple(v) if v else None for v in ob]
        self._tr_ext = [tuple(v) if v else None for v in tr]

    @staticmethod
    def _register(lst, k, x0, x1, y0, y1, nbx, nby, gx0, gy0, B):
        I0 = max(0, int((x0 - gx0) // B))
        I1 = min(nbx - 1, int((x1 - gx0) // B))
        J0 = max(0, int((y0 - gy0) // B))
        J1 = min(nby - 1, int((y1 - gy0) // B))
        for J in range(J0, J1 + 1):
            for I in range(I0, I1 + 1):
                b = J * nbx + I
                if lst[b] is None:
                    lst[b] = []
                lst[b].append(k)

    # ------------------------------------------------------------------ per tick
    def new_tick(self):
        """Clear the one-tick LOS cache (CONTRACTS 2: called once per obs)."""
        if self._los_cache:
            self._los_cache = {}

    # ------------------------------------------------------------------ heights
    def in_bounds(self, x, y):
        b = self._bd
        return b["minX"] <= x <= b["maxX"] and b["minY"] <= y <= b["maxY"]

    def terrain_h(self, x, y):
        """Bilinear terrain.i16 (protocol 11), clamped at the grid border."""
        fx = x - self.tminx
        fy = y - self.tminy
        nx, ny = self.nx, self.ny
        if fx < 0.0:
            fx = 0.0
        elif fx > nx - 1:
            fx = nx - 1.0
        if fy < 0.0:
            fy = 0.0
        elif fy > ny - 1:
            fy = ny - 1.0
        i = int(fx)
        j = int(fy)
        if i >= nx - 1:
            i = nx - 2
        if j >= ny - 1:
            j = ny - 2
        ax = fx - i
        ay = fy - j
        t = self.terr
        c = j * nx + i
        h00, h10, h01, h11 = t[c], t[c + 1], t[c + nx], t[c + nx + 1]
        return ((h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay) * 0.01

    def _cell1(self, x, y):
        i = int((x - self.gx0) // 1)
        j = int((y - self.gy0) // 1)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return None

    def surface_h(self, x, y):
        """Top of a building/stadium column at this 1 m cell, else bilinear terrain (11.2 surface_h).
        Bridges and overhangs are not included (see ground_under / overhangs)."""
        c = self._cell1(x, y)
        g = self.terrain_h(x, y)
        if c is None:
            return g
        s = self.surf[c]
        if s - self.terr[c] > 30:
            return max(g, s * 0.01)
        return g

    def ground_under(self, x, y, z):
        """Highest walkable surface <= z + STEP_UP at (x, y): surface_h or a bridge deck/overhang top."""
        best = self.surface_h(x, y)
        lim = z + STEP_UP
        if best > lim:
            return best
        for k, (bx0, bx1, by0, by1, base, top, edges) in enumerate(self._ohp):
            if top <= best or top > lim or not (bx0 <= x <= bx1 and by0 <= y <= by1):
                continue
            ins = False
            for ex0, ey0, ex1, ey1 in edges:
                if (ey0 > y) != (ey1 > y) and x < ex0 + (y - ey0) * (ex1 - ex0) / (ey1 - ey0):
                    ins = not ins
            if ins:
                best = top
        return best

    def clearance(self, x, y):
        """clear_dm in m (0 = blocked / outside the grid), capped 25.5."""
        c = self._cell1(x, y)
        if c is None:
            return 0.0
        if self.degraded:
            return self._clear_local(c)
        return self.clear[c] * 0.1

    def _blocked1(self, c):
        nav = self._nav1
        nx, ny = self.nx, self.ny
        i, j = c % nx, c // nx
        if not self.in_bounds(self.tminx + i, self.tminy + j):
            return True
        h = nav[c] - int(STEP_UP * 100)
        for di, dj in DIRS8:
            i2, j2 = i + di, j + dj
            if 0 <= i2 < nx and 0 <= j2 < ny and nav[j2 * nx + i2] < h:
                return True
        return False

    def _clear_local(self, c, cap=4):
        nx, ny = self.nx, self.ny
        i, j = c % nx, c // nx
        best = float(cap) + 0.5
        for dj in range(-cap, cap + 1):
            for di in range(-cap, cap + 1):
                d = math.hypot(di, dj)
                if d >= best:
                    continue
                i2, j2 = i + di, j + dj
                if not (0 <= i2 < nx and 0 <= j2 < ny) or self._blocked1(j2 * nx + i2):
                    best = d
        return max(0.0, best - 0.5)

    def roof_id(self, x, y):
        c = self._cell1(x, y)
        return self.foot[c] if c is not None else 0

    def roof(self, rid):
        """Roof by id (None for 0 / unknown)."""
        if 1 <= rid < len(self.roofs):
            return self.roofs[rid]
        return None

    # ------------------------------------------------------------------ 5 m and 2 m grids
    def _i5(self, x, y):
        I = int((x - self.gx0) // SIGHT_STEP)
        J = int((y - self.gy0) // SIGHT_STEP)
        if 0 <= I < self.S5_NX and 0 <= J < self.S5_NY:
            return J * self.S5_NX + I
        return -1

    def sight5(self, x, y):
        """Median sight distance (m) from eye 8 m above surface; NaN when unknown (degraded)."""
        if not self.has_sight:
            return _NAN
        k = self._i5(x, y)
        return self._s5[k] * 2.0 if k >= 0 else 0.0

    def sight5_dir(self, x, y, sector):
        """Best sight (m) in sector k = yaw in [45k, 45k + 45) deg; NaN when unknown."""
        if not self.has_sight:
            return _NAN
        k = self._i5(x, y)
        return self._s5d[k * 8 + (sector & 7)] * 2.0 if k >= 0 else 0.0

    def wall5(self, x, y):
        """Distance (m) to the nearest building wall (5 m grid; degraded: 4 m grid, coarser)."""
        if self.has_cover:
            k = self._i5(x, y)
            return self._w5[k] * 0.5 if k >= 0 else 0.0
        if self._dw4 is None:
            return 127.5
        c = self.cell4(x, y)
        return min(127.5, self._dw4[c]) if c >= 0 else 0.0

    def cover2(self, x, y):
        """Raw cover2 bits (COVER_WALL_MASK distance m, COVER_CORNER, COVER_WALKABLE)."""
        if self.has_cover:
            I = int((x - self.gx0) // COVER_STEP)
            J = int((y - self.gy0) // COVER_STEP)
            if 0 <= I < self.C2_NX and 0 <= J < self.C2_NY:
                return self._c2[J * self.C2_NX + I]
            return 0
        c = self.cell4(x, y)
        if c < 0:
            return 0
        v = min(63, int(self.wall5(x, y)))
        if self._nf[c] & NAV_F_WALK:
            v |= COVER_WALKABLE
        return v

    def cover2_cells(self, x, y, r):
        """2 m cells whose centre is within r of (x, y): [(cx, cy, bits)], nearest first."""
        I0 = int((x - r - self.gx0) // COVER_STEP)
        I1 = int((x + r - self.gx0) // COVER_STEP)
        J0 = int((y - r - self.gy0) // COVER_STEP)
        J1 = int((y + r - self.gy0) // COVER_STEP)
        I0, J0 = max(0, I0), max(0, J0)
        I1, J1 = min(self.C2_NX - 1, I1), min(self.C2_NY - 1, J1)
        r2 = r * r
        out = []
        for J in range(J0, J1 + 1):
            cy = self.gy0 + (J + 0.5) * COVER_STEP
            dy2 = (cy - y) * (cy - y)
            if dy2 > r2:
                continue
            for I in range(I0, I1 + 1):
                cx = self.gx0 + (I + 0.5) * COVER_STEP
                d2 = (cx - x) * (cx - x) + dy2
                if d2 <= r2:
                    out.append((d2, cx, cy))
        out.sort()
        if self.has_cover:
            c2, nxc, gx0, gy0 = self._c2, self.C2_NX, self.gx0, self.gy0
            return [(cx, cy, c2[int((cy - gy0) // COVER_STEP) * nxc + int((cx - gx0) // COVER_STEP)])
                    for d2, cx, cy in out]
        return [(cx, cy, self.cover2(cx, cy)) for d2, cx, cy in out]

    # ------------------------------------------------------------------ 4 m nav grid
    def cell4(self, x, y):
        """nav4 index j * nx4 + i, -1 outside the grid."""
        I = int((x - self.gx0) // NAV_STEP_I)
        J = int((y - self.gy0) // NAV_STEP_I)
        if 0 <= I < self.nx4 and 0 <= J < self.ny4:
            return J * self.nx4 + I
        return -1

    def cell4_xy(self, idx):
        return (self.gx0 + (idx % self.nx4 + 0.5) * NAV_STEP_I,
                self.gy0 + (idx // self.nx4 + 0.5) * NAV_STEP_I)

    def neighbor4(self, idx, k):
        """Neighbour index in direction NAV_DIRS[k], -1 outside."""
        di, dj = DIRS8[k]
        I, J = idx % self.nx4 + di, idx // self.nx4 + dj
        if 0 <= I < self.nx4 and 0 <= J < self.ny4:
            return J * self.nx4 + I
        return -1

    def nav_h(self, idx):
        return self._nh[idx] * 0.01

    def nav_flags(self, idx):
        return self._nf[idx]

    def walkable(self, idx):
        return bool(self._nf[idx] & NAV_F_WALK)

    def underpass_clear(self, idx):
        return self._nu[idx] * 0.1

    def _lazy_edges(self, c):
        w, j = nav4_edges_cell(c, self._nh, self._nf, self._nav1, self.nx, self.ny, self.nx4, self.ny4)
        self._nw[c] = w
        self._nj[c] = j
        self._nav_done[c] = 1

    def walk_edges(self, idx):
        if self._nav_lazy and not self._nav_done[idx]:
            self._lazy_edges(idx)
        return self._nw[idx]

    def jump_edges(self, idx):
        if self._nav_lazy and not self._nav_done[idx]:
            self._lazy_edges(idx)
        return self._nj[idx]

    # ------------------------------------------------------------------ LOS
    def los(self, a, b, *, canopy=False, trunks=False):
        """Line of sight / straight line a -> b (3D points). True = clear.

        Default (CONTRACTS 6.1, plan 4.6 / A-13): terrain + buildings + stadium + bridges/overhangs,
        trees ignored. canopy=True adds crown attenuation, trunks=True adds trunks (sight only)."""
        x0, y0, z0 = a[0], a[1], a[2]
        x1, y1, z1 = b[0], b[1], b[2]
        need_trees = (canopy or trunks) and self._tr_ext is not None
        ob, tr = [], []
        if self._grid_blocked(x0, y0, z0, x1, y1, z1, ob, tr if need_trees else None):
            return False
        if ob:
            if self._overhang_block(x0, y0, z0, x1, y1, z1, ob):
                return False
        if need_trees and tr:
            if self._trees_block(x0, y0, z0, x1, y1, z1, tr, canopy, trunks):
                return False
        return True

    def shot_clear(self, a, b):
        """Straight shot a -> b: trees never block shots (protocol 7/8)."""
        return self.los(a, b)

    def visible(self, eye, target, *, near=False):
        """Protocol 8 vision line: buildings, terrain, trunks and (unless near-range) canopy."""
        return self.los(eye, target, canopy=not near, trunks=True)

    def los_cached(self, key, a, b):
        """los(a, b) memoised for this tick under the caller's key (e.g. (track_id, cell_a, cell_b))."""
        r = self._los_cache.get(key)
        if r is None:
            r = self._los_cache[key] = self.los(a, b)
        return r

    # ------------------------------------------------------------------ names from the task brief
    ground = terrain_h
    surface = surface_h
    center_of = cell4_xy
    h = nav_h
    flags = nav_flags
    sight = sight5
    sight_dir = sight5_dir
    wall_dist = wall5
    cover_bits = cover2

    def cell_of(self, x, y):
        c = self.cell4(x, y)
        return c if c >= 0 else None

    def neighbor(self, idx, k):
        c = self.neighbor4(idx, k)
        return c if c >= 0 else None

    def _grid_blocked(self, x0, y0, z0, x1, y1, z1, ob, tr):
        """DDA over coarse 8 m blocks (skip when the ray is above the block max), then over 1 m
        cells. Collects overhang / tree candidates of every visited block into ob / tr."""
        du = x1 - x0
        dv = y1 - y0
        L = math.sqrt(du * du + dv * dv)
        if L < 2 * LOS_END_SKIP:
            return False
        nx, ny = self.nx, self.ny
        u0 = x0 - self.gx0
        v0 = y0 - self.gy0
        ta = LOS_END_SKIP / L
        tb = 1.0 - ta
        # clip to the grid rectangle [0, nx) x [0, ny)
        for p0, dp, hi in ((u0, du, nx), (v0, dv, ny)):
            if dp == 0.0:
                if p0 < 0.0 or p0 >= hi:
                    return False
            else:
                t1 = (0.0 - p0) / dp
                t2 = (hi - 1e-6 - p0) / dp
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > ta:
                    ta = t1
                if t2 < tb:
                    tb = t2
        if ta >= tb:
            return False
        cmax = self._cmax
        surf = self.surf
        zc0 = z0 * 100.0
        dzc = (z1 - z0) * 100.0
        up = dzc >= 0.0
        obx = self._ob_ext
        trx = self._tr_ext
        if cmax is None:
            return False
        # ---- coarse DDA (block units)
        B = COARSE
        nbx, nby = self._nbx, self._nby
        bu0, bv0, bdu, bdv = u0 / B, v0 / B, du / B, dv / B
        t = ta
        te = ta + 1e-9
        I = int(bu0 + bdu * te)
        J = int(bv0 + bdv * te)
        if I >= nbx:
            I = nbx - 1
        if J >= nby:
            J = nby - 1
        INF = 1e30
        if bdu > 0:
            si, tdu = 1, 1.0 / bdu
            tmu = (I + 1 - bu0) / bdu
        elif bdu < 0:
            si, tdu = -1, -1.0 / bdu
            tmu = (I - bu0) / bdu
        else:
            si, tdu, tmu = 0, INF, INF
        if bdv > 0:
            sj, tdv = 1, 1.0 / bdv
            tmv = (J + 1 - bv0) / bdv
        elif bdv < 0:
            sj, tdv = -1, -1.0 / bdv
            tmv = (J - bv0) / bdv
        else:
            sj, tdv, tmv = 0, INF, INF
        while True:
            tn = tmu if tmu < tmv else tmv
            if tn > tb:
                tn = tb
            b = J * nbx + I
            e = obx[b]
            if e is not None:
                ob.extend(e)
            if tr is not None:
                e = trx[b]
                if e is not None:
                    tr.extend(e)
            zlo = zc0 + dzc * (t if up else tn)
            if zlo < cmax[b]:
                # ---- fine DDA over [t, tn]
                tf = t + 1e-9
                i = int(u0 + du * tf)
                j = int(v0 + dv * tf)
                if i >= nx:
                    i = nx - 1
                if j >= ny:
                    j = ny - 1
                if du > 0:
                    fi, fdu = 1, 1.0 / du
                    fmu = (i + 1 - u0) / du
                elif du < 0:
                    fi, fdu = -1, -1.0 / du
                    fmu = (i - u0) / du
                else:
                    fi, fdu, fmu = 0, INF, INF
                if dv > 0:
                    fj, fdv = nx, 1.0 / dv
                    fmv = (j + 1 - v0) / dv
                elif dv < 0:
                    fj, fdv = -nx, -1.0 / dv
                    fmv = (j - v0) / dv
                else:
                    fj, fdv, fmv = 0, INF, INF
                c = j * nx + i
                s = t
                N = nx * ny
                while True:
                    sn = fmu if fmu < fmv else fmv
                    if sn > tn:
                        sn = tn
                    if (zc0 + dzc * (s if up else sn)) < surf[c]:
                        return True
                    if sn >= tn:
                        break
                    if fmu < fmv:
                        c += fi
                        s = fmu
                        fmu += fdu
                    else:
                        c += fj
                        s = fmv
                        fmv += fdv
                    if c < 0 or c >= N:
                        break
            if tn >= tb:
                return False
            if tmu < tmv:
                I += si
                t = tmu
                tmu += tdu
                if I < 0 or I >= nbx:
                    return False
            else:
                J += sj
                t = tmv
                tmv += tdv
                if J < 0 or J >= nby:
                    return False

    def _overhang_block(self, x0, y0, z0, x1, y1, z1, ids):
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        sx0, sx1 = (x0, x1) if x0 < x1 else (x1, x0)
        sy0, sy1 = (y0, y1) if y0 < y1 else (y1, y0)
        sz0, sz1 = (z0, z1) if z0 < z1 else (z1, z0)
        for k in set(ids):
            bx0, bx1, by0, by1, base, top, edges = self._ohp[k]
            if bx1 < sx0 or bx0 > sx1 or by1 < sy0 or by0 > sy1 or top < sz0 or base > sz1:
                continue
            ts = []
            inside = False
            for ex0, ey0, ex1, ey1 in edges:
                # point-in-polygon for the start point (even-odd)
                if (ey0 > y0) != (ey1 > y0):
                    if x0 < ex0 + (y0 - ey0) * (ex1 - ex0) / (ey1 - ey0):
                        inside = not inside
                fx, fy = ex1 - ex0, ey1 - ey0
                den = dx * fy - dy * fx
                if den == 0.0:
                    continue
                wx, wy = ex0 - x0, ey0 - y0
                t = (wx * fy - wy * fx) / den
                if t <= 0.0 or t >= 1.0:
                    continue
                s = (wx * dy - wy * dx) / den
                if 0.0 <= s < 1.0:
                    ts.append(t)
            ts.sort()
            ts.append(1.0)
            ta = 0.0
            for tcross in ts:
                if inside:
                    za, zb = z0 + dz * ta, z0 + dz * tcross
                    lo, hi = (za, zb) if za < zb else (zb, za)
                    if lo < top and hi > base:
                        return True
                inside = not inside
                ta = tcross
        return False

    def _trees_block(self, x0, y0, z0, x1, y1, z1, ids, canopy, trunks):
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        seg = math.sqrt(dx * dx + dy * dy + dz * dz)
        if seg <= 0.0:
            return False
        dd2 = dx * dx + dy * dy
        acc = 0.0
        lim = self._canopy_len
        trees = self.trees
        for k in set(ids):
            cx, cy, cz, R, Rz, tx, ty, trr, zb, zt = trees[k]
            if trunks and dd2 > 0.0:
                ox, oy = x0 - tx, y0 - ty
                b = ox * dx + oy * dy
                c = ox * ox + oy * oy - trr * trr
                disc = b * b - dd2 * c
                if disc > 0.0:
                    sq = math.sqrt(disc)
                    t1 = (-b - sq) / dd2
                    t2 = (-b + sq) / dd2
                    if t1 < 0.0:
                        t1 = 0.0
                    if t2 > 1.0:
                        t2 = 1.0
                    if t1 < t2:
                        za, zbb = z0 + dz * t1, z0 + dz * t2
                        lo, hi = (za, zbb) if za < zbb else (zbb, za)
                        if lo < zt and hi > zb:
                            return True
            if canopy:
                iR2 = 1.0 / (R * R)
                iZ2 = 1.0 / (Rz * Rz)
                ox, oy, oz = x0 - cx, y0 - cy, z0 - cz
                a = dd2 * iR2 + dz * dz * iZ2
                b = (ox * dx + oy * dy) * iR2 + oz * dz * iZ2
                c = (ox * ox + oy * oy) * iR2 + oz * oz * iZ2 - 1.0
                disc = b * b - a * c
                if disc > 0.0 and a > 0.0:
                    sq = math.sqrt(disc)
                    t1 = (-b - sq) / a
                    t2 = (-b + sq) / a
                    if t1 < 0.0:
                        t1 = 0.0
                    if t2 > 1.0:
                        t2 = 1.0
                    if t1 < t2:
                        acc += (t2 - t1) * seg
                        if acc > lim:
                            return True
        return False



def _btuple(b):
    return (b["minX"], b["minY"], b["maxX"], b["maxY"])


def _roof_placeholder():
    return Roof(id=0, top=0.0, ground=0.0, h=0.0, area=0.0, min_width=0.0, centroid=(0.0, 0.0), ring=[])


def _overhang_rec(k, p):
    return {"id": k, "kind": p.get("kind"), "base": p["base"], "top": p["top"],
            "ring": p["ring"], "holes": p.get("holes") or []}


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None
