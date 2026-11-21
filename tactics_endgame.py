"""X4 ENDGAME: FINAL_APPROACH, FINAL_HOLD, COLLAPSE (GP "375 s+: stop fighting"; plan 7.11 R11.1-R11.5).

R11.2 final point: next_center on walkable ground; inside a building footprint the roof above it when
climbable (height <= CLIMB_MAX_H with FINAL_ROOF_ENERGY_MARGIN, width >= ROOF_MIN_W, A-11), else the walkable
cell nearest the center; a point taken by another mech -> the nearest free cell. Re-evaluated on every
next_changed (modes.strategic_refresh clears it).
Aim (6.4 X4 row): AIM1 and AIM2 come from overlays; then the R11.3 self-defense target (a mech that damaged
us within SELF_DEFENSE_S, highest damage first) at prio X4_SELF_DEFENSE; else the arbiter's AIM10.
Fire: R11.3 self-defense only (modes sets fire_self_defense_only; the arbiter applies F5 and R11.5).
"""
from __future__ import annotations

import math

import modes as M


def final_point(ctx):
    m, C, world, mp = ctx.modes, ctx.C, ctx.world, ctx.mapdata
    me = world.me
    if m.final_point is not None:
        return m.final_point
    c = world.zone.next_center
    pt = (c[0], c[1])
    rid = 0
    try:
        rid = mp.roof_id(c[0], c[1])
    except Exception:
        rid = 0
    if rid:
        roof = None
        try:
            roof = mp.roofs[rid]
        except Exception:
            roof = None
        climbable = roof is not None and roof.h <= C.CLIMB_MAX_H and roof.min_width >= C.ROOF_MIN_W * (me.s or 1.0) and \
            C.CLIMB_COST_PER_M * roof.h + C.CLIMB_COST_PAD + C.FINAL_ROOF_ENERGY_MARGIN <= me.max_energy
        if not climbable:
            pt = _nearest_walkable(ctx, c)
    else:
        try:
            if mp.clearance(c[0], c[1]) <= 0:
                pt = _nearest_walkable(ctx, c)
        except Exception:
            pass
    # taken by another mech -> nearest free cell
    for tr in world.visible_mechs():
        if M.hdist(tr.pos, pt) < 2 * (me.radius or 2.5):
            u = M.unit(me.pos[0] - pt[0], me.pos[1] - pt[1]) or (1.0, 0.0)
            pt = (pt[0] + u[0] * 2.2 * (me.radius or 2.5), pt[1] + u[1] * 2.2 * (me.radius or 2.5))
            break
    m.final_point = pt
    return pt


def _nearest_walkable(ctx, c):
    mp = ctx.mapdata
    for r in range(2, 60, 2):
        for k in range(16):
            a = 2 * math.pi * k / 16
            x, y = c[0] + r * math.cos(a), c[1] + r * math.sin(a)
            try:
                if mp.clearance(x, y) > 0 and not mp.roof_id(x, y):
                    return (x, y)
            except Exception:
                return (c[0], c[1])
    return (c[0], c[1])


def _self_defense_aim(ctx) -> None:
    """R11.3 self-defense target (X4 aim list, prio X4_SELF_DEFENSE)."""
    m, C, world = ctx.modes, ctx.C, ctx.world
    best = None
    for tr in world.visible_mechs():
        if world.damaged_us_recently(tr.id, C.SELF_DEFENSE_S):
            k = (-tr.damage_to_us, tr.id)
            if best is None or k < best[0]:
                best = (k, tr)
    if best is not None:
        m.aim_track(ctx, "X4_SELF_DEFENSE", "R11.3", best[1])
        m.fire(ctx, best[1].id, None)


def run(sub: str, ctx) -> str:
    m, C, world, A = ctx.modes, ctx.C, ctx.world, ctx.A
    me, t = world.me, world.tick.t
    pt = final_point(ctx)
    dist = M.hdist(me.pos, pt)
    if t >= C.FINAL_SHRINK_T and sub != "COLLAPSE":
        m._log(t, "sub", sub, "COLLAPSE", None, "R11.5")
        sub = m.sub = "COLLAPSE"
    elif sub == "FINAL_APPROACH" and dist <= C.FINAL_POINT_TOL_M:
        m._log(t, "sub", sub, "FINAL_HOLD", None, "R11.1")
        sub = m.sub = "FINAL_HOLD"
    elif sub == "FINAL_HOLD" and dist > C.HOLD_ARRIVE_R:
        m._log(t, "sub", sub, "FINAL_APPROACH", None, "R11.1")
        sub = m.sub = "FINAL_APPROACH"
    _self_defense_aim(ctx)
    if sub == "COLLAPSE":                                              # R11.5: move = dir x min(1, dist x gain)
        d = M.unit(pt[0] - me.pos[0], pt[1] - me.pos[1])
        m.legs(ctx, "R11.5", d, min(1.0, dist * C.FINAL_MOVE_GAIN), pt, juke_ok=False)
        return "R11.5"
    if sub == "FINAL_HOLD":
        d = M.unit(pt[0] - me.pos[0], pt[1] - me.pos[1])
        m.legs(ctx, "R11.1", d, min(1.0, dist * C.FINAL_MOVE_GAIN), pt, juke_ok=False)
        return "R11.1"
    # FINAL_APPROACH (R11.1): via the zone field, then the path to the point
    st = m.go(ctx, "x4", pt)
    direction = st.direction
    try:
        eta, _ = ctx.nav.zone_eta(me.pos)
    except Exception:
        eta = 0.0
    if A.slack < C.ZONE_SLACK_SOFT_S and eta > 0:
        try:
            dd = ctx.nav.zone_descent(me.pos)
            if dd is not None:
                direction = dd
        except Exception:
            pass
    m.legs(ctx, "R11.1", direction, 1.0, pt)
    ws = float((me.stats or {}).get("walk_speed", 11.7)) or 11.7
    m.boost(ctx, "R11.1", dist / ws > C.FINAL_ARRIVE_T - t, 0.0)
    return "R11.1"
