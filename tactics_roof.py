"""Roof behaviors: ROOF_HOLD (M3 ROOF sub-state, plan RZ4 / R10.3 / R12.2) and ROOF_ASSAULT (X2 sub-state,
R10.2: lightning vs a laser roofer).

ROOF_HOLD: stand ROOF_EDGE_BACKOFF in from the watched edge (the edge facing the next circle center), strafe
along it every HOLD_SHUFFLE_S (the strafe never leaves the roof, so it never triggers M2: the M2 roof entry
is by foot_id), watch the edge arc; a jet sound / rising track -> look at it (laser it during the climb via
the F-gates); a climber on our roof within LIGHTNING_RANGE x m -> drop off the far side (GP: falling free).
ROOF_ASSAULT: approach via cover to the dead zone at the building base, climb with energy >=
CLIMB_COST(h) + ROOF_ASSAULT_RESERVE (jump + jets), fire lightning; else wait at the base up to
ROOF_WAIT_MAX_S (then modes exits X2 writing (FIGHT, it)).
"""
from __future__ import annotations

import math

from contracts import BoostRequest, JumpRequest

import modes as M


def _roof(ctx, rid):
    try:
        roofs = ctx.mapdata.roofs
        return roofs[rid] if 0 < rid < len(roofs) else None
    except Exception:
        return None


def _watched_edge(ctx, roof):
    """(a, b, outward normal) of the edge whose outward normal points most toward the next center."""
    zc = ctx.world.zone.next_center
    best = None
    for (a, b, n) in (roof.edges or ()):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        u = M.unit(zc[0] - mid[0], zc[1] - mid[1]) or (1.0, 0.0)
        k = n[0] * u[0] + n[1] * u[1]
        if best is None or k > best[0]:
            best = (k, (a, b, n))
    return best[1] if best else None


def roof_legs(ctx, rule) -> None:
    """Hold the roof position: backoff from the watched edge, strafe along it."""
    m, C, world = ctx.modes, ctx.C, ctx.world
    me, t = world.me, world.tick.t
    try:
        rid = ctx.mapdata.roof_id(me.pos[0], me.pos[1])
    except Exception:
        rid = 0
    roof = _roof(ctx, rid)
    edge = _watched_edge(ctx, roof) if roof is not None else None
    if edge is None:
        m.legs(ctx, rule, None)
        return
    a, b, n = edge
    tdir = M.unit(b[0] - a[0], b[1] - a[1]) or (1.0, 0.0)
    length = M.hdist(a, b)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    tac = m.tac if m.macro == "X2" else (m.gs.data if m.gs is not None else m.tac)
    if t - tac.get("roof_strafe_t", -1e9) >= C.HOLD_SHUFFLE_S:
        tac["roof_strafe_t"] = t
        tac["roof_off"] = ctx.rng.uniform(-0.3, 0.3) * length
    off = tac.get("roof_off", 0.0)
    back = C.ROOF_EDGE_BACKOFF
    goal = (mid[0] - n[0] * back + tdir[0] * off, mid[1] - n[1] * back + tdir[1] * off)
    if M.hdist(me.pos, goal) > C.HOLD_SPOT_TOL_M * 0.5:
        m.legs(ctx, rule, M.unit(goal[0] - me.pos[0], goal[1] - me.pos[1]), C.SILENT_MOVE, goal)
    else:
        m.legs(ctx, rule, None)
    if m.macro == "X1":
        m.aim_dir(ctx, "AIM9", rule, math.atan2(n[1], n[0]))            # sweep the watched arc (AIM10 jitters)


def _hold(ctx) -> str:
    m, C, world = ctx.modes, ctx.C, ctx.world
    me, t = world.me, world.tick.t
    try:
        rid = ctx.mapdata.roof_id(me.pos[0], me.pos[1])
    except Exception:
        rid = 0
    spot = m.spot
    if spot is not None and spot[2] and rid != spot[2]:
        m.walk_to(ctx, "roof", "RZ4", spot[0])                         # M2 normally does this
        return "RZ4"
    # climber on our roof (RZ4)
    for tr in world.tracks.values():
        if tr.kind != "mech" or not tr.visible:
            continue
        try:
            same = ctx.mapdata.roof_id(tr.pos[0], tr.pos[1]) == rid and rid
        except Exception:
            same = False
        if same:
            mlt = tr.lightning_range_mult or 1.0
            d = M.hdist(me.pos, tr.pos)
            away = M.unit(me.pos[0] - tr.pos[0], me.pos[1] - tr.pos[1]) or (1.0, 0.0)
            m.aim_track(ctx, "AIM9", "RZ4", tr)
            if M.has_weapon(tr, "lightning") and d <= C.LIGHTNING_RANGE * mlt:
                m.legs(ctx, "RZ4.drop", away, 1.0, None, juke_ok=False)   # drop off the far side
                return "RZ4.drop"
            m.legs(ctx, "RZ4.back", away, 1.0, None)                      # laser + gun backing to the far edge
            m.fire(ctx, tr.id, {"laser", "gun"})
            return "RZ4.back"
        if tr.jet_since is not None and t - tr.jet_since < 2.0:
            m.aim_track(ctx, "AIM9", "RZ4.jet", tr)                        # laser it during the climb
            m.fire(ctx, tr.id, {"laser"})
    roof_legs(ctx, "RZ4")
    return "RZ4"


def _assault(ctx) -> str:
    m, C, world = ctx.modes, ctx.C, ctx.world
    me, t = world.me, world.tick.t
    tr = world.tracks.get(m.target) if m.target is not None else None
    if tr is None:
        m.legs(ctx, "R10.2", None)
        return "R10.2"
    if not tr.on_roof:                                                   # it left the roof: normal fight
        m.sub = "BRAWL" if M.hdist(me.pos, tr.pos) <= C.LIGHTNING_RANGE else "RUSH"
        m._log(t, "sub", "ROOF_ASSAULT", m.sub, tr.id, "R10.2:left")
        import tactics_lightning
        return tactics_lightning.run(m.sub, ctx)
    m.aim_track(ctx, "AIM3", "R10.2", tr)
    m.fire(ctx, tr.id, {"lightning", "gun"})
    try:
        rid = ctx.mapdata.roof_id(tr.pos[0], tr.pos[1])
    except Exception:
        rid = 0
    roof = _roof(ctx, rid)
    if roof is None:
        m.legs(ctx, "R10.2", M.unit(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]))
        return "R10.2"
    faces = [f for f in (getattr(ctx.mapdata, "climbs", None) or ()) if f.roof_id == rid and not f.eave]
    base = min(faces, key=lambda f: M.hdist(f.pos, tr.pos)).pos if faces else roof.centroid
    min_pitch = abs(float((me.stats or {}).get("min_pitch", C.path("mech.min_pitch", -0.7))))
    muzzle = float(C.path("mech.muzzle_height", 7.0))
    dz = roof.top + muzzle - (me.pos[2] + 4.5 * (me.s or 1.0))
    dead = dz / math.tan(min_pitch) - C.DEADZONE_MARGIN if min_pitch > 0 else 0.0
    d_base = M.hdist(me.pos, base)
    on_roof = False
    try:
        on_roof = ctx.mapdata.roof_id(me.pos[0], me.pos[1]) == rid
    except Exception:
        pass
    if on_roof:
        m.legs(ctx, "R10.2.top", M.unit(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]), 1.0, None)
        return "R10.2.top"
    if d_base > max(dead, C.HOLD_ARRIVE_R):                               # 1) approach via cover to the dead zone
        m.walk_to(ctx, "assault", "R10.2.approach", base, 1.0, juke_ok=True)
        m.tac.pop("wait_t", None)
        return "R10.2.approach"
    need = C.CLIMB_COST_PER_M * roof.h + C.CLIMB_COST_PAD + C.ROOF_ASSAULT_RESERVE
    if "superconductor" in (me.upgrades or []):
        need -= C.ROOF_ASSAULT_SC_DISCOUNT
    if me.energy >= need or m.tac.get("climbing"):                       # 2) climb (jump, jets in air)
        m.tac["climbing"] = True
        toward = M.unit(base[0] - me.pos[0], base[1] - me.pos[1]) or M.unit(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1])
        m.legs(ctx, "R10.2.climb", toward, 1.0, None, juke_ok=False)
        if me.grounded:
            ctx.intent.jump.append(JumpRequest("ROOF_ASSAULT", "R10.2"))
        elif me.pos[2] < roof.top + C.CLIMB_CLEAR_M and me.energy > C.CLIMB_ABORT_ENERGY:
            ctx.intent.boost.append(BoostRequest("CLIMB", "R10.2", True, 0.0))
        if me.energy <= C.CLIMB_ABORT_ENERGY:
            m.tac["climbing"] = False
        return "R10.2.climb"
    w0 = m.tac.setdefault("wait_t", t)                                   # 3) wait at the base (silent)
    m.legs(ctx, "R10.2.wait", None)
    if t - w0 >= C.ROOF_WAIT_MAX_S:
        m.roof_wait_expired = True                                       # 4) skip: X2 exit with (FIGHT, it)
    return "R10.2.wait"


def run(sub: str, ctx) -> str:
    if sub == "ROOF_ASSAULT":
        return _assault(ctx)
    return _hold(ctx)
