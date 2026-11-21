"""PB_L FIGHT sub-states RUSH, BRAWL, STALL (GP "Lightning playbook"; plan R0.3 RL1-RL6, R9.2).

run(sub, ctx) writes the X2 owner LegRequest (L5), AIM3, FireRequests and the MODE boost; returns the
deciding rule id. Sub transitions: RUSH -> BRAWL at LIGHTNING_RANGE (RL1); RUSH -> STALL when the gap
closes slower than RUSH_CLOSING_MIN over RUSH_STALL_S of boosting (RL5); STALL -> BRAWL at
LIGHTNING_RANGE. RL2 (don't chase) and the timeouts are modes' X2 own exits.
"""
from __future__ import annotations

import math

from contracts import JukeStyle

import modes as M


def _r92(ctx) -> bool:
    """R9.2 empty energy: lightning off below LIGHTNING_MIN_ENERGY until + ENERGY_RESUME_MARGIN."""
    C, me, tac = ctx.C, ctx.world.me, ctx.modes.tac
    low = tac.get("r92", False)
    if me.energy < C.LIGHTNING_MIN_ENERGY:
        low = True
    elif me.energy >= C.LIGHTNING_MIN_ENERGY + C.ENERGY_RESUME_MARGIN:
        low = False
    tac["r92"] = low
    return low


def run(sub: str, ctx) -> str:
    m, C, world, A = ctx.modes, ctx.C, ctx.world, ctx.A
    me, t = world.me, world.tick.t
    tr = world.tracks.get(m.target) if m.target is not None else None
    if tr is None:
        m.legs(ctx, "RL1", None)
        return "RL1"
    ta = A.per_track.get(tr.id)
    d = ta.d if ta is not None else M.hdist(me.pos, tr.pos)
    low = _r92(ctx)
    kinds = {"gun"} if low else {"gun", "lightning", "mortar"}
    if sub == "RUSH" and d <= C.LIGHTNING_RANGE:
        sub = m.sub = "BRAWL"                                        # RL1: both weapons at LIGHTNING_RANGE
        m._log(t, "sub", "RUSH", "BRAWL", tr.id, "RL1")
    if sub == "STALL" and d <= C.LIGHTNING_RANGE:
        sub = m.sub = "BRAWL"
        m._log(t, "sub", "STALL", "BRAWL", tr.id, "RL5")
    m.aim_track(ctx, "AIM3", "AIM3", tr)
    bvec = M.unit(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]) or (1.0, 0.0)
    if sub == "RUSH":
        # RL5: gap closing slower than RUSH_CLOSING_MIN over RUSH_STALL_S -> STALL
        hist = m.tac.setdefault("gap", [])
        hist.append((t, d))
        while hist and t - hist[0][0] > C.RUSH_STALL_S + 1e-9:
            hist.pop(0)
        if hist and t - hist[0][0] >= C.RUSH_STALL_S - 1e-6 and (hist[0][1] - d) < C.RUSH_CLOSING_MIN * C.RUSH_STALL_S:
            m.sub = "STALL"
            m._log(t, "sub", "RUSH", "STALL", tr.id, "RL5")
            return run("STALL", ctx)
        # straight line, A* if blocked (RL1)
        direction = bvec
        try:
            if not ctx.nav.clear_dir(me.pos, bvec, C.LEG_PROBE_M):
                direction = m.go(ctx, "x2", (tr.pos[0], tr.pos[1])).direction
        except Exception:
            pass
        gun_holder = M.has_weapon(tr, "gun")
        juke_ok = gun_holder and d > max(C.JUKE_MIN_RANGE, C.LIGHTNING_RUSH_R)
        m.legs(ctx, "RL1", direction, 1.0, (tr.pos[0], tr.pos[1]), juke_ok=juke_ok)
        if juke_ok and ctx.intent.juke is not None:                  # closing juke angle (8.10, RL1)
            ctx.intent.juke = JukeStyle(ctx.intent.juke.side, math.radians(C.JUKE_CLOSE_ANGLE_DEG), "RL1")
        m.boost(ctx, "RL1", not low, C.BOOST_FLOOR_FIGHT)
        m.fire(ctx, tr.id, kinds)
        return "RL1"
    if sub == "STALL":
        # RL5: stop boosting, gun only, regen; walk
        m.legs(ctx, "RL5", bvec, 1.0, (tr.pos[0], tr.pos[1]))
        m.boost(ctx, "RL5", False, C.BOOST_FLOOR_FIGHT)
        m.fire(ctx, tr.id, {"gun"})
        return "RL5"
    # BRAWL: hold BRAWL_DIST, circle-strafe
    side = m.tac.get("side")
    if side is None or t - m.tac.get("side_t", t) > 2.0:
        side = 1 if ctx.rng.random() < 0.5 else -1
        m.tac["side"], m.tac["side_t"] = side, t
    tang = M.perp(bvec, side)
    radial = max(-1.0, min(1.0, (d - C.BRAWL_DIST) / max(1.0, C.BRAWL_DIST)))
    direction = M.unit(tang[0] + radial * bvec[0], tang[1] + radial * bvec[1])
    m.legs(ctx, "RL1.BRAWL", direction, 1.0, None, juke_ok=False)
    m.boost(ctx, "RL1.BRAWL", False, C.BOOST_FLOOR_FIGHT)
    m.fire(ctx, tr.id, kinds)
    return "RL1.BRAWL"
