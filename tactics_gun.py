"""PB_G FIGHT sub-state DUEL_GUN (plan RG1, gap-fill) and the weapon-agnostic CORNERED sub-state (R6.2).

DUEL_GUN: band GUN_HOLD_MIN-GUN_HOLD_MAX with juke, cover-heavy approach; mortar F4 between
MORTAR_SELF_SAFE_M and MORTAR_MAX_R, preferred against roof targets (walls do not block the splash).
CORNERED: face the threat, fire everything (gun F1, laser early release via F2, lightning, mortar only
when the impact is >= MORTAR_SELF_SAFE_M: the F4 gate), strafe perpendicular at walk speed.
"""
from __future__ import annotations

import modes as M
from contracts import AIM_PRIO, AimRequest


def _mortar_aim(ctx, tr) -> bool:
    """AIM3 at the mortar ballistic solution (8.7); False if none."""
    m, C, me = ctx.modes, ctx.C, ctx.world.me
    held = M.held_slots(me)
    if "mortar" not in held:
        return False
    try:
        sol = m.aim.mortar_solution(me, ctx.mapdata, tr.pos, tr.vel, me.weapons[held["mortar"]], C)
    except Exception:
        sol = None
    if sol is None:
        return False
    ctx.intent.aim.append(AimRequest(AIM_PRIO["AIM3"], "RG1.F4", sol[0], sol[1], tr.id))
    return True


def run(sub: str, ctx) -> str:
    m, C, world, A = ctx.modes, ctx.C, ctx.world, ctx.A
    me, t = world.me, world.tick.t
    tr = world.tracks.get(m.target) if m.target is not None else None
    if tr is None:
        m.legs(ctx, "RG1" if sub != "CORNERED" else "R6.2", None)
        return "RG1" if sub != "CORNERED" else "R6.2"
    ta = A.per_track.get(tr.id)
    d = ta.d if ta is not None else M.hdist(me.pos, tr.pos)
    bvec = M.unit(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]) or (1.0, 0.0)
    side = m.tac.get("side")
    if side is None or t - m.tac.get("side_t", t) > (1.0 if sub == "CORNERED" else 3.0):
        side = 1 if ctx.rng.random() < 0.5 else -1
        m.tac["side"], m.tac["side_t"] = side, t
    if sub == "CORNERED":
        m.aim_track(ctx, "AIM3", "R6.2", tr)
        m.legs(ctx, "R6.2", M.perp(bvec, side), 1.0, None)
        m.fire(ctx, tr.id, None)
        return "R6.2"
    held = M.held_slots(me)
    use_mortar = "mortar" in held and C.MORTAR_SELF_SAFE_M <= d <= C.MORTAR_MAX_R and \
        (tr.on_roof or "gun" not in held or d > C.GUN_HOLD_MAX)
    if not (use_mortar and _mortar_aim(ctx, tr)):
        use_mortar = False
        m.aim_track(ctx, "AIM3", "AIM3", tr)
    if d < C.GUN_HOLD_MIN:
        m.legs(ctx, "RG1", (-bvec[0], -bvec[1]), 1.0, None)
    elif d > C.GUN_HOLD_MAX:
        st = m.go(ctx, "x2", (tr.pos[0], tr.pos[1]), exposure=True)
        m.legs(ctx, "RG1", st.direction, 1.0, (tr.pos[0], tr.pos[1]))
    else:
        m.legs(ctx, "RG1", M.perp(bvec, side), 1.0, None)
    m.fire(ctx, tr.id, {"mortar"} if use_mortar else {"gun", "laser", "lightning"})
    return "RG1"
