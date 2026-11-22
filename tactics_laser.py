"""PB_Z FIGHT sub-state DUEL (GP "Gun + laser playbook"; plan RZ1, RZ3, RZ4 from a roof, R9.3).

Hold LASER_HOLD_MIN-LASER_HOLD_MAX (RZ3 vs a mortar holder: LASER_VS_MORTAR_MIN-MAX): lateral moves
inside the band (O7 adds the juke), back up when closer, close only along covered paths (A* with exposure
cost). Laser F2 when no bullets are inbound or the target's gun reloads; gun F1. ZL inside
LIGHTNING_RANGE plays PB_L (BRAWL). On a roof the legs keep the roof position (RZ4: "laser street
crossers (X2 DUEL from the roof)").
"""
from __future__ import annotations

import modes as M
import tactics_lightning
import tactics_roof


def run(sub: str, ctx) -> str:
    m, C, world, A = ctx.modes, ctx.C, ctx.world, ctx.A
    me, t = world.me, world.tick.t
    tr = world.tracks.get(m.target) if m.target is not None else None
    if tr is None:
        m.legs(ctx, "RZ1", None)
        return "RZ1"
    ta = A.per_track.get(tr.id)
    d = ta.d if ta is not None else M.hdist(me.pos, tr.pos)
    held = M.held_slots(me)
    if "lightning" in held and d <= C.LIGHTNING_RANGE:               # ZL: PB_L inside LIGHTNING_RANGE
        return tactics_lightning.run("BRAWL", ctx)
    lo, hi, rule = C.LASER_HOLD_MIN, C.LASER_HOLD_MAX, "RZ1"
    if M.has_weapon(tr, "mortar", C.CONF_SOUND):                     # RZ3
        lo, hi, rule = C.LASER_VS_MORTAR_MIN, C.LASER_VS_MORTAR_MAX, "RZ3"
    m.aim_track(ctx, "AIM3", "AIM3", tr)
    bvec = M.unit(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]) or (1.0, 0.0)
    on_roof = False
    try:
        on_roof = bool(ctx.mapdata.roof_id(me.pos[0], me.pos[1]))
    except Exception:
        on_roof = False
    if on_roof:
        tactics_roof.roof_legs(ctx, "RZ4.DUEL")
        rule = "RZ4.DUEL"
    elif d < lo:
        m.legs(ctx, rule, (-bvec[0], -bvec[1]), 1.0, None)           # back up
    elif d > hi:
        st = m.go(ctx, "x2", (tr.pos[0], tr.pos[1]), exposure=True)  # close along covered paths
        m.legs(ctx, rule, st.direction, 1.0, (tr.pos[0], tr.pos[1]), nav=st.rule != "8.9f")
    else:
        side = m.tac.get("side")
        if side is None or t - m.tac.get("side_t", t) > 3.0:
            side = 1 if ctx.rng.random() < 0.5 else -1
            m.tac["side"], m.tac["side_t"] = side, t
        m.legs(ctx, rule, M.perp(bvec, side), 1.0, None)             # lateral inside the band
    kinds = {"gun", "mortar"}
    bullets = getattr(A.threats, "bullets_inbound", None) or []
    if not bullets or tr.reloading_until > t or M.laser_charging(me):
        kinds.add("laser")
    if "lightning" in held:
        kinds.add("lightning")
    m.fire(ctx, tr.id, kinds)
    return rule
