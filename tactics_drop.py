"""Goal M7 DROP: APPROACH, WATCH, CAPTURE (GP "only when nobody is near it"; plan 7.8 R8.1-R8.6, 8.6, H6).

go_test(ctx, drop) -> (go, hard_fail)        8.6 / R8.6
check_exit(ctx, gs) -> (why | None, until)   M7 own exits and timeout (5.5): until = exit-memory end or None
run(sub, ctx) -> rule                        R8.2 (APPROACH/WATCH), R8.5 (CAPTURE)

H6 re-checks of the go test apply in APPROACH and WATCH; inside CAPTURE the ring rules decide (R8.3/R8.4
occupancy, R8.5 damage abort), so an intruder near (not in) the ring does not end a committed capture.
CAPTURE walks the nav path until it is inside the ring (it used to walk a straight line).
DROP_TIMEOUT_S counts from the landing time (world keeps lands_at fixed once down).
"""
from __future__ import annotations

import math

from contracts import ExitMemory

import modes as M


def _walk(ctx) -> float:
    return float((ctx.world.me.stats or {}).get("walk_speed", ctx.C.path("mech.walk_speed", 11.7))) or 11.7


def _eta(ctx, dr) -> float:
    try:
        tt = ctx.nav.travel_time(dr.pos[0], dr.pos[1])
    except Exception:
        tt = None
    if tt is None:
        tt = M.hdist(ctx.world.me.pos, dr.pos) / _walk(ctx) * ctx.C.ZONE_FALLBACK_DETOUR_K
    return tt


def _contacts(ctx, max_age):
    world, t = ctx.world, ctx.world.tick.t
    out = [(tr.id, tr.pos, t - tr.last_seen_t) for tr in world.tracks.values()
           if tr.kind == "mech" and t - tr.last_seen_t < max_age]
    out += [(gh.id, gh.pos, t - gh.last_t) for gh in world.ghosts.values() if t - gh.last_t < max_age]
    return out


def go_test(ctx, dr):
    """8.6: (1) ETA <= DROP_MAX_ETA_S and slack after arrival >= ZONE_SLACK_SOFT_S; (2) hp; (3) rivals == 0;
    (4) R8.6 open landing; (5) t < ENDGAME_T. Returns (go, hard_fail)."""
    C, world, A, me = ctx.C, ctx.world, ctx.A, ctx.world.me
    t = world.tick.t
    eta = _eta(ctx, dr)
    t_ready = max(eta, dr.lands_in)
    walk = _walk(ctx)
    rivals = 0
    near_ring = False
    for cid, pos, age in _contacts(ctx, C.DROP_RIVAL_INFO_S):
        dd = M.hdist(pos, dr.pos)
        if dd / walk <= t_ready + C.DROP_RIVAL_ETA_MARGIN_S:
            rivals += 1
        elif cid >= 0 and dd <= C.DROP_RIVAL_R:
            rivals += 1
        if cid >= 0 and dd <= C.DROP_RIVAL_R:
            near_ring = True
    inside_next = M.hdist(dr.pos, world.zone.next_center) <= world.zone.next_radius - C.ZONE_EDGE_MARGIN_M
    slack_ok = inside_next or (A.slack - eta) >= C.ZONE_SLACK_SOFT_S
    ok = eta <= C.DROP_MAX_ETA_S and slack_ok
    ok = ok and me.hp >= C.DROP_MIN_HP_FRAC * me.max_hp
    ok = ok and rivals == 0
    if ok:                                                            # R8.6 open landing
        try:
            cells = ctx.mapdata.cover2_cells(dr.pos[0], dr.pos[1], C.DROP_OPEN_COVER_R)
            open_land = not any(bits & 0x3F for (_, _, bits) in cells)
        except Exception:
            open_land = False
        if open_land and any(M.hdist(p, dr.pos) <= C.DROP_RIVAL_R for _, p, _ in _contacts(ctx, C.DROP_RIVAL_INFO_S)):
            ok = False
    ok = ok and t < C.ENDGAME_T
    hard = (A.slack < C.ZONE_SLACK_HARD_S or me.hp < (C.DROP_MIN_HP_FRAC - C.DROP_HARD_HP_DROP) * me.max_hp
            or near_ring)
    return ok, hard


def _ring_mechs(ctx, dr) -> int:
    return sum(1 for tr in ctx.world.visible_mechs() if M.hdist(tr.pos, dr.pos) <= dr.radius)


def check_exit(ctx, gs):
    C, world, me = ctx.C, ctx.world, ctx.world.me
    t = world.tick.t
    did = gs.key
    if any(ev.get("type") == "drop" and ev.get("id") == did for ev in world.events or ()):
        return "captured", None
    dr = world.drops.get(did)
    if dr is None:
        return "gone", None
    abandon = t + C.DROP_ABANDON_COOLDOWN_S
    landed = dr.lands_in <= 0
    if landed:
        others = _ring_mechs(ctx, dr)
        prog = dr.progress
        last = gs.data.get("other_prog")
        gs.data["other_prog"] = prog if (dr.holder is not None and dr.holder != me.id) else None
        if others >= 1:
            return ("contested" if others >= 2 else "occupied"), abandon               # R8.3 / R8.4
        if dr.holder is not None and dr.holder != me.id and last is not None and prog > last:
            return "occupied", abandon
        # our capture clock is running: let it finish. check_exit runs before run() in a decision, so prog_t is
        # last tick's; a first CAPTURE tick has none (counts as fresh). Frozen for max(DROP_STALL_S, capture_s)
        # (the 1 s probe alone is too tight for coarse progress updates) means something else holds the clock.
        stalled = t - gs.data.get("prog_t", t) >= max(C.DROP_STALL_S, dr.capture_s)
        ours = dr.holder == me.id and prog > 0.0 and not stalled
        if t - dr.lands_at >= C.DROP_TIMEOUT_S and not ours:
            return "timeout", ExitMemory.PERMANENT
    sub = ctx.modes.sub
    if sub == "CAPTURE":
        dmg = world.damage_since(gs.data.get("cap_t", t))
        if dmg > C.DROP_ABORT_DMG_FRAC * me.max_hp and dr.progress < C.DROP_COMMIT_PROGRESS_S:
            return "abort", abandon                                                    # R8.5
        return None, None
    if t - gs.data.get("last_check", -1e9) >= C.DROP_RECHECK_S:                         # H6
        gs.data["last_check"] = t
        ok, hard = go_test(ctx, dr)
        if hard:
            return "nogo_hard", abandon
        gs.data["fails"] = 0 if ok else gs.data.get("fails", 0) + 1
        if gs.data["fails"] >= C.DROP_FAIL_N:
            return "nogo", abandon
    return None, None


def _watch_point(ctx, dr):
    """R8.2: cover cell within DROP_WATCH_R of the ring with LOS to it, nearest to us; open terrain: the ring
    edge point toward us at DROP_WATCH_R."""
    C, me, mp = ctx.C, ctx.world.me, ctx.mapdata
    try:
        cells = mp.cover2_cells(dr.pos[0], dr.pos[1], C.DROP_WATCH_R)
    except Exception:
        cells = []
    best = None
    for (x, y, bits) in cells[:C.COVER_URGENT_CANDIDATES]:
        if not (bits & 0x3F) or M.hdist((x, y), dr.pos) <= dr.radius:
            continue
        try:
            z = mp.surface_h(x, y)
            if not mp.los((x, y, z + 8.0), (dr.pos[0], dr.pos[1], dr.pos[2] + 1.0)):
                continue
        except Exception:
            continue
        dd = M.hdist(me.pos, (x, y))
        if best is None or dd < best[0]:
            best = (dd, (x, y))
    if best is not None:
        return best[1]
    u = M.unit(me.pos[0] - dr.pos[0], me.pos[1] - dr.pos[1]) or (1.0, 0.0)
    return (dr.pos[0] + u[0] * C.DROP_WATCH_R, dr.pos[1] + u[1] * C.DROP_WATCH_R)


def run(sub: str, ctx) -> str:
    m, C, world, A = ctx.modes, ctx.C, ctx.world, ctx.A
    me, t, gs = world.me, world.tick.t, m.gs
    dr = world.drops.get(gs.key) if gs is not None else None
    if dr is None:
        m.legs(ctx, "R8.2", None)
        return "R8.2"
    ring_aim = M.bearing(me.pos, dr.pos)
    clear = not any(M.hdist(tr.pos, dr.pos) <= C.DROP_CLEAR_R and A.per_track.get(tr.id) is not None
                    and A.per_track[tr.id].los for tr in world.visible_mechs())
    quiet = not any(s.band == "near" for s in (world.sounds or ())) and t - gs.data.get("near_sound_t", -1e9) >= C.DROP_QUIET_S
    if any(s.band == "near" for s in (world.sounds or ())):
        gs.data["near_sound_t"] = t
    ready = dr.lands_in <= C.DROP_ENTER_LEAD_S and clear and quiet
    # CAPTURE commits when the drop is down and the ring is clear and quiet, wherever we are. A gate on
    # distance was tried on 2026-11-21 and cost captures (41 -> 30 in 90 matches): commitment is what wins
    # drops. What was wrong was the walk, a straight line into the nearest wall; CAPTURE now follows the
    # nav path until it is inside the ring, and DROP_TIMEOUT_S ends a capture that cannot be reached.
    if sub in ("APPROACH", "WATCH") and ready:
        sub = m.sub = "CAPTURE"
        gs.data["cap_t"] = t
        m._log(t, "sub", "WATCH", "CAPTURE", dr.id, "R8.2")
    if sub == "APPROACH":
        wp = gs.data.get("watch")
        if wp is None:
            wp = gs.data["watch"] = _watch_point(ctx, dr)
        st = m.walk_to(ctx, "m7", "R8.2", wp)
        if dr.lands_in < _eta(ctx, dr):
            m.boost(ctx, "R8.2", me.energy >= C.CONTEST_BOOST_MIN_ENERGY, 0.0)
        if st.arrived or M.hdist(me.pos, wp) <= C.HOLD_ARRIVE_R:
            m.sub = "WATCH"
            m._log(t, "sub", "APPROACH", "WATCH", dr.id, "R8.2")
        m.aim_dir(ctx, "AIM9", "R8.2", ring_aim)
        return "R8.2"
    if sub == "WATCH":
        m.legs(ctx, "R8.2", None)
        m.aim_dir(ctx, "AIM9", "R8.2", ring_aim)
        return "R8.2"
    # CAPTURE (R8.5): stay inside radius - DROP_RING_MARGIN, strafe randomly inside
    r_in = max(0.5, dr.radius - C.DROP_RING_MARGIN)
    dd = M.hdist(me.pos, dr.pos)
    if dd > dr.radius:                                                 # still outside the ring: follow the path
        m.walk_to(ctx, "m7", "R8.5", (dr.pos[0], dr.pos[1]))
    else:
        if dd > r_in * 0.6:
            direction = M.unit(dr.pos[0] - me.pos[0], dr.pos[1] - me.pos[1])
        else:
            if t - gs.data.get("strafe_t", -1e9) >= 0.5:
                gs.data["strafe_t"] = t
                gs.data["strafe_a"] = ctx.rng.uniform(-math.pi, math.pi)
            direction = M.yaw_vec(gs.data.get("strafe_a", 0.0))
        m.legs(ctx, "R8.5", direction, 1.0, (dr.pos[0], dr.pos[1]))
    mt = world.tracks.get(A.main_threat_id) if A.main_threat_id is not None else None
    if mt is not None and mt.visible:
        m.aim_track(ctx, "AIM9", "R8.5", mt)
    for tr in world.visible_mechs():                                   # F1-F4 at mechs entering DROP_CLEAR_R with LOS
        ta = A.per_track.get(tr.id)
        if ta is not None and ta.los and ta.d <= C.DROP_CLEAR_R:
            m.fire(ctx, tr.id, None, "")
            break
    # A-6: progress stalls with only an NPC inside -> kill it (R7.2 gates relaxed)
    prog = dr.progress
    if prog > gs.data.get("prog", -1.0):                               # rose since last tick (a reset restarts at 0)
        gs.data["prog_t"] = t
    gs.data["prog"] = prog
    if t - gs.data.get("prog_t", t) >= C.DROP_STALL_S:
        for tr in world.tracks.values():
            if tr.kind == "npc" and tr.visible and M.hdist(tr.pos, dr.pos) <= dr.radius:
                m.aim_track(ctx, "AIM9", "R8.npc", tr)
                m.fire(ctx, tr.id, {"gun", "lightning"}, allow_npc=True)
                break
    return "R8.5"
