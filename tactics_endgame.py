"""X4 ENDGAME: FINAL_APPROACH, FINAL_HOLD, COLLAPSE (GP "375 s+: stop fighting"; plan 7.11 R11.1-R11.5).

R11.2 final point: next_center on walkable ground; inside a building footprint the roof above it when
climbable (height <= CLIMB_MAX_H with FINAL_ROOF_ENERGY_MARGIN, width >= ROOF_MIN_W, A-11), else the walkable
cell nearest the center. Re-evaluated on every next_changed (modes.strategic_refresh clears it).
The point never moves off the center for another mech: in the last shrink the circle loses 1 m of radius per
second, so each metre off the center is a second of zone damage (260 HP), and a mech that was only standing
there when the point was first computed would cost that for the rest of the match. A mech on the point is
walked into instead: overlapping mechs are pushed apart (protocol 7), which leaves us as close as the
server allows and on the point as soon as it moves or dies.
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
            # the same test as _ring: with less room than our radius the body cannot be centred there, it
            # stops short of FINAL_POINT_TOL_M and FINAL_APPROACH would push at the wall for the rest of X4
            if mp.clearance(c[0], c[1]) < _stand_r(me):
                pt = _nearest_walkable(ctx, c)
        except Exception:
            pass
    m.final_point = pt
    return pt


def _stand_r(me) -> float:
    """Clearance a ground point needs for our body to be centred on it."""
    return me.radius or 2.5


_WALK_STEP_M = 0.5        # final ring spacing: half a metre off the center is half a second of zone damage
_WALK_COARSE_M = 2.0      # first pass, to keep the search to about a millisecond
_WALK_MAX_M = 60.0


def _ring(ctx, c, r):
    """Ground cell on the ring of radius r around c where our body can stand (clearance >= our radius:
    a point nearer the wall cannot be reached, and FINAL_HOLD would never start), nearest to us; or None."""
    mp, me = ctx.mapdata, ctx.world.me
    need = _stand_r(me)
    k_n = max(8, min(48, int(2 * math.pi * r / _WALK_STEP_M)))
    best = None
    for k in range(k_n):
        a = 2 * math.pi * k / k_n
        x, y = c[0] + r * math.cos(a), c[1] + r * math.sin(a)
        if mp.clearance(x, y) >= need and not mp.roof_id(x, y):
            dd = M.hdist(me.pos, (x, y))
            if best is None or dd < best[0]:
                best = (dd, (x, y))
    return None if best is None else best[1]


def _nearest_walkable(ctx, c):
    """Walkable ground cell nearest the center: 2 m rings find the distance, 0.5 m rings refine it."""
    try:
        r = _WALK_COARSE_M
        hit = None
        while r <= _WALK_MAX_M and hit is None:
            hit = _ring(ctx, c, r)
            r += _WALK_COARSE_M
        if hit is None:
            return (c[0], c[1])
        r_hit = r - _WALK_COARSE_M
        r = r_hit - _WALK_COARSE_M + _WALK_STEP_M
        while r < r_hit - 1e-9:
            fine = _ring(ctx, c, r)
            if fine is not None:
                return fine
            r += _WALK_STEP_M
        return hit
    except Exception:
        return (c[0], c[1])


_CONTACT_PAD_M = 1.0      # touching = centres within both radii plus this


def _held_off(ctx, pt, dist) -> bool:
    """A visible mech stands nearer the point than we do and we are touching it: this is as close as the
    server lets us get (overlapping mechs are pushed apart), so it counts as arrived. Without this the
    bot never reached FINAL_HOLD against an occupant and boosted into it from FINAL_ARRIVE_T on."""
    C, world = ctx.C, ctx.world
    me = world.me
    base = float(C.path("mech.radius", 2.5) or 2.5)
    k = float(C.path("cubes.size_per_cube", 0.05) or 0.0)
    r_me = me.radius or base
    for tr in world.visible_mechs():
        r_o = base * (1.0 + k * (tr.cubes or 0))
        if M.hdist(tr.pos, pt) < dist and M.hdist(tr.pos, me.pos) <= r_me + r_o + _CONTACT_PAD_M:
            return True
    return False


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


def _contact_m(ctx) -> float:
    """Centre distance at which we touch the biggest mech in sight (one of the base size when none is):
    how far out a mech that stands on the point keeps us."""
    C, world = ctx.C, ctx.world
    base = float(C.path("mech.radius", 2.5) or 2.5)
    k = float(C.path("cubes.size_per_cube", 0.05) or 0.0)
    r_o = max([base * (1.0 + k * (tr.cubes or 0)) for tr in world.visible_mechs()] or [base])
    return (world.me.radius or base) + r_o + _CONTACT_PAD_M


def _held_by_the_pile(ctx, pt, dist) -> bool:
    """_held_off by a mech that holds the point: one our body would touch if we stood on the point, or one
    that touches such a mech from further out, and so on outward (the crowd of the last shrink stands in
    rings around the point). Only then is where we are as close as the server lets us get. A mech we run
    into anywhere else holds nothing, and pressing into it is not the plan: O5's sidestep is what takes us
    round it (traced with one mech 15 m from a free point, straight in our way: without O5 the bot pressed
    into it 20 m from the point for the rest of the match)."""
    C, world = ctx.C, ctx.world
    me = world.me
    base = float(C.path("mech.radius", 2.5) or 2.5)
    k = float(C.path("cubes.size_per_cube", 0.05) or 0.0)
    r_me = me.radius or base
    pile = []                                                          # (pos, radius, distance to the point)
    for d_o, tr in sorted(((M.hdist(tr.pos, pt), tr) for tr in world.visible_mechs()), key=lambda e: e[0]):
        r_o = base * (1.0 + k * (tr.cubes or 0))
        # nearest the point first, so every member a mech can lean on is already in the list
        if d_o <= r_me + r_o + _CONTACT_PAD_M or \
                any(M.hdist(tr.pos, p) <= r_o + r_p + _CONTACT_PAD_M for p, r_p, _ in pile):
            pile.append((tr.pos, r_o, d_o))
    return any(d_p < dist and M.hdist(p, me.pos) <= r_me + r_p + _CONTACT_PAD_M for p, r_p, d_p in pile)


def _way_clear(ctx, pt, dist) -> bool:
    """No wall between us and the point: the arbiter's leg probe (nav.clear_dir: centre line and +-0.7 R),
    asked exactly as far as the point. The probe the arbiter runs itself looks LEG_PROBE_M past our rim,
    which is past a point that hugs a wall, and would drop the way to it."""
    me = ctx.world.me
    d = M.unit(pt[0] - me.pos[0], pt[1] - me.pos[1])
    if d is None:
        return True
    try:
        return bool(ctx.nav.clear_dir(me.pos, d, max(0.0, dist - _stand_r(me))))
    except Exception:
        return False


_HANDOVER_PAD_M = 1.0     # FINAL_HOLD is left this far beyond where the path follower handed over


def _hold_r(ctx, pt) -> float:
    """FINAL_HOLD is left beyond this distance: HOLD_ARRIVE_R, or where the follower stopped when that
    was further out (its path can end at the centre of the 4 m cell instead of on the point)."""
    C, mem = ctx.C, ctx.modes.tac.get("x4_hold")
    return max(C.HOLD_ARRIVE_R, mem[1]) if mem is not None and mem[0] == pt else C.HOLD_ARRIVE_R


def _servo(ctx, rule, pt, dist, held_off) -> str:
    """FINAL_HOLD and COLLAPSE legs: straight at the point, move = dir x min(1, dist x FINAL_MOVE_GAIN).
    Tells O5 when standing still under that command is the plan (nav.stuck_exempt): within HOLD_ARRIVE_R
    of the point; or with no wall between us and it, when we are no further out than one mech on the
    point keeps us, or the mechs that hold the point hold us off (_held_by_the_pile). Then only a mech
    can be what stops us, and the ladder's jump, sidestep, jet and relocation would each give up ground
    we hold (traced: a sidestep is 6 m, the relocation 10-20 m). With a wall on the way the detector
    stays on: that is terrain to get past. O5 decided this tick before us (main.Bot.act: overlays, then
    modes), on last tick's command; what we say here is about the command we give now, which it judges
    next tick."""
    m, C, me = ctx.modes, ctx.C, ctx.world.me
    clear = _way_clear(ctx, pt, dist)
    d = M.unit(pt[0] - me.pos[0], pt[1] - me.pos[1])
    m.legs(ctx, rule, d, min(1.0, dist * C.FINAL_MOVE_GAIN), pt, juke_ok=False, nav=clear)
    if dist <= C.HOLD_ARRIVE_R or (clear and (dist <= C.HOLD_ARRIVE_R + _contact_m(ctx)
                                              or (held_off and _held_by_the_pile(ctx, pt, dist)))):
        try:
            ctx.nav.stuck_exempt(ctx.world.tick.t)
        except Exception:
            pass
    return rule


def run(sub: str, ctx) -> str:
    m, C, world, A = ctx.modes, ctx.C, ctx.world, ctx.A
    me, t = world.me, world.tick.t
    pt = final_point(ctx)
    dist = M.hdist(me.pos, pt)
    held_off = _held_off(ctx, pt, dist)
    if t >= C.FINAL_SHRINK_T and sub != "COLLAPSE":
        m._log(t, "sub", sub, "COLLAPSE", None, "R11.5")
        sub = m.sub = "COLLAPSE"
    elif sub == "FINAL_APPROACH" and (dist <= C.FINAL_POINT_TOL_M or held_off):
        m._log(t, "sub", sub, "FINAL_HOLD", None, "R11.1")
        sub = m.sub = "FINAL_HOLD"
    elif sub == "FINAL_HOLD" and dist > _hold_r(ctx, pt) and not held_off:
        m._log(t, "sub", sub, "FINAL_APPROACH", None, "R11.1")
        sub = m.sub = "FINAL_APPROACH"
        m.tac.pop("x4_hold", None)
    _self_defense_aim(ctx)
    if sub == "COLLAPSE":                                              # R11.5: move = dir x min(1, dist x gain)
        return _servo(ctx, "R11.5", pt, dist, held_off)
    if sub == "FINAL_HOLD":
        return _servo(ctx, "R11.1", pt, dist, held_off)
    # FINAL_APPROACH (R11.1): via the zone field, then the path to the point
    st = m.go(ctx, "x4", pt)
    direction = st.direction
    nav_ok = st.rule != "8.9f"
    zone = False
    try:
        eta, _ = ctx.nav.zone_eta(me.pos)
    except Exception:
        eta = 0.0
    if A.slack < C.ZONE_SLACK_SOFT_S and eta > 0:
        zs = m.zone_go(ctx)
        if zs.direction is not None:
            direction = zs.direction
            nav_ok = zone = True
        elif zs.rule == "R13.2w":
            direction = None                                           # waiting at a climb's foot for its energy
            zone = True
    if st.arrived and not zone:
        # The follower is done WAYPOINT_REACH_M (2.5 m) from its goal, which is outside FINAL_POINT_TOL_M:
        # with nothing commanded after that, FINAL_APPROACH stood there until COLLAPSE. The servo of
        # FINAL_HOLD takes the rest.
        if dist > C.HOLD_ARRIVE_R:
            m.tac["x4_hold"] = (pt, dist + _HANDOVER_PAD_M)
        m._log(t, "sub", sub, "FINAL_HOLD", None, "R11.1")
        m.sub = "FINAL_HOLD"
        return _servo(ctx, "R11.1", pt, dist, held_off)
    m.legs(ctx, "R11.1", direction, 1.0, pt, nav=nav_ok)
    ws = float((me.stats or {}).get("walk_speed", 11.7)) or 11.7
    m.boost(ctx, "R11.1", dist > C.HOLD_ARRIVE_R and dist / ws > C.FINAL_ARRIVE_T - t, 0.0)
    return "R11.1"
