"""Overlays O1-O9 (plan 5.4, 6.2; CONTRACTS.md 6.8): reflexes that constrain legs/aim/boost and never
change the macro mode or the goal.

    ov = Overlays(C, mapdata, nav, rng)
    ostate = ov.update(world, A, nav, intent, clock, macro=modes.macro, sub=modes.sub, last_move=...)

Appends to `intent`:
  LegFilter  L2 (O2 mortar dodge), L4 (O1 zone, only when no L2 and no L3/L3b; D1 collapse variant),
             O8 (keep away, scope "low"), O9 (coast safe: circle clause "low", roof clause "all")
  LegRequest L3 (O3 cover dash), L3b (O4 blind-hit cover), L4 (O1 field descent)
  AimRequest AIM1, AIM2, AIM4 (O3), AIM7 (O4), AIM8 (sound LOOK, R4.1/H11)
  BoostRequest O1, O3, O4, O6, L2;  JumpRequest O5;  intent.juke (O7);  intent.unstick (O5)
  FireRequest F3.2 (lightning at a laser charger) and R9.6 (laser release when only full speed clears)
Scope rules use the previous tick's macro (`macro`): X3 disables O4 legs; X4 disables O4 legs, O8, AIM4
and AIM8 and limits O1's boost to "away from the point". The R4.1 sound-response matrix lives here;
its per-ghost result is published in `OverlayState.responses` for the goal selector (M3A, M8).

LegFilter.keep(dir, speed) receives a unit direction and the owner's speed scale; the "stop"
candidate is passed as dir=(0.0, 0.0), speed=0.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from contracts import (AIM_PRIO, BOOST_ORDER, LEG_OWNER_ORDER, AimRequest, BoostRequest, FireRequest,
                       JukeStyle, JumpRequest, LegFilter, LegRequest)

INF = float("inf")
KIND_ORDER = {"mortar": 0, "laser_charge": 1, "lightning": 2, "gun": 3}
BAND_ORDER = {"near": 0, "mid": 1, "far": 2}


def _wrap(a: float) -> float:
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


def _hd(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _has(track, kind, conf=0.0) -> bool:
    if kind in (getattr(track, "weapons", ()) or ()):
        return True
    c = (getattr(track, "known_weapons", None) or {}).get(kind, 0.0)
    return c >= conf if conf > 0 else c > 0


@dataclass
class OverlayState:
    active: Dict[str, float]            # overlay id ("O1".."O9") -> since t
    cover_target: Optional[Tuple[float, float]]   # L3/L3b cover cell
    zone_escape: bool
    keep_away_contacts: List[int]
    responses: Dict[int, frozenset] = field(default_factory=dict)   # R4.1 per ghost id
    look: Optional[Tuple[float, float, int]] = None                 # (bearing, until, ghost id) AIM8


def r41_response(kind: str, band: str, pb: str, C, *, t: float, energy: float, terrain_call: str,
                 ghost_dist: float = INF, launch_recent: bool = False) -> frozenset:
    """R4.1 response matrix (sounds not associated with a visible track)."""
    L = pb == "PB_L"
    window = (C.PHASE_T3_SHRINK <= t < C.PHASE_T4_SHRINK and terrain_call == "LIGHTNING") or \
        (C.INVESTIGATE_ANYTIME and t < C.ENDGAME_T)
    inv = {"INVESTIGATE"} if window else {"LOOK", "HEAT"}
    near, mid = band == "near", band == "mid"
    r: set = set()
    if kind == "gun":
        if near:
            r = ({"AMBUSH"} if energy >= C.RUSH_ENERGY_MIN else {"LOOK"}) if L else {"LOOK", "AVOID"}
        elif mid:
            r = inv if L else {"LOOK", "HEAT"}
        else:
            r = {"HEAT"}
    elif kind in ("laser", "laser_charge"):
        if near:
            r = {"LOOK", "AVOID"}
            if L and ghost_dist <= C.LIGHTNING_RANGE + C.AMBUSH_SEARCH_R:
                r.add("AMBUSH")
        elif mid:
            r = {"LOOK", "AVOID"}
        else:
            r = {"HEAT"}
    elif kind == "lightning":
        if near:
            r = {"LOOK"} if L else {"AVOID"}
        elif mid:
            r = inv if L else {"AVOID"}
    elif kind == "mortar":
        r = {"WARY", "LOOK"} if near else ({"WARY"} if mid else {"HEAT"})
    elif kind == "explosion":
        r = {"WARY"} if near else ({"WARY"} if mid and launch_recent else set())
    elif kind in ("step", "boost"):
        if near:
            r = {"AMBUSH"} if L else {"LOOK", "AVOID"}
        elif mid and kind == "boost":
            r = {"LOOK"}
    elif kind == "jet":
        r = {"LOOK"} if (near or mid) else set()
    elif kind == "reload":
        r = {"LOOK"} if near else set()
    elif kind == "crate":
        r = ({"AMBUSH"} if L else {"HEAT"}) if near else ({"HEAT"} if mid else set())
    elif kind == "npc_gun":
        r = {"HEAT"} if near else ((inv if L else {"HEAT"}) if mid else set())
    elif kind == "death":
        r = {"HEAT"} if (near or mid) else set()
    return frozenset(r)


class Overlays:
    def __init__(self, C, mapdata, nav, rng) -> None:
        self.C, self.map, self.nav, self.rng = C, mapdata, nav, rng
        self.active: Dict[str, float] = {}
        self._o1 = False
        self._o3_cover = None          # (point, since, charger id) H7
        self._o4_cover = None          # (point, since, hit t)
        self._juke: Optional[JukeStyle] = None
        self._juke_next = -INF
        self._juke_last_trigger = -INF
        self._o8_out_since: Dict[int, float] = {}
        self._o8_on: Dict[int, float] = {}
        self._still_since: Optional[float] = None
        self._last_move = None
        self._look = None
        self._look_hist: Dict[int, float] = {}
        self._resp: Dict[int, Tuple[frozenset, float]] = {}
        self._launch_t = -INF
        self._o6 = False

    def note_action(self, action: dict) -> None:
        """Main calls this with the last sent action (O5 compares commanded vs actual motion)."""
        self._last_move = (action or {}).get("move")

    # ------------------------------------------------------------------ helpers
    def _speed(self, me, scale: float, charging: bool) -> float:
        ws = float((me.stats or {}).get("walk_speed", self.C.path("mech.walk_speed", 11.7)))
        v = scale * ws
        if charging:
            v *= float(self.C.path("mech.charge_move_mult", 0.5))
        return v

    def _set(self, oid: str, on: bool, t: float) -> None:
        if on:
            self.active.setdefault(oid, t)
        else:
            self.active.pop(oid, None)

    # ------------------------------------------------------------------ main
    def update(self, world, A, nav, intent, clock, macro: str = "X1", sub: str = "", last_move=None) -> OverlayState:
        me, t = world.me, world.tick.t
        self.nav = nav
        if last_move is not None:
            self._last_move = last_move
        charging = any(w is not None and w.kind == "laser" and (w.charge or 0) > 0 for w in (me.weapons or ()))
        st = OverlayState(self.active, None, False, [])
        l2 = self._o2(world, A, intent, t, charging)                  # L2 filter (and D1 handling)
        l3 = self._o3(world, A, intent, clock, t, macro, st)           # L3 owner + AIM1/2/4 + F3.2
        l3b = self._o4(world, A, intent, clock, t, macro, st)          # L3b owner + AIM7
        self._o1_update(world, A, intent, t, macro, st, l2, l3 or l3b)
        self._o8(world, A, intent, t, macro, st)
        self._o9(world, intent, t)
        self._o7(world, A, intent, t, macro, sub, charging, l2 or l3 or l3b or self._o1)
        self._o5(world, intent, t, charging)
        self._o6_update(world, intent, t)
        self._sounds(world, A, intent, t, macro, st)
        st.active = dict(self.active)
        return st

    # ------------------------------------------------------------------ O2 MORTAR_DODGE (L2, R9.6, D1)
    def _o2(self, world, A, intent, t, charging) -> bool:
        C, me = self.C, world.me
        pos, vel = me.pos, me.vel
        shells = list(getattr(A.threats, "shells", None) or world.shells or [])
        thr = []
        for s in shells:
            tau = s.t_impact - t
            if tau < 0:
                continue
            pred = (pos[0] + vel[0] * tau, pos[1] + vel[1] * tau)
            if _hd(pred, s.impact) < C.MORTAR_DODGE_R:
                thr.append((s.impact, tau))
        if me.speed_h < C.STILL_SPEED:
            self._still_since = self._still_since if self._still_since is not None else t
        else:
            self._still_since = None
        wary = (getattr(world, "mortar_active_until", 0.0) > t and self._still_since is not None
                and t - self._still_since > C.MORTAR_STILL_MAX_S)
        if not thr and not wary:
            self._set("O2", False, t)
            return False
        self._set("O2", True, t)
        if not thr:                                    # wary mode: keep moving
            intent.leg_filters.append(LegFilter("L2", "O2.wary", lambda d, s: s > 0 and (d[0] or d[1])))
            return True
        dirs = _compass(int(C.IMPL_LEG_DIRS))

        def clears(v):
            return [d for d in dirs if all(_hd((pos[0] + d[0] * v * tau, pos[1] + d[1] * v * tau), I) >= C.MORTAR_DODGE_R
                                           for I, tau in thr)]

        mult = {"v_scale": 1.0}
        use_charge = charging
        if charging and not clears(self._speed(me, 1.0, True)) and clears(self._speed(me, 1.0, False)):
            # R9.6 (b): only full speed clears -> release the laser this tick, L2 uses full speed
            slot = next(i for i, w in enumerate(me.weapons) if w is not None and w.kind == "laser")
            intent.fire.append(FireRequest(slot, "R9.6", None, hold=False, release=True))
            use_charge = False
        if not clears(self._speed(me, 1.0, use_charge)):
            bs = float((me.stats or {}).get("boost_speed", C.path("mech.boost_speed", 19.5)))
            ws = float((me.stats or {}).get("walk_speed", C.path("mech.walk_speed", 11.7)))
            if me.energy > 0 and clears(bs):
                mult["v_scale"] = bs / ws
                intent.boost.append(BoostRequest("L2", "O2", True, 0.0))
        nav = self.nav

        def end(d, s):
            v = self._speed(me, max(s, 0.0) * mult["v_scale"], use_charge)
            return [(pos[0] + d[0] * v * tau, pos[1] + d[1] * v * tau, I) for I, tau in thr]

        def keep(d, s):
            return all(_hd(e[:2], e[2]) >= C.MORTAR_DODGE_R for e in end(d, s))

        def maxmin(d, s):
            return min(_hd(e[:2], e[2]) for e in end(d, s))

        def tieb(d, s):
            e = end(d, s)[0]
            try:
                return float(nav.zone_eta((e[0], e[1], pos[2]))[0])
            except Exception:
                return 0.0

        # D1 (ZONE_OUTRANKS_DODGE): in COLLAPSE (or always), omit the L2 hard filter if every candidate the
        # dodge keeps would leave the safe set; the zone then wins through an L4 filter.
        mode = C.ZONE_OUTRANKS_DODGE
        if mode == "always" or (mode == "collapse" and t >= C.FINAL_SHRINK_T):
            zone = world.zone
            tau0 = max(tau for _, tau in thr)
            c, r = zone.center_at(t + tau0), zone.radius_at(t + tau0)
            safe_r = max(r, C.ZONE_TINY_R)
            kept = [d for d in dirs + [(0.0, 0.0)] if keep(d, 1.0 if (d[0] or d[1]) else 0.0)]
            if not any(_hd(end(d, 1.0 if (d[0] or d[1]) else 0.0)[0][:2], c) <= safe_r for d in kept):
                def zkeep(d, s, c=c, safe_r=safe_r):
                    v = self._speed(me, max(s, 0.0), use_charge)
                    return _hd((pos[0] + d[0] * v * tau0, pos[1] + d[1] * v * tau0), c) <= safe_r
                intent.leg_filters.append(LegFilter("L4", "D1", zkeep, tieb))
                intent.trace.append("D1")
                return False                                   # no L2 this tick
        intent.leg_filters.append(LegFilter("L2", "O2", keep, tieb, "all", maxmin))
        return True

    # ------------------------------------------------------------------ O3 LASER_THREAT (L3, AIM1/2/4, F3.2)
    def _o3(self, world, A, intent, clock, t, macro, st) -> bool:
        C, me = self.C, world.me
        charge = float(C.path("weapons.laser.charge", 0.75))
        hold_s = float(C.path("weapons.laser.hold_s", 2.0))
        chargers = []
        for cid in (getattr(A.threats, "laser_chargers", None) or ()):
            tr = world.tracks.get(cid)
            ta = A.per_track.get(cid)
            if tr is None or ta is None or not tr.visible or not ta.charging_at_us:
                continue
            since = tr.charging_since if tr.charging_since is not None else t
            if t - since > charge + hold_s:                    # O3 timeout
                continue
            chargers.append((max(0.0, charge - (t - since)), ta.d, cid, tr, ta))
        if not chargers:
            self._set("O3", False, t)
            self._o3_cover = None
            return False
        chargers.sort(key=lambda c: (c[0], c[1], c[2]))
        self._set("O3", True, t)
        rem, d, cid, tr, ta = chargers[0]
        # aim (6.4): AIM1 > AIM2 > AIM4
        held = {w.kind: i for i, w in enumerate(me.weapons or ()) if w is not None}
        yaw = math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0])
        err = abs(_wrap(yaw - me.aim[0]))
        turn = float((me.stats or {}).get("turn_rate", C.path("mech.turn_rate", 4.0))) or 4.0
        d3 = math.sqrt(d * d + (tr.pos[2] - me.pos[2]) ** 2)
        pitch = math.atan2(tr.pos[2] + 4.5 - (me.muzzle_z or me.pos[2] + 7.0), max(0.5, d))
        tb = rem + d * 1e-4
        if "lightning" in held and d3 <= C.LIGHTNING_RANGE and err / turn <= rem:
            intent.aim.append(AimRequest(AIM_PRIO["AIM1"], "F3.2", yaw, pitch, cid, tb))      # AIM1
            intent.fire.append(FireRequest(held["lightning"], "F3.2", cid))
        else:
            front = math.radians(float(C.path("mech.front_arc_deg", 180.0))) / 2
            if err > front:
                tgt = A.per_track.get(A.target_id) if A.target_id is not None else None
                finish = macro == "X2" and tgt is not None and tgt.ttk_us_on_j < rem
                if not finish:
                    intent.aim.append(AimRequest(AIM_PRIO["AIM2"], "AIM2", yaw, pitch, cid, tb))
            elif macro not in ("X2", "X4"):
                intent.aim.append(AimRequest(AIM_PRIO["AIM4"], "AIM4", yaw, pitch, cid, tb))
        # legs: urgent cover query (3.3) reachable before the remaining charge
        pt = None
        if self._o3_cover is not None and self._o3_cover[2] == cid and t - self._o3_cover[1] < C.COVER_MIN_HOLD_S:
            pt = self._o3_cover[0]
        else:
            try:
                pt = self.nav.cover_query(me, [(tr.pos[0], tr.pos[1], tr.pos[2] + float(C.path("mech.eye_height", 8.0)))],
                                          rem, clock.slice(C.COVER_URGENT_MS))
            except Exception:
                pt = None
            self._o3_cover = (pt, t, cid) if pt is not None else None
        if pt is None:
            return False
        dvec = _unit(pt[0] - me.pos[0], pt[1] - me.pos[1])
        intent.legs.append(LegRequest("L3", "O3", dvec, 1.0, (pt[0], pt[1]), juke_ok=False, is_cover_dash=True))
        intent.boost.append(BoostRequest("O3", "O3", True, C.BOOST_FLOOR_ESCAPE))
        st.cover_target = (pt[0], pt[1])
        return True

    # ------------------------------------------------------------------ O4 BLIND_HIT (L3b, R3.1, AIM7)
    def _o4(self, world, A, intent, clock, t, macro, st) -> bool:
        C, me = self.C, world.me
        hits = [h for h in (getattr(A.threats, "unseen_hits", None) or ()) if t - h.t <= C.UNSEEN_REACT_S]
        hits = [h for h in hits if h.by is not None and not (world.tracks.get(h.by) is not None and world.tracks[h.by].visible)]
        if not hits:
            self._set("O4", False, t)
            self._o4_cover = None
            return False
        self._set("O4", True, t)
        h = max(hits, key=lambda r: r.t)
        b = h.bearing if h.bearing is not None else me.aim[0]
        intent.aim.append(AimRequest(AIM_PRIO["AIM7"], "R3.1", b, 0.0, h.by, -h.t))     # AIM7 (also in X4)
        held = {w.kind: i for i, w in enumerate(me.weapons or ()) if w is not None}
        pb = me.playbook
        if h.weapon == "lightning" and pb == "PB_L" and "lightning" in held:
            est = C.INFER_RANGE_LIGHTNING * C.TESLA_MUL ** C.TESLA_MAX_STACKS
            if est <= C.LIGHTNING_RANGE or C.INFER_RANGE_LIGHTNING <= C.LIGHTNING_RANGE:
                intent.fire.append(FireRequest(held["lightning"], "R3.1", None))
        if macro in ("X3", "X4"):                        # legs yield to FLEE; X4: aim only
            return False
        bv = (math.cos(b), math.sin(b))
        dvec, boost, cover_r = None, False, None
        if h.weapon in ("gun", "laser"):
            rng = C.INFER_RANGE_GUN if h.weapon == "gun" else C.INFER_RANGE_LASER
            att = (me.pos[0] + bv[0] * rng, me.pos[1] + bv[1] * rng, me.pos[2] + 8.0)
            pt = None
            if self._o4_cover is not None and t - self._o4_cover[1] < C.COVER_MIN_HOLD_S:
                pt = self._o4_cover[0]
            else:
                ws = float((me.stats or {}).get("walk_speed", 11.7)) or 11.7
                try:
                    pt = self.nav.cover_query(me, [att], C.COVER_SEARCH_R / ws, clock.slice(C.COVER_URGENT_MS))
                except Exception:
                    pt = None
                self._o4_cover = (pt, t, h.t) if pt is not None else None
            if pt is not None:
                dvec = _unit(pt[0] - me.pos[0], pt[1] - me.pos[1])
                cover_r = _hd(me.pos, pt)
                st.cover_target = (pt[0], pt[1])
            boost = h.weapon == "laser"
        elif h.weapon == "mortar":
            side = 1
            try:
                w1 = self.map.wall5(me.pos[0] - bv[1] * 10, me.pos[1] + bv[0] * 10)
                w2 = self.map.wall5(me.pos[0] + bv[1] * 10, me.pos[1] - bv[0] * 10)
                side = 1 if w1 <= w2 else -1
            except Exception:
                pass
            dvec = (-side * bv[1], side * bv[0])                     # perpendicular, toward cover
        elif h.weapon == "lightning":
            if pb != "PB_L":
                dvec = (-bv[0], -bv[1])                              # move away from the bearing
        elif h.weapon == "npc":
            tr = world.tracks.get(h.by)
            home = (tr.home or tr.pos) if tr is not None else (me.pos[0] + bv[0] * 30, me.pos[1] + bv[1] * 30)
            if _hd(me.pos, home) < C.NPC_AGGRO_R:
                dvec = _unit(me.pos[0] - home[0], me.pos[1] - home[1])
        if dvec is None:
            return False
        juke = cover_r is None or cover_r > C.COVER_DASH_R
        intent.legs.append(LegRequest("L3b", "R3.1", dvec, 1.0, None, juke_ok=juke and h.weapon == "gun",
                                      is_cover_dash=cover_r is not None))
        if boost:
            intent.boost.append(BoostRequest("O4", "R3.1", True, C.BOOST_FLOOR_ESCAPE))
        return True

    # ------------------------------------------------------------------ O1 ZONE_ESCAPE (L4)
    def _safe_set_dist(self, world) -> float:
        C, z, me = self.C, world.zone, world.me
        r = max(C.ZONE_TINY_R, z.next_radius - C.ZONE_EDGE_MARGIN_M)
        return max(0.0, _hd(me.pos, z.next_center) - r)

    def _o1_update(self, world, A, intent, t, macro, st, l2, l3) -> None:
        C, me = self.C, world.me
        inside = bool(me.in_zone)
        if self._o1:
            self._o1 = not (inside and A.slack >= C.ZONE_SLACK_HARD_S + C.ZONE_ESCAPE_EXIT_S)
        else:
            self._o1 = A.slack < C.ZONE_SLACK_HARD_S or not inside
        self._set("O1", self._o1, t)
        st.zone_escape = self._o1
        if not self._o1:
            return
        d = None
        try:
            d = self.nav.zone_descent(me.pos)
        except Exception:
            d = None
        if d is None:
            z = world.zone
            c = z.center_at(t) if not inside else z.next_center
            d = _unit(c[0] - me.pos[0], c[1] - me.pos[1])
        intent.legs.append(LegRequest("L4", "O1", d, 1.0, None, juke_ok=False))
        if d is not None and not l2 and not l3:
            cosmax = math.cos(math.radians(C.ZONE_FILTER_DEG))
            nav, pos = self.nav, me.pos

            def keep(c, s, d=d):
                return (c[0] or c[1]) and (c[0] * d[0] + c[1] * d[1]) >= cosmax - 1e-9

            def tieb(c, s):
                try:
                    return float(nav.zone_eta((pos[0] + c[0] * 4.0, pos[1] + c[1] * 4.0, pos[2]))[0])
                except Exception:
                    return 0.0
            intent.leg_filters.append(LegFilter("L4", "O1", keep, tieb))
        eta = max(0.0, A.zone_eta)
        want = me.energy > C.BOOST_FLOOR_TRAVEL or world.zone.dps * eta >= C.ZONE_OUT_DMG_MAX_FRAC * me.hp
        if macro == "X4" and self._safe_set_dist(world) <= C.FINAL_POINT_TOL_M:
            want = False                                  # X4 scope: no boost at the point
        intent.boost.append(BoostRequest("O1", "O1", want, 0.0))

    # ------------------------------------------------------------------ O8 KEEP_AWAY (filter "low"; R2.3, R4.1 AVOID)
    def _o8(self, world, A, intent, t, macro, st) -> None:
        C, me = self.C, world.me
        if macro == "X4":                                 # off in X4
            self._o8_on.clear()
            self._set("O8", False, t)
            return
        pb = me.playbook
        contacts: List[Tuple[int, Tuple[float, float], float]] = []   # (id, pos, radius)
        if pb in ("PB_Z", "PB_G"):
            for tr in world.tracks.values():
                if tr.kind != "mech" or t - tr.last_seen_t > C.TRACK_FORGET_S:
                    continue
                prior = not tr.weapons      # loadout never seen: R2.2 worst case (prior-only -> O8 only)
                if _has(tr, "lightning") or prior:
                    contacts.append((tr.id, (tr.pos[0], tr.pos[1]), C.LIGHTNING_SAFE_R * (tr.lightning_range_mult or 1.0)))
            gm = C.TESLA_MUL ** C.TESLA_MAX_STACKS
            for gh in world.ghosts.values():
                if gh.kinds.get("lightning", 0) > 0:
                    contacts.append((gh.id, gh.pos, C.LIGHTNING_SAFE_R * gm))
            for gid, (resp, _) in self._resp.items():
                gh = world.ghosts.get(gid)
                if gh is not None and "AVOID" in resp and all(c[0] != gid for c in contacts):
                    contacts.append((gid, gh.pos, INF))
        live = {}
        for cid, p, r in contacts:
            if _hd(me.pos, p) <= r:
                live[cid] = p
                self._o8_out_since.pop(cid, None)
                self._o8_on.setdefault(cid, t)
            elif cid in self._o8_on:
                out = self._o8_out_since.setdefault(cid, t)
                if t - out < C.KITE_EXIT_S:
                    live[cid] = p
                else:
                    self._o8_on.pop(cid, None)
                    self._o8_out_since.pop(cid, None)
        for cid in list(self._o8_on):
            if all(c[0] != cid for c in contacts):          # ghost expired / track forgotten
                self._o8_on.pop(cid, None)
                self._o8_out_since.pop(cid, None)
        self._set("O8", bool(live), t)
        st.keep_away_contacts = sorted(live)
        if not live:
            return
        pos = me.pos
        pts = list(live.values())

        def keep(d, s, intent=intent):
            if intent.macro == "X4":
                return True
            v = self._speed(me, max(s, 0.0), False)
            e = (pos[0] + d[0] * v, pos[1] + d[1] * v)
            return all(_hd(e, p) >= _hd(pos, p) - 1e-6 for p in pts)
        intent.leg_filters.append(LegFilter("O8", "O8", keep, None, "low"))

    # ------------------------------------------------------------------ O9 COAST_SAFE (R15.2b)
    def _o9(self, world, intent, t) -> None:
        C, me, z = self.C, world.me, world.zone
        coast = float(C.path("bot.coast_s", 1.0))
        pos = me.pos
        c1, r1 = z.center_at(t), z.radius_at(t)
        c2, r2 = z.center_at(t + coast), z.radius_at(t + coast)
        if _hd(pos, c1) <= r1:
            def keep_circle(d, s):
                v = self._speed(me, max(s, 0.0), False)
                return _hd((pos[0] + d[0] * v * coast, pos[1] + d[1] * v * coast), c2) <= r2
            intent.leg_filters.append(LegFilter("O9", "R15.2b", keep_circle, None, "low"))
        try:
            rid = self.map.roof_id(pos[0], pos[1])
        except Exception:
            rid = 0
        if rid:
            cos_tol = math.cos(math.pi / max(4, int(C.IMPL_LEG_DIRS)))
            mp = self.map

            def keep_roof(d, s, intent=intent, rid=rid):
                if not (d[0] or d[1]):
                    return True
                own = _owner(intent)
                if own is not None and own.direction is not None and \
                        d[0] * own.direction[0] + d[1] * own.direction[1] >= cos_tol:
                    return True                                       # the planned path may leave the roof
                v = self._speed(me, max(s, 0.0), False)
                try:
                    return mp.roof_id(pos[0] + d[0] * v * coast, pos[1] + d[1] * v * coast) == rid
                except Exception:
                    return True
            intent.leg_filters.append(LegFilter("O9", "R15.2b", keep_roof, None, "all"))
        self._set("O9", True, t)

    # ------------------------------------------------------------------ O7 JUKE (R12.1, 8.10)
    def _o7(self, world, A, intent, t, macro, sub, charging, reflex) -> None:
        C, me = self.C, world.me
        trig = bool(getattr(A.threats, "bullets_inbound", None) or [])
        if not trig:
            for tr in world.visible_mechs():
                ta = A.per_track.get(tr.id)
                d = ta.d if ta is not None else _hd(me.pos, tr.pos)
                if _has(tr, "gun") and C.JUKE_MIN_RANGE <= d <= C.JUKE_MAX_RANGE:
                    trig = True
                    break
        if trig:
            self._juke_last_trigger = t
        on = (t - self._juke_last_trigger) <= C.JUKE_EXIT_S
        if macro == "X2" and sub == "RUSH" and A.target_id in A.per_track and A.per_track[A.target_id].d <= C.LIGHTNING_RUSH_R:
            on = False
        if not on or charging or reflex:
            self._set("O7", False, t)
            return
        self._set("O7", True, t)
        if self._juke is None or t >= self._juke_next:
            side = 1 if self.rng.random() < 0.5 else -1
            ang = self.rng.uniform(C.JUKE_ANGLE_MIN_DEG, C.JUKE_ANGLE_MAX_DEG) + C.JUKE_ANGLE_PER_CUBE * me.cubes
            ang = min(ang, C.JUKE_ANGLE_CAP_DEG)
            self._juke = JukeStyle(side, math.radians(ang), "R12.1")
            self._juke_next = t + self.rng.uniform(C.JUKE_PERIOD_MIN_S, C.JUKE_PERIOD_MAX_S) * (me.s or 1.0)
        intent.juke = self._juke

    # ------------------------------------------------------------------ O5 NAV_RECOVER (R13.1)
    def _o5(self, world, intent, t, charging) -> None:
        if charging:
            self._set("O5", False, t)
            return
        cmd = tuple(self._last_move) if self._last_move else None
        try:
            leg, jump, boost = self.nav.stuck_update(world, cmd)
        except Exception:
            leg = jump = boost = None
        if leg is not None:
            intent.unstick = leg
        if jump is not None:
            intent.jump.append(JumpRequest("O5", jump.rule or "R13.1", jump.want))
        if boost is not None:
            owner = boost.owner if boost.owner in BOOST_ORDER else "CLIMB"
            intent.boost.append(BoostRequest(owner, boost.rule or "R13.1", boost.want, boost.floor))
        self._set("O5", leg is not None or jump is not None, t)

    # ------------------------------------------------------------------ O6 FALL_BRAKE (A-1, D9; off by default)
    def _o6_update(self, world, intent, t) -> None:
        C, me = self.C, world.me
        if not C.FALL_BRAKE_ENABLED:
            self._o6 = False
            return
        if me.grounded or me.vel[2] > C.FALL_BRAKE_EXIT_VZ:
            self._o6 = False
        else:
            try:
                h = me.pos[2] - self.map.surface_h(me.pos[0], me.pos[1])
            except Exception:
                h = INF
            if me.vel[2] < C.FALL_BRAKE_VZ and h < C.FALL_SAFE_DROP_M and me.energy >= C.FALL_BRAKE_MIN_ENERGY:
                self._o6 = True
        self._set("O6", self._o6, t)
        if self._o6:
            intent.boost.append(BoostRequest("O6", "O6", True, 0.0))

    # ------------------------------------------------------------------ sounds: R4.1 matrix, AIM8 LOOK (H11)
    def _sounds(self, world, A, intent, t, macro, st) -> None:
        C, me = self.C, world.me
        pb = me.playbook
        tc = getattr(A, "terrain_call", "MIXED")
        for s in world.sounds or ():
            if s.kind == "mortar":
                self._launch_t = t
        cands = []
        for s in world.sounds or ():
            if s.track_id is not None:
                tr = world.tracks.get(s.track_id)
                if tr is not None and tr.visible:
                    continue
            gid = s.ghost_id
            gh = world.ghosts.get(gid) if gid is not None else None
            gd = _hd(me.pos, gh.pos) if gh is not None else INF
            resp = r41_response(s.kind, s.band, pb, C, t=t, energy=me.energy, terrain_call=tc, ghost_dist=gd,
                                launch_recent=t - self._launch_t <= C.EXPLOSION_LINK_S)
            if gid is not None:
                prev = self._resp.get(gid)
                if prev is None or t - prev[1] >= C.SOUND_REACT_AGE_S:
                    self._resp[gid] = (resp, t)
            if "LOOK" in resp and s.band in ("near", "mid") and s.age <= C.SOUND_REACT_AGE_S:
                cands.append((KIND_ORDER.get(s.kind, 9), BAND_ORDER.get(s.band, 9), s.age, s.bearing, gid))
        for gid in list(self._resp):
            if gid not in world.ghosts or t - self._resp[gid][1] > C.GHOST_TTL_S:
                del self._resp[gid]
        st.responses = {g: r for g, (r, _) in self._resp.items()}
        if self._look is not None and t >= self._look[1]:
            self._look = None
        if cands:
            cands.sort()
            for k, b, age, brg, gid in cands:
                key = gid if gid is not None else -10**6
                if t - self._look_hist.get(key, -INF) < C.LOOK_REPEAT_S:
                    continue
                if self._look is None or (k, b) < self._look[3]:
                    self._look = (brg, t + C.SCAN_LOOK_S, key, (k, b))
                    self._look_hist[key] = t
                break
        if self._look is not None and macro != "X4":
            st.look = self._look[:3]
            intent.aim.append(AimRequest(AIM_PRIO["AIM8"], "AIM8", self._look[0], 0.0, None, 0.0))


def _owner(intent) -> Optional[LegRequest]:
    best = None
    for lr in intent.legs:
        if lr.level in LEG_OWNER_ORDER:
            i = LEG_OWNER_ORDER.index(lr.level)
            if best is None or i < best[0]:
                best = (i, lr)
    return best[1] if best else None


def _compass(n: int):
    return [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
