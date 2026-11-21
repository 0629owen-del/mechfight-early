"""Actuator resolution (plan 6.3-6.7; CONTRACTS.md section 5 and 6.10): the only code that turns an
`Intent` into an action.

    arb = Arbiter(C, mapdata)                 # optional: rng=, aim_mod=; main sets arb.runner_info
    action = arb.resolve(world, A, nav, intent, clock)

One owner per actuator (principle 3), recorded in `arb.last_owners` (telemetry / tests):
  legs   owner = first LEG_OWNER_ORDER level present; candidates d_own + IMPL_LEG_DIRS compass + stop,
         clearance probe, filters in LEG_FILTER_ORDER ("low" scope only under X4/L5-L8, a filter that
         would remove every candidate is ignored, L2 keeps its max-min argmax), rank by 6.3 step 2,
         ties: L2 tiebreak, closest to d_own, lower compass index; O5 unstick replaces d_own (only L2/L4
         filters); O7 juke rotation re-checked against the applied filters
  aim    lowest prio, lowest tiebreak, first appended; else AIM10 scan (aim.Scanner)
  fire   per slot through aim's F1-F4 gates; F5 self-defense filter; R11.5 clock guard; laser R9.5/R9.6
         safe window from `runner_info` (missed tick / hard overrun block starts; soft overruns block
         long holds only)
  reload intent.reload or F6;  boost BOOST_ORDER + O9 veto;  jump JUMP_ORDER;  take/drop TAKE_ORDER;  pick
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from contracts import AIM_PRIO, BOOST_ORDER, JUMP_ORDER, LEG_FILTER_ORDER, LEG_OWNER_ORDER, TAKE_ORDER, FireRequest

INF = float("inf")
LOW_SCOPE_OWNERS = ("X4", "L5", "L6", "L7", "L8")
COAST_VETO_EXEMPT = ("O1", "O6", "CLIMB")
_IMPLICIT_RULE = {"gun": "F1", "laser": "F2", "lightning": "F3", "mortar": "F4"}


def _wrap(a: float) -> float:
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def _hd(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_mech(tid) -> bool:
    return tid is not None and 0 <= int(tid) < 100


class _FallbackAim:
    """Minimal F-gates used only when `aim` cannot be imported (R15.1: aim/fire fallback = nearest
    visible mech with the F1 gate)."""

    def gun_lead(self, me, tr, gun, C):
        d = max(0.1, _hd(me.pos, tr.pos))
        yaw = math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0])
        return yaw, math.atan2(tr.pos[2] + 4.5 - (me.muzzle_z or me.pos[2] + 7.0), d), 0.0

    def gun_gate(self, me, tr, gun, aim_target, C):
        d = max(0.1, _hd(me.pos, tr.pos))
        tol = math.atan(C.GUN_FIRE_TOL * (2.5 + float(C.path("projectile.gun_radius", 0.3))) / d)
        err = math.hypot(_wrap(me.aim[0] - aim_target[0]), me.aim[1] - aim_target[1])
        ok_ammo = (gun.ammo is None or gun.ammo > 0) and not gun.reload
        return err < tol and d <= float(gun.stats.get("range", 230)) and ok_ammo

    def laser_point(self, me, tr, C):
        y, p, _ = self.gun_lead(me, tr, None, C)
        return y, p

    def laser_decision(self, me, tr, laser, los_now, los_breaks_next, safe_window_ok, C, t=None):
        full = float(laser.stats.get("charge", C.path("weapons.laser.charge", 0.75)))
        if laser.charge > 0:
            return True, laser.charge >= full
        return bool(tr is not None and los_now and safe_window_ok), False

    def lightning_gate(self, world, mapdata, slot, C, allow_npc=False):
        me = world.me
        best = None
        for tr in world.visible_mechs():
            d3 = math.sqrt(_hd(me.pos, tr.pos) ** 2 + (tr.pos[2] - me.pos[2]) ** 2)
            err = abs(_wrap(math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0]) - me.aim[0]))
            if d3 <= C.LIGHTNING_RANGE and err < math.radians(C.LIGHTNING_CONE_MARGIN * 15.0):
                if best is None or d3 < best[0]:
                    best = (d3, tr.id)
        ok = best is not None and me.energy >= float(C.path("weapons.lightning.energy", 3))
        return ok, (best[1] if best else None)

    def mortar_solution(self, me, mapdata, target, target_vel, mortar, C):
        return None

    def mortar_gate(self, me, solution, target_pred, mortar, C):
        return False


def _load_aim():
    try:
        import aim as _aim  # noqa: WPS433
        return _aim
    except Exception:
        return _FallbackAim()


class Arbiter:
    def __init__(self, C, mapdata, rng=None, aim_mod=None) -> None:
        self.C, self.map = C, mapdata
        self.rng = rng if rng is not None else random.Random(0)
        self.aim = aim_mod if aim_mod is not None else _load_aim()
        self.scanner = None
        self.runner_info = None          # main: arb.runner_info = bot.runner_info (R9.6 safe window)
        self.degrade_level = 0           # R15.3 (main keeps it current)
        self.last_action: Dict[str, Any] = {}
        self.last_owners: Dict[str, Any] = {}
        self.blocked_desire = None       # owner's wanted move when the clearance probe forced "stop" (O5 input)
        self._last_missed_tick = -10 ** 9
        self._laser_target: Optional[int] = None
        self._compass = None

    # ------------------------------------------------------------------ main
    def resolve(self, world, A, nav, intent, clock) -> dict:
        owners: Dict[str, Any] = {}
        if world.tick.missed_ticks > 0:
            self._last_missed_tick = world.tick.tick
        move, owners["legs"] = self._legs(world, A, nav, intent)
        aim_req = self._aim_owner(intent)
        if aim_req is not None:
            aim = [aim_req.yaw, aim_req.pitch]
            owners["aim"] = aim_req.rule
        else:
            aim = self._scan(world, move)
            owners["aim"] = "AIM10"
        fire, owners["fire"] = self._fire(world, A, intent, aim_req)
        reload, owners["reload"] = self._reload(world, intent)
        boost, owners["boost"] = self._boost(world, intent)
        jump, owners["jump"] = self._jump(intent)
        action: Dict[str, Any] = {"move": [move[0], move[1]], "boost": boost, "jump": jump, "aim": aim,
                                  "fire": fire, "reload": reload}
        take, drop, owners["take"] = self._take(world, intent)
        if take is not None:
            action["take"] = take
        if drop is not None:
            action["drop"] = drop
        if intent.pick is not None:
            action["pick"] = intent.pick
            owners["pick"] = intent.pick_rule or "R14.1"
        self.last_owners = owners
        self.last_action = action
        return action

    # ------------------------------------------------------------------ legs (6.3)
    def _legs(self, world, A, nav, intent):
        C, me = self.C, world.me
        owner = None
        for lr in intent.legs:
            if lr.level in LEG_OWNER_ORDER and (owner is None or
                                                LEG_OWNER_ORDER.index(lr.level) < LEG_OWNER_ORDER.index(owner.level)):
                owner = lr
        level = owner.level if owner is not None else None
        d_own = owner.direction if owner is not None else None
        speed = owner.speed if owner is not None else 1.0
        unstick = intent.unstick
        if unstick is not None:
            d_own = unstick.direction
            speed = unstick.speed or speed
        if d_own is not None and not (d_own[0] or d_own[1]):
            d_own = None
        mspeed = speed if speed > 0 else 1.0
        n = int(C.IMPL_LEG_DIRS)
        if self._compass is None or len(self._compass) != n:
            self._compass = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        cands: List[tuple] = []          # (dir, idx, speed)
        if d_own is not None:
            cands.append((d_own, -1, mspeed))
        for k, c in enumerate(self._compass):
            cands.append((c, k, mspeed))
        stop = ((0.0, 0.0), n, 0.0)
        pos = me.pos
        moving = [c for c in cands if self._clear(nav, pos, c[0])]
        cands = (moving if moving else cands) + [stop]
        # hard filters in LEG_FILTER_ORDER
        order = {lv: i for i, lv in enumerate(LEG_FILTER_ORDER)}
        filters = sorted((f for f in intent.leg_filters if f.level in order), key=lambda f: order[f.level])
        applied = []
        l2_tiebreak = None
        for f in filters:
            if unstick is not None and f.level not in ("L2", "L4"):
                continue
            if f.scope == "low" and level not in LOW_SCOPE_OWNERS and level is not None:
                continue
            kept = [c for c in cands if self._keep(f, c)]
            if kept:
                cands = kept
                applied.append(f)
            elif f.fallback_max_min is not None:
                cands = [max(cands, key=lambda c: (f.fallback_max_min(c[0], c[2]), -c[1]))]
                applied.append(f)
            else:
                continue                                     # would remove every candidate: ignored
            if f.level == "L2" and f.tiebreak is not None:
                l2_tiebreak = f.tiebreak
        # rank (6.3 step 2)
        d_field = None
        if A.slack < C.ZONE_SLACK_SOFT_S:
            try:
                d_field = nav.zone_descent(pos)
            except Exception:
                d_field = None
        hostile_dirs = []
        if me.playbook in ("PB_Z", "PB_G"):
            for tr in world.visible_mechs():
                if _hd(pos, tr.pos) < C.ARB_HOSTILE_R:
                    hostile_dirs.append(math.atan2(tr.pos[1] - pos[1], tr.pos[0] - pos[0]))
        hdeg = math.radians(C.ARB_HOSTILE_DIR_DEG)

        def score(c):
            d = c[0]
            if c[1] == n:
                s = 1.0 if d_own is None else 0.0
            else:
                s = (d[0] * d_own[0] + d[1] * d_own[1]) if d_own is not None else 0.0
                if d_field is not None:
                    s += C.ARB_W_ZONE * (d[0] * d_field[0] + d[1] * d_field[1])
                a = math.atan2(d[1], d[0])
                if any(abs(_wrap(a - h)) <= hdeg for h in hostile_dirs):
                    s -= C.ARB_W_HOSTILE
                try:
                    if world.heat_at(pos[0] + d[0] * 5.0, pos[1] + d[1] * 5.0) > C.ARB_HEAT_MAX:
                        s -= C.ARB_W_HEAT
                except Exception:
                    pass
            return s

        def ang(c):
            if d_own is None:
                return 0.0 if c[1] == n else math.pi
            if c[1] == n:
                return math.pi
            return math.acos(max(-1.0, min(1.0, c[0][0] * d_own[0] + c[0][1] * d_own[1])))

        scored = [(score(c), c) for c in cands]
        top = max(s for s, _ in scored)
        ties = [c for s, c in scored if s >= top - C.IMPL_SCORE_EPS]

        def tkey(c):
            tb = l2_tiebreak(c[0], c[2]) if l2_tiebreak is not None else 0.0
            return (tb, ang(c), c[1])
        win = min(ties, key=tkey)
        d, sp = win[0], win[2]
        # 6.3 "blocked ones dropped ...; then O5 handles it": the owner wants to move, its direction failed the
        # clearance probe and every clear candidate points away, so "stop" won. Report the wanted vector so
        # O5's stuck detector (commanded vs actual) sees the block instead of a voluntary stop (deadlock).
        self.blocked_desire = None
        if win[1] == n and d_own is not None and unstick is None and not self._clear(nav, pos, d_own):
            self.blocked_desire = (d_own[0] * mspeed, d_own[1] * mspeed)
        rule = (owner.rule if owner is not None else "none")
        # O7 juke (8.10): rotate, re-check filters, fail -> no juke
        if (intent.juke is not None and owner is not None and owner.juke_ok and not owner.is_cover_dash
                and unstick is None and win[1] != n and level not in ("L3", "L3b", "L4")):
            a = intent.juke.side * intent.juke.angle
            ca, sa = math.cos(a), math.sin(a)
            r = (d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca)
            rc = (r, win[1], sp)
            if self._clear(nav, pos, r) and all(self._keep(f, rc) for f in applied):
                d = r
                rule += "+" + intent.juke.rule
        owner_tag = (level or "none") + ":" + rule + (":unstick" if unstick is not None else "")
        return (d[0] * sp, d[1] * sp), owner_tag

    def _clear(self, nav, pos, d) -> bool:
        try:
            return bool(nav.clear_dir(pos, d, self.C.LEG_PROBE_M))
        except Exception:
            return True

    @staticmethod
    def _keep(f, c) -> bool:
        try:
            return bool(f.keep(c[0], c[2]))
        except Exception:
            return True

    # ------------------------------------------------------------------ aim (6.4)
    @staticmethod
    def _aim_owner(intent):
        best = None
        for i, r in enumerate(intent.aim):
            k = (r.prio, r.tiebreak, i)
            if best is None or k < best[0]:
                best = (k, r)
        return best[1] if best else None

    def _scan(self, world, move) -> list:
        try:
            if self.scanner is None:
                self.scanner = self.aim.Scanner(self.C, self.rng)
            y, p = self.scanner.scan(world, self.map)
            return [y, p]
        except Exception:
            if move[0] or move[1]:
                return [math.atan2(move[1], move[0]), 0.0]
            return [world.me.aim[0], 0.0]

    # ------------------------------------------------------------------ fire (6.6)
    def _laser_windows(self, world):
        """(safe_start, safe_long): missed ticks / hard overruns / degrade >= 1 block starts; soft overruns
        additionally block long holds (R9.5, R9.6, F2)."""
        C = self.C
        per = int(C.path("match.decide_every", 2) or 2)
        win = C.LASER_SAFE_WINDOW_N * per
        tick = world.tick.tick
        ri = self.runner_info
        hard_t = ri.last_hard_overrun_tick if ri is not None else -10 ** 9
        soft_t = ri.last_soft_overrun_tick if ri is not None else -10 ** 9
        recent = lambda x: x >= 0 and tick - x < win  # noqa: E731
        safe_start = self.degrade_level < 1 and not recent(self._last_missed_tick) and not recent(hard_t)
        safe_long = safe_start and not recent(soft_t)
        return safe_start, safe_long

    def _fire(self, world, A, intent, aim_req):
        C, me = self.C, world.me
        t = world.tick.t
        fire = [False, False]
        owners: List[Optional[str]] = [None, None]
        weapons = list(me.weapons or [])[:2] + [None] * max(0, 2 - len(me.weapons or []))
        if t >= C.CLOCK_HP_GUARD_T and world.tick.alive >= 2:            # R11.5: no firing at all
            for i, w in enumerate(weapons):
                if w is not None and w.kind == "laser" and w.charge > 0:
                    owners[i] = "R11.5"
            return fire, owners
        by_slot: Dict[int, list] = {}
        for fr in intent.fire:
            by_slot.setdefault(fr.slot, []).append(fr)
        aim_tid = aim_req.target_id if aim_req is not None else None
        # 6.6 F1-F4 fire "at the current aim owner's mech target (AIM1-AIM6)": a slot nobody asked for
        # follows the aim owner (X1 AIM6, X3 AIM5, O3 AIM2/AIM4); F5 still limits it to self-defense
        if aim_req is not None and aim_req.prio <= AIM_PRIO["AIM6"] and _is_mech(aim_tid):
            for slot, w in enumerate(weapons):
                if w is not None and slot not in by_slot:
                    by_slot[slot] = [FireRequest(slot, _IMPLICIT_RULE.get(w.kind, "F1"), aim_tid)]
        for slot, w in enumerate(weapons):
            if w is None:
                continue
            reqs = by_slot.get(slot, [])
            if intent.fire_self_defense_only:                            # F5
                reqs = [r for r in reqs if r.release or not _is_mech(r.target_id)
                        or world.damaged_us_recently(r.target_id, C.SELF_DEFENSE_S)]
            reqs.sort(key=lambda r: 0 if r.target_id == aim_tid else 1)
            try:
                if w.kind == "laser":
                    fire[slot], owners[slot] = self._laser(world, A, w, reqs, intent)
                elif not reqs:
                    continue
                elif w.kind == "gun":
                    fire[slot], owners[slot] = self._gun(world, w, reqs)
                elif w.kind == "lightning":
                    fire[slot], owners[slot] = self._lightning(world, w, slot, reqs, intent)
                elif w.kind == "mortar":
                    fire[slot], owners[slot] = self._mortar(world, w, reqs)
            except Exception:
                fire[slot], owners[slot] = False, "exc"
        return fire, owners

    def _gun(self, world, w, reqs):
        C, me = self.C, world.me
        for r in reqs:
            tid = r.target_id
            if tid is None:
                continue
            tr = world.tracks.get(tid)
            if tr is None:
                cr = world.crates.get(tid)
                if cr is None:
                    continue
                d = max(0.1, _hd(me.pos, cr.pos))
                yaw = math.atan2(cr.pos[1] - me.pos[1], cr.pos[0] - me.pos[0])
                tol = math.atan(C.GUN_FIRE_TOL * 1.5 / d)
                ok = (abs(_wrap(me.aim[0] - yaw)) < tol and d <= float(w.stats.get("range", 230))
                      and (w.ammo is None or w.ammo > 0) and not w.reload)
                if ok:
                    return True, r.rule
                continue
            yaw, pitch, _ = self.aim.gun_lead(me, tr, w, C)
            if self.aim.gun_gate(me, tr, w, (yaw, pitch), C):
                return True, r.rule
        return False, None

    def _lightning(self, world, w, slot, reqs, intent):
        C, me = self.C, world.me
        allow_npc = any(r.rule == "R7.2" for r in reqs)
        ok, tid = self.aim.lightning_gate(world, self.map, slot, C, allow_npc)
        if ok:
            if intent.fire_self_defense_only and _is_mech(tid) and not world.damaged_us_recently(tid, C.SELF_DEFENSE_S):
                return False, None
            return True, reqs[0].rule
        cost = float(w.stats.get("energy", C.path("weapons.lightning.energy", 3)))
        for r in reqs:                                       # R3.1: lightning along the bearing (PB_L, unseen)
            if r.rule == "R3.1" and me.energy >= cost and w.cooldown <= 0:
                return True, "R3.1"
        return False, None

    def _laser(self, world, A, w, reqs, intent):
        C, me = self.C, world.me
        t = world.tick.t
        charging = (w.charge or 0) > 0
        if charging and any(r.release for r in reqs):        # R9.6 (b): release now
            return False, "R9.6"
        reqs = [r for r in reqs if not r.release]
        if not charging and not reqs:
            return False, None
        tid = reqs[0].target_id if reqs else self._laser_target
        tr = world.tracks.get(tid) if tid is not None else None
        ta = A.per_track.get(tid) if tid is not None else None
        los = bool(ta is not None and ta.los)
        breaks = bool(getattr(ta, "los_breaks_next", False)) if ta is not None else False
        safe_start, safe_long = self._laser_windows(world)
        hold, release = self.aim.laser_decision(me, tr, w, los, breaks, safe_long if charging else safe_start, C, t=t)
        if charging:
            on = bool(hold and not release)
            return on, ("F2" if on else "F2:release")
        if hold and safe_start and t >= me.short_until and tr is not None:
            self._laser_target = tid
            return True, reqs[0].rule
        return False, None

    def _mortar(self, world, w, reqs):
        C, me = self.C, world.me
        for r in reqs:
            tr = world.tracks.get(r.target_id) if r.target_id is not None else None
            if tr is None or w.cooldown > 0:
                continue
            sol = self.aim.mortar_solution(me, self.map, tr.pos, tr.vel, w, C)
            if sol is not None and self.aim.mortar_gate(me, sol, tr.pos, w, C):
                return True, r.rule
        return False, None

    # ------------------------------------------------------------------ reload, boost, jump, take
    def _reload(self, world, intent):
        C, me = self.C, world.me
        if intent.reload:
            return True, "intent"
        for w in me.weapons or ():
            if w is not None and w.kind == "gun" and w.ammo is not None and not w.reload:
                mag = float(w.stats.get("mag", C.path("weapons.gun.mag", 12)))
                quiet = not [tr for tr in world.hostile_tracks(C.RELOAD_QUIET_S) if tr.kind == "mech"]
                if w.ammo < C.RELOAD_EARLY_FRAC * mag and quiet:
                    return True, "F6"
        return False, None

    def _boost(self, world, intent):
        C, me = self.C, world.me
        first: Dict[str, Any] = {}
        for b in intent.boost:
            first.setdefault(b.owner, b)
        for o in BOOST_ORDER:
            b = first.get(o)
            if b is None:
                continue
            on = bool(b.want and me.energy > b.floor)
            if on and me.energy < C.COAST_BOOST_MIN_ENERGY and o not in COAST_VETO_EXEMPT:
                return False, o + ":O9veto"                          # R15.2 (a)
            return on, o
        return False, None

    @staticmethod
    def _jump(intent):
        first = {}
        for j in intent.jump:
            first.setdefault(j.owner, j)
        for o in JUMP_ORDER:
            j = first.get(o)
            if j is not None:
                return bool(j.want), o
        return False, None

    def _take(self, world, intent):
        C, me = self.C, world.me
        first = {}
        for r in intent.take:
            first.setdefault(r.writer, r)
        for wtr in TAKE_ORDER:
            r = first.get(wtr)
            if r is None:
                continue
            if r.take is None:
                return None, r.drop, wtr                              # drop alone (R1.4 / D6 only)
            p = world.pickups.get(r.take)
            if p is None:
                return None, None, wtr + ":gone"
            if _hd(me.pos, p.pos) > float(C.path("loot.pickup_radius", 4.0)) or abs(p.pos[2] - me.pos[2]) > 4.0:
                return None, None, wtr + ":range"
            weapons = list(me.weapons or [])
            slot = r.drop
            if slot is None:
                slot = next((i for i, w in enumerate(weapons[:2]) if w is None), None)
            if slot is not None and slot < len(weapons):
                w = weapons[slot]
                if w is not None and w.kind == "laser" and (w.charge or 0) > 0:
                    return None, None, wtr + ":busy"
            return r.take, r.drop, wtr
        return None, None, None
