"""Macro modes, goal selector and exit memory (plan sections 5, 9; CONTRACTS.md 6.9).

    modes = Modes(C, mapdata, nav, rng)
    modes.strategic_refresh(world, assessor, nav, clock)      # 6.1 step 4 (also stores `assessor`)
    modes.decide(world, A, nav, ostate, intent, clock)        # 6.1 steps 7-8

decide() walks the macro transition rows T2-T6 of 5.6 top-down (T1/T7 are the runner's X5, T8 is
the constructor's X1), then the goal selector of 5.5 inside X1 (G1 stickiness, G2 reach override),
and writes exactly one owner LegRequest (X4, L5, L6, L7 or L8) plus aim (AIM3/5/6/9, X4 list), fire,
boost ("MODE"/"CLIMB"), jump ("CLIMB"/"ROOF_ASSAULT") and take requests. Every own exit and timeout
writes `exit_memory` (5.7, H12); every entry reads it. Timers never reset on re-selection (H13).

Rule ids appear in comments and in `intent.rule` / `intent.trace` (the telemetry label).
WIN/LOSE/CORNERED predicates come from `assessor.evaluate_fight` (8.2.3-8.2.6); this module applies
the T3/T4 exit-memory checks, the (e) bypass, the min holds (H1/H2), the committed-capture, D14
and D15 gates, and re-checks the (e)/(i) freshness rules defensively (they are cheap and idempotent).
Shared helpers used by the tactics_* modules live here (they receive the Modes object in TacticCtx).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from contracts import (AIM_PRIO, GOAL_NAMES, GOAL_ORDER, GOAL_ORDER_LOOT_WINDOW, AimRequest, BoostRequest,
                       ExitMemory, FightEval, FireRequest, JumpRequest, LegRequest, Steer, TakeRequest)

import tactics_drop  # noqa: E402  (tactics modules import this module lazily: no cycle at import time)
import tactics_endgame
import tactics_gun
import tactics_laser
import tactics_lightning
import tactics_roof

INF = float("inf")
OWNER_LEVELS = ("X4", "L5", "L6", "L7", "L8")      # the levels modes may own (exactly one per tick)
MACRO_LEVEL = {"X4": "X4", "X2": "L5", "X3": "L6"}
MECH_ID_MAX = 99                                   # protocol: mechs 0-15, NPCs 100+, crates 1000+
FIRE_RULE = {"gun": "F1", "laser": "F2", "lightning": "F3", "mortar": "F4"}
SUB_MODULE = {"RUSH": tactics_lightning, "BRAWL": tactics_lightning, "STALL": tactics_lightning,
              "DUEL": tactics_laser, "DUEL_GUN": tactics_gun, "CORNERED": tactics_gun,
              "ROOF_ASSAULT": tactics_roof, "ROOF_HOLD": tactics_roof,
              "APPROACH": tactics_drop, "WATCH": tactics_drop, "CAPTURE": tactics_drop,
              "FINAL_APPROACH": tactics_endgame, "FINAL_HOLD": tactics_endgame, "COLLAPSE": tactics_endgame}


# ------------------------------------------------------------------ small geometry helpers
def wrap(a: float) -> float:
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def unit(dx: float, dy: float) -> Optional[Tuple[float, float]]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


def hdist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a, b) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def yaw_vec(yaw: float) -> Tuple[float, float]:
    return (math.cos(yaw), math.sin(yaw))


def rot(v, a: float) -> Tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def perp(v, side: int) -> Tuple[float, float]:
    return (-side * v[1], side * v[0])


def held_slots(me) -> Dict[str, int]:
    """kind -> slot of the weapons we hold."""
    out: Dict[str, int] = {}
    for i, w in enumerate(getattr(me, "weapons", None) or ()):
        if w is not None and getattr(w, "kind", None):
            out.setdefault(w.kind, i)
    return out


def laser_charging(me) -> bool:
    for w in getattr(me, "weapons", None) or ():
        if w is not None and w.kind == "laser" and (w.charge or 0) > 0:
            return True
    return False


def has_weapon(track, kind: str, conf: float = 0.0) -> bool:
    """Seen (authoritative) or inferred with confidence >= conf (> 0 when conf == 0)."""
    if kind in (getattr(track, "weapons", ()) or ()):
        return True
    c = (getattr(track, "known_weapons", None) or {}).get(kind, 0.0)
    return c >= conf if conf > 0 else c > 0


def is_mech_id(tid) -> bool:
    return tid is not None and 0 <= int(tid) <= MECH_ID_MAX


# ------------------------------------------------------------------ aim fallback (used if aim.py is absent)
class _DirectAim:
    """Minimal aim used when `aim` cannot be imported (R15.1 fallback): direct yaw, center pitch."""

    @staticmethod
    def _direct(me, tr, C):
        s_t = 1.0 + float(C.path("cubes.size_per_cube", 0.05)) * float(getattr(tr, "cubes", 0) or 0)
        tz = tr.pos[2] + C.TARGET_CENTER_FRAC * float(C.path("mech.height", 9.0)) * s_t
        d = max(0.1, hdist(me.pos, tr.pos))
        yaw = bearing(me.pos, tr.pos)
        pitch = math.atan2(tz - (me.muzzle_z or me.pos[2] + 7.0), d)
        st = getattr(me, "stats", None) or {}
        return yaw, max(float(st.get("min_pitch", -0.7)), min(float(st.get("max_pitch", 1.3)), pitch))

    def gun_lead(self, me, tr, gun, C):
        y, p = self._direct(me, tr, C)
        return y, p, 0.0

    def laser_point(self, me, tr, C):
        return self._direct(me, tr, C)


def load_aim():
    try:
        import aim as _aim  # noqa: WPS433
        return _aim
    except Exception:
        return _DirectAim()


# ------------------------------------------------------------------ contract types owned here
@dataclass
class TacticCtx:
    C: object
    world: object
    A: object
    nav: object
    mapdata: object
    intent: object
    clock: object
    modes: "Modes"
    rng: object
    ostate: object = None


@dataclass
class GoalState:
    goal: str
    key: object
    entered_t: float
    active_s: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FightView:
    """FightEval after this module's gates (memory, committed capture, D14/D15, freshness)."""
    target_id: Optional[int] = None
    win: bool = False
    win_rule: str = ""
    win_sub: str = ""
    lose: frozenset = frozenset()          # clauses that hold now (after suppressions)
    lose_tracks: Dict[str, int] = field(default_factory=dict)
    cornered: bool = False
    t_close: float = INF
    cover_eta: Optional[float] = None      # optional FightEval extension (8.2.6 hysteresis)
    committed: bool = False
    d15: bool = False


# ------------------------------------------------------------------ Modes
class Modes:
    def __init__(self, C, mapdata, nav, rng, aim_mod=None) -> None:
        self.C, self.map, self.nav, self.rng = C, mapdata, nav, rng
        self.aim = aim_mod if aim_mod is not None else load_aim()
        self.assessor = None
        self.macro, self.sub, self.goal = "X1", "", None
        self.rule = "X0"
        self.exit_memory = ExitMemory()
        self._hard_fight_mem = ExitMemory()      # timeout / RL2 FIGHT memory (not bypassed by H2's A bonus)
        self.history: deque = deque(maxlen=8192)  # (t, layer, from, to, key, why)
        self.macro_since = 0.0
        self.terrain_call, self.desired_loadout = "MIXED", ""
        self._refreshed = False
        # X2
        self.target: Optional[int] = None
        self.x2_since: Optional[float] = None
        self.x2_last_hit_t = -INF
        self.x2_range_loss_since: Optional[float] = None
        self.corner_since: Optional[float] = None
        self.corner_checked_t = -INF
        self.roof_wait_expired = False           # tactics_roof sets it (R10.2 ROOF_WAIT_MAX_S)
        self.tac: Dict[str, Any] = {}            # per-episode tactic scratch (reset on macro entry)
        # X3
        self.x3_since: Optional[float] = None
        self.x3_entry: frozenset = frozenset()
        self.x3_e_track: Optional[int] = None
        self.x3_i_track: Optional[int] = None
        self.x3_threat: Optional[int] = None
        self.kite_open_since: Optional[float] = None
        self.e_los_broken_since: Optional[float] = None
        self.cover_point = None                  # (point, since) H7
        # bookkeeping
        self.last_hostile_los_t = -INF
        self.no_los_since = 0.0
        self.last_event_t = 0.0
        self.last_threat_sound_t = -INF          # near/mid gun/lightning/laser sound (R1.5, R7.2)
        self.neighbor: Optional[int] = None      # R1.2 (track id, or ghost id < 0)
        self.neighbor_sensed_t = -INF
        self.last_t = 0.0
        # goals
        self.gs: Optional[GoalState] = None
        self.reach: Optional[Dict[str, Any]] = None   # G2 state
        self.spot = None                         # (point, score, roof_id, kind, key)
        self.loot_item = None                    # (key, point, score, since, eta0, started_active_s)
        self._loot_scores: Dict[Any, float] = {}  # key -> 8.3 score of the last full _best_loot pass (H4)
        self.explore = None                      # (point, t)
        self.inv_points: List[Tuple[Tuple[float, float], float]] = []   # M8 near-point memory
        self.swap_lock_until = -INF              # H5
        self.final_point = None                  # R11.2 (tactics_endgame)
        self._paths: Dict[str, Tuple[Tuple[float, float], float]] = {}
        self._leg_writes = 0
        self._owner_level = "L8"
        self.n_changes = 0
        self.fv = FightView()

    # ============================================================== strategic refresh (6.1 step 4)
    def strategic_refresh(self, world, assessor, nav, clock) -> None:
        """R0 on next_changed (terrain call, desired loadout), hold-spot invalidation, zone field start."""
        self.assessor = assessor
        zone = world.zone
        if zone.next_changed or not self._refreshed:
            self._refreshed = True
            try:
                self.terrain_call, self.desired_loadout = assessor.terrain_call(world)   # R0.1, R0.2
            except Exception:
                pass
            self.spot = None                                                            # 8.4 invalid on next_changed
            self.final_point = None                                                     # R11.2 re-evaluated
            try:
                nav.zone_field_start(zone, not bool(world.me.in_zone))                  # 8.8
            except Exception:
                pass

    # ============================================================== main entry (6.1 steps 7-8)
    def decide(self, world, A, nav, ostate, intent, clock, assessor=None) -> None:
        if assessor is not None:
            self.assessor = assessor
        C = self.C
        t = world.tick.t
        dt = world.tick.dt if world.tick.dt and world.tick.dt > 0 else max(0.0, t - self.last_t) or 0.1
        self.last_t = t
        self.nav = nav
        self._leg_writes = 0
        self.exit_memory.prune(t)
        self._hard_fight_mem.prune(t)
        self._bookkeep(world, A, t)
        ctx = TacticCtx(C, world, A, nav, self.map, intent, clock, self, self.rng, ostate)
        fv = self._fight(world, A, t)
        self.fv = fv
        self._rows(ctx, fv, t)
        intent.macro = self.macro
        # F5 phase and goal limits (6.6) + R11.5 clock guard handled by the arbiter
        if self.macro == "X4" or (t < C.LOOT_END_T and self.macro != "X2"):
            intent.fire_self_defense_only = True
        if self.macro == "X4":
            self._owner_level = "X4"
            self.sub = self.sub or "FINAL_APPROACH"
            rule = tactics_endgame.run(self.sub, ctx)
            self.goal = None
        elif self.macro == "X2":
            self._owner_level = "L5"
            rule = self._run_x2(ctx, fv)
            self.goal = None
        elif self.macro == "X3":
            self._owner_level = "L6"
            rule = self._run_x3(ctx, fv)
            self.goal = None
        else:
            rule = self._run_x1(ctx, fv, dt)
        if self._leg_writes == 0:   # every branch must own the legs: hold still rather than leave none
            self.legs(ctx, rule or "6.2", None)
        intent.macro, intent.sub, intent.goal = self.macro, self.sub, self.goal
        intent.rule = rule or self.rule
        self.rule = intent.rule
        intent.trace.append(intent.rule)

    # ============================================================== bookkeeping
    def _bookkeep(self, world, A, t: float) -> None:
        los = False
        for tid in (A.hostiles or ()):
            ta = A.per_track.get(tid)
            tr = world.tracks.get(tid)
            if ta is not None and ta.los and tr is not None and tr.kind == "mech":
                los = True
                break
        if not los:
            for tr in world.visible_mechs():
                ta = A.per_track.get(tr.id)
                if ta is not None and ta.los:
                    los = True
                    break
        if los:
            self.last_hostile_los_t = t
        for ev in world.events or ():
            if ev.get("type") in ("damaged", "hit", "effect", "kill"):
                self.last_event_t = t
            if ev.get("type") == "hit" and self.target is not None and ev.get("target") == self.target:
                self.x2_last_hit_t = t
        for s in world.sounds or ():
            if s.band in ("near", "mid") and s.kind in ("gun", "lightning", "laser", "laser_charge"):
                self.last_threat_sound_t = t
        self._neighbor_update(world, t)
        evs = [ev.get("type") for ev in world.events or ()]
        if "took" in evs and "dropped" in evs:            # voluntary swap -> H5 lock
            self.note_swap(t)

    def _neighbor_update(self, world, t) -> None:
        """R1.2: t < SPAWN_PHASE_S and a mech seen, or a step/boost/gun/crate sound near/mid whose ghost
        estimate is within SPAWN_NEIGHBOR_R -> neighbor. Cleared beyond NEIGHBOR_CLEAR_R, unsensed for
        NEIGHBOR_CLEAR_S, or at LOOT_END_T."""
        C, me = self.C, world.me
        if t >= C.LOOT_END_T:
            self.neighbor = None
            return
        if self.neighbor is None and t < C.SPAWN_PHASE_S:
            best = None
            for tr in world.visible_mechs():
                d = hdist(me.pos, tr.pos)
                if best is None or d < best[0]:
                    best = (d, tr.id)
            if best is None:
                for snd in world.sounds or ():
                    gh = world.ghosts.get(snd.ghost_id) if snd.ghost_id is not None else None
                    if snd.kind in ("step", "boost", "gun", "crate") and snd.band in ("near", "mid") and gh is not None:
                        d = hdist(me.pos, gh.pos)
                        if d <= C.SPAWN_NEIGHBOR_R and (best is None or d < best[0]):
                            best = (d, gh.id)
            if best is not None:
                self.neighbor, self.neighbor_sensed_t = best[1], t
                self._log(t, "neighbor", None, best[1], best[1], "R1.2")
            return
        nb = self.neighbor
        if nb is None:
            return
        pos = None
        if nb >= 0 and nb in world.tracks:
            tr = world.tracks[nb]
            pos = tr.pos
            if tr.visible or t - tr.last_seen_t < 0.2:
                self.neighbor_sensed_t = t
        elif nb < 0 and nb in world.ghosts:
            pos = world.ghosts[nb].pos
            self.neighbor_sensed_t = max(self.neighbor_sensed_t, world.ghosts[nb].last_t)
        if pos is None or hdist(me.pos, pos) > C.NEIGHBOR_CLEAR_R or t - self.neighbor_sensed_t > C.NEIGHBOR_CLEAR_S:
            self._log(t, "neighbor", nb, None, nb, "clear")
            self.neighbor = None

    def _neighbor_pos(self, world):
        nb = self.neighbor
        if nb is None:
            return None
        if nb >= 0 and nb in world.tracks:
            return world.tracks[nb].pos
        if nb < 0 and nb in world.ghosts:
            return world.ghosts[nb].pos
        return None

    def _log(self, t, layer, frm, to, key, why) -> None:
        self.history.append((t, layer, frm, to, key, why))
        self.n_changes += 1

    # ============================================================== fight predicates (8.2) + gates
    def _committed_capture(self, world, A) -> bool:
        """R8.5 committed capture: CAPTURE with progress >= DROP_COMMIT_PROGRESS_S and
        (capture_s - progress) x incoming DPS < DROP_STAY_HP_FRAC x hp."""
        if self.goal != "M7" or self.sub != "CAPTURE" or self.gs is None:
            return False
        dr = world.drops.get(self.gs.key)
        if dr is None or dr.progress < self.C.DROP_COMMIT_PROGRESS_S:
            return False
        if dr.holder is not None and dr.holder != world.me.id:
            return False
        dps = sum(A.per_track[h].dps_to_us for h in (A.hostiles or ()) if h in A.per_track)
        return (dr.capture_s - dr.progress) * dps < self.C.DROP_STAY_HP_FRAC * world.me.hp

    def _d15(self, world, t) -> bool:
        """D15 (ALIVE2_FIGHT_IF_BEHIND): alive == 2, rival max HP (fresh) > ours x margin, window."""
        C = self.C
        if not C.ALIVE2_FIGHT_IF_BEHIND or world.tick.alive != 2 or not (C.ALIVE2_FIGHT_T <= t < C.ENDGAME_T):
            return False
        for tr in world.tracks.values():
            if tr.kind == "mech" and tr.max_hp is not None and t - tr.last_seen_t < C.RIVAL_INFO_MAX_AGE_S:
                if tr.max_hp > world.me.max_hp * C.ALIVE2_BEHIND_MARGIN:
                    return True
        return False

    def e_track_ok(self, world, A, tid, t) -> bool:
        """LOSE (e) / R6.1 track rule: seen within LIGHTNING_TRACK_MAX_AGE_S, lightning seen or
        inferred with confidence >= CONF_SOUND (prior-only never counts)."""
        tr = world.tracks.get(tid) if tid is not None else None
        if tr is None or t - tr.last_seen_t > self.C.LIGHTNING_TRACK_MAX_AGE_S:
            return False
        return has_weapon(tr, "lightning", self.C.CONF_SOUND)

    def e_reach_time(self, world, A, tid) -> float:
        """(d - LIGHTNING_RANGE x m) / closing speed; inf when not closing (8.2.4 suppression)."""
        ta = A.per_track.get(tid)
        tr = world.tracks.get(tid)
        if ta is None or tr is None or ta.closing_speed <= 0:
            return INF
        m = tr.lightning_range_mult or 1.0
        return max(0.0, ta.d - self.C.path("weapons.lightning.range", 45) * m) / ta.closing_speed

    def _fight(self, world, A, t) -> FightView:
        C = self.C
        committed = self._committed_capture(world, A)
        target = self.target if self.macro == "X2" else A.target_id
        ctx = {"exit_memory": self.exit_memory, "committed_capture": committed,
               "neighbor": self.neighbor,
               "x2_since": self.x2_since if self.macro == "X2" else None,
               "x3_since": self.x3_since if self.macro == "X3" else None,
               "alive": world.tick.alive, "cornered_active": self.macro == "X2" and self.sub == "CORNERED",
               "sub": self.sub if self.macro == "X2" else ""}
        fe = FightEval()
        if self.assessor is not None:
            fe = self.assessor.evaluate_fight(world, A, self.macro, target, ctx)
        lose = set(fe.lose_clauses or ())
        tracks = dict(fe.lose_tracks or {})
        held = held_slots(world.me)
        # (a), (b), (c) exist only in X2 (8.2.4)
        if self.macro != "X2":
            lose -= {"a", "b", "c"}
        # (e): fresh, confident lightning track only (R6.1; "stale track -> no flee", "prior-only -> O8 only")
        if "e" in lose and not self.e_track_ok(world, A, tracks.get("e"), t):
            lose.discard("e")
        # (i): R6.3 toggle and loadout (LASER_VIEW_RESPONSE "gl_only": lightning held, no laser)
        if "i" in lose:
            mode = C.LASER_VIEW_RESPONSE
            if mode == "off" or (mode == "gl_only" and not ("lightning" in held and "laser" not in held)):
                lose.discard("i")
        # (h): only while the R1.2 neighbor is set
        # committed capture (R8.5): suppress (a)(b)(c)(f)(g); (e) while its time to reach > remaining
        if committed:
            lose -= {"a", "b", "c", "f", "g"}
            if "e" in lose:
                dr = world.drops.get(self.gs.key) if self.gs else None
                remain = (dr.capture_s - dr.progress) if dr is not None else 0.0
                if self.e_reach_time(world, A, tracks.get("e")) > remain:
                    lose.discard("e")
        d15 = self._d15(world, t)
        if d15:                                   # D15: only (d) survives
            lose &= {"d"}
        win = bool(fe.win) and not lose
        sub, rule = fe.win_sub, fe.win_rule
        tid = fe.target_id if fe.target_id is not None else target
        if committed:                             # R8.5: T3 suppressed
            win = False
        alive = world.tick.alive
        if win and C.ALIVE_LE3_PASSIVE and alive <= C.ALIVE_ENDGAME_N and not d15:
            # D14: no initiation; WIN only for CORNERED or self-defense
            if not (fe.cornered or (tid is not None and world.damaged_us_recently(tid, C.SELF_DEFENSE_S))):
                win = False
        if d15 and not win and not lose:
            tr = world.tracks.get(tid) if tid is not None else None
            if tr is not None and tr.visible:
                win, rule = True, "D15"
        return FightView(tid, win, rule, sub, frozenset(lose), tracks, bool(fe.cornered), fe.t_close,
                         getattr(fe, "cover_eta", None), committed, d15)

    # ============================================================== transition rows (5.6)
    def _t4_ok(self, world, A, fv, t) -> Optional[str]:
        # LOSE clauses come from assess: (c) is R5.2 third party, (f) R9.1, (h) R1.2, (e) R6.1, (i) R6.3;
        # R10.1 (roof veto) and the R0.5 loot-only veto are inside assess's WIN.
        """T4: LOSE with (FLEE, *) clear; (e) also (FLEE, track) clear unless bypassed (LOS and closing,
        or LOS within LIGHTNING_KEEPOUT_R x m); (i) also (FLEE, track) clear (no bypass, R6.3).
        Returns the deciding clauses as a string or None."""
        if not fv.lose or not self.exit_memory.clear("FLEE", "*", t):
            return None
        ok = []
        for c in sorted(fv.lose):
            if c == "e":
                tid = fv.lose_tracks.get("e")
                if not (self.exit_memory.clear("FLEE", tid, t) or self._e_bypass(world, A, tid)):
                    continue
            elif c == "i":
                if not self.exit_memory.clear("FLEE", fv.lose_tracks.get("i"), t):
                    continue
            ok.append(c)
        return "".join(ok) or None

    def _e_bypass(self, world, A, tid) -> bool:
        ta = A.per_track.get(tid)
        tr = world.tracks.get(tid)
        if ta is None or tr is None or not ta.los:
            return False
        m = tr.lightning_range_mult or 1.0
        return ta.closing_speed > self.C.RUSH_CLOSING_MIN or ta.d <= self.C.LIGHTNING_KEEPOUT_R * m

    def _t3_ok(self, world, A, fv, t) -> bool:
        """T3: WIN(j) with (FIGHT, j) and (FIGHT, *) clear; the REENGAGE part of (FIGHT, j) is bypassed by
        A >= ENGAGE_A_MIN + REENGAGE_A_BONUS (H2)."""
        if not fv.win or fv.target_id is None:
            return False
        j = fv.target_id
        if not self.exit_memory.clear("FIGHT", "*", t):
            return False
        if self.exit_memory.clear("FIGHT", j, t):
            return True
        return self._hard_fight_mem.clear("FIGHT", j, t) and A.A >= self.C.ENGAGE_A_MIN + self.C.REENGAGE_A_BONUS

    def _enter_macro(self, to: str, sub: str, t: float, why: str, key=None) -> None:
        frm = self.macro
        if frm == "X1" and self.gs is not None:
            g = self.gs.goal
            if g in ("M3A", "M8"):        # "contact seen (L5/L6 decide)" is their own exit: write it
                self.exit_memory.write(GOAL_NAMES[g], self.gs.key, ExitMemory.PERMANENT)
            self._log(t, "goal", g, None, self.gs.key, "macro " + to)
        self._log(t, "macro", frm, to, key, why)
        self.macro, self.sub, self.macro_since = to, sub, t
        self.tac = {}
        self.reach = None
        if to == "X2":
            self.x2_since = t
            self.x2_last_hit_t = -INF
            self.x2_range_loss_since = None
            self.roof_wait_expired = False
            self.corner_since = t if sub == "CORNERED" else None
            self.corner_checked_t = t
        elif to == "X3":
            self.x3_since = t
            self.kite_open_since = None
            self.e_los_broken_since = None
            self.cover_point = None
        if to != "X1":
            self.gs = None
            self.goal = None

    def _rows(self, ctx, fv, t) -> None:
        C, world, A, intent = self.C, ctx.world, ctx.A, ctx.intent
        m = self.macro
        # T2 (R11.1): t >= ENDGAME_T, sticky
        if m in ("X1", "X2", "X3") and t >= C.ENDGAME_T:
            self._enter_macro("X4", "FINAL_APPROACH", t, "T2")
            intent.trace.append("T2")
            return
        if m == "X4":
            return
        if m == "X1":
            if self._t3_ok(world, A, fv, t):                                     # T3
                self.target = fv.target_id
                self._enter_macro("X2", self._default_sub(world, A, fv), t, "T3:" + (fv.win_rule or "WIN"),
                                  fv.target_id)
                intent.trace.append("T3")
                return
            cl = self._t4_ok(world, A, fv, t)                                    # T4
            if cl:
                self._enter_x3(world, A, fv, t, cl)
                intent.trace.append("T4:" + cl)
            return
        if m == "X2":
            held_ok = t - self.macro_since >= C.ENGAGE_MIN_HOLD_S
            cl = self._t4_ok(world, A, fv, t) if held_ok else None
            if cl and self.sub != "CORNERED":                                    # T4 after ENGAGE_MIN_HOLD_S
                if self.target is not None:                                      # H2: REENGAGE memory
                    self.exit_memory.write("FIGHT", self.target, t + C.REENGAGE_COOLDOWN_S)
                self._enter_x3(world, A, fv, t, cl)
                intent.trace.append("T4:" + cl)
                return
            self._x2_own_exit(ctx, fv, t)                                        # T5
            return
        if m == "X3":
            if fv.cornered:                                                      # 8.2.6 -> X2 CORNERED
                self.target = self.x3_threat if self.x3_threat is not None else fv.target_id
                self._enter_macro("X2", "CORNERED", t, "R6.2", self.target)
                intent.trace.append("R6.2")
                return
            if t - self.macro_since >= C.DISENGAGE_MIN_HOLD_S and self._t3_ok(world, A, fv, t):   # T3 from X3
                self.target = fv.target_id
                self._enter_macro("X2", self._default_sub(world, A, fv), t, "T3:" + (fv.win_rule or "WIN"),
                                  fv.target_id)
                intent.trace.append("T3")
                return
            self._x3_own_exit(ctx, fv, t)                                        # T6

    def _default_sub(self, world, A, fv) -> str:
        if fv.win_sub:
            return fv.win_sub
        if fv.win_rule == "R10.2":
            return "ROOF_ASSAULT"
        pb = A.playbook or world.me.playbook
        if pb == "PB_L":
            ta = A.per_track.get(fv.target_id)
            return "BRAWL" if ta is not None and ta.d <= self.C.LIGHTNING_RANGE else "RUSH"
        return "DUEL" if pb == "PB_Z" else "DUEL_GUN"

    def _enter_x3(self, world, A, fv, t, clauses: str) -> None:
        self.x3_entry = frozenset(clauses)
        self.x3_e_track = fv.lose_tracks.get("e") if "e" in clauses else None
        self.x3_i_track = fv.lose_tracks.get("i") if "i" in clauses else None
        th = None
        for c in ("e", "i", "g"):
            if c in clauses and fv.lose_tracks.get(c) is not None:
                th = fv.lose_tracks[c]
                break
        self.x3_threat = th if th is not None else (A.main_threat_id if A.main_threat_id is not None else fv.target_id)
        sub = "KITE" if clauses == "e" else "BREAK_LOS"
        self._enter_macro("X3", sub, t, "T4:" + clauses, self.x3_threat)

    # -------------------------------------------------------------- X2 own exits (T5, 5.3)
    def _x2_own_exit(self, ctx, fv, t) -> None:
        C, world, A = self.C, ctx.world, ctx.A
        j = self.target
        # R5.1: follow the H3-hysteresis target from assess while fighting (not in CORNERED/ROOF_ASSAULT)
        if self.sub not in ("CORNERED", "ROOF_ASSAULT") and A.target_id is not None and A.target_id != j:
            self.target = j = A.target_id
            self.x2_range_loss_since = None
        tr = world.tracks.get(j) if j is not None else None
        killed = any(ev.get("type") == "kill" and ev.get("target") == j for ev in world.events or ())
        if j is None or tr is None or killed or t - tr.last_seen_t > C.TARGET_LOST_S:
            self._exit_x2(t, "dead_or_lost", None, ctx)
            return
        ta = A.per_track.get(j)
        # CORNERED (R6.2 / 8.2.6): leave for X3 after CORNER_MIN_HOLD_S when a cover ETA clears the band;
        # A >= ENGAGE_A_MIN -> normal X2
        if self.sub == "CORNERED":
            if t - (self.x2_since or t) >= C.ENGAGE_TIMEOUT_S:            # X2 keeps its total timeout (15 exit check)
                self._exit_x2(t, "timeout", C.M4_TIMEOUT_COOLDOWN_S, ctx, hard=True)
                return
            since = self.corner_since if self.corner_since is not None else self.macro_since
            held = t - since >= C.CORNER_MIN_HOLD_S                   # CORNER_MIN_HOLD_S before any way out
            if held and A.A >= C.ENGAGE_A_MIN and not fv.lose:
                self.sub = self._default_sub(world, A, FightView(target_id=j))
                self._log(t, "sub", "CORNERED", self.sub, j, "R6.2:A")
                return
            if held and t - self.corner_checked_t >= C.CORNER_RECHECK_S - 1e-9:
                self.corner_checked_t = t
                if fv.cover_eta is not None:
                    leave = fv.cover_eta < fv.t_close - C.CORNER_COVER_MARGIN_S
                else:
                    leave = not fv.cornered
                if leave:
                    self.exit_memory.write("FIGHT", j, t + C.REENGAGE_COOLDOWN_S)
                    self._enter_x3(world, A, fv, t, "".join(sorted(fv.lose)))
                    ctx.intent.trace.append("R6.2:exit")
            return
        # timeouts: ENGAGE_TIMEOUT_S (ENGAGE_TIMEOUT_SHORT_S in the T3-T4 LASER window), ENGAGE_NO_HIT_S
        limit = C.ENGAGE_TIMEOUT_S
        tc = ctx.A.terrain_call or self.terrain_call
        if C.PHASE_T3_SHRINK <= t < C.PHASE_T4_SHRINK and tc == "LASER":
            limit = C.ENGAGE_TIMEOUT_SHORT_S                                     # R0.5
        no_hit = t - max(self.x2_since or t, self.x2_last_hit_t)
        if t - (self.x2_since or t) >= limit or no_hit >= C.ENGAGE_NO_HIT_S:
            self._exit_x2(t, "timeout", C.M4_TIMEOUT_COOLDOWN_S, ctx, hard=True)
            return
        if self.roof_wait_expired:                                                # R10.2 ROOF_WAIT_MAX_S
            self._exit_x2(t, "R10.2:wait", C.M4_TIMEOUT_COOLDOWN_S, ctx, hard=True)
            return
        # RL2: out of playbook range and not closing for ENGAGE_RANGE_LOSS_S
        if ta is not None and self.sub != "ROOF_ASSAULT" and self._out_of_range(world, A, ta) and ta.closing_speed <= 0:
            if self.x2_range_loss_since is None:
                self.x2_range_loss_since = t
            elif t - self.x2_range_loss_since >= C.ENGAGE_RANGE_LOSS_S:
                self._exit_x2(t, "RL2", C.NO_CHASE_COOLDOWN_S, ctx, hard=True)
                return
        else:
            self.x2_range_loss_since = None

    def _out_of_range(self, world, A, ta) -> bool:
        C = self.C
        pb = A.playbook or world.me.playbook
        if pb == "PB_L":
            return ta.d > C.LIGHTNING_RUSH_STILL_R or (ta.d > C.LIGHTNING_RANGE and -ta.closing_speed > C.RETREAT_SPEED_MIN)
        if pb == "PB_Z":
            return ta.d > float(C.path("weapons.laser.range", 280))
        return ta.d > float(C.path("weapons.gun.range", 230))

    def _exit_x2(self, t, why, cooldown, ctx, hard=False) -> None:
        j = self.target
        if cooldown is not None and j is not None:
            self.exit_memory.write("FIGHT", j, t + cooldown)
            if hard:
                self._hard_fight_mem.write("FIGHT", j, t + cooldown)
        self._enter_macro("X1", "", t, "T5:" + why, j)
        ctx.intent.trace.append("T5:" + why)
        self.target = None

    # -------------------------------------------------------------- X3 own exits (T6, 5.3, R6.1, R6.3)
    def _x3_own_exit(self, ctx, fv, t) -> None:
        C, world, A = self.C, ctx.world, ctx.A
        others = set(fv.lose) - {"e"}
        e_now = "e" in fv.lose
        e_tid = fv.lose_tracks.get("e") if e_now else self.x3_e_track
        if e_now and fv.lose_tracks.get("e") is not None:
            self.x3_e_track = fv.lose_tracks["e"]
        # X3 timeout (DISENGAGE_MAX_S), single rule (B4)
        if t - (self.x3_since or t) >= C.DISENGAGE_MAX_S:
            if world.damage_since(t - C.DISENGAGE_CLEAR_S) > 0:
                th = self.x3_threat if self.x3_threat is not None else A.main_threat_id
                self.target = th
                self._enter_macro("X2", "CORNERED", t, "T6:timeout_fire", th)   # no FLEE memory
                ctx.intent.trace.append("T6:CORNERED")
                return
            self.exit_memory.write("FLEE", "*", t + C.DISENGAGE_COOLDOWN_S)
            self._x3_track_memory(t, e_now, e_tid)
            self._enter_macro("X1", "", t, "T6:timeout", None)
            ctx.intent.trace.append("T6:timeout")
            return
        # which exit applies (mixed-clause rule)
        if others:
            use_r61 = False
        elif e_now:
            use_r61 = True
        else:
            use_r61 = self.x3_e_track is not None
        if use_r61 and e_tid is not None:
            self.sub = "KITE"
            if self._r61_exit(world, A, e_tid, t):
                self.exit_memory.write("FLEE", e_tid, t + C.DISENGAGE_COOLDOWN_S)     # R6.1
                self._x3_track_memory(t, False, None)
                self._enter_macro("X1", "", t, "T6:R6.1", e_tid)
                ctx.intent.trace.append("T6:R6.1")
            return
        # general exit: no hostile with LOS for DISENGAGE_CLEAR_S and the nearest threat beyond its reach
        reach = any(ta.in_its_reach for tid, ta in A.per_track.items()
                    if tid in world.tracks and world.tracks[tid].kind == "mech")
        no_los_since = max(self.last_hostile_los_t, self.x3_since if self.x3_since is not None else t)
        if t - no_los_since >= C.DISENGAGE_CLEAR_S and not reach:
            self._x3_track_memory(t, e_now, e_tid)
            self._enter_macro("X1", "", t, "T6:general", None)
            ctx.intent.trace.append("T6:general")

    def _x3_track_memory(self, t, e_now, e_tid) -> None:
        C = self.C
        if e_now and e_tid is not None:            # any X3 -> X1 exit while (e) holds writes (FLEE, track)
            self.exit_memory.write("FLEE", e_tid, t + C.DISENGAGE_COOLDOWN_S)
        if self.x3_i_track is not None:            # R6.3 exit memory
            self.exit_memory.write("FLEE", self.x3_i_track, t + C.DISENGAGE_COOLDOWN_S)

    def _r61_exit(self, world, A, tid, t) -> bool:
        """R6.1 exit: d > LIGHTNING_SAFE_R x m and opening for KITE_EXIT_S, or LOS broken for
        DISENGAGE_CLEAR_S, or the track is gone."""
        C = self.C
        tr = world.tracks.get(tid)
        ta = A.per_track.get(tid)
        if tr is None:
            return True
        m = tr.lightning_range_mult or 1.0
        if ta is not None and ta.d > C.LIGHTNING_SAFE_R * m and ta.closing_speed < 0:
            if self.kite_open_since is None:
                self.kite_open_since = t
        else:
            self.kite_open_since = None
        los = ta is not None and ta.los and tr.visible
        if not los:
            if self.e_los_broken_since is None:
                self.e_los_broken_since = t
        else:
            self.e_los_broken_since = None
        return ((self.kite_open_since is not None and t - self.kite_open_since >= C.KITE_EXIT_S)
                or (self.e_los_broken_since is not None and t - self.e_los_broken_since >= C.DISENGAGE_CLEAR_S))

    # ============================================================== shared actuator helpers
    def legs(self, ctx, rule, direction, speed=1.0, goal=None, juke_ok=True, level=None) -> None:
        """The single owner LegRequest of this tick (X4, L5, L6, L7 or L8)."""
        lvl = level or self._owner_level
        ctx.intent.legs.append(LegRequest(lvl, rule, direction, speed, goal, juke_ok))
        self._leg_writes += 1

    def aim_track(self, ctx, prio_name, rule, tr, tiebreak=0.0, kind=None) -> Optional[AimRequest]:
        """AimRequest at a track: gun lead when we hold a gun (8.7), the predicted laser point while our
        laser charges (R9.6), else straight at its center."""
        if tr is None:
            return None
        me, C = ctx.world.me, self.C
        held = held_slots(me)
        try:
            if laser_charging(me) or kind == "laser":
                yaw, pitch = self.aim.laser_point(me, tr, C)
            elif (kind in (None, "gun")) and "gun" in held:
                yaw, pitch, _ = self.aim.gun_lead(me, tr, me.weapons[held["gun"]], C)
            else:
                yaw, pitch = _DirectAim._direct(me, tr, C)
        except Exception:
            yaw, pitch = _DirectAim._direct(me, tr, C)
        req = AimRequest(AIM_PRIO[prio_name], rule, yaw, pitch, tr.id, tiebreak)
        ctx.intent.aim.append(req)
        return req

    def aim_dir(self, ctx, prio_name, rule, yaw, pitch=0.0, target_id=None) -> AimRequest:
        req = AimRequest(AIM_PRIO[prio_name], rule, yaw, pitch, target_id)
        ctx.intent.aim.append(req)
        return req

    def fire(self, ctx, target_id, kinds=None, rule_prefix="", allow_npc=False) -> None:
        """FireRequest per held slot (gates in the arbiter, 6.6). allow_npc only for R7.2 / A-6: during
        X2/X3 an NPC as nearest cone object blocks lightning (R7.1 "never target the NPC", F3)."""
        for kind, slot in held_slots(ctx.world.me).items():
            if kinds is not None and kind not in kinds:
                continue
            rule = FIRE_RULE.get(kind, "F")
            if allow_npc and kind == "lightning":
                rule = "R7.2"
            ctx.intent.fire.append(FireRequest(slot, rule_prefix + rule if rule_prefix else rule, target_id))

    def boost(self, ctx, rule, want, floor, owner="MODE") -> None:
        ctx.intent.boost.append(BoostRequest(owner, rule, bool(want), float(floor)))

    def go(self, ctx, key: str, goal, silent: bool = False, exposure: bool = True) -> Steer:
        """Path to `goal` through nav (A* request when the goal moved > PATH_DEVIATION_M), then follow.
        Climb/jump edges become CLIMB boost / jump requests (R13.2)."""
        C, me, t = self.C, ctx.world.me, ctx.world.tick.t
        g = (float(goal[0]), float(goal[1]))
        prev = self._paths.get(key)
        if prev is None or hdist(prev[0], g) > C.PATH_DEVIATION_M or t - prev[1] > 5.0:
            cost = {"exposure_tracks": [tr for tr in ctx.world.visible_mechs()] if exposure else [],
                    "heat": True, "zone": True, "silent": silent, "energy": me.energy,
                    "reserve": C.CLIMB_NAV_RESERVE}
            try:
                ctx.nav.path_request(key, me.pos, g, cost, t)
            except Exception:
                pass
            self._paths[key] = (g, t)
        try:
            st = ctx.nav.follow(me, key, g)
        except Exception:
            st = None
        if st is None:
            d = unit(g[0] - me.pos[0], g[1] - me.pos[1])
            st = Steer(d, arrived=d is None)
        if st.jump:
            ctx.intent.jump.append(JumpRequest("CLIMB", "R13.2"))
        if st.climb_boost:
            ctx.intent.boost.append(BoostRequest("CLIMB", "R13.2", True, 0.0))
        return st

    def walk_to(self, ctx, key, rule, goal, speed=1.0, silent=False, juke_ok=True) -> Steer:
        st = self.go(ctx, key, goal, silent)
        self.legs(ctx, rule, st.direction, speed, goal, juke_ok)
        return st

    def take(self, ctx, writer, rule, pickup_id, drop=None) -> None:
        ctx.intent.take.append(TakeRequest(writer, rule, pickup_id, drop))

    def wants_pickup(self, world, kind: str) -> Tuple[bool, Optional[int], float]:
        """8.5 via assess.slot_to_replace when available; (take?, drop slot or None, gain)."""
        me = world.me
        held = [w.kind if w is not None else None for w in (me.weapons or [None, None])]
        if kind in held:
            return False, None, 0.0
        if None in held:
            return True, None, 1.0
        tc = self.terrain_call
        try:
            import assess  # noqa: WPS433
            slot, gain = assess.slot_to_replace(held, kind, tc, self.C)
        except Exception:
            slot, gain = slot_to_replace_local(held, kind, tc, self.C)
        return (slot is not None and gain > 0), slot, gain

    # ============================================================== X2 FIGHT behavior
    def _run_x2(self, ctx, fv) -> str:
        world, A = ctx.world, ctx.A
        tr = world.tracks.get(self.target) if self.target is not None else None
        # AIM3 is written by the tactic (target lead / predicted laser point)
        sub = self.sub or self._default_sub(world, A, fv)
        self.sub = sub
        mod = SUB_MODULE.get(sub, tactics_gun)
        rule = mod.run(sub, ctx)
        if tr is not None:
            # empty-slot fill only (6.7) while fighting
            self._empty_slot_fill(ctx)
        return rule

    # ============================================================== X3 FLEE behavior (5.3, R6.1, R6.3)
    def _run_x3(self, ctx, fv) -> str:
        C, world, A, me = self.C, ctx.world, ctx.A, ctx.world.me
        th_id = self.x3_threat
        if th_id is None or th_id not in world.tracks:
            th_id = A.main_threat_id
        tr = world.tracks.get(th_id) if th_id is not None else None
        ta = A.per_track.get(th_id) if th_id is not None else None
        # aim AIM5: main threat (highest DPS on us)
        mt = world.tracks.get(A.main_threat_id) if A.main_threat_id is not None else tr
        self.aim_track(ctx, "AIM5", "AIM5", mt or tr)
        if tr is None:
            self.legs(ctx, "X3", None)
            return "X3"
        m = tr.lightning_range_mult or 1.0
        light_threat = has_weapon(tr, "lightning", C.CONF_SOUND)
        if self.sub == "KITE":
            rule = "R6.1"
            rng = C.path("weapons.lightning.range", 45) * m
            closing = ta.closing_speed if ta is not None else 0.0
            d = ta.d if ta is not None else hdist(me.pos, tr.pos)
            budget = (d - rng) / closing if closing > 0 else C.DISENGAGE_MAX_S
            pt = self._cover_point(ctx, [tr], max(0.0, budget))
            if pt is not None:
                st = self.go(ctx, "x3", pt, exposure=True)
                if st.arrived or hdist(me.pos, pt) < C.HOLD_ARRIVE_R:
                    # circle the building: move perpendicular to the threat bearing (keep the wall between)
                    away = unit(me.pos[0] - tr.pos[0], me.pos[1] - tr.pos[1]) or (1.0, 0.0)
                    self.legs(ctx, rule, perp(away, 1), 1.0, pt, juke_ok=False)
                else:
                    self.legs(ctx, rule, st.direction, 1.0, pt, juke_ok=False)
            else:
                away = unit(me.pos[0] - tr.pos[0], me.pos[1] - tr.pos[1])
                self.legs(ctx, rule, away, 1.0, None, juke_ok=False)       # backpedal straight away
            self.boost(ctx, rule, True, C.BOOST_FLOOR_ESCAPE)
            # fire: gun F1 every tick; laser only beyond LASER_CHARGE_MIN_GAP x m (or in range, no cover)
            kinds = {"gun", "lightning", "mortar"}
            if d >= C.LASER_CHARGE_MIN_GAP * m or (pt is None and d <= rng):
                kinds.add("laser")
            self.fire(ctx, tr.id, kinds)
            return rule
        # BREAK_LOS (general; R6.3 for LOSE (i)): path to the best cover point blocking all threats, backpedal
        rule = "R6.3" if "i" in self.x3_entry else "X3:" + ("".join(sorted(self.x3_entry)) or "-")
        threats = [world.tracks[i] for i in (A.hostiles or ()) if i in world.tracks] or [tr]
        pt = self._cover_point(ctx, threats, C.DISENGAGE_MAX_S)
        if pt is not None:
            st = self.go(ctx, "x3", pt, exposure=True)
            self.legs(ctx, rule, st.direction, 1.0, pt, juke_ok=True)
        else:
            away = unit(me.pos[0] - tr.pos[0], me.pos[1] - tr.pos[1])
            self.legs(ctx, rule, away, 1.0, None)
        if light_threat and ta is not None and ta.d <= C.LIGHTNING_THREAT_R * m:
            self.boost(ctx, rule, True, C.BOOST_FLOOR_ESCAPE)
        # fire: F1 at main threat; F2 only per R6.1 gap rule; F3 at anything in range
        kinds = {"gun", "lightning", "mortar"}
        if ta is not None and ta.d >= C.LASER_CHARGE_MIN_GAP * m:
            kinds.add("laser")
        self.fire(ctx, (mt or tr).id, kinds)
        return rule

    def _cover_point(self, ctx, threats, budget_s):
        """Cover point (8.4): kept COVER_MIN_HOLD_S (H7); urgent query first, ranked 'cover' spot second."""
        C, t = self.C, ctx.world.tick.t
        if self.cover_point is not None and t - self.cover_point[1] < C.COVER_MIN_HOLD_S:
            return self.cover_point[0]
        pt = None
        try:
            sl = ctx.clock.slice(C.COVER_URGENT_MS)
            pt = ctx.nav.cover_query(ctx.world.me, [tr.pos for tr in threats], budget_s, sl)
        except Exception:
            pt = None
        if pt is None:
            try:
                best = ctx.nav.best_spot("cover", ctx.world, ctx.A)
                pt = best[0] if best else None
            except Exception:
                pt = None
        self.cover_point = (pt, t) if pt is not None else None
        return pt

    # ============================================================== X1 GOAL (5.5)
    def _run_x1(self, ctx, fv, dt) -> str:
        world = ctx.world
        t = world.tick.t
        self._refresh_spot(ctx)
        self._select_goal(ctx, fv, t)
        gs = self.gs
        # F5 goal gate (gap-fill): M3H, M3A, M6 fire only in self-defense
        if self.goal in ("M3H", "M3A", "M6"):
            ctx.intent.fire_self_defense_only = True
        # G2 reach override (L7): suspends (pauses) the goal
        if self._reach(ctx, fv, t, dt):
            self._owner_level = "L8"
            return "G2"
        self._owner_level = "L8"
        if gs is not None:
            gs.active_s += dt
        self._aim6(ctx)
        return getattr(self, "_r_" + self.goal)(ctx, gs)

    # -------------------------------------------------------------- selector with stickiness (G1)
    def goal_order(self, t, alive) -> Tuple[str, ...]:
        C = self.C
        order = GOAL_ORDER_LOOT_WINDOW if t < C.LOOT_END_T else GOAL_ORDER
        if alive <= C.ALIVE_ENDGAME_N:                    # R11.4: no hunting
            order = tuple(g for g in order if g not in ("M8", "M3A"))
        return order

    def _mem_clear(self, g, key, t) -> bool:
        if not self.exit_memory.clear(GOAL_NAMES[g], key, t):
            return False
        return True

    def _select_goal(self, ctx, fv, t) -> None:
        order = self.goal_order(t, ctx.world.tick.alive)
        cur, gs = self.goal, self.gs
        exited = False
        if cur is not None and gs is not None:
            if cur not in order:
                exited = True
                self._log(t, "goal", cur, None, gs.key, "window")
            else:
                why = getattr(self, "_x_" + cur)(ctx, gs, t)
                if why:
                    exited = True
                    self._log(t, "goal", cur, None, gs.key, why)
                    ctx.intent.trace.append(f"{cur}:{why}")
        cache: Dict[str, Tuple[bool, Any]] = {}

        def entry(g):
            if g not in cache:
                cache[g] = getattr(self, "_e_" + g)(ctx, g == cur and not exited, t)
            return cache[g]

        chosen = None
        for g in order:
            ok, key = entry(g)
            if not ok:
                continue
            if g == cur and not exited:
                chosen = (g, gs.key)
                break
            if not self._mem_clear(g, key, t):
                continue
            chosen = (g, key)
            break
        if chosen is not None and cur is not None and not exited and chosen[0] != cur and cur in order:
            if order.index(chosen[0]) > order.index(cur):       # a lower goal: only after own exit or hold
                ok, _ = entry(cur)
                if ok or t - gs.entered_t < self.C.MODE_MIN_HOLD_S:
                    chosen = (cur, gs.key)
                else:
                    self._log(t, "goal", cur, None, gs.key, "predicate")
        if chosen is None:
            chosen = ("M3", self._spot_key())
        if exited or chosen[0] != cur:
            self.goal = chosen[0]
            self.gs = GoalState(chosen[0], chosen[1], t)
            self.sub = ""
            self._log(t, "goal", cur, chosen[0], chosen[1], "enter")
            getattr(self, "_init_" + chosen[0], lambda c, s: None)(ctx, self.gs)

    # -------------------------------------------------------------- G2 reach override (5.5)
    def _reach(self, ctx, fv, t, dt) -> bool:
        C, world, A, me = self.C, ctx.world, ctx.A, ctx.world.me
        if fv.committed or (self.goal == "M7" and self.sub == "CAPTURE") or self.goal in ("M3A", "M6"):
            self.reach = None
            return False
        for tr in world.visible_mechs():
            ta = A.per_track.get(tr.id)
            if ta is not None and ta.in_its_reach:
                self.reach = None
                return False
        pr = float(C.path("loot.pickup_radius", 4.0))
        item = None
        for p in world.pickups.values():
            if getattr(p, "ours", False) or hdist(me.pos, p.pos) > pr:
                continue
            ok, slot, _ = self.wants_pickup(world, p.kind)
            if ok and self.exit_memory.clear("LOOT", ("p", p.id), t):
                item = (("p", p.id), p.pos, p.id, slot)
                break
        if item is None:
            best = None
            for c in world.cubes.values():
                d = hdist(me.pos, c.pos)
                if d <= C.LOOT_REACH_R and self.exit_memory.clear("LOOT", ("c", c.id), t) and (best is None or d < best[0]):
                    best = (d, (("c", c.id), c.pos, None, None))
            item = best[1] if best else None
        if item is None:
            self.reach = None
            return False
        if self.reach is None or self.reach["key"] != item[0]:
            self.reach = {"key": item[0], "t": 0.0}
        self.reach["t"] += dt
        if self.reach["t"] >= C.LOOT_REACH_MAX_S:                       # G2 timeout: blacklist the item
            self.exit_memory.write("LOOT", item[0], t + C.LOOT_BLACKLIST_S)
            self._log(t, "reach", item[0], None, item[0], "timeout")
            self.reach = None
            return False
        d = unit(item[1][0] - me.pos[0], item[1][1] - me.pos[1])
        self.legs(ctx, "G2", d, 1.0, (item[1][0], item[1][1]), level="L7")
        if item[2] is not None:
            writer = "L7" if item[3] is None else "R0.4"
            if writer == "R0.4" and not self._switch_gates(world, t):
                writer = None
            if writer:
                self.take(ctx, writer, "G2", item[2], item[3])
        self._aim6(ctx)
        return True

    def _switch_gates(self, world, t) -> bool:
        """R0.4 gates: no enemy mech visible, our laser not charging, H5 lock."""
        return not world.visible_mechs() and not laser_charging(world.me) and t >= self.swap_lock_until

    def note_swap(self, t) -> None:
        """H5: lock a voluntary swap until the end of the first complete phase starting after t."""
        phases = self.C.path("zone.phases", []) or []
        start = 0.0
        for ph in phases:
            try:
                ln = float(ph["hold_s"]) + float(ph["shrink_s"])
            except Exception:
                break
            if start > t:
                self.swap_lock_until = start + ln
                return
            start += ln
        self.swap_lock_until = INF

    def _empty_slot_fill(self, ctx) -> None:
        world, me = ctx.world, ctx.world.me
        if None not in [w.kind if w else None for w in (me.weapons or [None, None])]:
            return
        pr = float(self.C.path("loot.pickup_radius", 4.0))
        for p in world.pickups.values():
            if not p.ours and hdist(me.pos, p.pos) <= pr and p.kind not in held_slots(me):
                self.take(ctx, "L7", "6.7", p.id, None)
                return

    # -------------------------------------------------------------- AIM6 / AIM9 defaults
    def _aim6(self, ctx) -> None:
        """AIM6 GP "torso on the biggest threat": a visible hostile mech, highest DPS(j -> us), nearest,
        lower id. In M3A only if within its weapon reach of us (the ambush bearing keeps the aim)."""
        world, A = ctx.world, ctx.A
        best = None
        for tr in world.visible_mechs():
            ta = A.per_track.get(tr.id)
            if ta is None:
                continue
            if self.goal == "M3A" and not ta.in_its_reach:
                continue
            k = (-ta.dps_to_us, ta.d, tr.id)
            if best is None or k < best[0]:
                best = (k, tr)
        if best is not None:
            self.aim_track(ctx, "AIM6", "AIM6", best[1])

    # -------------------------------------------------------------- spots (8.4, H7)
    def _spot_kind(self, ctx) -> str:
        pb = ctx.A.playbook or ctx.world.me.playbook
        tc = ctx.A.terrain_call or self.terrain_call
        if pb == "PB_Z" and tc in ("LIGHTNING", "MIXED"):   # RZ4 roof plan
            return "roof"
        return "hold"

    @staticmethod
    def spot_key(pt) -> Tuple[int, int]:
        return (int(round(pt[0])), int(round(pt[1])))

    def _spot_key(self):
        return self.spot[4] if self.spot is not None else "none"

    def _refresh_spot(self, ctx) -> None:
        t = ctx.world.tick.t
        if self.spot is not None:
            pt, _, _, kind, key = self.spot
            valid = True
            try:
                valid = bool(ctx.nav.spot_valid(kind, pt, ctx.world, ctx.A))
            except Exception:
                valid = True
            if not valid or not self.exit_memory.clear("SPOT", key, t):
                self.spot = None
        if self.spot is None:
            for kind in (self._spot_kind(ctx), "hold"):
                try:
                    best = ctx.nav.best_spot(kind, ctx.world, ctx.A)
                except Exception:
                    best = None
                if best:
                    key = self.spot_key(best[0])
                    if self.exit_memory.clear("SPOT", key, t):
                        self.spot = (best[0], best[1], int(best[2] or 0), kind, key)
                        break
                if kind == "hold":
                    break

    # ============================================================== goals: entry (_e_), exit (_x_), run (_r_)
    # ---- M2U ROTATE_URGENT
    def _e_M2U(self, ctx, active, t):
        C = self.C
        thr = C.ZONE_SLACK_SOFT_S + (C.ROTATE_URGENT_EXIT_MARGIN_S if active else 0.0)
        return ctx.A.slack < thr, "*"

    def _init_M2U(self, ctx, gs):
        gs.data["t0"] = 0.0
        gs.data["eta0"] = max(0.0, ctx.A.zone_eta)

    def _x_M2U(self, ctx, gs, t):
        C = self.C
        if ctx.A.slack >= C.ZONE_SLACK_SOFT_S + C.ROTATE_URGENT_EXIT_MARGIN_S:
            return "exit"
        if gs.active_s - gs.data.get("t0", 0.0) >= C.ROTATE_TIMEOUT_K * gs.data.get("eta0", 0.0) + C.ROTATE_TIMEOUT_ADD_S:
            gs.data["t0"] = gs.active_s                      # replan along the field and stay (no memory)
            gs.data["eta0"] = max(0.0, ctx.A.zone_eta)
            gs.data["timeouts"] = gs.data.get("timeouts", 0) + 1
            self._paths.pop("m2u", None)
            self._log(t, "timer", "M2U", "M2U", "*", "timeout_replan")
        return None

    def _r_M2U(self, ctx, gs) -> str:
        C, me = self.C, ctx.world.me
        d = None
        try:
            d = ctx.nav.zone_descent(me.pos)
        except Exception:
            d = None
        zone = ctx.world.zone
        if d is None:
            d = unit(zone.next_center[0] - me.pos[0], zone.next_center[1] - me.pos[1])
        inside = hdist(me.pos, zone.next_center) < max(0.0, zone.next_radius - C.ZONE_EDGE_MARGIN_M)
        if inside and self.spot is not None:
            self.walk_to(ctx, "m2u", "M2U", self.spot[0])
        else:
            self.legs(ctx, "M2U", d, 1.0)
            if d is not None:
                self.aim_dir(ctx, "AIM9", "M2U", math.atan2(d[1], d[0]))
        self.boost(ctx, "M2U", ctx.world.me.energy > C.BOOST_FLOOR_TRAVEL, C.BOOST_FLOOR_TRAVEL)
        return "M2U"

    # ---- M6 RECOVER
    def _e_M6(self, ctx, active, t):
        C, me = self.C, ctx.world.me
        if active:
            return me.hp < C.HEAL_EXIT_FRAC * me.max_hp, "*"
        return (me.hp < C.LOW_HP_FRAC * me.max_hp and t - self.last_hostile_los_t >= C.RECOVER_NO_LOS_S), "*"

    def _x_M6(self, ctx, gs, t):
        C, me = self.C, ctx.world.me
        if me.hp >= C.HEAL_EXIT_FRAC * me.max_hp:
            return "healed"
        if gs.active_s >= C.RECOVER_MAX_S:
            self.exit_memory.write("RECOVER", "*", t + C.RECOVER_RETRY_S)
            return "timeout"
        return None

    def _r_M6(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        nook = gs.data.get("nook")
        if nook is None:
            try:
                best = ctx.nav.best_spot("nook", world, ctx.A)
            except Exception:
                best = None
            nook = best[0] if best else (me.pos[0], me.pos[1])
            gs.data["nook"] = nook
        if hdist(me.pos, nook) > C.HOLD_SPOT_TOL_M:
            self.walk_to(ctx, "m6", "R9.1", nook, C.SILENT_MOVE, silent=True)
        else:
            self.legs(ctx, "R9.1", None)                                 # still (regen)
        # take within RECOVER_LOOT_R
        pr = float(C.path("loot.pickup_radius", 4.0))
        for p in world.pickups.values():
            if hdist(me.pos, p.pos) <= min(pr, C.RECOVER_LOOT_R) and not p.ours:
                ok, slot, _ = self.wants_pickup(world, p.kind)
                if ok and slot is None:
                    self.take(ctx, "L7", "M6", p.id)
                    break
        self._scan_aim(ctx, "M6")
        return "R9.1"

    # ---- M7 DROP
    def _e_M7(self, ctx, active, t):
        if active:
            return (self.gs.key in ctx.world.drops), self.gs.key
        for dr in sorted(ctx.world.drops.values(), key=lambda d: hdist(ctx.world.me.pos, d.pos)):
            if not self.exit_memory.clear("DROP", dr.id, t):
                continue
            ok, _hard = tactics_drop.go_test(ctx, dr)
            if ok:
                return True, dr.id
        return False, None

    def _init_M7(self, ctx, gs):
        gs.data["fails"] = 0
        gs.data["last_check"] = ctx.world.tick.t
        gs.data["hp0"] = ctx.world.me.hp
        self.sub = "APPROACH"

    def _x_M7(self, ctx, gs, t):
        why, until = tactics_drop.check_exit(ctx, gs)
        if why:
            if until is not None:
                self.exit_memory.write("DROP", gs.key, until)
            return why
        return None

    def _r_M7(self, ctx, gs) -> str:
        self.sub = self.sub or "APPROACH"
        return tactics_drop.run(self.sub, ctx)

    def _contact_seen(self, world, gh) -> bool:
        """M3A/M8 "contact seen": a visible mech on the ghost's bearing (within GHOST_MERGE_DEG) or within
        GHOST_REACH_M of its estimate. Other visible hostiles are handled by L5/L6 and AIM6."""
        if gh is None:
            return False
        me = world.me
        gb = bearing(me.pos, gh.pos)
        tol = math.radians(self.C.GHOST_MERGE_DEG)
        for tr in world.visible_mechs():
            if abs(wrap(bearing(me.pos, tr.pos) - gb)) <= tol or hdist(tr.pos, gh.pos) <= self.C.GHOST_REACH_M:
                return True
        return False

    # ---- M3A AMBUSH (R4.3)
    def _responses(self, ctx) -> Dict[int, frozenset]:
        return getattr(ctx.ostate, "responses", None) or {}

    def _e_M3A(self, ctx, active, t):
        world = ctx.world
        if active:
            return (self.gs.key in world.ghosts), self.gs.key
        for gid, resp in sorted(self._responses(ctx).items()):
            if "AMBUSH" in resp and gid in world.ghosts and self.exit_memory.clear("AMBUSH", gid, t):
                return True, gid
        return False, None

    def _x_M3A(self, ctx, gs, t):
        C, world = self.C, ctx.world
        g = gs.key
        if self._contact_seen(world, world.ghosts.get(g)):
            self.exit_memory.write("AMBUSH", g, ExitMemory.PERMANENT)
            return "seen"
        if g not in world.ghosts:
            self.exit_memory.write("AMBUSH", g, ExitMemory.PERMANENT)
            return "expired"
        if gs.active_s >= C.AMBUSH_MAX_S:
            self.exit_memory.write("AMBUSH", g, ExitMemory.PERMANENT)
            self.exit_memory.write("AMBUSH", "*", t + C.AMBUSH_RETRY_S)
            return "timeout"
        return None

    def _init_M3A(self, ctx, gs):
        """R4.3 spot: a cover2 corner cell within AMBUSH_SEARCH_R minimizing the bearing difference to the
        ghost with a wall between it and the ghost (one LOS check per candidate, nearest first)."""
        C, world, me = self.C, ctx.world, ctx.world.me
        gh = world.ghosts.get(gs.key)
        cell = (me.pos[0], me.pos[1])
        if gh is not None:
            try:
                cells = self.map.cover2_cells(me.pos[0], me.pos[1], C.AMBUSH_SEARCH_R)[:C.COVER_URGENT_CANDIDATES]
            except Exception:
                cells = []
            b0 = bearing(me.pos, gh.pos)
            best = None
            for (x, y, bits) in cells:
                if not (bits & 0x40):            # COVER_CORNER
                    continue
                try:
                    z = self.map.surface_h(x, y)
                    if self.map.los((x, y, z + 8.0), (gh.pos[0], gh.pos[1], z + 4.5)):
                        continue
                except Exception:
                    continue
                k = abs(wrap(bearing((x, y), gh.pos) - b0))
                if best is None or k < best[0]:
                    best = (k, (x, y))
            if best is not None:
                cell = best[1]
        gs.data["cell"] = cell

    def _r_M3A(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        cell = gs.data.get("cell") or (me.pos[0], me.pos[1])
        if hdist(me.pos, cell) > C.HOLD_SPOT_TOL_M:
            self.walk_to(ctx, "m3a", "R4.3", cell, C.SILENT_MOVE, silent=True)
        else:
            self.legs(ctx, "R4.3", None)                                  # still (silent)
        gh = world.ghosts.get(gs.key)
        if gh is not None:
            self.aim_dir(ctx, "AIM9", "R4.3", bearing(me.pos, gh.pos))     # ambush bearing (outranks AIM6 here)
        return "R4.3"

    # ---- M3H HIDE (R5.3)
    def _hide_pred(self, ctx, t) -> bool:
        C, world, A = self.C, ctx.world, ctx.A
        seen = [tr for tr in world.tracks.values() if tr.kind == "mech" and t - tr.last_seen_t <= C.HIDE_CLEAR_S]
        if len(seen) < 2:
            return False
        for tr in seen:
            ta = A.per_track.get(tr.id)
            if ta is not None and ta.aiming_at_us:
                return False
        return not self.fv.win

    def _e_M3H(self, ctx, active, t):
        if active:
            return True, "*"       # own exit handles "predicate false for MODE_MIN_HOLD_S"
        return self._hide_pred(ctx, t), "*"

    def _x_M3H(self, ctx, gs, t):
        C = self.C
        if self._hide_pred(ctx, t):
            gs.data["false_since"] = None
        elif gs.data.get("false_since") is None:
            gs.data["false_since"] = t
        fs = gs.data.get("false_since")
        if fs is not None and t - fs >= C.MODE_MIN_HOLD_S:
            self.exit_memory.write("HIDE", "*", t + C.HIDE_RETRY_S)
            return "exit"
        if gs.active_s >= C.HIDE_MAX_S:
            self.exit_memory.write("HIDE", "*", t + C.HIDE_RETRY_S)
            return "timeout"
        return None

    def _r_M3H(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        t = world.tick.t
        seen = sorted((tr for tr in world.tracks.values() if tr.kind == "mech" and t - tr.last_seen_t <= C.HIDE_CLEAR_S),
                      key=lambda tr: (hdist(me.pos, tr.pos), tr.id))
        near = seen[0] if seen else None
        pt = gs.data.get("cover")
        if pt is None and near is not None:
            pt = self._cover_point(ctx, [near], C.HIDE_MAX_S)
            gs.data["cover"] = pt                     # kept for the episode: reaching cover does not end it
        if pt is not None and hdist(me.pos, pt) > C.HOLD_SPOT_TOL_M:
            self.walk_to(ctx, "m3h", "R5.3", pt, C.SILENT_MOVE, silent=True)
        else:
            self.legs(ctx, "R5.3", None)
        if near is not None:
            self.aim_track(ctx, "AIM9", "R5.3", near)
        return "R5.3"

    # ---- M8 INVESTIGATE (R4.2)
    def _inv_point_clear(self, pt, t) -> bool:
        r = self.C.INVESTIGATE_CLEAR_R
        self.inv_points = [(p, u) for (p, u) in self.inv_points if u > t]
        return all(hdist(p, pt) > r for (p, _) in self.inv_points)

    def _e_M8(self, ctx, active, t):
        C, world, A, me = self.C, ctx.world, ctx.A, ctx.world.me
        if active:
            return (self.gs.key in world.ghosts), self.gs.key
        pb = A.playbook or me.playbook
        tc = A.terrain_call or self.terrain_call
        window = (C.PHASE_T3_SHRINK <= t < C.PHASE_T4_SHRINK and tc == "LIGHTNING") or (C.INVESTIGATE_ANYTIME and t < C.ENDGAME_T)
        if pb != "PB_L" or not window or world.tick.alive <= C.ALIVE_ENDGAME_N:
            return False, None
        if me.hp < C.INVESTIGATE_MIN_HP_FRAC * me.max_hp or me.energy < C.RUSH_ENERGY_MIN or A.slack < C.ZONE_SLACK_SOFT_S:
            return False, None
        for gid, resp in sorted(self._responses(ctx).items()):
            gh = world.ghosts.get(gid)
            if "INVESTIGATE" not in resp or gh is None or hdist(me.pos, gh.pos) > C.INVESTIGATE_MAX_R:
                continue
            others = [o for o in world.ghosts.values() if o.id != gid and hdist(o.pos, gh.pos) <= C.INVESTIGATE_CLEAR_R]
            others += [tr for tr in world.tracks.values() if tr.kind == "mech" and hdist(tr.pos, gh.pos) <= C.INVESTIGATE_CLEAR_R
                       and t - tr.last_seen_t <= C.TRACK_FORGET_S]
            if others or not self.exit_memory.clear("INVESTIGATE", gid, t) or not self._inv_point_clear(gh.pos, t):
                continue
            return True, gid
        return False, None

    def _init_M8(self, ctx, gs):
        gh = ctx.world.ghosts.get(gs.key)
        gs.data["pt"] = gh.pos if gh is not None else (ctx.world.me.pos[0], ctx.world.me.pos[1])

    def _x_M8(self, ctx, gs, t):
        C, world, me = self.C, ctx.world, ctx.world.me
        gh = world.ghosts.get(gs.key)
        pt = gh.pos if gh is not None else gs.data.get("pt", (me.pos[0], me.pos[1]))
        why = None
        if self._contact_seen(world, gh):
            why = "seen"
        elif gh is None:
            why = "expired"
        elif hdist(me.pos, pt) <= C.GHOST_REACH_M:
            why = "reached"
        elif gs.active_s >= C.INVESTIGATE_MAX_S:
            why = "timeout"
        if why:
            self.exit_memory.write("INVESTIGATE", gs.key, ExitMemory.PERMANENT)
            self.exit_memory.write("INVESTIGATE", ("pt",) + self.spot_key(pt), t + C.INVESTIGATE_RETRY_S)
            self.inv_points.append(((pt[0], pt[1]), t + C.INVESTIGATE_RETRY_S))
        return why

    def _r_M8(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        gh = world.ghosts.get(gs.key)
        pt = gh.pos if gh is not None else gs.data["pt"]
        near_contact = hdist(me.pos, pt) <= C.SILENT_NEAR_R
        self.walk_to(ctx, "m8", "R4.2", pt, C.SILENT_MOVE if near_contact else 1.0, silent=near_contact)
        self.aim_dir(ctx, "AIM9", "R4.2", bearing(me.pos, pt))
        return "R4.2"

    # ---- M1 LOOT (8.3)
    def _r72_ok(self, ctx, t, npc=None, farming: bool = False) -> bool:
        """R7.2 gates: no mech seen and no near/mid gun/lightning/laser sound for NPC_SAFE_QUIET_S; hp; time;
        slack; lightning energy or gun ammo.
        Integration gap-fills: the quiet gates are waived for an NPC that damaged us within SELF_DEFENSE_S
        (an aggroed drone chases and shoots: fight it instead of looting under fire); while a farm is under
        way (`farming`) or in that self-defense case the gun gate counts the magazine (reloads refill it),
        so the farm is not aborted after the first shots."""
        C, world, me = self.C, ctx.world, ctx.world.me
        if t >= C.NPC_FARM_END_T or me.hp < C.NPC_FARM_MIN_HP_FRAC * me.max_hp or ctx.A.slack < C.ZONE_SLACK_SOFT_S:
            return False
        defend = npc is not None and world.damaged_us_recently(npc.id, C.SELF_DEFENSE_S)
        if not defend:
            if t - self.last_threat_sound_t < C.NPC_SAFE_QUIET_S:
                return False
            if any(tr.kind == "mech" and t - tr.last_seen_t < C.NPC_SAFE_QUIET_S for tr in world.tracks.values()):
                return False
        held = held_slots(me)
        if "lightning" in held and me.energy >= C.NPC_LIGHTNING_MIN_ENERGY:
            return True
        gun = me.weapons[held["gun"]] if "gun" in held else None
        if gun is None:
            return False
        if farming or defend:
            return float(gun.stats.get("mag", C.path("weapons.gun.mag", 12))) >= C.NPC_GUN_MIN_AMMO
        return (gun.ammo or 0) >= C.NPC_GUN_MIN_AMMO

    def _r13_active(self, world, t) -> bool:
        """R1.3: t >= LOOT_END_T, slot 1 empty / no laser or lightning held, until LOOT_HARD_END_T."""
        C = self.C
        held = held_slots(world.me)
        return C.LOOT_END_T <= t < C.LOOT_HARD_END_T and "laser" not in held and "lightning" not in held

    def _e_M1(self, ctx, active, t):
        C, world = self.C, ctx.world
        if t < C.LOOT_END_T or self._r13_active(world, t):
            return True, "*"                                   # always true (best target or exploration)
        held = held_slots(world.me)
        r = C.HOLD_LOOT_R_NO_WEAPON if ("laser" not in held and "lightning" not in held) else C.HOLD_LOOT_R
        best = self._best_loot(ctx, t, max_r=r)
        return best is not None, "*"

    def _x_M1(self, ctx, gs, t):
        C = self.C
        if t < C.LOOT_END_T or self._r13_active(ctx.world, t):
            return None                                        # leaves only by preemption or window end
        if self.loot_item is None and gs.active_s > 0:
            return "done"
        return None

    def _loot_value(self, ctx, kind, t, empty_slot) -> float:
        C = self.C
        if empty_slot:
            want = self.desired_loadout or "ZL"
            best = {"L": "lightning", "Z": "laser"}
            letters = [best[c] for c in want if c in best]
            if kind == "mortar":
                f = C.KIND_FACTOR_MORTAR
                if t < C.EARLY_MORTAR_T and any(p.kind in ("laser", "lightning") for p in ctx.world.pickups.values()):
                    f = 0.0
            elif letters and kind == letters[0]:
                f = 1.0
            elif kind in ("laser", "lightning"):
                f = C.KIND_FACTOR_OTHER
            else:
                f = 0.0
            base = C.VAL_WEAPON_EMPTY_LATE if t >= C.LOOT_END_T else C.VAL_WEAPON_EMPTY
            return base * f
        ok, _, gain = self.wants_pickup(ctx.world, kind)
        if not ok or not self._switch_gates(ctx.world, t):
            return 0.0
        return C.VAL_WEAPON_UPGRADE * max(0.0, gain)

    def _travel(self, ctx, pt) -> float:
        tt = None
        try:
            tt = ctx.nav.travel_time(pt[0], pt[1])
        except Exception:
            tt = None
        if tt is None:
            ws = float((ctx.world.me.stats or {}).get("walk_speed", 11.7)) or 11.7
            tt = hdist(ctx.world.me.pos, pt) / ws * self.C.ZONE_FALLBACK_DETOUR_K
        return tt

    def _best_loot(self, ctx, t, max_r=None):
        """8.3 score_i = V x Risk / (T_travel + T_handle + pad); returns (key, point, score, eta, id, kind)."""
        C, world, me = self.C, ctx.world, ctx.world.me
        held = held_slots(me)
        empty = None in [w.kind if w else None for w in (me.weapons or [None, None])]
        cands = []
        for p in world.pickups.values():
            if p.ours or p.kind in held:
                continue
            v = self._loot_value(ctx, p.kind, t, empty)
            stale = C.LOOT_STALE_MULT if t - p.last_seen > C.LOOT_STALE_S else 1.0
            cands.append((("p", p.id), (p.pos[0], p.pos[1]), v * stale, C.WEAPON_HANDLE_S, p.id, p.kind))
        for c in world.cubes.values():
            cands.append((("c", c.id), (c.pos[0], c.pos[1]), C.VAL_CUBE, 0.0, c.id, "cube"))
        r13 = self._r13_active(world, t)
        gun = world.me.weapons[held["gun"]] if "gun" in held else None
        gun_dps = (float(gun.stats.get("damage", 45)) / max(0.05, float(gun.stats.get("cooldown", 0.24)))) if gun else 0.0
        crate_skip = self._crate_skip(world, t)
        for cr in world.crates.values():
            if crate_skip or (gun_dps <= 0 and "lightning" not in held):
                continue
            hp = cr.hp if cr.hp is not None else float(C.path("crates.hp", 300))
            handle = hp / max(1.0, gun_dps or 240.0)
            cands.append((("k", cr.id), (cr.pos[0], cr.pos[1]), C.VAL_CRATE_NO_WEAPON if r13 else C.VAL_CRATE, handle, cr.id, "crate"))
        r72 = self._r72_ok(ctx, t)                                   # R7.2 NPC farming
        npc_hp = float(C.path("npc.hp", 350))
        dps = gun_dps if "lightning" not in held else max(gun_dps, 240.0)
        for tr in world.tracks.values():
            if tr.kind == "npc" and (tr.hp is None or tr.hp > 0) and (r72 or self._r72_ok(ctx, t, tr)):
                hp = tr.hp if tr.hp is not None else npc_hp
                cands.append((("n", tr.id), (tr.pos[0], tr.pos[1]), C.VAL_NPC, hp / max(1.0, dps) + C.NPC_HANDLE_PAD_S,
                              tr.id, "npc"))
        for (pt, bt) in getattr(world, "loot_bursts", ()) or ():
            quiet = all(hdist(tr.pos, pt) > C.LOOT_ENEMY_NEAR_R or t - tr.last_seen_t > C.DEATH_LOOT_QUIET_S
                        for tr in world.tracks.values() if tr.kind == "mech")
            if quiet and t - bt < C.TRACK_FORGET_S:
                cands.append((("b", self.spot_key(pt)), (pt[0], pt[1]), C.VAL_DEATH_LOOT, C.DEATH_HANDLE_S, None, "burst"))
        best = None
        scores = {}
        zone = world.zone
        try:
            my_eta = ctx.nav.zone_eta(me.pos)[0]
        except Exception:
            my_eta = 0.0
        for key, pt, v, handle, iid, kind in cands:
            if v <= 0 or not self.exit_memory.clear("LOOT", key, t):
                continue
            if max_r is not None and hdist(me.pos, pt) > max_r:
                continue
            risk = 1.0
            for tr in world.tracks.values():
                if tr.kind == "mech" and t - tr.last_seen_t < C.TRACK_FORGET_S and hdist(tr.pos, pt) < C.LOOT_ENEMY_NEAR_R:
                    risk *= C.LOOT_ENEMY_PENALTY
            for gh in world.ghosts.values():
                if hdist(gh.pos, pt) < C.LOOT_ENEMY_NEAR_R:
                    risk *= C.LOOT_ENEMY_PENALTY
            nbp = self._neighbor_pos(world)                          # R1.2: targets the neighbor reaches first
            if nbp is not None and hdist(nbp, pt) < hdist(me.pos, pt) and hdist(nbp, pt) >= C.LOOT_ENEMY_NEAR_R:
                risk *= C.LOOT_ENEMY_PENALTY
            try:
                heat = world.heat_at(pt[0], pt[1])
            except Exception:
                heat = 0.0
            risk *= (1.0 - C.DANGER_HEAT_W * min(1.0, heat / C.HEAT_SAT))
            tt = self._travel(ctx, pt)
            outside = hdist(pt, zone.next_center) > zone.next_radius
            if outside:
                if ctx.A.slack - tt < C.ZONE_SLACK_SOFT_S:
                    continue                                        # zone_ok = 0
                try:
                    tt += C.ZONE_AWAY_PENALTY * max(0.0, ctx.nav.zone_eta((pt[0], pt[1], me.pos[2]))[0] - my_eta)   # R1.1
                except Exception:
                    pass
            score = v * risk / (tt + handle + C.LOOT_TIME_PAD_S)
            scores[key] = score
            if score > 0 and (best is None or score > best[2]):
                best = (key, pt, score, tt, iid, kind)
        if max_r is None:
            self._loot_scores = scores      # H4 compares against the current item's score of this tick
        return best

    def _crate_skip(self, world, t) -> bool:
        """R1.5 "skip while a mech is visible or a near/mid gun/lightning sound occurred within CRATE_QUIET_S":
        crates are no M1 targets then (standing at a crate without shooting it under fire helps nobody)."""
        return bool(world.visible_mechs()) or t - self.last_threat_sound_t < self.C.CRATE_QUIET_S

    def _r_M1(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        t = world.tick.t
        best = self._best_loot(ctx, t)
        cur = self.loot_item
        if cur is not None and cur[7] == "crate" and self._crate_skip(world, t):
            cur = None                      # R1.5 skip (no blacklist: the crate comes back when it is quiet)
        if cur is not None:                 # H4: current item's score now (stale selection-time score flip-flops)
            cur = cur[:2] + (self._loot_scores.get(cur[0], 0.0),) + cur[3:]
        # current item still valid? (H4: immediate switch if gone or blacklisted)
        if cur is not None:
            key = cur[0]
            gone = not self.exit_memory.clear("LOOT", key, t) or not self._loot_exists(world, key)
            if gone:
                cur = None
            elif gs.active_s - cur[5] >= C.LOOT_TARGET_TIMEOUT_K * cur[4] + C.LOOT_TARGET_TIMEOUT_ADD_S:
                self.exit_memory.write("LOOT", key, t + C.LOOT_BLACKLIST_S)    # item timeout -> blacklist
                self._log(t, "loot", key, None, key, "timeout")
                cur = None
        if best is not None:
            if cur is None:
                cur = (best[0], best[1], best[2], t, best[3], gs.active_s, best[4], best[5])
            elif best[0] != cur[0] and best[2] > cur[2] * C.LOOT_SWITCH_MARGIN and t - cur[3] >= C.LOOT_MIN_HOLD_S:
                cur = (best[0], best[1], best[2], t, best[3], gs.active_s, best[4], best[5])
            elif best[0] == cur[0]:
                cur = (cur[0], best[1], best[2], cur[3], cur[4], cur[5], cur[6], cur[7])
        self.loot_item = cur
        silent = any(hdist(me.pos, tr.pos) < C.SILENT_NEAR_R for tr in world.tracks.values()
                     if tr.kind == "mech" and t - tr.last_seen_t < C.TRACK_FORGET_S)
        speed = C.SILENT_MOVE if silent else 1.0
        if cur is None:
            if t < C.LOOT_END_T or self._r13_active(world, t):
                return self._explore(ctx, speed)                       # 8.3 exploration
            self.legs(ctx, "M1", None)
            return "M1"
        key, pt = cur[0], cur[1]
        kind = cur[7]
        if kind == "burst" and hdist(me.pos, pt) <= float(C.path("loot.pickup_radius", 4.0)) + C.WAYPOINT_REACH_M:
            # 8.3 death loot burst reached: its cubes / weapons are items of their own now ("taken")
            self.exit_memory.write("LOOT", key, t + C.LOOT_BLACKLIST_S)
            self._log(t, "loot", key, None, key, "burst_reached")
            self.loot_item = None
            return self._r_M1(ctx, gs)
        rule = "R1.5" if kind == "crate" else ("R7.2" if kind == "npc" else "8.3")
        if kind == "npc":
            return self._farm_npc(ctx, cur, speed)
        if kind == "crate" and hdist(me.pos, pt) <= C.CRATE_SHOOT_R:
            self.legs(ctx, rule, None)
            cr = world.crates.get(cur[6])
            if cr is not None and not world.visible_mechs():
                yaw = bearing(me.pos, cr.pos)
                pitch = math.atan2(cr.pos[2] + 1.0 - (me.muzzle_z or me.pos[2] + 7.0), max(0.5, hdist(me.pos, cr.pos)))
                self.aim_dir(ctx, "AIM9", rule, yaw, pitch, cr.id)
                kinds = {"gun"}
                if hdist(me.pos, cr.pos) <= C.LIGHTNING_RANGE and me.energy >= C.CRATE_LIGHTNING_MIN_ENERGY:
                    kinds.add("lightning")
                self.fire(ctx, cr.id, kinds)
            return rule
        self.walk_to(ctx, "m1", rule, pt, speed, silent=silent)
        nbp = self._neighbor_pos(world)
        if nbp is not None:
            ws = float((me.stats or {}).get("walk_speed", 11.7)) or 11.7
            if kind not in ("cube", "crate", "burst") and hdist(me.pos, pt) / ws <= hdist(nbp, pt) / ws + C.CONTEST_ETA_MARGIN_S:
                self.boost(ctx, "R1.2", me.energy >= C.CONTEST_BOOST_MIN_ENERGY, 0.0)   # contested weapon
            if math.fmod(t, C.SCAN_PERIOD_S) < C.SCAN_LOOK_S:                             # check its bearing
                self.aim_dir(ctx, "AIM9", "R1.2", bearing(me.pos, nbp))
        if kind not in ("cube", "crate", "burst") and cur[6] is not None:
            p = world.pickups.get(cur[6])
            if p is not None and hdist(me.pos, p.pos) <= float(C.path("loot.pickup_radius", 4.0)):
                ok, slot, _ = self.wants_pickup(world, p.kind)
                if ok:
                    writer = "L7" if slot is None else "R0.4"
                    self.take(ctx, writer, rule, p.id, slot)
        self._scan_aim(ctx, "M1")
        return rule

    def _farm_npc(self, ctx, cur, speed) -> str:
        """R7.2: approach to NPC_LIGHTNING_STANDOFF (PB_L) or NPC_GUN_STANDOFF, face it, fire; a failing
        gate aborts and blacklists the NPC for LOOT_BLACKLIST_S."""
        C, world, me = self.C, ctx.world, ctx.world.me
        t = world.tick.t
        tr = world.tracks.get(cur[6])
        if tr is None or not self._r72_ok(ctx, t, tr, farming=True):
            self.exit_memory.write("LOOT", cur[0], t + C.LOOT_BLACKLIST_S)
            self.loot_item = None
            self.legs(ctx, "R7.2", None)
            return "R7.2"
        held = held_slots(me)
        light = "lightning" in held and me.energy >= C.NPC_LIGHTNING_MIN_ENERGY
        stand = C.NPC_LIGHTNING_STANDOFF if light else C.NPC_GUN_STANDOFF
        d = hdist(me.pos, tr.pos)
        if d > stand:
            self.walk_to(ctx, "m1", "R7.2", (tr.pos[0], tr.pos[1]), speed)
        else:
            self.legs(ctx, "R7.2", None)
        self.aim_track(ctx, "AIM9", "R7.2", tr)
        if d <= stand + 5.0:
            self.fire(ctx, tr.id, {"gun", "lightning"} if light else {"gun"}, allow_npc=True)
        return "R7.2"

    @staticmethod
    def _loot_exists(world, key) -> bool:
        kind, iid = key
        if kind == "p":
            return iid in world.pickups
        if kind == "c":
            return iid in world.cubes
        if kind == "k":
            return iid in world.crates
        if kind == "n":
            return iid in world.tracks
        return True

    def _explore(self, ctx, speed) -> str:
        """8.3 exploration: the walkable point EXPLORE_STEP_M out (IMPL_LEG_DIRS directions) maximizing
        unseen cells x (1 + EXPLORE_CENTER_W cos(angle to next_center)); re-chosen on arrival or after
        EXPLORE_REPICK_S."""
        C, world, me = self.C, ctx.world, ctx.world.me
        t = world.tick.t
        ex = self.explore
        if ex is None or t - ex[1] >= C.EXPLORE_REPICK_S or hdist(me.pos, ex[0]) < C.WAYPOINT_REACH_M * 2:
            n = int(C.IMPL_LEG_DIRS)
            zc = world.zone.next_center
            zdir = bearing(me.pos, zc)
            best = None
            near = float(C.path("vision.near_range", 40.0))
            for k in range(n):
                a = 2 * math.pi * k / n
                px, py = me.pos[0] + C.EXPLORE_STEP_M * math.cos(a), me.pos[1] + C.EXPLORE_STEP_M * math.sin(a)
                try:
                    if self.map.clearance(px, py) <= 0:
                        continue
                except Exception:
                    pass
                unseen = 0
                for (ox, oy) in ((0, 0), (near / 2, 0), (-near / 2, 0), (0, near / 2), (0, -near / 2)):
                    try:
                        unseen += 0 if world.seen_at(px + ox, py + oy) else 1
                    except Exception:
                        unseen += 1
                score = unseen * (1 + C.EXPLORE_CENTER_W * math.cos(wrap(a - zdir)))
                if best is None or score > best[0] + 1e-9:
                    best = (score, (px, py))
            ex = (best[1] if best else (zc[0], zc[1]), t)
            self.explore = ex
        self.walk_to(ctx, "explore", "8.3x", ex[0], speed)
        self._scan_aim(ctx, "M1")
        return "8.3x"

    # ---- M2 ROTATE
    def _detour(self, ctx, t):
        """R0.4 approved detour: a remembered higher-rank pickup on our way within the detour bound."""
        C, world, me = self.C, ctx.world, ctx.world.me
        if not self._switch_gates(world, t) or None in [w.kind if w else None for w in (me.weapons or [None, None])]:
            return None
        zone = world.zone
        try:
            eta_next = ctx.nav.zone_eta(me.pos)[0]
        except Exception:
            eta_next = 0.0
        bound = zone.shrink_end - t - eta_next - C.SWITCH_SAFETY_S
        best = None
        for p in world.pickups.values():
            if p.ours:
                continue
            ok, slot, gain = self.wants_pickup(world, p.kind)
            if not ok or not self.exit_memory.clear("LOOT", ("p", p.id), t):
                continue
            tt = self._travel(ctx, (p.pos[0], p.pos[1]))
            if tt <= bound and (best is None or gain > best[0]):
                best = (gain, p)
        return best[1] if best else None

    def _e_M2(self, ctx, active, t):
        C, world, me = self.C, ctx.world, ctx.world.me
        p = self._detour(ctx, t)
        if p is not None:
            return True, ("detour", p.id)
        sp = self.spot
        zone = world.zone
        if sp is None:
            inside = hdist(me.pos, zone.next_center) < max(C.ZONE_TINY_R, zone.next_radius - C.ZONE_EDGE_MARGIN_M)
            return (not inside), "none"
        pt, _, roof_id, kind, key = sp
        if roof_id:
            try:
                on = self.map.roof_id(me.pos[0], me.pos[1]) == roof_id
            except Exception:
                on = False
            return (not on), key                    # never entered while standing on the chosen roof
        thr = C.HOLD_LEAVE_R if self.goal == "M3" else C.HOLD_ARRIVE_R
        return hdist(me.pos, pt) > thr, key

    def _init_M2(self, ctx, gs):
        pt = self._m2_goal(ctx, gs)
        gs.data["eta0"] = self._travel(ctx, pt) if pt is not None else 0.0

    def _m2_goal(self, ctx, gs):
        world = ctx.world
        if isinstance(gs.key, tuple) and gs.key and gs.key[0] == "detour":
            p = world.pickups.get(gs.key[1])
            return (p.pos[0], p.pos[1]) if p is not None else None
        if self.spot is None or gs.key == "none":
            return world.zone.next_center
        return self.spot[0]

    def _x_M2(self, ctx, gs, t):
        C, world, me = self.C, ctx.world, ctx.world.me
        if isinstance(gs.key, tuple) and gs.key and gs.key[0] == "detour":
            if gs.key[1] not in world.pickups or any(ev.get("type") == "took" for ev in world.events or ()):
                return "arrived"
        elif gs.key == "none":
            if self.spot is not None or hdist(me.pos, world.zone.next_center) < max(C.ZONE_TINY_R, world.zone.next_radius - C.ZONE_EDGE_MARGIN_M):
                return "arrived"
        elif self.spot is None or self.spot[4] != gs.key:
            return "respot"
        else:
            pt, _, roof_id, _, _ = self.spot
            if roof_id:
                try:
                    if self.map.roof_id(me.pos[0], me.pos[1]) == roof_id:
                        return "arrived"
                except Exception:
                    pass
            elif hdist(me.pos, pt) <= C.HOLD_ARRIVE_R:
                return "arrived"
        if gs.active_s >= C.ROTATE_TIMEOUT_K * gs.data.get("eta0", 0.0) + C.ROTATE_TIMEOUT_ADD_S:
            self.exit_memory.write("ROTATE", gs.key, t + C.SPOT_BLACKLIST_S)
            if not (isinstance(gs.key, tuple) and gs.key and gs.key[0] == "detour") and gs.key != "none":
                self.exit_memory.write("SPOT", gs.key, t + C.SPOT_BLACKLIST_S)
                self.spot = None
            elif gs.key != "none":
                self.exit_memory.write("LOOT", ("p", gs.key[1]), t + C.LOOT_BLACKLIST_S)
            return "timeout"
        return None

    def _r_M2(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        pt = self._m2_goal(ctx, gs)
        if pt is None:
            self.legs(ctx, "M2", None)
            return "M2"
        st = self.walk_to(ctx, "m2", "M2", pt)
        if st.direction is not None:
            self.aim_dir(ctx, "AIM9", "M2", math.atan2(st.direction[1], st.direction[0]))
        if ctx.A.slack < C.ZONE_SOFT_MULT * C.ZONE_SLACK_SOFT_S:
            self.boost(ctx, "M2", me.energy > C.BOOST_FLOOR_TRAVEL, C.BOOST_FLOOR_TRAVEL)
        if isinstance(gs.key, tuple) and gs.key and gs.key[0] == "detour":
            p = world.pickups.get(gs.key[1])
            if p is not None and hdist(me.pos, p.pos) <= float(C.path("loot.pickup_radius", 4.0)):
                ok, slot, _ = self.wants_pickup(world, p.kind)
                if ok and self._switch_gates(world, world.tick.t):
                    self.take(ctx, "R0.4", "R0.4", p.id, slot)
        return "M2"

    # ---- M3 HOLD
    def _e_M3(self, ctx, active, t):
        return True, self._spot_key()

    def _init_M3(self, ctx, gs):
        gs.data["shuffle_t"] = ctx.world.tick.t
        gs.data["anchor"] = self.spot[0] if self.spot else (ctx.world.me.pos[0], ctx.world.me.pos[1])

    def _x_M3(self, ctx, gs, t):
        C = self.C
        quiet_since = max(gs.entered_t, self.last_event_t)
        if gs.active_s >= C.HOLD_MAX_S and t - quiet_since >= C.HOLD_MAX_S:
            if gs.key != "none":
                self.exit_memory.write("SPOT", gs.key, t + C.SPOT_BLACKLIST_S)
            self.exit_memory.write("HOLD", gs.key, t + C.SPOT_BLACKLIST_S)
            self.spot = None
            return "timeout"
        return None

    def _r_M3(self, ctx, gs) -> str:
        C, world, me = self.C, ctx.world, ctx.world.me
        if self.spot is not None and self.spot[2]:
            self.sub = "ROOF"
            return tactics_roof.run("ROOF_HOLD", ctx)
        self.sub = "COVER"
        anchor = self.spot[0] if self.spot else gs.data.get("anchor", (me.pos[0], me.pos[1]))
        t = world.tick.t
        if t - gs.data.get("shuffle_t", t) >= C.HOLD_SHUFFLE_S:          # SILENT_MOVE shuffle
            gs.data["shuffle_t"] = t
            a = self.rng.uniform(-math.pi, math.pi)
            gs.data["shuffle_pt"] = (anchor[0] + C.HOLD_SPOT_TOL_M * 0.5 * math.cos(a),
                                     anchor[1] + C.HOLD_SPOT_TOL_M * 0.5 * math.sin(a))
        tgt = gs.data.get("shuffle_pt", anchor)
        if hdist(me.pos, tgt) > C.HOLD_SPOT_TOL_M * 0.25:
            self.legs(ctx, "M3", unit(tgt[0] - me.pos[0], tgt[1] - me.pos[1]), C.SILENT_MOVE, tgt)
        else:
            self.legs(ctx, "M3", None)
        self._scan_aim(ctx, "M3")
        return "M3"

    def _scan_aim(self, ctx, rule) -> None:
        """AIM9 default for goals without a target: none (the arbiter's AIM10 scan sweeps)."""
        return None


def slot_to_replace_local(held, new_kind, terrain_call, C) -> Tuple[Optional[int], float]:
    """8.5 fallback (assess.slot_to_replace preferred): empty -> fill; mortar first; SWAP_RULE 'gp' keeps
    RZ5 (GZ + lightning -> replace the gun); otherwise the higher-rank replacement, tie keeps the gun."""
    import config as _config
    rank = _config.LOADOUT_RANK
    tc = terrain_call if terrain_call in ("LIGHTNING", "LASER", "MIXED") else "MIXED"

    def r(kinds):
        return rank.get(_config.loadout_key([k for k in kinds if k]), {}).get(tc, 0.0)

    if None in held:
        return held.index(None), 1.0
    cur = r(held)
    if "mortar" in held:
        i = held.index("mortar")
        nk = list(held)
        nk[i] = new_kind
        return i, r(nk) - cur
    if getattr(C, "SWAP_RULE", "rank") == "gp" and sorted(held) == ["gun", "laser"] and new_kind == "lightning":
        return held.index("gun"), max(1e-6, r(["laser", "lightning"]) - cur)
    best = None
    for i in (0, 1):
        nk = list(held)
        nk[i] = new_kind
        g = r(nk) - cur
        keep_gun = 1 if held[i] == "gun" else 0
        k = (g, -keep_gun)
        if best is None or k > best[0]:
            best = (k, i, g)
    return best[1], best[2]
