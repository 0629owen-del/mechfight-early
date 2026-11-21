"""Assessment (plan 8.1 DPS, 8.2 fight evaluation, 8.5 loadout, 8.6 drop go/no-go, 8.8 slack, R0.1).

    assess.calibrate_rush_a_min(C)                 # at hello, fills C.RUSH_A_MIN (8.2.1)
    assessor = Assessor(C, mapdata)
    A = assessor.update(world, nav, clock)         # 6.1 step 5 -> contracts.Assessment
    fe = assessor.evaluate_fight(world, A, macro, target_id, ctx)

Extra read-only attributes on Assessor (telemetry / decision layer, not in the contract):
    aim1            [track id] lasers charging at us inside our lightning reach, turn time <= charge left
    shell_hits_us   [shell id] predicted impacts within MORTAR_DODGE_R of our extrapolated position
    laser_next_t    {track id: earliest t its laser can fire again} (laser-at-us timing, 4.5)
    hostile_dps     {track id: DPS on us incl. aim factor} for the members of H
    call_S, call_C, call_R, call_raw, call_n   last R0.1 inputs (median sight, wall share, radius)
    update_ms, los_calls, exceptions, last_error
Module helpers beyond the contract: raw_dps, hit_chance, enemy_wstats, hp_eff_enemy, hp_eff_self,
a_rush, tau_edge, zone_eta_fallback, terrain_stats, terrain_call_from_stats, lightning_cone_half.

A-19: our lightning fire gate uses +-LIGHTNING_CONE_MARGIN x cone/2; the nearest-object check and enemy
reach use +-cone (lightning_cone_half(C, worst=True)). A-20: enemy lightning reach x
Track.lightning_range_mult (world), ghosts x world.ghost_lightning_mult.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import config as _config
from contracts import Assessment, FightEval, Threats, TrackAssess

INF = math.inf
TWO_PI = 2.0 * math.pi
WEAPONS = ("gun", "laser", "lightning", "mortar")
DESIRED = {"LIGHTNING": "GL", "LASER": "GZ", "MIXED": "ZL"}     # R0.2 best loadout per call
CALL_MAX_SAMPLES = 2500            # R0.1 grid subsample cap (one call per next_changed, < 3 ms)


def wrap(a: float) -> float:
    a = math.fmod(a + math.pi, TWO_PI)
    if a <= 0.0:
        a += TWO_PI
    return a - math.pi


def _f(v, d=0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return x if x == x else d


def _interp(pts, d: float) -> float:
    """Piecewise-linear table [(d, p), ...], clamped at the ends (CONTRACTS 7.12)."""
    if not pts:
        return 0.0
    if d <= pts[0][0]:
        return float(pts[0][1])
    for (d0, p0), (d1, p1) in zip(pts, pts[1:]):
        if d <= d1:
            return float(p0) + (float(p1) - float(p0)) * (d - d0) / max(1e-9, d1 - d0)
    return float(pts[-1][1])


# ================================================================== 8.1 DPS model
def hit_chance(base: float, target_cubes: int, C, enemy: bool, size: bool = True) -> float:
    p = base
    if size:
        p *= 1.0 + C.SIZE_HIT_K * max(0, target_cubes)
    if enemy:
        p *= C.ENEMY_AIM_MULT
    return min(C.HIT_P_CAP, p)


def _k(C, cubes: int) -> float:
    return 1.0 + _f(C.path("cubes.damage_per_cube", 0.04), 0.04) * min(max(0, cubes), int(_f(C.path("cubes.stat_cap", 10), 10)))


def enemy_wstats(C, kind: str) -> dict:
    """Enemy weapon numbers: config base values (upgrades unknown, 8.1)."""
    if kind == "npc":
        return C.path("npc", {}) or {}
    return C.path("weapons." + kind, {}) or {}


def raw_dps(kind: str, wstats: dict, shooter_cubes: int, C) -> float:
    """Burst DPS without hit chance / sustain (GL at 0 cubes: gun 187.5 + lightning 240)."""
    if not wstats:
        wstats = enemy_wstats(C, kind)
    cd = _f(wstats.get("cooldown"), 1.0) or 1.0
    if kind == "laser":
        cd = _f(wstats.get("charge"), 0.75) + cd
    return _f(wstats.get("damage")) * _k(C, shooter_cubes) / cd


def dps(kind: str, wstats: dict, d: float, shooter_cubes: int, target_cubes: int, *, rear: bool,
        armor: float, energy: Optional[float], C, enemy: bool, d3: Optional[float] = None, los: bool = True,
        reloading: bool = False, shorted: bool = False, cover: bool = False, ammo: Optional[float] = None,
        range_mult: float = 1.0) -> float:
    """8.1 expected DPS of one weapon from shooter X at target Y, horizontal distance d."""
    if not wstats:
        wstats = enemy_wstats(C, kind)
    k = _k(C, shooter_cubes)
    rm = _f(C.path("mech.rear_mult", 1.5), 1.5) if rear else 1.0
    arm = 1.0 - min(max(armor, 0.0), 0.95)
    if kind == "gun":
        if reloading or d > _f(wstats.get("range"), 230.0):
            return 0.0
        cd = _f(wstats.get("cooldown"), 0.24) or 0.24
        mag = _f(wstats.get("mag"), 12.0)
        rel = _f(wstats.get("reload"), 2.4)
        n = mag if ammo is None else ammo
        sustain = mag * cd / (mag * cd + rel) if C.FIGHT_HORIZON_S > n * cd else 1.0
        p = hit_chance(_interp(C.HIT_GUN, d), target_cubes, C, enemy)
        return _f(wstats.get("damage")) * k / cd * sustain * p * rm * arm
    if kind == "laser":
        if shorted or d > _f(wstats.get("range"), 280.0):
            return 0.0
        p = C.HIT_LASER_COVER if cover else C.HIT_LASER_OPEN
        if enemy:
            p = min(C.HIT_P_CAP, p * C.ENEMY_AIM_MULT)
        per = _f(wstats.get("charge"), 0.75) + _f(wstats.get("cooldown"), 1.65)
        return _f(wstats.get("damage")) * k / max(1e-6, per) * p * rm * arm
    if kind == "lightning":
        dd = d if d3 is None else d3
        if not los or dd > _f(wstats.get("range"), 45.0) * range_mult:
            return 0.0
        cd = _f(wstats.get("cooldown"), 0.25) or 0.25
        e_need = _f(wstats.get("energy"), 3.0) / cd * C.FIGHT_HORIZON_S
        E = _f(C.path("mech.energy", 100), 100.0) if energy is None else max(0.0, energy)
        ef = min(1.0, E / e_need) if e_need > 0 else 1.0
        return _f(wstats.get("damage")) * k / cd * ef * rm * arm
    if kind == "mortar":
        v, g = _f(wstats.get("speed"), 72.0), _f(wstats.get("gravity"), 22.0) or 22.0
        dmax = v * v / g
        if enemy:
            dmin = _f(wstats.get("splash_radius"), 7.0) + _f(C.path("mech.radius", 2.5), 2.5) + 2.0
            dmg = _f(wstats.get("damage")) + _f(wstats.get("splash"))     # A-2 worst case
        else:
            dmin = C.MORTAR_SELF_SAFE_M
            dmg = _f(wstats.get("damage"))
        if d < dmin or d > dmax:
            return 0.0
        p = hit_chance(_interp(C.HIT_MORTAR, d), target_cubes, C, enemy)
        return dmg * k * p / (_f(wstats.get("cooldown"), 2.6) or 2.6) * arm
    if kind == "npc":
        if d > C.NPC_AGGRO_R:
            return 0.0
        return _f(wstats.get("damage"), 25.0) / (_f(wstats.get("cooldown"), 1.0) or 1.0) * C.HIT_NPC * arm
    return 0.0


def weapon_reach(kind: str, wstats: dict, C, range_mult: float = 1.0) -> float:
    if not wstats:
        wstats = enemy_wstats(C, kind)
    if kind == "lightning":
        return _f(wstats.get("range"), 45.0) * range_mult
    if kind == "mortar":
        g = _f(wstats.get("gravity"), 22.0) or 22.0
        return _f(wstats.get("speed"), 72.0) ** 2 / g
    if kind == "npc":
        return C.NPC_AGGRO_R
    return _f(wstats.get("range"), 0.0)


def hp_eff_enemy(tr, C) -> float:
    if tr.hp is not None:
        return max(1.0, tr.hp)
    if tr.kind == "npc":
        return _f(C.path("npc.hp", 350), 350.0)
    cap = int(_f(C.path("cubes.stat_cap", 10), 10))
    return _f(C.path("mech.hp", 1400), 1400.0) * (1.0 + _f(C.path("cubes.hp_per_cube", 0.04), 0.04) * min(tr.cubes, cap))


def hp_eff_self(me) -> float:
    return max(1.0, me.hp) / max(0.05, 1.0 - me.armor)


def lightning_cone_half(C, worst: bool) -> float:
    """A-19: worst=True -> +-cone_deg (nearest-object / threat), else +-MARGIN x cone/2 (our fire gate)."""
    cone = math.radians(_f(C.path("weapons.lightning.cone_deg", 30), 30.0))
    return cone if worst else C.LIGHTNING_CONE_MARGIN * cone / 2.0


# ================================================================== 8.5 loadout
def loadout_rank(kinds, terrain_call: str) -> float:
    row = _config.LOADOUT_RANK.get(_config.loadout_key([k for k in kinds if k]), {})
    return float(row.get(terrain_call, row.get("MIXED", 0.0)))


def slot_to_replace(held: List[Optional[str]], new_kind: str, terrain_call: str, C) -> Tuple[Optional[int], float]:
    """8.5: (slot, gain). Empty slot -> fill it. Full: a mortar is always the slot replaced; SWAP_RULE
    "gp" adds RZ5 (GZ + lightning -> replace the gun, gain counted > 0); otherwise the replacement with
    the higher rank, tie -> keep the gun. The slot is returned even when gain <= 0 (the caller takes
    iff gain > 0 and the R0.4 gates hold); None when the kind is already held or unknown."""
    held = list(held) + [None] * (2 - len(held))
    held = held[:2]
    if new_kind not in WEAPONS or new_kind in held:
        return None, 0.0
    cur = loadout_rank([k for k in held if k], terrain_call)
    for i, k in enumerate(held):
        if k is None:
            return i, loadout_rank([x for x in held if x] + [new_kind], terrain_call) - cur

    def after(i):
        return loadout_rank([new_kind if j == i else held[j] for j in range(2)], terrain_call)

    if "mortar" in held:                                            # (1)
        i = held.index("mortar")
        return i, after(i) - cur
    if C.SWAP_RULE == "gp" and set(held) == {"gun", "laser"} and new_kind == "lightning":   # (2) RZ5
        i = held.index("gun")
        return i, max(after(i) - cur, C.IMPL_SCORE_EPS)
    r0, r1 = after(0), after(1)                                     # (3)
    if r0 > r1 + 1e-9:
        i = 0
    elif r1 > r0 + 1e-9:
        i = 1
    else:
        i = 1 if held[0] == "gun" else 0                            # tie -> keep the gun
    return i, max(r0, r1) - cur


# ================================================================== 8.2.1 A_rush and calibration
def a_rush(C, d: float, v_c: float, hp_us: float, hp_j: float, energy: float, boost_drain: float,
           us_on_j: Callable[[float, float], float], j_on_us: Callable[[float], float],
           other_dps: float = 0.0) -> float:
    """8.2.1: approach from d to BRAWL_DIST in IMPL_RUSH_STEP_M segments (anchored at BRAWL_DIST, DPS
    at each segment midpoint) at closing speed v_c under fire from j (and the rest of H at constant
    DPS other_dps); A_rush = ((hp_us - loss) / DPS(H->us @ brawl)) / (hp_j / DPS(us->j @ brawl, E'))."""
    if v_c <= C.RUSH_CLOSING_MIN:
        return 0.0
    br = C.BRAWL_DIST
    step = max(1e-3, float(C.IMPL_RUSH_STEP_M))
    loss = 0.0
    if d > br:
        n = int(math.ceil((d - br) / step - 1e-9))
        for i in range(n):
            a = br + i * step
            b = min(d, a + step)
            loss += (j_on_us(0.5 * (a + b)) + other_dps) * (b - a) / v_c
    e_after = max(0.0, energy - boost_drain * max(0.0, d - br) / v_c)
    dps_them = j_on_us(br) + other_dps
    dps_us = us_on_j(br, e_after)
    if dps_us <= 0.0:
        return 0.0
    left = hp_us - loss
    if left <= 0.0:
        return 0.0
    ttk_them = left / dps_them if dps_them > 0 else INF
    return ttk_them / (hp_j / dps_us)


def _canonical_rush(C, d: float, retreat: float) -> float:
    """Canonical 8.2.1 case: us GL, 0 cubes, full HP, energy RUSH_ENERGY_MIN vs one GZ at full HP."""
    hp = _f(C.path("mech.hp", 1400), 1400.0)
    boost = _f(C.path("mech.boost_speed", 19.5), 19.5)
    drain = _f(C.path("mech.boost_drain", 17), 17.0)
    gun, lt, lz = enemy_wstats(C, "gun"), enemy_wstats(C, "lightning"), enemy_wstats(C, "laser")

    def us_on_j(s, e):
        return (dps("gun", gun, s, 0, 0, rear=False, armor=0.0, energy=e, C=C, enemy=False)
                + dps("lightning", lt, s, 0, 0, rear=False, armor=0.0, energy=e, C=C, enemy=False))

    def j_on_us(s):
        return (dps("gun", gun, s, 0, 0, rear=False, armor=0.0, energy=None, C=C, enemy=True)
                + dps("laser", lz, s, 0, 0, rear=False, armor=0.0, energy=None, C=C, enemy=True))

    return a_rush(C, d, boost - max(0.0, retreat), hp, hp, C.RUSH_ENERGY_MIN, drain, us_on_j, j_on_us)


def calibrate_rush_a_min(C) -> None:
    """Sets C.RUSH_A_MIN = min(A_rush(LIGHTNING_RUSH_R, GZ retreating at walk speed),
    A_rush(LIGHTNING_RUSH_STILL_R, GZ standing)) so GP's 75/110 m rush table passes in the canonical
    1v1 (8.2.1); RUSH_GATE != "calibrated" (D7) raises it to ENGAGE_A_MIN_SAFE."""
    try:
        walk = _f(C.path("mech.walk_speed", 11.7), 11.7)
        a1 = _canonical_rush(C, C.LIGHTNING_RUSH_R, walk)
        a2 = _canonical_rush(C, C.LIGHTNING_RUSH_STILL_R, 0.0)
        v = min(a1, a2)
        if getattr(C, "RUSH_GATE", "calibrated") != "calibrated":
            v = max(v, C.ENGAGE_A_MIN_SAFE)
        C.RUSH_A_MIN = v
        C.base["RUSH_A_MIN"] = v
        C.rush_calibration = {"retreat_75": a1, "standing_110": a2}
    except Exception as e:  # never break hello: fall back to the engage threshold
        C.RUSH_A_MIN = C.ENGAGE_A_MIN
        C.bind_errors["RUSH_A_MIN"] = f"{type(e).__name__}: {e}"


# ================================================================== 8.8 zone slack
def tau_edge(zone, p, t: float, steps: int) -> float:
    """Earliest tau >= t with |p - c(tau)| > r(tau) (bisection; inf if never; t if outside now)."""
    def outside(tau):
        c = zone.center_at(tau)
        return math.hypot(p[0] - c[0], p[1] - c[1]) > zone.radius_at(tau)
    if outside(t):
        return t
    end = max(t, zone.shrink_end)
    if not outside(end):
        return INF
    lo, hi = max(t, zone.shrink_start), end
    for _ in range(max(1, int(steps))):
        mid = 0.5 * (lo + hi)
        if outside(mid):
            hi = mid
        else:
            lo = mid
    return hi


def safe_set_dist(zone, p, C) -> float:
    """Distance from p to the 8.8 safe set S (next circle shrunk by ZONE_EDGE_MARGIN_M, or the
    ZONE_TINY_R disc around next_center)."""
    d = math.hypot(p[0] - zone.next_center[0], p[1] - zone.next_center[1])
    r = zone.next_radius - C.ZONE_EDGE_MARGIN_M
    if r < C.ZONE_TINY_R:
        r = C.ZONE_TINY_R
    return max(0.0, d - r)


def zone_eta_fallback(zone, p, C, walk_speed: float) -> float:
    """8.8 fallback ETA (straight line x ZONE_FALLBACK_DETOUR_K), x ZONE_ETA_SAFETY like the field."""
    return safe_set_dist(zone, p, C) / max(C.IMPL_MIN_SPEED, walk_speed) * C.ZONE_FALLBACK_DETOUR_K * C.ZONE_ETA_SAFETY


# ================================================================== R0.1 terrain call
def terrain_call_from_stats(S: float, Cs: float, R: float, t: float, C) -> str:
    """R0.1 first match; NaN S/C = unknown (degraded map) and never matches."""
    if t >= C.PHASE_T4_SHRINK:
        return "LIGHTNING"
    if R <= C.CALL_R_LIGHTNING:
        return "LIGHTNING"
    if (S == S and S <= C.CALL_S_LIGHTNING) or (Cs == Cs and Cs >= C.CALL_C_LIGHTNING):
        return "LIGHTNING"
    if S == S and Cs == Cs and S >= C.CALL_S_LASER and Cs < C.CALL_C_LASER:
        return "LASER"
    return "MIXED"


def terrain_stats(md, center, radius: float, C) -> Tuple[float, float, int]:
    """(S median sight, C share of cells with a wall within COVER_WALL_R, n samples) over the 5 m
    sight cells inside the circle (mapdata.sight5 / wall5), building footprints (roof_id > 0)
    excluded, subsampled to <= CALL_MAX_SAMPLES. S is NaN when sight5 is unknown (degraded)."""
    nan = float("nan")
    if md is None or radius <= 0:
        return nan, nan, 0
    step = 5.0
    gx0 = float(getattr(md, "gx0", (md.bounds[0] if isinstance(md.bounds, tuple) else -513.0) - 0.5))
    gy0 = float(getattr(md, "gy0", (md.bounds[1] if isinstance(md.bounds, tuple) else -334.0) - 0.5))
    cx, cy = center
    est = math.pi * radius * radius / (step * step)
    stride = max(1, int(math.ceil(math.sqrt(est / CALL_MAX_SAMPLES))))
    sight5, wall5, roof_id = md.sight5, md.wall5, md.roof_id
    sights: List[float] = []
    walls = 0
    n = 0
    j0 = int((cy - radius - gy0) // step)
    j1 = int((cy + radius - gy0) // step)
    r2 = radius * radius
    wall_r = C.COVER_WALL_R
    for j in range(j0, j1 + 1, stride):
        y = gy0 + (j + 0.5) * step
        dy = y - cy
        if dy * dy > r2:
            continue
        half = math.sqrt(r2 - dy * dy)
        i0 = int((cx - half - gx0) // step)
        i1 = int((cx + half - gx0) // step)
        for i in range(i0, i1 + 1, stride):
            x = gx0 + (i + 0.5) * step
            if not md.in_bounds(x, y) or roof_id(x, y):
                continue
            n += 1
            sv = sight5(x, y)
            if sv == sv:
                sights.append(sv)
            wv = wall5(x, y)
            if wv == wv and wv <= wall_r:
                walls += 1
    if n == 0:
        return nan, nan, 0
    S = nan
    if sights:
        sights.sort()
        m = len(sights)
        S = sights[m // 2] if m % 2 else 0.5 * (sights[m // 2 - 1] + sights[m // 2])
    return S, walls / n, n


# ================================================================== Assessor
class Assessor:
    def __init__(self, C, mapdata) -> None:
        self.C = C
        self.map = mapdata
        self.last = Assessment()
        self.call_raw = "MIXED"
        self.call_S = float("nan")
        self.call_C = float("nan")
        self.call_R = 0.0
        self.call_n = 0
        self._call_key = None
        self._target_id: Optional[int] = None
        self._target_since = -INF
        self._cornered_since: Optional[float] = None
        self._nav = None
        self.aim1: List[int] = []
        self.shell_hits_us: List[int] = []
        self.laser_next_t: Dict[int, float] = {}
        self.hostile_dps: Dict[int, float] = {}
        self._dps_from: Dict[int, float] = {}
        self._hp_eff: Dict[int, float] = {}
        self.update_ms = 0.0
        self.los_calls = 0
        self.exceptions = 0
        self.last_error = ""
        p = C.path
        self.front_half = math.radians(_f(p("mech.front_arc_deg", 180), 180.0)) / 2.0
        self.walk = _f(p("mech.walk_speed", 11.7), 11.7)
        self.boost = _f(p("mech.boost_speed", 19.5), 19.5)
        self.mech_h = _f(p("mech.height", 9), 9.0)
        self.npc_h = _f(p("npc.height", 3.5), 3.5)
        self.size_k = _f(p("cubes.size_per_cube", 0.05), 0.05)
        self.drop_extra = _f(p("cubes.death_drop_extra", 2), 2.0)
        self.laser_cfg = enemy_wstats(C, "laser")

    # ------------------------------------------------------------------ helpers
    def _wall_dist(self, x: float, y: float) -> Optional[float]:
        try:
            return self.map.wall5(x, y) if self.map is not None else None
        except Exception:
            return None

    def _enemy_kinds(self, world, tr, d: float, d3: float, los: bool) -> Tuple[str, ...]:
        """Sighted weapons, else confident inferences, padded with the worst case for this distance
        among kinds with P >= PRIOR_MIN_P (R2.2), at most 2 weapons."""
        if tr.kind == "npc":
            return ("npc",)
        x = world.tx.get(tr.id) if hasattr(world, "tx") else None
        if (x is not None and x.sighted) or tr.weapons:
            return tuple(tr.weapons)
        C = self.C
        known = [k for k in WEAPONS if tr.known_weapons.get(k, 0.0) >= C.CONF_SOUND]
        if len(known) >= 2:
            return tuple(known[:2])
        probs = world.weapon_probs(tr.id) if hasattr(world, "weapon_probs") else {"gun": 1.0}
        cands = [k for k in WEAPONS if k not in known and probs.get(k, 0.0) >= C.PRIOR_MIN_P]
        me = world.me
        best, bv = None, -1.0
        for k in cands:
            v = dps(k, enemy_wstats(C, k), d, tr.cubes, me.cubes, rear=False, armor=me.armor,
                    energy=tr.energy_est, C=C, enemy=True, d3=d3, los=los, range_mult=tr.lightning_range_mult)
            if v > bv:
                best, bv = k, v
        out = list(known)
        if best is not None:
            out.append(best)
        if len(out) < 2 and "gun" not in out and probs.get("gun", 0.0) >= C.PRIOR_MIN_P:
            out.append("gun")
        return tuple(out[:2])

    def _our_dps(self, world, tr, d: float, d3: float, los: bool, rear_on_j: bool, cover_j: bool,
                 energy: Optional[float] = None, t: Optional[float] = None) -> float:
        me = world.me
        C = self.C
        t = world.tick.t if t is None else t
        e = me.energy if energy is None else energy
        s = 0.0
        for i, w in enumerate(me.weapons):
            if w is None:
                continue
            if i < len(me.equip_until) and t < me.equip_until[i]:
                continue
            s += dps(w.kind, w.stats, d, me.cubes, tr.cubes, rear=rear_on_j, armor=0.0, energy=e, C=C,
                     enemy=False, d3=d3, los=los, reloading=(w.kind == "gun" and w.reload > 0.0),
                     shorted=(w.kind == "laser" and t < me.short_until), cover=cover_j, ammo=w.ammo)
        return s

    def _their_dps(self, world, tr, kinds, d: float, d3: float, los: bool, rear_on_us: bool,
                   cover_us: bool) -> float:
        me = world.me
        C = self.C
        t = world.tick.t
        s = 0.0
        for k in kinds:
            s += dps(k, enemy_wstats(C, k), d, tr.cubes, me.cubes, rear=rear_on_us, armor=me.armor,
                     energy=tr.energy_est, C=C, enemy=True, d3=d3, los=los,
                     reloading=(k == "gun" and tr.reloading_until > t), cover=cover_us,
                     range_mult=tr.lightning_range_mult)
        return s

    # ------------------------------------------------------------------ R0.1
    def terrain_call(self, world) -> Tuple[str, str]:
        """R0.1 -> (call, desired loadout key); grid scan only when the next circle changed."""
        zone = world.zone
        key = (round(zone.next_center[0], 2), round(zone.next_center[1], 2), round(zone.next_radius, 2))
        if key != self._call_key:
            self._call_key = key
            self.call_R = zone.next_radius
            if zone.next_radius <= self.C.CALL_R_LIGHTNING:
                self.call_S, self.call_C, self.call_n = float("nan"), float("nan"), 0
            else:
                self.call_S, self.call_C, self.call_n = terrain_stats(self.map, zone.next_center, zone.next_radius, self.C)
            self.call_raw = terrain_call_from_stats(self.call_S, self.call_C, self.call_R, -INF, self.C)
        call = "LIGHTNING" if world.tick.t >= self.C.PHASE_T4_SHRINK else self.call_raw
        return call, DESIRED[call]

    # ------------------------------------------------------------------ 6.1 step 5
    def update(self, world, nav, clock) -> Assessment:
        """Never raises: on an internal error the last Assessment is returned (R15.1 stage fallback)
        and the error is counted in self.exceptions (collect with take_exceptions())."""
        t_start = time.perf_counter()
        try:
            self.last = self._update(world, nav, clock)
        except Exception as e:
            self.exceptions += 1
            self.last_error = f"{type(e).__name__}: {e}"
        self.update_ms = (time.perf_counter() - t_start) * 1000.0
        return self.last

    def take_exceptions(self) -> int:
        n, self.exceptions = self.exceptions, 0
        return n

    def _update(self, world, nav, clock) -> Assessment:
        C = self.C
        self._nav = nav
        self.los_calls = 0
        me = world.me
        t = world.tick.t
        sl = clock.slice(C.LOS_MS_CAP) if clock is not None else None
        A = Assessment(t=t, playbook=me.playbook)
        call, desired = self.terrain_call(world)
        A.terrain_call, A.desired_loadout = call, desired
        A.loadout_rank = loadout_rank(world.held_kinds(), call)
        mx, my, mz = me.pos
        h_me = _f(me.stats.get("height"), self.mech_h * me.s)
        cz_me = mz + 0.5 * h_me
        hp_us = hp_eff_self(me)
        wd_me = self._wall_dist(mx, my)
        cover_us = wd_me is not None and wd_me <= C.COVER_DASH_R
        laser_tol_rad = C.LASER_AIM_TOL_RAD
        aim_tol_deg = math.radians(C.FIGHTING_AIM_TOL_DEG)
        lt_ours = None
        for w in me.weapons:
            if w is not None and w.kind == "lightning":
                lt_ours = w
        turn_rate = _f(me.stats.get("turn_rate"), 4.0) or 4.0
        fire_half = lightning_cone_half(C, worst=False)
        hostile_dps: Dict[int, float] = {}
        self._dps_from = {}
        self._hp_eff = {}
        chargers = []
        aim1 = []
        laser_next: Dict[int, float] = {}
        laser_period = _f(self.laser_cfg.get("cooldown"), 1.65) + _f(self.laser_cfg.get("charge"), 0.75)
        for tid, tr in world.tracks.items():
            x = world.tx.get(tid) if hasattr(world, "tx") else None
            dx, dy = tr.pos[0] - mx, tr.pos[1] - my
            d = math.hypot(dx, dy)
            h_t = self.npc_h if tr.kind == "npc" else self.mech_h * (1.0 + self.size_k * tr.cubes)
            d3 = math.hypot(d, tr.pos[2] + 0.5 * h_t - cz_me)
            b = math.atan2(dy, dx)
            b_back = wrap(b + math.pi)
            los = bool(tr.in_obs and x is not None and x.los_geo)
            rear_on_us = abs(wrap(b - me.aim[0])) > self.front_half
            rear_on_j = (not tr.est) and abs(wrap(b_back - tr.aim[0])) > self.front_half
            kinds = self._enemy_kinds(world, tr, d, d3, los)
            raw_them = self._their_dps(world, tr, kinds, d, d3, los, rear_on_us, cover_us)
            if tr.est:
                aiming = True
            else:
                tol = max(aim_tol_deg, math.atan2(C.LASER_AIM_TOL_M, max(d, 1.0)) + laser_tol_rad)
                aiming = abs(wrap(tr.aim[0] - b_back)) <= tol
            them = raw_them * (1.0 if aiming else C.NOT_AIMING_DPS_FRAC)
            reach = 0.0
            for k in kinds:
                reach = max(reach, weapon_reach(k, enemy_wstats(C, k), C, tr.lightning_range_mult))
            in_reach = los and (d3 if kinds == ("lightning",) else d) <= reach and raw_them > 0.0
            wd_j = self._wall_dist(tr.pos[0], tr.pos[1])
            ours = self._our_dps(world, tr, d, d3, los, rear_on_j, wd_j is not None and wd_j <= C.COVER_DASH_R)
            hpj = hp_eff_enemy(tr, C)
            ttk_us = hpj / ours if ours > 0 else INF
            ttk_them = hp_us / them if them > 0 else INF
            a_j = _ratio(ttk_them, ttk_us)
            dvx = tr.vel[0] - me.vel[0]
            dvy = tr.vel[1] - me.vel[1]
            closing = -(dx * dvx + dy * dvy) / d if d > 1e-6 else 0.0
            # O3: visible charger, aim on us, LOS from its muzzle to our center
            charging_at_us = False
            if tr.kind == "mech" and tr.visible and tr.charging:
                tol3 = math.atan2(C.LASER_AIM_TOL_M, max(d, 1.0)) + laser_tol_rad
                if abs(wrap(tr.aim[0] - b_back)) <= tol3:
                    if sl is not None and sl.expired():
                        charging_at_us = los
                    else:
                        self.los_calls += 1
                        mzz = tr.pos[2] + 7.0 * (1.0 + self.size_k * tr.cubes)
                        try:
                            charging_at_us = bool(self.map.los((tr.pos[0], tr.pos[1], mzz), (mx, my, cz_me)))
                        except Exception:
                            charging_at_us = los
            damaged_recent = t - tr.last_damaged_us_t <= C.SELF_DEFENSE_S
            in_h = (in_reach and them > 0.0) or damaged_recent
            if in_h:
                hostile_dps[tid] = them if them > 0 else raw_them
            prio = 0.0
            if tr.kind == "mech" and ttk_us < INF:
                prio = ((1.0 + C.THREAT_W * them / C.DPS_NORM)
                        * (1.0 + C.VALUE_W * (tr.cubes + self.drop_extra) / C.PRIO_CUBE_NORM) / ttk_us)
                if charging_at_us:
                    prio *= C.PRIO_CHARGER_MULT
            A.per_track[tid] = TrackAssess(id=tid, d=d, d3=d3, bearing=b, los=los, dps_to_us=them,
                                           dps_from_us=ours, ttk_us_on_j=ttk_us, ttk_them_on_us=ttk_them,
                                           A=a_j, prio=prio, aiming_at_us=aiming, in_its_reach=in_reach,
                                           charging_at_us=charging_at_us, closing_speed=closing)
            self._dps_from[tid] = ours
            self._hp_eff[tid] = hpj
            if charging_at_us:
                left = _f(self.laser_cfg.get("charge"), 0.75) - (t - (tr.charging_since if tr.charging_since is not None else t))
                chargers.append((left, d, tid))
                if lt_ours is not None:
                    lt_range = _f(lt_ours.stats.get("range"), C.LIGHTNING_RANGE)
                    err = abs(wrap(b - me.aim[0]))
                    if d3 <= lt_range and max(0.0, err - fire_half) / turn_rate <= max(0.0, left):
                        aim1.append((left, d, tid))
            if tr.kind == "mech" and ("laser" in tr.weapons or tr.known_weapons.get("laser", 0.0) > 0.0):
                laser_next[tid] = tr.last_laser_shot_t + laser_period if tr.last_laser_shot_t > -1e8 else t
        chargers.sort()
        aim1.sort()
        self.aim1 = [c[2] for c in aim1]
        self.laser_next_t = laser_next
        self.hostile_dps = hostile_dps
        A.hostiles = sorted(hostile_dps)
        # main threat: highest DPS on us among H mechs, nearest on ties
        mt, mv = None, None
        for tid in A.hostiles:
            tr = world.tracks[tid]
            if tr.kind != "mech":
                continue
            v = (hostile_dps[tid], -A.per_track[tid].d)
            if mv is None or v > mv:
                mt, mv = tid, v
        A.main_threat_id = mt
        # target with H3 hysteresis
        A.target_id = self._pick_target(world, A, t)
        sum_h = sum(hostile_dps.values())
        A.ttk_them_on_us = hp_us / sum_h if sum_h > 0 else INF
        if A.target_id is not None:
            ta = A.per_track[A.target_id]
            A.ttk_us_on_target = ta.ttk_us_on_j
            A.A = _ratio(A.ttk_them_on_us, ta.ttk_us_on_j)
            if me.playbook == "PB_L" and ta.d > C.LIGHTNING_RANGE:
                A.A_rush = self.a_rush_for(world, A, A.target_id)
        # threats
        th = Threats()
        th.laser_chargers = [c[2] for c in chargers]
        th.shells = list(world.shells)
        th.bullets_inbound = list(world.bullets)
        th.unseen_hits = [r for r in world.damage_log if r.unseen and t - r.t <= C.UNSEEN_REACT_S]
        th.mortar_active_until = world.mortar_active_until
        A.threats = th
        hits = []
        for s in world.shells:
            tau = max(0.0, min(s.t_impact - t, 2.0))
            px, py = mx + me.vel[0] * tau, my + me.vel[1] * tau
            if math.hypot(s.impact[0] - px, s.impact[1] - py) < C.MORTAR_DODGE_R:
                hits.append(s.id)
        self.shell_hits_us = hits
        # zone ETA and slack (8.8)
        eta, settled = None, False
        if nav is not None and hasattr(nav, "zone_eta"):
            try:
                r = nav.zone_eta(me.pos)
                if r is not None:
                    eta, settled = r[0], bool(r[1])
            except Exception as e:
                self.exceptions += 1
                self.last_error = f"nav.zone_eta: {type(e).__name__}: {e}"
                eta = None
        if eta is None or not (eta == eta) or eta == INF:
            eta, settled = zone_eta_fallback(world.zone, me.pos, C, _f(me.stats.get("walk_speed"), self.walk)), False
        A.zone_eta, A.zone_eta_settled = float(eta), settled
        A.tau_edge = tau_edge(world.zone, me.pos, t, C.IMPL_BISECT_STEPS)
        A.slack = A.tau_edge - t - A.zone_eta if A.tau_edge < INF else INF
        return A

    def _pick_target(self, world, A: Assessment, t: float) -> Optional[int]:
        C = self.C
        best, bv = None, None
        for tid, ta in A.per_track.items():
            tr = world.tracks[tid]
            if tr.kind != "mech" or not tr.visible:
                continue
            v = (ta.prio, -ta.d, -tid)
            if bv is None or v > bv:
                best, bv = tid, v
        cur = self._target_id
        tr = world.tracks.get(cur) if cur is not None else None
        cur_ok = tr is not None and tr.kind == "mech" and (tr.visible or t - tr.last_visible_t <= C.TARGET_LOST_S)
        if not cur_ok:
            self._target_id, self._target_since = best, t
            return best
        if best is not None and best != cur:
            cur_prio = A.per_track[cur].prio if cur in A.per_track else 0.0
            if best in self.aim1 or (bv[0] > cur_prio * C.TARGET_SWITCH_MARGIN and t - self._target_since >= C.TARGET_MIN_HOLD_S):
                self._target_id, self._target_since = best, t
                return best
        return cur

    # ------------------------------------------------------------------ A queries
    def a_rush_for(self, world, A: Assessment, tid: int, hostiles: Optional[List[int]] = None) -> float:
        C = self.C
        me = world.me
        tr = world.tracks.get(tid)
        ta = A.per_track.get(tid)
        if tr is None or ta is None:
            return 0.0
        d = ta.d
        retreat = (tr.vel[0] * (tr.pos[0] - me.pos[0]) + tr.vel[1] * (tr.pos[1] - me.pos[1])) / d if d > 1e-6 else 0.0
        boost = _f(me.stats.get("boost_speed"), self.boost)
        hs = A.hostiles if hostiles is None else hostiles
        other = sum(self.hostile_dps.get(h, 0.0) for h in hs if h != tid)
        kinds = self._enemy_kinds(world, tr, d, ta.d3, True)
        dz = ta.d3 * ta.d3 - d * d

        def j_on_us(s):
            s3 = math.sqrt(s * s + max(0.0, dz))
            return self._their_dps(world, tr, kinds, s, s3, True, False, False)

        def us_on_j(s, e):
            s3 = math.sqrt(s * s + max(0.0, dz))
            return self._our_dps(world, tr, s, s3, True, False, False, energy=e)

        drain = _f(me.stats.get("boost_drain"), _f(C.path("mech.boost_drain", 17), 17.0))
        return a_rush(C, d, boost - max(0.0, retreat), hp_eff_self(me), hp_eff_enemy(tr, C), me.energy, drain,
                      us_on_j, j_on_us, other)

    def _rush_exempt(self, world, tr, ta, macro: str, target_id, sub: str) -> bool:
        """Integration decision (UQ-2 must not cancel the GP lightning rush, KEEP): LOSE (i) / R6.3 skips a
        laser track that is our X2 target in sub-state RUSH, or that currently satisfies RL1's rush trigger
        (not backing away; <= LIGHTNING_RANGE with LIGHTNING_MIN_ENERGY, <= LIGHTNING_RUSH_R with the rush
        energy, or <= LIGHTNING_RUSH_STILL_R while it stands still / reloads)."""
        C, me, t = self.C, world.me, world.tick.t
        if macro == "X2" and tr.id == target_id and sub == "RUSH":
            return True
        if ta.closing_speed < -C.RETREAT_SPEED_MIN:
            return False
        d = ta.d
        if d <= C.LIGHTNING_RANGE:
            return me.energy >= C.LIGHTNING_MIN_ENERGY
        e_min = C.RUSH_ENERGY_MIN_SC if "lightning_efficiency" in me.upgrades else C.RUSH_ENERGY_MIN
        if d <= C.LIGHTNING_RUSH_R and me.energy >= e_min:
            return True
        still = tr.still_since is not None and t - tr.still_since >= C.STILL_MIN_S
        return d <= C.LIGHTNING_RUSH_STILL_R and (still or tr.reloading_until > t)

    def a_vs(self, world, track_id: int, hostiles: Optional[List[int]] = None) -> float:
        """A vs track_id with H = `hostiles` (None -> that track alone), from the last update."""
        me = world.me
        A = self.last
        ta = A.per_track.get(track_id)
        if ta is None:
            return 0.0
        hs = [track_id] if hostiles is None else hostiles
        tot = 0.0
        for h in hs:
            if h == track_id:
                tot += ta.dps_to_us
            else:
                tot += self.hostile_dps.get(h, A.per_track[h].dps_to_us if h in A.per_track else 0.0)
        ttk_them = hp_eff_self(me) / tot if tot > 0 else INF
        return _ratio(ttk_them, ta.ttk_us_on_j)

    # ------------------------------------------------------------------ 8.6 drop go/no-go
    def drop_go(self, world, drop_id: int, A: Optional[Assessment] = None, nav=None,
                exit_memory=None) -> Tuple[bool, Dict[str, object]]:
        C = self.C
        me = world.me
        t = world.tick.t
        dr = world.drops.get(drop_id)
        info: Dict[str, object] = {"fail": []}
        if dr is None:
            info["fail"].append("gone")
            return False, info
        nav = nav if nav is not None else self._nav
        walk = _f(me.stats.get("walk_speed"), self.walk)
        boost = _f(me.stats.get("boost_speed"), self.boost)
        dist = max(0.0, math.hypot(dr.pos[0] - me.pos[0], dr.pos[1] - me.pos[1]) - dr.radius)
        eta = None
        if nav is not None and hasattr(nav, "travel_time"):
            try:
                eta = nav.travel_time(dr.pos[0], dr.pos[1])
            except Exception:
                eta = None
        if eta is None:
            eta = dist / max(C.IMPL_MIN_SPEED, walk) * C.ZONE_FALLBACK_DETOUR_K
        if dr.lands_in < eta and me.energy >= C.CONTEST_BOOST_MIN_ENERGY:
            eta = eta * walk / max(walk, boost)
        t_ready = max(eta, dr.lands_in)
        info.update(eta=eta, t_ready=t_ready)
        rivals = []
        near_contacts = 0
        for tr in world.tracks.values():
            if tr.kind != "mech" or t - tr.last_seen_t >= C.DROP_RIVAL_INFO_S:
                continue
            dd = max(0.0, math.hypot(tr.pos[0] - dr.pos[0], tr.pos[1] - dr.pos[1]) - dr.radius)
            if dd <= C.DROP_RIVAL_R:
                near_contacts += 1
            if dd / max(C.IMPL_MIN_SPEED, self.walk) <= t_ready + C.DROP_RIVAL_ETA_MARGIN_S or \
                    (tr.in_obs and dd <= C.DROP_RIVAL_R):
                rivals.append(tr.id)
        for g in world.ghosts.values():
            if t - g.last_t >= C.DROP_RIVAL_INFO_S:
                continue
            dd = max(0.0, math.hypot(g.pos[0] - dr.pos[0], g.pos[1] - dr.pos[1]) - dr.radius)
            if dd <= C.DROP_RIVAL_R:
                near_contacts += 1
            if dd / max(C.IMPL_MIN_SPEED, self.walk) <= t_ready + C.DROP_RIVAL_ETA_MARGIN_S:
                rivals.append(g.id)
        info["rivals"] = rivals
        te = tau_edge(world.zone, dr.pos, t, C.IMPL_BISECT_STEPS)
        slack_after = te - t - t_ready - zone_eta_fallback(world.zone, dr.pos, C, walk) if te < INF else INF
        info["slack_after"] = slack_after
        fails = info["fail"]
        if not (eta <= C.DROP_MAX_ETA_S and slack_after >= C.ZONE_SLACK_SOFT_S):
            fails.append("1:eta_slack")
        if me.hp < C.DROP_MIN_HP_FRAC * me.max_hp:
            fails.append("2:hp")
        allowed = 0 if C.DROP_CONTEST == "gp" else 1
        if len(rivals) > allowed:
            fails.append("3:rivals")
        wd = self._wall_dist(dr.pos[0], dr.pos[1])
        open_landing = wd is None or wd > C.DROP_OPEN_COVER_R
        info["open"] = open_landing
        if open_landing and (rivals or near_contacts):
            fails.append("4:open")
        if t >= C.ENDGAME_T:
            fails.append("5:endgame")
        if exit_memory is not None and not (exit_memory.clear("DROP", drop_id, t) and exit_memory.clear("M7", drop_id, t)):
            fails.append("6:memory")
        return not fails, info

    # ------------------------------------------------------------------ 8.2.3 - 8.2.6
    def evaluate_fight(self, world, A: Assessment, macro: str, target_id: Optional[int], ctx: dict) -> FightEval:
        """WIN / LOSE clauses / CORNERED for the macro context. LOSE clauses carry their reach
        precondition; exit-memory checks, D14/D15 and the committed-capture suppression of LOSE are
        applied by modes (it reads ctx-independent clauses); WIN here applies R10.1, loot-only,
        R5.3 and the committed-capture veto."""
        C = self.C
        ctx = ctx or {}
        me = world.me
        t = world.tick.t
        fe = FightEval()
        tid = target_id if target_id is not None else A.target_id
        tr = world.tracks.get(tid) if tid is not None else None
        ta = A.per_track.get(tid) if tid is not None else None
        fe.target_id = tid if tr is not None else None
        held = world.held_kinds()
        hostiles = A.hostiles
        in_x2 = macro == "X2"
        # ---- reach precondition for (a)-(d), (f)-(i)
        reach = any((A.per_track[h].in_its_reach if h in A.per_track else False) for h in hostiles) or \
            any(t - r.t <= C.SELF_DEFENSE_S for r in world.damage_log if r.by is not None)
        ttk_j = ta.ttk_us_on_j if ta is not None else 0.0
        lose = set()
        lt: Dict[str, int] = {}
        if reach:
            if in_x2 and tr is not None and A.A < C.HOLD_A_MIN:
                lose.add("a")
            if in_x2 and tr is not None and me.hp < C.DISENGAGE_HP_FRAC * me.max_hp and tr.hp is not None and \
                    tr.max_hp and tr.hp > C.DISENGAGE_HP_FRAC * tr.max_hp:
                lose.add("b")
            if in_x2:
                since = ctx.get("x2_since")
                since = -INF if since is None else since
                for o in world.tracks.values():
                    if o.kind != "mech" or o.id == tid:
                        continue
                    x = world.tx.get(o.id)
                    vis_new = o.visible and x is not None and x.vis_since is not None and x.vis_since >= since
                    if vis_new or (o.last_damaged_us_t >= since and t - o.last_damaged_us_t <= C.SELF_DEFENSE_S):
                        lose.add("c")
                        lt.setdefault("c", o.id)
                        break
            if A.slack < C.ZONE_SLACK_HARD_S + (ttk_j if ttk_j < INF else 0.0):
                lose.add("d")
            if me.hp < C.LOW_HP_FRAC * me.max_hp and any(A.per_track[h].los for h in hostiles if h in A.per_track):
                lose.add("f")
            for h in hostiles:
                if h == tid or h not in A.per_track:
                    continue
                o = world.tracks.get(h)
                if o is None or o.kind != "mech":
                    continue
                pa = A.per_track[h]
                if (pa.charging_at_us or t - o.last_damaged_us_t <= C.SELF_DEFENSE_S) and self.a_vs(world, h) < C.HOLD_A_MIN:
                    lose.add("g")
                    lt.setdefault("g", h)
                    break
            nb = ctx.get("neighbor")
            if nb is not None and nb in A.per_track:
                o = world.tracks.get(nb)
                if o is not None and len(o.weapons) >= 2 and len(held) < 2 and self.a_vs(world, nb) < C.HOLD_A_MIN:
                    lose.add("h")
                    lt["h"] = nb
            if C.LASER_VIEW_RESPONSE != "off" and "lightning" in held and "laser" not in held:
                for h, pa in A.per_track.items():
                    o = world.tracks.get(h)
                    x = world.tx.get(h)
                    if o is None or o.kind != "mech" or "laser" not in o.weapons or x is None or x.los_since is None:
                        continue
                    if self._rush_exempt(world, o, pa, macro, tid, ctx.get("sub", "")):
                        continue
                    if t - x.los_since >= C.LASER_VIEW_DWELL_S and pa.d <= _f(self.laser_cfg.get("range"), 280.0) and \
                            self.a_vs(world, h) < C.HOLD_A_MIN:
                        lose.add("i")
                        lt["i"] = h
                        break
        # (e) R6.1 (no reach precondition)
        if me.playbook in ("PB_Z", "PB_G"):
            best_e = None
            for h, pa in A.per_track.items():
                o = world.tracks.get(h)
                if o is None or o.kind != "mech" or t - o.last_seen_t > C.LIGHTNING_TRACK_MAX_AGE_S:
                    continue
                if not ("lightning" in o.weapons or o.known_weapons.get("lightning", 0.0) >= C.CONF_SOUND):
                    continue
                m = o.lightning_range_mult or 1.0
                if (pa.d <= C.LIGHTNING_THREAT_R * m and pa.closing_speed > C.RUSH_CLOSING_MIN) or pa.d <= C.LIGHTNING_KEEPOUT_R * m:
                    if best_e is None or pa.d < A.per_track[best_e].d:
                        best_e = h
            if best_e is not None:
                lose.add("e")
                lt["e"] = best_e
        fe.lose_clauses = frozenset(lose)
        fe.lose_tracks = lt
        # ---- CORNERED (8.2.6): in X3, or held while X2 CORNERED is active
        cornered_active = bool(ctx.get("cornered_active"))
        if macro == "X3" or cornered_active:
            th_id = lt.get("e") or lt.get("g") or lt.get("i") or A.main_threat_id or tid
            pa = A.per_track.get(th_id) if th_id is not None else None
            if pa is not None:
                o = world.tracks[th_id]
                kinds = self._enemy_kinds(world, o, pa.d, pa.d3, pa.los)
                rch = max([weapon_reach(k, enemy_wstats(C, k), C, o.lightning_range_mult) for k in kinds] or [0.0])
                fe.t_close = max(0.0, pa.d - rch) / max(C.IMPL_MIN_SPEED, pa.closing_speed)
                wd = self._wall_dist(me.pos[0], me.pos[1])
                cover_eta = ctx.get("cover_eta")
                if cover_eta is None:
                    cover_eta = (wd / max(C.IMPL_MIN_SPEED, self.walk)) if (wd is not None and wd <= C.COVER_SEARCH_R) else INF
                no_break = pa.in_its_reach and (wd is None or wd > C.COVER_DASH_R)
                enter = cover_eta >= fe.t_close or no_break
                if cornered_active:
                    if self._cornered_since is None:
                        self._cornered_since = t
                    held_long = t - self._cornered_since >= C.CORNER_MIN_HOLD_S
                    fe.cornered = not (held_long and cover_eta < fe.t_close - C.CORNER_COVER_MARGIN_S)
                else:
                    fe.cornered = enter
                    self._cornered_since = t if enter else None
        else:
            self._cornered_since = None
        if fe.cornered:
            fe.win, fe.win_rule, fe.win_sub = True, "CORNERED", "CORNERED"
            return fe
        # ---- WIN(j)
        if tr is None or ta is None or tr.kind != "mech" or t >= C.ENDGAME_T:
            return fe
        visible = tr.visible or (in_x2 and t - tr.last_visible_t <= C.TARGET_LOST_S)
        if not visible:
            return fe
        a_min = C.HOLD_A_MIN if in_x2 else C.ENGAGE_A_MIN
        pb = me.playbook
        d = ta.d
        ok, rule, sub, a_gate = False, "", "", True
        tw = tr.weapons
        if pb == "PB_L":
            if C.PHASE_T4_SHRINK <= t < C.ENDGAME_T and d <= C.LIGHTNING_RANGE and me.energy >= C.LIGHTNING_MIN_ENERGY:
                ok, rule, sub, a_gate = True, "R0.5", "BRAWL", False
            elif tr.on_roof and "laser" in tw:
                h = max(0.0, tr.pos[2] - me.pos[2])
                need = C.CLIMB_COST_PER_M * h + C.CLIMB_COST_PAD + C.ROOF_ASSAULT_RESERVE
                if "lightning_efficiency" in me.upgrades:
                    need -= C.ROOF_ASSAULT_SC_DISCOUNT
                ok, rule, sub = me.energy >= need or in_x2, "R10.2", "ROOF_ASSAULT"
                a_gate = (A.A_rush if A.A_rush is not None else A.A) >= (C.RUSH_A_MIN or a_min)
                a_gate = a_gate or in_x2
            elif "laser" in held and d > C.LIGHTNING_RANGE:
                # ZL plays PB_Z rules beyond LIGHTNING_RANGE (GP "With laser + lightning")
                lr = _laser_range(me)
                ok = (C.LASER_HOLD_MIN <= d <= lr) or (in_x2 and d <= lr)
                rule, sub = "RZ1", "DUEL"
            else:
                backing = ta.closing_speed < -C.RETREAT_SPEED_MIN
                if backing or d > C.LIGHTNING_RUSH_STILL_R:
                    ok = False                                              # RL2
                else:
                    e_min = C.RUSH_ENERGY_MIN_SC if "lightning_efficiency" in me.upgrades else C.RUSH_ENERGY_MIN
                    still = tr.still_since is not None and t - tr.still_since >= C.STILL_MIN_S
                    reloading = tr.reloading_until > t
                    if d <= C.LIGHTNING_RANGE:
                        ok = me.energy >= C.LIGHTNING_MIN_ENERGY or in_x2
                    else:
                        ok = (d <= C.LIGHTNING_RUSH_R and me.energy >= e_min) or \
                             (d <= C.LIGHTNING_RUSH_STILL_R and (still or reloading)) or in_x2
                    if ok and "mortar" in tw:                               # RL3
                        boost = _f(me.stats.get("boost_speed"), self.boost)
                        ok = max(0.0, d - C.MORTAR_CLOSE_R) / max(C.IMPL_MIN_SPEED, boost) <= C.RUSH_MORTAR_MAX_S
                    if ok and ("lightning" in tw):                          # RL4
                        ok = hp_eff_self(me) > (tr.hp if tr.hp is not None else hp_eff_enemy(tr, C)) and \
                            me.energy > tr.energy_est + C.RL4_ENERGY_MARGIN
                    rule, sub = "RL1", ("BRAWL" if d <= C.LIGHTNING_RANGE else "RUSH")
                    if d > C.LIGHTNING_RANGE:
                        ar = A.A_rush if (A.A_rush is not None and tid == A.target_id) else self.a_rush_for(world, A, tid)
                        a_gate = ar >= (C.RUSH_A_MIN if C.RUSH_A_MIN is not None else a_min)
        elif pb == "PB_Z":
            lr = _laser_range(me)
            ok = (C.LASER_HOLD_MIN <= d <= lr) or (in_x2 and d <= lr)
            rule, sub = "RZ1", "DUEL"
            # R10.1 veto: enemy on a roof with laser/gun, we are inside its reach
            if ok and tr.on_roof and ("laser" in tw or "gun" in tw) and ta.in_its_reach:
                hurt = world.damaged_us_recently(tid, C.SELF_DEFENSE_S)
                wd = self._wall_dist(me.pos[0], me.pos[1])
                no_cover = wd is None or wd / max(C.IMPL_MIN_SPEED, self.walk) > C.COVER_REACH_S
                gun_only = "laser" not in tw
                if not ((hurt and no_cover) or (gun_only and d > C.LASER_HOLD_MIN)):
                    ok, rule = False, "R10.1"
        else:
            rng = 0.0
            for w in me.weapons:
                if w is not None:
                    rng = max(rng, weapon_reach(w.kind, w.stats, C))
            ok = d <= rng
            rule, sub = "RG1", "DUEL_GUN"
            if not in_x2:
                a_min = a_min + C.PB_G_A_EXTRA          # RG1 entry; X2 persists at HOLD_A_MIN (H2)
        if ok and a_gate and rule not in ("RL1", "R10.2"):
            a_gate = A.A >= a_min if tid == A.target_id else self.a_vs(world, tid, A.hostiles + ([] if tid in A.hostiles else [tid])) >= a_min
        elif ok and rule == "RL1" and d <= C.LIGHTNING_RANGE:
            a_gate = A.A >= a_min if tid == A.target_id else self.a_vs(world, tid) >= a_min
        if not (ok and a_gate):
            fe.win_rule = rule if rule == "R10.1" else ""
            return fe
        # (3) zone slack
        if A.slack < C.ZONE_SLACK_HARD_S + ttk_j:
            return fe
        # (4) vetoes
        if ctx.get("committed_capture"):
            return fe
        hurt = world.damaged_us_recently(tid, C.SELF_DEFENSE_S)
        if (t < C.LOOT_END_T or ctx.get("neighbor") == tid) and not (hurt or A.A >= C.EARLY_FIGHT_A_MIN):
            return fe
        if macro not in ("X2", "X3"):
            seen = [o for o in world.tracks.values() if o.kind == "mech" and t - o.last_seen_t <= C.HIDE_CLEAR_S]
            if len(seen) >= 2 and not any(A.per_track[o.id].aiming_at_us for o in seen if o.id in A.per_track):
                third = tr.fighting_with is not None and tr.fighting_with != me.id and tr.hp is not None and \
                    tr.max_hp and tr.hp < C.THIRD_PARTY_HP_FRAC * tr.max_hp
                if not third:
                    return fe
                rule = "R5.3"
        # (5) not LOSE
        if lose:
            return fe
        fe.win, fe.win_rule, fe.win_sub = True, rule, sub
        return fe


def _laser_range(me) -> float:
    for w in me.weapons:
        if w is not None and w.kind == "laser":
            return _f(w.stats.get("range"), 280.0)
    return 0.0


def _ratio(ttk_them: float, ttk_us: float) -> float:
    """A = ttk_them_on_us / ttk_us_on_j; 0 when we cannot hurt j, inf when nothing hurts us."""
    if ttk_us == INF:
        return 0.0
    if ttk_them == INF:
        return INF
    return ttk_them / ttk_us if ttk_us > 0 else INF
