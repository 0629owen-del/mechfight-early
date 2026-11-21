"""Aiming and fire gates (plan 8.7, 6.6 F1-F4, AIM8/AIM10 look timing). CONTRACTS 6.5.

Pure functions plus `Scanner`. Frames and units as in contracts.py: x east, y north, z up, yaw from +x
counter-clockwise, pitch up-positive, meters, seconds. Nothing here draws or prints.

Conventions chosen where the contract is silent:
  * `gun_lead` returns (yaw, pitch, tof) with tof = bullet flight time only (AIM_LATENCY_S is applied to
    the target's motion but not added to tof).
  * `mortar_solution(target, target_vel, ...)`: `target` is the point the shell should come down on (a
    ground splash point is the feet position); the impact is computed at that z. The returned impact is the
    forward-simulated landing point of the returned (yaw, pitch), so it equals the predicted target point.
  * `mortar_gate` simulates the ACTUAL torso aim (`me.aim`) down to the target's z and compares with
    `target_pred` (normally `solution[2]`), so it only opens once the torso has slewed onto the solution.
  * `laser_decision` has an optional trailing `t` keyword (match time) for the `short_until` check; the
    degrade-level and R9.6 conditions are folded into `safe_window_ok` by the caller.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from contracts import AIM_PRIO, AimRequest, ShellPred, SelfState, Track, Vec2, Vec3, WeaponState

TAU = 2.0 * math.pi


def wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, TAU)
    if a <= 0:
        a += TAU
    return a - math.pi


# ------------------------------------------------------------------ small helpers
def _dims(tr: Track, C) -> Tuple[float, float]:
    """(radius, height) of a track, scaled by its cubes (mechs) or the NPC constants."""
    if tr.kind == "npc":
        return float(C.path("npc.radius", 1.8)), float(C.path("npc.height", 3.5))
    k = float(C.path("cubes.size_per_cube", 0.05))
    s = 1.0 + k * float(tr.cubes or 0)
    return float(C.path("mech.radius", 2.5)) * s, float(C.path("mech.height", 9.0)) * s


def _muzzle_z(me: SelfState) -> float:
    if me.muzzle_z:
        return me.muzzle_z
    return me.pos[2] + float(me.stats.get("muzzle_height", 7.0))


def _eye_z(me: SelfState) -> float:
    if me.eye_z:
        return me.eye_z
    return me.pos[2] + float(me.stats.get("eye_height", 8.0))


def _pitch_limits(me: SelfState) -> Tuple[float, float]:
    return float(me.stats.get("min_pitch", -0.7)), float(me.stats.get("max_pitch", 1.3))


def _dir(yaw: float, pitch: float) -> Vec3:
    cp = math.cos(pitch)
    return (cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch))


def _ang_err(yaw_a: float, pitch_a: float, yaw_b: float, pitch_b: float) -> float:
    a, b = _dir(yaw_a, pitch_a), _dir(yaw_b, pitch_b)
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.acos(dot)


def _surface(mapdata, x: float, y: float) -> float:
    """Surface height tolerant of the mapdata API name (contract: surface_h)."""
    if mapdata is None:
        return -1e9
    fn = getattr(mapdata, "surface_h", None) or getattr(mapdata, "surface", None)
    if fn is None:
        return -1e9
    try:
        return float(fn(x, y))
    except Exception:
        return -1e9


# ------------------------------------------------------------------ gun (F1, 8.7)
def gun_lead(me: SelfState, tr: Track, gun: WeaponState, C) -> Tuple[float, float, float]:
    """Intercept aim for a constant-velocity target: (yaw, pitch, bullet flight time).

    The bullet leaves `R_self + SHOT_OFFSET_PAD` out along the aim, so the origin is part of the
    iteration. The target moves for AIM_LATENCY_S plus the flight time; a target that is rising or
    falling (|vz| > 0.5) also gets gravity."""
    speed = float(gun.stats.get("speed", C.path("weapons.gun.speed", 252.0)))
    _, h = _dims(tr, C)
    p0 = (tr.pos[0], tr.pos[1], tr.pos[2] + C.TARGET_CENTER_FRAC * h)
    v = tr.vel
    m = (me.pos[0], me.pos[1], _muzzle_z(me))
    off = float(me.radius) + C.SHOT_OFFSET_PAD
    lat = float(C.AIM_LATENCY_S)
    g = float(C.path("mech.gravity", 22.0))
    airborne = abs(v[2]) > 0.5

    def target_at(T: float) -> Vec3:
        z = p0[2] + v[2] * T - (0.5 * g * T * T if airborne else 0.0)
        return (p0[0] + v[0] * T, p0[1] + v[1] * T, z)

    d = (p0[0] - m[0], p0[1] - m[1], p0[2] - m[2])
    n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1.0
    dirv = (d[0] / n, d[1] / n, d[2] / n)
    tau = max(0.0, n - off) / speed
    for _ in range(max(1, int(C.IMPL_LEAD_ITERS)) + 1):
        o = (m[0] + off * dirv[0], m[1] + off * dirv[1], m[2] + off * dirv[2])
        P = target_at(lat + tau)
        d = (P[0] - o[0], P[1] - o[1], P[2] - o[2])
        n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1e-9
        tau = n / speed
        dirv = (d[0] / n, d[1] / n, d[2] / n)
    yaw = math.atan2(dirv[1], dirv[0])
    pitch = math.asin(max(-1.0, min(1.0, dirv[2])))
    lo, hi = _pitch_limits(me)
    return yaw, max(lo, min(hi, pitch)), tau


def gun_gate(me: SelfState, tr: Track, gun: WeaponState, aim_target: Tuple[float, float], C) -> bool:
    """F1: actual aim error to the lead point < atan(GUN_FIRE_TOL * (R_t + gun_radius) / d), d <= range,
    and a round is available."""
    if gun.ammo is not None and gun.ammo <= 0:
        return False
    if gun.reload and gun.reload > 0 and (gun.ammo is None or gun.ammo <= 0):
        return False
    r_t, h = _dims(tr, C)
    mx, my, mz = me.pos[0], me.pos[1], _muzzle_z(me)
    dx, dy = tr.pos[0] - mx, tr.pos[1] - my
    dz = tr.pos[2] + C.TARGET_CENTER_FRAC * h - mz
    d_h = math.hypot(dx, dy)
    rng = float(gun.stats.get("range", C.path("weapons.gun.range", 230.0)))
    if d_h > rng:
        return False
    d = max(math.sqrt(d_h * d_h + dz * dz), 0.1)
    tol = math.atan(C.GUN_FIRE_TOL * (r_t + float(C.path("projectile.gun_radius", 0.3))) / d)
    return _ang_err(me.aim[0], me.aim[1], aim_target[0], aim_target[1]) < tol


# ------------------------------------------------------------------ laser (F2, R9.5, R9.6)
def laser_point(me: SelfState, tr: Track, C) -> Tuple[float, float]:
    """Hitscan aim: the target's center advanced by AIM_LATENCY_S. Used for the whole charge."""
    _, h = _dims(tr, C)
    lat = float(C.AIM_LATENCY_S)
    px = tr.pos[0] + tr.vel[0] * lat
    py = tr.pos[1] + tr.vel[1] * lat
    pz = tr.pos[2] + C.TARGET_CENTER_FRAC * h + tr.vel[2] * lat
    dx, dy, dz = px - me.pos[0], py - me.pos[1], pz - _muzzle_z(me)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, max(math.hypot(dx, dy), 0.1))
    lo, hi = _pitch_limits(me)
    return yaw, max(lo, min(hi, pitch))


def laser_decision(me: SelfState, tr: Optional[Track], laser: WeaponState, los_now: bool, los_breaks_next: bool,
                   safe_window_ok: bool, C, t: Optional[float] = None) -> Tuple[bool, bool]:
    """(hold_trigger, release_now).

    Not charging: start only with a visible target in LOS that will not break LOS next tick, a clear R9.6
    safe window, off cooldown and not shorted (when `t` is given). Charging: keep holding while the target
    is in LOS; release at full charge once the aim error is inside atan(LASER_FIRE_TOL * R_t / d); release
    early at >= LASER_RELEASE_MIN_FRAC of the charge when LOS breaks next tick; release at once when the
    target or LOS is lost, and at full charge when the safe window is no longer ok (R9.5)."""
    full = float(laser.stats.get("charge", C.path("weapons.laser.charge", 0.75)))
    charge = float(laser.charge or 0.0)
    charging = charge > 0.0
    if not charging:
        if tr is None or not los_now or los_breaks_next or not safe_window_ok:
            return False, False
        if laser.cooldown and laser.cooldown > 0.0:
            return False, False
        if t is not None and t < me.short_until:
            return False, False
        return True, False
    if tr is None or not los_now:
        return False, True
    if los_breaks_next and charge >= C.LASER_RELEASE_MIN_FRAC * full:
        return False, True
    if charge + 1e-9 >= full:
        if not safe_window_ok:
            return False, True
        yaw, pitch = laser_point(me, tr, C)
        r_t, _ = _dims(tr, C)
        d = max(math.hypot(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1]), 0.1)
        tol = math.atan(C.LASER_FIRE_TOL * r_t / d)
        if _ang_err(me.aim[0], me.aim[1], yaw, pitch) < tol:
            return False, True
    return True, False


# ------------------------------------------------------------------ lightning (F3, A-19)
def lightning_gate(world, mapdata, slot: int, C, allow_npc: bool = False) -> Tuple[bool, Optional[int]]:
    """F3: (fire, target_id).

    A mech is a candidate when it is visible, within the weapon range in 3D, inside LIGHTNING_CONE_MARGIN
    of the narrow cone reading (cone_deg / 2) in yaw, and in LOS. The shot is allowed only when the nearest
    eligible object inside the WIDE cone reading (+-cone_deg) is a mech, or a crate within chain_range of
    the candidate; an NPC nearest blocks unless `allow_npc` (then, with no mech candidate, the nearest NPC
    in the narrow cone is itself the target)."""
    me = world.me
    if slot < 0 or slot >= len(me.weapons) or me.weapons[slot] is None or me.weapons[slot].kind != "lightning":
        return False, None
    w = me.weapons[slot]
    cost = float(w.stats.get("energy", C.path("weapons.lightning.energy", 3.0)))
    if me.energy < cost:
        return False, None
    rng = float(w.stats.get("range", C.LIGHTNING_RANGE))
    cone = float(w.stats.get("cone_deg", C.path("weapons.lightning.cone_deg", 30.0)))
    chain_r = float(w.stats.get("chain_range", C.path("weapons.lightning.chain_range", 14.0)))
    narrow = math.radians(C.LIGHTNING_CONE_MARGIN * cone / 2.0)
    wide = math.radians(cone)
    eye = (me.pos[0], me.pos[1], _eye_z(me))
    aim_yaw = me.aim[0]
    t_now = float(getattr(world.tick, "t", 0.0))

    def los(p: Vec3) -> bool:
        try:
            return bool(mapdata.los(eye, p)) if mapdata is not None else True
        except Exception:
            return True

    objs: List[Tuple[float, str, int, Vec3, float]] = []   # (d3, kind, id, center, yaw_err)
    for tr in world.tracks.values():
        if not tr.visible or tr.est:
            continue
        _, h = _dims(tr, C)
        c = (tr.pos[0], tr.pos[1], tr.pos[2] + C.TARGET_CENTER_FRAC * h)
        d3 = math.sqrt((c[0] - eye[0]) ** 2 + (c[1] - eye[1]) ** 2 + (c[2] - eye[2]) ** 2)
        if d3 > rng:
            continue
        yerr = abs(wrap(math.atan2(c[1] - eye[1], c[0] - eye[0]) - aim_yaw))
        if yerr > wide:
            continue
        objs.append((d3, "npc" if tr.kind == "npc" else "mech", tr.id, c, yerr))
    for cr in getattr(world, "crates", {}).values():
        if cr.last_seen < t_now - 0.11:
            continue
        c = (cr.pos[0], cr.pos[1], cr.pos[2] + 1.0)
        d3 = math.sqrt((c[0] - eye[0]) ** 2 + (c[1] - eye[1]) ** 2 + (c[2] - eye[2]) ** 2)
        if d3 > rng:
            continue
        yerr = abs(wrap(math.atan2(c[1] - eye[1], c[0] - eye[0]) - aim_yaw))
        if yerr > wide:
            continue
        objs.append((d3, "crate", cr.id, c, yerr))
    objs.sort(key=lambda o: o[0])
    objs = [o for o in objs if los(o[3])]
    if not objs:
        return False, None

    cand = next((o for o in objs if o[1] == "mech" and o[4] < narrow), None)
    if cand is None:
        if allow_npc:
            npc = next((o for o in objs if o[1] == "npc" and o[4] < narrow), None)
            if npc is not None and objs[0][1] in ("npc", "mech"):
                return True, npc[2]
        return False, None
    first = objs[0]
    if first[1] == "mech":
        return True, cand[2]
    if first[1] == "npc":
        return (True, cand[2]) if allow_npc else (False, None)
    # a crate is nearest: it takes the primary arc and chains on to the mech only if close enough
    cr = first[3]
    if math.sqrt(sum((cr[i] - cand[3][i]) ** 2 for i in range(3))) <= chain_r:
        return True, cand[2]
    return False, None


# ------------------------------------------------------------------ mortar (F4, 8.7)
def _mortar_params(mortar: WeaponState, C) -> Tuple[float, float]:
    v = float(mortar.stats.get("speed", C.path("weapons.mortar.speed", 72.0)))
    g = float(mortar.stats.get("gravity", C.path("weapons.mortar.gravity", 22.0)))
    return v, g


def _tube(me: SelfState, C) -> Vec3:
    return (me.pos[0], me.pos[1],
            me.pos[2] + float(C.path("mech.height", 9.0)) * float(me.s or 1.0) + C.MORTAR_TUBE_PAD)


def mortar_impact(me: SelfState, mortar: WeaponState, yaw: float, pitch: float, z_target: float,
                  C) -> Optional[Tuple[Vec3, float]]:
    """Forward ballistics: where a shell fired along (yaw, pitch) comes down through height z_target
    (descending branch). Returns (impact, flight_time) or None if it never reaches that height."""
    v, g = _mortar_params(mortar, C)
    off = float(me.radius) + C.SHOT_OFFSET_PAD
    tx, ty, tz = _tube(me, C)
    cp, sp = math.cos(pitch), math.sin(pitch)
    lx, ly, lz = tx + off * cp * math.cos(yaw), ty + off * cp * math.sin(yaw), tz + off * sp
    vz = v * sp
    a = 0.5 * g
    disc = vz * vz + 4.0 * a * (lz - z_target)
    if disc < 0:
        return None
    T = (vz + math.sqrt(disc)) / (2.0 * a)
    hd = v * cp * T
    return (lx + hd * math.cos(yaw), ly + hd * math.sin(yaw), z_target), T


def _arc_clear(mapdata, me: SelfState, yaw: float, pitch: float, v: float, g: float, D: float, C) -> bool:
    """Sample the arc every IMPL_ARC_SAMPLE_M of ground distance against the surface grid. The last meter
    before the impact is skipped (the target may stand beside a wall or on a roof edge)."""
    if mapdata is None:
        return True
    off = float(me.radius) + C.SHOT_OFFSET_PAD
    tx, ty, tz = _tube(me, C)
    cp, sp = math.cos(pitch), math.sin(pitch)
    lx, ly, lz = tx + off * cp * math.cos(yaw), ty + off * cp * math.sin(yaw), tz + off * sp
    vh = v * cp
    if vh < 1e-6:
        return False
    step = max(1.0, float(C.IMPL_ARC_SAMPLE_M))
    cy, sy = math.cos(yaw), math.sin(yaw)
    s = 0.0
    last = max(0.0, D - 1.0)
    while True:
        t = s / vh
        z = lz + v * sp * t - 0.5 * g * t * t
        if z < _surface(mapdata, lx + s * cy, ly + s * sy) - 0.3:
            return False
        if s >= last:
            return True
        s = min(s + step, last)


def mortar_solution(me: SelfState, mapdata, target: Vec3, target_vel: Vec3, mortar: WeaponState,
                    C) -> Optional[Tuple[float, float, Vec3]]:
    """(yaw, pitch, impact) for a shell to come down on `target` (moving with `target_vel`), or None.

    tan(theta) = (v^2 -+ sqrt(v^4 - g(gD^2 + 2Hv^2))) / (gD) with D and H measured from the shell's launch
    point (tube + R_self + SHOT_OFFSET_PAD along the aim) and the target advanced by AIM_LATENCY_S + flight
    time. Low arc first; a blocked arc falls through to the high arc; any pitch outside
    [min_pitch, max_pitch] is rejected (so the high arc is clamped away at short range)."""
    v, g = _mortar_params(mortar, C)
    lo, hi = _pitch_limits(me)
    tx, ty, tz = _tube(me, C)
    off = float(me.radius) + C.SHOT_OFFSET_PAD
    lat = float(C.AIM_LATENCY_S)
    for high in (False, True):
        theta = 0.7854
        tof = math.hypot(target[0] - tx, target[1] - ty) / v
        yaw = 0.0
        ok = True
        for _ in range(6):
            P = (target[0] + target_vel[0] * (tof + lat), target[1] + target_vel[1] * (tof + lat),
                 target[2] + target_vel[2] * (tof + lat))
            dx, dy = P[0] - tx, P[1] - ty
            Dt = math.hypot(dx, dy)
            yaw = math.atan2(dy, dx)
            D = Dt - off * math.cos(theta)
            H = P[2] - (tz + off * math.sin(theta))
            if D < 1.0:
                ok = False
                break
            disc = v ** 4 - g * (g * D * D + 2.0 * H * v * v)
            if disc < 0:
                ok = False
                break
            root = math.sqrt(disc)
            tan_t = (v * v + root) / (g * D) if high else (v * v - root) / (g * D)
            theta = math.atan(tan_t)
            tof = D / (v * math.cos(theta))
        if not ok or theta < lo or theta > hi:
            continue
        if not _arc_clear(mapdata, me, yaw, theta, v, g, D, C):
            continue
        res = mortar_impact(me, mortar, yaw, theta, P[2], C)
        if res is None:
            continue
        return yaw, theta, res[0]
    return None


def mortar_gate(me: SelfState, solution, target_pred: Vec3, mortar: WeaponState, C) -> bool:
    """F4: the shell fired along our ACTUAL aim lands within MORTAR_AIM_TOL_FRAC * splash_radius of
    `target_pred`, at least MORTAR_SELF_SAFE_M from us, and a clear solution exists."""
    if solution is None:
        return False
    res = mortar_impact(me, mortar, me.aim[0], me.aim[1], target_pred[2], C)
    if res is None:
        return False
    ip = res[0]
    splash = float(mortar.stats.get("splash_radius", C.path("weapons.mortar.splash_radius", 7.0)))
    if math.hypot(ip[0] - target_pred[0], ip[1] - target_pred[1]) > C.MORTAR_AIM_TOL_FRAC * splash:
        return False
    if math.hypot(ip[0] - me.pos[0], ip[1] - me.pos[1]) < C.MORTAR_SELF_SAFE_M:
        return False
    sol_imp = solution[2]
    return math.hypot(sol_imp[0] - me.pos[0], sol_imp[1] - me.pos[1]) >= C.MORTAR_SELF_SAFE_M


def shell_impact(mapdata, pos: Vec3, vel: Vec3, gravity: float, max_t: float = 8.0,
                 dt: float = 0.04) -> Optional[Tuple[Vec3, float]]:
    """Mortar impact predictor for our dodge: integrate a visible shell (pos, vel) under gravity until it
    drops below the surface grid. Returns (impact, time_from_now) or None if it does not land in max_t."""
    x, y, z = pos
    vx, vy, vz = vel
    t = 0.0
    while t < max_t:
        t2 = t + dt
        nx, ny = x + vx * dt, y + vy * dt
        nz = z + vz * dt - 0.5 * gravity * dt * dt
        vz -= gravity * dt
        if nz <= _surface(mapdata, nx, ny):
            # linear refinement inside this step
            g0 = z - _surface(mapdata, x, y)
            g1 = nz - _surface(mapdata, nx, ny)
            f = g0 / (g0 - g1) if g0 != g1 else 1.0
            f = max(0.0, min(1.0, f))
            ix, iy = x + (nx - x) * f, y + (ny - y) * f
            return (ix, iy, _surface(mapdata, ix, iy)), t + dt * f
        x, y, z, t = nx, ny, nz, t2
    return None


def predict_shell(mapdata, shell_id: int, pos: Vec3, vel: Vec3, t_now: float, C) -> Optional[ShellPred]:
    """ShellPred (4.5) for one visible mortar shell: impact point, absolute impact time, and the splash
    radius scaled by MORTAR_ENEMY_SPLASH_MULT."""
    g = float(C.path("weapons.mortar.gravity", 22.0))
    res = shell_impact(mapdata, pos, vel, g)
    if res is None:
        return None
    imp, dt = res
    splash = float(C.path("weapons.mortar.splash_radius", 7.0)) * C.MORTAR_ENEMY_SPLASH_MULT
    return ShellPred(id=shell_id, impact=imp, t_impact=t_now + dt, splash_r=splash)


# ------------------------------------------------------------------ scan pattern (AIM10) and look (AIM8)
_LOOK_KIND_ORDER = {"mortar": 0, "laser_charge": 1, "laser": 1, "lightning": 2, "gun": 3}
_BAND_ORDER = {"near": 0, "mid": 1}


class Scanner:
    """AIM10 sweep and AIM8 sound-look timing (H11)."""

    def __init__(self, C, rng) -> None:
        self.C = C
        self.rng = rng
        self._phase = rng.random() * C.SCAN_PERIOD_S
        self._watch: Optional[float] = None
        self._watch_t = -1e9
        self._look_gid: Optional[int] = None
        self._look_until = -1e9
        self._look_yaw = 0.0

    def _watch_dir(self, world, mapdata) -> float:
        me = world.me
        t = world.tick.t
        n = max(1, int(self.C.IMPL_SIGHT_SECTORS))
        best, best_v = None, -1.0
        sound_bias: Optional[float] = None
        gh = [g for g in getattr(world, "ghosts", {}).values() if t - g.last_t <= 2.0 * self.C.SOUND_REACT_AGE_S]
        if gh:
            sound_bias = max(gh, key=lambda g: g.last_t).bearing
        for k in range(n):
            try:
                v = float(mapdata.sight5_dir(me.pos[0], me.pos[1], k)) if mapdata is not None else float("nan")
            except Exception:
                v = float("nan")
            if v != v:
                continue
            yaw = wrap((k + 0.5) * TAU / n)
            if sound_bias is not None:
                v *= 1.0 + 0.5 * max(0.0, math.cos(wrap(yaw - sound_bias)))
            if v > best_v:
                best_v, best = v, yaw
        if best is None:
            return sound_bias if sound_bias is not None else (self._watch if self._watch is not None else me.aim[0])
        return best

    def scan(self, world, mapdata) -> Tuple[float, float]:
        """(yaw, pitch): a sinusoidal sweep of +-SCAN_HALF_ARC_DEG around the watch direction with period
        SCAN_PERIOD_S. The watch direction (most open sight sector, biased to the last sound) is re-chosen
        once per period so the sweep does not jitter."""
        C = self.C
        t = float(world.tick.t)
        if self._watch is None or t - self._watch_t >= C.SCAN_PERIOD_S:
            self._watch = self._watch_dir(world, mapdata)
            self._watch_t = t
        half = math.radians(C.SCAN_HALF_ARC_DEG)
        yaw = wrap(self._watch + half * math.sin(TAU * (t + self._phase) / C.SCAN_PERIOD_S))
        return yaw, 0.0

    def look(self, world) -> Optional[AimRequest]:
        """AIM8: look at the bearing of an unassociated near/mid sound heard within SOUND_REACT_AGE_S, for
        SCAN_LOOK_S. Order: kind (mortar > laser_charge > lightning > gun > others), nearer band, most
        recent. A ghost is looked at again only after a newer sound."""
        C = self.C
        t = float(world.tick.t)
        if self._look_gid is not None and t < self._look_until:
            return AimRequest(prio=AIM_PRIO["AIM8"], rule="AIM8", yaw=self._look_yaw, pitch=0.0)
        self._look_gid = None
        best = None
        best_key = None
        for g in getattr(world, "ghosts", {}).values():
            if g.band not in _BAND_ORDER or t - g.last_t > C.SOUND_REACT_AGE_S:
                continue
            if g.looked_at_t >= g.last_t:
                continue
            key = (_LOOK_KIND_ORDER.get(g.last_kind, 9), _BAND_ORDER[g.band], -g.last_t, g.id)
            if best_key is None or key < best_key:
                best, best_key = g, key
        if best is None:
            return None
        best.looked_at_t = t
        self._look_gid = best.id
        self._look_until = t + C.SCAN_LOOK_S
        self._look_yaw = best.bearing
        return AimRequest(prio=AIM_PRIO["AIM8"], rule="AIM8", yaw=best.bearing, pitch=0.0,
                          tiebreak=float(best_key[0] * 10 + best_key[1]))
