"""X5 SAFE controller (plan 5.3, X5 row): a pure function of (obs, bound config).

Used by runner.py when the T1 switch is on (too many exceptions), when act() raises (same tick),
when act() overruns HARD_DEADLINE_MS, and before the Bot finished __init__. No other bot modules,
no state: the periodic jump is derived from obs["t"]. Thread-safe (reads only).

    move:  straight to zone.next_center
    aim:   nearest mech in obs.mechs, else the move heading (pitch 0)
    fire:  gun if the actual aim error < the F1 tolerance and in range; lightning if a mech is in
           range (3D) and inside the cone margin with energy; laser / mortar never
    boost: if outside the zone
    jump:  every SAFE_JUMP_PERIOD_S while horizontal speed < STUCK_SPEED (and we want to move)
    pick:  offers are still answered (GP list for the held loadout, then UPGRADE_FALLBACK)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import config as _config


def _wrap(a: float) -> float:
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def pick_upgrade(offer: Optional[Dict[str, Any]], held_kinds: List[str], C: Any) -> Optional[int]:
    """R14.1 simplified: first offered id in GP's list for the held loadout, then UPGRADE_FALLBACK, else 0."""
    if not offer or not offer.get("choices"):
        return None
    ids = [c.get("id") if isinstance(c, dict) else c for c in offer["choices"]]
    key = _config.loadout_key(held_kinds)
    lists = getattr(_config, "UPGRADE_PRIORITY", {}) or {}
    if key in ("GL", "GZ", "ZL"):
        order = list(lists.get(key, []))
    elif key == "LM":  # GP: LM uses the ZL list without the lens
        order = [u for u in lists.get("ZL", []) if u != "laser_damage"]
    else:
        order = list(lists.get("MECH", []))
    order += list(getattr(C, "UPGRADE_FALLBACK", []) or [])
    for uid in order:
        if uid in ids:
            return ids.index(uid)
    return 0


def act(obs: Dict[str, Any], C: Any) -> Dict[str, Any]:
    """The SAFE action for this observation (without `tick`; the runner adds it)."""
    me = obs.get("self") or {}
    pos = me.get("pos") or [0.0, 0.0, 0.0]
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0
    st = me.get("stats") or {}
    zone = obs.get("zone") or {}
    goal = zone.get("next_center") or zone.get("center") or [x, y]
    gx, gy = float(goal[0]) - x, float(goal[1]) - y
    dist = math.hypot(gx, gy)
    action: Dict[str, Any] = {}
    if dist > 1.0:
        move = [gx / dist, gy / dist]
        heading = math.atan2(gy, gx)
    else:
        move = [0.0, 0.0]
        heading = float((me.get("aim") or [0.0, 0.0])[0])
    action["move"] = move

    # aim: nearest mech (obs.mechs may include remembered mechs: A-14; SAFE accepts that)
    target = None
    best = 1e18
    for m in obs.get("mechs") or ():
        if m.get("kind") != "mech":
            continue
        p = m.get("pos") or [0, 0, 0]
        d2 = (p[0] - x) ** 2 + (p[1] - y) ** 2
        if d2 < best:
            best, target = d2, m
    size_k = float(C.path("cubes.size_per_cube", 0.05))
    muzzle_z = z + float(st.get("muzzle_height", 7.0))
    fire = [False, False]
    if target is not None:
        tp = target["pos"]
        d = math.sqrt(best)
        s_t = 1.0 + size_k * float(target.get("cubes", 0) or 0)
        tz = tp[2] + C.TARGET_CENTER_FRAC * float(C.path("mech.height", 9.0)) * s_t
        yaw = math.atan2(tp[1] - y, tp[0] - x)
        pitch = math.atan2(tz - muzzle_z, max(d, 0.1))
        pitch = max(float(st.get("min_pitch", -0.7)), min(float(st.get("max_pitch", 1.3)), pitch))
        action["aim"] = [yaw, pitch]
        cur = me.get("aim") or [yaw, pitch]
        yaw_err = abs(_wrap(float(cur[0]) - yaw))
        err = math.hypot(yaw_err, float(cur[1]) - pitch)
        R_t = float(C.path("mech.radius", 2.5)) * s_t
        d3 = math.sqrt(best + (tz - muzzle_z) ** 2)
        energy = float(me.get("energy", 0) or 0)
        for i, w in enumerate((me.get("weapons") or [None, None])[:2]):
            if not isinstance(w, dict):
                continue
            ws = w.get("stats") or {}
            kind = w.get("kind")
            if kind == "gun":
                tol = math.atan(C.GUN_FIRE_TOL * (R_t + float(C.path("projectile.gun_radius", 0.3))) / max(d, 0.1))
                rng = float(ws.get("range", C.path("weapons.gun.range", 230)))
                if err < tol and d <= rng and (w.get("ammo") is None or w.get("ammo", 0) > 0):
                    fire[i] = True
            elif kind == "lightning":
                rng = float(ws.get("range", C.LIGHTNING_RANGE))
                cone = float(ws.get("cone_deg", C.path("weapons.lightning.cone_deg", 30)))
                cost = float(ws.get("energy", C.path("weapons.lightning.energy", 3)))
                if d3 <= rng and yaw_err < math.radians(C.LIGHTNING_CONE_MARGIN * cone / 2) and energy >= cost:
                    fire[i] = True
    else:
        action["aim"] = [heading, 0.0]
    action["fire"] = fire

    action["boost"] = not bool(me.get("in_zone", True)) and float(me.get("energy", 0) or 0) > 0
    vel = me.get("vel") or [0, 0, 0]
    speed = math.hypot(float(vel[0]), float(vel[1]))
    t = float(obs.get("t", 0.0))
    period = max(0.1, float(C.SAFE_JUMP_PERIOD_S))
    step = float(C.path("match.decide_every", 2)) / float(C.path("match.tick_hz", 20))
    wants_move = move[0] != 0.0 or move[1] != 0.0
    # one decision per period: the decision whose time falls in the first `step` of the period
    action["jump"] = bool(wants_move and me.get("grounded", True) and speed < C.STUCK_SPEED
                          and math.fmod(t, period) < step - 1e-6)

    pk = pick_upgrade(obs.get("offer"), [w.get("kind") for w in (me.get("weapons") or []) if isinstance(w, dict)], C)
    if pk is not None:
        action["pick"] = pk
    return action
