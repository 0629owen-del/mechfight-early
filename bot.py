"""Early-game loot-rotate bot (curiosity submit)."""
from __future__ import annotations

import math
from typing import Any, Optional

import mechfight as mf

EARLY_T = 90.0
WEAK_HP_FRAC = 0.5
LOOT_CHASE = 55.0
SCAN_RATE = 1.8


def _xy(p) -> tuple[float, float]:
    return float(p[0]), float(p[1])


def _dist(a, b) -> float:
    return math.dist(_xy(a), _xy(b))


def _yaw_to(frm, to) -> float:
    x, y = _xy(frm)
    tx, ty = _xy(to)
    return math.atan2(ty - y, tx - x)


def _norm(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _move_toward(frm, to, scale: float = 1.0) -> list[float]:
    x, y = _xy(frm)
    tx, ty = _xy(to)
    dx, dy = tx - x, ty - y
    n = math.hypot(dx, dy) or 1.0
    s = max(0.0, min(1.0, scale))
    return [dx / n * s, dy / n * s]


def _clamp_move(v: list[float]) -> list[float]:
    n = math.hypot(v[0], v[1])
    if n <= 1.0 or n < 1e-9:
        return v
    return [v[0] / n, v[1] / n]


def _kind(w) -> Optional[str]:
    return w.get("kind") if isinstance(w, dict) else None


def _weak(m: dict) -> bool:
    hp, mx = m.get("hp"), m.get("max_hp")
    if hp is None or not mx:
        return False
    return float(hp) / float(mx) <= WEAK_HP_FRAC


def _pick_upgrade(offer: dict, holding: set[str]) -> int:
    scored = []
    for i, c in enumerate(offer.get("choices") or []):
        cid = c.get("id", "")
        if cid.startswith("gun_") or cid == "sensors":
            score = -100
        elif cid.startswith("lightning_") and "lightning" in holding:
            score = 50
        elif cid.startswith("laser_") and "laser" in holding:
            score = 40
        elif cid in ("armor", "plating"):
            score = 30
        elif cid == "servos":
            score = 20
        else:
            score = 0
        scored.append((score, i))
    scored.sort(reverse=True)
    return scored[0][1] if scored else 0


class Bot:
    name = "early-loot"

    def __init__(self, hello: mf.Hello):
        self.pickup_r = float(hello["config"]["loot"]["pickup_radius"])
        self.scan_phase = 0.0
        self._last_t = 0.0

    def act(self, obs: mf.Observation) -> mf.Action:
        # Curiosity submit: always run early policy.
        return self.act_early(obs)

    def is_early(self, obs: mf.Observation) -> bool:
        slot1 = obs["self"]["weapons"][1] if len(obs["self"]["weapons"]) > 1 else None
        return float(obs["t"]) < EARLY_T or slot1 is None

    def act_early(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        pos = me["pos"]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        dt = max(0.05, float(obs["t"]) - self._last_t)
        self._last_t = float(obs["t"])

        act: dict[str, Any] = {"tick": obs["tick"]}
        holding = {k for k in (_kind(w) for w in me["weapons"]) if k}

        if obs.get("offer"):
            act["pick"] = _pick_upgrade(obs["offer"], holding)

        slot1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        slot1_kind = _kind(slot1)

        # Lightning > laser for slot 1
        loot_goal = None
        pickups = []
        for p in obs.get("pickups") or []:
            kind = p.get("kind")
            if kind not in ("lightning", "laser"):
                continue
            if kind in holding and kind != slot1_kind:
                continue
            pickups.append(((0 if kind == "lightning" else 1), _dist(pos, p["pos"]), p))
        pickups.sort()
        if pickups:
            _, d, p = pickups[0]
            need = slot1_kind not in ("lightning", "laser") or (
                p["kind"] == "lightning" and slot1_kind == "laser"
            )
            if need:
                if d <= self.pickup_r + 0.5:
                    act["take"] = p["id"]
                    if slot1 is not None and slot1_kind != p["kind"]:
                        act["drop"] = 1
                elif d < LOOT_CHASE:
                    loot_goal = p["pos"]

        # Cubes / crates nearby
        if loot_goal is None:
            best_d, best_p = LOOT_CHASE, None
            for c in list(obs.get("cubes") or []) + list(obs.get("crates") or []):
                d = _dist(pos, c["pos"])
                if d < best_d:
                    best_d, best_p = d, c["pos"]
            loot_goal = best_p

        zone = obs["zone"]
        safe = zone["next_center"]
        dist_safe = _dist(pos, safe)
        next_r = float(zone["next_radius"])
        in_next = dist_safe < next_r * 0.85
        shrink_soon = float(obs["t"]) >= float(zone["shrink_start"]) - 15.0

        goal = loot_goal if (loot_goal is not None and (in_next or not shrink_soon)) else safe
        move = _move_toward(pos, goal)

        # Soft avoid combat sounds
        for s in obs.get("sounds") or []:
            if s.get("kind") not in ("gun", "explosion", "laser", "laser_charge", "mortar", "lightning"):
                continue
            if s.get("dist") == "far":
                continue
            b = float(s["bearing"])
            w = 0.55 if s.get("dist") == "near" else 0.3
            move = [move[0] - math.cos(b) * w, move[1] - math.sin(b) * w]
        move = _clamp_move(move)

        energy = float(me.get("energy") or 0)
        need_boost = (not me.get("in_zone", True)) or (shrink_soon and dist_safe > next_r * 0.9)
        act["move"] = move
        act["boost"] = bool(need_boost and energy > 15)
        act["jump"] = False

        enemies = sorted(
            (m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"),
            key=lambda m: _dist(pos, m["pos"]),
        )
        target = next((m for m in enemies if _weak(m)), None)

        face_yaw = None
        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("bearing") is not None:
                face_yaw = float(ev["bearing"])

        gun = me["weapons"][0] if me["weapons"] else None
        gun_ready = (
            isinstance(gun, dict)
            and float(gun.get("ammo") or 0) > 0
            and float(gun.get("reload") or 0) <= 0
            and float(gun.get("cooldown") or 0) <= 0.02
        )

        fire = [False, False]
        aim_yaw = _yaw_to(pos, goal)
        aim_pitch = 0.0

        if target is not None:
            tpos = target["pos"]
            aim_yaw = _yaw_to(pos, tpos)
            horiz = _dist(pos, tpos) or 1.0
            aim_pitch = math.atan2(float(tpos[2]) - z - 2.5, horiz)
            face_yaw = aim_yaw
            if gun_ready:
                fire[0] = True

            s1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
            if _kind(s1) == "lightning" and isinstance(s1, dict):
                if _dist(pos, tpos) < 40 and float(s1.get("cooldown") or 0) <= 0.02 and energy > 10:
                    fire[1] = True
                    act["move"] = _clamp_move(
                        [move[0] + _move_toward(pos, tpos, 0.7)[0], move[1] + _move_toward(pos, tpos, 0.7)[1]]
                    )
            elif _kind(s1) == "laser" and isinstance(s1, dict) and _dist(pos, tpos) < 120:
                fire[1] = True

            my_frac = float(me["hp"]) / max(float(me["max_hp"]), 1.0)
            thp, tmax = target.get("hp"), target.get("max_hp")
            their_frac = float(thp) / float(tmax) if thp is not None and tmax else my_frac
            if my_frac + 0.05 < their_frac:
                # kite away, keep facing
                act["move"] = _move_toward(pos, [x - (float(tpos[0]) - x), y - (float(tpos[1]) - y)])

            if isinstance(gun, dict) and float(gun.get("reload") or 0) <= 0:
                ammo = float(gun.get("ammo") or 0)
                if ammo <= 0:
                    act["reload"] = True
                elif ammo <= 4 and my_frac >= their_frac and not fire[0]:
                    act["reload"] = True
        else:
            self.scan_phase = _norm(self.scan_phase + SCAN_RATE * dt)
            aim_yaw = _norm(aim_yaw + 0.65 * math.sin(self.scan_phase))
            if face_yaw is not None:
                aim_yaw = face_yaw
            if isinstance(gun, dict) and not enemies:
                ammo = float(gun.get("ammo") or 0)
                mag = float((gun.get("stats") or {}).get("mag") or 12)
                if ammo < mag and float(gun.get("reload") or 0) <= 0:
                    act["reload"] = True

        act["aim"] = [face_yaw if (face_yaw is not None and target is None) else aim_yaw, aim_pitch]
        act["fire"] = fire

        mf.draw.label(f"early t={obs['t']:.0f} hp={me['hp']:.0f}")
        mf.draw.point([float(safe[0]), float(safe[1]), z], "#40c0ff", 5)
        if target is not None:
            tp = target["pos"]
            mf.draw.line([x, y, z + 7], [tp[0], tp[1], float(tp[2]) + 4], "#ff5040")
        return act  # type: ignore[return-value]


mf.run(Bot)
