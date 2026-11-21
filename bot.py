"""Early edge-camp loot bot: self-defense, rim play, lightning/laser prio."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import mechfight as mf

LOOT_CHASE = 90.0
THREAT_COVER_M = 75.0
RIM_INNER = 0.83  # stay in outer band of current zone (e.g. 250–300 on a 300 r circle)
RIM_OUTER = 0.98
SCAN_RATE = 2.2
DEFEND_MEMORY_S = 1.2


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


def _hp_frac(entity: dict, fallback: float = 1.0) -> float:
    hp, mx = entity.get("hp"), entity.get("max_hp")
    if hp is None or not mx:
        return fallback
    return float(hp) / float(mx)


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


def _load_building_points(map_dir: str) -> list[tuple[float, float]]:
    """Sample points just outside building rings for crude cover seeking."""
    path = os.path.join(map_dir, "v2", "world.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            world = json.load(f)
    except OSError:
        return []
    pts: list[tuple[float, float]] = []
    for p in world.get("prisms") or []:
        if p.get("kind") not in ("building", "stadium"):
            continue
        ring = p.get("ring") or []
        if len(ring) < 6:
            continue
        xs, ys = ring[0::2], ring[1::2]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        # edge midpoints pushed slightly outward from centroid
        n = len(xs)
        for i in range(0, n, max(1, n // 8)):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]
            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            dx, dy = mx - cx, my - cy
            L = math.hypot(dx, dy) or 1.0
            pts.append((mx + dx / L * 6.0, my + dy / L * 6.0))
        if len(pts) > 4000:
            break
    return pts


class Bot:
    name = "early-loot"

    def __init__(self, hello: mf.Hello):
        self.pickup_r = float(hello["config"]["loot"]["pickup_radius"])
        self.scan_phase = 0.0
        self._last_t = 0.0
        self.defend_until = -1.0
        self.defend_yaw: Optional[float] = None
        map_dir = hello.get("map", {}).get("dir") or "/arena"
        self.cover_pts = _load_building_points(map_dir)
        mf.log(f"cover pts loaded: {len(self.cover_pts)} from {map_dir}")

    def _nearest_cover(self, pos, enemy_pos=None) -> Optional[tuple[float, float]]:
        if not self.cover_pts:
            return None
        x, y = _xy(pos)
        best, best_d = None, 1e18
        for cx, cy in self.cover_pts:
            d = math.hypot(cx - x, cy - y)
            if d < 8 or d > 80:
                continue
            # prefer cover that puts building between us and enemy
            if enemy_pos is not None:
                ex, ey = _xy(enemy_pos)
                # score: closer cover that is not past the enemy
                toward_enemy = (ex - x) * (cx - x) + (ey - y) * (cy - y)
                if toward_enemy > 0 and d > _dist(pos, enemy_pos):
                    continue
            if d < best_d:
                best_d, best = d, (cx, cy)
        return best

    def _rim_goal(self, pos, zone: dict) -> list[float]:
        """Stay in the outer band of the *current* circle, not next_center."""
        cx, cy = float(zone["center"][0]), float(zone["center"][1])
        R = float(zone["radius"])
        x, y = _xy(pos)
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / dist, dy / dist
        # tangential wander so we spin/orbit the rim while scanning
        tx, ty = -uy, ux
        orbit = 0.55 * math.sin(self.scan_phase)
        inner, outer = R * RIM_INNER, R * RIM_OUTER
        target_r = 0.5 * (inner + outer)
        if dist < inner:
            # too far in — push out to rim band
            return [cx + ux * target_r + tx * 40 * orbit, cy + uy * target_r + ty * 40 * orbit]
        if dist > outer:
            # near / past edge — ease back into band
            return [cx + ux * target_r + tx * 25 * orbit, cy + uy * target_r + ty * 25 * orbit]
        # already in band — keep orbiting
        return [cx + ux * dist + tx * 55 * orbit, cy + uy * dist + ty * 55 * orbit]

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        pos = me["pos"]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        t = float(obs["t"])
        dt = max(0.05, t - self._last_t)
        self._last_t = t
        self.scan_phase = _norm(self.scan_phase + SCAN_RATE * dt)

        act: dict[str, Any] = {"tick": obs["tick"]}
        holding = {k for k in (_kind(w) for w in me["weapons"]) if k}
        my_frac = _hp_frac(me)

        if obs.get("offer"):
            act["pick"] = _pick_upgrade(obs["offer"], holding)

        # --- self-defense trigger from damage ---
        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("bearing") is not None:
                self.defend_until = t + DEFEND_MEMORY_S
                self.defend_yaw = float(ev["bearing"])

        defending = t <= self.defend_until and self.defend_yaw is not None

        slot1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        slot1_kind = _kind(slot1)

        # --- loot: lightning > laser, also cubes/crates; prefer outward (longer chase) ---
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

        if loot_goal is None:
            best_d, best_p = LOOT_CHASE, None
            for c in list(obs.get("cubes") or []) + list(obs.get("crates") or []):
                d = _dist(pos, c["pos"])
                if d < best_d:
                    best_d, best_p = d, c["pos"]
            loot_goal = best_p

        zone = obs["zone"]
        rim_goal = self._rim_goal(pos, zone)
        # Prefer visible loot over rim; never hard-commit to next_center.
        goal = loot_goal if loot_goal is not None else rim_goal
        move = _move_toward(pos, goal)

        # Soft sound avoid — but don't cancel loot: only side-slip if we have a loot goal
        for s in obs.get("sounds") or []:
            if s.get("kind") not in ("gun", "explosion", "laser", "laser_charge", "mortar", "lightning"):
                continue
            if s.get("dist") == "far":
                continue
            b = float(s["bearing"])
            w = 0.25 if loot_goal is not None else (0.45 if s.get("dist") == "near" else 0.25)
            move = [move[0] - math.cos(b) * w, move[1] - math.sin(b) * w]

        enemies = sorted(
            (m for m in (obs.get("mechs") or []) if m.get("kind") == "mech"),
            key=lambda m: _dist(pos, m["pos"]),
        )
        nearest = enemies[0] if enemies else None
        nearest_d = _dist(pos, nearest["pos"]) if nearest else 1e9

        # Engage rules: defend always; else if equal/better HP shoot; if worse run
        fight = None
        flee = False
        if defending:
            # prefer visible enemy matching damage direction
            fight = nearest
            if nearest is None:
                fight = {"pos": [x + math.cos(self.defend_yaw), y + math.sin(self.defend_yaw), z]}
        elif nearest is not None:
            their = _hp_frac(nearest, fallback=1.0)
            if my_frac + 0.02 >= their:
                fight = nearest  # same or more HP → shoot first
            else:
                flee = True

        committing_fight = fight is not None and not flee and (
            defending or (nearest is not None and _hp_frac(nearest) <= my_frac + 0.02)
        )

        # Within 75m of an opponent: break to building cover unless we're starting the fight
        if nearest is not None and nearest_d <= THREAT_COVER_M and not committing_fight:
            cover = self._nearest_cover(pos, nearest["pos"])
            if cover is not None:
                move = _move_toward(pos, cover)
        elif flee and nearest is not None:
            cover = self._nearest_cover(pos, nearest["pos"])
            if cover is not None:
                move = _move_toward(pos, cover)
            else:
                # run away from them
                move = _move_toward(pos, [x - (float(nearest["pos"][0]) - x), y - (float(nearest["pos"][1]) - y)])

        move = _clamp_move(move)

        energy = float(me.get("energy") or 0)
        # Boost if outside zone or deep outside rim; not for next_center chase
        R = float(zone["radius"])
        dist_c = _dist(pos, zone["center"])
        need_boost = (not me.get("in_zone", True)) or dist_c > R * 0.99
        act["move"] = move
        act["boost"] = bool(need_boost and energy > 12)
        act["jump"] = False

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

        if fight is not None and committing_fight:
            tpos = fight["pos"]
            if defending and nearest is None and self.defend_yaw is not None:
                aim_yaw = self.defend_yaw
            else:
                aim_yaw = _yaw_to(pos, tpos)
                horiz = _dist(pos, tpos) or 1.0
                aim_pitch = math.atan2(float(tpos[2]) - z - 2.5, horiz)
            if gun_ready:
                fire[0] = True
            s1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
            if nearest is not None and _kind(s1) == "lightning" and isinstance(s1, dict):
                if nearest_d < 40 and float(s1.get("cooldown") or 0) <= 0.02 and energy > 8:
                    fire[1] = True
            elif nearest is not None and _kind(s1) == "laser" and isinstance(s1, dict) and nearest_d < 140:
                fire[1] = True
        elif flee and nearest is not None:
            # face while retreating
            aim_yaw = _yaw_to(pos, nearest["pos"])
            if gun_ready and nearest_d < 100:
                fire[0] = True  # shoot while running if we can
        else:
            # spin scan for loot / threats
            aim_yaw = _norm(aim_yaw + 1.1 * math.sin(self.scan_phase))
            if defending and self.defend_yaw is not None:
                aim_yaw = self.defend_yaw
                if gun_ready:
                    fire[0] = True
            if isinstance(gun, dict) and not enemies and not defending:
                ammo = float(gun.get("ammo") or 0)
                mag = float((gun.get("stats") or {}).get("mag") or 12)
                if ammo < mag and float(gun.get("reload") or 0) <= 0:
                    act["reload"] = True

        if isinstance(gun, dict) and float(gun.get("reload") or 0) <= 0:
            ammo = float(gun.get("ammo") or 0)
            if ammo <= 0 and not fire[0]:
                act["reload"] = True

        act["aim"] = [aim_yaw, aim_pitch]
        act["fire"] = fire

        mf.draw.label(f"rim t={t:.0f} hp={me['hp']:.0f} {'DEF' if defending else ''}")
        mf.draw.point([float(rim_goal[0]), float(rim_goal[1]), z], "#40c0ff", 5)
        if loot_goal is not None:
            mf.draw.point([float(loot_goal[0]), float(loot_goal[1]), z], "#ffd060", 6)
        if nearest is not None:
            tp = nearest["pos"]
            mf.draw.line([x, y, z + 7], [tp[0], tp[1], float(tp[2]) + 4], "#ff5040")
        return act  # type: ignore[return-value]


mf.run(Bot)
