#!/usr/bin/env python3
"""Scripted built-in-like opponents for minisim (stdlib only; each runs as its own bot process).

    python3 -u tools/minisim_bots.py duelist|rusher|lobber|looter

Archetypes (all: follow the zone, loot their preferred weapon early, pick up cubes, shoot crates when
nothing is around, lead moving targets, turn towards unseen attackers, scan with the torso, get unstuck
by jumping and jetting):

- duelist  gun + laser, holds 93-150 m (the recorded built-ins' band, median 139 m), strafes, charges the
           laser to full and releases when the aim is on the lead point.
- rusher   gun + lightning, boosts in to < 30 m and arcs.
- lobber   gun + mortar, keeps 70-200 m and lobs low-arc shells with lead.
- looter   takes anything, farms crates/cubes/NPCs, avoids mechs unless attacked (coward).

Speaks mechfight/1 directly (newest-observation rule, tick echo); no kit import needed.

Engagement knobs (below) are calibrated against the one real recording (tools/calibrate.py, MINISIM.md
"Calibration"). Profile v2 (default) = calibrated; v1 = the original fire-on-sight bots. Select with
MINISIM_BOTS_PROFILE=v1|v2 (minisim spec `scripted:<arch>_v1` sets it); override any knob with
MINISIM_BOTS_<NAME>=value, e.g. MINISIM_BOTS_PAUSE_S=0.
"""
import json
import math
import os
import queue
import random
import sys
import threading

# ------------------------------------------------------------------------------------- calibration knobs
# Grounded in the real recording (slot 3 = kit example bot; built-in id 10, gun+laser, engaged it). The
# v2 values were tuned with tools/calibrate.py (MINISIM.md "Calibration"); one recording, so treat them
# as a plausible fit, not a measurement.
#   PEACE_S, PEACE_JITTER_S  early looting window: until PEACE_S + U(0, JITTER) a bot shoots at nothing
#                            (mechs, crates, NPCs) and ignores mechs, unless a mech damages it. Real: the
#                            first gun heard (320 m carry) was at 3.8 s; v1 bots fire from t = 0.2 s.
#   REACT_S, REACT_JITTER_S  delay from a new contact (mech not tracked for CONTACT_GAP_S) to the first
#                            trigger pull at it: REACT_S + U(0, JITTER); torso turning comes on top. Real:
#                            aim on target -> laser charge start 0.9 s (6.2 -> 7.1 s) and 0.4 s (18.1 -> 18.5 s).
#   ENGAGE_MAX_M             open fire on a mech only within this range, unless it hit us in the last 4 s.
#                            Real: every shot at the observer came from 151 m or closer (55-151 m).
#   GUN_INTERVAL_S, GUN_INTERVAL_JITTER_S  gun trigger cadence for any target: next shot no sooner than
#                            INTERVAL + U(-JITTER, +JITTER) after the last (weapon cooldown 0.24 s, decisions
#                            every 0.1 s, so 0.3/0.4/0.5 s). Real: id 10's gun sounds came 0.3-0.5 s apart
#                            (mean 0.36 s).
#   GUN_WHILE_CHARGING       1 = keep firing the gun while the laser charges (v1). Real: no gun sound from
#                            id 10 during any of its 4 observed charges; it opened with a full laser charge.
#   BURST_S, BURST_JITTER_S  after BURST_S + U(0, JITTER) of shooting at mechs, stop for PAUSE_S +
#   PAUSE_S, PAUSE_JITTER_S  U(0, JITTER) (keep moving and aiming, finish a laser charge, no new shots).
#                            Real: id 10 fired 7.1-14.1, 18.5-19.8, 24.2-24.4 and from ~35 s; the pauses
#                            (4.4, 4.4, ~10.7 s) were with the target in view and in range.
#   CONTACT_GAP_S            a mech absent from `mechs` this long is a new contact (react again).
PROFILES = {
    "v1": dict(PEACE_S=0.0, PEACE_JITTER_S=0.0, REACT_S=0.0, REACT_JITTER_S=0.0, ENGAGE_MAX_M=1e9,
               GUN_INTERVAL_S=0.0, GUN_INTERVAL_JITTER_S=0.0, GUN_WHILE_CHARGING=1, BURST_S=1e9, BURST_JITTER_S=0.0,
               PAUSE_S=0.0, PAUSE_JITTER_S=0.0, CONTACT_GAP_S=1.0),
    "v2": dict(PEACE_S=3.0, PEACE_JITTER_S=2.0, REACT_S=0.5, REACT_JITTER_S=0.6, ENGAGE_MAX_M=160.0,
               GUN_INTERVAL_S=0.32, GUN_INTERVAL_JITTER_S=0.1, GUN_WHILE_CHARGING=0, BURST_S=1.0, BURST_JITTER_S=6.0,
               PAUSE_S=4.0, PAUSE_JITTER_S=7.0, CONTACT_GAP_S=1.0),
}
PROFILE = os.environ.get("MINISIM_BOTS_PROFILE", "v2").strip().lower() or "v2"
if PROFILE not in PROFILES:
    PROFILE = "v2"


def _knob(name):
    v = os.environ.get("MINISIM_BOTS_" + name, "").strip()
    return float(v) if v else float(PROFILES[PROFILE][name])


PEACE_S = _knob("PEACE_S")
PEACE_JITTER_S = _knob("PEACE_JITTER_S")
REACT_S = _knob("REACT_S")
REACT_JITTER_S = _knob("REACT_JITTER_S")
ENGAGE_MAX_M = _knob("ENGAGE_MAX_M")
GUN_INTERVAL_S = _knob("GUN_INTERVAL_S")
GUN_INTERVAL_JITTER_S = _knob("GUN_INTERVAL_JITTER_S")
GUN_WHILE_CHARGING = _knob("GUN_WHILE_CHARGING") >= 0.5
BURST_S = _knob("BURST_S")
BURST_JITTER_S = _knob("BURST_JITTER_S")
PAUSE_S = _knob("PAUSE_S")
PAUSE_JITTER_S = _knob("PAUSE_JITTER_S")
CONTACT_GAP_S = _knob("CONTACT_GAP_S")
HURT_MEMORY_S = 4.0

PREF = {"duelist": ["laser", "lightning", "mortar"], "rusher": ["lightning", "laser", "mortar"],
        "lobber": ["mortar", "laser", "lightning"], "looter": ["laser", "lightning", "mortar"]}
UPG = {"duelist": ["laser_damage", "laser_charge", "gun_damage", "armor", "plating", "gun_rate"],
       "rusher": ["lightning_efficiency", "lightning_range", "servos", "armor", "plating", "capacitor"],
       "lobber": ["mortar_reload", "mortar_splash", "armor", "plating", "gun_damage"],
       "looter": ["plating", "armor", "nanites", "servos", "sensors"]}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ScriptBot:
    def __init__(self, hello, arch):
        self.arch = arch
        self.cfg = hello["config"]
        self.me_id = hello["slot"]
        self.rng = random.Random(hello.get("seed", 0))
        self.name = f"{arch}-{hello['slot']}"
        self.strafe = 1
        self.strafe_t = 0.0
        self.stuck_t = 0.0
        self.detour_until = -1.0
        self.detour = (0.0, 0.0)
        self.jet_until = -1.0
        self.scan_phase = self.rng.random() * 6.28
        self.hurt_bearing = None
        self.hurt_t = -99.0
        self.lp = None
        self.wander = None
        # calibrated engagement state (knobs above); own RNG so profile v1 replays the original bots exactly
        self.krng = random.Random(hello.get("seed", 0) * 31 + 7)
        self.peace_end = PEACE_S + self.krng.random() * PEACE_JITTER_S
        self.seen = {}            # mech id -> last time it was in `mechs`
        self.react_until = {}     # mech id -> no trigger pull at it before this time
        self.hurt_by = {}         # mech id -> last time it damaged us
        self.hurt_mech_t = -99.0
        self.gun_next = -1.0
        self.burst_start = None
        self.burst_len = 0.0
        self.pause_until = -1.0
        self.last_fire_t = -99.0

    # -------------------------------------------------------------- helpers
    def zone_live(self, z, t):
        c, r = z["center"], z["radius"]
        if t <= z["shrink_start"] or z["shrink_end"] <= z["shrink_start"]:
            return c[0], c[1], r
        f = min(1.0, (t - z["shrink_start"]) / (z["shrink_end"] - z["shrink_start"]))
        nc, nr = z["next_center"], z["next_radius"]
        return c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f, r + (nr - r) * f

    def lead(self, me, tgt, speed, mh):
        mx, my, mz = me["pos"][0], me["pos"][1], me["pos"][2] + mh
        px, py, pz = tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + 4.5
        vx, vy, vz = tgt["vel"]
        tt = math.dist((mx, my, mz), (px, py, pz)) / speed
        for _ in range(3):
            qx, qy, qz = px + vx * tt, py + vy * tt, pz + max(-3.0, min(3.0, vz * tt))
            tt = math.dist((mx, my, mz), (qx, qy, qz)) / speed
        return math.atan2(qy - my, qx - mx), math.atan2(qz - mz, math.hypot(qx - mx, qy - my))

    def mortar_aim(self, me, tgt, ws):
        v, g = ws["speed"], ws["gravity"]
        mx, my = me["pos"][0], me["pos"][1]
        mz = me["pos"][2] + me["stats"].get("height", 9) + 0.5
        px, py, pz = tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + 1.0
        vx, vy = tgt["vel"][0], tgt["vel"][1]
        th = 0.6
        for _ in range(3):
            d = math.hypot(px - mx, py - my)
            h = pz - mz
            disc = v ** 4 - g * (g * d * d + 2 * h * v * v)
            th = math.pi / 4 if disc < 0 else math.atan((v * v - math.sqrt(disc)) / (g * max(d, 1.0)))
            tt = d / max(1.0, v * math.cos(th))
            px, py = tgt["pos"][0] + vx * tt, tgt["pos"][1] + vy * tt
        return math.atan2(py - my, px - mx), th

    def slot_of(self, me, kind):
        for i, w in enumerate(me["weapons"]):
            if w is not None and w["kind"] == kind:
                return i
        return None

    def gate_fire(self, t, me, tgt, dtg, fire, ls):
        """Apply the calibrated engagement rules to `fire` (shots at mech `tgt`), in place."""
        charging = ls is not None and me["weapons"][ls]["charge"] > 0
        if t - self.last_fire_t > 2.0:
            self.burst_start = None  # target lost or quiet: the next shot starts a new burst
        hold = (t < self.pause_until or t < self.react_until.get(tgt["id"], -1.0)
                or (dtg > ENGAGE_MAX_M and t - self.hurt_by.get(tgt["id"], -99.0) > HURT_MEMORY_S))
        for i in range(len(fire)):
            if fire[i] and hold and not (i == ls and charging):
                fire[i] = False  # no new shots; a laser charge in progress still completes
        if any(fire):
            self.last_fire_t = t
            if not hold:
                if self.burst_start is None:
                    self.burst_start = t
                    self.burst_len = BURST_S + self.krng.random() * BURST_JITTER_S
                elif t - self.burst_start > self.burst_len:
                    self.pause_until = t + PAUSE_S + self.krng.random() * PAUSE_JITTER_S
                    self.burst_start = None

    def gun_cadence(self, t, me, fire, gun_s):
        """Gun trigger cadence (any target): GUN_INTERVAL_S, and no gun while the laser charges."""
        if gun_s is None or not fire[gun_s]:
            return
        ls = self.slot_of(me, "laser")
        g = me["weapons"][gun_s]
        if (ls is not None and me["weapons"][ls]["charge"] > 0 and not GUN_WHILE_CHARGING) or t + 1e-6 < self.gun_next:
            fire[gun_s] = False
        elif g["cooldown"] <= 0 and g.get("reload", 0) == 0 and g.get("ammo", 1) > 0:
            self.gun_next = t + GUN_INTERVAL_S + (2 * self.krng.random() - 1) * GUN_INTERVAL_JITTER_S

    # -------------------------------------------------------------- policy
    def act(self, obs):
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        act = {"tick": obs["tick"]}
        pref = PREF[self.arch]
        held = [w["kind"] for w in me["weapons"] if w is not None]
        mh = me["stats"].get("muzzle_height", 7)
        if obs.get("offer"):
            ch = [c["id"] for c in obs["offer"]["choices"]]
            best = next((i for u in UPG[self.arch] for i, c in enumerate(ch) if c == u), None)
            if best is not None:
                act["pick"] = best
            elif self.rng.random() < 0.5:
                act["pick"] = self.rng.randrange(len(ch))
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("by") is not None:
                self.hurt_bearing, self.hurt_t = e["bearing"], t
                if e["by"] < 100:
                    self.hurt_by[e["by"]] = t
                    self.hurt_mech_t = t
        zx, zy, zr = self.zone_live(obs["zone"], t)
        nzx, nzy = obs["zone"]["next_center"]
        dz_live = math.hypot(x - zx, y - zy)
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        peaceful = t < self.peace_end and t - self.hurt_mech_t > HURT_MEMORY_S
        if peaceful:
            enemies = []  # early looting window: ignore mechs until attacked
        for m in enemies:
            if t - self.seen.get(m["id"], -99.0) > CONTACT_GAP_S:
                self.react_until[m["id"]] = t + REACT_S + self.krng.random() * REACT_JITTER_S
            self.seen[m["id"]] = t
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        tgt = min(enemies, key=lambda m: math.dist(m["pos"], me["pos"])) if enemies else None
        dtg = math.dist(tgt["pos"], me["pos"]) if tgt else 1e9
        # ---------------- legs
        goal = None
        boost = False
        zone_urgent = dz_live > zr - 15 or (t > obs["zone"]["shrink_start"] - 10 and
                                             math.hypot(x - nzx, y - nzy) > obs["zone"]["next_radius"] - 20)
        if zone_urgent:
            goal = (nzx, nzy)
            boost = not me["in_zone"] or dz_live > zr - 5
        want_kind = None
        if len(held) < 2 or (pref[0] not in held and t < 200):
            for k in pref:
                if k not in held:
                    want_kind = k
                    break
        loot_tgt = None
        if not zone_urgent:
            cands = [p for p in obs["pickups"] if p["kind"] in pref and p["kind"] not in held]
            if len(held) >= 2:
                cands = [p for p in cands if p["kind"] == pref[0]]
            if cands:
                loot_tgt = min(cands, key=lambda p: (pref.index(p["kind"]), math.dist(p["pos"][:2], (x, y))))
                if math.dist(loot_tgt["pos"][:2], (x, y)) > 120:
                    loot_tgt = None
        if loot_tgt is not None and (tgt is None or dtg > 90 or self.arch == "looter"):
            goal = tuple(loot_tgt["pos"][:2])
            if math.dist(loot_tgt["pos"][:2], (x, y)) < self.cfg["loot"]["pickup_radius"] - 0.3:
                act["take"] = loot_tgt["id"]
                if len(held) >= 2:
                    worst = max(range(2), key=lambda i: (pref.index(me["weapons"][i]["kind"])
                                                         if me["weapons"][i]["kind"] in pref else (-1 if me["weapons"][i]["kind"] == "gun" else 9)))
                    if me["weapons"][worst]["kind"] != "gun":
                        act["drop"] = worst
                    else:
                        act.pop("take")
        cube = None
        if obs["cubes"] and goal is None:
            cube = min(obs["cubes"], key=lambda c: math.dist(c["pos"][:2], (x, y)))
            if math.dist(cube["pos"][:2], (x, y)) < 70 and (tgt is None or dtg > 80):
                goal = tuple(cube["pos"][:2])
        for d in obs["drops"]:
            if goal is None and d["lands_in"] < 6 and math.hypot(d["pos"][0] - x, d["pos"][1] - y) < 140 and \
                    self.arch in ("duelist", "rusher", "looter") and (tgt is None or dtg > 60):
                goal = (d["pos"][0], d["pos"][1])
        mv = (0.0, 0.0)
        if tgt is not None and not zone_urgent and (loot_tgt is None or dtg <= 90):
            bx, by = (tgt["pos"][0] - x) / max(dtg, 1), (tgt["pos"][1] - y) / max(dtg, 1)
            px, py = -by * self.strafe, bx * self.strafe
            if t > self.strafe_t:
                self.strafe = -self.strafe
                self.strafe_t = t + 1.5 + self.rng.random() * 2.0
            if self.arch == "duelist":
                lo, hi = 93, 150
            elif self.arch == "rusher":
                lo, hi = 12, 28
            elif self.arch == "lobber":
                lo, hi = 70, 190
            else:
                lo, hi = 160, 400
            if self.arch == "looter" and t - self.hurt_t > 4 and me["hp"] > 0.5 * me["max_hp"] and dtg > 60:
                pass
            if dtg > hi:
                mv = (bx * 0.85 + px * 0.3, by * 0.85 + py * 0.3)
                boost = self.arch == "rusher" and me["energy"] > 25
            elif dtg < lo:
                mv = (-bx * 0.8 + px * 0.5, -by * 0.8 + py * 0.5)
                boost = self.arch == "looter" and me["energy"] > 20
            else:
                mv = (px, py)
        elif goal is not None:
            gx, gy = goal[0] - x, goal[1] - y
            gl = math.hypot(gx, gy)
            if gl > 1.0:
                mv = (gx / gl, gy / gl)
        else:
            if self.wander is None or math.hypot(self.wander[0] - x, self.wander[1] - y) < 10 or \
                    math.hypot(self.wander[0] - zx, self.wander[1] - zy) > zr * 0.6:
                a = self.rng.random() * 6.283
                rr = zr * 0.5 * math.sqrt(self.rng.random())
                self.wander = (zx + rr * math.cos(a), zy + rr * math.sin(a))
            gx, gy = self.wander[0] - x, self.wander[1] - y
            gl = math.hypot(gx, gy) or 1
            mv = (gx / gl * 0.8, gy / gl * 0.8)
        # NPC avoidance (except looters with HP)
        for n in npcs:
            dn = math.dist(n["pos"][:2], (x, y))
            if dn < 55 and not (self.arch == "looter" and me["hp"] > 0.6 * me["max_hp"]):
                ax, ay = (x - n["pos"][0]) / max(dn, 1), (y - n["pos"][1]) / max(dn, 1)
                mv = (mv[0] + ax * 0.7, mv[1] + ay * 0.7)
        # stuck: commanded but not moving -> detour + jump + jet
        sp = math.hypot(me["vel"][0], me["vel"][1])
        if math.hypot(*mv) > 0.3 and sp < 2.0 and me["grounded"]:
            self.stuck_t += 0.1
        else:
            self.stuck_t = max(0.0, self.stuck_t - 0.05)
        jump = False
        if self.stuck_t > 0.6:
            self.stuck_t = 0.0
            a = math.atan2(mv[1], mv[0]) + self.rng.choice((-1.57, 1.57, 2.4, -2.4))
            self.detour = (math.cos(a), math.sin(a))
            self.detour_until = t + 1.2 + self.rng.random()
            jump = True
            self.jet_until = t + 1.0
        if t < self.detour_until:
            mv = (self.detour[0] * 0.7 + mv[0] * 0.3, self.detour[1] * 0.7 + mv[1] * 0.3)
        if t < self.jet_until and not me["grounded"] and me["energy"] > 15:
            boost = True
        ml = math.hypot(*mv)
        if ml > 1:
            mv = (mv[0] / ml, mv[1] / ml)
        act["move"] = [round(mv[0], 3), round(mv[1], 3)]
        act["boost"] = bool(boost and me["energy"] > 5)
        act["jump"] = jump
        # ---------------- aim and fire
        fire = [False, False]
        gun_s = self.slot_of(me, "gun")
        aim = None
        if tgt is not None and (self.arch != "looter" or dtg < 120 or t - self.hurt_t < 4):
            gs = self.cfg["weapons"]["gun"]["speed"]
            if gun_s is not None:
                gs = me["weapons"][gun_s]["stats"]["speed"]
            aim = self.lead(me, tgt, gs, mh)
            err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
            if gun_s is not None and dtg < me["weapons"][gun_s]["stats"]["range"] and err < 0.05:
                fire[gun_s] = True
            ls = self.slot_of(me, "laser")
            if ls is not None:
                w = me["weapons"][ls]
                if w["charge"] > 0:
                    # hitscan: aim at the target itself (small lead for our own torso slew)
                    aim = self.lead(me, tgt, 3000.0, mh)
                    err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
                    full = w["stats"]["charge"]
                    fire[ls] = not (w["charge"] >= full - 0.06 and err < 2.0 / max(dtg, 20.0))
                    if gun_s is not None:
                        fire[gun_s] = fire[gun_s] and err < 0.05
                elif w["cooldown"] <= 0 and dtg < w["stats"]["range"] and err < 0.25:
                    fire[ls] = True
            zs = self.slot_of(me, "lightning")
            if zs is not None:
                w = me["weapons"][zs]
                if dtg < w["stats"]["range"] * 0.95 and abs(wrap(aim[0] - me["aim"][0])) < 0.3 and me["energy"] > w["stats"]["energy"]:
                    fire[zs] = True
            ms = self.slot_of(me, "mortar")
            if ms is not None:
                w = me["weapons"][ms]
                if 25 < dtg < 230:
                    my_, mp = self.mortar_aim(me, tgt, w["stats"])
                    if dtg > 60 or gun_s is None:
                        aim = (my_, min(mp, me["stats"].get("max_pitch", 1.3)))
                        if w["cooldown"] <= 0 and abs(wrap(my_ - me["aim"][0])) < 0.04 and abs(mp - me["aim"][1]) < 0.04:
                            fire[ms] = True
                        if gun_s is not None:
                            fire[gun_s] = False
            self.gate_fire(t, me, tgt, dtg, fire, ls)
        elif peaceful:
            pass  # early looting window: no shooting at all (also no crates or NPCs)
        elif npcs and self.arch == "looter" and me["hp"] > 0.6 * me["max_hp"]:
            n = min(npcs, key=lambda m: math.dist(m["pos"], me["pos"]))
            if math.dist(n["pos"], me["pos"]) < 120 and gun_s is not None:
                aim = self.lead(me, dict(n, pos=[n["pos"][0], n["pos"][1], n["pos"][2] - 2.75]), 252, mh)
                if abs(wrap(aim[0] - me["aim"][0])) < 0.05:
                    fire[gun_s] = True
        elif obs["crates"] and gun_s is not None:
            c = min(obs["crates"], key=lambda c: math.dist(c["pos"], me["pos"]))
            dc = math.dist(c["pos"], me["pos"])
            if dc < 70:
                aim = (math.atan2(c["pos"][1] - y, c["pos"][0] - x),
                       math.atan2(c["pos"][2] + 1.2 - (z + mh), math.hypot(c["pos"][0] - x, c["pos"][1] - y)))
                if abs(wrap(aim[0] - me["aim"][0])) < 0.04:
                    fire[gun_s] = True
                zs = self.slot_of(me, "lightning")
                if zs is not None and dc < 40:
                    fire[zs] = True
        if aim is None:
            if self.hurt_bearing is not None and t - self.hurt_t < 3:
                aim = (self.hurt_bearing, 0.0)
            else:
                self.scan_phase += 0.12
                base = math.atan2(mv[1], mv[0]) if math.hypot(*mv) > 0.2 else me["aim"][0]
                aim = (base + 1.0 * math.sin(self.scan_phase), 0.0)
        self.gun_cadence(t, me, fire, gun_s)
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire
        if gun_s is not None and tgt is None:
            g = me["weapons"][gun_s]
            if g.get("ammo", 99) < g["stats"]["mag"] * 0.4 and g.get("reload", 0) == 0:
                act["reload"] = True
        return act


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "duelist"
    if arch not in PREF:
        arch = "duelist"
    q = queue.Queue()

    def reader():
        for line in sys.stdin:
            if line.strip():
                q.put(line)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    bot = None
    while True:
        batch = [q.get()]
        while True:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        latest = None
        for raw in batch:
            if raw is None:
                return
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            k = msg.get("type")
            if k == "hello":
                bot = ScriptBot(msg, arch)
                sys.stdout.write(json.dumps({"type": "ready", "name": bot.name}) + "\n")
                sys.stdout.flush()
            elif k == "obs":
                latest = msg
            elif k == "end":
                return
        if latest is None or bot is None:
            continue
        try:
            act = bot.act(latest)
        except Exception as e:  # keep playing; log for debugging
            sys.stderr.write(f"scripted {arch} error at tick {latest.get('tick')}: {e!r}\n")
            act = {"tick": latest["tick"]}
        act["tick"] = latest["tick"]
        sys.stdout.write(json.dumps(act, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
