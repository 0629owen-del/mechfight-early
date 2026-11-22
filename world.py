"""World model (plan section 4): self, tracks, ghosts, item memory, projectiles, zone, seen cells.

    world = WorldModel(C, mapdata, slot)
    tick = world.update(obs, clock)        # 6.1 steps 1-2, every observation (also during X5 SAFE)

Everything other modules read is a plain attribute (CONTRACTS.md 6.2). Extra read-only state for
telemetry and the decision layer (not in the contract, all plain attributes):
    world.tx[track_id]          _TrackExtra: sighted, vis_since, los_since, los_geo, fight_*, ...
    world.heat                  {5 m cell: [heat, t]} danger heat (4.3 step 7)
    world.seen                  bytearray, 5 m seen-cells bitmap (4.4); world.seen_count
    world.explosions            [(Vec2, t)] heard explosion points (last GHOST_TTL_S)
    world.last_launch_t         last near/mid mortar launch sound
    world.phases                [(hold_start, shrink_start, shrink_end, radius, dps)] zone timeline
    world.phase_idx             index into phases at the current t
    world.last_swap_t           t of our last `took` (H5 lock)
    world.ghost_lightning_mult  TESLA_MUL ** TESLA_MAX_STACKS (A-20, ghosts / unknown tracks)
    world.take_failed           take_failed events of this tick
    world.update_ms, world.los_calls, world.exceptions, world.last_error
Queries beyond the contract: weapon_probs(id), possible_kinds(id), zone_phase(t), take_exceptions().

Two kinds of ray (A-13 / A-14, protocol 7 and 8):
- SHOT line, _los() = mapdata.los (buildings, stadium, bridges, terrain; trees ignored). A mech
  listed in obs.mechs has its real, current position and trees do not stop shots, so Track.visible,
  tx.los_geo / los_since / vis_since (fire permission, aiming, "can it hit us") stay on this line.
  Rays are cached per track keyed by (our 1 m cell, its 1 m cell, z); when the LOS_MS_CAP slice
  runs out the previous result is reused (geometry is static, so a cache hit is exact).
- SIGHT line, _sight() = mapdata.visible (the shot line + tree trunks, + dense canopy beyond
  vision.near_range): "would the server list something at this spot for us right now". Used where
  NOT being listed is taken as evidence: a remembered item / NPC whose spot we look at is gone, a
  ghost whose spot we look at is empty, a 5 m cell has been looked at (seen cells).
"""
from __future__ import annotations

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

from contracts import (BulletThreat, CrateMem, CubeObs, DamageRec, DropState, Ghost, HitRec, OfferState,
                       PickupMem, SelfState, ShellPred, SoundRec, TickInfo, Track, Vec2, Vec3,
                       WeaponState, ZoneState)

TWO_PI = 2.0 * math.pi
WEAPONS = ("gun", "laser", "lightning", "mortar")
LOG_KEEP_S = 30.0                    # damage_log / hits horizon (contract: "last 30 s")
SEEN_STEP = 5.0                      # 4.4 seen-cells and 4.3 heat grid (5 m sight cells)
SEEN_LOS_PER_TICK = 20               # 4.4 "sampled, 20 cells per tick"
SEEN_SCAN_PER_TICK = 400             # offsets visited per tick while looking for unmarked cells
SHELL_STEP_S = 0.05                  # 4.5 integration step
ITEM_GONE_GRACE_S = 0.5              # remembered item unlisted this long while in near range -> gone

# 4.3 step 3: sound kind -> inferred weapon
SOUND_WEAPON = {"gun": "gun", "laser": "laser", "laser_charge": "laser", "lightning": "lightning",
                "mortar": "mortar", "reload": "gun"}
NPC_SOUNDS = frozenset(("npc_gun",))


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, TWO_PI)
    if a <= 0.0:
        a += TWO_PI
    return a - math.pi


def _f(v, d=0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return x if x == x and x not in (math.inf, -math.inf) else d


def _v3(v, d=(0.0, 0.0, 0.0)) -> Vec3:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (_f(v[0]), _f(v[1]), _f(v[2]) if len(v) > 2 else 0.0)
    return d


def _v2(v, d=(0.0, 0.0)) -> Vec2:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (_f(v[0]), _f(v[1]))
    return d


class _TrackExtra:
    """Per-track state that does not fit the slotted contract Track."""
    __slots__ = ("sighted", "last_evidence_t", "last_spend_t", "prev_vz", "prev_t", "fight_id", "fight_t",
                 "obs_mult", "vis_since", "los_since", "los_geo", "los_key", "last_jet_t", "last_move_snd_t",
                 "last_hit_by_us_t", "in_view")

    def __init__(self):
        self.sighted = False          # weapons list seen in obs.mechs at least once
        self.last_evidence_t = -1e9   # last obs listing, sound, beam or damage
        self.last_spend_t = -1e9      # last observed energy spend (energy_est regen delay)
        self.prev_vz = 0.0
        self.prev_t = None
        self.fight_id = None          # fighting_with candidate and when last supported
        self.fight_t = -1e9
        self.obs_mult = 1.0           # observed lightning range multiplier (A-20)
        self.vis_since = None         # t when it became visible (None = not visible)
        self.los_since = None         # t since geometric LOS eye<->center held continuously
        self.los_geo = False          # geometric LOS this tick (ignores FOV)
        self.los_key = None
        self.last_jet_t = -1e9
        self.last_move_snd_t = -1e9
        self.last_hit_by_us_t = -1e9
        self.in_view = False          # inside our vision wedge/near range (geometry only)


class WorldModel:
    def __init__(self, C, mapdata, slot: int) -> None:
        self.C = C
        self.map = mapdata
        self.slot = int(slot)
        p = C.path
        # static rule numbers (hello.config); live stats come from obs.self every tick
        self.mech_radius = _f(p("mech.radius", 2.5), 2.5)
        self.mech_height = _f(p("mech.height", 9), 9.0)
        self.walk_speed = _f(p("mech.walk_speed", 11.7), 11.7)
        self.boost_drain = _f(p("mech.boost_drain", 17), 17.0)
        self.jet_drain = _f(p("mech.jet_drain", 15), 15.0)
        self.mech_energy = _f(p("mech.energy", 100), 100.0)
        self.energy_regen = _f(p("mech.energy_regen", 18), 18.0)
        self.energy_regen_delay = _f(p("mech.energy_regen_delay", 0.6), 0.6)
        self.gravity = _f(p("mech.gravity", 22), 22.0)
        self.regen_delay = _f(p("mech.regen_delay", 5), 5.0)
        self.vis_range = _f(p("vision.range", 200), 200.0)
        self.vis_near = _f(p("vision.near_range", 45), 45.0)
        self.vis_half_fov = math.radians(_f(p("vision.fov_deg", 120), 120.0)) / 2.0
        self.vis_canopy = bool(p("vision.canopy_blocks", True))
        self.snd_near = _f(p("sound.near", 60), 60.0)
        self.snd_mid = _f(p("sound.mid", 160), 160.0)
        self.snd_noise_deg = _f(p("sound.bearing_noise_deg", 7), 7.0)
        self.snd_range = p("sound.range", {}) or {}
        self.size_per_cube = _f(p("cubes.size_per_cube", 0.05), 0.05)
        self.thresholds = [t for t in (p("cubes.upgrade_thresholds", []) or []) if isinstance(t, (int, float))]
        self.npc_height = _f(p("npc.height", 3.5), 3.5)
        self.npc_aggro = _f(p("npc.aggro_range", 45), 45.0)
        self.equip_s = _f(p("loot.equip_s", 1), 1.0)
        self.short_s = _f(p("weapons.lightning.short_s", 2), 2.0)
        self.lt_range = _f(p("weapons.lightning.range", 45), 45.0)
        self.lt_energy = _f(p("weapons.lightning.energy", 3), 3.0)
        self.mortar_g = _f(p("weapons.mortar.gravity", 22), 22.0)
        self.mortar_ttl = _f(p("weapons.mortar.ttl", 7), 7.0)
        self.mortar_splash = _f(p("weapons.mortar.splash_radius", 7), 7.0)
        self.gun_reload = _f(p("weapons.gun.reload", 2.4), 2.4)
        self.decide_every = max(1, int(_f(p("match.decide_every", 2), 2)))
        self.wcfg = p("weapons", {}) or {}
        self.ghost_lightning_mult = float(C.TESLA_MUL) ** int(C.TESLA_MAX_STACKS)
        # zone timeline (4.7)
        self.phases: List[Tuple[float, float, float, float, float]] = []
        tt = 0.0
        for ph in (p("zone.phases", []) or []):
            try:
                hs, ss = float(ph["hold_s"]), float(ph["shrink_s"])
            except (KeyError, TypeError, ValueError):
                continue
            self.phases.append((tt, tt + hs, tt + hs + ss, _f(ph.get("radius")), _f(ph.get("dps"))))
            tt += hs + ss
        self.start_radius = _f(p("zone.start_radius", 0.0))
        self.phase_idx = 0
        # contract state
        self.tick = TickInfo()
        self.me = SelfState(id=self.slot)
        self.tracks: Dict[int, Track] = {}
        self.tx: Dict[int, _TrackExtra] = {}
        self.ghosts: Dict[int, Ghost] = {}
        self.sounds: List[SoundRec] = []
        self.beams: List[dict] = []
        self.events: List[dict] = []
        self.pickups: Dict[int, PickupMem] = {}
        self.crates: Dict[int, CrateMem] = {}
        self.cubes: Dict[int, CubeObs] = {}
        self.drops: Dict[int, DropState] = {}
        self.zone = ZoneState()
        self.offer: Optional[OfferState] = None
        self.damage_log: collections.deque = collections.deque()
        self.hits: collections.deque = collections.deque()
        self.shells: List[ShellPred] = []
        self.bullets: List[BulletThreat] = []
        self.npc_bullets: List[BulletThreat] = []
        self.mortar_active_until = -1e9
        self.kills = 0
        self.loot_bursts: List[Tuple[Vec2, float]] = []
        # extras
        self.heat: Dict[int, List[float]] = {}
        self.explosions: List[Tuple[Vec2, float]] = []
        self.last_launch_t = -1e9
        self.last_swap_t = -1e9
        self.take_failed: List[dict] = []
        self.exceptions = 0
        self.last_error = ""
        self.update_ms = 0.0
        self.los_calls = 0
        self._next_ghost = -1
        self._last_tick: Optional[int] = None
        self._prev_next: Optional[Tuple[float, float, float]] = None
        self._zone_key = None
        self._shell_cache: Dict[int, Tuple[Vec3, float]] = {}
        self._seen_ptr = 0
        # 5 m grid (seen cells + heat), aligned with mapdata's 5 m grids when present
        b = getattr(mapdata, "bounds", None)
        if isinstance(b, dict):
            bx0, by0, bx1, by1 = b.get("minX", -513), b.get("minY", -334), b.get("maxX", 556), b.get("maxY", 389)
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            bx0, by0, bx1, by1 = b
        else:
            bx0, by0, bx1, by1 = -513.0, -334.0, 556.0, 389.0
        self.g0x = float(getattr(mapdata, "gx0", bx0 - 0.5))
        self.g0y = float(getattr(mapdata, "gy0", by0 - 0.5))
        self.gnx = int((bx1 - self.g0x) // SEEN_STEP) + 1
        self.gny = int((by1 - self.g0y) // SEEN_STEP) + 1
        self.seen = bytearray(self.gnx * self.gny)
        self.seen_count = 0
        R = int(math.ceil(max(C.SEEN_FOV_R, self.vis_near) / SEEN_STEP))
        offs = []
        for dj in range(-R, R + 1):
            for di in range(-R, R + 1):
                d = math.hypot(di, dj) * SEEN_STEP
                if 0 < d <= max(C.SEEN_FOV_R, self.vis_near):
                    offs.append((di, dj, d, math.atan2(dj, di)))
        # interleave distances so a partial scan still covers near and far cells
        offs.sort(key=lambda o: (o[2] // (SEEN_STEP * 4), o[3]))
        self._seen_offs = offs

    # ================================================================== public queries
    def held_kinds(self) -> Tuple[str, ...]:
        return tuple(w.kind for w in self.me.weapons if w is not None)

    def visible_mechs(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.kind == "mech" and tr.visible]

    def hostile_tracks(self, max_age_s: float, include_npc: bool = False) -> List[Track]:
        t = self.tick.t
        out = []
        for tr in self.tracks.values():
            if tr.kind != "mech" and not include_npc:
                continue
            if t - tr.last_seen_t <= max_age_s:
                out.append(tr)
        return out

    def damage_since(self, t: float, by: Optional[int] = None) -> float:
        s = 0.0
        for r in reversed(self.damage_log):
            if r.t < t:
                break
            if by is None or r.by == by:
                s += r.amount
        return s

    def damaged_us_recently(self, track_id: int, window_s: float) -> bool:
        t0 = self.tick.t - window_s
        for r in reversed(self.damage_log):
            if r.t < t0:
                return False
            if r.by == track_id:
                return True
        return False

    def heat_at(self, x: float, y: float) -> float:
        c = self._gcell(x, y)
        h = self.heat.get(c) if c is not None else None
        if not h:
            return 0.0
        return h[0] * 0.5 ** ((self.tick.t - h[1]) / max(1e-6, self.C.DANGER_HEAT_DECAY_S))

    def seen_at(self, x: float, y: float) -> bool:
        c = self._gcell(x, y)
        return bool(c is not None and self.seen[c])

    def weapon_probs(self, ident: int) -> Dict[str, float]:
        """4.3 step 5: P(kind held) for a track (id >= 0) or ghost (id < 0). Sighted weapons are
        authoritative; otherwise inferred confidences, the spawn gun, and PRIOR_SECOND for slot 2."""
        C = self.C
        if ident < 0:
            g = self.ghosts.get(ident)
            known = dict(g.kinds) if g else {}
            sighted = None
        else:
            tr = self.tracks.get(ident)
            if tr is None:
                return {}
            if tr.kind != "mech":
                return {}
            known = dict(tr.known_weapons)
            sighted = tr.weapons if self.tx.get(ident) and self.tx[ident].sighted else None
        if sighted is not None:
            return {k: (1.0 if k in sighted else 0.0) for k in WEAPONS}
        out = {"gun": max(known.get("gun", 0.0), 1.0)}   # protocol: every mech spawns with the gun
        second = [k for k in ("laser", "lightning", "mortar") if known.get(k, 0.0) >= C.CONF_SOUND]
        for k in ("laser", "lightning", "mortar"):
            if second and k not in second:
                out[k] = known.get(k, 0.0)
            else:
                out[k] = max(known.get(k, 0.0), float(C.PRIOR_SECOND.get(k, 0.0)))
        return out

    def possible_kinds(self, ident: int, min_p: Optional[float] = None) -> List[str]:
        mp = self.C.PRIOR_MIN_P if min_p is None else min_p
        return [k for k, p in self.weapon_probs(ident).items() if p >= mp]

    def zone_phase(self, t: float) -> int:
        for i, ph in enumerate(self.phases):
            if t < ph[2]:
                return i
        return max(0, len(self.phases) - 1)

    def take_exceptions(self) -> int:
        n, self.exceptions = self.exceptions, 0
        return n

    # ================================================================== update
    def update(self, obs: dict, clock=None) -> TickInfo:
        t_start = time.perf_counter()
        self.los_calls = 0
        sl = clock.slice(self.C.LOS_MS_CAP) if clock is not None else None
        if not isinstance(obs, dict):
            self.exceptions += 1
            self.last_error = "obs not a dict"
            return self.tick
        self._guard(self._tick_info, obs)
        self._guard(self._self_state, obs.get("self"))
        self._guard(self._zone_update, obs.get("zone"))
        self._guard(self._tracks_update, obs.get("mechs"), sl)
        self._guard(self._beams_update, obs.get("beams"))
        self._guard(self._events_update, obs.get("events"))
        self._guard(self._sounds_update, obs.get("sounds"))
        self._guard(self._projectiles_update, obs.get("projectiles"))
        self._guard(self._items_update, obs, sl)
        self._guard(self._offer_update, obs.get("offer"))
        self._guard(self._tracks_maintain, sl)
        self._guard(self._ghosts_maintain, sl)
        self._guard(self._seen_update, sl)
        self._guard(self._prune)
        self.update_ms = (time.perf_counter() - t_start) * 1000.0
        return self.tick

    def _guard(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # counted, never silent (CONTRACTS 0)
            self.exceptions += 1
            self.last_error = f"{fn.__name__}: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------ 6.1 step 1
    def _tick_info(self, obs: dict) -> None:
        tick = int(_f(obs.get("tick"), self.tick.tick))
        t = _f(obs.get("t"), self.tick.t)
        prev_t = self.tick.t
        missed = 0
        if self._last_tick is not None and tick > self._last_tick:
            missed = max(0, (tick - self._last_tick) // self.decide_every - 1)
        was_charging = missed > 0 and self.me.laser_charge > 0.0
        self.tick = TickInfo(tick=tick, t=t, dt=(t - prev_t) if self._last_tick is not None else 0.0,
                             missed_ticks=missed, was_charging_before_gap=was_charging,
                             alive=int(_f(obs.get("alive"), self.tick.alive)))
        self._last_tick = tick
        self.phase_idx = self.zone_phase(t)

    # ------------------------------------------------------------------ 4.1
    def _self_state(self, s) -> None:
        if not isinstance(s, dict):
            return
        me = self.me
        t = self.tick.t
        me.id = int(_f(s.get("id"), me.id))
        me.pos = _v3(s.get("pos"), me.pos)
        me.vel = _v3(s.get("vel"), me.vel)
        me.aim = _v2(s.get("aim"), me.aim)
        me.hp = _f(s.get("hp"), me.hp)
        me.max_hp = _f(s.get("max_hp"), me.max_hp)
        me.energy = _f(s.get("energy"), me.energy)
        me.max_energy = _f(s.get("max_energy"), me.max_energy)
        me.grounded = bool(s.get("grounded", True))
        me.cubes = int(_f(s.get("cubes"), me.cubes))
        me.upgrades = [u for u in (s.get("upgrades") or []) if isinstance(u, str)]
        me.in_zone = bool(s.get("in_zone", True))
        st = s.get("stats")
        if isinstance(st, dict):
            me.stats = st
        ws = s.get("weapons") or []
        out: List[Optional[WeaponState]] = []
        me.laser_slot = None
        me.laser_charge = 0.0
        for i in range(2):
            w = ws[i] if i < len(ws) else None
            if not isinstance(w, dict) or not w.get("kind"):
                out.append(None)
                continue
            ammo = w.get("ammo")
            ws_ = WeaponState(kind=str(w["kind"]), cooldown=_f(w.get("cooldown")),
                              ammo=_f(ammo) if ammo is not None else None, reload=_f(w.get("reload")),
                              charge=_f(w.get("charge")), stats=w.get("stats") or {})
            out.append(ws_)
            if ws_.kind == "laser":
                me.laser_slot = i
                me.laser_charge = ws_.charge
        me.weapons = out
        me.s = 1.0 + self.size_per_cube * me.cubes
        st = me.stats
        me.radius = _f(st.get("radius"), self.mech_radius * me.s)
        me.eye_z = me.pos[2] + _f(st.get("eye_height"), 8.0 * me.s)
        me.muzzle_z = me.pos[2] + _f(st.get("muzzle_height"), 7.0 * me.s)
        me.armor = _f(st.get("armor"), 0.0)
        me.speed_h = math.hypot(me.vel[0], me.vel[1])
        kinds = self.held_kinds()
        me.playbook = "PB_L" if "lightning" in kinds else ("PB_Z" if "laser" in kinds else "PB_G")
        if not me.in_zone:
            me.regen_block_until = max(me.regen_block_until, t + _f(st.get("regen_delay"), self.regen_delay))

    # ------------------------------------------------------------------ 4.7
    def _zone_update(self, z) -> None:
        if not isinstance(z, dict):
            self.zone.next_changed = False
            return
        t = self.tick.t
        zs = self.zone
        zs.center = _v2(z.get("center"), zs.center)
        zs.radius = _f(z.get("radius"), zs.radius)
        zs.next_center = _v2(z.get("next_center"), zs.next_center)
        zs.next_radius = _f(z.get("next_radius"), zs.next_radius)
        zs.shrink_start = _f(z.get("shrink_start"), zs.shrink_start)
        zs.shrink_end = _f(z.get("shrink_end"), zs.shrink_end)
        zs.dps = _f(z.get("dps"), zs.dps)
        nxt = (zs.next_center[0], zs.next_center[1], zs.next_radius)
        zs.next_changed = self._prev_next is None or any(abs(a - b) > 1e-6 for a, b in zip(nxt, self._prev_next))
        self._prev_next = nxt
        key = (zs.shrink_start, nxt)
        if t <= zs.shrink_start:
            zs.from_center, zs.from_radius = zs.center, zs.radius
        else:
            # protocol §9 says obs.zone is "the current circle"; whether that is the live (moving) circle
            # or the phase-start circle during a shrink is unverified. Robust to both readings: an
            # observed radius equal to the phase-start radius (within ZONE_PHASE_START_TOL_M) means the
            # phase-start reading -> interpolate from it; anything else is the live circle -> back-compute
            # the phase-start circle so that c(t), r(t) pass through it.
            f = zs.frac(t)
            r0 = zs.from_radius if key == self._zone_key else self._phase_start_radius(zs.shrink_start)
            if f >= 1.0 - 1e-6:
                zs.from_center, zs.from_radius = zs.next_center, zs.next_radius
            elif r0 is not None and abs(zs.radius - r0) <= self.C.ZONE_PHASE_START_TOL_M:
                if key != self._zone_key:
                    zs.from_center, zs.from_radius = zs.center, zs.radius
            else:
                k = 1.0 / (1.0 - f)
                zs.from_center = ((zs.center[0] - zs.next_center[0] * f) * k,
                                  (zs.center[1] - zs.next_center[1] * f) * k)
                zs.from_radius = (zs.radius - zs.next_radius * f) * k
            # consumers read zone.center / zone.radius as the live circle
            zs.center, zs.radius = zs.center_at(t), zs.radius_at(t)
        self._zone_key = key

    def _phase_start_radius(self, shrink_start: float) -> Optional[float]:
        """Radius at the start of the shrink that begins at `shrink_start` (config phases: the previous
        phase's radius, zone.start_radius for the first); None when no phase matches."""
        for i, ph in enumerate(self.phases):
            if abs(ph[1] - shrink_start) <= 0.05:
                return self.phases[i - 1][3] if i > 0 else self.start_radius
        return None

    # ------------------------------------------------------------------ 4.2 tracks from obs.mechs
    def _tracks_update(self, mechs, sl) -> None:
        t = self.tick.t
        me = self.me
        C = self.C
        for tr in self.tracks.values():
            tr.in_obs = False
            tr.visible = False
        if not isinstance(mechs, list):
            return
        mx, my = me.pos[0], me.pos[1]
        near = self.vis_near + _f(me.stats.get("near_range_bonus"), 0.0)
        far = self.vis_range + _f(me.stats.get("vision_bonus"), 0.0)
        yaw = me.aim[0]
        for m in mechs:
            if not isinstance(m, dict) or m.get("id") is None:
                continue
            tid = int(_f(m.get("id"), -1))
            if tid == me.id or tid < 0:
                continue
            kind = "npc" if m.get("kind") == "npc" else "mech"
            pos = _v3(m.get("pos"))
            d = math.hypot(pos[0] - mx, pos[1] - my)
            tr = self.tracks.get(tid)
            x = self.tx.get(tid)
            if tr is None:
                tr = Track(id=tid, kind=kind, first_seen_t=t, first_seen_dist=d, last_seen_t=t)
                if kind == "npc":
                    tr.home = pos
                tr.energy_est = self.mech_energy
                self.tracks[tid] = tr
            if x is None:
                x = self.tx[tid] = _TrackExtra()
            if tr.est and tr.first_seen_dist == 0.0:
                tr.first_seen_dist = d
            tr.kind = kind
            tr.est = False
            tr.in_obs = True
            tr.pos = pos
            tr.vel = _v3(m.get("vel"))
            tr.aim = _v2(m.get("aim"), tr.aim)
            if m.get("hp") is not None:
                tr.hp = _f(m.get("hp"))
            if m.get("max_hp") is not None:
                tr.max_hp = _f(m.get("max_hp"))
            tr.cubes = int(_f(m.get("cubes"), tr.cubes))
            if kind == "npc" and tr.home is None:
                tr.home = pos
            if kind == "mech":
                ws = m.get("weapons")
                if isinstance(ws, list):
                    tr.weapons = tuple(str(w) for w in ws if isinstance(w, str))
                    tr.known_weapons = {w: float(C.CONF_HARD) for w in tr.weapons}   # sighting overrides
                    x.sighted = True
                charging = bool(m.get("charging", False))
                if charging and not tr.charging:
                    tr.charging_since = t
                elif not charging:
                    tr.charging_since = None
                tr.charging = charging
            tr.last_seen_t = t
            x.last_evidence_t = t
            sp = math.hypot(tr.vel[0], tr.vel[1])
            if sp < C.STILL_SPEED:
                if tr.still_since is None:
                    tr.still_since = t
            else:
                tr.still_since = None
            # "visible" = listed + inside our vision geometry + SHOT line (A-13 / A-14): a listed mech
            # behind canopy keeps its real position and can be shot, so trees are not tested here
            b = math.atan2(pos[1] - my, pos[0] - mx)
            x.in_view = d <= near or (d <= far and abs(wrap(b - yaw)) <= self.vis_half_fov)
            x.los_geo = self._track_los(tr, x, sl)
            if x.los_geo:
                if x.los_since is None:
                    x.los_since = t
            else:
                x.los_since = None
            tr.visible = x.in_view and x.los_geo
            if tr.visible:
                tr.last_visible_t = t
                if x.vis_since is None:
                    x.vis_since = t
            else:
                x.vis_since = None
            self._energy_seen(tr, x)
        for tid, tr in self.tracks.items():
            if not tr.in_obs:
                x = self.tx.get(tid)
                if x is not None:
                    x.vis_since = None
                    x.los_since = None
                    x.los_geo = False

    def _height_of(self, tr: Track) -> float:
        if tr.kind == "npc":
            return self.npc_height
        return self.mech_height * (1.0 + self.size_per_cube * tr.cubes)

    def _track_los(self, tr: Track, x: _TrackExtra, sl) -> bool:
        me = self.me
        h = self._height_of(tr)
        key = (int(me.pos[0]), int(me.pos[1]), int(me.eye_z * 2), int(tr.pos[0]), int(tr.pos[1]),
               int(tr.pos[2] * 2), int(h))
        if key == x.los_key:
            return x.los_geo
        if sl is not None and sl.expired() and x.los_key is not None:
            return x.los_geo       # over budget: reuse last result
        eye = (me.pos[0], me.pos[1], me.eye_z)
        ok = self._los(eye, (tr.pos[0], tr.pos[1], tr.pos[2] + 0.5 * h)) or \
            self._los(eye, (tr.pos[0], tr.pos[1], tr.pos[2] + h))
        x.los_key = key
        return ok

    def _los(self, a, b) -> bool:
        """SHOT line a -> b: buildings and terrain only (trees never stop shots, protocol 7)."""
        self.los_calls += 1
        try:
            return bool(self.map.los(a, b))
        except Exception:
            return True

    def _sight(self, eye, b, near: bool) -> bool:
        """SIGHT line eye -> b (protocol 8): the shot line plus tree trunks, plus dense canopy unless
        `near` (b inside vision.near_range: proximity sensing is not blocked by canopy) or the match
        runs with vision.canopy_blocks false. True = the server would show us an object at b.
        CONTRACTS 6.1 only binds mapdata.los: a map without visible() answers with the shot line."""
        vis = getattr(self.map, "visible", None)
        if vis is None:
            return self._los(eye, b)
        self.los_calls += 1
        try:
            return bool(vis(eye, b, near=near or not self.vis_canopy))
        except Exception:
            return True

    def _energy_seen(self, tr: Track, x: _TrackExtra) -> None:
        """4.2 energy_est while seen: boost on the ground, jets while rising."""
        if tr.kind != "mech":
            return
        t = self.tick.t
        dt = (t - x.prev_t) if x.prev_t is not None else 0.0
        dt = min(max(dt, 0.0), 0.5)
        sp = math.hypot(tr.vel[0], tr.vel[1])
        vz = tr.vel[2]
        spend = 0.0
        if abs(vz) < 0.5 and sp > self.walk_speed * self.C.BOOST_DETECT_K:
            spend += self.boost_drain * dt
        if vz > 0.5 and dt > 0 and (vz - x.prev_vz) > -0.5 * self.gravity * dt:
            spend += self.jet_drain * dt
            x.last_jet_t = t
            if tr.jet_since is None:
                tr.jet_since = t
        if spend > 0:
            tr.energy_est = max(0.0, tr.energy_est - spend)
            x.last_spend_t = t
        x.prev_vz = vz
        x.prev_t = t

    # ------------------------------------------------------------------ 4.5 beams
    def _beams_update(self, beams) -> None:
        self.beams = beams if isinstance(beams, list) else []
        t = self.tick.t
        C = self.C
        for bm in self.beams:
            if not isinstance(bm, dict) or bm.get("by") is None:
                continue
            by = int(_f(bm.get("by"), -1))
            if by < 0 or by == self.me.id:
                continue
            kind = bm.get("kind")
            path = bm.get("path") or []
            age = _f(bm.get("age"), 0.0)
            p0 = _v3(path[0]) if path else None
            tr = self.tracks.get(by)
            if tr is None:
                if p0 is None:
                    continue
                tr = self._new_est_track(by, "mech", (p0[0], p0[1], p0[2] - 7.0))
            x = self.tx[by]
            x.last_evidence_t = t
            if not tr.in_obs and p0 is not None:
                tr.pos = (p0[0], p0[1], p0[2] - 7.0 * (1.0 + self.size_per_cube * tr.cubes))
                tr.est = True
            if kind in WEAPONS:
                if not (x.sighted and tr.in_obs):
                    tr.known_weapons[kind] = max(tr.known_weapons.get(kind, 0.0), float(C.CONF_HARD))
            if kind == "laser":
                tr.last_laser_shot_t = max(tr.last_laser_shot_t, t - age)
            elif kind == "lightning":
                tr.energy_est = max(0.0, tr.energy_est - self.lt_energy)
                x.last_spend_t = t
                if len(path) >= 2:
                    a, b = _v3(path[0]), _v3(path[1])
                    seg = math.dist(a, b)
                    if self.lt_range > 0:
                        x.obs_mult = max(x.obs_mult, seg / self.lt_range)
            # fighting_with from where the first segment ends
            if len(path) >= 2:
                e = _v3(path[1])
                tgt = self._nearest_body(e, exclude=by)
                if tgt is not None:
                    x.fight_id, x.fight_t = tgt, t

    def _nearest_body(self, p: Vec3, exclude: int, tol: float = 4.0) -> Optional[int]:
        best, bd = None, tol + self.mech_radius
        me = self.me
        d = math.hypot(p[0] - me.pos[0], p[1] - me.pos[1])
        if d < bd + me.radius - self.mech_radius:
            best, bd = me.id, d
        for tid, tr in self.tracks.items():
            if tid == exclude:
                continue
            d = math.hypot(p[0] - tr.pos[0], p[1] - tr.pos[1])
            if d < bd:
                best, bd = tid, d
        return best

    # ------------------------------------------------------------------ events (4.1-4.4)
    def _events_update(self, events) -> None:
        self.events = events if isinstance(events, list) else []
        self.take_failed = []
        t = self.tick.t
        me = self.me
        C = self.C
        regen_delay = _f(me.stats.get("regen_delay"), self.regen_delay)
        for ev in self.events:
            if not isinstance(ev, dict):
                continue
            typ = ev.get("type")
            if typ == "damaged":
                by = ev.get("by")
                by = int(_f(by, -1)) if by is not None else None
                bearing = ev.get("bearing")
                bearing = _f(bearing) if bearing is not None else None
                weapon = str(ev.get("weapon") or "")
                tr = self.tracks.get(by) if by is not None else None
                unseen = by is not None and not (tr is not None and tr.visible)
                rec = DamageRec(t=t, amount=_f(ev.get("amount")), weapon=weapon, bearing=bearing, by=by,
                                rear=bool(ev.get("rear", False)), unseen=unseen)
                self.damage_log.append(rec)
                me.regen_block_until = max(me.regen_block_until, t + regen_delay)
                if weapon == "mortar":
                    self.mortar_active_until = max(self.mortar_active_until, t + C.MORTAR_WARY_S)
                if by is not None and weapon != "zone":
                    self._infer_from_damage(by, weapon, bearing)
            elif typ == "hit":
                tgt = int(_f(ev.get("target"), -1))
                amt = _f(ev.get("amount"))
                self.hits.append(HitRec(t=t, target=tgt, amount=amt, weapon=str(ev.get("weapon") or ""),
                                        effect=ev.get("effect")))
                me.regen_block_until = max(me.regen_block_until, t + regen_delay)
                tr = self.tracks.get(tgt)
                if tr is not None:
                    self.tx[tgt].last_hit_by_us_t = t
                    if not tr.in_obs and tr.hp is not None:
                        tr.hp = max(0.0, tr.hp - amt)
                        if tr.hp <= 0.0 and tr.kind == "npc":
                            self._track_died(tgt)
                cr = self.crates.get(tgt)
                if cr is not None and cr.hp is not None:
                    cr.hp = max(0.0, cr.hp - amt)
            elif typ == "effect":
                if ev.get("effect") == "short":
                    me.short_until = max(me.short_until, t + self.short_s)
            elif typ == "kill":
                self.kills += 1
                tgt = int(_f(ev.get("target"), -1))
                self._track_died(tgt)
            elif typ == "took":
                slot = int(_f(ev.get("slot"), 0))
                kind = str(ev.get("kind") or "")
                cd = _f((self.wcfg.get(kind) or {}).get("cooldown"), 0.0)
                w = me.weapons[slot] if 0 <= slot < len(me.weapons) else None
                if w is not None and w.kind == kind:
                    cd = _f(w.stats.get("cooldown"), cd)
                if 0 <= slot < 2:
                    me.equip_until[slot] = t + max(self.equip_s, cd)
                self.pickups.pop(int(_f(ev.get("id"), -1)), None)
                self.last_swap_t = t
            elif typ == "dropped":
                pid = int(_f(ev.get("id"), -1))
                if pid >= 0:
                    self.pickups[pid] = PickupMem(id=pid, kind=str(ev.get("kind") or ""), pos=me.pos,
                                                  last_seen=t, ours=True)
            elif typ == "take_failed":
                self.take_failed.append(ev)
                if ev.get("reason") == "gone":
                    self.pickups.pop(int(_f(ev.get("id"), -1)), None)

    def _new_est_track(self, tid: int, kind: str, pos: Vec3) -> Track:
        t = self.tick.t
        tr = Track(id=tid, kind=kind, pos=pos, est=True, first_seen_t=t, first_seen_dist=0.0, last_seen_t=t)
        tr.energy_est = self.mech_energy
        if kind == "npc":
            tr.home = pos
        self.tracks[tid] = tr
        x = self.tx[tid] = _TrackExtra()
        x.last_evidence_t = t
        return tr

    def _infer_from_damage(self, by: int, weapon: str, bearing: Optional[float]) -> None:
        """4.3 step 4: damaged.by + weapon -> track `by` holds `weapon` (CONF_HARD); creates it."""
        C = self.C
        t = self.tick.t
        me = self.me
        kind = "npc" if (weapon == "npc" or by >= 100) else "mech"
        rng = {"gun": C.INFER_RANGE_GUN, "laser": C.INFER_RANGE_LASER, "mortar": C.INFER_RANGE_MORTAR,
               "lightning": C.INFER_RANGE_LIGHTNING * self.ghost_lightning_mult}.get(weapon, self.npc_aggro)
        tr = self.tracks.get(by)
        if tr is not None and tr.in_obs:
            x = self.tx[by]
        else:
            if bearing is None:
                bearing = 0.0 if tr is None else math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0])
            d = float(rng)
            if tr is not None:
                # keep a stale estimate's range when its bearing agrees with the hit
                dd = math.hypot(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1])
                bb = math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0])
                if dd > 1.0 and abs(wrap(bb - bearing)) <= self._assoc_tol():
                    d = dd
            px, py = me.pos[0] + d * math.cos(bearing), me.pos[1] + d * math.sin(bearing)
            pz = self._ground(px, py)
            if tr is None:
                tr = self._new_est_track(by, kind, (px, py, pz))
            else:
                tr.pos = (px, py, pz)
                tr.est = True
                tr.last_seen_t = t       # est tracks: last_seen_t = last evidence (forget timer)
            x = self.tx[by]
        x.last_evidence_t = t
        if weapon in WEAPONS and kind == "mech" and not (x.sighted and tr.in_obs):
            tr.known_weapons[weapon] = max(tr.known_weapons.get(weapon, 0.0), float(C.CONF_HARD))
        if kind == "npc":
            tr.aggro_on_us = True

    def _track_died(self, tid: int) -> None:
        tr = self.tracks.pop(tid, None)
        self.tx.pop(tid, None)
        if tr is not None:
            self.loot_bursts.append(((tr.pos[0], tr.pos[1]), self.tick.t))

    def _ground(self, x: float, y: float) -> float:
        try:
            return float(self.map.terrain_h(x, y))
        except Exception:
            return self.me.pos[2]

    # ------------------------------------------------------------------ 4.3 sounds
    def _assoc_tol(self) -> float:
        hearing = _f(self.me.stats.get("hearing"), 1.0) or 1.0
        return math.radians(self.C.ASSOC_SIGMA_K * self.snd_noise_deg * hearing)

    def _band_range(self, band: str, kind: str) -> Tuple[float, float]:
        if band == "near":
            return 0.0, self.snd_near
        if band == "mid":
            return self.snd_near, self.snd_mid
        return self.snd_mid, max(self.snd_mid, _f(self.snd_range.get(kind), 600.0))

    def _sounds_update(self, sounds) -> None:
        self.sounds = []
        if not isinstance(sounds, list):
            return
        C = self.C
        t = self.tick.t
        me = self.me
        mx, my = me.pos[0], me.pos[1]
        tol = self._assoc_tol()
        edge = C.ASSOC_BAND_EDGE_M
        merge = math.radians(C.GHOST_MERGE_DEG)
        rep = C.SOUND_BAND_REP
        # predicted bearing/distance of candidate tracks, once per tick
        cands = []
        for tid, tr in self.tracks.items():
            x = self.tx.get(tid)
            last = max(tr.last_seen_t, x.last_evidence_t if x else -1e9)
            if t - last > C.TRACK_FORGET_S:
                continue
            age = 0.0 if tr.in_obs else max(0.0, t - last)
            d = math.hypot(tr.pos[0] - mx, tr.pos[1] - my)
            slack = self.walk_speed * age
            ang = math.atan2(slack, max(d, 1.0)) if slack > 0 else 0.0
            cands.append((tr, x, math.atan2(tr.pos[1] - my, tr.pos[0] - mx), d, slack, ang))
        gb_by_band: Dict[str, List[list]] = {"near": [], "mid": [], "far": []}
        for g in self.ghosts.values():
            gb_by_band.setdefault(g.band, []).append([g, math.atan2(g.pos[1] - my, g.pos[0] - mx)])
        for s in sounds:
            if not isinstance(s, dict):
                continue
            kind = str(s.get("kind") or "")
            b = _f(s.get("bearing"))
            band = s.get("dist") if s.get("dist") in ("near", "mid", "far") else "far"
            age = _f(s.get("age"))
            rec = SoundRec(kind=kind, bearing=b, band=band, age=age)
            self.sounds.append(rec)
            r = float(rep.get(band, 110.0))
            est = (mx + r * math.cos(b), my + r * math.sin(b))
            if kind == "drop":
                continue
            if kind == "explosion":
                self.explosions.append((est, t - age))
                self._add_heat(est[0], est[1], 1.0)
                if band == "near" or (band == "mid" and t - self.last_launch_t <= C.EXPLOSION_LINK_S):
                    self.mortar_active_until = max(self.mortar_active_until, t + C.MORTAR_WARY_S)
                continue
            if kind == "death":
                self.loot_bursts.append((est, t - age))
                self._add_heat(est[0], est[1], 1.0)
                continue
            if kind == "mortar" and band in ("near", "mid"):
                self.last_launch_t = t - age
                self.mortar_active_until = max(self.mortar_active_until, t + C.MORTAR_WARY_S)
            want_npc = kind in NPC_SOUNDS
            weapon = SOUND_WEAPON.get(kind)
            lo, hi = self._band_range(band, kind)
            best, berr = None, None
            for tr, x, tb, d, slack, ang in cands:
                if (tr.kind == "npc") != want_npc:
                    continue
                if weapon and tr.in_obs and x is not None and x.sighted and weapon not in tr.weapons:
                    continue   # authoritative loadout excludes this source
                err = abs((b - tb + math.pi) % TWO_PI - math.pi)
                if err > tol + ang:
                    continue
                if d < lo - edge - slack or d > hi + edge + slack:
                    continue
                if berr is None or err < berr or (err == berr and tr.id < best.id):
                    best, berr = tr, err
            if best is not None:
                rec.track_id = best.id
                self._infer_sound(best, kind, age)
                self._add_heat(best.pos[0], best.pos[1], 1.0)
                continue
            # ghost association / creation
            gbest, gerr, gent = None, None, None
            for ent in gb_by_band.get(band, ()):
                err = abs((b - ent[1] + math.pi) % TWO_PI - math.pi)
                if err <= merge and (gerr is None or err < gerr):
                    gbest, gerr, gent = ent[0], err, ent
            if gbest is None:
                gbest = Ghost(id=self._next_ghost, pos=est, bearing=b, band=band, first_t=t - age, last_t=t - age)
                self.ghosts[gbest.id] = gbest
                self._next_ghost -= 1
                gb_by_band.setdefault(band, []).append([gbest, b])
            else:
                gbest.pos, gbest.bearing = est, b
                gbest.last_t = max(gbest.last_t, t - age)
                gent[1] = b
            gbest.last_kind = kind
            if weapon:
                gbest.kinds[weapon] = max(gbest.kinds.get(weapon, 0.0), float(C.CONF_SOUND))
            rec.ghost_id = gbest.id
            self._add_heat(est[0], est[1], 1.0)

    def _infer_sound(self, tr: Track, kind: str, age: float) -> None:
        """4.3 step 3 on an associated track."""
        C = self.C
        t = self.tick.t
        x = self.tx.get(tr.id)
        if x is None:
            x = self.tx[tr.id] = _TrackExtra()
        x.last_evidence_t = t
        weapon = SOUND_WEAPON.get(kind)
        if weapon and tr.kind == "mech" and not (x.sighted and tr.in_obs):
            tr.known_weapons[weapon] = max(tr.known_weapons.get(weapon, 0.0), float(C.CONF_SOUND))
        if kind == "reload":
            tr.reloading_until = max(tr.reloading_until, t - age + self.gun_reload)
        elif kind == "jet":
            x.last_jet_t = t
            if tr.jet_since is None:
                tr.jet_since = t - age
            x.last_spend_t = t
        elif kind in ("step", "boost"):
            x.last_move_snd_t = t
            if kind == "boost":
                x.last_spend_t = t

    def _gcell(self, x: float, y: float) -> Optional[int]:
        i = int((x - self.g0x) // SEEN_STEP)
        j = int((y - self.g0y) // SEEN_STEP)
        if 0 <= i < self.gnx and 0 <= j < self.gny:
            return j * self.gnx + i
        return None

    def _add_heat(self, x: float, y: float, amount: float) -> None:
        c = self._gcell(x, y)
        if c is None:
            return
        t = self.tick.t
        h = self.heat.get(c)
        if h is None:
            self.heat[c] = [amount, t]
        else:
            h[0] = h[0] * 0.5 ** ((t - h[1]) / max(1e-6, self.C.DANGER_HEAT_DECAY_S)) + amount
            h[1] = t

    # ------------------------------------------------------------------ 4.5 projectiles
    def _projectiles_update(self, projs) -> None:
        C = self.C
        t = self.tick.t
        me = self.me
        self.shells = []
        self.bullets = []
        self.npc_bullets = []
        if not isinstance(projs, list):
            self._shell_cache.clear()
            return
        live = set()
        splash_r = self.mortar_splash * C.MORTAR_ENEMY_SPLASH_MULT
        R = me.radius + C.BULLET_INBOUND_PAD_M
        height = _f(me.stats.get("height"), self.mech_height * me.s)
        for pj in projs:
            if not isinstance(pj, dict):
                continue
            pid = int(_f(pj.get("id"), -1))
            kind = pj.get("kind")
            p = _v3(pj.get("pos"))
            v = _v3(pj.get("vel"))
            if kind == "mortar":
                live.add(pid)
                hit = self._shell_cache.get(pid)
                if hit is None:
                    imp, tau = self._shell_impact(p, v)
                    hit = (imp, t + tau)
                    self._shell_cache[pid] = hit
                self.shells.append(ShellPred(id=pid, impact=hit[0], t_impact=hit[1], splash_r=splash_r))
                self.mortar_active_until = max(self.mortar_active_until, hit[1])
            elif kind in ("gun", "npc"):
                rx, ry = p[0] - me.pos[0], p[1] - me.pos[1]
                vx, vy = v[0] - me.vel[0], v[1] - me.vel[1]
                vv = vx * vx + vy * vy
                if vv < 1e-6:
                    continue
                tau = -(rx * vx + ry * vy) / vv
                if tau < 0.0 or tau > C.BULLET_INBOUND_S:
                    continue
                cx, cy = rx + vx * tau, ry + vy * tau
                miss = math.hypot(cx, cy)
                if miss >= R:
                    continue
                z = p[2] + v[2] * tau
                if z < me.pos[2] - C.BULLET_INBOUND_PAD_M or z > me.pos[2] + height + C.BULLET_INBOUND_PAD_M:
                    continue
                bt = BulletThreat(id=pid, kind=kind, t_closest=t + tau, miss_dist=miss)
                (self.bullets if kind == "gun" else self.npc_bullets).append(bt)
        for k in [k for k in self._shell_cache if k not in live]:
            del self._shell_cache[k]
        self.shells.sort(key=lambda s: s.t_impact)

    def _shell_impact(self, p: Vec3, v: Vec3) -> Tuple[Vec3, float]:
        """4.5: integrate z(tau) = z0 + vz tau - g tau^2 / 2 against surface_h in 0.05 s steps."""
        g = self.mortar_g
        md = self.map
        tau = 0.0
        x, y, z = p
        while tau < self.mortar_ttl:
            tau += SHELL_STEP_S
            x = p[0] + v[0] * tau
            y = p[1] + v[1] * tau
            z = p[2] + v[2] * tau - 0.5 * g * tau * tau
            try:
                s = md.surface_h(x, y)
            except Exception:
                s = -1e9
            if z <= s:
                return (x, y, s), tau
        return (x, y, z), tau

    # ------------------------------------------------------------------ 4.4 items
    def _items_update(self, obs: dict, sl) -> None:
        t = self.tick.t
        listed_p = set()
        for pk in obs.get("pickups") or []:
            if not isinstance(pk, dict) or pk.get("id") is None:
                continue
            pid = int(_f(pk.get("id"), -1))
            listed_p.add(pid)
            m = self.pickups.get(pid)
            if m is None:
                self.pickups[pid] = PickupMem(id=pid, kind=str(pk.get("kind") or ""), pos=_v3(pk.get("pos")),
                                              last_seen=t)
            else:
                m.kind = str(pk.get("kind") or m.kind)
                m.pos = _v3(pk.get("pos"), m.pos)
                m.last_seen = t
        listed_c = set()
        for cr in obs.get("crates") or []:
            if not isinstance(cr, dict) or cr.get("id") is None:
                continue
            cid = int(_f(cr.get("id"), -1))
            listed_c.add(cid)
            hp = cr.get("hp")
            hp = _f(hp) if hp is not None else None
            m = self.crates.get(cid)
            if m is None:
                self.crates[cid] = CrateMem(id=cid, pos=_v3(cr.get("pos")), hp=hp, last_seen=t)
            else:
                m.pos = _v3(cr.get("pos"), m.pos)
                m.hp = hp
                m.last_seen = t
        self.cubes = {}
        for cu in obs.get("cubes") or []:
            if isinstance(cu, dict) and cu.get("id") is not None:
                cid = int(_f(cu.get("id"), -1))
                self.cubes[cid] = CubeObs(id=cid, pos=_v3(cu.get("pos")))
        # forget remembered items whose spot is in near view (SIGHT line: a trunk can hide it) but not listed
        for store, listed, h in ((self.pickups, listed_p, 0.5), (self.crates, listed_c, 1.0)):
            gone = []
            for iid, m in store.items():
                if iid in listed or t - m.last_seen < ITEM_GONE_GRACE_S:
                    continue
                if self._point_in_view(m.pos, h, sl, near_only=True):
                    gone.append(iid)
            for iid in gone:
                del store[iid]
        listed_d = set()
        for dr in obs.get("drops") or []:
            if not isinstance(dr, dict) or dr.get("id") is None:
                continue
            did = int(_f(dr.get("id"), -1))
            listed_d.add(did)
            lands_in = _f(dr.get("lands_in"))
            holder = dr.get("holder")
            m = self.drops.get(did)
            if m is None:
                m = self.drops[did] = DropState(id=did, pos=_v3(dr.get("pos")), radius=_f(dr.get("radius"), 8.0),
                                                lands_in=lands_in, capture_s=_f(dr.get("capture_s"), 4.0),
                                                progress=_f(dr.get("progress")), holder=None, first_seen_t=t,
                                                lands_at=t + lands_in)
            m.pos = _v3(dr.get("pos"), m.pos)
            m.radius = _f(dr.get("radius"), m.radius)
            m.lands_in = lands_in
            m.capture_s = _f(dr.get("capture_s"), m.capture_s)
            m.progress = _f(dr.get("progress"))
            m.holder = int(_f(holder, -1)) if holder is not None else None
            if lands_in > 0.0:              # once landed the time stays: M7's DROP_TIMEOUT_S counts from it
                m.lands_at = t + lands_in
        for did in [d for d in self.drops if d not in listed_d]:
            del self.drops[did]

    def _point_in_view(self, pos: Vec3, h: float, sl, near_only: bool = False) -> bool:
        """Would the server show us an object at pos now (protocol 8: vision geometry + SIGHT line)?
        Inside near range trunks hide it, beyond it trunks and dense canopy do. near_only: proximity
        range only (4.4: item / NPC memory is only dropped on a near look)."""
        me = self.me
        dx, dy = pos[0] - me.pos[0], pos[1] - me.pos[1]
        d = math.hypot(dx, dy)
        near = self.vis_near + _f(me.stats.get("near_range_bonus"), 0.0)
        far = self.vis_range + _f(me.stats.get("vision_bonus"), 0.0)
        if d > near and (near_only or d > far or abs(wrap(math.atan2(dy, dx) - me.aim[0])) > self.vis_half_fov):
            return False
        if sl is not None and sl.expired():
            return False
        return self._sight((me.pos[0], me.pos[1], me.eye_z), (pos[0], pos[1], pos[2] + h), d <= near)

    def _offer_update(self, of) -> None:
        if not isinstance(of, dict) or of.get("id") is None:
            self.offer = None
            return
        oid = int(_f(of.get("id"), -1))
        choices = []
        for c in of.get("choices") or []:
            if isinstance(c, dict):
                choices.append(str(c.get("id", "")))
            elif isinstance(c, str):
                choices.append(c)
        if self.offer is not None and self.offer.id == oid:
            self.offer.choices = choices
            self.offer.expires = _f(of.get("expires"), self.offer.expires)
        else:
            self.offer = OfferState(id=oid, choices=choices, expires=_f(of.get("expires")))

    # ------------------------------------------------------------------ 4.2 maintenance
    def _tracks_maintain(self, sl) -> None:
        C = self.C
        t = self.tick.t
        me = self.me
        dt = min(max(self.tick.dt, 0.0), 1.0)
        # per-track damage to us over SELF_DEFENSE_S
        dmg: Dict[int, float] = {}
        last: Dict[int, float] = {}
        t0 = t - C.SELF_DEFENSE_S
        for r in self.damage_log:
            if r.by is None:
                continue
            if r.t >= t0:
                dmg[r.by] = dmg.get(r.by, 0.0) + r.amount
            if r.t > last.get(r.by, -1e9):
                last[r.by] = r.t
        offers = 0
        drop = []
        vis_mechs = [tr for tr in self.tracks.values() if tr.visible and tr.kind == "mech"]
        fight_tol = math.radians(C.FIGHTING_AIM_TOL_DEG)
        for tid, tr in self.tracks.items():
            x = self.tx.get(tid)
            if x is None:
                x = self.tx[tid] = _TrackExtra()
            tr.damage_to_us = dmg.get(tid, 0.0)
            tr.last_damaged_us_t = max(tr.last_damaged_us_t, last.get(tid, -1e9))
            if tr.kind == "npc":
                tr.aggro_on_us = (t - tr.last_damaged_us_t <= C.NPC_PERSIST_S
                                  or t - x.last_hit_by_us_t <= C.NPC_PERSIST_S)
            # energy regen (unseen time counts as regen)
            if t - x.last_spend_t >= self.energy_regen_delay:
                tr.energy_est = min(self.mech_energy, tr.energy_est + self.energy_regen * dt)
            if tr.jet_since is not None and t - x.last_jet_t > 1.0:
                tr.jet_since = None
            if tr.in_obs:
                try:
                    tr.on_roof = tr.pos[2] - self.map.terrain_h(tr.pos[0], tr.pos[1]) > C.ROOF_Z_MIN
                except Exception:
                    tr.on_roof = False
            # A-20 lightning range multiplier
            if x.sighted or tr.in_obs:
                offers = sum(1 for th in self.thresholds if th <= tr.cubes)
                worst = float(C.TESLA_MUL) ** min(int(C.TESLA_MAX_STACKS), offers)
            else:
                worst = self.ghost_lightning_mult
            tr.lightning_range_mult = max(x.obs_mult, worst)
            # fighting_with from aim (another track or us within FIGHTING_AIM_TOL_DEG)
            if tr.visible and tr.kind == "mech":
                best, bd = None, None
                for o in vis_mechs:
                    if o.id == tid:
                        continue
                    dd = math.hypot(o.pos[0] - tr.pos[0], o.pos[1] - tr.pos[1])
                    if dd < 1.0 or abs(wrap(math.atan2(o.pos[1] - tr.pos[1], o.pos[0] - tr.pos[0]) - tr.aim[0])) > fight_tol:
                        continue
                    if bd is None or dd < bd:
                        best, bd = o.id, dd
                dd = math.hypot(me.pos[0] - tr.pos[0], me.pos[1] - tr.pos[1])
                if dd >= 1.0 and abs(wrap(math.atan2(me.pos[1] - tr.pos[1], me.pos[0] - tr.pos[0]) - tr.aim[0])) <= fight_tol:
                    if bd is None or dd < bd:
                        best, bd = me.id, dd
                if best is not None:
                    x.fight_id, x.fight_t = best, t
            tr.fighting_with = x.fight_id if (x.fight_id is not None and t - x.fight_t <= C.FIGHTING_WINDOW_S) else None
            # forgetting
            lastev = max(tr.last_seen_t, x.last_evidence_t)
            if tr.kind == "mech":
                if t - lastev > C.TRACK_FORGET_S:
                    drop.append(tid)
            else:
                # NPCs: kept while alive; gone once its last spot is in view, not listed, memory over
                if (not tr.in_obs and t - tr.last_seen_t > _f(C.path("vision.memory_s", 5), 5.0) + 1.0
                        and self._point_in_view(tr.pos, 0.5 * self.npc_height, sl, near_only=True)):
                    drop.append(tid)
        for tid in drop:
            self.tracks.pop(tid, None)
            self.tx.pop(tid, None)

    # ------------------------------------------------------------------ 4.3 steps 6 + merge
    def _ghosts_maintain(self, sl) -> None:
        C = self.C
        t = self.tick.t
        me = self.me
        merge = math.radians(C.GHOST_MERGE_DEG)
        vis = [tr for tr in self.tracks.values() if tr.visible and tr.kind == "mech"]
        dead = []
        for gid, g in self.ghosts.items():
            if t - g.last_t > C.GHOST_TTL_S:
                dead.append(gid)
                continue
            gb = math.atan2(g.pos[1] - me.pos[1], g.pos[0] - me.pos[0])
            lo, hi = self._band_range(g.band, g.last_kind)
            merged = None
            gk = [k for k, c in g.kinds.items() if c >= C.CONF_SOUND]
            for tr in vis:
                x = self.tx.get(tr.id)
                if gk and x is not None and x.sighted and any(k not in tr.weapons for k in gk):
                    continue   # its seen loadout cannot have made these sounds
                d = math.hypot(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1])
                if abs(wrap(math.atan2(tr.pos[1] - me.pos[1], tr.pos[0] - me.pos[0]) - gb)) <= merge and \
                        lo - C.ASSOC_BAND_EDGE_M <= d <= hi + C.ASSOC_BAND_EDGE_M:
                    merged = tr
                    break
            if merged is not None:
                x = self.tx.get(merged.id)
                if not (x and x.sighted):
                    for k, c in g.kinds.items():
                        merged.known_weapons[k] = max(merged.known_weapons.get(k, 0.0), c)
                dead.append(gid)
                continue
            if t - g.last_t >= C.SOUND_REACT_AGE_S:
                gz = self._ground(g.pos[0], g.pos[1]) + 4.5
                if self._point_in_view((g.pos[0], g.pos[1], gz), 0.0, sl):
                    if not any(math.hypot(tr.pos[0] - g.pos[0], tr.pos[1] - g.pos[1]) <= C.GHOST_REACH_M for tr in vis):
                        dead.append(gid)
        for gid in dead:
            self.ghosts.pop(gid, None)

    # ------------------------------------------------------------------ 4.4 seen cells
    def _seen_update(self, sl) -> None:
        me = self.me
        ci = int((me.pos[0] - self.g0x) // SEEN_STEP)
        cj = int((me.pos[1] - self.g0y) // SEEN_STEP)
        c = self._gcell(me.pos[0], me.pos[1])
        if c is not None and not self.seen[c]:
            self.seen[c] = 1
            self.seen_count += 1
        offs = self._seen_offs
        n = len(offs)
        if n == 0:
            return
        near = self.vis_near + _f(me.stats.get("near_range_bonus"), 0.0)
        far = min(self.C.SEEN_FOV_R, self.vis_range + _f(me.stats.get("vision_bonus"), 0.0))
        yaw = me.aim[0]
        half = self.vis_half_fov
        eye = (me.pos[0], me.pos[1], me.eye_z)
        gnx, gny = self.gnx, self.gny
        seen = self.seen
        checks = 0
        p = self._seen_ptr
        for _ in range(min(n, SEEN_SCAN_PER_TICK)):
            di, dj, d, ang = offs[p]
            p += 1
            if p >= n:
                p = 0
            i, j = ci + di, cj + dj
            if not (0 <= i < gnx and 0 <= j < gny):
                continue
            k = j * gnx + i
            if seen[k]:
                continue
            if d > near and (d > far or abs(wrap(ang - yaw)) > half):
                continue
            if sl is not None and sl.expired():
                break
            x = self.g0x + (i + 0.5) * SEEN_STEP
            y = self.g0y + (j + 0.5) * SEEN_STEP
            try:
                z = self.map.surface_h(x, y) + 1.0
            except Exception:
                z = me.pos[2] + 1.0
            # SIGHT line: a far cell under a dense crown or behind a trunk has not been looked at
            if self._sight(eye, (x, y, z), math.hypot(x - eye[0], y - eye[1]) <= near):
                seen[k] = 1
                self.seen_count += 1
            checks += 1
            if checks >= SEEN_LOS_PER_TICK:
                break
        self._seen_ptr = p

    # ------------------------------------------------------------------ pruning
    def _prune(self) -> None:
        t = self.tick.t
        C = self.C
        while self.damage_log and t - self.damage_log[0].t > LOG_KEEP_S:
            self.damage_log.popleft()
        while self.hits and t - self.hits[0].t > LOG_KEEP_S:
            self.hits.popleft()
        if self.explosions:
            self.explosions = [e for e in self.explosions if t - e[1] <= C.GHOST_TTL_S]
        if self.loot_bursts:
            self.loot_bursts = [b for b in self.loot_bursts if t - b[1] <= C.LOOT_STALE_S]
        if len(self.heat) > 4096:
            hl = C.DANGER_HEAT_DECAY_S
            self.heat = {c: h for c, h in self.heat.items() if h[0] * 0.5 ** ((t - h[1]) / hl) > 0.01}
