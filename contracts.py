"""Shared data types for the bot pipeline (see CONTRACTS.md for the module APIs that use them).

Only plain data lives here (plus the tiny Clock/Slice budget helper every stage needs), so world,
assess, nav, aim, overlays, modes, tactics and arbiter can be written in parallel against it.
Conventions (plan section 0): meters, seconds (match time `t` from obs), radians; x east, y north,
z up; yaw from +x counter-clockwise, wrap(a) in (-pi, pi]; distances horizontal unless "3d".
Vectors are tuples: Vec2 = (x, y), Vec3 = (x, y, z). Ids are ints (protocol ids are JSON numbers;
world converts). Absolute times are match time `t` (obs["t"]); `*_until` fields are absolute.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

WEAPON_KINDS = ("gun", "laser", "lightning", "mortar")
SOUND_BANDS = ("near", "mid", "far")

# Single mode order (plan 5.2, H1): X5 SAFE > X4 ENDGAME > X2 FIGHT > X3 FLEE > X1 GOAL.
MACRO_ORDER = ("X5", "X4", "X2", "X3", "X1")
MACRO_NAMES = {"X0": "BOOT", "X1": "GOAL", "X2": "FIGHT", "X3": "FLEE", "X4": "ENDGAME", "X5": "SAFE"}
# Goal order inside X1 (5.5) and the loot-window order (t < LOOT_END_T).
GOAL_ORDER = ("M2U", "M6", "M7", "M3A", "M3H", "M8", "M1", "M2", "M3")
GOAL_ORDER_LOOT_WINDOW = ("M2U", "M1", "M6", "M2", "M3")
GOAL_NAMES = {"M2U": "ROTATE_URGENT", "M6": "RECOVER", "M7": "DROP", "M3A": "AMBUSH", "M3H": "HIDE",
              "M8": "INVESTIGATE", "M1": "LOOT", "M2": "ROTATE", "M3": "HOLD"}
PLAYBOOKS = ("PB_L", "PB_Z", "PB_G")
TERRAIN_CALLS = ("LIGHTNING", "LASER", "MIXED")

# Leg owner precedence (5.2 / 6.2): first present level owns d_own. L2 (mortar) is a filter only.
# X5 replaces L2-L8; X4 replaces L5-L8 (it sits at L5's place).
LEG_OWNER_ORDER = ("X5", "L3", "L3b", "L4", "X4", "L5", "L6", "L7", "L8")
# Leg filter application order (6.3 step 1).
LEG_FILTER_ORDER = ("L2", "L4", "O8", "O9")
# Aim priority (6.4): lower number wins. X4 ENDGAME publishes its own list with the same numbers.
AIM_PRIO = {"AIM1": 1, "AIM2": 2, "AIM3": 3, "AIM4": 4, "AIM5": 5, "AIM6": 6, "AIM7": 7, "AIM8": 8,
            "AIM9": 9, "AIM10": 10, "X4_SELF_DEFENSE": 3}
# Boost owner order (6.5): first owner with an opinion sets boost.
BOOST_ORDER = ("O6", "O1", "L2", "O3", "O4", "CLIMB", "MODE")
# Jump owner order (6.5).
JUMP_ORDER = ("CLIMB", "ROOF_ASSAULT", "O5", "X5")
# Take/drop single-writer order (6.7).
TAKE_ORDER = ("R1.4", "R0.4", "L7", "RETRY")


# ------------------------------------------------------------------ world model (plan section 4)
@dataclass(slots=True)
class WeaponState:
    """One held weapon (obs.self.weapons[i]); stats are this weapon's numbers after upgrades."""
    kind: str
    cooldown: float = 0.0
    ammo: Optional[float] = None      # gun only
    reload: float = 0.0               # gun only: seconds of reload left
    charge: float = 0.0               # laser only: seconds charged (0 = not charging)
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SelfState:
    """4.1. Updated every tick from obs.self plus events."""
    id: int = -1
    pos: Vec3 = (0.0, 0.0, 0.0)
    vel: Vec3 = (0.0, 0.0, 0.0)
    aim: Vec2 = (0.0, 0.0)            # actual (slewed) [yaw, pitch]
    hp: float = 0.0
    max_hp: float = 0.0
    energy: float = 0.0
    max_energy: float = 0.0
    grounded: bool = True
    cubes: int = 0
    upgrades: List[str] = field(default_factory=list)
    weapons: List[Optional[WeaponState]] = field(default_factory=lambda: [None, None])
    in_zone: bool = True
    stats: Dict[str, float] = field(default_factory=dict)
    # derived
    s: float = 1.0                    # size factor 1 + size_per_cube * cubes
    radius: float = 2.5               # stats.radius
    eye_z: float = 0.0                # pos.z + stats.eye_height
    muzzle_z: float = 0.0             # pos.z + stats.muzzle_height
    armor: float = 0.0
    playbook: str = "PB_G"            # 5.1, follows held weapons (H10)
    short_until: float = 0.0          # lightning short on our laser
    equip_until: List[float] = field(default_factory=lambda: [0.0, 0.0])
    laser_slot: Optional[int] = None
    laser_charge: float = 0.0
    speed_h: float = 0.0              # horizontal speed
    regen_block_until: float = 0.0    # t + regen_delay after last dealt/taken damage


@dataclass(slots=True)
class Track:
    """4.2. One per mech or NPC id seen in obs.mechs (or created by damage inference)."""
    id: int
    kind: str                         # "mech" | "npc"
    pos: Vec3 = (0.0, 0.0, 0.0)
    vel: Vec3 = (0.0, 0.0, 0.0)
    aim: Vec2 = (0.0, 0.0)
    hp: Optional[float] = None
    max_hp: Optional[float] = None
    cubes: int = 0
    weapons: Tuple[str, ...] = ()     # authoritative when seen (obs.mechs[].weapons)
    charging: bool = False
    in_obs: bool = False              # listed in obs.mechs this tick (may be memory, A-14)
    visible: bool = False             # in obs.mechs AND our LOS check passes this tick
    est: bool = False                 # position is an estimate (inferred from damage/sound), not seen
    first_seen_t: float = 0.0
    first_seen_dist: float = 0.0
    last_seen_t: float = 0.0          # last tick in obs.mechs
    last_visible_t: float = -1e9
    known_weapons: Dict[str, float] = field(default_factory=dict)   # kind -> confidence 0..1
    charging_since: Optional[float] = None
    still_since: Optional[float] = None
    energy_est: float = 100.0
    reloading_until: float = 0.0
    last_laser_shot_t: float = -1e9
    jet_since: Optional[float] = None
    on_roof: bool = False
    fighting_with: Optional[int] = None
    damage_to_us: float = 0.0         # over the last SELF_DEFENSE_S
    last_damaged_us_t: float = -1e9
    lightning_range_mult: float = 1.0  # ENEMY_LIGHTNING_RANGE_MULT for this track (A-20)
    home: Optional[Vec3] = None       # NPCs: first seen position (leash center)
    aggro_on_us: bool = False         # NPCs: shooting at us recently


@dataclass(slots=True)
class Ghost:
    """4.3. A sound contact not associated with a track. Ids are negative (world assigns -1, -2, ...)."""
    id: int
    pos: Vec2                         # estimate: self.pos + SOUND_BAND_REP[band] * (cos b, sin b)
    bearing: float                    # from our position at last_t
    band: str                         # "near" | "mid" | "far"
    first_t: float
    last_t: float
    last_kind: str = ""
    kinds: Dict[str, float] = field(default_factory=dict)  # inferred weapon kind -> confidence
    looked_at_t: float = -1e9         # H11: last AIM8 LOOK it triggered


@dataclass(slots=True)
class SoundRec:
    """One sound of this tick after association (4.3)."""
    kind: str
    bearing: float
    band: str
    age: float
    track_id: Optional[int] = None    # associated track
    ghost_id: Optional[int] = None    # associated or created ghost (None for explosions)


@dataclass(slots=True)
class PickupMem:
    """4.4. A weapon pickup seen (or dropped by us)."""
    id: int
    kind: str
    pos: Vec3
    last_seen: float
    ours: bool = False                # we dropped it


@dataclass(slots=True)
class CrateMem:
    id: int
    pos: Vec3
    hp: Optional[float]
    last_seen: float


@dataclass(slots=True)
class CubeObs:
    id: int
    pos: Vec3


@dataclass(slots=True)
class DropState:
    """Supply drop from obs.drops (known to all from the call)."""
    id: int
    pos: Vec3
    radius: float
    lands_in: float
    capture_s: float
    progress: float
    holder: Optional[int]
    first_seen_t: float
    lands_at: float                   # landing time: t + lands_in, refreshed while it falls, fixed once down


@dataclass(slots=True)
class DamageRec:
    """One `damaged` event (4.2/4.3, R3.1)."""
    t: float
    amount: float
    weapon: str                       # gun, mortar, laser, lightning, npc, zone
    bearing: Optional[float]
    by: Optional[int]
    rear: bool = False
    unseen: bool = False              # `by` not visible when it landed (O4 trigger)


@dataclass(slots=True)
class HitRec:
    t: float
    target: int
    amount: float
    weapon: str
    effect: Optional[str] = None


@dataclass(slots=True)
class ShellPred:
    """4.5. Predicted mortar impact."""
    id: int
    impact: Vec3
    t_impact: float                   # absolute match time
    splash_r: float                   # splash_radius * MORTAR_ENEMY_SPLASH_MULT


@dataclass(slots=True)
class BulletThreat:
    """4.5. A gun/npc bullet whose closest approach passes within R_self + BULLET_INBOUND_PAD_M."""
    id: int
    kind: str                         # "gun" | "npc"
    t_closest: float                  # absolute time of closest approach
    miss_dist: float


@dataclass(slots=True)
class ZoneState:
    """4.7. Current circle and the announced next one; c(t), r(t) linear during the shrink."""
    center: Vec2 = (0.0, 0.0)
    radius: float = 0.0
    next_center: Vec2 = (0.0, 0.0)
    next_radius: float = 0.0
    shrink_start: float = 0.0
    shrink_end: float = 0.0
    dps: float = 0.0
    next_changed: bool = False        # next_center/next_radius differ from last tick
    # circle at shrink_start (from which c(t), r(t) interpolate)
    from_center: Vec2 = (0.0, 0.0)
    from_radius: float = 0.0

    def frac(self, t: float) -> float:
        if t <= self.shrink_start:
            return 0.0
        if t >= self.shrink_end or self.shrink_end <= self.shrink_start:
            return 1.0
        return (t - self.shrink_start) / (self.shrink_end - self.shrink_start)

    def center_at(self, t: float) -> Vec2:
        f = self.frac(t)
        return (self.from_center[0] + (self.next_center[0] - self.from_center[0]) * f,
                self.from_center[1] + (self.next_center[1] - self.from_center[1]) * f)

    def radius_at(self, t: float) -> float:
        f = self.frac(t)
        return self.from_radius + (self.next_radius - self.from_radius) * f


@dataclass(slots=True)
class OfferState:
    id: int
    choices: List[str]                # upgrade ids, index = pick value
    expires: float
    answered_tick: Optional[int] = None


@dataclass(slots=True)
class TickInfo:
    """Per-tick bookkeeping produced by world.update (6.1 step 1)."""
    tick: int = 0
    t: float = 0.0
    dt: float = 0.0                   # match time since previous decision
    missed_ticks: int = 0             # decisions skipped (tick gap / decide_every - 1)
    was_charging_before_gap: bool = False
    alive: int = 16


# ------------------------------------------------------------------ map / nav data (sections 8.9, 11)
@dataclass(slots=True)
class Roof:
    """11.2 `roofs` blob entry."""
    id: int                           # index into MapData.roofs (foot_id value; 0 = none, so ids start at 1)
    top: float                        # roof surface z
    ground: float
    h: float                          # top - ground
    area: float
    min_width: float
    centroid: Vec2
    ring: List[Vec2]                  # simplified outline
    edges: List[Tuple[Vec2, Vec2, Vec2]] = field(default_factory=list)  # (a, b, outward normal)
    neighbors: List[int] = field(default_factory=list)                  # roofs within ROOF_NEIGHBOR_R
    sight: float = 0.0                # mean roof sight


@dataclass(slots=True)
class ClimbFace:
    """11.2 `climbs` blob entry."""
    roof_id: int
    ground_cell: int                  # nav4 index
    roof_cell: int                    # nav4 index
    pos: Vec2                         # wall foot point
    h: float                          # climb height
    eave: bool                        # eave overhangs the face (jet blocked) -> excluded


@dataclass(slots=True)
class Path:
    """nav A* result (8.9). points[0] is near the start; complete=False is a usable partial path."""
    key: str
    goal: Vec2
    points: List[Vec2]
    complete: bool
    eta_s: float
    climbs: List[int] = field(default_factory=list)   # indices into points where a climb/jump edge starts
    created_t: float = 0.0


@dataclass(slots=True)
class Steer:
    """nav.follow output: what the legs should do this tick to follow a path or reach a point."""
    direction: Optional[Vec2]         # unit; None = arrived / stop
    jump: bool = False                # R13.2 jump edge within JUMP_TRIGGER_DIST_M
    climb_boost: bool = False         # R13.2 climb in progress (boost owner "CLIMB")
    arrived: bool = False
    blocked: bool = False             # no clear direction (O5 takes over)
    rule: str = "8.9"


class ExitMemory:
    """5.7: (macro_or_goal, key) -> until_t. key = target / drop / ghost / spot id or "*".
    PERMANENT = until the match ends."""
    __slots__ = ("_d",)
    PERMANENT = float("inf")

    def __init__(self) -> None:
        self._d: Dict[Tuple[str, object], float] = {}

    def write(self, kind: str, key: object, until_t: float) -> None:
        cur = self._d.get((kind, key), -1e18)
        self._d[(kind, key)] = max(cur, until_t)

    def clear(self, kind: str, key: object, t: float) -> bool:
        """True when no memory blocks (kind, key) at time t (checks the exact key and '*')."""
        return self._d.get((kind, key), -1e18) <= t and self._d.get((kind, "*"), -1e18) <= t

    def until(self, kind: str, key: object) -> float:
        return self._d.get((kind, key), -1e18)

    def prune(self, t: float) -> None:
        for k in [k for k, u in self._d.items() if u <= t]:
            del self._d[k]

    def items(self):
        return list(self._d.items())


# ------------------------------------------------------------------ assessment (plan section 8)
@dataclass(slots=True)
class TrackAssess:
    """Per-track numbers from assess (8.1, 8.2)."""
    id: int
    d: float                          # horizontal distance
    d3: float
    bearing: float                    # from us to it
    los: bool                         # our muzzle/eye to its center (cached per tick)
    dps_to_us: float                  # Σ_w DPS(j -> us, d), aiming factor applied
    dps_from_us: float                # Σ_w DPS(us -> j, d)
    ttk_us_on_j: float                # inf if 0
    ttk_them_on_us: float             # j alone
    A: float                          # ttk_them_on_us / ttk_us_on_j (j alone)
    prio: float                       # 8.2.2
    aiming_at_us: bool
    in_its_reach: bool                # we are inside its weapon reach with LOS
    charging_at_us: bool              # O3 test
    closing_speed: float              # >0 = closing
    A_entry: float = 0.0              # 8.2.1 engage ratio: j assumed aiming at us, no reload/rear discounts
    dps_to_us_entry: float = 0.0      # DPS(j -> us) under the A_entry assumptions


@dataclass(slots=True)
class Threats:
    laser_chargers: List[int] = field(default_factory=list)     # O3 candidates, least remaining charge first
    shells: List[ShellPred] = field(default_factory=list)
    bullets_inbound: List[BulletThreat] = field(default_factory=list)
    unseen_hits: List[DamageRec] = field(default_factory=list)  # within UNSEEN_REACT_S
    mortar_active_until: float = 0.0


@dataclass(slots=True)
class FightEval:
    """8.2.3-8.2.6 predicates for the current macro context (assess.evaluate_fight)."""
    target_id: Optional[int] = None
    win: bool = False
    win_rule: str = ""                # RL1, RZ1, RG1, R10.2, R0.5, R5.3, CORNERED
    win_sub: str = ""                 # X2 sub-state suggested by the playbook (RUSH, DUEL, ...)
    lose_clauses: frozenset = frozenset()   # subset of "abcdefghi"
    lose_tracks: Dict[str, int] = field(default_factory=dict)  # clause -> track id (e, g, i)
    cornered: bool = False
    t_close: float = float("inf")     # 8.2.6 escape budget (lightning: time to close; ranged: CORNER_TTK_FRAC x ttk)
    cover_eta: Optional[float] = None # 8.2.6 escape ETA (cover point, else out of reach)


@dataclass(slots=True)
class Assessment:
    """assess.Assessor.update output (6.1 step 5). Read-only for everything downstream."""
    t: float = 0.0
    per_track: Dict[int, TrackAssess] = field(default_factory=dict)
    target_id: Optional[int] = None   # best-priority hostile mech with H3 hysteresis applied
    main_threat_id: Optional[int] = None   # highest DPS on us (AIM5/AIM6)
    hostiles: List[int] = field(default_factory=list)   # H: can damage us now (8.2.1)
    A: float = 0.0                    # vs target with all of H
    A_rush: Optional[float] = None    # PB_L closing (8.2.1)
    A_entry: float = 0.0              # vs target with all of H, engage assumptions (8.2.1)
    ttk_them_on_us_entry: float = float("inf")   # Σ H under the A_entry assumptions
    ttk_us_on_target: float = float("inf")
    ttk_them_on_us: float = float("inf")
    threats: Threats = field(default_factory=Threats)
    zone_eta: float = 0.0             # ETA to the safe set (field or fallback, x ZONE_ETA_SAFETY)
    zone_eta_settled: bool = False
    tau_edge: float = float("inf")
    slack: float = float("inf")       # 8.8
    terrain_call: str = "MIXED"       # R0.1
    desired_loadout: str = ""         # LOADOUT_RANK key
    loadout_rank: float = 0.0         # rank of what we hold under terrain_call
    playbook: str = "PB_G"


# ------------------------------------------------------------------ per-tick intent (plan section 6)
@dataclass(slots=True)
class LegRequest:
    """A desired leg direction from one level (owner candidates, 6.2/6.3)."""
    level: str                        # one of LEG_OWNER_ORDER
    rule: str                         # rule id for the trace, e.g. "RZ1", "R3.1"
    direction: Optional[Vec2]         # unit world-frame vector; None = stop
    speed: float = 1.0                # owner speed scale (1, SILENT_MOVE, ...)
    goal: Optional[Vec2] = None       # optional target point (follower / telemetry)
    juke_ok: bool = True              # False: straight dash / charging / RUSH inside LIGHTNING_RUSH_R
    is_cover_dash: bool = False
    nav: bool = False                 # direction from the navigation layer (path follower, climb, zone field,
                                      # stuck ladder): kept even when it fails the leg clearance probe


@dataclass(slots=True)
class LegFilter:
    """A hard leg filter (6.3 step 1). keep(dir, speed) -> bool; tiebreak(dir, speed) -> lower is better.
    scope "low": applies only when the leg owner is L5-L8/X4; "all": also under L2-L4 (O9 roof clause, L2)."""
    level: str                        # one of LEG_FILTER_ORDER
    rule: str
    keep: Callable[[Vec2, float], bool]
    tiebreak: Optional[Callable[[Vec2, float], float]] = None
    scope: str = "all"
    fallback_max_min: Optional[Callable[[Vec2, float], float]] = None  # L2: none kept -> maximize this


@dataclass(slots=True)
class JukeStyle:
    """O7 output: rotate the winning leg vector by side * angle if filters still pass."""
    side: int                         # -1 | +1
    angle: float                      # radians
    rule: str = "R12.1"


@dataclass(slots=True)
class AimRequest:
    """6.4: the request with the lowest prio number owns `aim`."""
    prio: int                         # AIM_PRIO value
    rule: str
    yaw: float
    pitch: float
    target_id: Optional[int] = None   # mech/NPC/crate id the aim is at (F1-F4 gates use it)
    tiebreak: float = 0.0             # within the same prio: lower wins


@dataclass(slots=True)
class FireRequest:
    """6.6: per slot. The arbiter applies the F-gates (aim.gun_gate / laser / lightning / mortar)."""
    slot: int
    rule: str                         # "F1".."F4", "F3.2", ...
    target_id: Optional[int] = None
    hold: bool = True                 # trigger held this decision (laser: keep charging)
    release: bool = False             # laser: release now (F2/R9.5/R9.6)


@dataclass(slots=True)
class BoostRequest:
    owner: str                        # one of BOOST_ORDER
    rule: str
    want: bool
    floor: float = 0.0                # energy floor for this owner (BOOST_FLOOR_*)


@dataclass(slots=True)
class JumpRequest:
    owner: str                        # one of JUMP_ORDER
    rule: str
    want: bool = True


@dataclass(slots=True)
class TakeRequest:
    writer: str                       # one of TAKE_ORDER
    rule: str
    take: Optional[int] = None        # pickup id
    drop: Optional[int] = None        # slot 0/1


@dataclass(slots=True)
class Intent:
    """Everything the decision layer wants this tick; arbiter.resolve(...) turns it into an action."""
    tick: int = 0
    t: float = 0.0
    macro: str = "X1"
    sub: str = ""                     # macro sub-state (RUSH, DUEL, KITE, COLLAPSE, ...)
    goal: Optional[str] = None        # active goal id inside X1
    rule: str = ""                    # deciding rule id of the leg owner (label/trace)
    legs: List[LegRequest] = field(default_factory=list)
    leg_filters: List[LegFilter] = field(default_factory=list)
    juke: Optional[JukeStyle] = None
    unstick: Optional[LegRequest] = None   # O5 replaces the owner's direction (still filtered by L2-L4)
    aim: List[AimRequest] = field(default_factory=list)
    fire: List[FireRequest] = field(default_factory=list)
    fire_self_defense_only: bool = False   # F5 phase/goal limit
    boost: List[BoostRequest] = field(default_factory=list)
    jump: List[JumpRequest] = field(default_factory=list)
    take: List[TakeRequest] = field(default_factory=list)
    reload: bool = False
    pick: Optional[int] = None
    pick_rule: str = ""
    trace: List[str] = field(default_factory=list)   # rule ids that fired this tick (telemetry)

    def label(self) -> str:
        g = f"/{self.goal}" if self.goal else ""
        s = f".{self.sub}" if self.sub else ""
        return f"{self.macro}{s}{g} {self.rule}".strip()


@dataclass(slots=True)
class RunnerInfo:
    """Set by runner.py on the bot (bot.runner_info) and updated every tick before act()."""
    hard_overruns: int = 0
    last_hard_overrun_tick: int = -1
    soft_overruns: int = 0
    last_soft_overrun_tick: int = -1
    exceptions: int = 0
    safe_until_t: float = -1.0        # X5 active while obs.t < safe_until_t
    last_compute_ms: float = 0.0
    init_late: bool = False           # __init__ missed its deadline; bot attached later
    last_source: str = ""             # who answered the last tick: act, safe:boot|T1|busy|exc|late


# ------------------------------------------------------------------ time budget helper (3.3)
class Clock:
    """Deadlines for one decision, measured from the reader thread's receipt time t0 (perf_counter s).

        clk = Clock(t0, C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS)
        sl = clk.slice(C.NAV_MS_CAP)          # min(now + cap, soft deadline)
        while work and not sl.expired(): ...
    """
    __slots__ = ("t0", "soft", "hard", "_now")

    def __init__(self, t0: float, soft_ms: float, hard_ms: float, now: Callable[[], float] = time.perf_counter):
        self.t0 = t0
        self.soft = t0 + soft_ms / 1000.0
        self.hard = t0 + hard_ms / 1000.0
        self._now = now

    def now(self) -> float:
        return self._now()

    def elapsed_ms(self) -> float:
        return (self._now() - self.t0) * 1000.0

    def soft_left_ms(self) -> float:
        return (self.soft - self._now()) * 1000.0

    def hard_left_ms(self) -> float:
        return (self.hard - self._now()) * 1000.0

    def soft_passed(self) -> bool:
        return self._now() >= self.soft

    def hard_passed(self) -> bool:
        return self._now() >= self.hard

    def slice(self, cap_ms: float, guaranteed_ms: float = 0.0) -> "Slice":
        """A sub-budget: ends at min(now + cap, soft deadline), but never before now + guaranteed_ms
        (ZONE_FIELD_MIN_MS) as long as the hard deadline allows."""
        now = self._now()
        end = min(now + cap_ms / 1000.0, self.soft)
        if guaranteed_ms > 0:
            end = max(end, min(now + guaranteed_ms / 1000.0, self.hard))
        return Slice(end, self._now)


class Slice:
    __slots__ = ("end", "_now")

    def __init__(self, end: float, now: Callable[[], float] = time.perf_counter):
        self.end = end
        self._now = now

    def expired(self) -> bool:
        return self._now() >= self.end

    def left_ms(self) -> float:
        return (self.end - self._now()) * 1000.0
