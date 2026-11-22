# Module contracts (mechfight/1 bot)

The interface every parallel agent codes against. Plan: `mechfight-kit/docs/game-plan-v2.md`
(section numbers below refer to it). Shared data types are real code in `contracts.py`; the module
APIs below are stubs: names, signatures and return types are binding, bodies are yours.

## 0. Ground rules

- Python 3.13 stdlib only (no 3.14-only features), 1 CPU, 512 MB, 50 ms per decision, read-only FS
  except `/tmp`, stdout = protocol only (the runner points `sys.stdout` at stderr; still, never print).
  Logs and telemetry go to stderr through `telemetry` (rate limited). Debug drawings: `label` only.
- Units / frames per plan section 0. Vectors are tuples (`Vec2`, `Vec3` in `contracts.py`). Times are
  match time `t` (`obs["t"]`); `*_until` fields are absolute. Ids are `int`.
- Config: `C = config.bind(hello["config"])`, then `C.NAME` for every section-10 constant. Raw config
  values: `C.path("weapons.laser.charge")` or `C.cfg[...]`. Live rows (`LIGHTNING_RANGE`,
  `JUKE_MIN_RANGE`, `SILENT_MOVE`, ...) are refreshed by `config.update_live(C, obs["self"])`, which
  `main.Bot.act` calls first every tick. `config.LOADOUT_RANK`, `config.UPGRADE_PRIORITY`,
  `config.loadout_key(kinds)` hold the 8.5 rank table and GP upgrade lists. Never hard-code a number
  that has a section-10 name.
- Every module constructor takes `(C, mapdata, ...)` and must be cheap (`Bot.__init__` has a
  deadline). Long work is sliced through `Clock`/`Slice` (section 3).
- No module raises out of its public per-tick call on bad input if it can return a fallback; when it
  cannot, raise and `main.Bot` catches it per stage (R15.1), counts it (`self._exceptions`) and uses the
  stage fallback. Never swallow silently: count it.
- Determinism: use only the `random.Random` passed in (seeded from `hello.seed`).

### File ownership

| Agent | Files |
| --- | --- |
| foundation | `Dockerfile`, `runner.py`, `main.py`, `config.py`, `safe.py`, `contracts.py`, `CONTRACTS.md`, `README.md`, `.dockerignore`, `tests/test_foundation.py`, `tools/gen_config.py` |
| world / assess | `world.py`, `assess.py` |
| map / nav | `mapdata.py`, `nav.py`, `tools/precompute.py`, `precomp/map.bin` |
| aim / telemetry | `aim.py`, `telemetry.py`, `upgrades.py` |
| decision layer | `overlays.py`, `modes.py`, `arbiter.py`, `tactics_*.py` |
| integration (later) | wires the pipeline into `main.py` at the marked INTEGRATION POINTs |

Only `contracts.py` is shared code. Need a new shared field? Add it to your own module first and
ask for it to be moved; do not edit `contracts.py` in parallel.

## 1. Runner and Bot (implemented: `runner.py`, `main.py`, `safe.py`)

```python
# main.py
class Bot:
    name: str                                    # ready name (<= 32 chars)
    runner_info: contracts.RunnerInfo            # set by the runner; read hard/soft overruns here (R9.6)
    def __init__(self, hello: dict, deadline: float) -> None: ...
        # deadline = perf_counter() seconds; return before it (cooperative checks between load steps);
        # the runner sends `ready` at the deadline anyway and answers with SAFE until __init__ returns
    def act(self, obs: dict, t0: float) -> dict: ...
        # t0 = reader-thread receipt time of this obs line (perf_counter). Return an Action dict
        # without `tick` (the runner sets it). Must return by C.HARD_DEADLINE_MS after t0, else the
        # runner sends SAFE's action and treats the late result as discarded (state changes persist).
    def observe(self, obs: dict, t0: float) -> None: ...   # called instead of act() during X5 SAFE (world update only)
    def take_exceptions(self) -> int: ...                  # per-stage exceptions since last call (runner T1 counter)
    def on_end(self, msg: dict) -> None: ...               # `end` hook (telemetry summary), < 0.5 s

# runner.py
def run(make_bot) -> int                         # make_bot(hello, deadline) -> Bot
def sanitize(action: dict, max_label=80, debug_full=False) -> dict   # schema-valid subset of an action
class Runner: ...                                # testable with injected inp/out streams

# safe.py  (X5 controller; pure, stateless, thread-safe)
def act(obs: dict, C) -> dict
def pick_upgrade(offer: dict | None, held_kinds: list[str], C) -> int | None
```

Runner guarantees: one answer per answered obs with `tick` echoed; newest queued obs only; a SAFE
answer on the same tick when `act()` raises or overruns `HARD_DEADLINE_MS`; X5 SAFE (T1) for
`SAFE_MODE_S` once `SAFE_MODE_EXC_N` exceptions land within `SAFE_MODE_EXC_WINDOW_S` (exceptions
escaping `act()`, late jobs' exceptions, and `take_exceptions()` counts all count; new ones re-arm);
`gc.freeze()` + `gc.disable()` after `__init__`, gen-0 collections only after the answer is out on
ticks under `GC_IDLE_MS`; actions are sanitized (finite numbers, 2-vectors, `fire` padded to 2, debug
reduced to `label` unless `TLM_DEBUG_CHANNEL`) and kept under `bot.max_line_bytes`.

Threads: reader, worker (runs `__init__`, `act`, `observe`), main. `act()` runs on the worker; the
main thread may run `safe.act` concurrently, so `safe.py` touches nothing but `obs` and `C`.

## 2. Pipeline (plan 6.1) as `main.Bot.act` will call it

```python
clock = Clock(t0, C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS)
config.update_live(C, obs["self"])
mapdata.new_tick()
tick   = world.update(obs, clock)                         # 1-2
intent = Intent(tick=tick.tick, t=tick.t)
intent.pick, intent.pick_rule = upgrades.pick(world)       # 3  (L1)
modes.strategic_refresh(world, assessor, nav, clock)       # 4  (R0 on next_changed, playbook, zone field start)
A      = assessor.update(world, nav, clock)                # 5
ostate = overlays.update(world, A, nav, intent, clock)     # 6  (O1-O9, urgent cover query)
modes.decide(world, A, nav, ostate, intent, clock)         # 7-8 (T1-T8 rows, goal selector, tactics_*)
action = arbiter.resolve(world, A, nav, intent, clock)     # 9
nav.anytime(world, A, clock)                               # 10 (slices until the soft deadline)
tlm.record(world, A, intent, action, clock)                # 11
action["debug"] = {"label": intent.label()}
```

As integrated: `upgrades.pick` runs right after the arbiter (it writes `action["pick"]`, so a failed arbiter still
answers the offer); `overlays.update(..., macro=modes.macro, sub=modes.sub, last_move=...)` gets the last sent
move, or `arbiter.blocked_desire` when the clearance probe forced "stop" (O5 input); main keeps
`arbiter.runner_info`, `arbiter.degrade_level` and `nav.degrade_level` current, issues the loot flood
(`nav.flood_request`) while loot is known, and exposes `bot.last_stage_ms`.

Fallbacks per stage (R15.1): world -> keep last state and advance time; assess -> last Assessment;
overlays -> none active; modes -> last intent's leg/aim owners; arbiter -> `safe.act(obs, C)`.

## 3. Clock / budget helper (`contracts.Clock`, implemented)

```python
class Clock:
    def __init__(self, t0: float, soft_ms: float, hard_ms: float, now=time.perf_counter): ...
    t0: float; soft: float; hard: float          # absolute perf_counter seconds
    def now(self) -> float
    def elapsed_ms(self) -> float
    def soft_left_ms(self) -> float; def hard_left_ms(self) -> float
    def soft_passed(self) -> bool;  def hard_passed(self) -> bool
    def slice(self, cap_ms: float, guaranteed_ms: float = 0.0) -> Slice
        # ends at min(now + cap, soft deadline); with guaranteed_ms, never before now + guaranteed
        # (capped by the hard deadline): ZONE_FIELD_MIN_MS while slack is short (3.3)
class Slice:
    end: float
    def expired(self) -> bool; def left_ms(self) -> float
```

Stage caps (3.3): `LOS_MS_CAP`, `COVER_URGENT_MS`, `ZONE_FIELD_MS` / `ZONE_FIELD_MIN_MS`,
`NAV_MS_CAP` (flood **or** A*, one per tick), `RANK_MS_CAP`. Check `expired()` at least every ~0.5 ms
of work (e.g. every 64 heap pops). Resumable work keeps its state in the module object.

## 4. Shared data types (`contracts.py`, implemented)

World (section 4): `WeaponState`, `SelfState`, `Track`, `Ghost`, `SoundRec`, `PickupMem`, `CrateMem`,
`CubeObs`, `DropState`, `DamageRec`, `HitRec`, `ShellPred`, `BulletThreat`, `ZoneState`
(`center_at(t)`, `radius_at(t)`), `OfferState`, `TickInfo`.
Map/nav: `Roof`, `ClimbFace`, `Path`, `Steer`, `ExitMemory` (`write/clear/until/prune`).
Assessment: `TrackAssess`, `Threats`, `FightEval`, `Assessment`.
Intent: `LegRequest` (`nav` flag), `LegFilter`, `JukeStyle`, `AimRequest`, `FireRequest`, `BoostRequest`,
`JumpRequest`, `TakeRequest`, `Intent`. Runner: `RunnerInfo`.
Order tables: `MACRO_ORDER`, `GOAL_ORDER`, `GOAL_ORDER_LOOT_WINDOW`, `LEG_OWNER_ORDER`,
`LEG_FILTER_ORDER`, `AIM_PRIO`, `BOOST_ORDER`, `JUMP_ORDER`, `TAKE_ORDER`.

Read the field comments in `contracts.py`; they are part of the contract.

## 5. Intent and how the arbiter resolves it (plan 6.2-6.7)

Producers append requests; nobody overwrites another producer's request. The arbiter is the only
code that turns an `Intent` into an action.

| Actuator | Producers | Resolution (arbiter) |
| --- | --- | --- |
| legs owner | `LegRequest(level=...)`: X5 safe (runner, not via Intent), L3/L3b (overlays O3/O4 cover dash), L4 (O1 zone steer), X4/L5/L6/L7/L8 (modes: exactly one of them) | owner = first level in `LEG_OWNER_ORDER` present; `d_own` = its direction; candidates = `d_own` + `IMPL_LEG_DIRS` compass dirs + stop; drop those blocked by `nav.clear_dir(pos, dir, LEG_PROBE_M)` unless all are; the owner candidate is kept when the request in force (unstick, else owner) has `nav=True` (a direction from the navigation layer: follower, climb, zone steer, ladder jump/jet) and its speed is >= `STUCK_MOVE_MIN` (a slower, silent move could never trigger O5, so it keeps the probe); the arbiter writes that verdict to `nav.leg_nav` (O5 pacing reads it), and O7 does not rotate a nav direction while a climb runs. A raw straight-line fallback (rule `8.9f`) is `nav=False`. `Modes.walk_to` and `Modes.zone_go` write `nav=True`; own vectors (away, kite, perp, juke) do not |
| leg filters | `LegFilter` L2 (O2), L4 (O1, only when owner is not L3/L3b and no L2), O8, O9 (`scope="all"` roof clause, `scope="low"` circle clause) | apply in `LEG_FILTER_ORDER`; a filter that would remove every candidate is ignored (L2: keep the `fallback_max_min` argmax); `scope="low"` filters only when owner is X4/L5-L8; rank survivors by 6.3 step 2 score; ties: closest to `d_own`, then lower compass index; L2 tie-break = its `tiebreak` (zone time-to-safety) |
| unstick | `intent.unstick` (O5, from `nav.stuck_update`) | replaces the owner's direction before filtering (still filtered by L2-L4) |
| juke | `intent.juke` (O7) | rotate the winner by `side*angle`; re-check filters; fail -> no juke. Skipped when owner request has `juke_ok=False` |
| speed | owner's `LegRequest.speed` | `move = dir * speed` |
| aim | `AimRequest(prio=AIM_PRIO[...])` from overlays (AIM1, AIM2, AIM4, AIM7, AIM8), modes/tactics (AIM3, AIM5, AIM6, AIM9, X4 list), arbiter's own AIM10 scan if nothing else | lowest `prio`, then lowest `tiebreak`, then first appended |
| fire | `FireRequest(slot, rule, target_id, hold, release)` from tactics/modes/overlays (F3.2) | per slot: gates via `aim` (F1 gun gate, F2 laser start/release, F3 lightning gate, F4 mortar gate); `intent.fire_self_defense_only` (F5) drops mech targets that did not damage us within `SELF_DEFENSE_S`; laser R9.6 safe window uses `bot.runner_info` |
| reload | `intent.reload` or F6 (arbiter computes F6 itself) | `reload: true` |
| boost | `BoostRequest(owner in BOOST_ORDER, want, floor)` | first owner in `BOOST_ORDER` that has a request decides, except that every request (want True or False) from an owner other than O6 / CLIMB is skipped while `self.grounded` is false (in the air boost is a jet, not a sprint); `want and energy > floor`; O9 veto below `COAST_BOOST_MIN_ENERGY` unless owner in (O1, O6, CLIMB) |
| jump | `JumpRequest(owner in JUMP_ORDER)` | first owner present |
| take/drop | `TakeRequest(writer in TAKE_ORDER, take, drop)` | first writer present; never `drop` alone and `take` on one tick; `take` only within `loot.pickup_radius` (h) and 4 m (v) and the replaced slot is not a charging laser |
| pick | `intent.pick` (upgrades, L1) | copied |
| label | `intent.macro/sub/goal/rule` | `intent.label()` -> `debug.label` |

D1 (`ZONE_OUTRANKS_DODGE="collapse"`): only when `t >= C.FINAL_SHRINK_T` does overlays emit the L4
filter ahead of L2 (it sets `LegFilter(level="L4", ...)` and omits L2's hard filter if the dodge would
leave the safe set). The arbiter does not special-case it.

## 6. Module APIs (stubs; bodies owned by the agents in section 0)

### 6.1 `mapdata.py` (sections 4.6, 11)

```python
NAV_STEP = 4.0                      # m per nav4 cell
NAV_DIRS: tuple[tuple[int, int], ...]   # 8 neighbor offsets (di, dj) in edge-bit order: E, NE, N, NW, W, SW, S, SE
NAV_WALKABLE, NAV_ROOF, NAV_DECK, NAV_UNDERPASS, NAV_EDGE = 1, 2, 4, 8, 16   # nav4 flag bits
COVER_WALL_MASK, COVER_CORNER, COVER_WALKABLE = 0x3F, 0x40, 0x80             # cover2 bits

class MapData:
    degraded: bool                      # precompute missing / hash mismatch / boot deadline (11.3)
    degraded_reason: str
    bounds: tuple[float, float, float, float]   # min_x, min_y, max_x, max_y
    roofs: list[Roof]                   # roofs[0] is a placeholder (foot_id 0 = none)
    climbs: list[ClimbFace]
    nx4: int; ny4: int; min_x4: float; min_y4: float

    @classmethod
    def load(cls, map_dir: str, C, deadline: float) -> "MapData": ...   # never raises; degraded on any failure
    def new_tick(self) -> None: ...                       # clear one-tick LOS cache
    # heights
    def terrain_h(self, x: float, y: float) -> float: ...     # bilinear terrain.i16
    def surface_h(self, x: float, y: float) -> float: ...     # 1 m: max(terrain, roof tops) (11.2)
    def ground_under(self, x: float, y: float, z: float) -> float: ...  # highest walkable surface <= z + STEP_UP (decks)
    # LOS (4.6): DDA on surface_h + exact bridge/overhang prisms, trees ignored
    def los(self, a: Vec3, b: Vec3) -> bool: ...
    def los_cached(self, key: tuple, a: Vec3, b: Vec3) -> bool: ...   # per-tick cache, key e.g. (track_id, cell_a, cell_b)
    # 1 m grids
    def clearance(self, x: float, y: float) -> float: ...     # clear_dm in m (0 = blocked)
    def roof_id(self, x: float, y: float) -> int: ...         # foot_id, 0 = none
    # 4 m nav grid
    def cell4(self, x: float, y: float) -> int: ...           # index j*nx4+i, -1 outside
    def cell4_xy(self, idx: int) -> Vec2: ...                 # cell center
    def nav_h(self, idx: int) -> float: ...
    def nav_flags(self, idx: int) -> int: ...
    def walk_edges(self, idx: int) -> int: ...                # bit k = NAV_DIRS[k] walkable
    def jump_edges(self, idx: int) -> int: ...
    def underpass_clear(self, idx: int) -> float: ...         # m, 0 = none
    # 5 m and 2 m grids
    def sight5(self, x: float, y: float) -> float: ...        # median sight distance, m
    def sight5_dir(self, x: float, y: float, sector: int) -> float: ...   # sector 0..7, 45 deg each from east CCW
    def wall5(self, x: float, y: float) -> float: ...         # distance to nearest wall, m
    def cover2(self, x: float, y: float) -> int: ...          # raw bits
    def cover2_cells(self, x: float, y: float, r: float) -> list[tuple[float, float, int]]: ...  # nearest first
```

Degraded mode still provides every method (coarser data); `sight5*` return `float("nan")` when absent
and callers treat NaN as unknown.

### 6.2 `world.py` (section 4)

```python
class WorldModel:
    def __init__(self, C, mapdata: MapData, slot: int) -> None: ...
    def update(self, obs: dict, clock: Clock) -> TickInfo: ...   # 6.1 steps 1-2; LOS within LOS_MS_CAP
    # state (read-only for other modules)
    tick: TickInfo
    me: SelfState
    tracks: dict[int, Track]            # mechs and NPCs; dropped TRACK_FORGET_S after last_seen_t (NPCs kept)
    ghosts: dict[int, Ghost]            # negative ids
    sounds: list[SoundRec]              # this tick, after association
    beams: list[dict]                   # raw obs.beams this tick
    events: list[dict]                  # raw obs.events this tick
    pickups: dict[int, PickupMem]
    crates: dict[int, CrateMem]
    cubes: dict[int, CubeObs]           # visible this tick only
    drops: dict[int, DropState]
    zone: ZoneState
    offer: OfferState | None
    damage_log: collections.deque[DamageRec]   # last 30 s
    hits: collections.deque[HitRec]            # last 30 s
    shells: list[ShellPred]
    bullets: list[BulletThreat]         # gun bullets inbound (juke trigger)
    npc_bullets: list[BulletThreat]
    mortar_active_until: float
    kills: int
    loot_bursts: list[tuple[Vec2, float]]      # (point, t) from death sounds / kills
    # queries
    def held_kinds(self) -> tuple[str, ...]: ...
    def visible_mechs(self) -> list[Track]: ...         # kind == "mech" and visible
    def hostile_tracks(self, max_age_s: float) -> list[Track]: ...
    def damage_since(self, t: float, by: int | None = None) -> float: ...
    def damaged_us_recently(self, track_id: int, window_s: float) -> bool: ...
    def heat_at(self, x: float, y: float) -> float: ...  # danger heat (4.3 step 7), decayed to now
    def seen_at(self, x: float, y: float) -> bool: ...   # 5 m seen-cells bitmap (4.4)
```

`world.update` also: playbook (5.1) into `me.playbook`; `me.short_until`, `me.equip_until` from
events; `tick.missed_ticks` from the tick gap; `zone.next_changed`; ghost creation/merge/expiry;
damage inference creates tracks (`est=True`); per-track `lightning_range_mult` (A-20).

### 6.3 `assess.py` (section 8.1-8.2, 8.5, R0)

```python
def calibrate_rush_a_min(C) -> None: ...      # sets C.RUSH_A_MIN at hello (8.2.1 canonical case)
def dps(kind: str, wstats: dict, d: float, shooter_cubes: int, target_cubes: int, *, rear: bool,
        armor: float, energy: float | None, C, enemy: bool) -> float: ...   # 8.1 one weapon
def loadout_rank(kinds, terrain_call: str) -> float: ...                     # 8.5 via config.LOADOUT_RANK
def slot_to_replace(held: list[str | None], new_kind: str, terrain_call: str, C) -> tuple[int | None, float]: ...
    # (slot or None, gain) per 8.5 with SWAP_RULE

class Assessor:
    def __init__(self, C, mapdata: MapData) -> None: ...
    def update(self, world: WorldModel, nav: "Nav", clock: Clock) -> Assessment: ...
    def terrain_call(self, world: WorldModel) -> tuple[str, str]: ...   # R0.1 -> (call, desired loadout); on next_changed
    def evaluate_fight(self, world: WorldModel, A: Assessment, macro: str, target_id: int | None,
                       ctx: dict) -> FightEval: ...
        # ctx keys: "exit_memory": ExitMemory, "committed_capture": bool, "neighbor": int | None,
        #           "x2_since": float | None, "x3_since": float | None, "alive": int,
        #           "cornered_active": bool, "sub": str (X2 sub-state; LOSE (i) rush exemption, UQ-2 KEEP)
    def a_vs(self, world: WorldModel, track_id: int, hostiles: list[int] | None = None) -> float: ...
```

### 6.4 `nav.py` (sections 7.13, 8.3, 8.4, 8.8, 8.9)

```python
class Nav:
    def __init__(self, C, mapdata: MapData) -> None: ...
    # zone time-to-safety field (8.8)
    def zone_field_start(self, zone: ZoneState, outside_current: bool) -> None: ...
    def zone_eta(self, pos: Vec3) -> tuple[float, bool]: ...    # (eta_s incl. ZONE_ETA_SAFETY, settled)
    def zone_descent(self, pos: Vec3) -> Vec2 | None: ...       # steepest-descent direction (fallback per 8.8); pure query, no climb / jump
    def zone_steer(self, me: SelfState, t: float | None = None) -> Steer: ...           # zone_descent as a full Steer for leg owners that must get in: runs / begins a climb (its approach waits, rule R13.2w, direction None, until the energy covers it; never inside a ladder episode), jumps a JUMP edge; cached per tick on the world time `t` (callers that run before the first world sync pass it); a zone climb its caller stops stepping for ZONE_CLIMB_IDLE_S is dropped without blocking its face
    def zone_field_done(self) -> bool: ...
    #   grown mech (R12.3): a tight cell (centre within radius-2.5+1 m of a wall) is walked into in a closed search only along a tight chain rooted at the search origin (a search that starts on open ground never walks into one; a climb landing may be tight and is left onto open ground; a closed zone field also enters a tight seed cell), in every search except the widened zone field and the A* retry below, which also enter it from open ground at SIZE_TIGHT_ENTER_COST_S (tight to tight: SIZE_TIGHT_COST_S); status 2 (low underpass) is closed in all of them
    # loot flood (8.3)
    def flood_request(self, origin: Vec3, radius: float, cost: dict) -> None: ...      # closed mode: never enters a tight cell from open ground
    def travel_time(self, x: float, y: float) -> float | None: ...   # latest complete flood; None = unknown
    # A* (8.9): resumable, keyed
    def path_request(self, key: str, start: Vec3, goal: Vec2, cost: dict, t: float) -> None: ...   # closed mode first; a grown mech whose search ends short of the goal is searched once more with tight cells open (stats["tight_retry"])
    def path(self, key: str) -> Path | None: ...                 # complete or partial
    def follow(self, me: SelfState, key: str | None, goal: Vec2 | None) -> Steer: ...   # pure pursuit / interim steering
    def clear_dir(self, pos: Vec3, direction: Vec2, dist: float) -> bool: ...          # clearance probe (6.3); height-aware: ground below pos.z counts as at pos.z
    # stuck ladder (R13.1, O5)
    def stuck_update(self, world: WorldModel, commanded: Vec2 | None) -> tuple[LegRequest | None, JumpRequest | None, BoostRequest | None]: ...
    #   trigger: commanded and slow (STUCK_T), or commanded, `nav.leg_nav` (the winning direction is a navigation one) and under STUCK_PACE_M net over STUCK_PACE_S (pacing);
    #   jump / jet LegRequests are nav=True; WEDGE_REPEAT_N episodes that end within WEDGE_MOVE_M of where the first began -> one "lift" episode per spot (a later episode there is the normal ladder with its escape) (jump + CLIMB jet up past the highest surface around, toward the most open side, WEDGE_JET_MAX_S)
    def block_edge(self, a: Vec3, b: Vec3, ttl_s: float, t: float) -> None: ...
    # spots and cover (8.4)
    def cover_query(self, me: SelfState, threats: list[Vec3], budget_s: float, sl: Slice) -> Vec2 | None: ...  # urgent; closed mode: a standing-in-the-open mech never gets a tight-cell cover point
    def best_spot(self, kind: str, world: WorldModel, A: Assessment) -> tuple[Vec2, float, int] | None: ...
        # kind in {"cover", "hold", "roof", "nook"}; (point, score, roof_id or 0) from the ranking slices
    def spot_valid(self, kind: str, point: Vec2, world: WorldModel, A: Assessment) -> bool: ...
    # anytime work (6.1 step 10): zone field slice (own budget) then one of flood / A* (NAV_MS_CAP), then ranking (RANK_MS_CAP)
    def anytime(self, world: WorldModel, A: Assessment, clock: Clock) -> None: ...
```

`cost` dict keys (8.9): `"exposure_tracks": list[Track]`, `"heat": bool`, `"zone": bool`,
`"silent": bool`, `"energy": float` (climb feasibility), `"reserve": float`.

### 6.5 `aim.py` (section 8.7, 6.6)

```python
def wrap(a: float) -> float: ...
def gun_lead(me: SelfState, tr: Track, gun: WeaponState, C) -> tuple[float, float, float]: ...   # yaw, pitch, tof
def gun_gate(me: SelfState, tr: Track, gun: WeaponState, aim_target: tuple[float, float], C) -> bool: ...   # F1
def laser_point(me: SelfState, tr: Track, C) -> tuple[float, float]: ...
def laser_decision(me: SelfState, tr: Track | None, laser: WeaponState, los_now: bool, los_breaks_next: bool,
                   safe_window_ok: bool, C) -> tuple[bool, bool]: ...   # (hold_trigger, release_now): F2/R9.5/R9.6
def lightning_gate(world: WorldModel, mapdata: MapData, slot: int, C, allow_npc: bool = False) -> tuple[bool, int | None]: ...  # F3
def mortar_solution(me: SelfState, mapdata: MapData, target: Vec3, target_vel: Vec3, mortar: WeaponState,
                    C) -> tuple[float, float, Vec3] | None: ...   # yaw, pitch, impact
def mortar_gate(me: SelfState, solution, target_pred: Vec3, mortar: WeaponState, C) -> bool: ...  # F4
class Scanner:                       # AIM10 / AIM8 look timing (H11)
    def __init__(self, C, rng) -> None: ...
    def scan(self, world: WorldModel, mapdata: MapData) -> tuple[float, float]: ...
    def look(self, world: WorldModel) -> AimRequest | None: ...
```

### 6.6 `upgrades.py` (R14.1)

```python
class UpgradePicker:
    def __init__(self, C) -> None: ...
    def pick(self, world: WorldModel) -> tuple[int | None, str]: ...  # (choice index, rule); resends per OFFER_RESEND_TICKS
```

### 6.7 `telemetry.py` (section 14.4)

```python
class Telemetry:
    def __init__(self, C, slot: int, seed: int) -> None: ...
    def event(self, kind: str, **fields) -> None: ...        # one JSONL record (rate limited TLM_MAX_LINES_PER_S; summaries never dropped)
    def record(self, world, A, intent: Intent, action: dict, clock: Clock, stage_ms: dict[str, float] | None = None) -> None: ...
    def end_summary(self, msg: dict, runner_info: RunnerInfo) -> None: ...
```

stderr only; one JSON object per line with `"ev"`, `"t"`, `"tick"`. Never writes to debug drawings.

### 6.8 `overlays.py` (section 5.4, 6.2)

```python
@dataclass
class OverlayState:
    active: dict[str, float]            # overlay id ("O1".."O9") -> since t
    cover_target: Vec2 | None           # L3/L3b cover cell
    zone_escape: bool
    keep_away_contacts: list[int]

class Overlays:
    def __init__(self, C, mapdata: MapData, nav: "Nav", rng) -> None: ...
    def update(self, world, A: Assessment, nav, intent: Intent, clock: Clock, macro: str = "X1") -> OverlayState: ...
        # appends: LegFilter L2/L4/O8/O9, LegRequest L3/L3b/L4, AimRequest AIM1/AIM2/AIM4/AIM7/AIM8,
        # BoostRequest O1/O3/O4/O6/L2, intent.juke (O7), intent.unstick (O5 via nav.stuck_update).
        # Uses the previous tick's macro for scope rules (X3 disables O4 legs; X4 scope).
```

### 6.9 `modes.py` (sections 5, 9) and `tactics_*.py`

```python
class Modes:
    macro: str; sub: str; goal: str | None
    exit_memory: ExitMemory
    def __init__(self, C, mapdata: MapData, nav: "Nav", rng) -> None: ...
    def strategic_refresh(self, world, assessor, nav, clock: Clock) -> None: ...   # R0 on next_changed, hold-spot invalidation, zone field start
    def decide(self, world, A: Assessment, nav, ostate: OverlayState, intent: Intent, clock: Clock) -> None: ...
        # T1 is the runner's; rows T2-T8 then the goal selector (G1 stickiness, G2 reach override),
        # writes exactly one owner LegRequest (X4 or L5-L8), AIM3/5/6/9 aims, FireRequests,
        # BoostRequest("MODE"), JumpRequest("CLIMB"/"ROOF_ASSAULT"), TakeRequest, intent.macro/sub/goal/rule

@dataclass
class TacticCtx:
    C: object; world: object; A: Assessment; nav: object; mapdata: MapData
    intent: Intent; clock: Clock; modes: Modes; rng: object

# each tactics module exposes one entry point; returns the deciding rule id
def run(sub: str, ctx: TacticCtx) -> str: ...
# tactics_lightning: RUSH, BRAWL, STALL (RL*)      tactics_laser: DUEL (RZ*)       tactics_gun: DUEL_GUN (RG1)
# tactics_drop: APPROACH, WATCH, CAPTURE (R8)      tactics_roof: ROOF_HOLD, ROOF_ASSAULT (R10, RZ4)
# tactics_endgame: FINAL_APPROACH, FINAL_HOLD, COLLAPSE (R11)
```

`OverlayState` and `TacticCtx` live in their owners' modules (`overlays.py`, `modes.py`).

### 6.10 `arbiter.py` (section 6.3-6.7)

```python
class Arbiter:
    def __init__(self, C, mapdata: MapData) -> None: ...
    def resolve(self, world, A: Assessment, nav, intent: Intent, clock: Clock) -> dict: ...  # Action without tick
    last_action: dict
```

## 7. Config derivations (choices made where the plan was ambiguous)

`config.py` implements one explicit formula per DERIVED row (`_derived`); live rows in
`update_live`. Readings chosen:

1. Phase times use "the last phase" for `phases[5]` and `ENDGAME_T = Σ phases[0..n-2]`; `PHASE_T3/T4`
   use fixed indices 2/3 (clamped to the phases present). Empty phases -> table defaults kept.
2. `SOUND_BAND_REP`: near = 0.75 × `sound.near`, mid = (near + mid)/2, far = `sound.mid` + `sound.near`
   (reproduces 45 / 110 / 220 from 60 / 160).
3. `JUKE_MAX_RANGE` = `gun.range` × the Rail barrel range mul read from `upgrades[gun_velocity]`
   (1.15 fallback).
4. `LASER_HOLD_MIN` = max(75 (GP), `JUKE_MIN_RANGE`) at bind and live; 75.04 at 0 cubes.
5. `NPC_GUN_MIN_AMMO` = ceil(`npc.hp` / `gun.damage`) = 8.
6. `RECOVER_MAX_S` = 29.9 (formula), not the rounded 30 in the table.
7. `FINAL_POINT_TOL_M` and `AB_ACCEPT_ONE` are DERIVED in name only (no formula over config): table
   values kept.
8. `RUSH_A_MIN` = None after bind; `assess.calibrate_rush_a_min(C)` sets it at hello (it needs the DPS model).
9. `ENEMY_LIGHTNING_RANGE_MULT` = 1.0 globally; the per-track value is `Track.lightning_range_mult`.
10. Live rows: `LIGHTNING_RANGE` / `LIGHTNING_MIN_ENERGY` from our held lightning's stats (base value
    when not held); `MORTAR_SELF_SAFE_M` uses our held mortar's `splash_radius` (Cluster) and
    `stats.radius`; `MORTAR_DODGE_R` uses base splash × `MORTAR_ENEMY_SPLASH_MULT` + `stats.radius`;
    `ROOF_EDGE_BACKOFF` = 6.5 × `s_self`; `COAST_BOOST_MIN_ENERGY` = `stats.boost_drain` × `coast_s`;
    `CLIMB_COST_PER_M` from stats; `CLIMB_MAX_H` uses `max_energy`; `LASER_HOLD_MAX` and
    `NEIGHBOR_CLEAR_R` = `vision.range + stats.vision_bonus`; `SILENT_MOVE` from `stats.walk_speed`.
    `C.base[name]` keeps the bind-time value.
11. `UPGRADE_FALLBACK` is parsed as the id list (plating ... nanites); "then held-weapon upgrades in GP
    order, mortar upgrades last" is behavior for `upgrades.py`.
12. `HIT_GUN` / `HIT_MORTAR` are lists of `(d, p)` points (interpolate linearly, clamp at the ends);
    `PRIOR_SECOND` / `SOUND_BAND_REP` are dicts.
13. `LOADOUT_RANK` keys use letters G (gun), Z (laser), L (lightning), M (mortar) in that order
    (`config.loadout_key`); "L alone, Z alone" -> keys `L` and `Z`.
14. `UPGRADE_PRIORITY` comes from GP's "Upgrades by loadout" table (keys GL, GZ, ZL, MECH), names
    mapped to ids via the recorded hello. LM = ZL list without `laser_damage` (GP note; in `safe.py`
    and expected in `upgrades.py`).

## 8. Regenerating config

`python3 tools/gen_config.py` rewrites only the block between the GENERATED markers in `config.py`
from plan section 10 (+ 8.5 rank table, GP upgrade table, IMPL_* paragraph); `--check` fails if stale
(also a unit test). User decisions of 16.1 are asserted (and forced) by the generator.
