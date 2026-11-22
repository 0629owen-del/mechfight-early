"""Mech Royale bot entry point: `python3 -u /bot/main.py` (Dockerfile ENTRYPOINT).

Bot(hello, deadline) binds the config, loads the precompute and builds every pipeline module with
cooperative deadline checks (X0); act(obs, t0) runs the per-tick pipeline of plan 6.1 / CONTRACTS 2:

    world -> upgrades pick (L1) -> strategic refresh -> assess -> overlays (reflexes, urgent cover)
    -> modes (rows, goals, tactics_*) -> arbiter (actuators) -> nav anytime (zone field / flood / A* /
    ranking, until SOFT_DEADLINE_MS) -> telemetry -> label

Every stage runs in its own try/except (R15.1): a failure is counted (take_exceptions -> runner T1)
and the stage fallback is used. Budget (3.3): optional work only while the soft deadline has not
passed; past HARD_DEADLINE_MS the current action goes out at once. R15.3 degrade levels follow the
compute-time window (p95 > soft or any hard-late tick).
"""
from __future__ import annotations

import os
import random
import sys
import time
import traceback
from collections import deque
from typing import Any, Dict, Optional

sys.dont_write_bytecode = True  # read-only image (also PYTHONDONTWRITEBYTECODE=1 in the Dockerfile)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
import runner  # noqa: E402
import safe  # noqa: E402
from contracts import Assessment, Clock, Intent, RunnerInfo  # noqa: E402

_STAGES = ("world", "pick", "refresh", "assess", "overlays", "modes", "arbiter", "anytime", "telemetry")
_MAP_FILE = os.path.join("v2", "world.json")


def _map_dir(hello_dir: Optional[str]) -> str:
    """hello.map.dir, else a known arena location (the conformance checker sends a path relative to the
    kit; unit tests run from the bot directory)."""
    cands = [hello_dir or "/arena", "/arena", os.path.join(HERE, "..", "mechfight-kit", "arena")]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, _MAP_FILE)):
            return d
    return hello_dir or "/arena"


class Bot:
    name = "dogma"

    def __init__(self, hello: Dict[str, Any], deadline: float):
        """Returns before `deadline` (perf_counter s): every load step checks the time left; missing
        modules leave the bot on the SAFE controller (plan X0, 11.3)."""
        self.deadline = deadline
        self.hello = hello
        self.C = C = config.bind(hello.get("config"))
        self.slot = int(hello.get("slot", 0) or 0)
        self.seed = int(hello.get("seed", 0) or 0)
        self.rng = random.Random(self.seed)
        self.map_dir = _map_dir((hello.get("map") or {}).get("dir"))
        self.runner_info = RunnerInfo()  # replaced by the runner's shared instance
        self._exceptions = 0             # per-stage exceptions caught inside act() (runner counts them)
        self.stage_exc: Dict[str, int] = {s: 0 for s in _STAGES}
        self.last_tick: Optional[int] = None
        self.degraded_nav = True
        self.ready = False               # full pipeline available (else SAFE)
        self.map = self.world = self.nav = self.assess = self.overlays = None
        self.modes = self.arbiter = self.upgrades = self.tlm = None
        self._A = Assessment()
        self._last_intent: Optional[Intent] = None
        self._last_action: Dict[str, Any] = {}
        # R15.3 degrade state
        self.degrade_level = 0
        self._win: deque = deque()
        self._hard_seen = 0
        self._l2_hard = 0                # hard-late ticks at L2 in the current window (R15.3 L3 gate)
        self._prev_late = False          # the previous act() was itself over the hard deadline
        self._ticks = 0
        self.boot_ms: Dict[str, float] = {}
        self.last_stage_ms: Dict[str, float] = {}   # per-stage ms of the last act() (tools/bench.py)
        t_boot = time.perf_counter()
        # Baseline build: BOT_SAFE_ONLY=1 (one ENV line in the Dockerfile) plays the whole match on the
        # SAFE controller, so a submission or a minisim batch shows what the full pipeline adds.
        if os.environ.get("BOT_SAFE_ONLY") == "1":
            runner.log("BOT_SAFE_ONLY=1: SAFE controller only (baseline build)")
        else:
            try:
                self._boot(C)
            except Exception:
                runner.log(f"boot failed (SAFE controller only):\n{traceback.format_exc()}")
                self.ready = False
        self.boot_ms["total"] = (time.perf_counter() - t_boot) * 1000.0
        if self.tlm is not None:
            self.tlm.event("boot", _force=True, ready=self.ready, map_dir=self.map_dir,
                           degraded=bool(getattr(self.map, "degraded", True)),
                           reason=getattr(self.map, "degraded_reason", ""),
                           rush_a_min=C.RUSH_A_MIN, ms={k: round(v, 1) for k, v in self.boot_ms.items()})

    def _step(self, name: str, fn, margin_s: float = 0.05):
        if self._time_left() < margin_s:
            raise TimeoutError(f"boot deadline before {name}")
        t = time.perf_counter()
        r = fn()
        self.boot_ms[name] = (time.perf_counter() - t) * 1000.0
        return r

    def _boot(self, C) -> None:
        import mapdata
        import telemetry
        self.tlm = self._step("telemetry", lambda: telemetry.Telemetry(C, self.slot, self.seed))
        self.map = self._step("mapdata", lambda: mapdata.MapData.load(self.map_dir, C, self.deadline - 0.5))
        self.degraded_nav = bool(self.map.degraded)
        import assess
        self._step("rush_cal", lambda: assess.calibrate_rush_a_min(C))       # 8.2.1 (needs bound C)
        import world
        import nav
        import overlays
        import modes
        import arbiter
        import upgrades
        self.world = self._step("world", lambda: world.WorldModel(C, self.map, self.slot))
        self.nav = self._step("nav", lambda: nav.Nav(C, self.map, random.Random(self.seed + 1)))
        self.assess = self._step("assess", lambda: assess.Assessor(C, self.map))
        self.overlays = self._step("overlays", lambda: overlays.Overlays(C, self.map, self.nav,
                                                                         random.Random(self.seed + 2)))
        self.modes = self._step("modes", lambda: modes.Modes(C, self.map, self.nav, random.Random(self.seed + 3)))
        self.arbiter = self._step("arbiter", lambda: arbiter.Arbiter(C, self.map, rng=random.Random(self.seed + 4)))
        self.upgrades = self._step("upgrades", lambda: upgrades.UpgradePicker(C))
        self._OverlayState = overlays.OverlayState
        self._base_los = float(C.base.get("LOS_MS_CAP", C.LOS_MS_CAP))
        self._base_rank = float(getattr(self.nav, "_RANK_MS_CAP", C.RANK_MS_CAP))
        self.ready = True

    def _time_left(self) -> float:
        return self.deadline - time.perf_counter()

    # -------------------------------------------------------------- runner hooks
    def take_exceptions(self) -> int:
        """Per-stage exceptions since the last call (runner adds them to its T1 counter)."""
        n, self._exceptions = self._exceptions, 0
        return n

    def _fail(self, stage: str) -> None:
        self._exceptions += 1
        self.stage_exc[stage] = self.stage_exc.get(stage, 0) + 1
        n = self.stage_exc[stage]
        if n <= 3 or n % 100 == 0:
            runner.log(f"stage {stage} failed at tick {self.last_tick} ({n} so far):\n{traceback.format_exc()}")
        if self.tlm is not None:
            self.tlm.count("exc_" + stage)

    def _module_exceptions(self) -> None:
        """World/assess swallow sub-step errors behind their own fallbacks: they count as stage failures
        (R15.1). Nav keeps its own fallbacks (straight-line interim steering): telemetry only."""
        for stage, mod in (("world", self.world), ("assess", self.assess)):
            try:
                n = int(mod.take_exceptions() or 0)
            except Exception:
                n = 0
            if n:
                self._exceptions += n
                self.stage_exc[stage] = self.stage_exc.get(stage, 0) + n
                if self.tlm is not None:
                    self.tlm.count("exc_" + stage, n)
        try:
            n = int(getattr(self.nav, "exceptions", 0) or 0)
            if n:
                self.nav.exceptions = 0
                if self.tlm is not None:
                    self.tlm.count("exc_nav_internal", n)
        except Exception:
            pass

    def observe(self, obs: Dict[str, Any], t0: float) -> None:
        """Called by the runner instead of act() while X5 SAFE is active: keep the world model fresh."""
        config.update_live(self.C, obs.get("self"))
        self.last_tick = obs.get("tick")
        if not self.ready:
            return
        try:
            self.world.update(obs, Clock(t0, self.C.SOFT_DEADLINE_MS, self.C.HARD_DEADLINE_MS))
        except Exception:
            self._fail("world")
        self._module_exceptions()

    # -------------------------------------------------------------- R15.3
    def _degrade_update(self, el_ms: float) -> None:
        C = self.C
        n = int(C.DEGRADE_WINDOW_N)
        late = el_ms > C.HARD_DEADLINE_MS
        ri = self.runner_info
        replaced = ri is not None and ri.hard_overruns > self._hard_seen   # late answers the runner replaced
        if replaced:
            self._hard_seen = ri.hard_overruns
        # The runner counts a late act() after that act() has already been counted here by its own time,
        # so its count arrives one call later: one late tick must not count as two (the L3 gate below).
        hard = late or (replaced and not self._prev_late)
        self._prev_late = late
        self._win.append(el_ms)
        old = self.degrade_level
        # L1 and L2 only cut search budgets; L3 hands the legs to SAFE (straight to the zone center), so it
        # takes DEGRADE_L3_HARD_N hard-late ticks at L2 inside one window, and is left after
        # DEGRADE_L3_EXIT_N clean ticks instead of a full window.
        need = int(C.DEGRADE_L3_EXIT_N) if old >= 3 else n
        if hard:
            if old < 2:
                self.degrade_level = old + 1
                self._l2_hard = 0
            elif old == 2:
                self._l2_hard += 1
                if self._l2_hard >= int(C.DEGRADE_L3_HARD_N):
                    self.degrade_level = 3
                    self._l2_hard = 0
            self._win.clear()
        elif len(self._win) >= need:
            s = sorted(self._win)
            p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
            if p95 > C.SOFT_DEADLINE_MS:
                self.degrade_level = min(2, old + 1)    # a soft overrun never holds L3: only hard ticks do
                self._win.clear()
            elif old > 0:
                self.degrade_level = old - 1
                self._win.clear()
            else:
                self._win.popleft()
            self._l2_hard = 0               # the L3 count does not outlive a window
        if self.degrade_level != old:
            self._apply_degrade()
            if self.tlm is not None:
                self.tlm.event("degrade", _force=True, level=self.degrade_level, frm=old, el_ms=round(el_ms, 2))

    def _apply_degrade(self) -> None:
        """L1+: LOS cap x DEGRADE_SCALE (flood radius in _flood); the arbiter blocks laser starts / long holds."""
        C = self.C
        lv = self.degrade_level
        C.LOS_MS_CAP = self._base_los * (C.DEGRADE_SCALE if lv >= 1 else 1.0)
        self.arbiter.degrade_level = lv
        self.nav.degrade_level = lv

    def _flood(self, world) -> None:
        """8.3: loot flood while targets exist. nav.flood_request reuses the running / fresh flood unless we
        moved > 2 cells or NAV_FLOOD_REFRESH_S passed, so travel times stay relative to where we are.
        R15.3: radius x DEGRADE_SCALE at L1, no flood from L2."""
        if self.degrade_level >= 2 or not (world.pickups or world.crates or world.cubes):
            return
        C, me = self.C, world.me
        r = float(C.NAV_FLOOD_R) * (C.DEGRADE_SCALE if self.degrade_level >= 1 else 1.0)
        self.nav.flood_request(me.pos, r, {"heat": True, "zone": True, "energy": me.energy,
                                           "reserve": C.CLIMB_NAV_RESERVE})

    # -------------------------------------------------------------- pipeline
    def act(self, obs: Dict[str, Any], t0: float) -> Dict[str, Any]:
        C = self.C
        clock = Clock(t0, C.SOFT_DEADLINE_MS, C.HARD_DEADLINE_MS)
        config.update_live(C, obs.get("self"))
        self.last_tick = obs.get("tick")
        if not self.ready:
            return safe.act(obs, C)
        self._ticks += 1
        ms: Dict[str, float] = {}
        now = time.perf_counter
        world = self.world
        self.arbiter.runner_info = self.runner_info

        # 1-2 world
        s = now()
        try:
            self.map.new_tick()
            world.update(obs, clock)
        except Exception:
            self._fail("world")
        ms["world"] = (now() - s) * 1000.0
        intent = Intent(tick=int(obs.get("tick", 0) or 0), t=float(obs.get("t", 0.0) or 0.0))

        # 4 strategic refresh (R0 on next_changed, zone field start)
        s = now()
        try:
            self.modes.strategic_refresh(world, self.assess, self.nav, clock)
        except Exception:
            self._fail("refresh")
        # 5 assessment
        try:
            self._A = self.assess.update(world, self.nav, clock)
        except Exception:
            self._fail("assess")
        A = self._A
        ms["assess"] = (now() - s) * 1000.0

        # 6 reflexes (O1-O9, urgent cover)
        s = now()
        try:
            last_move = self.arbiter.blocked_desire or self._last_action.get("move")   # O5 commanded vector
            ostate = self.overlays.update(world, A, self.nav, intent, clock, macro=self.modes.macro,
                                          sub=self.modes.sub, last_move=last_move)
        except Exception:
            self._fail("overlays")
            ostate = self._OverlayState({}, None, False, [])
        ms["overlays"] = (now() - s) * 1000.0

        # 7-8 modes, goals, tactics
        s = now()
        try:
            self.modes.decide(world, A, self.nav, ostate, intent, clock, assessor=self.assess)
        except Exception:
            self._fail("modes")
            self._modes_fallback(intent)
        ms["modes"] = (now() - s) * 1000.0

        # 9 actuators
        s = now()
        try:
            action = self.arbiter.resolve(world, A, self.nav, intent, clock)
        except Exception:
            self._fail("arbiter")
            action = safe.act(obs, C)
        if self.degrade_level >= 3:                       # R15.3 L3: SAFE legs, full aim/fire
            try:
                sa = safe.act(obs, C)
                for k in ("move", "boost", "jump"):
                    if k in sa:
                        action[k] = sa[k]
            except Exception:
                self._fail("arbiter")
        # 3 L1 offer (after the actuators so a failed arbiter still answers the offer)
        try:
            intent.pick, intent.pick_rule = self.upgrades.pick(world)
            if intent.pick is not None:
                action["pick"] = intent.pick
        except Exception:
            self._fail("pick")
        ms["arbiter"] = (now() - s) * 1000.0

        # 10 anytime work until the soft deadline
        s = now()
        if not clock.soft_passed():
            try:
                self._flood(world)
                lv = self.degrade_level
                if lv >= 2:                               # R15.3 L2: ranking every DEGRADE_RANK_EVERY ticks
                    self.nav._RANK_MS_CAP = self._base_rank if self._ticks % int(C.DEGRADE_RANK_EVERY) == 0 else 0.0
                else:
                    self.nav._RANK_MS_CAP = self._base_rank
                self.nav.anytime(world, A, clock)
            except Exception:
                self._fail("anytime")
        ms["anytime"] = (now() - s) * 1000.0

        # 11 telemetry + label
        action["debug"] = {"label": intent.label()}
        self._last_intent = intent
        self._last_action = action
        self._module_exceptions()
        if not clock.hard_passed():
            s = now()
            try:
                ms["total"] = clock.elapsed_ms()
                self.tlm.record(world, A, intent, action, clock, ms)
            except Exception:
                self._fail("telemetry")
        el = clock.elapsed_ms()
        ms["total"] = el
        self.last_stage_ms = ms
        try:
            self._degrade_update(el)
        except Exception:
            self._fail("telemetry")
        return action

    def _modes_fallback(self, intent: Intent) -> None:
        """R15.1 modes fallback: last intent's leg and aim owners (the arbiter drops stale fire)."""
        li = self._last_intent
        if li is None:
            return
        intent.legs = [lr for lr in li.legs if lr.level not in ("L3", "L3b", "L4")]
        intent.aim = [a for a in li.aim if a.prio >= 3]
        intent.macro, intent.sub, intent.goal = li.macro, li.sub, li.goal
        intent.rule = "R15.1"

    def on_end(self, msg: Dict[str, Any]) -> None:
        if self.tlm is not None:
            try:
                self.tlm.count("degrade_final", self.degrade_level)
                self.tlm.end_summary(msg, self.runner_info)
                return
            except Exception:
                pass
        runner.log(f'{{"ev":"end","place":{msg.get("place")},"kills":{msg.get("kills")},'
                   f'"exc":{self.runner_info.exceptions},"hard_overruns":{self.runner_info.hard_overruns}}}')


if __name__ == "__main__":
    sys.exit(runner.run(Bot))
