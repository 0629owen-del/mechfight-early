"""Telemetry (plan 14.4). CONTRACTS 6.7.

One JSON object per line on stderr, each with "ev", "t" (match time) and "tick". Volume is bounded:
  * at most TLM_MAX_LINES_PER_S lines per second and TLM_MAX_BYTES_PER_S bytes per second (token
    buckets on wall time; the optional bytes constant defaults to 4096), a line over TLM_MAX_LINE_BYTES
    (default 1024) is cut down to its header;
  * dropped lines are counted and reported in the next line that gets through (`"dropped": n`);
  * summaries (`ev: "summary"`) and `_force=True` events always go out.
Everything else is aggregated in memory and shows up in the periodic `status` line (every
TLM_DEBUG_PERIOD_S) and the end-of-match summary. Nothing is ever written to debug drawings; stderr is
invisible in real matches, only the local minisim captures it.

`record()` is on the per-tick path: constant work, no allocation of big structures, and it never raises
(an internal failure is counted in `errors`).
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import config as _config
from contracts import Intent, RunnerInfo

_ALWAYS = frozenset({"summary"})


def _clean(v: Any, depth: int = 0) -> Any:
    """JSON-safe copy: finite floats rounded to 3 decimals, non-finite -> None, tuples -> lists."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return round(v, 3) if math.isfinite(v) else None
    if depth > 4:
        return str(v)
    if isinstance(v, dict):
        return {str(k): _clean(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x, depth + 1) for x in list(v)[:64]]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _clean(dataclasses.asdict(v), depth + 1)
    return str(v)


class Telemetry:
    def __init__(self, C, slot: int, seed: int, stream=None, now=time.monotonic) -> None:
        self.C = C
        self.slot = slot
        self.seed = seed
        self.stream = stream               # None = sys.stderr at write time
        self._now = now
        g = lambda n, d: getattr(C, n, d) if C is not None else d
        self.max_lines = float(g("TLM_MAX_LINES_PER_S", 5))
        self.max_bytes = float(g("TLM_MAX_BYTES_PER_S", 4096))
        self.max_line = int(g("TLM_MAX_LINE_BYTES", 1024))
        self.period = float(g("TLM_DEBUG_PERIOD_S", 30))
        self.hit_match_s = float(g("TLM_HIT_MATCH_S", 0.5))
        self.bucket_m = float(g("TLM_RANGE_BUCKET_M", 25))
        self.hold_a = float(g("HOLD_A_MIN", 0.9))
        self.engage_a = float(g("ENGAGE_A_MIN", 1.0))
        self.self_def_s = float(g("SELF_DEFENSE_S", 3))
        # rate limiter
        self._tok_l = self.max_lines
        self._tok_b = max(self.max_bytes, self.max_line + 64.0)
        self._tok_t = self._now()
        self.dropped = 0
        self.dropped_total = 0
        self.lines = 0
        self.bytes = 0
        self.errors = 0
        # clock
        self._t = 0.0
        self._tick = 0
        # counters
        self.counters: Counter = Counter()
        self.macro_time: Dict[str, float] = defaultdict(float)
        self.goal_time: Dict[str, float] = defaultdict(float)
        self.rule_count: Counter = Counter()
        self.stage_sum: Dict[str, float] = defaultdict(float)
        self.stage_max: Dict[str, float] = defaultdict(float)
        self.stage_n = 0
        self.dmg_taken: Dict[str, float] = defaultdict(float)
        self.t_outside = 0.0
        self.zone_damage = 0.0
        self.missed_ticks = 0
        self.flips = 0
        self._macro: Optional[str] = None
        self._goal: Optional[str] = None
        self._last_t: Optional[float] = None
        # shots and hits: key (weapon, bucket, moving) -> [shots, hits]
        self.hitrate: Dict[Tuple[str, int, int], List[int]] = defaultdict(lambda: [0, 0])
        self._pending: List[List[Any]] = []        # [t_expire, weapon, key, hit?]
        self._prev_w: List[Optional[Tuple[float, Optional[float], float]]] = [None, None]
        # sightings
        self._seen: set = set()
        self.sight_hist: Counter = Counter()
        # fights
        self._fight: Optional[Dict[str, Any]] = None
        self.fights: List[Dict[str, Any]] = []
        self.fight_table: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # key -> [wins, losses+other]
        # A band
        self.a_band_s = 0.0
        self.a_band_flips = 0
        self.a_at_entry: List[float] = []
        # last context (death cause)
        self._ctx: Dict[str, Any] = {}
        self._recent_dmg: List[Tuple[float, float, str, Any]] = []
        self._last_status_t = -1e9
        self._kills_prev = 0
        self._hp_prev: Optional[float] = None

    # ------------------------------------------------------------------ output
    def _refill(self) -> None:
        n = self._now()
        dt = max(0.0, n - self._tok_t)
        self._tok_t = n
        self._tok_l = min(self.max_lines, self._tok_l + dt * self.max_lines)
        cap_b = max(self.max_bytes, self.max_line + 64.0)
        self._tok_b = min(cap_b, self._tok_b + dt * self.max_bytes)

    def _write(self, line: str) -> None:
        try:
            out = self.stream if self.stream is not None else sys.stderr
            out.write(line + "\n")
            out.flush()
        except Exception:
            self.errors += 1

    def event(self, kind: str, **fields) -> bool:
        """Emit one JSONL record (rate limited). Returns True when it was written."""
        force = bool(fields.pop("_force", False)) or kind in _ALWAYS
        rec: Dict[str, Any] = {"ev": kind, "t": round(self._t, 3), "tick": self._tick}
        for k, v in fields.items():
            if k != "ev":
                rec[k] = _clean(v)
        self._refill()
        if not force and self._tok_l < 1.0:
            self.dropped += 1
            self.dropped_total += 1
            return False
        if self.dropped:
            rec["dropped"] = self.dropped
        try:
            line = json.dumps(rec, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            self.errors += 1
            line = json.dumps({"ev": kind, "t": rec["t"], "tick": rec["tick"], "bad": True}, separators=(",", ":"))
        limit = self.max_line if not force else max(self.max_line, 16384)
        if len(line) > limit:
            line = json.dumps({"ev": kind, "t": rec["t"], "tick": rec["tick"], "trunc": len(line)},
                              separators=(",", ":"))
        if not force and self._tok_b < len(line):
            self.dropped += 1
            self.dropped_total += 1
            return False
        self._tok_l = max(0.0, self._tok_l - 1.0)
        self._tok_b = max(0.0, self._tok_b - len(line))
        self.dropped = 0
        self.lines += 1
        self.bytes += len(line) + 1
        self._write(line)
        return True

    def count(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    # ------------------------------------------------------------------ per tick
    def record(self, world, A, intent: Intent, action: dict, clock, stage_ms: Optional[Dict[str, float]] = None) -> None:
        try:
            self._record(world, A, intent, action, stage_ms)
        except Exception:
            self.errors += 1

    def _record(self, world, A, intent: Intent, action: dict, stage_ms) -> None:
        tk = world.tick
        t = float(tk.t)
        dt = float(tk.dt) if tk.dt and tk.dt > 0 else (t - self._last_t if self._last_t is not None else 0.0)
        dt = max(0.0, min(dt, 2.0))
        self._t, self._tick, self._last_t = t, int(tk.tick), t
        me = world.me
        macro, goal = intent.macro, intent.goal
        # time shares and flips
        self.macro_time[macro] += dt
        if goal:
            self.goal_time[goal] += dt
        if self._macro is not None and macro != self._macro:
            self.flips += 1
        if self._goal != goal and self._macro is not None:
            self.counters["goal_changes"] += 1
        prev_macro, self._macro, self._goal = self._macro, macro, goal
        for r in intent.trace:
            self.rule_count[r] += 1
        if intent.rule and intent.rule not in intent.trace:     # modes appends the deciding rule to trace
            self.rule_count[intent.rule] += 1
        if getattr(tk, "missed_ticks", 0):
            self.missed_ticks += tk.missed_ticks
        if stage_ms:
            self.stage_n += 1
            for k, v in stage_ms.items():
                self.stage_sum[k] += v
                if v > self.stage_max[k]:
                    self.stage_max[k] = v
        if not me.in_zone:
            self.t_outside += dt
        # shots first: a hitscan laser's `hit` arrives in the same observation as its cooldown jump
        self._detect_shots(t, me, A, intent, world)
        # damage events
        for ev in world.events:
            ty = ev.get("type")
            if ty == "damaged":
                amt = float(ev.get("amount", 0.0))
                w = str(ev.get("weapon"))
                self.dmg_taken[w] += amt
                if w == "zone":
                    self.zone_damage += amt
                self._recent_dmg.append((t, amt, w, ev.get("by")))
            elif ty == "hit":
                self._match_hit(t, ev)
            elif ty in ("kill", "pickup", "upgrade", "drop", "took", "dropped", "take_failed"):
                self.counters["ev_" + ty] += 1
        if len(self._recent_dmg) > 32:
            self._recent_dmg = self._recent_dmg[-32:]
        self._pending = [p for p in self._pending if p[0] >= t - 1e-9] if self._pending else self._pending
        # first sightings
        for tr in world.tracks.values():
            if tr.kind == "mech" and tr.id not in self._seen and (tr.in_obs or tr.visible):
                self._seen.add(tr.id)
                d = tr.first_seen_dist or math.hypot(tr.pos[0] - me.pos[0], tr.pos[1] - me.pos[1])
                self.sight_hist[int(d // self.bucket_m)] += 1
                self.event("sight", id=tr.id, d=d, weapons=list(tr.weapons))
        # A band and fights
        a = getattr(A, "A", None) if A is not None else None
        if a is not None and self.hold_a <= a < self.engage_a and A.target_id is not None:
            self.a_band_s += dt
        self._fights(world, A, intent, macro, prev_macro, t, me, a)
        self._ctx = {"macro": macro, "goal": goal, "rule": intent.rule, "hp": me.hp, "in_zone": me.in_zone,
                     "alive": tk.alive, "pos": me.pos}
        if t - self._last_status_t >= self.period:
            self._last_status_t = t
            self.event("status", macro=macro, goal=goal, hp=me.hp, alive=tk.alive, cubes=me.cubes,
                       kills=getattr(world, "kills", 0), flips=self.flips, missed=self.missed_ticks,
                       zone_out_s=self.t_outside, ms=self._ms_summary(), lines=self.lines, dropped=self.dropped_total)

    # shots ---------------------------------------------------------------
    def _bucket(self, d: float) -> int:
        return int(d // self.bucket_m)

    def _detect_shots(self, t: float, me, A, intent: Intent, world) -> None:
        for i, w in enumerate(me.weapons[:2]):
            if w is None:
                self._prev_w[i] = None
                continue
            cur = (float(w.cooldown or 0.0), w.ammo, float(w.charge or 0.0))
            prev = self._prev_w[i]
            self._prev_w[i] = cur
            if prev is None:
                continue
            if w.kind == "gun":
                fired = (prev[1] is not None and cur[1] is not None and cur[1] < prev[1])
                if not fired and cur[1] is None:
                    fired = cur[0] > prev[0] + 1e-6
            else:
                fired = cur[0] > prev[0] + 1e-6
            if not fired:
                continue
            tid = next((fr.target_id for fr in intent.fire if fr.slot == i and fr.target_id is not None), None)
            if tid is None and A is not None:
                tid = A.target_id
            d, moving = 0.0, 0
            ta = A.per_track.get(tid) if (A is not None and tid is not None) else None
            if ta is not None:
                d = ta.d
                tr = world.tracks.get(tid)
                if tr is not None:
                    moving = 1 if math.hypot(tr.vel[0], tr.vel[1]) > 1.0 else 0
            key = (w.kind, self._bucket(d), moving)
            self.hitrate[key][0] += 1
            flight = 0.0
            if w.kind == "gun":
                flight = d / max(1.0, float(w.stats.get("speed", 252.0)))
            elif w.kind == "mortar":
                flight = d / max(1.0, 0.7 * float(w.stats.get("speed", 72.0))) + 0.3
            self._pending.append([t + self.hit_match_s + flight, w.kind, key, False])
            if len(self._pending) > 64:
                self._pending = self._pending[-64:]

    def _match_hit(self, t: float, ev: dict) -> None:
        wk = ev.get("weapon")
        tgt = ev.get("target")
        if wk in (None, "npc") or (isinstance(tgt, int) and tgt >= 1000):
            return                                 # crates, cubes and NPC targets are not mech hits
        for p in self._pending:
            if p[1] == wk and not p[3]:
                p[3] = True
                self.hitrate[p[2]][1] += 1
                return
        self.counters["hit_unmatched"] += 1

    # fights ---------------------------------------------------------------
    def _loadout(self, world, tid) -> str:
        tr = world.tracks.get(tid) if tid is not None else None
        return _config.loadout_key(tr.weapons) if tr is not None and tr.weapons else "?"

    def _fights(self, world, A, intent: Intent, macro: str, prev_macro: Optional[str], t: float, me, a) -> None:
        f = self._fight
        kills = int(getattr(world, "kills", 0) or 0)
        if macro == "X2" and f is None:
            tid = intent.fire[0].target_id if intent.fire else (A.target_id if A is not None else None)
            if tid is None and A is not None:
                tid = A.target_id
            ta = A.per_track.get(tid) if (A is not None and tid is not None) else None
            self._fight = f = {"t0": t, "target": tid, "ours": _config.loadout_key(world.held_kinds()),
                               "theirs": self._loadout(world, tid), "d0": ta.d if ta is not None else None,
                               "A0": a, "hp0": me.hp, "kills0": kills}
            if a is not None:
                self.a_at_entry.append(a)
            self.counters["fights"] += 1
        elif f is not None and macro != "X2":
            outcome = "kill" if kills > f["kills0"] else ("flee" if macro == "X3" else "end")
            if macro == "X3" and a is not None:
                self.a_band_flips += 1
                self.event("aband", A=a, to="X3", target=f["target"])
            rec = {"dur": t - f["t0"], "target": f["target"], "ours": f["ours"], "theirs": f["theirs"],
                   "d0": f["d0"], "A0": f["A0"], "hp0": f["hp0"], "hp1": me.hp, "out": outcome}
            self._finish_fight(rec)
            self._fight = None
        # a kill count increase while fighting counts for the episode; nothing else to do per tick

    def _finish_fight(self, rec: Dict[str, Any]) -> None:
        d0 = rec.get("d0")
        key = f"{rec['ours']}>{rec['theirs']}@{self._bucket(d0) if d0 is not None else -1}"
        row = self.fight_table[key]
        row[0 if rec["out"] == "kill" else 1] += 1
        if len(self.fights) < 64:
            self.fights.append(rec)
        self.event("fight", **rec)

    # ------------------------------------------------------------------ summaries
    def _ms_summary(self) -> Dict[str, Any]:
        n = max(1, self.stage_n)
        return {k: [round(v / n, 2), round(self.stage_max[k], 2)] for k, v in list(self.stage_sum.items())[:12]}

    def death_cause(self) -> Dict[str, Any]:
        t = self._t
        by_w: Dict[str, float] = defaultdict(float)
        by_a: Dict[Any, float] = defaultdict(float)
        for (ts, amt, w, who) in self._recent_dmg:
            if ts >= t - self.self_def_s - 0.5:
                by_w[w] += amt
                by_a[who] += amt
        top_w = max(by_w, key=by_w.get) if by_w else None
        top_a = max(by_a, key=by_a.get) if by_a else None
        return {"weapon": top_w, "by": top_a, "dmg": dict(by_w), "macro": self._ctx.get("macro"),
                "goal": self._ctx.get("goal"), "rule": self._ctx.get("rule"), "hp": self._ctx.get("hp"),
                "in_zone": self._ctx.get("in_zone"), "alive": self._ctx.get("alive")}

    def hitrate_table(self) -> List[List[Any]]:
        rows = []
        for (w, b, mv), (s, h) in sorted(self.hitrate.items()):
            rows.append([w, int(b * self.bucket_m), mv, s, h])
        return rows

    def end_summary(self, msg: dict, runner_info: Optional[RunnerInfo] = None) -> None:
        try:
            if self._fight is not None:
                f = self._fight
                self._finish_fight({"dur": self._t - f["t0"], "target": f["target"], "ours": f["ours"],
                                    "theirs": f["theirs"], "d0": f["d0"], "A0": f["A0"], "hp0": f["hp0"],
                                    "hp1": self._ctx.get("hp"), "out": "end"})
                self._fight = None
            total_t = max(self._t, 1e-9)
            end = {k: v for k, v in (msg or {}).items() if isinstance(v, (int, float, str, bool, type(None)))}
            self.event(
                "summary",
                end=end,
                death=self.death_cause(),
                time_s=self._t,
                macro_s=dict(self.macro_time),
                goal_s=dict(self.goal_time),
                flips_per_min=self.flips / max(total_t / 60.0, 1e-9),
                rules=dict(self.rule_count.most_common(40)),
                hitrate=self.hitrate_table(),
                sight=sorted(self.sight_hist.items()),
                fights=dict(self.fight_table),
                a_band_s=self.a_band_s,
                a_band_flips=self.a_band_flips,
                a_entry=self.a_at_entry[:16],
                dmg_taken=dict(self.dmg_taken),
                zone_out_s=self.t_outside,
                zone_damage=self.zone_damage,
                missed_ticks=self.missed_ticks,
                ms=self._ms_summary(),
                counters=dict(self.counters),
                lines=self.lines,
                bytes=self.bytes,
                dropped=self.dropped_total,
                errors=self.errors,
                runner=runner_info,
                slot=self.slot,
                seed=self.seed,
                _force=True,
            )
        except Exception:
            self.errors += 1
