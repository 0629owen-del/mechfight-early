"""Our copy of the kit's mechfight.run() loop (plan 3.2, R15.1-R15.3).

Kept from the kit: answer `hello` with `ready`; answer only the newest queued observation; echo
`tick`; exit on `end` / stdin EOF. Added:
- reader thread stamps each line's receipt time t0 (time.perf_counter) for deadlines;
- Bot.__init__(hello, deadline) and act(obs, t0) run on a worker thread; the main thread waits at
  most until the deadline, so `ready` is always sent within the hello budget (BOOT_MAX_MS from t0)
  and an action always goes out by HARD_DEADLINE_MS from t0 (SAFE's output if act() is late);
- act() exceptions (and per-stage ones the bot reports via take_exceptions()) are counted; a throw
  gets SAFE's controller on the same tick; >= SAFE_MODE_EXC_N in SAFE_MODE_EXC_WINDOW_S switches
  to X5 SAFE for SAFE_MODE_S (T1), re-armed by every new exception (T7 back after quiet);
- on_end(msg) hook; gc.collect()+gc.freeze() after __init__, then gen-0 collections on idle ticks;
- stdout is reserved: sys.stdout is pointed at stderr so a stray print() cannot corrupt the
  protocol; every action is sanitized to the schema (finite numbers, 2-item vectors, label-only
  debug) and kept under bot.max_line_bytes.
"""
from __future__ import annotations

import gc
import json
import math
import queue
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Dict, Optional

import config as _config
import safe as _safe
from contracts import RunnerInfo

_CONTINUOUS = ("move", "aim", "boost")


def log(*args: Any) -> None:
    """stderr only (matches never show it; the conformance check shows its tail on failure)."""
    try:
        print(*args, file=sys.stderr, flush=True)
    except Exception:
        pass


class _Job:
    __slots__ = ("fn", "args", "done", "result", "exc", "tb", "consumed", "kind", "tick")

    def __init__(self, fn: Callable, args: tuple, kind: str, tick: Any = None):
        self.fn, self.args, self.kind, self.tick = fn, args, kind, tick
        self.done = threading.Event()
        self.result: Any = None
        self.exc: Optional[BaseException] = None
        self.tb = ""
        self.consumed = False


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def sanitize(action: Any, max_label: int = 80, debug_full: bool = False) -> Dict[str, Any]:
    """Keep only schema-valid Action fields (drops anything malformed). `tick` is added by the caller."""
    out: Dict[str, Any] = {}
    if not isinstance(action, dict):
        return out
    for k in ("move", "aim"):
        v = action.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 2 and _finite(v[0]) and _finite(v[1]):
            out[k] = [float(v[0]), float(v[1])]
    if "move" in out:
        n = math.hypot(*out["move"])
        if n > 1.0:
            out["move"] = [out["move"][0] / n, out["move"][1] / n]
    for k in ("boost", "jump", "reload"):
        v = action.get(k)
        if isinstance(v, bool):
            out[k] = v
    f = action.get("fire")
    if isinstance(f, (list, tuple)):
        fl = [bool(x) for x in list(f)[:2]]
        out["fire"] = fl + [False] * (2 - len(fl))
    for k in ("pick", "take", "drop"):
        v = action.get(k)
        if _finite(v):
            out[k] = int(v) if float(v).is_integer() else v
    d = action.get("debug")
    if isinstance(d, dict):
        if debug_full:
            out["debug"] = d
        elif isinstance(d.get("label"), str):
            out["debug"] = {"label": d["label"][:max_label]}
    return out


class Runner:
    def __init__(self, make_bot: Callable[..., Any], inp=None, out=None, clock: Callable[[], float] = time.perf_counter):
        self.make_bot = make_bot
        self.inp = inp if inp is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self.clock = clock
        self.q: "queue.Queue" = queue.Queue()
        self.jobs: "queue.Queue" = queue.Queue()
        self.bot: Any = None
        self.C: Any = _config.bind(None)
        self.info = RunnerInfo()
        self.inflight: Optional[_Job] = None   # last job handed to the worker (may still run)
        self.init_job: Optional[_Job] = None
        self.exc_times: deque = deque()
        self.exc_logged = 0
        self.last: Dict[str, Any] = {}
        self.gc_frozen = False
        self.ticks = 0
        self.idle_gcs = 0
        self.max_line = 65536
        self.sent = 0

    # -------------------------------------------------------------- threads
    def _reader(self) -> None:
        try:
            for line in self.inp:
                if line.strip():
                    self.q.put((self.clock(), line))
        except Exception:
            log(f"stdin reader failed:\n{traceback.format_exc()}")
        self.q.put((self.clock(), None))

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                job.result = job.fn(*job.args)
            except BaseException as e:  # noqa: BLE001 - every failure is counted, never fatal
                job.exc = e
                job.tb = traceback.format_exc()
            job.done.set()

    def _submit(self, fn: Callable, args: tuple, kind: str, tick: Any = None) -> _Job:
        job = _Job(fn, args, kind, tick)
        self.inflight = job
        self.jobs.put(job)
        return job

    def _wait(self, job: _Job, deadline: float) -> bool:
        """Wait for `job` until `deadline` (perf_counter s). Lock/Event timeouts can wake several ms
        late (seen on macOS with a GIL-holding worker), so wait coarsely, then poll with short sleeps."""
        coarse = deadline - self.clock() - 0.008
        if coarse > 0 and job.done.wait(coarse):
            return True
        while not job.done.is_set():
            left = deadline - self.clock()
            if left <= 0:
                return False
            time.sleep(min(0.0003, left))
        return True

    def _worker_busy(self) -> bool:
        return self.inflight is not None and not self.inflight.done.is_set()

    # -------------------------------------------------------------- output
    def _write(self, msg: Dict[str, Any]) -> None:
        self.out.write(json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n")
        self.out.flush()

    def _send_action(self, action: Any, tick: Any) -> None:
        label_max = int(self.C.path("debug.max_text_chars", 80) or 80)
        full = bool(getattr(self.C, "TLM_DEBUG_CHANNEL", False))
        a = sanitize(action, label_max, full)
        a["tick"] = tick
        try:
            line = json.dumps(a, separators=(",", ":"), allow_nan=False)
            if len(line.encode()) > self.max_line:
                a.pop("debug", None)
                line = json.dumps(a, separators=(",", ":"), allow_nan=False)
            if len(line.encode()) > self.max_line:
                raise ValueError("action line too long")
        except (TypeError, ValueError):
            log(f"unserializable action at tick {tick}:\n{traceback.format_exc()}")
            a = {"tick": tick, **{k: v for k, v in self.last.items() if k in _CONTINUOUS}}
            line = json.dumps(a, separators=(",", ":"))
        self.out.write(line + "\n")
        self.out.flush()
        self.sent += 1
        self.last = {k: a[k] for k in _CONTINUOUS if k in a}

    # -------------------------------------------------------------- exceptions / T1
    def _note_exception(self, t: float, where: str, tb: str = "", n: int = 1) -> None:
        for _ in range(max(0, n)):
            self.exc_times.append(t)
        self.info.exceptions += n
        self.exc_logged += n
        every = int(getattr(self.C, "EXC_LOG_EVERY", 100) or 100)
        if tb and (self.info.exceptions <= 5 or self.exc_logged >= every):
            self.exc_logged = 0
            log(f"exception in {where} at t={t:.1f} ({self.info.exceptions} so far):\n{tb}")
        window = float(self.C.SAFE_MODE_EXC_WINDOW_S)
        while self.exc_times and self.exc_times[0] < t - window:
            self.exc_times.popleft()
        in_safe = t < self.info.safe_until_t
        if in_safe or len(self.exc_times) >= int(self.C.SAFE_MODE_EXC_N):
            if not in_safe:
                log(f"T1: switching to SAFE at t={t:.1f} for {self.C.SAFE_MODE_S} s")
            self.info.safe_until_t = t + float(self.C.SAFE_MODE_S)

    def _collect_late(self, t: float) -> None:
        """Account for a job that finished after its deadline (or a fire-and-forget observe)."""
        j = self.inflight
        if j is not None and j.done.is_set() and not j.consumed:
            j.consumed = True
            if j.exc is not None:
                self._note_exception(t, f"{j.kind} (late)", j.tb)

    def _bot_reported(self, t: float) -> None:
        take = getattr(self.bot, "take_exceptions", None)
        if take is None:
            return
        try:
            n = int(take() or 0)
        except Exception:
            n = 0
        if n > 0:
            self._note_exception(t, "bot stage", "", n)

    # -------------------------------------------------------------- messages
    def _attach(self, bot: Any) -> None:
        self.bot = bot
        try:
            bot.runner_info = self.info
        except Exception:
            pass
        if not self.gc_frozen:
            gc.collect()
            gc.freeze()
            gc.disable()  # collections only after an answer is out (_after_send)
            self.gc_frozen = True

    def _on_hello(self, msg: Dict[str, Any], t0: float) -> None:
        self.C = _config.bind(msg.get("config"))
        self.max_line = int(self.C.path("bot.max_line_bytes", 65536) or 65536)
        if self.C.bind_errors:
            log(f"config bind errors: {self.C.bind_errors}")
        deadline = t0 + float(self.C.BOOT_MAX_MS) / 1000.0
        job = self._submit(self.make_bot, (msg, deadline), "init")
        self.init_job = job
        if self._wait(job, deadline):
            self._finish_init(job)
        else:
            self.info.init_late = True
            log("Bot.__init__ missed its deadline: answering with SAFE until it returns")
        name = getattr(self.bot, "name", None) or getattr(self.make_bot, "name", None) or "bot"
        self._write({"type": "ready", "name": str(name)[:32]})

    def _finish_init(self, job: _Job) -> None:
        job.consumed = True
        self.init_job = None
        if job.exc is not None:
            log(f"Bot.__init__ failed (SAFE for the whole match):\n{job.tb}")
            return
        self._attach(job.result)

    def _on_obs(self, obs: Dict[str, Any], t0: float) -> None:
        tick = obs.get("tick")
        t = float(obs.get("t", 0.0) or 0.0)
        if self.init_job is not None and self.init_job.done.is_set():
            self._finish_init(self.init_job)
        self._collect_late(t)
        hard_s = float(self.C.HARD_DEADLINE_MS) / 1000.0
        action: Any = None
        source = "act"
        if self.bot is None or self.init_job is not None:
            source = "safe:boot"
        elif t < self.info.safe_until_t:
            source = "safe:T1"
            observe = getattr(self.bot, "observe", None)
            if observe is not None and not self._worker_busy():
                self._submit(observe, (obs, t0), "observe", tick)  # keep the world fresh; not awaited
        elif self._worker_busy():
            source = "safe:busy"  # previous act() still running past its deadline
            self._overrun(tick)
        else:
            job = self._submit(self.bot.act, (obs, t0), "act", tick)
            if self._wait(job, t0 + hard_s):
                job.consumed = True
                if job.exc is not None:
                    self._note_exception(t, "act()", job.tb)
                    source = "safe:exc"
                else:
                    action = job.result
            else:
                source = "safe:late"
                self._overrun(tick)
            self._bot_reported(t)
        if action is None:
            action = self._safe(obs)
        self.info.last_source = source
        self._send_action(action, tick)
        el_ms = (self.clock() - t0) * 1000.0
        self.info.last_compute_ms = el_ms
        if el_ms > float(self.C.SOFT_DEADLINE_MS):
            self.info.soft_overruns += 1
            self.info.last_soft_overrun_tick = tick if isinstance(tick, int) else -1
        self._after_send(el_ms)

    def _overrun(self, tick: Any) -> None:
        self.info.hard_overruns += 1
        self.info.last_hard_overrun_tick = tick if isinstance(tick, int) else -1
        if self.info.hard_overruns <= 5 or self.info.hard_overruns % 100 == 0:
            log(f"hard deadline overrun at tick {tick} ({self.info.hard_overruns} so far)")

    def _safe(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _safe.act(obs, self.C)
        except Exception:
            log(f"SAFE controller failed:\n{traceback.format_exc()}")
            return dict(self.last)

    def _after_send(self, el_ms: float) -> None:
        if not self.gc_frozen:
            return
        self.ticks += 1
        try:
            if el_ms < float(self.C.GC_IDLE_MS) or gc.get_count()[0] > 100000:
                self.idle_gcs += 1
                gc.collect(1 if self.idle_gcs % 50 == 0 else 0)
            if self.ticks % 600 == 0:
                gc.collect()
        except Exception:
            pass

    def _on_end(self, msg: Dict[str, Any]) -> None:
        if self.bot is None:
            return
        hook = getattr(self.bot, "on_end", None)
        if hook is None:
            return
        if self.inflight is not None:
            self.inflight.done.wait(0.5)
        try:
            hook(msg)
        except Exception:
            log(f"on_end failed:\n{traceback.format_exc()}")

    # -------------------------------------------------------------- loop
    def run(self) -> int:
        sys.setswitchinterval(0.0005)  # the main thread must win the GIL promptly at the hard deadline
        threading.Thread(target=self._reader, name="reader", daemon=True).start()
        threading.Thread(target=self._worker, name="worker", daemon=True).start()
        while True:
            batch = [self.q.get()]
            while True:  # drain: only the newest observation matters
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            latest = None
            latest_t0 = 0.0
            for t0, raw in batch:
                if raw is None:
                    return 0
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")
                if kind == "hello":
                    self._on_hello(msg, t0)
                elif kind == "obs":
                    latest, latest_t0 = msg, t0
                elif kind == "end":
                    self._on_end(msg)
                    return 0
            if latest is not None:
                try:
                    self._on_obs(latest, latest_t0)
                except (BrokenPipeError, OSError):
                    return 0
                except Exception:  # last line of defense: always answer the tick
                    log(f"runner failed at tick {latest.get('tick')}:\n{traceback.format_exc()}")
                    try:
                        self._send_action(dict(self.last), latest.get("tick"))
                    except Exception:
                        pass


def run(make_bot: Callable[..., Any]) -> int:
    """Run a bot over real stdin/stdout. `make_bot(hello, deadline)` returns an object with
    act(obs, t0) -> action dict and optionally name, observe(obs, t0), take_exceptions(), on_end(msg)."""
    real_out = sys.stdout
    sys.stdout = sys.stderr  # stray print() cannot reach the protocol stream
    try:
        return Runner(make_bot, inp=sys.stdin, out=real_out).run()
    finally:
        sys.stdout = real_out
