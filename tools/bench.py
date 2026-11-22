#!/usr/bin/env python3
"""CPU bench for the bot: per-tick wall time of Bot.act on the recorded and the synthetic worst-case stream.

    python3 tools/bench.py                         # in-process, recorded + worst case
    python3 tools/bench.py --stream worst --n 1200 --profile
    python3 tools/bench.py --subprocess            # end-to-end answer times through runner.py (stdin/stdout, like check.py)
    python3 tools/bench.py --slow 12 --gate        # with 12 CPU burners running; exit 1 if p99 >= soft or max >= hard

Numbers are LOCAL (this machine), not the 1-CPU 512 MB container. `--slow N` approximates a slower CPU by
running N busy-loop processes at the same time: use N >= number of cores (os.cpu_count()) to actually take cores
away from the bot; with N = 2 x cores the bot gets roughly half a core.

In-process mode mirrors runner.py: JSON parse of the line is inside the timed section (t0 = "receipt"),
gc.collect()+gc.freeze()+gc.disable() after Bot.__init__, gen-0 collection after the answer on ticks under GC_IDLE_MS.
Stage breakdown: if the bot exposes per-tick stage timings (`bot.last_stage_ms`, `bot.stage_ms`,
`bot.tlm.last_stage_ms`, `bot.tlm.stage_ms`: dict name -> ms for the last tick; or telemetry's cumulative `bot.tlm.stage_sum`, differenced per tick) they are tabulated; `--profile`
adds a cProfile pass (own time per module and top functions) which works with any bot internals.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
for _p in (BOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import worstcase as W  # noqa: E402


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def summarize(times_ms, soft, hard):
    return {
        "n": len(times_ms), "p50": pct(times_ms, 0.5), "p95": pct(times_ms, 0.95), "p99": pct(times_ms, 0.99),
        "max": max(times_ms) if times_ms else 0.0, "mean": statistics.fmean(times_ms) if times_ms else 0.0,
        "late_soft": sum(1 for t in times_ms if t > soft), "late_hard": sum(1 for t in times_ms if t > hard),
    }


def deadlines(hello):
    import config
    C = config.bind(hello.get("config"))
    return float(C.SOFT_DEADLINE_MS), float(C.HARD_DEADLINE_MS), C


# ---------------------------------------------------------------------------- slow-CPU burners
class Burners:
    def __init__(self, n):
        self.procs = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]

    def stop(self):
        for p in self.procs:
            p.kill()
        for p in self.procs:
            p.wait()

    def __enter__(self):
        time.sleep(0.2 if self.procs else 0)
        return self

    def __exit__(self, *a):
        self.stop()


# ---------------------------------------------------------------------------- in-process
_PREV_SUM = {}


def _stage_probe(bot):
    """Per-tick stage ms: a dict the bot keeps, else the per-tick delta of telemetry's cumulative `stage_sum`."""
    tl = getattr(bot, "tlm", None)
    ss = getattr(tl, "stage_sum", None)
    if isinstance(ss, dict) and ss and not any(isinstance(getattr(bot, a, None), dict) and getattr(bot, a) for a in ("last_stage_ms", "stage_ms")):
        prev = _PREV_SUM.get(id(bot), {})
        out = {k: float(v) - prev.get(k, 0.0) for k, v in ss.items()}
        _PREV_SUM[id(bot)] = {k: float(v) for k, v in ss.items()}
        return out
    for path in (("last_stage_ms",), ("stage_ms",), ("tlm", "last_stage_ms"), ("tlm", "stage_ms")):
        o = bot
        for k in path:
            o = getattr(o, k, None)
            if o is None:
                break
        if isinstance(o, dict) and o and all(isinstance(v, (int, float)) for v in o.values()):
            return dict(o)
    return None


def run_inproc(hello, obs_list, soft, hard, C, profile=False, parse=True, gc_mode=True):
    """Returns dict(times_ms, stages{name: [ms]}, init_ms, exceptions, profile_text)."""
    import main
    hello = copy.deepcopy(hello)
    hello["map"]["dir"] = W.ARENA
    lines = [json.dumps(o, separators=(",", ":")) for o in obs_list]
    t_i = time.perf_counter()
    bot = main.Bot(hello, time.perf_counter() + C.BOOT_MAX_MS / 1000.0)
    init_ms = (time.perf_counter() - t_i) * 1000
    gc_was = gc.isenabled()
    if gc_mode:
        gc.collect()
        gc.freeze()
        gc.disable()
    times, stages = [], {}
    exc0 = getattr(bot, "_exceptions", 0)
    prof = None
    if profile:
        import cProfile
        prof = cProfile.Profile()
    try:
        for line in lines:
            t0 = time.perf_counter()
            if prof:
                prof.enable()
            obs = json.loads(line) if parse else None
            act = bot.act(obs, t0)
            if prof:
                prof.disable()
            el = (time.perf_counter() - t0) * 1000
            times.append(el)
            st = _stage_probe(bot)
            if st:
                for k, v in st.items():
                    stages.setdefault(k, []).append(float(v))
            if gc_mode and el < C.GC_IDLE_MS:
                gc.collect(0)
    finally:
        if gc_mode:
            gc.unfreeze()
            if gc_was:
                gc.enable()
    ptxt = ""
    if prof:
        import io
        import pstats
        buf = io.StringIO()
        ps = pstats.Stats(prof, stream=buf)
        ps.sort_stats("tottime")
        by_file = {}
        for (fn, ln, name), (cc, nc, tt, ct, callers) in ps.stats.items():
            key = os.path.basename(fn) if fn.startswith(BOT + os.sep) else "(stdlib/builtin)"
            by_file[key] = by_file.get(key, 0.0) + tt
        tot = sum(by_file.values()) or 1.0
        ptxt = "own time per file (profiled run, includes profiler overhead):\n"
        for k, v in sorted(by_file.items(), key=lambda kv: -kv[1])[:12]:
            ptxt += f"  {k:24s} {v * 1000 / len(lines):7.3f} ms/tick  {100 * v / tot:5.1f}%\n"
        ptxt += "top functions by own time (ms/tick, calls/tick):\n"
        rows = sorted(ps.stats.items(), key=lambda kv: -kv[1][2])[:15]
        for (fn, ln, name), (cc, nc, tt, ct, callers) in rows:
            ptxt += f"  {tt * 1000 / len(lines):7.3f} ms  {nc / len(lines):8.1f}x  cum {ct * 1000 / len(lines):7.3f}  {os.path.basename(fn)}:{ln}({name})\n"
    return {"times_ms": times, "stages": stages, "init_ms": init_ms,
            "exceptions": getattr(bot, "_exceptions", 0) - exc0, "profile": ptxt}


# ---------------------------------------------------------------------------- subprocess (runner, stdin/stdout)
class Proc:
    def __init__(self, cmd, cwd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0, cwd=cwd)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()

    def _rd(self):
        for raw in self.p.stdout:
            self.q.put((time.perf_counter(), raw))
        self.q.put((time.perf_counter(), None))

    def send(self, msg):
        try:
            self.p.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def get(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


def run_subprocess(hello, obs_list, soft, hard, python=None, cmd=None, budget_s=None):
    """End-to-end: send hello, wait `ready`, then one obs at a time; time = send -> answer line with the same tick."""
    hello = copy.deepcopy(hello)
    hello["map"]["dir"] = W.ARENA
    cmd = cmd or [python or sys.executable, "-u", os.path.join(BOT, "main.py")]
    budget_s = budget_s or hello["config"]["bot"]["tick_budget_ms"] / 1000.0
    pr = Proc(cmd, BOT)
    t_s = time.perf_counter()
    pr.send(hello)
    ready_ms = None
    while True:
        got = pr.get(hello["config"]["bot"]["hello_budget_ms"] / 1000.0)
        if got is None or got[1] is None:
            pr.p.kill()
            raise RuntimeError("bot gave no `ready`")
        try:
            if json.loads(got[1]).get("type") == "ready":
                ready_ms = (got[0] - t_s) * 1000
                break
        except ValueError:
            pass
    times, missing, wrong = [], 0, 0
    for o in obs_list:
        sent = time.perf_counter()
        pr.send(o)
        while True:
            got = pr.get(max(0.0, sent + budget_s + 0.5 - time.perf_counter()))
            if got is None:
                missing += 1
                break
            at, raw = got
            if raw is None:
                raise RuntimeError("bot exited early")
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            if m.get("tick") == o["tick"]:
                times.append((at - sent) * 1000)
                break
            wrong += 1
    pr.send({"type": "end", "place": 1, "kills": 0})
    try:
        pr.p.stdin.close()
        pr.p.wait(timeout=3)
    except Exception:
        pr.p.kill()
    return {"times_ms": times, "ready_ms": ready_ms, "missing": missing, "stale": wrong}


# ---------------------------------------------------------------------------- reporting
def fmt_row(name, s):
    return (f"{name:28s} n={s['n']:4d}  p50 {s['p50']:6.2f}  p95 {s['p95']:6.2f}  p99 {s['p99']:6.2f}  max {s['max']:6.2f}  "
            f"mean {s['mean']:6.2f}  late>soft {s['late_soft']:3d}  late>hard {s['late_hard']:3d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stream", choices=("recorded", "worst", "both"), default="both")
    ap.add_argument("--n", type=int, default=None, help="worst-case observations (default WORSTCASE_TICKS)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--subprocess", action="store_true", help="run the bot through runner.py as a child process (end-to-end answer times)")
    ap.add_argument("--python", default=None, help="interpreter for --subprocess (default: this one)")
    ap.add_argument("--cmd", nargs=argparse.REMAINDER, help="full bot command for --subprocess (overrides --python)")
    ap.add_argument("--slow", type=int, default=0, metavar="N", help="run N CPU-burner processes during the bench")
    ap.add_argument("--profile", action="store_true", help="in-process cProfile pass (own time per file / function)")
    ap.add_argument("--no-parse", action="store_true", help="in-process: do not include json.loads in the timed section")
    ap.add_argument("--no-gc", action="store_true", help="in-process: leave the garbage collector alone")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each stream, pooling the samples")
    ap.add_argument("--gate", action="store_true", help="exit 1 unless p99 < soft and max < hard on every stream")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    a = ap.parse_args(argv)

    hello, rec, _ = W.load_stream()
    soft, hard, C = deadlines(hello)
    streams = []
    if a.stream in ("recorded", "both"):
        streams.append(("recorded", rec))
    if a.stream in ("worst", "both"):
        n = a.n or int(C.WORSTCASE_TICKS)
        streams.append(("worst-case", W.generate(n, a.seed)))
    results, ok = {}, True
    mode = "subprocess" if a.subprocess else "in-process"
    if not a.json:
        print(f"mode: {mode}  slow burners: {a.slow}  cpus: {os.cpu_count()}  python {sys.version.split()[0]}  soft {soft:g} ms  hard {hard:g} ms")
    with Burners(a.slow):
        for name, obs in streams:
            times, extra, stages, prof = [], {}, {}, ""
            for _ in range(a.repeat):
                if a.subprocess:
                    r = run_subprocess(hello, obs, soft, hard, python=a.python, cmd=a.cmd)
                    extra = {"ready_ms": r["ready_ms"], "missing": r["missing"], "stale": r["stale"]}
                else:
                    r = run_inproc(hello, obs, soft, hard, C, profile=a.profile and _ == 0, parse=not a.no_parse, gc_mode=not a.no_gc)
                    extra = {"init_ms": r["init_ms"], "exceptions": r["exceptions"]}
                    for k, v in r["stages"].items():
                        stages.setdefault(k, []).extend(v)
                    prof = prof or r["profile"]
                times += r["times_ms"]
            s = summarize(times, soft, hard)
            s.update(extra)
            s["stages"] = {k: summarize(v, soft, hard) for k, v in stages.items()}
            results[name] = s
            ok &= s["p99"] < soft and s["max"] < hard
            if not a.json:
                print(fmt_row(f"{name} ({mode})", s))
                print("  " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in extra.items()))
                if stages:
                    print("  stage ms:            p50     p95     p99     max")
                    for k, v in sorted(s["stages"].items(), key=lambda kv: -kv[1]["p99"]):
                        print(f"    {k:18s} {v['p50']:7.2f} {v['p95']:7.2f} {v['p99']:7.2f} {v['max']:7.2f}")
                elif not a.subprocess:
                    print("  (bot exposes no stage timings; use --profile)")
                if prof:
                    print("  " + prof.replace("\n", "\n  "))
    if a.json:
        print(json.dumps({"mode": mode, "slow": a.slow, "soft_ms": soft, "hard_ms": hard, "results": results}, default=float))
    if a.gate:
        print("GATE:", "PASS" if ok else "FAIL (local proxy for the 1-CPU container)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
