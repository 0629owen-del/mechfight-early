#!/usr/bin/env python3
"""climbcheck: how often does a climb our navigation starts end on the level it was climbing to?

Measured inside tools/minisim.py, whose physics and map model are its own (tools/navsim.py builds its
surface from the bot's nav_surface_1m, so a climb that works there only agrees with nav's own reading
of the map).

Usage

    python3 tools/climbcheck.py --bot <dir with main.py> [--seeds N] [--seed0 S] [--duration T]
                                [--out DIR] [-j J]
    python3 tools/climbcheck.py --bot old=/path/a --bot new=/path/b --seeds 8 --seed0 800 -j 2 --out runs/cc
    python3 tools/climbcheck.py --bot old=/path/a --bot new=/path/b --out runs/cc --rejudge --land-r 4
    python3 tools/climbcheck.py --selftest

--bot may repeat (optionally label=dir); every build plays the same seeds. One match per build and seed:
our bot in one slot (run through tools/tracebot.py with TRACEBOT_DIR=<dir>, so the code under test is
that directory's, checked from the wrapper's TRACEBOT line) and minisim's scripted bots in the others,
`--peace` (no weapon damage, so the bot lives until the zone takes it; --no-peace turns that off).
Other options: --mechs, --sim (the minisim.py to run, default the one next to this file), --kit,
--limit-jet, --limit-hop, --land-r, --z-tol, --hold, --rejudge (judge the match directories already in
--out again, e.g. with other limits, without running anything; it takes every s<seed> directory it
finds there, so batches run into one --out with different --seed0 add up).

It prints attempts, successes, rate, failures by class and a per-seed table for every build, and writes
<out>/summary.json (every attempt with its outcome) and <out>/summary.txt (the printed report).
Match files are kept under <out>/<label>/s<seed>/ (match.jsonl, slotNN.stderr, summary.json).

What is counted

A climb attempt = nav starts a climb, taken from the bot side (tracebot's CLIMB lines):
- jet: Nav._climb_begin, the jump-and-jet sequence up a climb face. Target top = the height nav
  climbs to, landing = the face's landing point on the upper level.
- hop: a nav Steer asks for a jump up a ledge in (step_up 1.2 m, jump height] (path jump edge, zone
  field step, interim steering). One attempt per jump; a jump still asked for on the next decision is
  the same attempt. Target top = nav's surface height just past the edge, landing = the point one mech
  radius past the edge along the steer direction.
The stuck ladder's jumps and jets (O5) and tactics_roof's own roof assault are not navigation climbs
and are not counted.

The outcome is judged only from the simulator's positions of our mech (the `self` pos / grounded of
every observation, copied by the wrapper as CP lines before the bot sees it) and minisim's death
record:

    success = at some decision after the start and within the time limit (--limit-jet 15 s,
              --limit-hop 3 s) the mech is grounded, within --land-r (6 m) horizontally of the landing
              point and at a height >= target top - --z-tol (0.5 m).

The window also closes when the next attempt of the same kind starts (a later climb must not be
credited to an earlier one). That is one observation on the level: a mech that touches the edge of a
roof and is off it again two decisions later is a success by this rule. So every success also gets

    stayed  = for --hold (1 s) after that observation the mech is never more than step_up (1.2 m)
              below the level (None when the track ends first),

and the report gives both counts: `ok` (touched down on the level) and `stay` (and stayed on it).
Failures, by the first thing that happened in the window:

    fell_back        after the jump was commanded the mech left the ground and came down below the level
    abandoned        nav ended the climb itself before that (abort: timeout / retry / energy, dropped,
                     replaced); the nav-side ending is kept as `nav_end` for every jet attempt
    level_elsewhere  it stood at the target height or above, but further than --land-r from the landing
    died             minisim logged the mech's death
    timed_out        none of these: still on the way, or never left the ground, when the window closed
Not counted as attempts: cut_off (the match ended inside the window), no_target (no ledge found for a
hop), already_up (the simulator had the mech on the level at the landing when the climb began).

`sites` groups attempts by landing cell (4 m) and target top (1 m): a mech that jumps at one ledge ten
times is ten attempts but one site. `top_sim` is minisim's own highest surface at or below the target
top (+0.5 m), at the landing point of a jet climb and at the edge of a hop. For a jet climb (a flat roof)
a difference of more than --z-tol means nav and the simulator disagree about the map there
(`map_mismatch`); a hop's ledge is usually sloped terrain, where the two surfaces are sampled
differently, so it is listed but not counted.

Matches are not reproducible run to run (the bot's searches are budgeted in wall-clock time), and the
same seed gives each build the same map, zone and spawns but not the same route: builds are compared
by rate, not attempt by attempt. `missed` / `late` in the per-seed table are our bot's decisions that
minisim did not get inside the 50 ms budget (a loaded machine makes them, and they can cost a climb).

Development tool only; stdlib only.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures as cf
import glob
import hashlib
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRACEBOT = os.path.join(HERE, "tracebot.py")
STEP_UP = 1.2          # config mech.step_up: a drop of more than this below the level is "off it"
LAUNCH_GAP_S = 1.0     # a jet attempt begun in the air this soon after the one before ended is the same launch
FAIL_CLASSES = ("fell_back", "abandoned", "level_elsewhere", "died", "timed_out")
EXCLUDED = ("cut_off", "no_target", "already_up")
EPS = 1e-6


# ============================================================================ judging (pure)
def judge(att, track, t_death, t_close, nav_end, P, floor=None):
    """Outcome of one attempt from the simulator's track.

    att: the CLIMB start record (kind, t, top, land) plus `t_go` (time of the first jump command, None
    for a jet climb that never jumped). track: [(t, x, y, z, grounded)] sorted by t. t_death: minisim's
    death time of the mech or None. t_close: start of the next attempt of the same kind or None.
    nav_end: the CLIMB end record of a jet attempt or None. P: limit_jet, limit_hop, land_r, z_tol, hold.
    floor: minisim's floor_at(x, y, zmax) -> height (None: `off_how` cannot be told).
    Returns {outcome, t_ok, dt, dist, stayed, off_how, air_start, closed_by, ...}."""
    kind = att["kind"]
    t0 = att["t"]
    out = {"outcome": None, "t_ok": None, "dt": None, "dist": None, "closed_by": "limit", "air_start": False}
    if att.get("top") is None or att.get("land") is None:
        out["outcome"] = "no_target"
        return out
    top, (lx, ly) = float(att["top"]), att["land"]
    level = top - P["z_tol"]
    t_go = att.get("t_go")
    # a jet climb may wait at the foot for energy before it jumps (nav does not time that out): its
    # limit runs from the jump command then
    t_ref = t_go if (kind == "jet" and t_go is not None and t_go > t0) else t0
    t1 = t_ref + (P["limit_jet"] if kind == "jet" else P["limit_hop"])
    if t_close is not None and t_close < t1:
        t1, out["closed_by"] = t_close, "next"
    out["t_end"] = round(t1, 2)
    times = [r[0] for r in track]
    i0 = bisect.bisect_left(times, t0 - EPS)
    if i0 < len(track) and abs(track[i0][0] - t0) < 0.051:
        _, x, y, z, g = track[i0]
        out["z0"] = z
        out["air_start"] = not g       # begun in the air: nav restarted a climb inside a flight
        if g and z >= level and math.hypot(x - lx, y - ly) <= P["land_r"]:
            out["outcome"] = "already_up"
            return out
        i0 += 1
    left = False                       # airborne after the jump command
    ev = []                            # (time, class) of the things that end an attempt badly
    zmax = None
    last_t = None
    i_ok = None
    for i in range(i0, len(track)):
        t, x, y, z, g = track[i]
        if t > t1 + EPS:
            break
        last_t = t
        zmax = z if zmax is None or z > zmax else zmax
        d = math.hypot(x - lx, y - ly)
        if g and z >= level:
            if d <= P["land_r"]:
                out.update(outcome="success", t_ok=t, dt=round(t - t0, 2), dist=round(d, 2), z_ok=z)
                i_ok = i
                break
            if not any(c == "level_elsewhere" for _, c in ev):
                ev.append((t, "level_elsewhere"))
                out["dist_elsewhere"] = round(d, 2)
        if t_go is not None and t > t_go + EPS:
            if not g:
                left = True
            elif left and z < level and not any(c == "fell_back" for _, c in ev):
                ev.append((t, "fell_back"))
    out["zmax"] = zmax
    if out["outcome"] == "success":
        # did it stay? (a touch on the roof edge that is off it again two decisions later is not standing there)
        t_hold = out["t_ok"] + P["hold"]
        stayed = None
        j_off = len(track)
        for j in range(i_ok + 1, len(track)):
            if track[j][0] > t_hold + EPS:
                break
            if track[j][3] < level - STEP_UP:
                stayed, out["t_off"], j_off = False, track[j][0], j
                break
        if stayed is None and track[-1][0] >= t_hold - 0.11:
            stayed = True
        out["stayed"] = stayed
        if stayed is False:
            # was it ever standing on the level (minisim's surface under the mech's centre is where its
            # feet are), or only held by its rim on the edge (minisim's edge landing) until it slid off?
            if floor is None:
                out["off_how"] = "unknown"
            else:
                stood = any(g and z >= level and floor(x, y, z + 0.05) >= z - 0.05
                            for _, x, y, z, g in track[i_ok:j_off])
                out["off_how"] = "left" if stood else "rim"
        return out
    if nav_end is not None and nav_end.get("how") != "done" and nav_end["t"] <= t1 + EPS:
        ev.append((nav_end["t"], "abandoned"))
    if t_death is not None and t_death <= t1 + EPS:
        ev.append((t_death, "died"))
    if ev:
        ev.sort(key=lambda e: e[0])
        out["outcome"], out["t_fail"] = ev[0][1], ev[0][0]
    elif last_t is None or last_t < t1 - 0.11:
        out["outcome"] = "cut_off"          # the track stops inside the window and the mech did not die
    else:
        out["outcome"] = "timed_out"
    return out


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 3), round(min(1.0, mid + half), 3)]


# ============================================================================ one match
def parse_match(d, bot_dir, P, geo=None):
    """Attempts (judged) and match facts of the match directory d. Raises ValueError on a bad match."""
    path = os.path.join(d, "match.jsonl")
    if not os.path.exists(path):
        raise ValueError("no match.jsonl")
    meta, death, place, summ = None, None, None, None
    slot = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ty = r.get("type")
            if ty == "meta":
                meta = r
                slot = next((s["slot"] for s in r["slots"] if "tracebot.py" in s["spec"]), None)
            elif ty == "death" and r.get("id") == slot:
                death = r
            elif ty == "place" and r.get("id") == slot:
                place = r
            elif ty == "summary":
                summ = r
    if meta is None or slot is None or summ is None or place is None:
        raise ValueError("match did not finish (meta / traced slot / summary missing)")
    where, starts, ends, jumps, track, errs = None, [], {}, {}, [], 0
    with open(os.path.join(d, f"slot{slot:02d}.stderr"), errors="replace") as f:
        for line in f:
            if line.startswith("CP ["):
                try:
                    v = json.loads(line[3:])
                except ValueError:
                    continue
                if v[1] is not None:
                    track.append((float(v[0]), float(v[1]), float(v[2]), float(v[3]), bool(v[4])))
            elif line.startswith("CLIMB {"):
                try:
                    r = json.loads(line[6:])
                except ValueError:
                    continue
                if r["ev"] == "start":
                    starts.append(r)
                elif r["ev"] == "end":
                    ends[r["id"]] = r
                elif r["ev"] == "jump":
                    jumps.setdefault(r["id"], []).append(r["t"])
            elif line.startswith("TRACEBOT {"):
                where = json.loads(line[9:])
            elif line.startswith(("CLIMB err", "CP err")):
                errs += 1
    if where is None:
        raise ValueError("no TRACEBOT line: the wrapper did not start")
    want = os.path.realpath(bot_dir)
    if (where["dir"] != want or os.path.dirname(where["main"]) != want or os.path.dirname(where["nav"]) != want
            or where["foreign"]):
        raise ValueError(f"wrapper ran other code than {want}: {where}")
    track.sort(key=lambda r: r[0])
    t_death = death["t"] if death else None
    by_kind = collections.defaultdict(list)
    for s in starts:
        by_kind[s["kind"]].append(s)
    atts = []
    for kind, lst in by_kind.items():
        lst.sort(key=lambda s: (s["t"], s["id"]))
        for k, s in enumerate(lst):
            a = dict(s)
            a["seed"] = meta["seed"]
            a["t_go"] = s["t"] if kind == "hop" else (jumps.get(s["id"]) or [None])[0]
            a["jumps"] = 1 if kind == "hop" else len(jumps.get(s["id"], ()))
            end = ends.get(s["id"])
            a["nav_end"] = None if kind == "hop" else (
                "open" if end is None else end["how"] + (":" + str(end["why"]) if end.get("why") else ""))
            a["nav_end_t"] = end["t"] if end else None
            a["zone"] = bool(end.get("zone")) if end else None
            t_close = lst[k + 1]["t"] if k + 1 < len(lst) else None
            a.update(judge(a, track, t_death, t_close, end, P,
                           (lambda x, y, zmax: geo.floor_at(x, y, zmax)[0]) if geo is not None else None))
            if geo is not None and a.get("top") is not None and a.get("land") is not None:
                qx, qy = a["land"]
                if kind == "hop":                     # the edge: `edge` metres from pos towards land
                    px, py = a["pos"][0], a["pos"][1]
                    ln = math.hypot(qx - px, qy - py) or 1.0
                    qx, qy = px + (qx - px) * a["edge"] / ln, py + (qy - py) * a["edge"] / ln
                a["top_sim"] = round(geo.floor_at(qx, qy, a["top"] + 0.5)[0], 2)
                if kind == "jet":
                    a["map_mismatch"] = abs(a["top_sim"] - a["top"]) > P["z_tol"]
            atts.append(a)
    atts.sort(key=lambda a: (a["t"], a["id"]))
    facts = {"seed": meta["seed"], "dir": d, "slot": slot, "mechs": meta["n"], "duration": meta["duration"],
             "peace": bool(meta["opts"].get("peace")), "t_end": summ["t_end"], "death_t": t_death,
             "death_cause": death["cause"] if death else None, "place": place["place"],
             "answered": place["answered"], "missed": place["missed"], "late": place["late"],
             "resp_ms_p99": place["resp_ms_p99"], "decisions_seen": len(track), "trace_errors": errs}
    return atts, facts


# ============================================================================ aggregation
def site_key(a):
    return (a["kind"], int(math.floor(a["land"][0] / 4.0)), int(math.floor(a["land"][1] / 4.0)), int(round(a["top"])))


def tally(atts):
    counted = [a for a in atts if a["outcome"] not in EXCLUDED]
    ok = [a for a in counted if a["outcome"] == "success"]
    stay = [a for a in ok if a.get("stayed")]
    out = {"attempts": len(counted), "successes": len(ok), "rate": round(len(ok) / len(counted), 3) if counted else None,
           "rate_ci95": wilson(len(ok), len(counted)),
           "stayed": len(stay), "stayed_unknown": sum(1 for a in ok if a.get("stayed") is None),
           "rate_stayed": round(len(stay) / len(counted), 3) if counted else None,
           "rate_stayed_ci95": wilson(len(stay), len(counted)),
           "failures": {c: sum(1 for a in counted if a["outcome"] == c) for c in FAIL_CLASSES},
           "excluded": {c: sum(1 for a in atts if a["outcome"] == c) for c in EXCLUDED}}
    off = collections.Counter(a.get("off_how") for a in ok if a.get("stayed") is False)
    out["not_stayed"] = {k: off.get(k, 0) for k in ("rim", "left", "unknown")}
    # launches: nav can begin a jet climb again while the mech is still in the air of the one before
    # (its timeout or energy abort fired in flight): two attempts, one launch
    launches, prev, cur = [], None, None
    for a in sorted(counted, key=lambda a: (a["seed"], a["t"], a["id"])):
        good = a["outcome"] == "success"
        if a["kind"] != "jet":
            launches.append([good, good and bool(a.get("stayed"))])
            continue
        if not (a.get("air_start") and prev is not None and prev["seed"] == a["seed"]
                and (prev.get("nav_end_t") is None or a["t"] - prev["nav_end_t"] <= LAUNCH_GAP_S)):
            cur = [False, False]
            launches.append(cur)
        cur[0] |= good
        cur[1] |= good and bool(a.get("stayed"))
        prev = a
    out["air_starts"] = sum(1 for a in counted if a.get("air_start"))
    out["launches"] = {"n": len(launches), "reached": sum(1 for v in launches if v[0]),
                       "stayed": sum(1 for v in launches if v[1])}
    out["by_kind"] = {}
    for kind in ("jet", "hop"):
        ck = [a for a in counted if a["kind"] == kind]
        out["by_kind"][kind] = {"attempts": len(ck), "successes": sum(1 for a in ck if a["outcome"] == "success"),
                                "stayed": sum(1 for a in ck if a["outcome"] == "success" and a.get("stayed")),
                                "failures": {c: sum(1 for a in ck if a["outcome"] == c) for c in FAIL_CLASSES}}
    sites = collections.defaultdict(lambda: [False, False])
    for a in counted:
        v = sites[(a["seed"],) + site_key(a)]
        v[0] |= a["outcome"] == "success"
        v[1] |= a["outcome"] == "success" and bool(a.get("stayed"))
    out["sites"] = {"n": len(sites), "reached": sum(1 for v in sites.values() if v[0]),
                    "stayed": sum(1 for v in sites.values() if v[1])}
    jets = [a for a in counted if a["kind"] == "jet"]
    out["jet_jumped"] = {"attempts": sum(1 for a in jets if a["t_go"] is not None),
                         "successes": sum(1 for a in jets if a["t_go"] is not None and a["outcome"] == "success")}
    out["nav_end"] = dict(collections.Counter(f"{a['outcome']}|{a['nav_end']}" for a in jets).most_common())
    mm = [a for a in counted if a.get("map_mismatch")]
    out["map_mismatch"] = len(mm) if any("map_mismatch" in a for a in counted) else None
    dts = sorted(a["dt"] for a in ok)
    out["success_dt_s"] = {"median": dts[len(dts) // 2], "max": dts[-1]} if dts else None
    ds = sorted(a["dist"] for a in ok)
    out["success_dist_m"] = {"median": ds[len(ds) // 2], "max": ds[-1]} if ds else None
    return out


def fmt_rate(k, n):
    return f"{k}/{n}" + (f" ({100.0 * k / n:.0f}%)" if n else "")


def report_build(label, b, hold):
    T = b["total"]
    L = [f"== {label}: {b['dir']}",
         f"   code md5 {b['code_md5']}   matches {len(b['per_seed'])}   seeds {b['seeds']}"]
    if b.get("errors"):
        L.append("   MATCH ERRORS: " + "; ".join(f"seed {s}: {e}" for s, e in b["errors"].items()))
    terr = [(r["facts"]["seed"], r["facts"]["trace_errors"]) for r in b["per_seed"] if r["facts"]["trace_errors"]]
    if terr:
        L.append("   WRAPPER ERRORS (CLIMB err / CP err lines: attempts or positions may be missing): "
                 + ", ".join(f"seed {s}: {n}" for s, n in terr))
    ci, cs = T["rate_ci95"], T["rate_stayed_ci95"]
    L.append(f"   attempts {T['attempts']}   successes {T['successes']}   rate "
             + (f"{T['rate']:.3f} (95% CI {ci[0]:.2f}-{ci[1]:.2f})" if T["attempts"] else "n/a"))
    L.append(f"   and stayed on the level for {hold:g} s: {T['stayed']}   rate "
             + (f"{T['rate_stayed']:.3f} (95% CI {cs[0]:.2f}-{cs[1]:.2f})" if T["attempts"] else "n/a")
             + (f"   (track ended first: {T['stayed_unknown']})" if T["stayed_unknown"] else ""))
    ns = T["not_stayed"]
    if sum(ns.values()):
        L.append(f"     the other {sum(ns.values())}: only on the rim of the level and slid off {ns['rim']}, "
                 f"stood on it and left it again {ns['left']}" + (f", not known {ns['unknown']}" if ns["unknown"] else ""))
    for kind in ("jet", "hop"):
        k = T["by_kind"][kind]
        L.append(f"     {kind}: ok {fmt_rate(k['successes'], k['attempts'])}, stayed {k['stayed']}   failures: "
                 + (", ".join(f"{c} {n}" for c, n in k["failures"].items() if n) or "none"))
    L.append("   failures by class: " + ", ".join(f"{c} {T['failures'][c]}" for c in FAIL_CLASSES))
    L.append("   not counted: " + ", ".join(f"{c} {T['excluded'][c]}" for c in EXCLUDED))
    L.append(f"   launches ({T['air_starts']} attempts began in the air, inside the flight of the one before): "
             f"{fmt_rate(T['launches']['reached'], T['launches']['n'])} reached, {T['launches']['stayed']} stayed")
    L.append(f"   sites (landing cell x top): {fmt_rate(T['sites']['reached'], T['sites']['n'])} reached, "
             f"{T['sites']['stayed']} stayed"
             f"   jet climbs that got to their jump: {fmt_rate(T['jet_jumped']['successes'], T['jet_jumped']['attempts'])}")
    if T["nav_end"]:
        L.append("   jet outcome | how nav ended it: " + ", ".join(f"{k} {v}" for k, v in T["nav_end"].items()))
    extra = []
    if T["success_dt_s"]:
        extra.append(f"success after median {T['success_dt_s']['median']} s (max {T['success_dt_s']['max']} s), "
                     f"{T['success_dist_m']['median']} m from the landing point (max {T['success_dist_m']['max']} m)")
    if T["map_mismatch"] is not None:
        extra.append(f"jet target top vs minisim's surface at the landing point differs by more than z-tol: "
                     f"{T['map_mismatch']} of {T['by_kind']['jet']['attempts']}")
    for e in extra:
        L.append("   " + e)
    L.append(f"   {'seed':>6s} {'att':>4s} {'ok':>4s} {'stay':>4s} {'rate':>5s} | {'jet':>7s} {'hop':>7s} | "
             f"{'fell':>4s} {'aban':>4s} {'else':>4s} {'died':>4s} {'tout':>4s} | {'excl':>4s} | "
             f"{'missed':>6s} {'late':>4s} {'p99ms':>6s} {'death_t':>7s}")
    for r in b["per_seed"]:
        t, f = r["tally"], r["facts"]
        fl = t["failures"]
        jk, hk = t["by_kind"]["jet"], t["by_kind"]["hop"]
        L.append(f"   {f['seed']:6d} {t['attempts']:4d} {t['successes']:4d} {t['stayed']:4d} "
                 f"{(('%.2f' % t['rate']) if t['attempts'] else '-'):>5s} | "
                 f"{str(jk['successes']) + '/' + str(jk['attempts']):>7s} {str(hk['successes']) + '/' + str(hk['attempts']):>7s} | "
                 f"{fl['fell_back']:4d} {fl['abandoned']:4d} {fl['level_elsewhere']:4d} {fl['died']:4d} {fl['timed_out']:4d} | "
                 f"{sum(t['excluded'].values()):4d} | {f['missed']:6d} {f['late']:4d} "
                 f"{(f['resp_ms_p99'] if f['resp_ms_p99'] is not None else 0):6.1f} "
                 f"{(('%.1f' % f['death_t']) if f['death_t'] is not None else '-'):>7s}")
    return "\n".join(L)


def report_compare(builds):
    L = ["== comparison (same seeds)",
         f"   {'build':24s} {'attempts':>8s} {'ok':>4s} {'rate':>6s} {'95% CI':>11s} {'stay':>4s} {'rate':>6s} | "
         f"{'jet':>9s} {'hop':>9s} | {'sites':>9s} | {'fell':>4s} {'aban':>4s} {'else':>4s} {'died':>4s} {'tout':>4s}"]
    for label, b in builds.items():
        T = b["total"]
        ci = T["rate_ci95"]
        fl = T["failures"]
        L.append(f"   {label[:24]:24s} {T['attempts']:8d} {T['successes']:4d} "
                 f"{(('%.3f' % T['rate']) if T['attempts'] else '-'):>6s} "
                 f"{(('%.2f-%.2f' % (ci[0], ci[1])) if ci else '-'):>11s} {T['stayed']:4d} "
                 f"{(('%.3f' % T['rate_stayed']) if T['attempts'] else '-'):>6s} | "
                 f"{str(T['by_kind']['jet']['successes']) + '/' + str(T['by_kind']['jet']['attempts']):>9s} "
                 f"{str(T['by_kind']['hop']['successes']) + '/' + str(T['by_kind']['hop']['attempts']):>9s} | "
                 f"{str(T['sites']['reached']) + '/' + str(T['sites']['n']):>9s} | "
                 f"{fl['fell_back']:4d} {fl['abandoned']:4d} {fl['level_elsewhere']:4d} {fl['died']:4d} {fl['timed_out']:4d}")
    return "\n".join(L)


# ============================================================================ running
def code_md5(bot_dir):
    h = hashlib.md5()
    for p in sorted(glob.glob(os.path.join(bot_dir, "*.py"))):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def run_one(a, label, bot_dir, seed):
    out = os.path.join(a.out, label, f"s{seed}")
    os.makedirs(out, exist_ok=True)
    q = shlex.quote
    spec = (f"TRACEBOT_DIR={q(bot_dir)} TRACE_POS=1 TRACE_EVERY_S=1000000 "
            f"{q(sys.executable)} -u {q(TRACEBOT)}")
    cmd = [sys.executable, a.sim, "--out", out, "--seed", str(seed), "--quiet", "--kit", a.kit, "--bot", spec]
    if not a.no_peace:
        cmd.append("--peace")
    if a.duration:
        cmd += ["--duration", str(a.duration)]
    if a.mechs:
        cmd += ["--mechs", str(a.mechs)]
    if a.fill:
        cmd += ["--fill", a.fill]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"       # the build directories under test stay as they are
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return label, seed, out, p.returncode, time.perf_counter() - t0, p.stderr[-1500:]


def load_geo(sim, kit):
    """minisim's own map model (for top_sim), or None when it cannot be loaded."""
    try:
        spec = importlib.util.spec_from_file_location("_climbcheck_minisim", sim)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        dont = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = dont
        return mod.get_geo(os.path.join(os.path.abspath(kit), "arena"))
    except Exception as e:  # noqa: BLE001
        print(f"climbcheck: minisim geometry not loaded ({e!r}); top_sim is left out", file=sys.stderr)
        return None


def parse_bots(specs):
    bots = collections.OrderedDict()
    for s in specs:
        label, d = (s.split("=", 1) if "=" in s and not os.path.isdir(s) else (None, s))
        d = os.path.realpath(d)
        if not os.path.isfile(os.path.join(d, "main.py")):
            sys.exit(f"climbcheck: {d} has no main.py")
        label = label or os.path.basename(d.rstrip(os.sep)) or "bot"
        base, k = label, 2
        while label in bots:
            label, k = f"{base}_{k}", k + 1
        bots[label] = d
    return bots


def selftest():
    P = {"limit_jet": 15.0, "limit_hop": 3.0, "land_r": 6.0, "z_tol": 0.5, "hold": 1.0}

    def tr(rows):                      # (t, x, z, grounded): y = 0
        return [(t, x, 0.0, z, g) for t, x, z, g in rows]

    def steps(t0, t1, fn):
        return [fn(round(t0 + 0.1 * i, 2)) for i in range(int(round((t1 - t0) / 0.1)) + 1)]

    jet = {"kind": "jet", "t": 10.0, "top": 12.0, "land": [5.0, 0.0], "t_go": 10.2}
    up = tr([(10.0, 0.0, 0.0, True), (10.1, 0.0, 0.0, True), (10.2, 0.0, 0.0, True)]
            + steps(10.3, 11.6, lambda t: (t, 0.0, (t - 10.2) * 9.0, False))
            + [(11.7, 3.0, 12.0, True), (11.8, 4.0, 12.0, True)] + steps(11.9, 14.0, lambda t: (t, 5.0, 12.0, True)))
    r = judge(jet, up, None, None, {"how": "done", "t": 11.7}, P)
    assert r["outcome"] == "success" and r["t_ok"] == 11.7 and r["dist"] == 2.0 and r["stayed"] is True, r
    # touches the roof edge, is off it two decisions later and back on the ground: a success that did not stay
    slid = up[:up.index((11.8, 4.0, 0.0, 12.0, True)) + 1] + tr(
        [(11.9, 2.0, 11.9, False), (12.0, 1.0, 11.0, False), (12.1, 0.0, 9.5, False), (12.5, 0.0, 0.0, True)]
        + steps(12.6, 14.0, lambda t: (t, 0.0, 0.0, True)))
    r = judge(jet, slid, None, None, {"how": "done", "t": 11.7}, P)
    assert r["outcome"] == "success" and r["stayed"] is False and r["t_off"] == 12.1, r
    assert judge(jet, up[:up.index((11.8, 4.0, 0.0, 12.0, True)) + 1], None, None, None, P)["stayed"] is None
    # jumps twice in place, nav gives up: the first thing the simulator shows is the fall back
    fall = tr([(10.0, 0.0, 0.0, True), (10.2, 0.0, 0.0, True), (10.3, 0.0, 1.0, False), (10.7, 0.0, 2.7, False),
               (11.2, 0.0, 0.0, True), (11.3, 0.0, 1.0, False), (12.2, 0.0, 0.0, True)]
              + steps(12.3, 26.0, lambda t: (t, 0.0, 0.0, True)))
    r = judge(jet, fall, None, None, {"how": "abort", "why": "retry", "t": 12.2}, P)
    assert r["outcome"] == "fell_back" and r["t_fail"] == 11.2, r
    # never jumps: nav times the approach out
    flat = tr(steps(10.0, 26.0, lambda t: (t, 0.0, 0.0, True)))
    r = judge(dict(jet, t_go=None), flat, None, None, {"how": "abort", "why": "timeout", "t": 25.5}, P)
    assert r["outcome"] == "timed_out", r           # the abort came after the 15 s window
    r = judge(dict(jet, t_go=None), flat, None, None, {"how": "abort", "why": "timeout", "t": 14.0}, P)
    assert r["outcome"] == "abandoned", r
    r = judge(dict(jet, t_go=None), flat, 13.0, None, {"how": "abort", "why": "timeout", "t": 14.0}, P)
    assert r["outcome"] == "died", r
    r = judge(dict(jet, t_go=None), flat[:30], None, None, None, P)
    assert r["outcome"] == "cut_off", r
    # at the height but 9 m from the landing point
    far = tr([(10.0, 0.0, 0.0, True), (10.3, 0.0, 3.0, False), (11.7, -4.0, 12.0, True)]
             + steps(11.8, 26.0, lambda t: (t, -4.0, 12.0, True)))
    r = judge(jet, far, None, None, {"how": "done", "t": 11.7}, P)
    assert r["outcome"] == "level_elsewhere" and r["dist_elsewhere"] == 9.0, r
    assert judge(jet, far, None, None, None, dict(P, land_r=9.5))["outcome"] == "success"
    # the level is reached only after the next jet climb began: not this attempt's success
    r = judge(jet, up, None, 11.0, {"how": "replaced", "t": 11.0}, P)
    assert r["outcome"] == "abandoned" and r["closed_by"] == "next", r
    # half a metre short of the top is not the level; 0.5 m below counts
    low = [(t, x, y, min(z, 11.4), g) for t, x, y, z, g in up]
    assert judge(jet, low, None, None, None, P)["outcome"] != "success"
    assert judge(jet, [(t, x, y, min(z, 11.5), g) for t, x, y, z, g in up], None, None, None, P)["outcome"] == "success"
    hop = {"kind": "hop", "t": 5.0, "top": 2.0, "land": [4.0, 0.0], "t_go": 5.0}
    h_ok = tr([(5.0, 0.0, 0.0, True), (5.1, 0.5, 0.5, False), (5.5, 2.5, 2.6, False), (5.8, 4.0, 2.0, True)]
              + steps(5.9, 9.0, lambda t: (t, 6.0, 2.0, True)))
    assert judge(hop, h_ok, None, None, None, P)["outcome"] == "success"
    h_bad = tr([(5.0, 0.0, 0.0, True), (5.1, 0.0, 0.5, False), (5.5, 0.0, 2.6, False), (6.0, 0.0, 0.0, True)]
               + steps(6.1, 9.0, lambda t: (t, 0.0, 0.0, True)))
    assert judge(hop, h_bad, None, None, None, P)["outcome"] == "fell_back"
    h_none = tr(steps(5.0, 9.0, lambda t: (t, 0.0, 0.0, True)))
    assert judge(hop, h_none, None, None, None, P)["outcome"] == "timed_out"
    assert judge(dict(hop, top=None, land=None), h_none, None, None, None, P)["outcome"] == "no_target"
    assert judge(hop, tr([(5.0, 4.0, 2.0, True), (5.1, 4.0, 2.0, True)]), None, None, None, P)["outcome"] == "already_up"
    # waits 20 s at the foot for energy, then climbs in 1.5 s: the limit runs from the jump command
    wait = tr(steps(10.0, 30.0, lambda t: (t, 0.0, 0.0, True))
              + steps(30.1, 31.4, lambda t: (t, 0.0, (t - 30.0) * 9.0, False))
              + steps(31.5, 34.0, lambda t: (t, 5.0, 12.0, True)))
    r = judge(dict(jet, t_go=30.0), wait, None, None, {"how": "done", "t": 31.5}, P)
    assert r["outcome"] == "success" and r["t_ok"] == 31.5 and r["t_end"] == 45.0, r
    assert judge(dict(jet, t_go=None), wait, None, None, None, P)["outcome"] == "timed_out"
    # the roof is x >= 3.5: at x = 3 the mech hangs on the edge by its rim, at x = 4 it stands on the roof
    roof = lambda x, y, zmax: 12.0 if (x >= 3.5 and zmax >= 12.0) else 0.0      # noqa: E731
    r = judge(jet, slid, None, None, {"how": "done", "t": 11.7}, P, roof)
    assert r["stayed"] is False and r["off_how"] == "left", r
    rim = [q for q in slid if q[0] != 11.8]
    r = judge(jet, rim, None, None, {"how": "done", "t": 11.7}, P, roof)
    assert r["outcome"] == "success" and r["stayed"] is False and r["off_how"] == "rim", r
    assert judge(jet, rim, None, None, None, P)["off_how"] == "unknown"
    assert "off_how" not in judge(jet, up, None, None, None, P, roof)
    # nav begins the climb again in the air (its timeout fired in flight): two attempts, one launch
    air = dict(jet, t=11.0, t_go=11.0)
    r1 = judge(jet, up, None, 11.0, {"how": "abort", "why": "timeout", "t": 11.0}, P)
    r2 = judge(air, up, None, None, {"how": "done", "t": 11.7}, P)
    assert (r1["outcome"], r1["air_start"], r2["outcome"], r2["air_start"]) == ("abandoned", False, "success", True)
    T = tally([dict(jet, id=1, seed=1, nav_end="abort:timeout", nav_end_t=11.0, **r1),
               dict(air, id=2, seed=1, nav_end="done", nav_end_t=11.7, **r2),
               dict(air, id=1, seed=2, nav_end="done", nav_end_t=11.7, **r2)])      # other match: its own launch
    assert (T["attempts"], T["successes"], T["air_starts"]) == (3, 2, 2), T
    assert T["launches"] == {"n": 2, "reached": 2, "stayed": 2}, T
    atts = [dict(jet, id=1, seed=1, nav_end="done", **judge(jet, up, None, None, None, P)),
            dict(jet, id=2, seed=1, nav_end="abort:retry", **judge(jet, fall, None, None, {"how": "abort", "t": 12.2}, P)),
            dict(hop, id=3, seed=1, nav_end=None, **judge(hop, h_ok, None, None, None, P)),
            dict(jet, id=4, seed=1, nav_end="done", **judge(jet, rim, None, None, None, P, roof))]
    T = tally(atts)
    assert (T["attempts"], T["successes"], T["stayed"], T["failures"]["fell_back"]) == (4, 3, 2, 1), T
    assert T["not_stayed"] == {"rim": 1, "left": 0, "unknown": 0}, T
    assert T["sites"] == {"n": 2, "reached": 2, "stayed": 2}, T
    assert T["launches"] == {"n": 4, "reached": 3, "stayed": 2} and T["air_starts"] == 0, T
    assert wilson(0, 0) is None and wilson(5, 10) == [0.237, 0.763]
    print("climbcheck selftest: ok")


def main():
    ap = argparse.ArgumentParser(description="Climb success of our bot inside minisim (see the module docstring).")
    ap.add_argument("--bot", action="append", help="bot directory with main.py, or label=dir (repeatable)")
    ap.add_argument("--seeds", type=int, default=4, help="number of matches per build (default 4)")
    ap.add_argument("--seed0", type=int, default=800, help="first seed (default 800)")
    ap.add_argument("--duration", type=float, default=0, help="match length in s (default: the config's 480)")
    ap.add_argument("--out", default=None, help="output directory (default: a new temporary directory)")
    ap.add_argument("-j", type=int, default=2, help="matches in parallel (default 2)")
    ap.add_argument("--mechs", type=int, default=0, help="mechs per match (default: the config's 16)")
    ap.add_argument("--fill", default=None, help="minisim --fill spec for the other slots (default scripted:mix)")
    ap.add_argument("--no-peace", action="store_true", help="weapons do damage (default: minisim --peace)")
    ap.add_argument("--sim", default=os.path.join(HERE, "minisim.py"), help="the minisim.py to run")
    ap.add_argument("--kit", default=os.path.join(os.path.dirname(REPO), "mechfight-kit"), help="mechfight-kit directory")
    ap.add_argument("--limit-jet", type=float, default=15.0, help="time limit of a jet climb, s from its start")
    ap.add_argument("--limit-hop", type=float, default=3.0, help="time limit of a hop, s from its start")
    ap.add_argument("--land-r", type=float, default=6.0, help="horizontal distance to the landing point, m")
    ap.add_argument("--z-tol", type=float, default=0.5, help="how far below the target top still counts, m")
    ap.add_argument("--hold", type=float, default=1.0, help="`stayed`: s on the level after the first touch")
    ap.add_argument("--rejudge", action="store_true", help="judge the matches already in --out, run nothing")
    ap.add_argument("--selftest", action="store_true", help="check the judging rules on made-up tracks and exit")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.bot:
        ap.error("--bot is required")
    bots = parse_bots(a.bot)
    a.sim = os.path.abspath(a.sim)
    a.kit = os.path.abspath(a.kit)
    if a.out is None:
        if a.rejudge:
            ap.error("--rejudge needs --out")
        a.out = tempfile.mkdtemp(prefix="climbcheck_")
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    P = {"limit_jet": a.limit_jet, "limit_hop": a.limit_hop, "land_r": a.land_r, "z_tol": a.z_tol, "hold": a.hold}
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    run_errors = collections.defaultdict(dict)
    if not a.rejudge:
        jobs = [(label, d, s) for s in seeds for label, d in bots.items()]      # builds interleaved per seed
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=max(1, a.j)) as ex:
            futs = [ex.submit(run_one, a, label, d, s) for label, d, s in jobs]
            for fu in cf.as_completed(futs):
                label, seed, out, rc, wall, err = fu.result()
                print(f"climbcheck: {label} seed {seed}: rc={rc} wall={wall:.0f}s", file=sys.stderr)
                if rc != 0:
                    run_errors[label][seed] = f"minisim rc={rc}: {err[-300:]}"
        print(f"climbcheck: {len(jobs)} matches in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    geo = load_geo(a.sim, a.kit)
    builds = collections.OrderedDict()
    for label, d in bots.items():
        if a.rejudge:
            found = sorted(int(os.path.basename(p)[1:]) for p in glob.glob(os.path.join(a.out, label, "s[0-9]*")))
        else:
            found = seeds
        b = {"dir": d, "code_md5": code_md5(d), "seeds": found, "per_seed": [], "errors": dict(run_errors.get(label, {})),
             "attempts": []}
        for s in found:
            try:
                atts, facts = parse_match(os.path.join(a.out, label, f"s{s}"), d, P, geo)
            except (ValueError, OSError, KeyError) as e:
                b["errors"][s] = f"{b['errors'].get(s, '')} {e}".strip()
                continue
            b["per_seed"].append({"facts": facts, "tally": tally(atts)})
            b["attempts"] += atts
        b["total"] = tally(b["attempts"])
        b["errors"] = {str(k): v for k, v in b["errors"].items()}
        builds[label] = b
    txt = "\n\n".join(report_build(label, b, a.hold) for label, b in builds.items())
    head = (f"climbcheck: sim {a.sim}\n"
            f"   limits jet {a.limit_jet:g} s, hop {a.limit_hop:g} s; landing radius {a.land_r:g} m; z tolerance {a.z_tol:g} m; "
            f"hold {a.hold:g} s; "
            f"{'weapons on' if a.no_peace else 'peace'}; mechs {a.mechs or 'config'}; duration {a.duration or 'config'}")
    txt = head + "\n\n" + txt
    if len(builds) > 1:
        txt += "\n\n" + report_compare(builds)
    print(txt)
    summ = {"params": dict(P, seeds=seeds if not a.rejudge else None, peace=not a.no_peace, mechs=a.mechs or None,
                           duration=a.duration or None, fill=a.fill, sim=a.sim, kit=a.kit, tracebot=TRACEBOT),
            "builds": builds}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nclimbcheck: wrote {os.path.join(a.out, 'summary.json')}", file=sys.stderr)
    return 1 if any(b["errors"] for b in builds.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
