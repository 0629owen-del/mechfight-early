#!/usr/bin/env python3
"""Our bot plus a per-decision trace on stderr, for reading one bad match tick by tick.

    python3 tools/minisim.py --out runs/t1 --seed 503 --mechs 2 --bot "python3 -u tools/tracebot.py"
    grep '^TR ' runs/t1/slotNN.stderr        # one JSON line every TRACE_EVERY_S of match time
    grep '^NAVFIN ' runs/t1/slotNN.stderr    # one JSON line per finished path search

TR: t, hp, pos, label, leg owner, move, boost, energy, in_zone, zone slack, stuck step, goal / sub-state,
drops [id, pos, lands_in, progress, holder, distance, seconds since landing], path jobs
{key: [complete, points, search running, failed, stale]}.
NAVFIN: key, found, expansions, start, goal, blocked edges, path points.

Climbs (read by tools/climbcheck.py), one JSON line per event, `t` = the observation time of the decision:

    grep '^CLIMB ' runs/t1/slotNN.stderr

    {"ev":"start","id":N,"kind":"jet","t":..,"top":..,"land":[x,y],"stand":[x,y],"h":..,"pos":[x,y,z],"e":..,"src":..}
        nav began a climb sequence (Nav._climb_begin): top = the height it climbs to, land = the landing
        point on the upper level, stand = the foot point, h = the face height, src = path | direct (the
        zone field or a caller without a path job).
    {"ev":"start","id":N,"kind":"hop","t":..,"top":..,"land":[x,y],"edge":..,"rise":..,"pos":[x,y,z],"e":..,"rule":..}
        a nav Steer asked for a jump up a ledge in (STEP_UP, JUMP_H] (path jump edge, zone field, interim
        steering). One line per jump: a jump still being asked for on the next decision is the same hop.
        top = nav's surface height just past the edge, edge = metres to the edge along the steer
        direction, land = the point one mech radius past the edge.
    {"ev":"jump","id":N,"t":..,"n":k}        the k-th jump command of jet climb N (k = 2: its one retry)
    {"ev":"end","id":N,"t":..,"how":"done|abort|dropped|replaced","why":..,"phase":..,"tries":..,"zone":..}
        how nav ended jet climb N: done (it saw itself grounded at the top), abort (why = timeout | retry |
        energy | phase), dropped (the climb was cleared without either, e.g. an idle zone climb), replaced
        (another climb began). A hop has no end on the nav side.
    CL [t, legs owner ("level:rule"), action label, move x, move y, boost, jump]
        one line per decision while a jet climb runs and for CL_AFTER_S (3 s) after it ended or a hop began:
        who had the legs and what went out, to see why a climb failed or the mech left the level again.

Environment:
    TRACE_EVERY_S   seconds of match time between TR lines (default 2)
    TRACE_POS=1     also one `CP [t,x,y,z,grounded,hp,energy]` line per decision, copied from the raw
                    observation (`self`), i.e. the simulator's own position of the mech
    TRACEBOT_DIR    import the bot (main, nav, runner, ...) from this directory instead of the repo this
                    file is in; the first stderr line `TRACEBOT {"dir":..,"main":..,"nav":..,"foreign":[..]}`
                    says where the code came from (foreign = modules loaded from this repo all the same)

Development tool only (it patches main.Bot.act and nav.Nav._finish_job, _climb_begin, _climb_done,
_climb_abort, follow, zone_steer, climb_steer); the match build never loads it.
"""
import json
import math
import os
import sys

_REPO = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOT_DIR = os.path.realpath(os.environ.get("TRACEBOT_DIR") or _REPO)
sys.path.insert(0, BOT_DIR)
import main  # noqa: E402
import nav as NAV  # noqa: E402
import runner  # noqa: E402

TRACE_EVERY_S = float(os.environ.get("TRACE_EVERY_S", "2"))
TRACE_POS = os.environ.get("TRACE_POS", "") not in ("", "0")
_act = main.Bot.act
_finish = NAV.Nav._finish_job
_begin, _done, _abort = NAV.Nav._climb_begin, NAV.Nav._climb_done, NAV.Nav._climb_abort
# climb trace state: n = last id, t = observation time of the decision being computed, open = the jet climb
# nav is running ({id, obj: nav._climb, jumps, jump_t}), hop_prev / hop_now = a hop jump was asked for on the
# previous / this decision
_cl = {"n": 0, "t": 0.0, "open": None, "hop_prev": False, "hop_now": False, "until": -1.0}
CL_AFTER_S = 3.0


def _emit(tag, rec):
    sys.stderr.write(tag + " " + json.dumps(rec, separators=(",", ":"), default=str) + "\n")


def _where():
    """Where the bot code came from (TRACEBOT_DIR must not be mixed with this repo's modules)."""
    foreign = []
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or name == "__main__":
            continue
        f = os.path.realpath(f)
        if f.startswith(_REPO + os.sep) and not f.startswith(BOT_DIR + os.sep):
            foreign.append(name)
    return {"dir": BOT_DIR, "main": os.path.realpath(main.__file__), "nav": os.path.realpath(NAV.__file__),
            "foreign": sorted(foreign)}


def _r(v, n=2):
    return [round(float(x), n) for x in v]


def _climb_end(how, why=None):
    o = _cl["open"]
    if o is None:
        return
    _cl["open"] = None
    _cl["until"] = _cl["t"] + CL_AFTER_S
    cl = o["obj"]
    _emit("CLIMB", {"ev": "end", "id": o["id"], "t": _cl["t"], "how": how, "why": why, "phase": cl.get("phase"),
                    "tries": cl.get("tries"), "zone": bool(cl.get("zone"))})


def climb_begin(self, rec, job, after_idx):
    r = _begin(self, rec, job, after_idx)
    try:
        _climb_end("replaced")               # a climb still open here ended without done / abort
        cl = self._climb
        _cl["n"] += 1
        _cl["open"] = {"id": _cl["n"], "obj": cl, "jumps": 0, "jump_t": None}
        _emit("CLIMB", {"ev": "start", "id": _cl["n"], "kind": "jet", "t": _cl["t"], "top": round(float(cl["top"]), 2),
                        "land": _r((rec["rx"], rec["ry"])), "stand": _r((rec["x"], rec["y"])),
                        "h": round(float(rec["h"]), 2), "pos": _r(self._pos) if self._pos is not None else None,
                        "e": round(float(self.energy), 1), "src": "path" if job is not None else "direct"})
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("CLIMB err %r\n" % (e,))
    return r


def climb_done(self):
    try:
        _climb_end("done")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("CLIMB err %r\n" % (e,))
    return _done(self)


def climb_abort(self, why):
    try:
        _climb_end("abort", why)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("CLIMB err %r\n" % (e,))
    return _abort(self, why)


def _ledge(nav, pos, d):
    """(rise, metres to the edge, surface height past it) of the first ledge along d: the scan of
    Nav._ledge_ahead, which only returns the rise."""
    z = nav._h1(pos[0], pos[1])
    if z is None:
        return None
    if len(pos) > 2:
        z = max(z, pos[2])
    reach = nav.radius + nav._JUMP_TRIGGER_DIST_M
    s = 0.5
    while s <= reach + 1e-9:
        h = nav._h1(pos[0] + d[0] * s, pos[1] + d[1] * s)
        if h is None:
            return None
        if h - z > nav._STEP_UP + 0.05:
            return h - z, s, h
        s += 0.5
    return None


def _on_steer(nav, me, st):
    """A Steer nav handed out with jump set: the jump of the running jet climb, or a hop up a ledge."""
    if st is None or not getattr(st, "jump", False) or me is None:
        return
    if st.rule == "R13.2j":
        o = _cl["open"]
        if o is not None and nav._climb is o["obj"] and o["jump_t"] != _cl["t"]:   # (zone_steer caches per tick)
            o["jumps"] += 1
            o["jump_t"] = _cl["t"]
            _emit("CLIMB", {"ev": "jump", "id": o["id"], "t": _cl["t"], "n": o["jumps"]})
        return
    d = st.direction
    if d is None:
        return
    first = not (_cl["hop_prev"] or _cl["hop_now"])
    _cl["hop_now"] = True
    if not first:
        return
    pos = tuple(me.pos)
    led = _ledge(nav, pos, d)
    _cl["n"] += 1
    _cl["until"] = max(_cl["until"], _cl["t"] + CL_AFTER_S)
    rec = {"ev": "start", "id": _cl["n"], "kind": "hop", "t": _cl["t"]}
    if led is None:
        rec.update(top=None, land=None, edge=None, rise=None)
    else:
        rise, s, h = led
        reach = s + nav.radius
        rec.update(top=round(h, 2), land=_r((pos[0] + d[0] * reach, pos[1] + d[1] * reach)), edge=s,
                   rise=round(rise, 2))
    rec.update(pos=_r(pos), e=round(float(me.energy), 1), rule=st.rule)
    _emit("CLIMB", rec)


def _steer_hook(fn):
    def hooked(self, me, *a, **k):
        st = fn(self, me, *a, **k)
        try:
            _on_steer(self, me, st)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("CLIMB err %r\n" % (e,))
        return st
    hooked.__name__ = getattr(fn, "__name__", "hooked")
    return hooked


def finish(self, job, sl=None):
    s = job.search
    info = {"key": job.key, "found": bool(s.found), "exp": s.exp, "max": s.max_exp, "t": round(self._t, 1),
            "start": [round(v, 1) for v in job.start], "goal": [round(v, 1) for v in job.goal],
            "blocked": len(self._blocked), "cost": sorted(k for k, v in job.cost.items() if v)}
    r = _finish(self, job, sl)
    info["pts"] = len(job.path.points) if job.path else None
    sys.stderr.write("NAVFIN " + json.dumps(info, separators=(",", ":")) + "\n")
    return r


def act(self, obs, t0):
    try:
        me0 = obs.get("self") or {}
        _cl["t"] = float(obs.get("t", 0.0) or 0.0)
        _cl["hop_prev"], _cl["hop_now"] = _cl["hop_now"], False
        if TRACE_POS:
            p = me0.get("pos") or (None, None, None)
            sys.stderr.write("CP " + json.dumps([_cl["t"], p[0], p[1], p[2], 1 if me0.get("grounded") else 0,
                                                 me0.get("hp"), me0.get("energy")], separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("CP err %r\n" % (e,))
    a = _act(self, obs, t0)
    try:
        o = _cl["open"]
        if o is not None and (self.nav is None or self.nav._climb is not o["obj"]):
            _climb_end("dropped")            # cleared without _climb_done / _climb_abort
        if _cl["open"] is not None or _cl["t"] <= _cl["until"]:
            mv = a.get("move") or [0, 0]
            _emit("CL", [_cl["t"], self.arbiter.last_owners.get("legs"), (a.get("debug") or {}).get("label"),
                         round(mv[0], 2), round(mv[1], 2), 1 if a.get("boost") else 0, 1 if a.get("jump") else 0])
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("CLIMB err %r\n" % (e,))
    try:
        w = self.world
        if w is None:
            return a
        me, t = w.me, w.tick.t
        if int(round(t * 10)) % max(1, int(round(TRACE_EVERY_S * 10))) == 0:
            rec = {"t": t, "hp": round(me.hp), "pos": [round(v, 1) for v in me.pos],
                   "lab": (a.get("debug") or {}).get("label"), "own": self.arbiter.last_owners.get("legs"),
                   "mv": [round(x, 2) for x in a.get("move", [0, 0])], "boost": a.get("boost"),
                   "e": round(me.energy), "in_zone": me.in_zone, "slack": round(self._A.slack, 1),
                   "stuck": self.nav._stuck.get("step_name") if self.nav._stuck.get("active") else None,
                   "goal": self.modes.goal, "sub": self.modes.sub,
                   "drops": [[d.id, [round(v, 1) for v in d.pos], round(d.lands_in, 1), round(d.progress, 1), d.holder,
                              round(math.hypot(d.pos[0] - me.pos[0], d.pos[1] - me.pos[1]), 1),
                              round(t - d.lands_at, 1)] for d in w.drops.values()],
                   "jobs": {k: [j.path is not None and j.path.complete, j.path is not None and len(j.path.points),
                                j.search is not None and not j.search.done, bool(j.fail), bool(j.stale)]
                            for k, j in self.nav._paths.items()}}
            sys.stderr.write("TR " + json.dumps(rec, separators=(",", ":"), default=str) + "\n")
    except Exception as e:  # noqa: BLE001  (a trace must never change the bot's answer)
        sys.stderr.write("TR err %r\n" % (e,))
    return a


NAV.Nav._finish_job = finish
NAV.Nav._climb_begin, NAV.Nav._climb_done, NAV.Nav._climb_abort = climb_begin, climb_done, climb_abort
for _name in ("follow", "zone_steer", "climb_steer"):      # every nav call that returns a Steer for our mech
    if hasattr(NAV.Nav, _name):
        setattr(NAV.Nav, _name, _steer_hook(getattr(NAV.Nav, _name)))
main.Bot.act = act
_emit("TRACEBOT", _where())

if __name__ == "__main__":
    sys.exit(runner.run(main.Bot))
