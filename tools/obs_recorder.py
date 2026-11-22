#!/usr/bin/env python3
"""Observer harness for minisim calibration (stdlib only).

Runs the kit's example bot (`mechfight-kit/python/example_bot.py`) UNCHANGED and copies every message
it receives to stderr as one compact JSON line, so a minisim match yields the same kind of stream as the
one real recording (`mechfight-kit/check/stream.jsonl.gz`, slot 3 = the example bot):

    python3 tools/minisim.py --out runs/c1 --bot "python3 -u tools/obs_recorder.py" --fill scripted:mix
    python3 tools/calibrate.py runs/cal_batch          # observer metrics vs the real recording

minisim stores each slot's stderr as `<out>/slotNN.stderr`. Lines written here:
    {"type":"hello","slot":..,"seed":..}           (config dropped: it is the recorded config)
    {"type":"obs",...}                             every observation while t <= OBS_REC_UNTIL, with the
                                                   static `self.stats` and `weapons[i].stats` dropped
    {"type":"end",...}
Anything else on stderr (example-bot tracebacks) is not JSON and is ignored by calibrate.py.

Env: OBS_REC_UNTIL  match seconds to record (default 60; the real recording covers 0-36.8 s)
     OBS_REC_FILE   write the lines to this file instead of stderr
     MECHFIGHT_KIT  kit directory (default: ../mechfight-kit next to the repo)

Every received line is recorded (also observations the bot skips under the newest-observation rule),
from the reader thread, before the bot sees it.
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.environ.get("MECHFIGHT_KIT") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "mechfight-kit")
UNTIL = float(os.environ.get("OBS_REC_UNTIL", "60"))
OUT = open(os.environ["OBS_REC_FILE"], "w") if os.environ.get("OBS_REC_FILE") else sys.stderr


def compact(msg):
    """Strip the static parts of a server message; None = don't record."""
    kind = msg.get("type")
    if kind == "hello":
        return {"type": "hello", "slot": msg.get("slot"), "seed": msg.get("seed")}
    if kind == "obs":
        if msg.get("t", 0) > UNTIL:
            return None
        me = msg.get("self") or {}
        me.pop("stats", None)
        for w in me.get("weapons") or []:
            if w is not None:
                w.pop("stats", None)
        return msg
    if kind == "end":
        return msg
    return None


def dump(line):
    try:
        rec = compact(json.loads(line))
    except (ValueError, AttributeError):
        return
    if rec is not None:
        OUT.write(json.dumps(rec, separators=(",", ":")) + "\n")
        OUT.flush()


def main():
    sys.path.insert(0, os.path.join(KIT, "python"))
    import mechfight as mf

    def reader(q):
        for line in sys.stdin:
            if line.strip():
                dump(line)
                q.put(line)
        q.put(None)

    mf._reader = reader  # example_bot's `import mechfight` gets this module (already in sys.modules)
    runpy.run_path(os.path.join(KIT, "python", "example_bot.py"), run_name="__main__")


if __name__ == "__main__":
    main()
