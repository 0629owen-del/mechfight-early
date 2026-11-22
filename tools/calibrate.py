#!/usr/bin/env python3
"""Compare minisim against the one real recorded match, from the example bot's point of view (stdlib only).

    python3 tools/run_matches.py -k 30 -j 4 --out runs/cal --bot "python3 -u tools/obs_recorder.py"
    python3 tools/calibrate.py runs/cal                 # one batch vs the real recording
    python3 tools/calibrate.py runs/cal_v1 runs/cal     # several batches side by side (before / after)

The real recording (`mechfight-kit/check/stream.jsonl.gz`) is the observation stream of slot 3, the kit's
example bot, for t = 0-36.8 s (with a recording gap at 29.5-35.9 s). `tools/obs_recorder.py` produces the
same stream in minisim. The SAME function (`observer_metrics`) computes every observer-side metric for
both, and the sim stream is masked to the real recording's observation times (so the gap drops the
same sounds and events). For every metric: the real value, the minisim median and IQR over matches,
and the real value's percentile in the minisim distribution.

Observer-side metrics (window 0-36.8 s):
- deaths by 20 s / 36.8 s, first death (from `alive`; minisim: ground truth, because the observer may
  die first), alive count every 5 s
- sounds heard by kind x band, first gun sound. Counts are scaled to the real recording's covered
  seconds when the minisim observer died early (matches with < 10 s covered are left out)
- the primary attacker (the mech that dealt the observer the most damage): first sight (time,
  distance), reaction = first shot at the observer - first sight, engagement range (median distance at
  its damage events), damage, DPS while engaged (damage / (last - first damage + 1 s)), DPS in bursts,
  bursts (damage events closer than 3 s; mean length and pause), damage duty cycle (share of covered 1 s bins between its
  first and last damage with damage), laser hit rate (laser damage events / its beams passing within
  40 m), gun hit rate (gun damage events / gun sounds attributable to it: bearing within 0.35 rad,
  matching distance band, its aim within 0.12 rad of the observer)
- observer: spawn distance from the zone centre (sounds heard depend strongly on it; --spawn-band
  restricts minisim to spawns near the real one), hits dealt, HP at the window end (0 if dead), NPC
  gun sounds heard

minisim ground truth (no real counterpart): deaths by 10/20/30/36.8/60/120 s, shots per scripted
bot-second alive in the first 60 s (all / gun / laser), the share of scripted bots that fired by 10 s,
and the primary attacker's true shots/hits on the observer and reaction from ITS first sight.
"""
import argparse
import bisect
import collections
import glob
import gzip
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "mechfight-kit")
REAL = os.path.join(KIT, "check", "stream.jsonl.gz")
WINDOW = 36.8
DT = 0.1
BURST_GAP = 3.0
BEAM_NEAR = 40.0
GUN_BEARING = 0.35     # 3 sigma of the 7 deg bearing noise
AIM_AT_ME = 0.12
MIN_COVER = 10.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def band(d):
    return "near" if d < 60 else ("mid" if d < 160 else "far")


def seg_dist(p, a, b):
    d = [b[i] - a[i] for i in range(3)]
    L2 = sum(x * x for x in d) or 1e-9
    u = max(0.0, min(1.0, sum((p[i] - a[i]) * d[i] for i in range(3)) / L2))
    return math.dist(p, [a[i] + u * d[i] for i in range(3)])


# ---------------------------------------------------------------------------------------------- loading
def load_stream(path):
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def load_match(d):
    recs = []
    with open(os.path.join(d, "match.jsonl")) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    if not any(r.get("type") == "summary" for r in recs):
        return None
    meta = next(r for r in recs if r["type"] == "meta")
    obs_slot = None
    for s in meta["slots"]:
        if "obs_recorder" in s["spec"]:
            obs_slot = s["slot"]
    if obs_slot is None:
        return None
    stream = load_stream(os.path.join(d, f"slot{obs_slot:02d}.stderr"))
    return {"dir": d, "recs": recs, "meta": meta, "obs_slot": obs_slot,
            "obs": [m for m in stream if m.get("type") == "obs"]}


# ---------------------------------------------------------------------------------------- observer side
def observer_metrics(obs, window=WINDOW, mask=None, cover_ref=None):
    """Metrics from one observation stream (real or minisim). mask: allowed obs times (0.1 s grid)."""
    obs = [o for o in obs if o["t"] <= window + 1e-6]
    if mask is not None:
        obs = [o for o in obs if round(o["t"], 1) in mask]
    m = {}
    if not obs:
        return m
    p0, zc = obs[0]["self"]["pos"], obs[0]["zone"]["center"]
    m["obs_spawn_r"] = round(math.hypot(p0[0] - zc[0], p0[1] - zc[1]), 1)
    covered = len(obs) * DT
    m["covered_s"] = round(covered, 1)
    m["observer_alive_at_end"] = obs[-1]["t"] >= window - 0.05
    # alive / deaths as the observer saw them
    dead_at = []
    prev = 16
    for o in obs:
        if o["alive"] < prev:
            dead_at += [o["t"]] * (prev - o["alive"])
            prev = o["alive"]
    m["_obs_deaths"] = dead_at
    # sounds
    snd = collections.Counter()
    first_gun = None
    for o in obs:
        for s in o["sounds"]:
            snd[(s["kind"], s["dist"])] += 1
            snd[(s["kind"], "all")] += 1
            if s["kind"] == "gun":
                tg = o["t"] - s.get("age", 0)
                first_gun = tg if first_gun is None else min(first_gun, tg)
    scale = None
    if covered >= MIN_COVER:
        scale = (cover_ref / covered) if cover_ref else 1.0
    m["_sounds"] = {f"{k}_{b}": v for (k, b), v in snd.items()}
    m["_sound_scale"] = scale
    m["first_gun_t"] = round(first_gun, 2) if first_gun is not None else None
    # observer's own output and state
    m["obs_hits"] = sum(1 for o in obs for e in o["events"] if e["type"] == "hit")
    m["obs_hp_end"] = obs[-1]["self"]["hp"] if m["observer_alive_at_end"] else 0.0
    m["obs_mechs_seen"] = len({x["id"] for o in obs for x in o["mechs"] if x["kind"] == "mech"})
    # attackers
    dmg = collections.defaultdict(list)
    for o in obs:
        for e in o["events"]:
            if e["type"] == "damaged" and isinstance(e.get("by"), (int, float)) and 0 <= e["by"] < 100:
                dmg[int(e["by"])].append((o["t"], e["weapon"], e["amount"]))
    m["n_attackers"] = len(dmg)
    m["dmg_taken_mechs"] = round(sum(a for v in dmg.values() for _, _, a in v), 1)
    first_dmg = min((v[0][0] for v in dmg.values()), default=None)
    m["first_dmg_t"] = first_dmg
    if not dmg:
        return m
    A = max(dmg, key=lambda k: sum(a for _, _, a in dmg[k]))
    ev = dmg[A]
    m["att_id"] = A
    track = {}  # t -> (dist, bearing me->A, aim error of A towards me)
    first_sight = None
    for o in obs:
        me = o["self"]["pos"]
        for x in o["mechs"]:
            if x["id"] == A:
                d = math.dist(x["pos"], me)
                brg = math.atan2(x["pos"][1] - me[1], x["pos"][0] - me[0])
                aim_err = abs(wrap(x["aim"][0] - math.atan2(me[1] - x["pos"][1], me[0] - x["pos"][0])))
                track[round(o["t"], 1)] = (d, brg, aim_err)
                if first_sight is None:
                    first_sight = (o["t"], d)
    times = sorted(track)

    def near_track(t, tol=0.55):
        if not times:
            return None
        i = bisect.bisect_left(times, t - tol)
        best = None
        while i < len(times) and times[i] <= t + tol:
            if best is None or abs(times[i] - t) < abs(best - t):
                best = times[i]
            i += 1
        return track[best] if best is not None else None

    # shots at me: beams by A passing near me, gun sounds attributable to A while it aims at me, damage
    laser_shots, gun_shots = [], []
    prev_aim_ok = False
    for o in obs:
        me = o["self"]["pos"]
        for b in o.get("beams", []):
            if b["by"] == A and b["kind"] == "laser" and seg_dist(me, b["path"][0], b["path"][-1]) < BEAM_NEAR:
                laser_shots.append(o["t"] - b.get("age", 0))
        tr = track.get(round(o["t"], 1))
        aim_ok = tr is not None and tr[2] < AIM_AT_ME
        if tr is not None:
            for s in o["sounds"]:
                if s["kind"] == "gun" and s["dist"] == band(tr[0]) and abs(wrap(s["bearing"] - tr[1])) < GUN_BEARING \
                        and (aim_ok or prev_aim_ok):
                    gun_shots.append(o["t"] - s.get("age", 0))
        prev_aim_ok = aim_ok
    shot_ts = [t for t, _, _ in ev] + laser_shots + gun_shots
    first_shot = min(shot_ts)
    m["att_first_sight_t"] = round(first_sight[0], 2) if first_sight else None
    m["att_first_sight_dist"] = round(first_sight[1], 1) if first_sight else None
    m["att_first_shot_t"] = round(first_shot, 2)
    m["att_reaction_s"] = round(first_shot - first_sight[0], 2) if first_sight else None
    ranges = [r[0] for r in (near_track(t) for t, _, _ in ev) if r is not None]
    m["att_range_median"] = round(statistics.median(ranges), 1) if ranges else None
    m["att_closest"] = round(min(v[0] for v in track.values()), 1) if track else None
    total = sum(a for _, _, a in ev)
    t0, t1 = ev[0][0], ev[-1][0]
    m["att_dmg"] = round(total, 1)
    m["att_dps_engaged"] = round(total / (t1 - t0 + 1.0), 1)
    bursts = [[ev[0][0], ev[0][0]]]
    for t, _, _ in ev[1:]:
        if t - bursts[-1][1] > BURST_GAP:
            bursts.append([t, t])
        else:
            bursts[-1][1] = t
    m["att_bursts"] = len(bursts)
    m["att_burst_len_mean"] = round(statistics.mean(b - a for a, b in bursts), 2)
    m["att_pause_mean"] = (round(statistics.mean(bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)), 2)
                           if len(bursts) > 1 else None)
    m["att_dps_burst"] = round(total / sum(b - a + 1.0 for a, b in bursts), 1)
    cov_bins = {int(o["t"] + 1e-6) for o in obs}
    dmg_bins = {int(t + 1e-6) for t, _, _ in ev}
    span = [b for b in range(int(t0 + 1e-6), int(t1 + 1e-6) + 1) if b in cov_bins]
    m["att_duty"] = round(len(dmg_bins) / len(span), 3) if len(span) >= 3 else None
    nl = sum(1 for _, w, _ in ev if w == "laser")
    ng = sum(1 for _, w, _ in ev if w == "gun")
    m["att_laser_dmg_events"] = nl
    m["att_gun_dmg_events"] = ng
    m["att_laser_shots"] = max(len(laser_shots), nl)
    m["att_gun_shots_est"] = max(len(gun_shots), ng)
    m["att_laser_hit"] = round(nl / m["att_laser_shots"], 3) if m["att_laser_shots"] else None
    m["att_gun_hit"] = round(ng / m["att_gun_shots_est"], 3) if m["att_gun_shots_est"] >= 3 else None
    return m


def finish_obs_metrics(m, deaths):
    """Deaths (from `deaths`: list of times) and scaled sound counts -> flat metric dict."""
    out = {k: v for k, v in m.items() if not k.startswith("_")}
    for T in (20.0, WINDOW):
        out[f"deaths_by_{T:g}"] = sum(1 for t in deaths if t <= T + 1e-6)
    out["first_death_t"] = min(deaths) if deaths else None
    for T in (5, 10, 15, 20, 25, 30, 35):
        out[f"alive_at_{T}"] = 16 - sum(1 for t in deaths if t <= T + 1e-6)
    sc = m.get("_sound_scale")
    for k in SOUND_KEYS:
        out[f"snd_{k}"] = None if sc is None else round(m["_sounds"].get(k, 0) * sc, 1)
    return out


SOUND_KEYS = ["gun_all", "gun_far", "gun_mid", "gun_near", "laser_all", "laser_mid", "laser_near", "laser_far",
              "explosion_all", "mortar_all", "lightning_all", "crate_all", "death_all", "laser_charge_all",
              "boost_all", "step_all", "npc_gun_all"]


# -------------------------------------------------------------------------------------- ground truth
def gt_metrics(match, window=WINDOW):
    recs, meta, ob = match["recs"], match["meta"], match["obs_slot"]
    label = {s["slot"]: s["label"] for s in meta["slots"]}
    scripted = {s for s, lb in label.items() if lb.startswith("scripted:")}
    deaths = sorted(r["t"] for r in recs if r["type"] == "death")
    tdeath = {r["id"]: r["t"] for r in recs if r["type"] == "death"}
    g = {}
    for T in (10, 20, 30, window, 60, 120):
        g[f"gt_deaths_by_{T:g}"] = sum(1 for t in deaths if t <= T + 1e-6)
    alive_s = sum(min(60.0, tdeath.get(s, 60.0)) for s in scripted)
    shots = [r for r in recs if r["type"] == "shot" and r["t"] < 60 and r["by"] in scripted]
    g["gt_shots_per_bot_s_60"] = round(len(shots) / alive_s, 3) if alive_s else None
    g["gt_gun_per_bot_s_60"] = round(sum(1 for r in shots if r["weapon"] == "gun") / alive_s, 3) if alive_s else None
    g["gt_laser_per_bot_s_60"] = round(sum(1 for r in shots if r["weapon"] == "laser") / alive_s, 3) if alive_s else None
    early = {r["by"] for r in recs if r["type"] == "shot" and r["t"] <= 10 and r["by"] in scripted
             and r.get("target") is not None and r["target"] < 100}
    g["gt_fired_at_mech_by_10_share"] = round(len(early) / max(1, len(scripted)), 3)
    at_mech = [r for r in shots if r.get("target") is not None and r["target"] < 100]
    g["gt_range_median_60"] = round(statistics.median(r["range"] for r in at_mech), 1) if at_mech else None
    g["gt_first_mech_shot_t"] = min((r["t"] for r in at_mech), default=None)
    g["gt_observer_death_t"] = tdeath.get(ob)
    return g, deaths


def gt_attacker(match, att, window=WINDOW):
    recs, ob = match["recs"], match["obs_slot"]
    sh = [r for r in recs if r["type"] == "shot" and r["by"] == att and r.get("target") == ob and r["t"] <= window]
    g = {}
    for w in ("gun", "laser"):
        s = [r for r in sh if r["weapon"] == w]
        g[f"gt_att_{w}_shots"] = len(s)
        g[f"gt_att_{w}_hit"] = round(sum(1 for r in s if r.get("hit")) / len(s), 3) if s else None
    sight = next((r["t"] for r in recs if r["type"] == "sight" and r["by"] == att and r["target"] == ob), None)
    first = min((r["t"] for r in sh), default=None)
    g["gt_att_reaction_s"] = round(first - sight, 2) if (first is not None and sight is not None) else None
    g["gt_att_sight_dist"] = next((r["dist"] for r in recs if r["type"] == "sight" and r["by"] == att
                                   and r["target"] == ob), None)
    return g


# ------------------------------------------------------------------------------------------- reporting
ROWS = [
    ("1 deaths", ["deaths_by_20", "deaths_by_36.8", "first_death_t", "alive_at_10", "alive_at_30"]),
    ("2 sounds (per 30 s covered)", ["snd_gun_all", "snd_gun_far", "snd_gun_mid", "snd_gun_near", "first_gun_t",
                                     "snd_laser_all", "snd_explosion_all", "snd_mortar_all", "snd_lightning_all",
                                     "snd_death_all", "snd_crate_all", "snd_laser_charge_all", "snd_boost_all",
                                     "snd_step_all"]),
    ("3 primary attacker", ["n_attackers", "first_dmg_t", "att_first_sight_t", "att_first_sight_dist",
                            "att_reaction_s", "att_range_median", "att_closest", "att_dmg", "att_dps_engaged",
                            "att_dps_burst", "att_bursts", "att_burst_len_mean", "att_pause_mean", "att_duty", "att_laser_shots",
                            "att_laser_hit", "att_gun_shots_est", "att_gun_dmg_events", "att_gun_hit"]),
    ("4 observer", ["obs_spawn_r", "obs_hits", "obs_hp_end", "obs_mechs_seen", "dmg_taken_mechs", "snd_npc_gun_all"]),
    ("5 minisim ground truth", ["gt_deaths_by_10", "gt_deaths_by_20", "gt_deaths_by_30", "gt_deaths_by_36.8",
                                "gt_deaths_by_60", "gt_deaths_by_120", "gt_shots_per_bot_s_60", "gt_gun_per_bot_s_60",
                                "gt_laser_per_bot_s_60", "gt_fired_at_mech_by_10_share", "gt_range_median_60",
                                "gt_first_mech_shot_t", "gt_observer_death_t", "gt_att_reaction_s",
                                "gt_att_sight_dist", "gt_att_gun_shots", "gt_att_gun_hit", "gt_att_laser_shots",
                                "gt_att_laser_hit"]),
]


def quant(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    p = q * (len(xs) - 1)
    lo = int(math.floor(p))
    hi = min(len(xs) - 1, lo + 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def percentile(xs, v):
    if v is None or not xs:
        return None
    below = sum(1 for x in xs if x < v - 1e-9)
    eq = sum(1 for x in xs if abs(x - v) <= 1e-9)
    return round(100.0 * (below + 0.5 * eq) / len(xs))


def summarize(per_match, real):
    out = {}
    keys = {k for m in per_match for k in m} | set(real)
    for k in keys:
        xs = [m[k] for m in per_match if isinstance(m.get(k), (int, float)) and not isinstance(m[k], bool)]
        rv = real.get(k)
        if not xs:
            out[k] = {"n": 0, "real": rv}
            continue
        q1, md, q3 = quant(xs, 0.25), quant(xs, 0.5), quant(xs, 0.75)
        out[k] = {"n": len(xs), "real": rv, "median": round(md, 3), "q25": round(q1, 3), "q75": round(q3, 3),
                  "mean": round(statistics.mean(xs), 3), "min": min(xs), "max": max(xs),
                  "pct": percentile(xs, rv) if isinstance(rv, (int, float)) and not isinstance(rv, bool) else None,
                  "in_iqr": (q1 <= rv <= q3) if isinstance(rv, (int, float)) and not isinstance(rv, bool) else None}
    return out


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 10 else f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
    return str(v)


def real_metrics(path=REAL):
    stream = load_stream(path)
    obs = [m for m in stream if m.get("type") == "obs"]
    mask = {round(o["t"], 1) for o in obs if o["t"] <= WINDOW + 1e-6}
    m = observer_metrics(obs, mask=mask)
    flat = finish_obs_metrics(m, m["_obs_deaths"])
    return flat, mask, m["covered_s"]


def batch_metrics(root, mask, cover_ref, spawn_r=None, spawn_band=0.0):
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "match.jsonl"), recursive=True)})
    per = []
    for d in dirs:
        mt = load_match(d)
        if mt is None:
            continue
        g, deaths = gt_metrics(mt)
        m = observer_metrics(mt["obs"], mask=mask, cover_ref=cover_ref)
        flat = finish_obs_metrics(m, deaths) if m else {}
        flat.update(g)
        if flat.get("att_id") is not None:
            flat.update(gt_attacker(mt, flat["att_id"]))
        flat["_dir"] = os.path.basename(d)
        if spawn_band > 0 and abs(flat.get("obs_spawn_r", -1e9) - spawn_r) > spawn_band:
            continue
        per.append(flat)
    return per


def main():
    ap = argparse.ArgumentParser(description="minisim vs the real recording, from the example bot's view")
    ap.add_argument("batches", nargs="+", help="run_matches.py output dirs (observer = tools/obs_recorder.py)")
    ap.add_argument("--real", default=REAL)
    ap.add_argument("--json", default=None, help="write the full comparison here (default <first batch>/calibration.json)")
    ap.add_argument("--spawn-band", type=float, default=0.0,
                    help="only matches whose observer spawned within this many metres of the real observer's distance "
                         "from the zone centre (347 m; sounds heard depend strongly on it)")
    a = ap.parse_args()
    real, mask, cover = real_metrics(a.real)
    summ = []
    for b in a.batches:
        per = batch_metrics(b, mask, cover, real["obs_spawn_r"], a.spawn_band)
        summ.append((b, per, summarize(per, real)))
    L = [f"real recording: {a.real}  window 0-{WINDOW} s, covered {cover:.1f} s (gap 29.5-35.9 s masked in minisim too)"]
    if a.spawn_band > 0:
        L.append(f"only matches with observer spawn radius {real['obs_spawn_r']:.0f} +- {a.spawn_band:.0f} m")
    for b, per, _ in summ:
        died = sum(1 for m in per if m.get("observer_alive_at_end") is False)
        L.append(f"batch {b}: {len(per)} matches; observer dead before {WINDOW} s in {died}")
    L.append("")
    head = f"{'metric':28s} {'real':>7s}"
    for _ in summ:
        head += f" | {'median':>7s} {'IQR':>13s} {'pct':>4s} {'n':>3s}"
    L.append(head)
    for title, keys in ROWS:
        L.append(f"-- {title}")
        for k in keys:
            rv = real.get(k)
            row = f"{k:28s} {fmt(rv):>7s}"
            for _, _, s in summ:
                e = s.get(k, {})
                if not e.get("n"):
                    row += f" | {'-':>7s} {'-':>13s} {'-':>4s} {0:3d}"
                    continue
                iqr = f"[{fmt(e['q25'])}, {fmt(e['q75'])}]"
                mark = "*" if e.get("in_iqr") else " "
                pct = "-" if e.get("pct") is None else f"{e['pct']}{mark}"
                row += f" | {fmt(e['median']):>7s} {iqr:>13s} {pct:>4s} {e['n']:3d}"
            L.append(row)
    L.append("")
    L.append("pct = percentile of the real value in the minisim distribution; * = inside the IQR")
    txt = "\n".join(L)
    print(txt)
    out = a.json or os.path.join(a.batches[0], "calibration.json")
    with open(out, "w") as f:
        json.dump({"real": real, "batches": [{"dir": b, "summary": s, "per_match": per} for b, per, s in summ]},
                  f, indent=1, default=str)
    with open(os.path.splitext(out)[0] + ".txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
