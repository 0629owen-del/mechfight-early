#!/usr/bin/env python3
"""Run K minisim matches (optionally in parallel) and aggregate their ground-truth logs.

    python3 tools/run_matches.py -k 8 -j 4 --out runs/b1 --bot "python3 -u main.py"
    python3 tools/run_matches.py -k 4 --out runs/b2 --bot example --fill scripted:mix -- --duration 240
    python3 tools/run_matches.py --aggregate runs/b1            # re-aggregate existing match dirs

Everything after `--` is passed to tools/minisim.py unchanged. Output: <out>/mNNN/ (one dir per match:
match.jsonl, slotNN.stderr, summary.json) plus <out>/summary.json and <out>/summary.txt with:

- placement and score (16 - place) per bot spec, win and top-3 rates, kills, survival, missed decisions
- hit rate by weapon x range bucket (shots aimed at a mech; per spec and overall)
- first-sighting distance distribution (per observer spec)
- fight win/loss by loadout x start-range bucket (per spec and overall)
- death causes per spec
"""
import argparse
import bisect
import collections
import concurrent.futures as cf
import glob
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT_BUCKETS = [0, 25, 50, 75, 100, 125, 150, 200, 300]
FIGHT_BUCKETS = [0, 30, 60, 100, 150, 250]
SIGHT_BIN = 25


def bucket(v, edges):
    if v is None:
        return "none"
    i = bisect.bisect_right(edges, v) - 1
    if i >= len(edges) - 1:
        return f"{edges[-1]}+"
    return f"{edges[max(0, i)]}-{edges[max(0, i) + 1]}"


def run_one(i, a, extra):
    out = os.path.join(a.out, f"m{i:03d}")
    cmd = [sys.executable, os.path.join(HERE, "minisim.py"), "--out", out, "--seed", str(a.seed + i), "--quiet"]
    for b in a.bot or []:
        cmd += ["--bot", b]
    if a.fill:
        cmd += ["--fill", a.fill]
    cmd += extra
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return i, out, p.returncode, time.perf_counter() - t0, p.stderr[-2000:]


def load(dirs):
    matches = []
    for d in dirs:
        path = os.path.join(d, "match.jsonl")
        if not os.path.exists(path):
            continue
        recs = []
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
        if any(r.get("type") == "summary" for r in recs):
            matches.append((d, recs))
    return matches


def rate(h, n):
    return round(h / n, 3) if n else None


def aggregate(matches):
    place = collections.defaultdict(list)
    shots = collections.defaultdict(lambda: [0, 0])
    shots_mv = collections.defaultdict(lambda: [0, 0])
    sights = collections.defaultdict(list)
    sight_modes = collections.defaultdict(collections.Counter)
    fights = collections.defaultdict(lambda: [0, 0, 0])
    causes = collections.defaultdict(collections.Counter)
    walls = []
    for d, recs in matches:
        meta = next(r for r in recs if r["type"] == "meta")
        label = {s["slot"]: s["label"] for s in meta["slots"]}
        n = meta["n"]
        for r in recs:
            ty = r["type"]
            if ty == "place":
                place[r["label"]].append(r)
            elif ty == "shot" and r.get("target") is not None:
                lb = label[r["by"]]
                b = bucket(r.get("range"), SHOT_BUCKETS)
                hit = 1 if (r.get("hit") and r["target"] in r.get("hit_ids", [r["target"]])) else 0
                for key in ((lb, r["weapon"], b), ("ALL", r["weapon"], b)):
                    shots[key][0] += 1
                    shots[key][1] += hit
                mv = "moving" if r.get("moving") else "still"
                for key in ((lb, r["weapon"], b, mv), ("ALL", r["weapon"], b, mv)):
                    shots_mv[key][0] += 1
                    shots_mv[key][1] += hit
            elif ty == "sight":
                lb = label[r["by"]]
                for k in (lb, "ALL"):
                    sights[k].append(r["dist"])
                    sight_modes[k][r["mode"]] += 1
            elif ty == "fight":
                b = bucket(r["range"], FIGHT_BUCKETS)
                for me, other in ((r["a"], r["b"]), (r["b"], r["a"])):
                    lo = "+".join(sorted(r["loadout"][str(me)])) or "none"
                    lo_o = "+".join(sorted(r["loadout"][str(other)])) or "none"
                    res = 0 if r["winner"] == me else (1 if r["winner"] == other else 2)
                    for key in ((label[me], lo, b), ("ALL", lo, b), ("VS", f"{lo} vs {lo_o}", b)):
                        fights[key][res] += 1
            elif ty == "death":
                causes[label[r["id"]]][r["cause"]] += 1
                causes["ALL"][r["cause"]] += 1
            elif ty == "summary":
                walls.append((r["wall_s"], r["t_end"]))
    out = {"matches": len(matches), "wall_s": [w for w, _ in walls],
           "mean_wall_s": round(statistics.mean([w for w, _ in walls]), 1) if walls else None}
    pl = {}
    for lb, rs in place.items():
        pl[lb] = {"n": len(rs), "mean_place": round(statistics.mean(r["place"] for r in rs), 2),
                  "mean_score": round(statistics.mean(r["score"] for r in rs), 2),
                  "win_rate": rate(sum(1 for r in rs if r["place"] == 1), len(rs)),
                  "top3_rate": rate(sum(1 for r in rs if r["place"] <= 3), len(rs)),
                  "mean_kills": round(statistics.mean(r["kills"] for r in rs), 2),
                  "mean_dmg_dealt": round(statistics.mean(r["dmg_dealt"] for r in rs), 1),
                  "mean_dmg_taken": round(statistics.mean(r["dmg_taken"] for r in rs), 1),
                  "mean_survival_s": round(statistics.mean(r["t_death"] if r["t_death"] is not None else 480 for r in rs), 1),
                  "missed_decisions": sum(r["missed"] for r in rs),
                  "late_answers": sum(r["late"] for r in rs),
                  "resp_ms_p99_max": max((r["resp_ms_p99"] or 0) for r in rs)}
    out["placement"] = dict(sorted(pl.items(), key=lambda kv: kv[1]["mean_place"]))
    out["hit_rate"] = {f"{k[0]}|{k[1]}|{k[2]}": {"shots": v[0], "hits": v[1], "rate": rate(v[1], v[0])}
                       for k, v in sorted(shots.items())}
    out["hit_rate_target_motion"] = {"|".join(k): {"shots": v[0], "hits": v[1], "rate": rate(v[1], v[0])}
                                     for k, v in sorted(shots_mv.items())}
    sd = {}
    for k, ds in sights.items():
        ds = sorted(ds)
        hist = collections.Counter(int(x // SIGHT_BIN) * SIGHT_BIN for x in ds)
        sd[k] = {"n": len(ds), "median": round(statistics.median(ds), 1),
                 "p25": round(ds[len(ds) // 4], 1), "p75": round(ds[(3 * len(ds)) // 4], 1),
                 "modes": dict(sight_modes[k]), "hist": {f"{b}-{b + SIGHT_BIN}": hist[b] for b in sorted(hist)}}
    out["first_sighting"] = sd
    out["fights"] = {f"{k[0]}|{k[1]}|{k[2]}": {"win": v[0], "loss": v[1], "draw": v[2],
                                               "win_rate": rate(v[0], v[0] + v[1])}
                     for k, v in sorted(fights.items())}
    out["death_causes"] = {k: dict(v.most_common()) for k, v in causes.items()}
    return out


def text_report(s):
    L = []
    L.append(f"matches: {s['matches']}   mean wall per match: {s['mean_wall_s']} s")
    L.append("")
    L.append("PLACEMENT (score = 16 - place)")
    L.append(f"{'spec':44s} {'n':>4s} {'place':>6s} {'score':>6s} {'win':>5s} {'top3':>5s} {'kills':>5s} {'surv_s':>6s} {'missed':>6s} {'p99ms':>6s}")
    for lb, r in s["placement"].items():
        L.append(f"{lb[:44]:44s} {r['n']:4d} {r['mean_place']:6.2f} {r['mean_score']:6.2f} {r['win_rate']:5.2f} "
                 f"{r['top3_rate']:5.2f} {r['mean_kills']:5.2f} {r['mean_survival_s']:6.1f} {r['missed_decisions']:6d} "
                 f"{r['resp_ms_p99_max']:6.1f}")
    L.append("")
    L.append("HIT RATE by weapon x range (shots aimed at a mech), spec=ALL")
    rows = collections.defaultdict(dict)
    for k, v in s["hit_rate"].items():
        lb, wp, b = k.split("|")
        if lb == "ALL":
            rows[wp][b] = v
    cols = [bucket(e, SHOT_BUCKETS) for e in SHOT_BUCKETS[:-1]] + [f"{SHOT_BUCKETS[-1]}+"]
    L.append(f"{'weapon':10s} " + " ".join(f"{c:>11s}" for c in cols))
    for wp, d in sorted(rows.items()):
        L.append(f"{wp:10s} " + " ".join(f"{(str(round(100 * d[c]['rate'])) + '%/' + str(d[c]['shots'])) if c in d and d[c]['rate'] is not None else '-':>11s}" for c in cols))
    L.append("")
    L.append("FIRST SIGHTING distance (m)")
    for k, v in sorted(s["first_sighting"].items()):
        L.append(f"  {k[:44]:44s} n={v['n']:4d} median {v['median']:6.1f}  p25 {v['p25']:6.1f}  p75 {v['p75']:6.1f}  {v['modes']}")
    if "ALL" in s["first_sighting"]:
        L.append("  histogram (ALL): " + "  ".join(f"{b}:{c}" for b, c in s["first_sighting"]["ALL"]["hist"].items()))
    L.append("")
    L.append("FIGHTS by loadout x start range (ALL): win/loss/draw (win rate of decided)")
    fr = collections.defaultdict(dict)
    for k, v in s["fights"].items():
        lb, lo, b = k.split("|")
        if lb == "ALL":
            fr[lo][b] = v
    fcols = [bucket(e, FIGHT_BUCKETS) for e in FIGHT_BUCKETS[:-1]] + [f"{FIGHT_BUCKETS[-1]}+"]
    L.append(f"{'loadout':22s} " + " ".join(f"{c:>14s}" for c in fcols))
    for lo, d in sorted(fr.items()):
        L.append(f"{lo[:22]:22s} " + " ".join(
            f"{(str(d[c]['win']) + '/' + str(d[c]['loss']) + '/' + str(d[c]['draw'])) if c in d else '-':>14s}" for c in fcols))
    L.append("")
    L.append("DEATH CAUSES")
    for k, v in sorted(s["death_causes"].items()):
        L.append(f"  {k[:44]:44s} " + ", ".join(f"{c}:{n}" for c, n in v.items()))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Run and aggregate minisim matches.")
    ap.add_argument("-k", type=int, default=4, help="number of matches")
    ap.add_argument("-j", type=int, default=2, help="parallel matches")
    ap.add_argument("--out", default="runs/batch")
    ap.add_argument("--seed", type=int, default=1, help="first match seed (match i uses seed+i)")
    ap.add_argument("--bot", action="append", help="bot spec (repeatable), see minisim.py")
    ap.add_argument("--fill", default=None, help="spec for remaining slots (minisim default scripted:mix)")
    ap.add_argument("--aggregate", default=None, help="only aggregate the match dirs under this directory")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- extra minisim.py arguments")
    a = ap.parse_args()
    extra = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    if a.aggregate:
        root = a.aggregate
        dirs = sorted(set(os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "match.jsonl"), recursive=True)))
    else:
        root = a.out
        os.makedirs(root, exist_ok=True)
        dirs = []
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=max(1, a.j)) as ex:
            futs = [ex.submit(run_one, i, a, extra) for i in range(a.k)]
            for fu in cf.as_completed(futs):
                i, out, rc, wall, err = fu.result()
                dirs.append(out)
                print(f"match {i:3d}: rc={rc} wall={wall:.1f}s -> {out}", file=sys.stderr)
                if rc != 0:
                    print(err, file=sys.stderr)
        print(f"all matches: {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        dirs.sort()
    s = aggregate(load(dirs))
    txt = text_report(s)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    with open(os.path.join(root, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
