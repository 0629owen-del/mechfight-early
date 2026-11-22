"""Telemetry tests (14.4): rate limiting, record schema, shot/hit tables, fights, summary.

    cd bot && python3 -m unittest tests.test_telemetry -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

import config  # noqa: E402
import telemetry  # noqa: E402
from contracts import (Assessment, FireRequest, Intent, RunnerInfo, SelfState, TickInfo, Track,  # noqa: E402
                       TrackAssess, WeaponState)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def lines(buf):
    return [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]


def make(C=None, **over):
    C = C or config.bind(None)
    for k, v in over.items():
        setattr(C, k, v)
    buf = io.StringIO()
    clk = FakeClock()
    tl = telemetry.Telemetry(C, slot=3, seed=42, stream=buf, now=clk)
    return tl, buf, clk


class TestRateLimit(unittest.TestCase):
    def test_burst_is_limited_and_dropped_reported(self):
        tl, buf, clk = make()
        n = int(tl.max_lines)
        written = sum(tl.event("x", i=i) for i in range(50))
        self.assertEqual(written, n)
        self.assertEqual(len(lines(buf)), n)
        clk.t += 1.0
        self.assertTrue(tl.event("x", i=99))
        last = lines(buf)[-1]
        self.assertEqual(last["dropped"], 50 - n)
        clk.t += 1.0
        tl.event("x")
        self.assertNotIn("dropped", lines(buf)[-1])

    def test_sustained_rate_never_exceeds_limit(self):
        tl, buf, clk = make()
        for step in range(200):                      # 20 s at 10 attempts per second
            clk.t += 0.1
            tl.event("x", step=step)
        n = len(lines(buf))
        self.assertLessEqual(n, tl.max_lines * 20 + tl.max_lines)
        self.assertGreaterEqual(n, tl.max_lines * 18)

    def test_byte_budget(self):
        tl, buf, clk = make(TLM_MAX_BYTES_PER_S=500)
        for _ in range(20):
            tl.event("big", blob="y" * 300)
        self.assertLessEqual(len(buf.getvalue()), 1200 + 500)
        self.assertGreater(tl.dropped_total, 10)

    def test_summary_never_dropped(self):
        tl, buf, clk = make()
        for i in range(30):
            tl.event("x", i=i)
        tl.end_summary({"type": "end", "place": 4}, RunnerInfo())
        rows = lines(buf)
        self.assertEqual(rows[-1]["ev"], "summary")
        self.assertEqual(rows[-1]["end"]["place"], 4)
        self.assertIn("dropped", rows[-1])

    def test_long_line_is_cut_to_header(self):
        tl, buf, clk = make(TLM_MAX_LINE_BYTES=200)
        self.assertTrue(tl.event("huge", blob="z" * 5000))
        row = lines(buf)[0]
        self.assertEqual(set(row), {"ev", "t", "tick", "trunc"})

    def test_force_flag_bypasses_limit(self):
        tl, buf, clk = make()
        for _ in range(20):
            tl.event("x")
        self.assertTrue(tl.event("boom", _force=True))
        self.assertEqual(lines(buf)[-1]["ev"], "boom")


class TestSchema(unittest.TestCase):
    def test_records_have_ev_t_tick_and_valid_json(self):
        tl, buf, clk = make()
        tl._t, tl._tick = 12.5, 250
        tl.event("custom", a=1, b=float("nan"), c=(1.0, 2.0), d={"k": float("inf")}, e=object)
        row = lines(buf)[0]
        self.assertEqual((row["ev"], row["t"], row["tick"]), ("custom", 12.5, 250))
        self.assertIsNone(row["b"])
        self.assertEqual(row["c"], [1.0, 2.0])
        self.assertIsNone(row["d"]["k"])

    def test_ev_cannot_be_overridden_t_and_tick_can(self):
        tl, buf, clk = make()
        tl.event("a", ev="evil", t=3.0, tick=7)
        row = lines(buf)[0]
        self.assertEqual((row["ev"], row["t"], row["tick"]), ("a", 3.0, 7))

    def test_nothing_goes_to_stdout(self):
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            tl, buf, clk = make()
            tl.event("x")
            tl.end_summary({}, None)
            self.assertEqual(sys.stdout.getvalue(), "")
        finally:
            sys.stdout = old


# ---------------------------------------------------------------- record() scenarios
def world(t, tick, me, tracks=None, events=(), kills=0, alive=10, dt=0.1, held=("gun", "lightning")):
    return SimpleNamespace(tick=TickInfo(tick=tick, t=t, dt=dt, alive=alive), me=me, tracks=tracks or {},
                           events=list(events), kills=kills, held_kinds=lambda: held)


def me_with(gun_ammo=12, gun_cd=0.0, in_zone=True, hp=1400.0):
    me = SelfState(id=1, hp=hp, max_hp=1400.0, in_zone=in_zone)
    me.weapons = [WeaponState(kind="gun", ammo=gun_ammo, cooldown=gun_cd,
                              stats={"speed": 252.0, "cooldown": 0.24}), None]
    return me


def assess(d, A=1.0, tid=5):
    ta = TrackAssess(id=tid, d=d, d3=d, bearing=0.0, los=True, dps_to_us=0, dps_from_us=0, ttk_us_on_j=1,
                     ttk_them_on_us=1, A=A, prio=1, aiming_at_us=False, in_its_reach=False,
                     charging_at_us=False, closing_speed=0)
    return Assessment(per_track={tid: ta}, target_id=tid, A=A)


class TestRecord(unittest.TestCase):
    def test_shots_and_hit_matching_by_bucket(self):
        tl, buf, clk = make()
        intent = Intent(macro="X2")
        intent.fire.append(FireRequest(slot=0, rule="F1", target_id=5))
        tr = Track(id=5, kind="mech", vel=(8.0, 0.0, 0.0), in_obs=True, visible=True)
        A = assess(d=110.0)
        t, ammo = 10.0, 12
        for i in range(6):
            ev = []
            if i == 3:
                ev = [{"type": "hit", "target": 5, "amount": 45, "weapon": "gun"}]
            w = world(t, i, me_with(gun_ammo=ammo), {5: tr}, ev)
            tl.record(w, A, intent, {}, None)
            t += 0.1
            ammo -= 1           # one round per decision
        self.assertEqual(tl.errors, 0)
        rows = tl.hitrate_table()
        self.assertEqual(len(rows), 1)
        weapon, bucket_from, moving, shots, hits = rows[0]
        self.assertEqual((weapon, bucket_from, moving), ("gun", 100, 1))
        self.assertEqual(shots, 5)       # five ammo drops seen
        self.assertEqual(hits, 1)

    def test_shot_detected_by_cooldown_for_lightning(self):
        tl, buf, clk = make()
        intent = Intent(macro="X2")
        me1 = me_with()
        me1.weapons[1] = WeaponState(kind="lightning", cooldown=0.0)
        me2 = me_with()
        me2.weapons[1] = WeaponState(kind="lightning", cooldown=0.25)
        tr = Track(id=5, kind="mech", in_obs=True)
        tl.record(world(1.0, 1, me1, {5: tr}), assess(30.0), intent, {}, None)
        tl.record(world(1.1, 2, me2, {5: tr}), assess(30.0), intent, {}, None)
        self.assertEqual(tl.hitrate_table(), [["lightning", 25, 0, 1, 0]])

    def test_unmatched_and_non_mech_hits_do_not_crash(self):
        tl, buf, clk = make()
        ev = [{"type": "hit", "target": 1007, "amount": 45, "weapon": "gun"},
              {"type": "hit", "target": 5, "amount": 45, "weapon": "laser"}]
        tl.record(world(1.0, 1, me_with(), {}, ev), assess(50.0), Intent(), {}, None)
        self.assertEqual(tl.counters["hit_unmatched"], 1)
        self.assertEqual(tl.errors, 0)

    def test_first_sighting_and_histogram(self):
        tl, buf, clk = make(TLM_MAX_LINES_PER_S=100)
        tr = Track(id=5, kind="mech", in_obs=True, visible=True, first_seen_dist=130.0, weapons=("gun",))
        npc = Track(id=104, kind="npc", in_obs=True, first_seen_dist=40.0)
        for i in range(3):
            tl.record(world(2.0 + 0.1 * i, i, me_with(), {5: tr, 104: npc}), None, Intent(), {}, None)
        sights = [r for r in lines(buf) if r["ev"] == "sight"]
        self.assertEqual(len(sights), 1)
        self.assertEqual((sights[0]["id"], sights[0]["d"]), (5, 130.0))
        self.assertEqual(dict(tl.sight_hist), {5: 1})

    def test_fight_episode_and_a_band(self):
        tl, buf, clk = make(TLM_MAX_LINES_PER_S=100)
        tr = Track(id=5, kind="mech", in_obs=True, visible=True, weapons=("gun", "lightning"))
        fi = Intent(macro="X2")
        fi.fire.append(FireRequest(slot=0, rule="F1", target_id=5))
        t = 20.0
        for i in range(10):                         # 1 s in X2 with A inside [HOLD, ENGAGE)
            tl.record(world(t, i, me_with(hp=1400 - 50 * i), {5: tr}), assess(80.0, A=0.95), fi, {}, None)
            t += 0.1
        flee = Intent(macro="X3")
        tl.record(world(t, 11, me_with(hp=800), {5: tr}, kills=0), assess(80.0, A=0.8), flee, {}, None)
        fights = [r for r in lines(buf) if r["ev"] == "fight"]
        self.assertEqual(len(fights), 1)
        f = fights[0]
        self.assertEqual((f["target"], f["out"], f["ours"], f["theirs"]), (5, "flee", "GL", "GL"))
        self.assertAlmostEqual(f["A0"], 0.95)
        self.assertAlmostEqual(f["d0"], 80.0)
        self.assertEqual(f["hp0"], 1400)
        self.assertGreater(f["dur"], 0.9)
        self.assertEqual([r["ev"] for r in lines(buf)].count("aband"), 1)
        self.assertGreater(tl.a_band_s, 0.9)
        self.assertEqual(tl.fight_table["GL>GL@3"], [0, 1])

    def test_fight_kill_outcome(self):
        tl, buf, clk = make(TLM_MAX_LINES_PER_S=100)
        tr = Track(id=5, kind="mech", in_obs=True, visible=True, weapons=("gun",))
        tl.record(world(1.0, 1, me_with(), {5: tr}, kills=0), assess(60.0), Intent(macro="X2"), {}, None)
        tl.record(world(1.1, 2, me_with(), {5: tr}, kills=1), assess(60.0), Intent(macro="X1"), {}, None)
        f = [r for r in lines(buf) if r["ev"] == "fight"][0]
        self.assertEqual(f["out"], "kill")

    def test_death_cause_from_recent_damage_and_summary(self):
        tl, buf, clk = make()
        intent = Intent(macro="X3", goal=None, rule="R3.1")
        evs = [{"type": "damaged", "amount": 300, "weapon": "laser", "by": 7},
               {"type": "damaged", "amount": 60, "weapon": "zone", "by": None}]
        tl.record(world(50.0, 500, me_with(in_zone=False, hp=10), {}, evs, alive=3), None, intent, {}, None)
        tl.end_summary({"type": "end", "place": 3, "ignored": {"a": 1}}, RunnerInfo(hard_overruns=2))
        s = lines(buf)[-1]
        self.assertEqual(s["ev"], "summary")
        self.assertEqual(s["death"]["weapon"], "laser")
        self.assertEqual(s["death"]["by"], 7)
        self.assertEqual(s["death"]["macro"], "X3")
        self.assertFalse(s["death"]["in_zone"])
        self.assertEqual(s["zone_damage"], 60)
        self.assertEqual(s["runner"]["hard_overruns"], 2)
        self.assertNotIn("ignored", s["end"])
        for key in ("macro_s", "hitrate", "sight", "fights", "a_band_s", "dmg_taken", "ms", "lines", "dropped",
                    "errors", "rules", "flips_per_min", "slot", "seed"):
            self.assertIn(key, s)

    def test_time_shares_flips_stage_ms(self):
        tl, buf, clk = make(TLM_MAX_LINES_PER_S=100)
        seq = ["X1", "X1", "X2", "X1"]
        for i, m in enumerate(seq):
            tl.record(world(1.0 + 0.1 * i, i, me_with()), None, Intent(macro=m, goal="M1" if m == "X1" else None),
                      {}, None, stage_ms={"world": 1.0 + i, "modes": 2.0})
        self.assertEqual(tl.flips, 2)
        self.assertAlmostEqual(tl.macro_time["X1"], 0.3)
        self.assertEqual(tl.stage_max["world"], 4.0)

    def test_record_never_raises(self):
        tl, buf, clk = make()
        tl.record(None, None, Intent(), {}, None)
        tl.record(SimpleNamespace(), None, None, {}, None)
        self.assertEqual(tl.errors, 2)

    def test_status_line_period_and_volume(self):
        tl, buf, clk = make()
        me = me_with()
        for i in range(4000):                        # 400 s of 10 Hz decisions
            clk.t += 0.1
            tl.record(world(0.1 * i, i, me), None, Intent(macro="X1"), {}, None)
        st = [r for r in lines(buf) if r["ev"] == "status"]
        self.assertEqual(len(st), 14)                # 400 / 30 + the first one
        self.assertLess(len(buf.getvalue()), 8000)


if __name__ == "__main__":
    unittest.main()
