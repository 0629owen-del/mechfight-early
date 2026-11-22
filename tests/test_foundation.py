"""Foundation tests: config binding (plan 14.1 "Config binding"), runner fallbacks, SAFE, stream replay.

    cd bot && python3 -m unittest tests.test_foundation -v

Uses the kit next to this repo (../mechfight-kit) for the recorded hello/stream, the plan and the
schema; tests that need it are skipped when it is absent.
"""
from __future__ import annotations

import copy
import gc
import gzip
import importlib.util
import json
import math
import os
import queue
import sys
import threading
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.dont_write_bytecode = True

import config  # noqa: E402
import runner  # noqa: E402
import safe  # noqa: E402

KIT = os.path.join(os.path.dirname(REPO), "mechfight-kit")
STREAM = os.path.join(KIT, "check", "stream.jsonl.gz")
PLAN = os.path.join(KIT, "docs", "game-plan-v2.md")
SCHEMA = os.path.join(KIT, "mechfight.schema.json")
HAVE_KIT = os.path.exists(STREAM)


def load_stream():
    with gzip.open(STREAM, "rt") as f:
        msgs = [json.loads(l) for l in f if l.strip()]
    return msgs[0], [m for m in msgs if m.get("type") == "obs"], msgs[-1]


@unittest.skipUnless(HAVE_KIT, "mechfight-kit not found next to the repo")
class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hello, cls.obs, cls.end = load_stream()
        cls.cfg = cls.hello["config"]
        cls.C = config.bind(cls.cfg)

    def test_binds_without_errors(self):
        self.assertEqual(self.C.bind_errors, {})

    def test_cfg_rows_read_from_hello(self):
        self.assertTrue(config.CFG_PATHS)
        for name, path in config.CFG_PATHS.items():
            self.assertEqual(getattr(self.C, name), config.cfg_get(self.cfg, path), name)
        self.assertEqual(self.C.TESLA_MUL, 1.2)
        self.assertEqual(self.C.TESLA_MAX_STACKS, 2)
        self.assertEqual(self.C.DROP_RIVAL_ETA_MARGIN_S, self.cfg["drops"]["capture_s"])

    @unittest.skipUnless(os.path.exists(PLAN), "plan not found")
    def test_every_section10_name_present(self):
        import gen_config
        with open(PLAN, encoding="utf-8") as f:
            rows, order = gen_config.parse_plan(f.read())
        self.assertGreater(len(order), 400)
        for name in order:
            self.assertTrue(hasattr(self.C, name), name)
        self.assertEqual(set(order), set(config.DEFAULTS))

    @unittest.skipUnless(os.path.exists(PLAN), "plan not found")
    def test_generated_block_up_to_date(self):
        import gen_config
        block = gen_config.build_block(gen_config.DEF_PLAN, gen_config.DEF_GP, gen_config.DEF_HELLO)
        with open(os.path.join(REPO, "config.py"), encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(gen_config.splice(text, block), text, "rerun tools/gen_config.py")

    def test_every_derived_row_has_a_formula(self):
        V = dict(config.DEFAULTS)
        for n, p in config.CFG_PATHS.items():
            V[n] = config.cfg_get(self.cfg, p)
        out = config._derived(V, self.cfg)
        derived = {n for n, (k, _) in config.SOURCES.items() if k == "DERIVED"}
        self.assertEqual(derived - set(out), set())

    def test_derived_match_plan_defaults(self):
        """Every DERIVED row reproduces its table default on the recorded hello (2% or 0.5 abs)."""
        for name, (kind, _) in config.SOURCES.items():
            if kind != "DERIVED" or config.DEFAULTS[name] is None:
                continue
            want, got = config.DEFAULTS[name], getattr(self.C, name)
            if isinstance(want, dict):
                for k in want:
                    self.assertAlmostEqual(got[k], want[k], delta=0.5, msg=f"{name}[{k}]")
            else:
                self.assertAlmostEqual(got, want, delta=max(0.5, 0.02 * abs(want)), msg=name)

    def test_derived_values_explicit(self):
        C = self.C
        self.assertEqual(C.ENDGAME_T, 375)
        self.assertEqual(C.FINAL_SHRINK_T, 390)
        self.assertEqual(C.ZONE_CLOSED_T, 415)
        self.assertEqual(C.PHASE_T3_SHRINK, 230)
        self.assertEqual(C.PHASE_T4_SHRINK, 295)
        self.assertEqual(C.LOOT_END_T, 60)
        self.assertEqual(C.BOOT_MAX_MS, 15000)
        self.assertAlmostEqual(C.AIM_LATENCY_S, 0.1)
        self.assertAlmostEqual(C.RECOVER_MAX_S, 30, delta=0.2)          # 5 + 0.8*1400/45 = 29.9
        self.assertEqual(C.COAST_BOOST_MIN_ENERGY, 17)
        self.assertAlmostEqual(C.JUKE_MIN_RANGE, 75, delta=0.1)
        self.assertAlmostEqual(C.LASER_CHARGE_MIN_GAP, 64.6, delta=0.05)
        self.assertAlmostEqual(C.JUMP_H, 2.75)
        self.assertEqual(C.LOOT_HARD_END_T, self.cfg["drops"]["times"][1])
        self.assertAlmostEqual(C.LASER_VS_MORTAR_MIN, 235.6, delta=0.05)
        self.assertAlmostEqual(C.CLIMB_MAX_H, 51.3, delta=0.1)
        self.assertEqual(C.SOUND_BAND_REP, {"near": 45, "mid": 110, "far": 220})
        self.assertEqual(C.NPC_GUN_MIN_AMMO, 8)
        self.assertAlmostEqual(C.MORTAR_DODGE_R, 10)
        self.assertAlmostEqual(C.MORTAR_SELF_SAFE_M, 11.5)
        self.assertIsNone(C.RUSH_A_MIN)  # calibrated by assess at hello

    def test_juke_min_range_live_with_cubes(self):
        C = config.bind(self.cfg)
        me = copy.deepcopy(self.obs[0]["self"])
        me["cubes"] = 10
        me["stats"]["radius"] = 2.5 * 1.5
        config.update_live(C, me)
        self.assertAlmostEqual(C.JUKE_MIN_RANGE, 108, delta=0.6)
        self.assertAlmostEqual(C.LASER_HOLD_MIN, C.JUKE_MIN_RANGE)
        self.assertAlmostEqual(C.ROOF_EDGE_BACKOFF, 6.5 * 1.5)
        # base values are untouched
        self.assertAlmostEqual(C.base["JUKE_MIN_RANGE"], 75.04, delta=0.01)

    def test_live_held_lightning_and_upgrades(self):
        C = config.bind(self.cfg)
        me = copy.deepcopy(self.obs[0]["self"])
        me["weapons"][1] = {"kind": "lightning", "cooldown": 0, "stats": dict(self.cfg["weapons"]["lightning"], range=54,
                                                                             energy=1.95)}
        me["stats"]["vision_bonus"] = 25
        me["stats"]["boost_drain"] = 17 * 0.75
        config.update_live(C, me)
        self.assertEqual(C.LIGHTNING_RANGE, 54)
        self.assertAlmostEqual(C.LIGHTNING_MIN_ENERGY, 5.85)
        self.assertEqual(C.LASER_HOLD_MAX, 225)
        self.assertEqual(C.NEIGHBOR_CLEAR_R, 225)
        self.assertAlmostEqual(C.COAST_BOOST_MIN_ENERGY, 12.75)

    def test_modified_hello_propagates(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["weapons"]["lightning"]["range"] = 54
        cfg["zone"]["phases"][5]["hold_s"] = 20
        cfg["bot"]["hello_budget_ms"] = 10000
        C = config.bind(cfg)
        self.assertEqual(C.bind_errors, {})
        self.assertEqual(C.LIGHTNING_RANGE, 54)
        self.assertEqual(C.LIGHTNING_KEEPOUT_R, 59)
        self.assertAlmostEqual(C.LASER_CHARGE_MIN_GAP, 73.625)
        self.assertEqual(C.FINAL_SHRINK_T, 395)
        self.assertEqual(C.ZONE_CLOSED_T, 420)
        self.assertEqual(C.FINAL_ARRIVE_T, 390)
        self.assertEqual(C.BOOT_MAX_MS, 5000)

    def test_toggles(self):
        C = self.C
        # 16.1 user decisions
        self.assertEqual(C.ZONE_OUTRANKS_DODGE, "collapse")
        self.assertEqual(C.SWAP_RULE, "rank")
        self.assertIs(C.ALIVE_LE3_PASSIVE, True)
        self.assertIs(C.ALIVE2_FIGHT_IF_BEHIND, True)
        self.assertEqual(C.LASER_VIEW_RESPONSE, "gl_only")
        self.assertIs(C.TLM_DEBUG_CHANNEL, False)
        # everything else stays GP
        self.assertEqual(C.ENGAGE_A_MIN, 1.0)
        self.assertEqual(C.AB_D10_ENGAGE_A_MIN, 1.2)
        for name, want in (("ENDGAME_FIRE_POLICY", "gp"), ("DROP_CONTEST", "gp"), ("INVESTIGATE_ANYTIME", False),
                           ("SWAP_GATE", "gp"), ("RUSH_GATE", "calibrated"), ("FIRE_DISCIPLINE", "gp"),
                           ("THIRD_PARTY_RULE", "gp"), ("COMMIT_ENABLED", False), ("LASER_PEEK", False),
                           ("FALL_BRAKE_ENABLED", False), ("MORTAR_DROP_ON_LASER", False)):
            self.assertEqual(getattr(C, name), want, name)

    def test_side_tables(self):
        self.assertEqual(config.LOADOUT_RANK["GL"]["LIGHTNING"], 3)
        self.assertEqual(config.LOADOUT_RANK["ZL"]["MIXED"], 3)
        self.assertEqual(config.LOADOUT_RANK["empty"]["LASER"], 0)
        self.assertEqual(config.UPGRADE_PRIORITY["GL"][0], "lightning_efficiency")
        self.assertEqual(config.UPGRADE_PRIORITY["MECH"], ["plating", "armor", "sensors", "servos", "nanites"])
        self.assertEqual(config.loadout_key(["lightning", "gun"]), "GL")
        self.assertEqual(config.loadout_key(["lightning", "laser"]), "ZL")
        self.assertEqual(config.loadout_key([]), "empty")
        ids = {u["id"] for u in self.cfg["upgrades"]}
        for lst in config.UPGRADE_PRIORITY.values():
            self.assertTrue(set(lst) <= ids)
        self.assertTrue(set(self.C.UPGRADE_FALLBACK) <= ids)

    def test_bind_never_raises(self):
        for bad in (None, {}, {"zone": {"phases": []}}, {"mech": "x"}):
            C = config.bind(bad)
            self.assertEqual(C.ENDGAME_T, 375)  # table default kept


@unittest.skipUnless(HAVE_KIT, "mechfight-kit not found next to the repo")
class TestSafe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hello, cls.obs, _ = load_stream()
        cls.C = config.bind(cls.hello["config"])

    def test_moves_to_next_center(self):
        o = self.obs[0]
        a = safe.act(o, self.C)
        x, y, _ = o["self"]["pos"]
        cx, cy = o["zone"]["next_center"]
        d = math.hypot(cx - x, cy - y)
        self.assertAlmostEqual(a["move"][0], (cx - x) / d)
        self.assertAlmostEqual(a["move"][1], (cy - y) / d)
        self.assertAlmostEqual(a["aim"][0], math.atan2(cy - y, cx - x))
        self.assertEqual(a["fire"], [False, False])
        self.assertFalse(a["boost"])

    def test_aims_and_fires_gun_at_mech(self):
        o = copy.deepcopy(self.obs[0])
        x, y, z = o["self"]["pos"]
        o["mechs"] = [{"id": 5, "kind": "mech", "pos": [x + 50, y, z], "vel": [0, 0, 0], "aim": [0, 0]}]
        a = safe.act(o, self.C)
        self.assertAlmostEqual(a["aim"][0], 0.0)
        o["self"]["aim"] = list(a["aim"])
        a = safe.act(o, self.C)
        self.assertEqual(a["fire"], [True, False])
        o["self"]["in_zone"] = False
        self.assertTrue(safe.act(o, self.C)["boost"])

    def test_lightning_gate_and_pick(self):
        o = copy.deepcopy(self.obs[0])
        x, y, z = o["self"]["pos"]
        o["self"]["weapons"][1] = {"kind": "lightning", "cooldown": 0, "stats": dict(self.hello["config"]["weapons"]["lightning"])}
        o["mechs"] = [{"id": 5, "kind": "mech", "pos": [x, y + 30, z], "vel": [0, 0, 0], "aim": [0, 0]}]
        o["self"]["aim"] = [math.pi / 2, 0.0]
        a = safe.act(o, self.C)
        self.assertTrue(a["fire"][1])
        o["offer"] = {"id": 1, "expires": 5, "choices": [{"id": "gyros", "name": "", "desc": ""},
                                                         {"id": "plating", "name": "", "desc": ""},
                                                         {"id": "lightning_efficiency", "name": "", "desc": ""}]}
        self.assertEqual(safe.act(o, self.C)["pick"], 2)  # GL list starts with Superconductor


# ------------------------------------------------------------------ runner harness
class Feed:
    def __init__(self):
        self.q = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        x = self.q.get()
        if x is None:
            raise StopIteration
        return x

    def send(self, msg):
        self.q.put(json.dumps(msg) + "\n")


class Sink:
    def __init__(self):
        self.q = queue.Queue()
        self.raw = []

    def write(self, s):
        for part in s.splitlines():
            if part:
                self.raw.append(part)
                self.q.put(json.loads(part))

    def flush(self):
        pass


class FakeBot:
    name = "fake"
    mode = "ok"            # ok | raise | sleep
    sleep_s = 0.2
    init_sleep = 0.0
    calls = 0

    def __init__(self, hello, deadline):
        if FakeBot.init_sleep:
            time.sleep(FakeBot.init_sleep)
        self.deadline = deadline

    def act(self, obs, t0):
        FakeBot.calls += 1
        if FakeBot.mode == "raise":
            raise RuntimeError("boom")
        if FakeBot.mode == "sleep":
            time.sleep(FakeBot.sleep_s)
        if FakeBot.mode == "spin":  # CPU-bound: holds the GIL
            end = time.perf_counter() + FakeBot.sleep_s
            while time.perf_counter() < end:
                pass
        if FakeBot.mode == "nan":
            return {"move": [float("nan"), 0.0], "aim": [1.0, 0.0], "fire": [True], "debug": {"label": "x" * 200,
                                                                                         "lines": [1]}}
        return {"move": [1.0, 0.0], "aim": [0.5, 0.0], "tick": -99}


@unittest.skipUnless(HAVE_KIT, "mechfight-kit not found next to the repo")
class TestRunner(unittest.TestCase):
    def setUp(self):
        self.hello, self.obs, self.end = load_stream()
        FakeBot.mode, FakeBot.calls, FakeBot.init_sleep = "ok", 0, 0.0
        self.feed, self.sink = Feed(), Sink()
        self.r = runner.Runner(FakeBot, inp=self.feed, out=self.sink)
        self.th = threading.Thread(target=self.r.run, daemon=True)
        self.th.start()

    def tearDown(self):
        self.feed.q.put(None)
        self.th.join(2)
        gc.enable()
        gc.unfreeze()

    def start(self, hello=None):
        self.feed.send(hello or self.hello)
        ready = self.sink.q.get(timeout=5)
        self.assertEqual(ready, {"type": "ready", "name": "fake"})

    def ask(self, o, timeout=2.0):
        t = time.perf_counter()
        self.feed.send(o)
        got = self.sink.q.get(timeout=timeout)
        return got, (time.perf_counter() - t) * 1000

    def test_tick_echo(self):
        self.start()
        for o in self.obs[:5]:
            a, _ = self.ask(o)
            self.assertEqual(a["tick"], o["tick"])
            self.assertEqual(a["move"], [1.0, 0.0])
        self.feed.send(self.end)
        self.th.join(2)
        self.assertFalse(self.th.is_alive())

    def test_exception_falls_back_to_safe_same_tick(self):
        self.start()
        FakeBot.mode = "raise"
        o = self.obs[3]
        a, _ = self.ask(o)
        want = runner.sanitize(safe.act(o, self.r.C))
        want["tick"] = o["tick"]
        self.assertEqual(a, want)
        self.assertEqual(self.r.info.exceptions, 1)

    def test_t1_switch_to_safe(self):
        self.start()
        FakeBot.mode = "raise"
        C = self.r.C
        for o in self.obs[: int(C.SAFE_MODE_EXC_N)]:
            a, _ = self.ask(o)
            self.assertEqual(a["tick"], o["tick"])
        self.assertGreater(self.r.info.safe_until_t, self.obs[int(C.SAFE_MODE_EXC_N) - 1]["t"])
        calls = FakeBot.calls
        FakeBot.mode = "ok"
        nxt = self.obs[int(C.SAFE_MODE_EXC_N)]
        a, _ = self.ask(nxt)
        self.assertEqual(FakeBot.calls, calls)  # act() not called while SAFE is active
        self.assertEqual(a["tick"], nxt["tick"])
        # T7: after SAFE_MODE_S quiet the bot is back
        later = next(o for o in self.obs if o["t"] >= self.r.info.safe_until_t)
        a, _ = self.ask(later)
        self.assertEqual(a["move"], [1.0, 0.0])

    def test_hard_deadline_guard(self):
        self.start()
        FakeBot.mode = "sleep"
        o = self.obs[4]
        a, ms = self.ask(o)
        self.assertEqual(a["tick"], o["tick"])
        self.assertLess(ms, self.r.C.HARD_DEADLINE_MS + 30)
        self.assertEqual(a["move"], runner.sanitize(safe.act(o, self.r.C))["move"])
        self.assertEqual(self.r.info.hard_overruns, 1)
        # the next observation arrives while act() is still running: SAFE again, on time
        a, ms = self.ask(self.obs[5])
        self.assertEqual(a["tick"], self.obs[5]["tick"])
        self.assertLess(ms, self.r.C.HARD_DEADLINE_MS)

    def test_hard_deadline_guard_cpu_bound(self):
        self.start()
        FakeBot.mode = "spin"
        a, ms = self.ask(self.obs[6])
        self.assertEqual(a["tick"], self.obs[6]["tick"])
        self.assertLess(ms, self.r.C.HARD_DEADLINE_MS + 6)
        self.assertEqual(self.r.info.hard_overruns, 1)

    def test_sanitize_bad_action(self):
        self.start()
        FakeBot.mode = "nan"
        a, _ = self.ask(self.obs[1])
        self.assertNotIn("move", a)
        self.assertEqual(a["fire"], [True, False])
        self.assertEqual(a["debug"], {"label": "x" * 80})
        self.assertNotIn("NaN", self.sink.raw[-1])

    def test_ready_within_deadline_when_init_is_slow(self):
        h = copy.deepcopy(self.hello)
        h["config"]["bot"]["hello_budget_ms"] = 5200  # BOOT_MAX_MS = 200 ms
        FakeBot.init_sleep = 0.6
        t = time.perf_counter()
        self.start(h)
        self.assertLess(time.perf_counter() - t, 0.45)
        a, _ = self.ask(self.obs[0])  # init still running: SAFE
        self.assertEqual(a["tick"], self.obs[0]["tick"])
        self.assertNotEqual(a.get("move"), [1.0, 0.0])
        time.sleep(0.6)
        a, _ = self.ask(self.obs[1])  # attached now
        self.assertEqual(a["move"], [1.0, 0.0])


@unittest.skipUnless(HAVE_KIT and os.path.exists(SCHEMA), "kit not found")
class TestStreamReplay(unittest.TestCase):
    def test_bot_over_recorded_stream(self):
        spec = importlib.util.spec_from_file_location("kit_check", os.path.join(KIT, "check", "check.py"))
        chk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chk)
        schema = chk.Schema(SCHEMA)
        import main
        hello, obs, end = load_stream()
        bot = main.Bot(hello, time.perf_counter() + 15)
        for o in obs:
            a = runner.sanitize(bot.act(o, time.perf_counter()))
            a["tick"] = o["tick"]
            self.assertEqual(schema.errors(a, "Action"), [], o["tick"])
        self.assertEqual(bot.take_exceptions(), 0)


if __name__ == "__main__":
    unittest.main()
