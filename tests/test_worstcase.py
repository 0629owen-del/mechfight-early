"""Worst-case CPU gate, generated-observation validity and end-to-end scenario assertions (stdlib unittest).

    python3 -m unittest tests.test_worstcase -v

The CPU gate here is a LOCAL PROXY for the 1-CPU / 512 MB container (game-plan-v2 14.3 item 4): it runs on
this machine's cores, with this machine's clock. A pass here does not prove the docker `--cpus 1` gate; a
fail here is a real fail. Run `tools/bench.py --slow N` for a loaded-machine approximation.

Scenario tests drive the real `main.Bot` through `Bot(hello, deadline)` / `act(obs, t0)` only.
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import bench  # noqa: E402
import scenarios as S  # noqa: E402
import worstcase as W  # noqa: E402

HELLO, RECORDED, _END = W.load_stream()
SOFT, HARD, C = bench.deadlines(HELLO)
VALIDATOR = W.schema_validator() or W.MiniSchema()


# ------------------------------------------------------------------------------------ generated observations
class WorstCaseStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = W.generate(int(C.WORSTCASE_TICKS), seed=1)

    def test_schema_valid(self):
        bad = []
        for o in self.obs:
            e = VALIDATOR.errors(o, "Observation")
            if e:
                bad.append((o["tick"], e[:2]))
        self.assertEqual(bad[:3], [], f"{len(bad)} invalid observations")

    def test_other_seeds_schema_valid(self):
        for seed in (2, 3):
            for o in W.generate(120, seed=seed):
                self.assertEqual(VALIDATOR.errors(o, "Observation"), [], f"seed {seed} tick {o['tick']}")

    def test_deterministic(self):
        a, b = W.generate(30, seed=7), W.generate(30, seed=7)
        self.assertEqual(a, b)

    def test_ticks_and_time_advance(self):
        for p, q in zip(self.obs, self.obs[1:]):
            self.assertEqual(q["tick"] - p["tick"], 2)
            self.assertAlmostEqual(q["t"] - p["t"], 0.1, places=3)

    def test_load_every_tick(self):
        for o in self.obs:
            kinds = [m["kind"] for m in o["mechs"]]
            self.assertEqual(kinds.count("mech"), 16)
            self.assertEqual(kinds.count("npc"), 12)
            proj = [p["kind"] for p in o["projectiles"]]
            self.assertEqual(len(proj) - proj.count("mortar"), 20)
            self.assertEqual(proj.count("mortar"), 6)
            self.assertIsNotNone(o["offer"])
            self.assertEqual(len(o["offer"]["choices"]), 3)

    def test_entities_on_walkable_cells(self):
        g = W.Ground(W.load_map(), __import__("random").Random(0))
        for o in self.obs[::25]:
            for m in o["mechs"]:
                self.assertTrue(g.ok(m["pos"][0], m["pos"][1]), f"tick {o['tick']} actor {m['id']} off the walkable grid")
            self.assertTrue(g.ok(o["self"]["pos"][0], o["self"]["pos"][1]))

    def test_forces_expensive_paths(self):
        # zone field rebuild: next circle changes repeatedly
        nxt = {(o["zone"]["next_center"][0], o["zone"]["next_radius"]) for o in self.obs}
        self.assertGreaterEqual(len(nxt), len(self.obs) // 30)
        # flood: new loot ids keep arriving
        ids = {p["id"] for o in self.obs for p in o["pickups"]}
        self.assertGreaterEqual(len(ids), len(self.obs) // 25)
        # A*: a drop is always >= 200 m away somewhere in the stream, goal far from self
        far = 0
        for o in self.obs:
            sx, sy = o["self"]["pos"][:2]
            far += any(math.hypot(d["pos"][0] - sx, d["pos"][1] - sy) > 200 for d in o["drops"])
        self.assertGreater(far, len(self.obs) // 2)
        # urgent cover: a laser mech charging at us, in range, aimed at us
        hits = 0
        for o in self.obs:
            sx, sy = o["self"]["pos"][:2]
            for m in o["mechs"]:
                if m.get("charging"):
                    d = math.hypot(m["pos"][0] - sx, m["pos"][1] - sy)
                    err = abs(S.wrap(m["aim"][0] - S.yaw_to(sx - m["pos"][0], sy - m["pos"][1])))
                    if 40 < d < 120 and err < 0.4 and "laser" in m["weapons"]:
                        hits += 1
                        break
        self.assertGreaterEqual(hits, len(self.obs) // 5)

    def test_scenario_fixtures_schema_valid(self):
        sc = S.Scenario.quiet(S.find_open_site(1)).weapons("gun", "laser")
        sc.laser_at_us(70, 0.3)
        sc.add_bullet(60, 1.0)
        sc.add_shell(1.0, 1.0)
        sc.damage(90, "gun", 0.5, by=77)
        sc.offer()
        sc.pickup("laser", 2.5, 0.0)
        sc.drop(120, 2.0)
        sc.persist(3)
        self.assertEqual(sc.validate(), [])
        rec = S.Scenario.from_recorded(120)
        rec.add_mech(80, 0.2)
        rec.offer(["armor", "plating", "servos"])
        self.assertEqual(rec.validate(), [])


# ------------------------------------------------------------------------------------ CPU gate (LOCAL PROXY)
class CpuGateTests(unittest.TestCase):
    """p99 < SOFT_DEADLINE_MS and max < HARD_DEADLINE_MS over WORSTCASE_TICKS ticks.

    LOCAL PROXY: measured on this machine with Bot.act in-process, not in the docker `--cpus 1` container."""

    def _gate(self, name, obs):
        r = bench.run_inproc(HELLO, obs, SOFT, HARD, C)
        s = bench.summarize(r["times_ms"], SOFT, HARD)
        msg = f"{name}: p50 {s['p50']:.2f} p95 {s['p95']:.2f} p99 {s['p99']:.2f} max {s['max']:.2f} ms (soft {SOFT:g}, hard {HARD:g}; local proxy)"
        print("\n  " + msg)
        self.assertLess(s["p99"], SOFT, msg)
        self.assertLess(s["max"], HARD, msg)
        self.assertEqual(r["exceptions"], 0, f"{name}: per-stage exceptions inside act()")

    def test_recorded_stream(self):
        self._gate("recorded", RECORDED)

    def test_worst_case_stream(self):
        self._gate("worst-case", W.generate(int(C.WORSTCASE_TICKS), seed=1))


# ------------------------------------------------------------------------------------ scenarios vs the real Bot
def open_line_site(dist, start=0.0, seed0=0):
    """(site, bearing) with an open (canopy-aware) line of sight `dist` m along `bearing` on real ground."""
    for seed in range(seed0, seed0 + 60):
        site = S.find_open_site(seed)
        b = S.clear_bearing(site, dist, start)
        if b is not None:
            return site, b
    raise unittest.SkipTest("no open line of sight found on this map")


class ScenarioTests(unittest.TestCase):
    """One assertion per plan rule; each builds a fixture with tools/scenarios.py and checks the emitted action."""

    def test_laser_charging_at_us_moves_toward_cover(self):
        cs = S.find_cover_site(seed=0)
        sc = S.Scenario.quiet(cs["site"]).weapons("gun", "laser")
        sc.laser_at_us(pos=cs["charger"])
        act = sc.run()
        d = S.move_dir(act)
        self.assertIsNotNone(d, f"stands still: {act}")
        sx, sy = cs["site"][:2]
        best = min(S.angle_between(d, (q[0] - sx, q[1] - sy)) for q in cs["cover_cells"] if math.hypot(q[0] - sx, q[1] - sy) <= C.COVER_WALL_R)
        self.assertLess(math.degrees(best), 50, f"move {d} does not head for any cover cell within {C.COVER_WALL_R} m (best {math.degrees(best):.0f} deg)")

    def test_offer_picks_gp_order_index(self):
        for ids in (("armor", "laser_damage", "plating"), ("servos", "sensors", "armor"), ("gun_rate", "capacitor", "laser_damage")):
            with self.subTest(choices=ids):
                sc = S.Scenario.quiet(S.find_open_site(2)).weapons("gun", "laser")
                sc.offer(ids)
                act = sc.run()
                self.assertEqual(act.get("pick"), S.gp_pick_index(list(ids), ["gun", "laser"]), f"action {act}")

    def test_mortar_shell_at_our_feet_moves_away(self):
        site = S.find_open_site(3)
        sc = S.Scenario.quiet(site).weapons("gun", "laser")
        sc.add_shell(land_dx=2.0, land_dy=0.0, eta=1.0, shooter_bearing=1.0)
        sc.add_sound("mortar", 1.0, "mid")
        act = sc.run()
        d = S.move_dir(act)
        self.assertIsNotNone(d, f"stands still: {act}")
        self.assertLess(math.degrees(S.angle_between(d, (-1.0, 0.0))), 75, f"move {d} is not away from the landing point east of us")
        self.assertGreater(math.hypot(*act["move"]), 0.5)

    def test_visible_gun_enemy_is_aimed_at_and_fired_on(self):
        site, b = open_line_site(90.0, 0.8, seed0=4)
        sc = S.Scenario.quiet(site).weapons("gun", "laser")
        sc.add_mech(90.0, b, weapons=("gun",))
        act = sc.run()
        self.assertLess(abs(S.wrap(act["aim"][0] - b)), 0.2, f"aim {act.get('aim')} vs bearing {b:.2f}")
        self.assertTrue(any(act.get("fire") or []), f"no weapon fired at a visible enemy at 90 m: {act}")

    def test_outside_zone_heads_for_circle(self):
        sc = S.Scenario.quiet(S.find_open_site(5)).weapons("gun", "laser")
        sc.outside_zone(60.0)
        act = sc.run()
        d = S.move_dir(act)
        self.assertIsNotNone(d, f"stands still outside the zone: {act}")
        z = sc.cur["zone"]
        px, py = sc.pos[:2]
        self.assertLess(math.degrees(S.angle_between(d, (z["next_center"][0] - px, z["next_center"][1] - py))), 60)

    def test_weapon_pickup_in_reach_is_taken(self):
        sc = S.Scenario.quiet(S.find_open_site(6)).weapons("gun")
        pk = sc.pickup("laser", 2.5, 0.7)
        act = sc.run()
        self.assertEqual(act.get("take"), pk["id"], f"action {act}")

    def test_lone_drop_in_circle_is_the_goal(self):
        site, b = open_line_site(100.0, 2.0, seed0=7)
        sc = S.Scenario.quiet(site).weapons("gun", "laser")
        sc.drop(100.0, b)
        act = sc.run()
        d = S.move_dir(act)
        self.assertIsNotNone(d, f"stands still with a drop at 100 m: {act}")
        self.assertLess(math.degrees(S.angle_between(d, (math.cos(b), math.sin(b)))), 60)


if __name__ == "__main__":
    unittest.main()
