"""FSM loop freedom (plan 5.7, 14.1 "FSM loop freedom"): the 600-decision property test for every timeout
and exit-memory writer, the "own exit with trigger still true" class (every macro/goal own exit, LOSE
(a)-(i) as the X3 entry trigger), and the CORNERED <-> X3 bounded-rate check.

Each fixture holds the entry predicate true for 600 decisions (60 s) and asserts:
  * total changes <= designed exits + timeouts (computed from the cooldowns),
  * no re-entry with the same key before its exit memory expires,
  * no A -> B -> A cycle shorter than the relevant cooldown,
  * every timer that should fire fires.

    python3 -m unittest tests.test_loopfree -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import (FightEvalX, Harness, build_A, goal_entries, goal_exits, macro_changes, set_flags)  # noqa: E402

from contracts import DropState, FightEval, ExitMemory  # noqa: E402
import tactics_drop  # noqa: E402

N = 600
EPS = 1e-6


def seq_of(changes):
    """[(t, to_state)] from history records."""
    return [(h[0], h[3]) for h in changes]


def short_cycles(seq, period):
    """A -> B -> A with t(A2) - t(A1) < period."""
    out = []
    for i in range(len(seq) - 2):
        if seq[i][1] == seq[i + 2][1] and seq[i + 2][0] - seq[i][0] < period - EPS:
            out.append((seq[i], seq[i + 1], seq[i + 2]))
    return out


class LoopFreeMixin:
    def assertSpacing(self, times, gap, msg=""):
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(b - a, gap - 1e-3, f"{msg}: {a:.2f} -> {b:.2f} < {gap}")

    def bound(self, period, n_per_cycle=2, extra=1):
        return n_per_cycle * math.ceil(N * 0.1 / period) + extra


# ====================================================================== macro timeouts / exits
class TestX2Exits(unittest.TestCase, LoopFreeMixin):
    def _fight(self, pos, events=True, weapons=("gun", "laser")):
        h = Harness(t=100.0, weapons=weapons)
        h.world.add_mech(5, pos, weapons=("gun", "laser"))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, win=True, win_rule="RZ1", win_sub="DUEL")
        if events:
            h.world.persist_events = [{"type": "hit", "target": 5, "amount": 10, "weapon": "gun"}]
        return h

    def _check(self, h, why, cooldown, active):
        mc = macro_changes(h.modes)
        enters = [r[0] for r in mc if r[3] == "X2"]
        exits = [r for r in mc if r[2] == "X2" and r[3] == "X1"]
        self.assertTrue(exits, "timer never fired")
        self.assertTrue(all(r[5] == "T5:" + why for r in exits), [r[5] for r in exits])
        for r in exits:                                   # no re-entry with the same key before until
            later = [e for e in enters if e > r[0]]
            if later:
                self.assertGreaterEqual(later[0] - r[0], cooldown - 1e-3)
        self.assertLessEqual(len(mc), self.bound(active + cooldown))
        self.assertFalse(short_cycles(seq_of(mc), cooldown))
        return exits

    def test_x2_engage_timeout(self):
        h = self._fight((150.0, 0.0, 0.0))
        h.run(N)
        ex = self._check(h, "timeout", h.C.M4_TIMEOUT_COOLDOWN_S, h.C.ENGAGE_TIMEOUT_S)
        self.assertAlmostEqual(ex[0][0] - 100.0, h.C.ENGAGE_TIMEOUT_S, delta=0.21)

    def test_x2_no_hit_timeout(self):
        h = self._fight((150.0, 0.0, 0.0), events=False)
        h.run(N)
        ex = self._check(h, "timeout", h.C.M4_TIMEOUT_COOLDOWN_S, h.C.ENGAGE_NO_HIT_S)
        self.assertAlmostEqual(ex[0][0] - 100.0, h.C.ENGAGE_NO_HIT_S, delta=0.21)

    def test_x2_rl2_no_chase(self):
        h = self._fight((300.0, 0.0, 0.0))            # beyond laser.range, not closing (PB_Z)
        h.run(N)
        self._check(h, "RL2", h.C.NO_CHASE_COOLDOWN_S, h.C.ENGAGE_RANGE_LOSS_S)


class TestX3Timeout(unittest.TestCase, LoopFreeMixin):
    def _flee(self, under_fire):
        h = Harness(t=100.0)
        h.world.add_mech(5, (150.0, 0.0, 0.0), weapons=("gun", "laser"))
        h.assessor.fn = lambda w, A, macro, target, ctx: FightEval(target_id=5, lose_clauses=frozenset("f"))
        h.A_fn = lambda hh: build_A(hh.world, A_ratio=0.5)
        if under_fire:
            h.pre = lambda hh: hh.world.damage(5.0, 5)
        return h

    def test_quiet_branch_writes_flee_star(self):
        h = self._flee(False)
        h.run(N)
        C = h.C
        mc = macro_changes(h.modes)
        exits = [r for r in mc if r[2] == "X3"]
        self.assertTrue(exits)
        self.assertTrue(all(r[5] == "T6:timeout" for r in exits))
        enters = [r[0] for r in mc if r[3] == "X3"]
        self.assertSpacing(enters, C.DISENGAGE_MAX_S + C.DISENGAGE_COOLDOWN_S, "X3 re-entry")
        self.assertLessEqual(len(mc), self.bound(C.DISENGAGE_MAX_S + C.DISENGAGE_COOLDOWN_S))
        self.assertFalse(short_cycles(seq_of(mc), C.DISENGAGE_COOLDOWN_S))

    def test_under_fire_branch_cornered_no_memory(self):
        h = self._flee(True)
        h.run(N)
        C = h.C
        mc = macro_changes(h.modes)
        to_corner = [r for r in mc if r[2] == "X3" and r[5] == "T6:timeout_fire"]
        self.assertTrue(to_corner, [r[5] for r in mc])
        self.assertFalse(any(k == ("FLEE", "*") for k, _ in h.modes.exit_memory.items()))
        # bounded: each CORNERED stint >= CORNER_MIN_HOLD_S, each X3 stint == DISENGAGE_MAX_S
        self.assertLessEqual(len(mc), self.bound(C.DISENGAGE_MAX_S + C.CORNER_MIN_HOLD_S))
        self.assertFalse(short_cycles(seq_of(mc), C.CORNER_MIN_HOLD_S))


# ====================================================================== goal timeouts / exits
class TestGoalTimeouts(unittest.TestCase, LoopFreeMixin):
    def test_m6_recover_timeout(self):
        h = Harness(t=100.0)
        h.world.me.hp = 300.0
        h.run(N)
        C = h.C
        ent = goal_entries(h.modes, "M6")
        exits = goal_exits(h.modes, "M6")
        self.assertTrue(exits and exits[0][5] == "timeout")
        self.assertAlmostEqual(exits[0][0] - ent[0][0], C.RECOVER_MAX_S, delta=0.21)
        self.assertSpacing([e[0] for e in ent], C.RECOVER_MAX_S + C.RECOVER_RETRY_S, "M6")

    def test_m7_abandon_occupied(self):
        h = Harness(t=100.0)
        h.world.drops[20000] = DropState(20000, (0.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 90.0, 95.0)
        h.world.add_mech(6, (3.0, 0.0, 0.0), weapons=("gun",))                # inside the ring
        with mock.patch.object(tactics_drop, "go_test", lambda ctx, dr: (True, False)):
            h.run(N)
        C = h.C
        ent = goal_entries(h.modes, "M7")
        exits = goal_exits(h.modes, "M7")
        self.assertTrue(exits and all(r[5] in ("occupied", "contested") for r in exits))
        self.assertSpacing([e[0] for e in ent], C.DROP_ABANDON_COOLDOWN_S, "M7")
        self.assertLessEqual(len(ent), math.ceil(N * 0.1 / C.DROP_ABANDON_COOLDOWN_S) + 1)

    def test_m7_timeout_permanent(self):
        h = Harness(t=100.0)
        h.world.drops[20000] = DropState(20000, (0.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 90.0, 95.0)
        with mock.patch.object(tactics_drop, "go_test", lambda ctx, dr: (True, False)):
            h.run(max(N, int((h.C.DROP_TIMEOUT_S + 10.0) / 0.1)))    # the timeout counts from the landing (t = 95)
        ent = goal_entries(h.modes, "M7")
        exits = goal_exits(h.modes, "M7")
        self.assertEqual(len(ent), 1)
        self.assertEqual(exits[0][5], "timeout")
        self.assertAlmostEqual(exits[0][0], 95.0 + h.C.DROP_TIMEOUT_S, delta=0.21)
        self.assertEqual(h.modes.exit_memory.until("DROP", 20000), ExitMemory.PERMANENT)

    def test_m3a_ambush_timeout(self):
        h = Harness(t=100.0, weapons=("gun", "lightning"))
        h.world.add_ghost(-1, (40.0, 0.0), "near")
        h.pre = lambda hh: hh.world.hear("step", -1, "near")
        h.run(N)
        C = h.C
        ent = goal_entries(h.modes, "M3A")
        self.assertEqual(len(ent), 1)
        ex = goal_exits(h.modes, "M3A")
        self.assertEqual(ex[0][5], "timeout")
        self.assertAlmostEqual(ex[0][0] - ent[0][0], C.AMBUSH_MAX_S, delta=0.21)
        self.assertEqual(h.modes.exit_memory.until("AMBUSH", -1), ExitMemory.PERMANENT)

    def test_m3h_hide_timeout(self):
        h = Harness(t=100.0)
        h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.world.add_mech(6, (0.0, 150.0, 0.0))
        h.run(N)
        C = h.C
        ent = goal_entries(h.modes, "M3H")
        ex = goal_exits(h.modes, "M3H")
        self.assertTrue(ex and all(r[5] == "timeout" for r in ex))
        self.assertSpacing([e[0] for e in ent], C.HIDE_MAX_S + C.HIDE_RETRY_S, "M3H")

    def test_m8_investigate_timeout(self):
        h = Harness(t=240.0, weapons=("gun", "lightning"))
        h.A_fn = lambda hh: build_A(hh.world, terrain_call="LIGHTNING")
        h.world.add_ghost(-1, (100.0, 0.0), "mid")
        h.pre = lambda hh: hh.world.hear("gun", -1, "mid")
        h.run(N)
        C = h.C
        ent = goal_entries(h.modes, "M8")
        self.assertEqual(len(ent), 1)
        ex = goal_exits(h.modes, "M8")
        self.assertEqual(ex[0][5], "timeout")
        self.assertAlmostEqual(ex[0][0] - ent[0][0], C.INVESTIGATE_MAX_S, delta=0.21)
        self.assertEqual(h.modes.exit_memory.until("INVESTIGATE", -1), ExitMemory.PERMANENT)

    def test_m1_item_timeout_blacklist(self):
        from contracts import PickupMem
        h = Harness(t=0.0, weapons=("gun",))
        h.world.pickups[8001] = PickupMem(8001, "laser", (100.0, 0.0, 0.0), 0.0)
        h.run(N)
        C = h.C
        to = [r for r in h.modes.history if r[1] == "loot" and r[5] == "timeout"]
        self.assertGreaterEqual(len(to), 1)
        self.assertSpacing([r[0] for r in to], C.LOOT_BLACKLIST_S, "item re-target")
        self.assertEqual(len(goal_entries(h.modes, "M1")), 1)        # M1 stays (loot window)

    def test_m2_spot_timeout(self):
        h = Harness(t=100.0)
        h.nav.spots["hold"] = ((200.0, 0.0), 1.0, 0)
        h.run(N)
        C = h.C
        ex = goal_exits(h.modes, "M2")
        self.assertTrue(ex and ex[0][5] == "timeout")
        ent = goal_entries(h.modes, "M2")
        self.assertSpacing([e[0] for e in ent], C.SPOT_BLACKLIST_S, "M2 same spot")
        self.assertFalse(short_cycles(seq_of(goal_entries(h.modes)), C.MODE_MIN_HOLD_S))

    def test_m3_hold_max(self):
        h = Harness(t=100.0)
        h.nav.spots["hold"] = ((0.0, 0.0), 1.0, 0)
        h.run(N)
        C = h.C
        ex = goal_exits(h.modes, "M3")
        self.assertTrue(ex and ex[0][5] == "timeout")
        self.assertAlmostEqual(ex[0][0] - 100.0, C.HOLD_MAX_S, delta=0.21)
        self.assertLessEqual(len(ex), math.ceil(N * 0.1 / C.HOLD_MAX_S))

    def test_m2u_timeout_replans_and_stays(self):
        h = Harness(t=100.0)
        h.A_fn = lambda hh: build_A(hh.world, slack=5.0)
        h.run(N)
        self.assertEqual(len(goal_entries(h.modes)), 1)
        self.assertEqual(h.modes.goal, "M2U")
        timers = [r for r in h.modes.history if r[1] == "timer"]
        self.assertGreaterEqual(len(timers), 3)                         # timer fires, no memory, stays


# ====================================================================== own exit with the trigger still true
class TestOwnExitTriggerTrue(unittest.TestCase, LoopFreeMixin):
    """LOSE (a)-(i) as the X3 entry trigger while the X3 own exit (general / R6.1) is also true."""

    def _run(self, clause, weapons=("gun", "laser"), x2_first=False):
        h = Harness(t=100.0, weapons=weapons)
        wk = ("gun", "lightning") if clause == "e" else ("gun", "laser")
        h.world.add_mech(5, (48.0 if clause == "e" else 150.0, 0.0, 0.0), weapons=wk)
        h.A_fn = lambda hh: build_A(hh.world, los={5: False}, A_ratio=0.5)

        def fn(w, A, macro, target, ctx):
            if x2_first and macro != "X2":
                return FightEval(target_id=5, win=True, win_sub="DUEL")
            return FightEval(target_id=5, lose_clauses=frozenset(clause), lose_tracks={clause: 5})
        h.assessor.fn = fn
        if clause not in ("e",):
            h.pre = lambda hh: hh.world.damage(5.0, 5)           # reach precondition via damage
        h.run(N)
        return h, macro_changes(h.modes)

    def test_general_exit_clauses(self):
        for c in "dfgh":
            with self.subTest(clause=c):
                h, mc = self._run(c)
                C = h.C
                enters = [r[0] for r in mc if r[3] == "X3"]
                self.assertTrue(enters)
                # general exit has no memory: bounded by DISENGAGE_CLEAR_S per stint (designed exits)
                self.assertSpacing(enters, C.DISENGAGE_CLEAR_S, c)
                self.assertLessEqual(len(mc), self.bound(C.DISENGAGE_CLEAR_S))

    def test_clause_e_r61_exit_memory(self):
        h, mc = self._run("e")
        C = h.C
        enters = [r[0] for r in mc if r[3] == "X3"]
        exits = [r for r in mc if r[2] == "X3"]
        self.assertTrue(exits and all(r[5] == "T6:R6.1" for r in exits))
        self.assertSpacing(enters, C.DISENGAGE_CLEAR_S + C.DISENGAGE_COOLDOWN_S, "e")
        self.assertFalse(short_cycles(seq_of(mc), C.DISENGAGE_COOLDOWN_S))

    def test_clause_i_track_memory(self):
        h, mc = self._run("i", weapons=("gun", "lightning"))
        C = h.C
        enters = [r[0] for r in mc if r[3] == "X3"]
        self.assertTrue(enters)
        self.assertSpacing(enters, C.DISENGAGE_CLEAR_S + C.DISENGAGE_COOLDOWN_S, "i")

    def test_x2_only_clauses(self):
        for c in "abc":
            with self.subTest(clause=c):
                h, mc = self._run(c, x2_first=True)
                C = h.C
                x2 = [r[0] for r in mc if r[3] == "X2"]
                to_x3 = [r[0] for r in mc if r[2] == "X2" and r[3] == "X3"]
                self.assertTrue(to_x3)
                for t3 in to_x3:                                   # REENGAGE memory respected
                    later = [t for t in x2 if t > t3]
                    if later:
                        self.assertGreaterEqual(later[0] - t3, C.REENGAGE_COOLDOWN_S - 1e-3)
                self.assertFalse(short_cycles(seq_of(mc), C.DISENGAGE_CLEAR_S))

    def test_goal_own_exit_trigger_true_m3h(self):
        """M3H: predicate false for MODE_MIN_HOLD_S then true again at once -> (HIDE, *) blocks re-entry."""
        h = Harness(t=100.0)
        a = h.world.add_mech(5, (150.0, 0.0, 0.0))
        h.world.add_mech(6, (0.0, 150.0, 0.0))

        def pre(hh):
            ph = (hh.world.tick.t - 100.0) % 4.0
            if ph < 1.5:
                set_flags(a, fresh=False)
                a.last_seen_t = hh.world.tick.t - 30.0
                a.visible = False
            else:
                set_flags(a, fresh=True)
                a.last_seen_t = hh.world.tick.t
                a.visible = True
        h.pre = pre
        h.run(N)
        C = h.C
        ent = goal_entries(h.modes, "M3H")
        self.assertTrue(ent)
        self.assertSpacing([e[0] for e in ent], C.HIDE_RETRY_S, "M3H re-entry")

    def test_goal_own_exit_trigger_true_m3a_m8(self):
        """M3A / M8 'contact seen' while the ghost and its response persist -> permanent memory, 1 entry."""
        h = Harness(t=100.0, weapons=("gun", "lightning"))
        h.world.add_ghost(-1, (40.0, 0.0), "near")
        h.pre = lambda hh: hh.world.hear("step", -1, "near")
        h.run(5)
        h.world.add_mech(5, (40.0, 0.0, 0.0), weapons=("gun",))     # the contact appears on the bearing
        h.run(N)
        self.assertEqual(len(goal_entries(h.modes, "M3A")), 1)
        self.assertEqual(h.modes.exit_memory.until("AMBUSH", -1), ExitMemory.PERMANENT)

    def test_goal_own_exit_trigger_true_m7(self):
        """M7 abandon (ring occupied) while the go test keeps passing: one entry per cooldown."""
        h = Harness(t=100.0)
        h.world.drops[20000] = DropState(20000, (0.0, 0.0, 0.0), 8.0, 0.0, 4.0, 0.0, None, 90.0, 95.0)
        h.world.add_mech(6, (2.0, 0.0, 0.0), weapons=("gun",))
        with mock.patch.object(tactics_drop, "go_test", lambda ctx, dr: (True, False)):
            h.run(N)
        self.assertSpacing([e[0] for e in goal_entries(h.modes, "M7")], h.C.DROP_ABANDON_COOLDOWN_S, "M7")


# ====================================================================== CORNERED <-> X3 bounded rate
class TestCorneredRate(unittest.TestCase, LoopFreeMixin):
    def _run(self, amp, period):
        h = Harness(t=100.0)
        h.world.add_mech(5, (100.0, 0.0, 0.0), weapons=("gun", "lightning"))
        h.A_fn = lambda hh: build_A(hh.world, A_ratio=0.5)
        t_close = 5.0
        log = []

        def fn(w, A, macro, target, ctx):
            eta = t_close + amp * math.sin(2 * math.pi * (w.tick.t - 100.0) / period)
            fe = FightEvalX(target_id=5, lose_clauses=frozenset("f"), cornered=eta >= t_close, t_close=t_close)
            fe.cover_eta = eta
            log.append((w.tick.t, eta))
            return fe
        h.assessor.fn = fn
        h.run(N)
        return h, log

    def test_inside_band_no_flips(self):
        h, _ = self._run(0.5 * 1.0, 3.0)          # amplitude < CORNER_COVER_MARGIN_S
        C = h.C
        mc = macro_changes(h.modes)
        flips = [r for r in mc if r[2] == "X2" and r[3] == "X3"]
        self.assertEqual(flips, [], "CORNERED left inside the margin band")
        corner = [r[0] for r in mc if r[3] == "X2" and r[5] == "R6.2"]
        self.assertSpacing(corner, C.CORNER_MIN_HOLD_S, "CORNERED entries")

    def test_outside_band_rate_bounded(self):
        h, log = self._run(2.5, 3.0)
        C = h.C
        mc = macro_changes(h.modes)
        corner = [r[0] for r in mc if r[3] == "X2" and r[5] == "R6.2"]
        self.assertTrue(corner)
        self.assertSpacing(corner, C.CORNER_MIN_HOLD_S, "CORNERED entries")
        etas = dict((round(t, 3), e) for t, e in log)
        for r in mc:
            if r[2] == "X2" and r[3] == "X3":          # left only when a cover ETA < t_close - margin
                self.assertLess(etas[round(r[0], 3)], 5.0 - C.CORNER_COVER_MARGIN_S)


# ====================================================================== randomized sequences
class TestRandomized(unittest.TestCase, LoopFreeMixin):
    def test_random_scenarios_no_short_cycles(self):
        import random
        rng = random.Random(7)
        for k in range(40):
            h = Harness(t=100.0 + rng.random() * 100, seed=k)
            h.world.add_mech(5, (rng.uniform(30, 250), rng.uniform(-50, 50), 0.0),
                             weapons=rng.choice([("gun", "laser"), ("gun", "lightning"), ("gun",)]))
            seq = [rng.choice(["", "win", "f", "e", "g", "d"]) for _ in range(12)]

            def fn(w, A, macro, target, ctx, seq=seq, t0=h.world.tick.t):
                s = seq[min(len(seq) - 1, int((w.tick.t - t0) / 5.0))]
                if s == "win":
                    return FightEval(target_id=5, win=True, win_sub="DUEL")
                if s:
                    return FightEval(target_id=5, lose_clauses=frozenset(s), lose_tracks={s: 5})
                return FightEval(target_id=5)
            h.assessor.fn = fn
            h.A_fn = lambda hh, r=rng.random(): build_A(hh.world, A_ratio=0.5 + r)
            h.run(N)
            mc = macro_changes(h.modes)
            # X2 <-> X3 flips need ENGAGE_MIN_HOLD_S / DISENGAGE_MIN_HOLD_S (H2) except CORNERED
            for a, b in zip(mc, mc[1:]):
                if a[3] == "X2" and b[3] == "X3" and a[5] != "R6.2":
                    self.assertGreaterEqual(b[0] - a[0], h.C.ENGAGE_MIN_HOLD_S - 1e-3)
                if a[3] == "X3" and b[3] == "X2" and b[5].startswith("T3"):
                    self.assertGreaterEqual(b[0] - a[0], h.C.DISENGAGE_MIN_HOLD_S - 1e-3)
            self.assertLessEqual(len(mc), 2 * N * 0.1 / min(h.C.ENGAGE_MIN_HOLD_S, h.C.DISENGAGE_CLEAR_S) + 2)


if __name__ == "__main__":
    unittest.main()
