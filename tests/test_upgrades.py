"""Upgrade picker tests (R14.1): offers built from the recorded hello's upgrade list.

    cd bot && python3 -m unittest tests.test_upgrades -v
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import unittest
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

import config  # noqa: E402
import safe  # noqa: E402
import upgrades  # noqa: E402
from contracts import OfferState  # noqa: E402

STREAM = os.path.join(os.path.dirname(REPO), "mechfight-kit", "check", "stream.jsonl.gz")
HAVE_KIT = os.path.exists(STREAM)


@unittest.skipUnless(HAVE_KIT, "recorded hello not available")
class TestPicker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(STREAM, "rt") as f:
            cls.cfg = json.loads(f.readline())["config"]
        cls.C = config.bind(cls.cfg)
        cls.defs = {u["id"]: u for u in cls.cfg["upgrades"]}

    def offer(self, *ids, oid=1):
        for i in ids:
            self.assertIn(i, self.defs)
        return OfferState(id=oid, choices=list(ids), expires=99.0)

    def world(self, kinds, offer):
        return SimpleNamespace(offer=offer, held_kinds=lambda: tuple(kinds))

    def pick(self, kinds, *ids):
        p = upgrades.UpgradePicker(self.C)
        idx, rule = p.pick(self.world(kinds, self.offer(*ids)))
        return ids[idx], rule

    def test_gp_lists_per_loadout(self):
        self.assertEqual(self.pick(["gun", "lightning"], "armor", "plating", "gun_damage")[0], "plating")
        self.assertEqual(self.pick(["gun", "lightning"], "armor", "servos", "gun_mag")[0], "servos")
        self.assertEqual(self.pick(["gun", "lightning"], "armor", "gun_damage", "lightning_efficiency")[0],
                         "lightning_efficiency")
        self.assertEqual(self.pick(["gun", "laser"], "gun_rate", "sensors", "armor")[0], "sensors")
        self.assertEqual(self.pick(["gun", "laser"], "armor", "laser_damage", "gun_mag")[0], "laser_damage")
        self.assertEqual(self.pick(["laser", "lightning"], "servos", "capacitor", "armor")[0], "servos")
        self.assertEqual(self.pick(["laser", "lightning"], "capacitor", "armor", "laser_damage")[0], "laser_damage")

    def test_lm_uses_zl_without_lens(self):
        self.assertEqual(self.pick(["lightning", "mortar"], "laser_damage", "capacitor", "armor")[0], "capacitor")
        self.assertEqual(self.pick(["lightning", "mortar"], "laser_damage", "armor", "gyros")[0], "armor")

    def test_mech_only_list_for_other_loadouts(self):
        self.assertEqual(self.pick(["gun"], "gun_rate", "nanites", "sensors")[0], "sensors")
        self.assertEqual(self.pick(["mortar", "laser"], "nanites", "armor", "servos")[0], "armor")
        self.assertEqual(self.pick(["gun", "mortar"], "servos", "nanites", "gyros")[0], "servos")

    def test_fallback_list_then_held_weapon_then_mortar(self):
        # nothing in the GP list; plating-first fallback: capacitor beats reactor beats gyros
        self.assertEqual(self.pick(["gun"], "gyros", "reactor", "capacitor")[0], "capacitor")
        self.assertEqual(self.pick(["gun"], "gun_rate", "gun_velocity", "mortar_splash"), ("gun_rate", "R14.1.held"))
        self.assertEqual(self.pick(["mortar"], "mortar_splash", "mortar_reload", "mortar_velocity")[1],
                         "R14.1.mortar")
        self.assertEqual(self.pick(["gun", "mortar"], "mortar_reload", "gun_mag", "mortar_splash")[0], "gun_mag")
        # a laser upgrade is never taken without the laser; with only unknown ids we still answer
        p = upgrades.UpgradePicker(self.C)
        idx, rule = p.pick(self.world(["gun"], self.offer("laser_charge", "laser_pierce", "lightning_chain")))
        self.assertIsNotNone(idx)

    def test_never_defaults_silently_to_choice_zero(self):
        # best choice at index 2: we answer with 2, so the offer cannot expire into choice 0
        ids = ("mortar_splash", "gyros", "plating")
        p = upgrades.UpgradePicker(self.C)
        idx, _ = p.pick(self.world(["gun", "laser"], self.offer(*ids)))
        self.assertEqual(idx, 2)

    def test_no_offer(self):
        p = upgrades.UpgradePicker(self.C)
        self.assertEqual(p.pick(self.world(["gun"], None)), (None, ""))

    def test_agrees_with_safe_controller_on_main_loadouts(self):
        cases = [(["gun", "lightning"], ["armor", "servos", "gun_mag"]),
                 (["gun", "laser"], ["gun_rate", "sensors", "armor"]),
                 (["laser", "lightning"], ["capacitor", "armor", "laser_damage"]),
                 (["gun"], ["gyros", "nanites", "sensors"])]
        for kinds, ids in cases:
            idx, _ = upgrades.UpgradePicker(self.C).pick(self.world(kinds, self.offer(*ids)))
            offer = {"choices": [{"id": i, "name": i, "desc": ""} for i in ids]}
            self.assertEqual(idx, safe.pick_upgrade(offer, kinds, self.C), (kinds, ids))

    def test_dict_choices_as_in_the_protocol(self):
        o = OfferState(id=3, choices=[{"id": "gun_rate"}, {"id": "armor"}, {"id": "sensors"}], expires=9.0)
        p = upgrades.UpgradePicker(self.C)
        idx, _ = p.pick(self.world(["gun"], o))
        self.assertEqual(idx, 1)          # MECH list: armor before sensors

    def test_resend_policy(self):
        p = upgrades.UpgradePicker(self.C)
        w = self.world(["gun", "lightning"], self.offer("armor", "plating", "gun_damage"))
        n = self.C.OFFER_RESEND_TICKS
        results = [p.pick(w)[0] for _ in range(2 * n + 1)]
        self.assertEqual(results[0], 1)
        self.assertTrue(all(r is None for r in results[1:n]))
        self.assertEqual(results[n], 1)           # resent after OFFER_RESEND_TICKS decisions

    def test_offer_queue_two_offers(self):
        p = upgrades.UpgradePicker(self.C)
        kinds = ["gun", "lightning"]
        o1 = self.offer("armor", "plating", "gun_damage", oid=1)
        o2 = self.offer("gun_mag", "armor", "servos", oid=2)
        self.assertEqual(p.pick(self.world(kinds, o1))[0], 1)
        self.assertIsNone(p.pick(self.world(kinds, o1))[0])          # still shown, just answered
        # the queued offer replaces it: answered on its first decision, with the held-loadout list
        idx, _ = p.pick(self.world(kinds, o2))
        self.assertEqual(idx, 2)
        self.assertIsNone(p.pick(self.world(kinds, None))[0])
        # the same id shown again later (expired resend window) is answered again
        for _ in range(self.C.OFFER_RESEND_TICKS):
            p.pick(self.world(kinds, o2))
        self.assertEqual(p.pick(self.world(kinds, o2))[0], 2)

    def test_loadout_follows_current_hold(self):
        p = upgrades.UpgradePicker(self.C)
        ids = ("servos", "sensors", "lightning_efficiency")
        self.assertEqual(ids[p.pick(self.world(["gun", "laser"], self.offer(*ids, oid=1)))[0]], "sensors")
        self.assertEqual(ids[p.pick(self.world(["gun", "lightning"], self.offer(*ids, oid=2)))[0]],
                         "lightning_efficiency")


if __name__ == "__main__":
    unittest.main()
