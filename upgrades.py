"""Upgrade offers (plan 7.14, R14.1). CONTRACTS 6.6.

Pick the first offered choice found, in order, in: GP's priority list for the CURRENTLY held loadout
(GL, GZ, ZL; LM = ZL list without the lens; everything else = the mech-only list), then
`UPGRADE_FALLBACK`, then upgrades for held weapons (gun, laser, lightning), then mortar upgrades, then
whatever is first. Always answers: the offer must never expire into the default choice 0.

An answered offer id is re-sent after OFFER_RESEND_TICKS decisions while it is still shown; a new offer id
(the queue's next offer) is answered on its first decision.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import config as _config


def _ids(choices) -> List[str]:
    return [c.get("id") if isinstance(c, dict) else c for c in (choices or [])]


class UpgradePicker:
    def __init__(self, C) -> None:
        self.C = C
        self._decisions = 0                 # pick() calls so far (one per decision)
        self._answered: dict = {}           # offer id -> decision index of the last send
        self.last_choice: Optional[str] = None

    def order_for(self, held_kinds) -> List[str]:
        """The full ordered preference list for a held loadout (GP list, fallback, weapon ups, mortar)."""
        key = _config.loadout_key(held_kinds)
        lists = _config.UPGRADE_PRIORITY
        if key in ("GL", "GZ", "ZL"):
            order = list(lists.get(key, []))
        elif key == "LM":
            order = [u for u in lists.get("ZL", []) if u != "laser_damage"]
        else:
            order = list(lists.get("MECH", []))
        order += list(getattr(self.C, "UPGRADE_FALLBACK", []) or [])
        return order

    def choose(self, choices, held_kinds) -> Tuple[Optional[int], str]:
        ids = _ids(choices)
        if not ids:
            return None, ""
        fb = list(getattr(self.C, "UPGRADE_FALLBACK", []) or [])
        order = self.order_for(held_kinds)
        gp = order[:len(order) - len(fb)]
        for uid in order:
            if uid in ids:
                return ids.index(uid), "R14.1" if uid in gp else "R14.1.fb"
        for kind in ("gun", "laser", "lightning"):
            if kind in held_kinds:
                for j, uid in enumerate(ids):
                    if isinstance(uid, str) and uid.startswith(kind + "_"):
                        return j, "R14.1.held"
        for j, uid in enumerate(ids):
            if isinstance(uid, str) and uid.startswith("mortar_"):
                return j, "R14.1.mortar"
        return 0, "R14.1.any"

    def pick(self, world) -> Tuple[Optional[int], str]:
        """(choice index, rule) for this decision, or (None, "") when nothing is to be sent."""
        self._decisions += 1
        offer = getattr(world, "offer", None)
        if offer is None or not offer.choices:
            # an old answered id never comes back once the queue moved on; keep the table small
            if len(self._answered) > 64:
                self._answered.clear()
            return None, ""
        last = self._answered.get(offer.id)
        resend = int(getattr(self.C, "OFFER_RESEND_TICKS", 2))
        if last is not None and self._decisions - last < resend:
            return None, ""
        held = tuple(world.held_kinds())
        idx, rule = self.choose(offer.choices, held)
        if idx is None:
            return None, ""
        self._answered[offer.id] = self._decisions
        ids = _ids(offer.choices)
        self.last_choice = ids[idx]
        return idx, rule
