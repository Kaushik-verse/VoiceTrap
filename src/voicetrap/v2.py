"""Multi-owner benchmark (v2): every owner and attacker voice comes from the unseen test split."""

from __future__ import annotations

from pathlib import Path

OWNERS = ["Aman (English (India))", "Moira (English (Ireland))", "Daniel (English (UK))"]
ATTACKERS = {
    "Aman (English (India))": ["Daniel (English (UK))", "Sandy (English (US))", "Anna", "Paulina"],
    "Moira (English (Ireland))": ["Tessa", "Sandy (English (UK))", "Milena", "Damayanti"],
    "Daniel (English (UK))": ["Aman (English (India))", "Tessa", "Ioana", "Joana"],
}
ROOT = Path("data/v2")


def build():
    from voicetrap.scenarios import build_v2

    return build_v2(ROOT, OWNERS, ATTACKERS)
