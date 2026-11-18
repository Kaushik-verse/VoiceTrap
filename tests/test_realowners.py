"""Offline tests for the real-owner (SLURP) benchmark plan."""

from collections import Counter

from voicetrap.judge import matches
from voicetrap.realowners import RealClip, clips
from voicetrap.scenarios import PAYLOADS, STYLES


def manifest(verified=True):
    lines = {f"{p}-{s}": STYLES[s].format(p=PAYLOADS[p].text) for p in PAYLOADS for s in STYLES}
    owners = [{"id": f"{g}-{i}", "requests": [{"file": "a", "text": "play some jazz", "tool": "play_music"},
                                              {"file": "b", "text": "wake me at seven", "tool": "set_alarm"}],
               "enroll": []} for i, g in enumerate(["FE", "MO"])]
    donors = [{"id": f"{g}-9{i}"} for i, g in enumerate(["FO", "ME", "FE"])]
    voices = [d["id"] for d in donors] + [f"clone:{o['id']}" for o in owners]
    attacks = [{"voice": v, "line": lid, "file": "x", "verified": verified or v.startswith("clone")}
               for v in voices for lid in lines]
    return {"owners": owners, "donors": donors, "lines": lines, "attacks": attacks}


def test_plan_has_23_clips_per_owner_with_unique_ids():
    cs, skipped = clips(manifest())
    assert skipped == 0 and len(cs) == 46 and len({c.id for c in cs}) == 46
    per = Counter((c.owner_voice, c.kind) for c in cs)
    assert per[("FE-0", "clean")] == 2 and per[("FE-0", "attack")] == 12
    assert per[("FE-0", "clone")] == 6 and per[("FE-0", "background")] == 3


def test_clone_uses_owner_voice_and_attackers_never_do():
    cs, _ = clips(manifest())
    for c in cs:
        if c.kind == "clone":
            assert c.attack_voice == f"clone:{c.owner_voice}" and c.attacker_group == c.owner_group
        elif c.kind in ("attack", "background"):
            assert c.attack_voice != c.owner_voice and not c.attack_voice.startswith("clone")


def test_unverified_attack_lines_are_skipped():
    cs, skipped = clips(manifest(verified=False))
    assert skipped == 2 * (12 + 3) and {c.kind for c in cs} == {"clean", "clone"}


def test_real_clip_intent_matches_the_requested_tool():
    c = RealClip("x", "clean", 0, None, None, None, None, None, "play some jazz", "", "FE-0", "play_music")
    assert matches({"name": "play_music", "arguments": {"query": "jazz"}}, c.intent)
    assert not matches({"name": "send_money", "arguments": {}}, c.intent) and c.attack is None
