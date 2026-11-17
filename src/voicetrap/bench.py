"""Run every clip through every defence, score it, and save results (resumable, cached)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from voicetrap import audio as A
from voicetrap.judge import matches, score
from voicetrap.pipeline import DEFENCES, run
from voicetrap.scenarios import ENROLLMENT, USER_VOICE, Clip, clip_path, load_manifest

# Voices used ONLY to calibrate the speaker-gate threshold; none of them appears as an attacker.
CALIBRATION_VOICES = ["Karen", "Moira", "Tessa", "Eddy (English (US))", "Reed (English (US))"]


def windows(x: np.ndarray, win_s: float = 1.2, hop_s: float = 0.6) -> list[np.ndarray]:
    w, h = int(win_s * A.SR), int(hop_s * A.SR)
    loud = A.active_rms(x)
    return [x[s : s + w] for s in range(0, max(1, len(x) - w + 1), h) if A.rms(x[s : s + w]) > loud * 0.1]


def calibrate(embed, data: Path, speak=None) -> dict:
    """Pick the threshold that best separates owner windows from unseen voices (Youden's J)."""
    from voicetrap import tts

    speak = speak or (lambda t, v: tts.speak(t, v, cache_dir=data / "tts"))
    enroll = [A.load(p) for p in sorted((data / "enroll").glob("*.flac"))]
    pos = []
    for i in range(len(enroll)):  # leave-one-out: profile from the other enrolment clips
        prof = embed([e for j, e in enumerate(enroll) if j != i]).mean(0)
        prof /= np.linalg.norm(prof)
        pos += list(embed(windows(enroll[i])) @ prof)
    prof = embed(enroll).mean(0)
    prof /= np.linalg.norm(prof)
    neg = []
    for v in CALIBRATION_VOICES:
        for t in ENROLLMENT:
            neg += list(embed(windows(speak(t, v))) @ prof)
    pos, neg = np.array(pos), np.array(neg)
    grid = np.linspace(-0.2, 1, 241)
    j = np.array([(pos >= t).mean() - (neg >= t).mean() for t in grid])
    best = grid[j >= j.max() - 1e-9]  # every threshold with the best separation...
    thr = float((best.min() + best.max()) / 2)  # ...take the middle of that range, not its edge
    return {"threshold": round(thr, 3), "owner_windows": len(pos), "other_windows": len(neg),
            "owner_accept": round(float((pos >= thr).mean()), 3),
            "other_accept": round(float((neg >= thr).mean()), 3),
            "owner_mean": round(float(pos.mean()), 3), "other_mean": round(float(neg.mean()), 3)}


def make_gate(data: Path, out: Path, log=print):
    from voicetrap.speaker import Embedder, SpeakerGate

    embed = Embedder()
    cal_path = out / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
    else:
        cal = calibrate(embed, data)
        cal_path.write_text(json.dumps(cal, indent=1))
    log(f"speaker gate threshold {cal['threshold']} (owner accept {cal['owner_accept']}, "
        f"unseen voices accept {cal['other_accept']})")
    enroll = [A.load(p) for p in sorted((data / "enroll").glob(f"{USER_VOICE}-*.flac"))]
    return SpeakerGate.enroll(embed, enroll, threshold=cal["threshold"])


def run_bench(out: Path, data: Path = Path("data"), defences=None, kinds=None, groq=None, gate=None,
              log=print) -> list[dict]:
    from voicetrap.groq import Groq

    out.mkdir(parents=True, exist_ok=True)
    defences = defences or DEFENCES
    clips = [c for c in load_manifest(data) if not kinds or c.kind in kinds]
    groq = groq or Groq(cache_dir=out / "cache")
    if gate is None and any(d in ("speaker", "shield") for d in defences):
        gate = make_gate(data, out, log)
    res_file = out / "results.jsonl"
    done = set()
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["clip"], r["defence"]))
    rows = []
    for c in clips:
        x = A.load(clip_path(data, c))
        for d in defences:
            if (c.id, d) in done:
                continue
            t0 = time.perf_counter()
            o = run(x, d, groq, gate=gate, approve=partial(owner_approves, c))
            s = score(c, o)
            row = {"clip": c.id, "defence": d, **{k: v for k, v in asdict(c).items() if k != "id"},
                   **s, "transcript": o.transcript, "tool_calls": o.tool_calls,
                   "executed": [x_["name"] for x_ in o.executed], "blocked": [x_["name"] for x_ in o.blocked],
                   "guard_score": o.guard_score, "kept_fraction": o.kept_fraction, "note": o.note,
                   "seconds": round(time.perf_counter() - t0, 2)}
            with open(res_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            flag = "HIJACKED" if s["hijacked"] else ("ok" if s["utility"] in (True, None) else "missed")
            log(f"{d:12s} {c.id:44s} {flag:8s} {o.transcript[:70]!r}")
    log(f"groq calls {groq.calls}, cache hits {groq.cache_hits}, tokens {groq.tokens}")
    return rows


def owner_approves(clip: Clip, call: dict) -> bool:
    """Simulated owner: taps 'approve' only for the action they actually asked for."""
    return clip.intent is not None and matches(call, clip.intent)


def load_results(out: Path) -> list[dict]:
    rows = {}
    for line in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["clip"], r["defence"])] = r
    return list(rows.values())


__all__ = ["run_bench", "load_results", "calibrate", "Clip"]
