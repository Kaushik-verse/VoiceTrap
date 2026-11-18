"""Real-owner benchmark: owners are real people from SLURP, attackers are F5-TTS voices.

Inputs come from ``colab/slurp_attacks.ipynb`` (unzipped to ``data/slurp_bench``; SLURP audio is CC BY-NC 4.0, so it is
git-ignored and never redistributed). The speaker gate is calibrated on SLURP speakers that are neither owners nor
attackers, and uses the fine-tuned ECAPA from ``colab/real_voice_gate.ipynb`` only if that model beat the pretrained
one on its validation speakers.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from voicetrap import audio as A
from voicetrap.scenarios import PAYLOADS, STYLES, TIMINGS, Clip, Intent, clip_path, render

ROOT = Path("data/slurp_bench")
OUT = Path("results/slurp")
GATE_FT = Path("data/gate_ft")  # ecapa_finetuned_best.pt + finetune_history.json from Colab
DEFENCES = ["none", "promptguard", "speaker", "shield"]
LEVELS = (-10, 0)


@dataclass
class RealClip(Clip):
    tool: str | None = None              # the tool the owner's request needs
    owner_group: str | None = None       # FE / FO / ME / MO: gender + native (E) or non-native (O) English
    attacker_group: str | None = None

    @property
    def intent(self) -> Intent | None:
        return Intent(self.user_text, self.tool, {}) if self.tool else None


def load(root: Path = ROOT) -> dict:
    return json.loads((root / "manifest.json").read_text())


def tv(x: np.ndarray, seed: int) -> np.ndarray:
    """The hidden command plays from a TV speaker across the room: band-limited, with room echo."""
    from scipy.signal import butter, fftconvolve, sosfilt

    rng = np.random.default_rng(seed)
    y = sosfilt(butter(4, [150, 7000], btype="band", fs=A.SR, output="sos"), x)
    rt60 = rng.uniform(0.3, 0.6)
    t = np.arange(int(rt60 * A.SR)) / A.SR
    h = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    h[0] = 4.0
    y = fftconvolve(y, h)[: len(x)]
    return (y * A.rms(x) / max(A.rms(y), 1e-9)).astype("float32")


def clips(m: dict) -> tuple[list[RealClip], int]:
    """23 clips per owner. Returns the clips and how many were skipped for an unverified attack line."""
    ok = {(a["voice"], a["line"]) for a in m["attacks"] if a["verified"]}
    groups = {d["id"]: d["id"].split("-")[0] for d in m["owners"] + m["donors"]}
    donors = [d["id"] for d in m["donors"]]
    pay, sty = list(PAYLOADS), list(STYLES)
    out, skipped = [], 0

    def add(c: RealClip):
        nonlocal skipped
        if c.kind != "clean" and (c.attack_voice, f"{c.payload}-{c.style}") not in ok:
            skipped += 1
        else:
            out.append(c)

    for oi, o in enumerate(m["owners"]):
        own, g = o["id"], groups[o["id"]]
        for r, req in enumerate(o["requests"]):
            add(RealClip(f"{own}-clean-r{r}", "clean", r, None, None, None, None, None, req["text"], "", own,
                         req["tool"], g))
            for pi, p in enumerate(pay):
                s = sty[(oi + r + pi) % 3]
                d = donors[(oi * 5 + r * 3 + pi) % len(donors)]
                for li, lvl in enumerate(LEVELS):
                    tm = TIMINGS[(r + pi + li) % 2]
                    add(RealClip(f"{own}-attack-r{r}-{p}-{s}-{lvl:+d}db-{tm}", "attack", r, p, s, d, lvl, tm,
                                 req["text"], STYLES[s].format(p=PAYLOADS[p].text), own, req["tool"], g, groups[d]))
        req = o["requests"][0]
        for pi, p in enumerate(pay):
            s = sty[(oi + pi) % 3]
            for lvl in LEVELS:
                add(RealClip(f"{own}-clone-r0-{p}-{s}-{lvl:+d}db-gap", "clone", 0, p, s, f"clone:{own}", lvl, "gap",
                             req["text"], STYLES[s].format(p=PAYLOADS[p].text), own, req["tool"], g, g))
        for pi, p in enumerate(pay):
            s = sty[(oi + pi + 1) % 3]
            d = donors[(oi * 5 + pi + 7) % len(donors)]
            add(RealClip(f"{own}-background-{p}-{s}", "background", None, p, s, d, None, None, "",
                         STYLES[s].format(p=PAYLOADS[p].text), own, None, g, groups[d]))
    return out, skipped


def build(root: Path = ROOT) -> list[RealClip]:
    m = load(root)
    line_of = {text: lid for lid, text in m["lines"].items()}
    files = {(a["voice"], a["line"]): a["file"] for a in m["attacks"]}
    requests = {(o["id"], r["text"]): r["file"] for o in m["owners"] for r in o["requests"]}

    def speak(text: str, voice: str) -> np.ndarray:
        if (voice, text) in requests:
            return A.load(root / requests[(voice, text)])
        lid = line_of[text]
        return tv(A.load(root / files[(voice, lid)]), zlib.crc32(f"{voice}|{lid}".encode()))

    cs, skipped = clips(m)
    for c in cs:
        p = clip_path(root, c)
        if not p.exists():
            A.save(p, render(c, speak))
    (root / "clips.json").write_text(json.dumps([asdict(c) for c in cs], indent=1))
    print(f"{len(cs)} clips for {len(m['owners'])} owners ({skipped} skipped: attack line failed the Whisper check)")
    return cs


def load_clips(root: Path = ROOT) -> list[RealClip]:
    return [RealClip(**d) for d in json.loads((root / "clips.json").read_text())]


def gate_weights() -> str | None:
    """The fine-tuned ECAPA, but only if it beat the pretrained model on its validation speakers."""
    hist, w = GATE_FT / "finetune_history.json", GATE_FT / "ecapa_finetuned_best.pt"
    if not (hist.exists() and w.exists()):
        return None
    h = json.loads(hist.read_text())
    pre = next(e["val_eer"] for e in h if e["epoch"] == -1)
    return str(w) if min(e["val_eer"] for e in h if e["epoch"] >= 0) < pre else None


def calibrate(embed, m: dict, root: Path = ROOT, frr: float = 0.05) -> dict:
    """Threshold = 5% owner-window rejection on calibration speakers (never owners or attackers)."""
    from voicetrap.bench import windows

    profs, wins = [], []
    for c in m["calibration"]:
        audio = [A.load(root / f) for f in c["files"]]
        p = embed(audio[:5]).mean(0)
        profs.append(p / np.linalg.norm(p))
        wins.append(embed([w for x in audio[5:] for w in windows(x)]))
    P = np.stack(profs)
    pos = np.concatenate([wins[i] @ P[i] for i in range(len(P))])
    neg = np.concatenate([wins[j] @ P[i] for i in range(len(P)) for j in range(len(P)) if i != j])
    thr = float(np.quantile(pos, frr))
    return {"threshold": round(thr, 4), "owner_accept": round(float((pos >= thr).mean()), 3),
            "other_accept": round(float((neg >= thr).mean()), 3), "owner_windows": int(len(pos)),
            "other_windows": int(len(neg)), "speakers": len(P)}


class Gates:
    def __init__(self, m: dict, root: Path = ROOT, out: Path = OUT) -> None:
        from voicetrap.speaker import Embedder

        self.weights = gate_weights()
        self.embed = Embedder(weights=self.weights)
        model = "ecapa_finetuned" if self.weights else "ecapa_pretrained"
        cal_path = out / "calibration.json"
        cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        if cal.get("model") != model:
            cal = {"model": model, **calibrate(self.embed, m, root)}
            out.mkdir(parents=True, exist_ok=True)
            cal_path.write_text(json.dumps(cal, indent=1))
        print(f"speaker gate: {model}, threshold {cal['threshold']} (calibration speakers: owner windows kept "
              f"{cal['owner_accept']}, other voices kept {cal['other_accept']})")
        self.threshold, self.root, self.cache = cal["threshold"], root, {}
        self.enroll = {o["id"]: o["enroll"] for o in m["owners"]}

    def __call__(self, clip, defence: str):
        from voicetrap.speaker import SpeakerGate

        if clip.owner_voice not in self.cache:
            audio = [A.load(self.root / f) for f in self.enroll[clip.owner_voice]]
            self.cache[clip.owner_voice] = SpeakerGate.enroll(self.embed, audio, threshold=self.threshold)
        return self.cache[clip.owner_voice]


def bench(defences=DEFENCES, root: Path = ROOT, out: Path = OUT, groq=None):
    from voicetrap.bench import run_bench

    m = load(root)
    cs = load_clips(root) if (root / "clips.json").exists() else build(root)
    return run_bench(out, data=root, defences=list(defences), gate_for=Gates(m, root, out), clips=cs, groq=groq)
