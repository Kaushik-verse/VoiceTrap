"""Voice corpus for training and testing the speaker gate.

Two sources:

* **Real people**: Mini LibriSpeech (openslr.org/31, CC BY 4.0): read audiobook speech, about 68 speakers.
* **Synthetic**: every usable macOS voice (about 140 distinct voices, many accents) reading the same
  sentence pool.

Speakers are split into train / val / test **by speaker group**, so test voices are never heard in
training. Locale variants of one macOS voice (e.g. "Eddy (English (UK))" and "Eddy (English (US))")
form one group, so they can't leak across the split.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from voicetrap import audio as A

NOVELTY = {"Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Good News", "Jester",
           "Organ", "Superstar", "Trinoids", "Whisper", "Zarvox", "Albert", "Fred", "Junior", "Ralph", "Kathy"}

SENTENCES = [
    "The weather today is bright with a light wind from the west.",
    "Please remember to buy milk, bread and a dozen eggs on the way home.",
    "Our train leaves the central station at a quarter past nine.",
    "She opened the window and listened to the rain on the roof.",
    "Technology changes quickly, but good habits last for years.",
    "I would like a cup of coffee with a little sugar, please.",
    "The children played football in the park until it got dark.",
    "Turn left at the second signal and the bank is on your right.",
    "A good book can keep you company on a long journey.",
    "The meeting has been moved to Thursday afternoon at three.",
    "Fresh mangoes are sold at the market every summer morning.",
    "He fixed the old radio with a screwdriver and some patience.",
]


@dataclass
class Speaker:
    id: str            # unique id, e.g. "tts:Eddy (English (UK))" or "real:1272"
    group: str         # split unit, e.g. "tts:Eddy" or "real:1272"
    source: str        # tts | real
    lang: str          # locale for tts, "en" for real
    gender: str = "?"
    files: list[str] = field(default_factory=list)
    split: str = ""


def macos_voices() -> list[tuple[str, str]]:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    seen, voices = set(), []
    for line in out.splitlines():
        m = re.match(r"^(.*?)\s+([a-z]{2,3}_[A-Z0-9]{2,3})\s+#", line)
        if not m:
            continue
        name, loc = m.group(1).strip(), m.group(2)
        if name in seen or name.split(" (")[0] in NOVELTY:
            continue
        seen.add(name)
        voices.append((name, loc))
    return voices


def build_tts(root: Path, voices=None, sentences=SENTENCES, workers: int = 8) -> list[Speaker]:
    """Synthesize every voice x sentence in parallel (resumable: existing clips are skipped)."""
    from concurrent.futures import ThreadPoolExecutor

    from voicetrap import tts

    voices = voices or macos_voices()
    bad = root / "tts" / "_failed.txt"
    failed = set(bad.read_text().splitlines()) if bad.exists() else set()

    def job(args):
        name, i, s = args
        path = root / "tts" / f"{_slug(name)}-{i}.flac"
        if path.exists() or str(path) in failed:
            return
        try:
            x = tts.speak(s, name, cache_dir=root / "tts_cache")
        except Exception:
            x = np.zeros(0, dtype="float32")
        if len(x) < A.SR * 1.0 or A.rms(x) < 1e-3:  # voice can't read English: remember and skip
            with open(bad, "a") as f:
                f.write(f"{path}\n")
            return
        A.save(path, A.normalize(x))

    (root / "tts").mkdir(parents=True, exist_ok=True)
    jobs = [(n, i, s) for n, _ in voices for i, s in enumerate(sentences)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, jobs))
    speakers = []
    for name, loc in voices:
        sp = Speaker(f"tts:{name}", f"tts:{name.split(' (')[0]}", "tts", loc)
        sp.files = [str(p) for i in range(len(sentences))
                    if (p := root / "tts" / f"{_slug(name)}-{i}.flac").exists()]
        if len(sp.files) >= 6:
            speakers.append(sp)
    return speakers


def build_real(libri: Path, per_speaker: int = 12, max_s: float = 8.0, seed: int = 0) -> list[Speaker]:
    genders = {}
    for txt in libri.rglob("SPEAKERS.TXT"):
        for line in txt.read_text(errors="ignore").splitlines():
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                genders[parts[0]] = parts[1]
    rng = random.Random(seed)
    speakers = []
    for spk_dir in sorted(p for p in libri.glob("LibriSpeech/*/*") if p.is_dir()):
        files = sorted(spk_dir.rglob("*.flac"))
        if len(files) < 6:
            continue
        rng.shuffle(files)
        sid = spk_dir.name
        speakers.append(Speaker(f"real:{sid}", f"real:{sid}", "real", "en", genders.get(sid, "?"),
                                [str(f) for f in files[:per_speaker]]))
    return speakers


def split(speakers: list[Speaker], val: float = 0.15, test: float = 0.3, seed: int = 7) -> None:
    """Assign splits by group, stratified by source, so test speakers are unseen in training."""
    rng = random.Random(seed)
    for source in ("tts", "real"):
        groups = sorted({s.group for s in speakers if s.source == source})
        rng.shuffle(groups)
        n_test, n_val = round(len(groups) * test), round(len(groups) * val)
        which = {g: "test" for g in groups[:n_test]}
        which |= {g: "val" for g in groups[n_test : n_test + n_val]}
        for s in speakers:
            if s.source == source:
                s.split = which.get(s.group, "train")


def build(root: Path = Path("data/corpus"), libri: Path = Path("data/librispeech")) -> list[Speaker]:
    root.mkdir(parents=True, exist_ok=True)
    speakers = build_tts(root) + (build_real(libri) if libri.exists() else [])
    split(speakers)
    (root / "speakers.json").write_text(json.dumps([asdict(s) for s in speakers], indent=1))
    return speakers


def load(root: Path = Path("data/corpus")) -> list[Speaker]:
    return [Speaker(**d) for d in json.loads((root / "speakers.json").read_text())]


def load_utt(path: str, max_s: float = 8.0) -> np.ndarray:
    x = A.load(path)
    return x[: int(max_s * A.SR)]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
