"""Text-to-speech with the voices built into macOS (`say`): free, offline and reproducible.

Generated clips are cached under ``data/tts`` and the benchmark clips are committed, so other
platforms can run the benchmark without generating audio.

Every clip is checked after synthesis: under heavy parallel load `say` occasionally returns a
truncated file, so a clip whose speaking rate is implausibly fast is regenerated (and rejected if it
stays truncated).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from voicetrap.audio import SR, load

CACHE = Path("data/tts")
MAX_WORDS_PER_S = 5.2  # natural speech at -r 175 is ~2.5-4.8 words/s (verified with Whisper); faster = cut off


class TruncatedSpeech(RuntimeError):
    pass


def available() -> bool:
    return shutil.which("say") is not None


def plausible(text: str, seconds: float) -> bool:
    words = max(1, len(text.split()))
    return seconds >= 0.4 and words / max(seconds, 1e-3) <= MAX_WORDS_PER_S


def cache_path(text: str, voice: str, rate: int = 175, cache_dir: Path = CACHE) -> Path:
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    return cache_dir / f"{key}.wav"


def speak(text: str, voice: str, rate: int = 175, cache_dir: Path = CACHE, retries: int = 4) -> np.ndarray:
    path = cache_path(text, voice, rate, cache_dir)
    if path.exists():
        x = load(path)
        if plausible(text, len(x) / SR):
            return x
        path.unlink()  # a truncated clip from an earlier run: regenerate it
    if not available():
        raise RuntimeError("macOS `say` is not available; use the committed audio in data/clips")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.wav"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(out),
                            "--data-format=LEI16@16000", text], check=True)
            x = load(out)
            if plausible(text, len(x) / SR):
                shutil.move(out, path)
                return x
    raise TruncatedSpeech(f"{voice!r} kept producing truncated audio for {text!r}")
