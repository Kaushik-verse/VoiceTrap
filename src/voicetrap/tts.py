"""Text-to-speech with the voices built into macOS (`say`): free, offline and reproducible.

Generated clips are cached under ``data/tts`` and the benchmark clips are committed, so other
platforms can run the benchmark without generating audio.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from voicetrap.audio import load

CACHE = Path("data/tts")


def available() -> bool:
    return shutil.which("say") is not None


def speak(text: str, voice: str, rate: int = 175, cache_dir: Path = CACHE) -> np.ndarray:
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.wav"
    if not path.exists():
        if not available():
            raise RuntimeError("macOS `say` is not available; use the committed audio in data/clips")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.wav"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(out),
                            "--data-format=LEI16@16000", text], check=True)
            shutil.move(out, path)
    return load(path)
