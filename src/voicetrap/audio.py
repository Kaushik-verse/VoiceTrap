"""Small audio toolkit: 16 kHz mono float32 arrays, mixing at a target level, procedural backgrounds.

Everything is deterministic (seeded) so the benchmark audio can be regenerated bit-for-bit.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000


def load(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = resample(x, sr, SR)
    return x


def save(path: str | Path, x: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), SR, subtype="PCM_16")
    return path


def to_wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def active_rms(x: np.ndarray, frame: int = 400) -> float:
    """RMS over the louder half of 25 ms frames, so leading/trailing silence doesn't skew levels."""
    n = len(x) // frame
    if n == 0:
        return rms(x)
    e = np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame), dtype=np.float64), axis=1))
    top = np.sort(e)[n // 2 :]
    return float(np.sqrt(np.mean(np.square(top))) + 1e-12)


def gain_for(x: np.ndarray, ref: np.ndarray, rel_db: float) -> float:
    """Gain that puts ``x`` at ``rel_db`` decibels relative to ``ref`` (speech-active RMS)."""
    return (active_rms(ref) / active_rms(x)) * 10 ** (rel_db / 20)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype="float32")


def overlay(base: np.ndarray, x: np.ndarray, at_s: float, gain: float = 1.0) -> np.ndarray:
    start = int(at_s * SR)
    out = np.zeros(max(len(base), start + len(x)), dtype="float32")
    out[: len(base)] += base
    out[start : start + len(x)] += x * gain
    return out


def music_bed(seconds: float, seed: int = 0) -> np.ndarray:
    """A soft procedural chord loop (royalty-free stand-in for an ad jingle or a song)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    roots = rng.choice([220.0, 246.9, 261.6, 293.7, 329.6], size=4)
    out = np.zeros_like(t)
    seg = len(t) / 4
    for i, f in enumerate(roots):
        a, b = int(i * seg), int((i + 1) * seg)
        tt = t[a:b]
        env = np.minimum(1, (tt - tt[0]) * 8) * np.exp(-(tt - tt[0]) * 0.9)
        for k, m in enumerate((1.0, 1.25, 1.5)):  # major triad
            out[a:b] += env * np.sin(2 * np.pi * f * m * tt) / (k + 1)
        beat = (np.sin(2 * np.pi * 2.0 * tt) > 0.95).astype(float) * rng.normal(0, 0.4, len(tt))
        out[a:b] += 0.3 * beat
    return (out / (np.abs(out).max() + 1e-9) * 0.3).astype("float32")


def room_noise(seconds: float, seed: int = 0) -> np.ndarray:
    """Pink-ish background noise (fan / street hum)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, int(SR * seconds))
    f = np.fft.rfft(w)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    x = np.fft.irfft(f, n=len(w))
    return (x / (np.abs(x).max() + 1e-9) * 0.05).astype("float32")


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.abs(x).max())
    return x if m < 1e-9 else (x / m * peak).astype("float32")
