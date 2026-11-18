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


GATE_DIR = Path("results/gate")


class Gates:
    """One speaker gate per (owner, scorer). Thresholds come only from training + validation speakers
    (5% owner-window rejection), never from the test voices used here."""

    def __init__(self, gate_dir: Path = GATE_DIR) -> None:
        import json

        import numpy as np

        from voicetrap.gatetrain import Scorer
        from voicetrap.speaker import Embedder

        self.embed = Embedder()
        m = json.loads((gate_dir / "metrics_v2.json").read_text())
        pos = []
        for split in ("train", "val"):  # cosine operating point on the same train+val pool
            d = np.load(gate_dir / "embeddings" / f"{split}.npz", allow_pickle=True)
            y = np.array([t["label"] for t in d["trials"]])
            pos.append(np.sum(d["E"] * d["P"], axis=1)[y == 1])
        self.thresholds = {"cosine": float(np.quantile(np.concatenate(pos), 0.05)),
                           "trained": float(m["operating_point"])}
        self.scorers = {"cosine": Scorer("cosine"), "trained": Scorer.load_v2(gate_dir / "gate_model_v2.joblib")}
        self.cache: dict = {}

    def __call__(self, clip, defence: str):
        from voicetrap import audio as A
        from voicetrap.scenarios import _tag
        from voicetrap.speaker import SpeakerGate

        kind = "cosine" if defence == "speaker" else "trained"
        key = (clip.owner_voice, kind)
        if key not in self.cache:
            enroll = [A.load(p) for p in sorted((ROOT / "enroll").glob(f"{_tag(clip.owner_voice)}-*.flac"))]
            self.cache[key] = SpeakerGate.enroll(self.embed, enroll, threshold=self.thresholds[kind],
                                                 scorer=self.scorers[kind])
        return self.cache[key]


def bench(defences=("none", "promptguard", "speaker", "speaker_trained", "shield"), out: Path = Path("results/v2")):
    from voicetrap.bench import run_bench

    needs_gate = any(d in ("speaker", "speaker_trained", "shield") for d in defences)
    return run_bench(out, data=ROOT, defences=list(defences), gate_for=Gates() if needs_gate else None)
