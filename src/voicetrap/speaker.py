"""Speaker gate: keep only the parts of the audio spoken by the enrolled owner.

A pretrained ECAPA-TDNN speaker-embedding model (SpeechBrain, trained on VoxCeleb; no training
here) embeds sliding windows of the input. Windows whose embedding is close to the owner's
enrolment profile are kept; everything else is muted before transcription.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voicetrap import audio as A

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


class Embedder:
    def __init__(self, savedir: str = "pretrained_models/ecapa", weights: str | None = None) -> None:
        """``weights``: a fine-tuned ECAPA state dict (from colab/real_voice_gate.ipynb) to load over the pretrained one."""
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.model = EncoderClassifier.from_hparams(source=MODEL_ID, savedir=savedir,
                                                    run_opts={"device": "cpu"})
        if weights:
            self.model.mods.embedding_model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.mods.eval()

    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        n = max(len(x) for x in batch)
        wav = np.zeros((len(batch), n), dtype="float32")
        lens = np.zeros(len(batch), dtype="float32")
        for i, x in enumerate(batch):
            wav[i, : len(x)] = x
            lens[i] = len(x) / n
        with self.torch.no_grad():
            e = self.model.encode_batch(self.torch.from_numpy(wav), self.torch.from_numpy(lens))
        e = e.squeeze(1).numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)


@dataclass
class GateResult:
    audio: np.ndarray
    kept_fraction: float
    windows: list = field(default_factory=list)  # (start_s, end_s, similarity, kept)


def cosine(E: np.ndarray, p: np.ndarray) -> np.ndarray:
    return E @ p


class SpeakerGate:
    """``scorer(window_embeddings, profile) -> scores``: plain cosine, or the trained model
    (see ``gatetrain.Scorer``). ``threshold`` applies to whichever score the scorer returns."""

    def __init__(self, embed, profile: np.ndarray, threshold: float = 0.45, win_s: float = 1.2,
                 hop_s: float = 0.3, pad_s: float = 0.15, scorer=cosine) -> None:
        self.embed, self.profile, self.threshold, self.scorer = embed, profile, threshold, scorer
        self.win, self.hop, self.pad = int(win_s * A.SR), int(hop_s * A.SR), int(pad_s * A.SR)

    @classmethod
    def enroll(cls, embed, clips: list[np.ndarray], **kw) -> SpeakerGate:
        p = embed(clips).mean(axis=0)
        return cls(embed, p / np.linalg.norm(p), **kw)

    def similarity(self, x: np.ndarray) -> float:
        return float(self.scorer(self.embed([x]), self.profile)[0])

    def __call__(self, x: np.ndarray) -> GateResult:
        starts = list(range(0, max(1, len(x) - self.win + 1), self.hop))
        frames = [x[s : s + self.win] for s in starts]
        loud = A.active_rms(x)
        speech = [A.rms(f) > loud * 0.1 for f in frames]  # skip near-silent windows
        sims = np.full(len(frames), -1.0)
        idx = [i for i, sp in enumerate(speech) if sp]
        if idx:
            sims[idx] = self.scorer(self.embed([frames[i] for i in idx]), self.profile)
        mask = np.zeros(len(x), dtype=bool)
        windows = []
        for s, sim, sp in zip(starts, sims, speech, strict=True):
            keep = bool(sp and sim >= self.threshold)
            windows.append((round(s / A.SR, 2), round((s + self.win) / A.SR, 2), round(float(sim), 3), keep))
            if keep:
                mask[max(0, s - self.pad) : s + self.win + self.pad] = True
        return GateResult((x * mask).astype("float32"), float(mask.mean()), windows)
