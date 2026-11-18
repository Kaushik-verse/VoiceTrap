"""Train and test the speaker gate's scoring model on our own generated trials.

A *trial* = (owner's enrolment profile, one 1.2 s window of audio). Label 1 if the window is the owner
speaking (possibly with music, noise or a quieter voice behind them), 0 if it is someone else
(clean, inside an ad jingle, in noise). The pretrained ECAPA model stays frozen; we train a small
scorer on pair features of (profile, window embedding) and compare it with plain cosine similarity.

All splits are speaker-disjoint: test owners and test impostors never appear in training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicetrap import audio as A
from voicetrap.corpus import Speaker, load_utt

WIN_S = 1.2
POS_CONDITIONS = ["clean", "music", "noise", "quiet_other", "phone"]
NEG_CONDITIONS = ["clean", "ad", "noise", "phone"]


@dataclass
class Trial:
    owner: str
    other: str | None
    label: int
    condition: str
    owner_source: str
    other_source: str | None


def _crop(x: np.ndarray, rng: random.Random) -> np.ndarray:
    w = int(WIN_S * A.SR)
    if len(x) <= w:
        return np.pad(x, (0, w - len(x)))
    loud = A.active_rms(x)
    for _ in range(20):
        s = rng.randrange(0, len(x) - w)
        if A.rms(x[s : s + w]) > loud * 0.3:
            return x[s : s + w]
    return x[:w]


def _phone(x: np.ndarray) -> np.ndarray:
    """Crude 300-3400 Hz band-limit: sounds like a TV speaker or phone line."""
    f = np.fft.rfft(x)
    hz = np.fft.rfftfreq(len(x), 1 / A.SR)
    f[(hz < 300) | (hz > 3400)] = 0
    return np.fft.irfft(f, n=len(x)).astype("float32")


def augment(x: np.ndarray, condition: str, rng: random.Random, other: np.ndarray | None = None) -> np.ndarray:
    seed = rng.randrange(1 << 30)
    if condition in ("music", "ad"):
        bed = A.music_bed(len(x) / A.SR + 0.1, seed)[: len(x)]
        x = x + bed * A.gain_for(bed, x, rng.uniform(-12, -4))
    elif condition == "noise":
        n = A.room_noise(len(x) / A.SR + 0.1, seed)[: len(x)]
        x = x + n * A.gain_for(n, x, rng.uniform(-18, -8))
    elif condition == "quiet_other" and other is not None:
        o = other[: len(x)] if len(other) >= len(x) else np.pad(other, (0, len(x) - len(other)))
        x = x + o * A.gain_for(o, x, rng.uniform(-20, -8))
    elif condition == "phone":
        x = _phone(x)
    return A.normalize(x, peak=rng.uniform(0.3, 0.95))


def make_trials(speakers: list[Speaker], split: str, n_pos: int, n_neg: int, seed: int = 0,
                n_enroll: int = 3):
    """Return (profile_audio_per_owner, window_audio, trials) for one split."""
    rng = random.Random(seed)
    pool = [s for s in speakers if s.split == split]
    cache: dict[str, np.ndarray] = {}

    def utt(p):
        if p not in cache:
            cache[p] = load_utt(p)
        return cache[p]

    enroll, windows, trials = {}, [], []
    for owner in pool:
        enroll[owner.id] = [utt(p) for p in owner.files[:n_enroll]]
        rest = owner.files[n_enroll:]
        others = [s for s in pool if s.group != owner.group]
        for _ in range(n_pos):
            cond = rng.choice(POS_CONDITIONS)
            o = rng.choice(others)
            win = _crop(utt(rng.choice(rest)), rng)
            other_win = _crop(utt(rng.choice(o.files)), rng) if cond == "quiet_other" else None
            windows.append(augment(win, cond, rng, other_win))
            trials.append(Trial(owner.id, o.id if other_win is not None else None, 1, cond, owner.source,
                                o.source if other_win is not None else None))
        for _ in range(n_neg):
            cond = rng.choice(NEG_CONDITIONS)
            o = rng.choice(others)
            windows.append(augment(_crop(utt(rng.choice(o.files)), rng), cond, rng))
            trials.append(Trial(owner.id, o.id, 0, cond, owner.source, o.source))
    return enroll, windows, trials


def embed_all(embed, clips: list[np.ndarray], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(clips), batch):
        out.append(embed(clips[i : i + batch]))
    return np.concatenate(out) if out else np.zeros((0, 192))


def profiles(embed, enroll: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    prof = {}
    for k, clips in enroll.items():
        p = embed(clips).mean(0)
        prof[k] = p / np.linalg.norm(p)
    return prof


def pair_features(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """[cosine, elementwise product, absolute difference] of window embedding and owner profile."""
    cos = np.sum(E * P, axis=1, keepdims=True)
    return np.hstack([cos, E * P, np.abs(E - P)]).astype("float32")


# ---------------- metrics ----------------
def eer(y: np.ndarray, s: np.ndarray) -> float:
    order = np.sort(np.unique(s))
    best = 1.0
    for t in order:
        far = np.mean(s[y == 0] >= t)
        frr = np.mean(s[y == 1] < t)
        best = min(best, max(far, frr))
    return float(best)


def far_at_frr(y: np.ndarray, s: np.ndarray, frr: float = 0.05) -> tuple[float, float]:
    """Threshold that rejects at most ``frr`` of owner windows, and the impostor accept rate there."""
    t = float(np.quantile(s[y == 1], frr))
    return float(np.mean(s[y == 0] >= t)), t


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


@dataclass
class Scorer:
    """Callable used by the speaker gate: (window embeddings, profile) -> scores."""

    kind: str  # cosine | trained
    model: object = None

    cohort: np.ndarray | None = None
    features: str = "pair"  # pair (v1 model) | compact | full (v2 models, need the cohort)

    def __call__(self, E: np.ndarray, p: np.ndarray) -> np.ndarray:
        P = np.repeat(p[None, :], len(E), axis=0)
        if self.kind == "cosine":
            return np.sum(E * P, axis=1)
        if self.features == "pair":
            return self.model.predict_proba(pair_features(E, P))[:, 1]
        return self.model.predict_proba(Feats(E, P, self.cohort).get(self.features))[:, 1]

    @classmethod
    def load_v2(cls, path: Path) -> "Scorer":
        import joblib

        d = joblib.load(path)
        feats = "full" if "full" in d["name"] else "compact"
        return cls("trained", d["model"], d["cohort"].astype("float32"), feats)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def _split_data(embed, speakers, split, n_pos, n_neg, seed, cache: Path):
    f = cache / f"{split}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return d["E"], d["P"], [Trial(**t) for t in d["trials"]]
    enroll, windows, trials = make_trials(speakers, split, n_pos, n_neg, seed)
    prof = profiles(embed, enroll)
    E = embed_all(embed, windows)
    P = np.stack([prof[t.owner] for t in trials])
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, E=E, P=P, trials=np.array([t.__dict__ for t in trials], dtype=object))
    return E, P, trials


def _report(y, s, trials) -> dict:
    far5, thr = far_at_frr(y, s, 0.05)
    out = {"eer": round(eer(y, s), 4), "auc": round(auc(y, s), 4), "far_at_frr5": round(far5, 4),
           "threshold_frr5": round(thr, 4), "n": int(len(y))}
    by = {}
    conds = sorted({t.condition for t in trials})
    for c in conds:  # each condition vs all negatives (pos) or all positives (neg)
        m = np.array([t.condition == c for t in trials])
        if c in POS_CONDITIONS:
            mm = m & (y == 1) | (y == 0)
            by[f"owner:{c}"] = {"frr_at_t": round(float(np.mean(s[m & (y == 1)] < thr)), 4)}
            by[f"owner:{c}"]["eer"] = round(eer(y[mm], s[mm]), 4)
        if c in NEG_CONDITIONS:
            by[f"impostor:{c}"] = {"far_at_t": round(float(np.mean(s[m & (y == 0)] >= thr)), 4)}
    for src in ("tts", "real"):
        m = np.array([t.owner_source == src for t in trials])
        if m.any():
            by[f"owners:{src}"] = {"eer": round(eer(y[m], s[m]), 4), "n": int(m.sum())}
    out["by"] = by
    return out


def run_experiment(out: Path = Path("results/gate"), corpus_root: Path = Path("data/corpus"),
                   seed: int = 0, log=print) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from voicetrap.corpus import load
    from voicetrap.speaker import Embedder

    speakers = load(corpus_root)
    counts = {sp: {src: len({s.group for s in speakers if s.split == sp and s.source == src})
                   for src in ("tts", "real")} for sp in ("train", "val", "test")}
    log(f"speaker groups per split: {counts}")
    embed = Embedder()
    cache = out / "embeddings"
    data = {
        "train": _split_data(embed, speakers, "train", 16, 16, seed, cache),
        "val": _split_data(embed, speakers, "val", 16, 24, seed + 1, cache),
        "test": _split_data(embed, speakers, "test", 16, 32, seed + 2, cache),
    }
    X = {k: pair_features(E, P) for k, (E, P, _) in data.items()}
    Y = {k: np.array([t.label for t in tr]) for k, (_, _, tr) in data.items()}
    log({k: f"{len(v)} trials" for k, v in Y.items()})

    candidates = {f"logreg C={c}": make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                  for c in (0.01, 0.1, 1.0)}
    candidates |= {f"mlp h={h}": make_pipeline(StandardScaler(), MLPClassifier(
        (h,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=seed)) for h in (32, 128)}
    val_scores = {}
    for name, m in candidates.items():
        m.fit(X["train"], Y["train"])
        val_scores[name] = eer(Y["val"], m.predict_proba(X["val"])[:, 1])
        log(f"  {name:14s} val EER {val_scores[name]:.4f}")
    best = min(val_scores, key=val_scores.get)
    model = candidates[best]
    cos_val = eer(Y["val"], X["val"][:, 0])
    log(f"cosine val EER {cos_val:.4f}; selected {best}")

    tr = data["test"][2]
    results = {
        "splits": counts, "trials": {k: int(len(v)) for k, v in Y.items()},
        "selected": best, "val_eer": {**{k: round(v, 4) for k, v in val_scores.items()}, "cosine": round(cos_val, 4)},
        "test": {"cosine": _report(Y["test"], X["test"][:, 0], tr),
                 "trained": _report(Y["test"], model.predict_proba(X["test"])[:, 1], tr)},
    }
    import joblib

    joblib.dump(model, out / "gate_model.joblib")
    _, thr_cos = far_at_frr(Y["val"], X["val"][:, 0], 0.05)
    _, thr_tr = far_at_frr(Y["val"], model.predict_proba(X["val"])[:, 1], 0.05)
    results["operating_points_from_val"] = {"cosine": round(thr_cos, 4), "trained": round(thr_tr, 4)}
    save_json(out / "metrics.json", results)
    return results


# ---------------- v2: cohort normalisation + grouped cross-validation ----------------
def cohort_stats(C: np.ndarray, X: np.ndarray, k: int = 200, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of each row of X against its top-k most similar cohort embeddings (AS-norm).
    Computed in chunks so memory stays small (chunk x cohort floats at a time)."""
    k = min(k, len(C))
    mu, sd = np.empty(len(X), "float32"), np.empty(len(X), "float32")
    for i in range(0, len(X), chunk):
        S = X[i : i + chunk] @ C.T
        top = np.partition(S, S.shape[1] - k, axis=1)[:, -k:]
        mu[i : i + chunk], sd[i : i + chunk] = top.mean(1), top.std(1) + 1e-6
    return mu, sd


def subsample(C: np.ndarray, n: int = 4000, seed: int = 0) -> np.ndarray:
    if len(C) <= n:
        return C
    return C[np.random.default_rng(seed).choice(len(C), n, replace=False)]


def norm_features(E: np.ndarray, P: np.ndarray, C: np.ndarray, k: int = 200) -> np.ndarray:
    """[cosine, AS-norm score, owner-side mean/std, window-side mean/std] — six compact features."""
    s = np.sum(E * P, axis=1)
    mp, sp = cohort_stats(C, P, k)
    me, se = cohort_stats(C, E, k)
    asn = 0.5 * ((s - mp) / sp + (s - me) / se)
    return np.stack([s, asn, mp, sp, me, se], axis=1).astype("float32")


def _candidates(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    c = {"cosine": None, "asnorm": None}
    for C_ in (0.01, 0.1, 1.0):
        c[f"logreg-compact C={C_}"] = ("compact", make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=3000)))
    c["mlp-compact h=16"] = ("compact", make_pipeline(StandardScaler(), MLPClassifier(
        (16,), alpha=1e-2, max_iter=500, early_stopping=True, random_state=seed)))
    for C_ in (0.01, 0.1):
        c[f"logreg-full C={C_}"] = ("full", make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=3000)))
    c["mlp-full h=128"] = ("full", make_pipeline(StandardScaler(), MLPClassifier(
        (128,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=seed)))
    return c


class Feats:
    """Compact (AS-norm) and full feature matrices for one (E, P, cohort), computed once and reused."""

    def __init__(self, E, P, C):
        self.E, self.P = E, P
        self.compact = norm_features(E, P, C)
        self._full = None

    def get(self, kind):
        if kind == "compact":
            return self.compact
        if self._full is None:
            self._full = np.hstack([self.compact, pair_features(self.E, self.P)])
        return self._full


def _score(name, spec, F: Feats, fitted=None):
    if name == "cosine":
        return F.compact[:, 0]
    if name == "asnorm":
        return F.compact[:, 1]
    return fitted.predict_proba(F.get(spec[0]))[:, 1]


def run_experiment2(out: Path = Path("results/gate"), corpus_root: Path = Path("data/corpus"), seed: int = 0,
                    folds: int = 5, extra_train_seeds=(10,), log=print) -> dict:
    """Model selection by speaker-grouped K-fold CV on train+val; one final evaluation on unseen test speakers."""
    import copy

    from sklearn.model_selection import GroupKFold

    from voicetrap.corpus import load
    from voicetrap.speaker import Embedder

    speakers = load(corpus_root)
    group = {s.id: s.group for s in speakers}
    embed = Embedder()
    cache = out / "embeddings"
    parts = [_split_data(embed, speakers, "train", 16, 16, seed, cache),
             _split_data(embed, speakers, "val", 16, 24, seed + 1, cache)]
    for s_ in extra_train_seeds:  # more trials from the same training speakers, new crops/augmentations
        f = cache / f"train_s{s_}.npz"
        if not f.exists():
            enroll, windows, trials = make_trials(speakers, "train", 16, 16, s_)
            prof = profiles(embed, enroll)
            np.savez_compressed(f, E=embed_all(embed, windows), P=np.stack([prof[t.owner] for t in trials]),
                                trials=np.array([t.__dict__ for t in trials], dtype=object))
        d = np.load(f, allow_pickle=True)
        parts.append((d["E"], d["P"], [Trial(**t) for t in d["trials"]]))
    E = np.concatenate([p[0] for p in parts])
    P = np.concatenate([p[1] for p in parts])
    T = [t for p in parts for t in p[2]]
    y = np.array([t.label for t in T])
    g = np.array([group[t.owner] for t in T])
    log(f"train+val pool: {len(y)} trials from {len(set(g))} owner groups")

    cands = _candidates(seed)
    oof = {name: np.zeros(len(y)) for name in cands}
    for k, (tr, te) in enumerate(GroupKFold(folds).split(E, y, g)):
        C = subsample(E[tr], seed=k)  # cohort = windows of the training-fold speakers only
        Ftr, Fte = Feats(E[tr], P[tr], C), Feats(E[te], P[te], C)
        for name, spec in cands.items():
            fitted = None if spec is None else copy.deepcopy(spec[1]).fit(Ftr.get(spec[0]), y[tr])
            oof[name][te] = _score(name, spec, Fte, fitted)
        del Ftr, Fte
        log(f"  fold {k + 1}/{folds} done")
    cv = {name: round(eer(y, s), 4) for name, s in oof.items()}
    for name, v in sorted(cv.items(), key=lambda kv: kv[1]):
        log(f"  CV EER {v:.4f}  {name}")
    best = min(cv, key=cv.get)

    # final fit on all train+val speakers; operating point from out-of-fold scores (no test peeking)
    C_all = subsample(E, seed=99)
    spec = cands[best]
    F_all = Feats(E, P, C_all)
    final = None if spec is None else copy.deepcopy(spec[1]).fit(F_all.get(spec[0]), y)
    _, thr = far_at_frr(y, oof[best], 0.05)
    Et, Pt, Tt = _split_data(embed, speakers, "test", 16, 32, seed + 2, cache)
    yt = np.array([t.label for t in Tt])
    Ft = Feats(Et, Pt, C_all)
    test = {name: _report(yt, _score(name, cands[name], Ft, final if name == best else None), Tt)
            for name in dict.fromkeys(("cosine", "asnorm", best))}
    st = _score(best, spec, Ft, final)
    test["selected_at_cv_threshold"] = {"threshold": round(float(thr), 4),
                                        "owner_reject": round(float(np.mean(st[yt == 1] < thr)), 4),
                                        "impostor_accept": round(float(np.mean(st[yt == 0] >= thr)), 4)}
    results = {"selected": best, "cv_eer": cv, "pool_trials": int(len(y)), "test_trials": int(len(yt)),
               "test": test, "operating_point": round(float(thr), 4)}
    import joblib

    joblib.dump({"name": best, "model": final, "cohort": C_all.astype("float16"), "threshold": float(thr)},
                out / "gate_model_v2.joblib")
    save_json(out / "metrics_v2.json", results)
    return results
