"""Offline tests for the speaker split, trial augmentation and metrics."""

import random

import numpy as np
import pytest

from voicetrap import audio as A
from voicetrap.corpus import Speaker, split, split_of
from voicetrap.gatetrain import Scorer, auc, augment, eer, far_at_frr, make_trials, pair_features


def test_split_is_stable_and_speaker_disjoint():
    assert split_of("tts:Eddy") == split_of("tts:Eddy")
    sp = [Speaker(f"tts:Eddy ({loc})", "tts:Eddy", "tts", loc) for loc in ("en_US", "fr_FR", "de_DE")]
    sp += [Speaker(f"real:{i}", f"real:{i}", "real", "en") for i in range(200)]
    split(sp)
    assert len({s.split for s in sp if s.group == "tts:Eddy"}) == 1  # locale variants never straddle splits
    shares = {k: sum(s.split == k for s in sp[3:]) / 200 for k in ("train", "val", "test")}
    assert 0.2 < shares["test"] < 0.4 and 0.08 < shares["val"] < 0.25


def test_metrics_on_known_scores():
    y = np.array([1, 1, 1, 0, 0, 0])
    assert eer(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0
    assert auc(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == pytest.approx(1.0)
    assert auc(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(0.0)
    far, thr = far_at_frr(y, np.array([0.9, 0.8, 0.7, 0.75, 0.2, 0.1]), frr=0.0)
    assert thr == pytest.approx(0.7) and far == pytest.approx(1 / 3)


def test_pair_features_and_cosine_scorer():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(5, 192))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    p = E[0]
    X = pair_features(E, np.repeat(p[None], 5, 0))
    assert X.shape == (5, 385) and X[0, 0] == pytest.approx(1.0)
    assert Scorer("cosine")(E, p)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("cond", ["clean", "music", "ad", "noise", "quiet_other", "phone"])
def test_augment_keeps_length_and_range(cond):
    t = np.arange(int(1.2 * A.SR)) / A.SR
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32")
    y = augment(x, cond, random.Random(1), other=x[::-1].copy())
    assert len(y) == len(x) and np.abs(y).max() <= 0.95 + 1e-6


def test_make_trials_never_pairs_owner_with_own_group(tmp_path):
    t = np.arange(int(2 * A.SR)) / A.SR
    sps = []
    for k in range(4):
        files = []
        for j in range(6):
            p = A.save(tmp_path / f"s{k}-{j}.flac", (0.3 * np.sin(2 * np.pi * (150 + 40 * k + j) * t)).astype("float32"))
            files.append(str(p))
        sps.append(Speaker(f"tts:v{k}", f"tts:v{k % 2}", "tts", "en_US", files=files, split="train"))
    enroll, windows, trials = make_trials(sps, "train", 3, 3, seed=0)
    group = {s.id: s.group for s in sps}
    assert len(windows) == len(trials) == 4 * 6 and set(enroll) == {s.id for s in sps}
    assert all(group[t.other] != group[t.owner] for t in trials if t.other)
    assert all(len(w) == int(1.2 * A.SR) for w in windows)
