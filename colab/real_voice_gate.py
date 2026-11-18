# %% [markdown]
# # VoiceTrap: speaker gate on real voices (Colab T4)
#
# Trains and tests the speaker gate **only on real people**:
#
# | Role | Data | Speakers |
# |---|---|---|
# | Fine-tune | LibriSpeech train-clean-100 | 251 |
# | Pick the best epoch | LibriSpeech dev-clean | 40 |
# | Test, same dataset | LibriSpeech test-clean | 40 |
# | Test, **different dataset** | VCTK 0.92 (mic1) | 110 |
#
# Compared on identical trials (1.2 s windows: clean, music, noise, room echo, phone line, a second voice):
# 1. pretrained ECAPA-TDNN (what VoiceTrap uses now)
# 2. pretrained WavLM-base-plus-sv
# 3. ECAPA fine-tuned here with AAM-softmax and the same augmentations
#
# **Before running:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all.
# Everything is saved to Google Drive (`MyDrive/voicetrap_gpu`), so after a disconnect just run all again:
# finished steps are skipped and training resumes from the last epoch.

# %%
import os
import subprocess
import sys
from pathlib import Path

SMOKE = os.environ.get("SMOKE") == "1"  # tiny local test run on CPU
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain>=1.0,<2"], check=True)
    WORK, OUT = Path("/content/work"), Path("/content/drive/MyDrive/voicetrap_gpu")
else:
    WORK = Path(os.environ.get("WORK", "work"))
    OUT = WORK / "out"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
print("work:", WORK, "| results:", OUT, "| smoke:", SMOKE)

# %% [markdown]
# ## 1. Download the datasets (Hugging Face copies: a few minutes on Colab)
# The original servers (openslr.org, datashare.ed.ac.uk) run at under 1 MB/s from Colab, so the same FLAC files are
# read from the Hugging Face copies: `openslr/librispeech_asr` (CC BY 4.0) and `sanchit-gandhi/vctk` (VCTK 0.92,
# CC BY 4.0). VCTK keeps the mic1 recordings, up to 20 per speaker.

# %%
from collections import Counter

LIBRI_HF = {"train-clean-100": "train.clean.100", "dev-clean": "validation.clean", "test-clean": "test.clean"}


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def unpack(repo, pattern, dest, name_of, keep=lambda row: True):
    """Download parquet shards from the Hugging Face Hub and write each row's original FLAC bytes to dest/speaker/."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    if (dest / ".done").exists():
        return dest
    hf = WORK / "hf"
    snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern], local_dir=hf, max_workers=8)
    for f in sorted(hf.glob(pattern)):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if keep(row):
                    out = dest / str(row["speaker_id"]) / name_of(row)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(row["audio"]["bytes"])
        f.unlink()  # free the disk as we go
    (dest / ".done").touch()
    return dest


def fetch_libri(split):
    return unpack("openslr/librispeech_asr", f"all/{LIBRI_HF[split]}/*.parquet", WORK / "LibriSpeech" / split,
                  lambda r: f"{r['id']}.flac")


def fetch_vctk(per_speaker=20):
    kept = Counter()

    def keep(r):
        if "_mic1" not in r["audio"]["path"] or kept[r["speaker_id"]] >= per_speaker:
            return False
        kept[r["speaker_id"]] += 1
        return True

    return unpack("sanchit-gandhi/vctk", "data/*.parquet", WORK / "vctk", lambda r: r["audio"]["path"], keep)


def speakers(root):
    """speaker id -> sorted list of FLAC files (the speaker is the first folder under root)."""
    out = {}
    for f in sorted(root.rglob("*.flac")):
        out.setdefault(f.relative_to(root).parts[0], []).append(f)
    return {k: v for k, v in out.items() if len(v) >= 6}


if SMOKE:
    local = Path(os.environ["LIBRI_LOCAL"])
    TRAIN = speakers(local / "train-clean-5")
    dev = speakers(local / "dev-clean-2")
    keys = sorted(dev)
    VAL = {k: dev[k] for k in keys[: len(keys) // 2]}
    TESTS = {"librispeech-test": {k: dev[k] for k in keys[len(keys) // 2 :]}}
else:
    TRAIN = speakers(fetch_libri("train-clean-100"))
    VAL = speakers(fetch_libri("dev-clean"))
    TESTS = {"librispeech-test": speakers(fetch_libri("test-clean")), "vctk": speakers(fetch_vctk())}
assert not set(TRAIN) & set(VAL) and all(not set(TRAIN) & set(t) for t in TESTS.values())
print("speakers  train", len(TRAIN), "| val", len(VAL), "|", {k: len(v) for k, v in TESTS.items()})

# %% [markdown]
# ## 2. Audio and augmentation
# The same six conditions are used for training and testing. Room echo is a synthetic exponentially decaying impulse
# response; the phone line is a 300-3400 Hz band limit through 8 kHz.

# %%
import random

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 16000
CONDITIONS = ["clean", "music", "noise", "reverb", "phone", "second_voice"]
_PHONE = butter(6, [300, 3400], btype="band", fs=SR, output="sos")


def read(path, seconds=None, rng=None):
    """Read a file (or a random `seconds`-long crop of it) as 16 kHz mono float32."""
    info = sf.info(str(path))
    native = int(seconds * info.samplerate) if seconds else None
    start = 0
    if native and info.frames > native:
        start = (rng or random).randrange(info.frames - native)
    x, sr = sf.read(str(path), start=start, frames=native or -1, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    if seconds:
        n = int(seconds * SR)
        x = np.pad(x, (0, max(0, n - len(x))))[:n]
    return x


def rms(x):
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def mix(x, y, snr_db):
    return x + y * (rms(x) / rms(y)) * 10 ** (-snr_db / 20)


def pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(np.maximum(f, 1 / n))
    return np.fft.irfft(spec, n).astype("float32")


def music(n, rng):
    t = np.arange(n) / SR
    root = 110 * 2 ** (rng.integers(0, 12) / 12)
    y = sum(np.sin(2 * np.pi * root * r * t + rng.uniform(0, 6.28)) for r in (1, 1.26, 1.5, 2, 3))
    beat = 0.6 + 0.4 * (np.sin(2 * np.pi * rng.uniform(1.5, 3) * t) > 0)
    return (y * beat).astype("float32")


def reverb(x, rng):
    rt60 = rng.uniform(0.2, 0.8)
    t = np.arange(int(rt60 * SR)) / SR
    h = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    h[0] = 4.0  # direct path
    y = fftconvolve(x, h)[: len(x)]
    return (y * rms(x) / rms(y)).astype("float32")


def phone(x):
    y = resample_poly(sosfilt(_PHONE, x), 1, 2)
    y = resample_poly(np.clip(y * 1.5, -0.6, 0.6), 2, 1)[: len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype("float32")


def augment(x, cond, rng, other=None):
    if cond == "music":
        x = mix(x, music(len(x), rng), rng.uniform(0, 12))
    elif cond == "noise":
        x = mix(x, pink(len(x), rng), rng.uniform(0, 15))
    elif cond == "reverb":
        x = reverb(x, rng)
    elif cond == "phone":
        x = phone(x)
    elif cond == "second_voice":
        x = mix(x, other, rng.uniform(10, 20))  # someone else talking quietly in the room
    peak = np.abs(x).max()
    return (x * (0.9 / peak) if peak > 0.9 else x).astype("float32")


# %% [markdown]
# ## 3. Fixed test trials
# For every test speaker: an owner profile from 3 clean recordings, then 1.2 s windows of the owner and of other
# speakers under every condition. Trials are generated once with a fixed seed and saved, so every model is scored on
# exactly the same audio.

# %%
WIN_S = 1.2
PER_COND = 2 if SMOKE else 4  # owner windows and impostor windows per speaker per condition


def make_trials(spk, name, seed=0):
    path = OUT / f"trials-{name}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    ids = sorted(spk)
    enroll, owner, label, cond, windows = [], [], [], [], []
    for i, s in enumerate(ids):
        files = spk[s]
        enroll.append(np.concatenate([read(f)[: 8 * SR] for f in files[:3]]))
        rest = files[3:]
        for c in CONDITIONS:
            for k in range(PER_COND * 2):
                is_owner = k < PER_COND
                who = s if is_owner else ids[(i + 1 + rng.integers(len(ids) - 1)) % len(ids)]
                f = prng.choice(rest if is_owner else spk[who])
                x = read(f, WIN_S, prng)
                other = None
                if c == "second_voice":
                    third = ids[(i + 1 + rng.integers(len(ids) - 1)) % len(ids)]
                    other = read(prng.choice(spk[third]), WIN_S, prng)
                windows.append(augment(x, c, rng, other))
                owner.append(i)
                label.append(int(is_owner))
                cond.append(c)
    out = {"enroll": np.array(enroll, dtype=object), "windows": (np.stack(windows) * 32767).astype("int16"),
           "owner": np.array(owner), "label": np.array(label), "cond": np.array(cond), "speakers": np.array(ids)}
    np.savez(path, **out)
    return out


VAL_TRIALS = make_trials(VAL, "val", seed=1)
TEST_TRIALS = {k: make_trials(v, k, seed=2) for k, v in TESTS.items()}
print({k: len(v["label"]) for k, v in TEST_TRIALS.items()}, "test trials; val", len(VAL_TRIALS["label"]))

# %% [markdown]
# ## 4. Metrics and scoring (same as `gatetrain.py`)

# %%
import json

from sklearn.metrics import roc_auc_score, roc_curve


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def far_at_frr(y, s, frr=0.05):
    thr = np.quantile(s[y == 1], frr)
    return float((s[y == 0] >= thr).mean())


def evaluate(embed, trials, batch=64):
    """embed(list of float32 arrays) -> L2-normalised [N, D] tensor."""
    prof = torch.stack([embed([e]).squeeze(0) for e in trials["enroll"]])
    prof = torch.nn.functional.normalize(prof, dim=-1)
    W = trials["windows"].astype("float32") / 32767
    E = torch.cat([embed(list(W[i : i + batch])) for i in range(0, len(W), batch)])
    s = (E * prof[trials["owner"]]).sum(-1).cpu().numpy()
    y, c = trials["label"], trials["cond"]
    res = {"eer": eer(y, s), "auc": float(roc_auc_score(y, s)), "far_at_frr5": far_at_frr(y, s), "n": int(len(y))}
    res["by_condition"] = {k: round(eer(y[c == k], s[c == k]), 4) for k in CONDITIONS}
    return res


RESULTS = OUT / "results.json"
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def record(model, name, res):
    results.setdefault(model, {})[name] = res
    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"{model:22s} {name:18s} EER {100 * res['eer']:.2f}%  AUC {res['auc']:.4f}  "
          f"FA@5%FR {100 * res['far_at_frr5']:.2f}%  {res['by_condition']}")


# %% [markdown]
# ## 5. Baselines: pretrained ECAPA and WavLM

# %%
from speechbrain.inference.speaker import EncoderClassifier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(WORK / "ecapa"),
                                       run_opts={"device": DEV})


def ecapa_embed_fn(model):
    @torch.no_grad()
    def embed(xs):
        n = max(len(x) for x in xs)
        wav = torch.from_numpy(np.stack([np.pad(x, (0, n - len(x))) for x in xs])).to(DEV)
        lens = torch.tensor([len(x) / n for x in xs], device=DEV)
        feats = ecapa.mods.mean_var_norm(ecapa.mods.compute_features(wav), lens)
        return torch.nn.functional.normalize(model(feats, lens).squeeze(1), dim=-1)

    return embed


def run_baseline(model_name, embed):
    for name, tr in TEST_TRIALS.items():
        if name not in results.get(model_name, {}):
            record(model_name, name, evaluate(embed, tr))


ecapa.mods.eval()
run_baseline("ecapa_pretrained", ecapa_embed_fn(ecapa.mods.embedding_model))

try:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEV).eval()

    @torch.no_grad()
    def wavlm_embed(xs):
        inp = fe(list(xs), sampling_rate=SR, return_tensors="pt", padding=True).to(DEV)
        return torch.nn.functional.normalize(wavlm(**inp).embeddings, dim=-1)

    run_baseline("wavlm_pretrained", wavlm_embed)
except ImportError:
    print("transformers not installed: skipping WavLM")

# %% [markdown]
# ## 6. Fine-tune ECAPA on 251 real speakers
# AAM-softmax (margin 0.2, scale 30), random 2 s crops, the augmentations above on 70% of crops.
# The classifier starts from each speaker's mean pretrained embedding, and the backbone uses a small learning rate,
# so it adapts without forgetting what it learned from VoxCeleb. The best epoch is chosen on **dev-clean only**.

# %%
import copy

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EPOCHS = 1 if SMOKE else 6
STEPS = 3 if SMOKE else 1500
BATCH = 4 if SMOKE else 64
CROP_S = 2.0
spk_ids = sorted(TRAIN)


class Crops(Dataset):
    """Random 2 s crops. Seeded by (epoch, index) so every epoch sees different crops and augmentations."""

    def __init__(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return STEPS * BATCH

    def __getitem__(self, i):
        rng = np.random.default_rng([self.epoch, i])
        prng = random.Random(int(rng.integers(2**31)))
        y = int(rng.integers(len(spk_ids)))
        x = read(prng.choice(TRAIN[spk_ids[y]]), CROP_S, prng)
        if rng.random() < 0.7:
            c = CONDITIONS[1 + int(rng.integers(len(CONDITIONS) - 1))]
            other = None
            if c == "second_voice":
                o = spk_ids[(y + 1 + int(rng.integers(len(spk_ids) - 1))) % len(spk_ids)]
                other = read(prng.choice(TRAIN[o]), CROP_S, prng)
            x = augment(x, c, rng, other)
        return torch.from_numpy(x), y


class AAM(torch.nn.Module):
    def __init__(self, W, m=0.2, s=30.0):
        super().__init__()
        self.W, self.m, self.s = torch.nn.Parameter(W.clone()), m, s

    def forward(self, e, y):
        cos = F.linear(F.normalize(e), F.normalize(self.W)).clamp(-1 + 1e-6, 1 - 1e-6)
        target = torch.cos(torch.acos(cos) + self.m)
        logits = self.s * torch.where(F.one_hot(y, cos.shape[1]).bool(), target, cos)
        return F.cross_entropy(logits, y)


model = copy.deepcopy(ecapa.mods.embedding_model).to(DEV)
pre_embed = ecapa_embed_fn(ecapa.mods.embedding_model)
CKPT = OUT / "finetune_ckpt.pt"
BEST = OUT / "ecapa_finetuned_best.pt"
if CKPT.exists():
    ck = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ck["model"])
    head = AAM(ck["head"]).to(DEV)
    start, best_eer, history = ck["epoch"] + 1, ck["best_eer"], ck["history"]
else:
    centres = torch.stack([pre_embed([read(f, 4.0) for f in TRAIN[s][:6]]).mean(0) for s in spk_ids])
    head = AAM(F.normalize(centres, dim=-1).cpu()).to(DEV)
    val0 = evaluate(pre_embed, VAL_TRIALS)
    start, best_eer, history = 0, val0["eer"], [{"epoch": -1, "val_eer": val0["eer"]}]
    torch.save(model.state_dict(), BEST)  # epoch -1 = the pretrained model; fine-tuning must beat it
    print(f"pretrained val EER {100 * val0['eer']:.2f}%")

opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 2e-5},
                         {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
if CKPT.exists():
    opt.load_state_dict(ck["opt"])
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
for epoch in range(start, EPOCHS):
    loader = DataLoader(Crops(epoch), batch_size=BATCH, num_workers=0 if SMOKE else os.cpu_count(), drop_last=True)
    model.train()
    total = 0.0
    for step, (wav, y) in enumerate(loader):
        wav, y = wav.to(DEV), y.to(DEV)
        lens = torch.ones(len(wav), device=DEV)
        with torch.no_grad():
            feats = ecapa.mods.mean_var_norm(ecapa.mods.compute_features(wav), lens)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            loss = head(model(feats, lens).squeeze(1).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        if step % 250 == 0:
            print(f"epoch {epoch} step {step}/{STEPS} loss {loss.item():.3f}")
    model.eval()
    v = evaluate(ecapa_embed_fn(model), VAL_TRIALS)
    history.append({"epoch": epoch, "loss": total / STEPS, "val_eer": v["eer"]})
    print(f"epoch {epoch}: mean loss {total / STEPS:.3f}, val EER {100 * v['eer']:.2f}% (best {100 * best_eer:.2f}%)")
    if v["eer"] < best_eer:
        best_eer = v["eer"]
        torch.save(model.state_dict(), BEST)
    torch.save({"model": model.state_dict(), "head": head.W.detach().cpu(), "opt": opt.state_dict(),
                "epoch": epoch, "best_eer": best_eer, "history": history}, CKPT)

(OUT / "finetune_history.json").write_text(json.dumps(history, indent=1))

# %% [markdown]
# ## 7. Test the fine-tuned model (best epoch by dev-clean) on the unseen test sets

# %%
model.load_state_dict(torch.load(BEST, map_location=DEV))
model.eval()
for name, tr in TEST_TRIALS.items():
    record("ecapa_finetuned", name, evaluate(ecapa_embed_fn(model), tr))

print("\n| Model | Test set | EER | AUC | Wrong voices accepted at 5% owner rejects |")
print("|---|---|:-:|:-:|:-:|")
for m, sets in results.items():
    for name, r in sets.items():
        print(f"| {m} | {name} | {100 * r['eer']:.2f}% | {r['auc']:.4f} | {100 * r['far_at_frr5']:.2f}% |")
print("\nSaved to", OUT, "- download ecapa_finetuned_best.pt and results.json")
