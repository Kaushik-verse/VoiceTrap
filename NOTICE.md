# Notices, licences and attribution

VoiceTrap's **source code** is released under the MIT License (see `LICENSE`), © 2026 Ch Kaushik.
That covers the code, the tests and the original diagrams in `docs/`. Nothing else in this file is covered by it:
the speech data, voices and models below all keep their own terms, and several of them are **non-commercial**.

If you only want the short version:

- The code is MIT. Use it however you like.
- **No third-party audio is stored in this repository.** Every recording is downloaded or synthesised on your own
  machine when you run the scripts.
- Four of the things the experiments rely on are **non-commercial**: Apple's macOS voices, the SLURP recordings,
  the F5-TTS weights and the WavLM weights (ShareAlike). The benchmark results are research results. If you want to
  build a product on this, swap those parts out first.

## Speech data

| What | Used for | Licence | In this repo? |
|---|---|---|---|
| **macOS system voices** (`say`) | owners and attackers in the v1 and v2 benchmarks; part of the gate training set | Apple macOS Software Licence Agreement, §2F: personal, non-commercial use only | **No.** Built locally by `./run_all.sh` |
| [**LibriSpeech**](https://www.openslr.org/12/) (Panayotov et al., 2015), read through the [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) copy | training and testing the speaker gate | CC BY 4.0 | No |
| [**Mini LibriSpeech**](https://www.openslr.org/31/) | real speakers in the first gate dataset | CC BY 4.0 | No |
| [**CSTR VCTK Corpus 0.92**](https://datashare.ed.ac.uk/handle/10283/3443) (Yamagishi, Veaux & MacDonald, 2019), read through [`sanchit-gandhi/vctk`](https://huggingface.co/datasets/sanchit-gandhi/vctk) | testing the speaker gate on new accents and microphones | CC BY 4.0 | No |
| [**SLURP**](https://github.com/pswietojanski/slurp) (Bastianelli et al., 2020) | the real owners in the real-owner benchmark | text CC BY 4.0, **audio CC BY-NC 4.0** | No. `data/slurp_bench/` is git-ignored |

### About the macOS voices

The licence on every Mac (`/Library/Documentation/License.lpdf`, §2F, macOS 27) lets you use the system voices
"to create your own original content and projects for your personal, non-commercial use" and rules out
"publishing or redistribution" of them "in a profit, non-profit, public sharing or commercial context".

So the audio made with them never leaves the machine that made it. Earlier versions of this repository did include
those clips, along with voice embeddings and a small classifier trained on them. They were removed once the problem
was spotted. To reproduce the v1 and v2 benchmarks you need a Mac, and the scripts rebuild the same audio from the
same text and settings. The published numbers are measurements, not voice recordings.

### About SLURP

The SLURP recordings are CC BY-NC 4.0. The Colab notebook downloads them from Zenodo into your own Google Drive,
and nothing is copied into this repository. Results computed on SLURP (`results/slurp/`) are shared for research
and education. If you need SLURP for commercial work, its authors say a less strict licence can be arranged
through Emotech (<info@emotech.co>).

## Models

| Model | Used for | Licence | How it is used |
|---|---|---|---|
| [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (ECAPA-TDNN, trained on VoxCeleb) | the speaker gate | Apache-2.0 | downloaded at run time |
| ECAPA fine-tuned on LibriSpeech train-clean-100 (`ecapa_finetuned_best.pt`) | a candidate gate | derived from the two above | **not published.** You get it by running `colab/real_voice_gate.ipynb` |
| [`microsoft/wavlm-base-plus-sv`](https://huggingface.co/microsoft/wavlm-base-plus-sv) | comparison only, in Colab | [CC BY-SA 3.0](https://github.com/microsoft/UniSpeech/blob/main/LICENSE) | downloaded at run time, never redistributed |
| [F5-TTS](https://github.com/SWivid/F5-TTS) (Chen et al., 2024) | attacker voices and voice clones in the real-owner benchmark | code MIT, **weights CC BY-NC 4.0** (trained on Emilia) | run in Colab; the generated audio is never published |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), through Groq | speech recognition in the agent | MIT | API |
| [Whisper small](https://huggingface.co/openai/whisper-small) | checking that every synthetic attack line says what it should | Apache-2.0 | run in Colab |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), through Groq | the voice agent | Apache-2.0 | API |
| [Llama Prompt Guard 2 86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M), through Groq | the "Prompt Guard" baseline defence | Llama 4 Community Licence | API |

Llama Prompt Guard 2 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc.
All Rights Reserved.

The Groq API is used on its free tier, under [Groq's terms](https://groq.com/terms-of-use). Responses are cached
locally in `results/*/cache/`, which is git-ignored.

## Software

| Library | Licence |
|---|---|
| NumPy, SciPy, scikit-learn, pandas | BSD-3-Clause |
| PyTorch, torchaudio | BSD-3-Clause |
| SpeechBrain | Apache-2.0 |
| soundfile | BSD-3-Clause |
| httpx | BSD-3-Clause |
| python-dotenv | BSD-3-Clause |
| Streamlit | Apache-2.0 |
| Altair | BSD-3-Clause |
| Hugging Face `transformers`, `huggingface_hub`, PyArrow | Apache-2.0 |
| pytest | MIT |

They are all installed from PyPI, and none is copied into this repository.

## Papers

If you use this work, please also cite the work it rests on:

- Desplanques, Thienpondt & Demuynck. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
  TDNN Based Speaker Verification.* Interspeech 2020.
- Chen et al. *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE JSTSP 2022.
- Panayotov, Chen, Povey & Khudanpur. *LibriSpeech: an ASR corpus based on public domain audio books.* ICASSP 2015.
- Yamagishi, Veaux & MacDonald. *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit
  (version 0.92).* University of Edinburgh, 2019.
- Bastianelli, Vanzo, Swietojanski & Rieser. *SLURP: A Spoken Language Understanding Resource Package.* EMNLP 2020.
- Chen et al. *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.* arXiv:2410.06885, 2024.
- Radford et al. *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML 2023.
- Deng, Guo, Xue & Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019
  (the AAM-softmax loss used for fine-tuning).

## Responsible use

VoiceTrap builds hidden voice commands and cloned voices **to measure a defence against them**.

- The attacks only ever target a sandbox. The "tools" record calls and do nothing real.
- The cloned voices copy speakers from SLURP, a public research dataset, and are made only to test whether a
  voiceprint can be fooled. They are never published. If you rerun this, please keep them private too, and never
  clone someone outside a research dataset without their consent.
- Please don't point the attack scripts at a real assistant or device you don't own.

## No warranty

The defence makes mistakes, and the README reports how often. It reduces risk. It doesn't make a voice assistant
safe on its own.

VoiceTrap is an independent research project. It isn't affiliated with or endorsed by Apple, Meta, Microsoft,
OpenAI, Groq or the dataset authors. Their names are used only to say whose work is being used.
