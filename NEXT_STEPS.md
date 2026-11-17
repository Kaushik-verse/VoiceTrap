# Where VoiceTrap stands (saved before shutdown)

Done
- Agent, attacks, 5 defences, scoring, report: `src/voicetrap/`
- 75 benchmark clips (single owner): `data/clips`, results so far: `results/main/results.jsonl`
- Real voices downloaded: `data/librispeech` (Mini LibriSpeech, CC BY 4.0)
- Synthetic voice corpus: `data/corpus/tts` (built in parallel; resumable)
- Trainable speaker-gate scorer: `src/voicetrap/gatetrain.py`; multi-owner scenarios: `scenarios.matrix_v2`

Resume (each step skips work already done)
1. Finish the voice corpus and write the speaker manifest:
   `uv run python -c "from voicetrap.corpus import build; s=build(); print(len(s))"`
2. Train + test the gate on speaker-disjoint splits (offline, no API):
   `uv run python -c "from voicetrap.gatetrain import run_experiment; run_experiment()"`
3. Finish the single-owner benchmark (needs internet + Groq):
   `uv run voicetrap bench --run main && uv run voicetrap report --run main`
4. Multi-owner end-to-end benchmark with the trained gate (to be wired up next).
5. README with real numbers.
