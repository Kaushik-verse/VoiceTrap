# Status

Done (reproduce with `./run_all.sh`)
- Verified voice corpus: 219 speakers (62 macOS voice families + 54 LibriSpeech readers), 2,446 synthetic clips
- Cross-validated speaker gate: `results/gate/metrics_v2.json`, `results/gate_v2.log`
- Single-owner benchmark, 375 runs: `results/main/summary.md`
- Three-owner benchmark, 540 runs: `results/v2/summary.md`
- README with final numbers

Ideas
- Play attacks through a real speaker and record them in a room
- Ask for confirmation on read_messages (every remaining Shield hijack)
- Phone-channel augmentation when enrolling the owner
