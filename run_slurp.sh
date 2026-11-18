#!/bin/bash
# Real-owner benchmark. Needs data/slurp_bench (from colab/slurp_attacks.ipynb) and, optionally,
# data/gate_ft/{ecapa_finetuned_best.pt,finetune_history.json} (from colab/real_voice_gate.ipynb).
#
# Resumable, and it waits out Groq's daily free-tier limit by itself: when every key's quota is used up it
# sleeps an hour and tries again, until every clip has run through every defence.
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

remaining() {
  .venv/bin/python - <<'EOF'
import json
from pathlib import Path
from voicetrap.realowners import DEFENCES, ROOT, load_clips
done = set()
p = Path("results/slurp/results.jsonl")
if p.exists():
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    done = {(r["clip"], r["defence"]) for r in rows}
print(sum((c.id, d) not in done for c in load_clips(ROOT) for d in DEFENCES))
EOF
}

while true; do
  left=$(remaining)
  echo "$(date '+%F %T')  runs left: $left" | tee -a results/slurp.log
  [ "$left" = "0" ] && break
  .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from voicetrap.realowners import bench
bench()
" 2>&1 | grep -v -i warn | tee -a results/slurp.log
  [ "$(remaining)" = "0" ] && break
  echo "$(date '+%F %T')  stopped early (most likely the daily free quota); retrying in an hour" | tee -a results/slurp.log
  sleep 3600
done
.venv/bin/voicetrap report --run slurp > /dev/null
echo "$(date '+%F %T')  ALL DONE" | tee -a results/slurp.log
