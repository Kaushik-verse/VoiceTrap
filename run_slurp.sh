#!/bin/sh
# Real-owner benchmark. Needs data/slurp_bench (from colab/slurp_attacks.ipynb) and, optionally,
# data/gate_ft/{ecapa_finetuned_best.pt,finetune_history.json} (from colab/real_voice_gate.ipynb).
# Resumable: run it again after a stop or a daily Groq limit and it continues where it left off.
set -e
export PYTHONUNBUFFERED=1
uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from voicetrap.realowners import bench
bench()
" 2>&1 | grep -v -i warn | tee -a results/slurp.log
uv run voicetrap report --run slurp > /dev/null
echo ALL DONE
