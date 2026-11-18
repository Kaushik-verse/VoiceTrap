#!/bin/sh
# Full reproducible pipeline, one heavy step at a time (8 GB RAM friendly).
set -e
export PYTHONUNBUFFERED=1
uv run python -c "from voicetrap.corpus import build; from collections import Counter; s=build(); print(len(s), Counter((x.source, x.split) for x in s))"
uv run python -c "
from voicetrap.gatetrain import run_experiment2
import json
r = run_experiment2()
print(json.dumps({k: r[k] for k in ('selected', 'operating_point', 'cv_eer')}, indent=1))
for k, v in r['test'].items():
    print(k, json.dumps({x: v[x] for x in v if x != 'by'}))
" 2>&1 | grep -v -i warn > results/gate_v2.log
uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from voicetrap.v2 import bench
bench()
" > results/v2.log 2>&1
uv run voicetrap bench --run main > results/main.log 2>&1
uv run voicetrap report --run v2 > /dev/null
uv run voicetrap report --run main > /dev/null
echo ALL DONE
