"""voicetrap command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> None:
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser(prog="voicetrap", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="synthesize the benchmark clips (macOS voices)")
    b = sub.add_parser("bench", help="run every clip through every defence")
    b.add_argument("--run", default="main")
    b.add_argument("--defences", default="", help="comma list, default all")
    b.add_argument("--kinds", default="", help="clean,attack,clone,background")
    r = sub.add_parser("report", help="write summary.md and charts")
    r.add_argument("--run", default="main")
    t = sub.add_parser("try", help="run one audio file through a defence")
    t.add_argument("audio")
    t.add_argument("--defence", default="none")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from voicetrap.scenarios import build

        clips = build()
        print(f"{len(clips)} clips in data/clips, enrolment in data/enroll")
    elif a.cmd == "bench":
        from voicetrap.bench import run_bench

        run_bench(Path("results") / a.run, defences=[d for d in a.defences.split(",") if d] or None,
                  kinds=[k for k in a.kinds.split(",") if k] or None)
    elif a.cmd == "report":
        from voicetrap.report import build_report

        print(build_report(Path("results") / a.run))
    elif a.cmd == "try":
        from voicetrap import audio as A
        from voicetrap.bench import make_gate
        from voicetrap.groq import Groq
        from voicetrap.pipeline import run

        gate = make_gate(Path("data"), Path("results/main")) if a.defence in ("speaker", "shield") else None
        o = run(A.load(a.audio), a.defence, Groq(), gate=gate, approve=lambda call: False)
        print(json.dumps({"transcript": o.transcript, "tool_calls": o.tool_calls,
                          "executed": o.executed, "blocked": o.blocked, "note": o.note}, indent=2))


if __name__ == "__main__":
    main()
