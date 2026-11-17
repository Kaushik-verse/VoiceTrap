"""Turn results.jsonl into summary.md and two charts (SVG)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from voicetrap.bench import load_results
from voicetrap.pipeline import DEFENCES
from voicetrap.scenarios import LEVELS_DB

LABEL = {"none": "No defence", "prompt": "Prompt rule", "promptguard": "Prompt Guard 2",
         "speaker": "Speaker gate", "shield": "VoiceTrap Shield"}
COLOR = {"none": "#B4552D", "prompt": "#D99A6C", "promptguard": "#C9B79C", "speaker": "#7FA1C9",
         "shield": "#1E4D8C"}


def _rate(rows, key) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return None if not vals else sum(bool(v) for v in vals) / len(vals)


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["defence"]].append(r)
    table = {}
    for d in DEFENCES:
        rs = by.get(d, [])
        if not rs:
            continue
        att = [r for r in rs if r["kind"] == "attack"]
        table[d] = {
            "hijack_attack": _rate(att, "hijacked"),
            "hijack_clone": _rate([r for r in rs if r["kind"] == "clone"], "hijacked"),
            "hijack_background": _rate([r for r in rs if r["kind"] == "background"], "hijacked"),
            "utility_clean": _rate([r for r in rs if r["kind"] == "clean"], "utility"),
            "utility_under_attack": _rate(att + [r for r in rs if r["kind"] == "clone"], "utility"),
            "attack_heard": _rate(att, "attack_in_transcript"),
            "reached_owner": _rate([r for r in rs if r["kind"] != "clean"], "attack_reached_owner"),
            "n": len(rs),
            "by_level": {lvl: _rate([r for r in att if r["level_db"] == lvl], "hijacked") for lvl in LEVELS_DB},
        }
    return table


def chart_levels(table: dict) -> str:
    W, H, L, R, T, B = 1280, 420, 110, 300, 90, 70
    pw, ph = W - L - R, H - T - B
    xs = {lvl: L + i * pw / (len(LEVELS_DB) - 1) for i, lvl in enumerate(LEVELS_DB)}
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'role="img" aria-label="Hijack rate by attacker loudness for each defence">',
         f'<rect width="{W}" height="{H}" rx="22" fill="#F7F6F2"/>',
         '<g font-family="-apple-system,\'Segoe UI\',Helvetica,Arial,sans-serif">',
         '<text x="48" y="50" font-family="Georgia,serif" font-size="24" fill="#1A1C1E">'
         'How often the hidden command took over</text>',
         '<text x="48" y="74" font-size="14" fill="#5E6368">Hijack rate by how loud the hidden command is, '
         'relative to the owner\'s voice</text>']
    for k in range(0, 101, 25):
        y = T + ph - ph * k / 100
        s.append(f'<line x1="{L}" y1="{y}" x2="{L + pw}" y2="{y}" stroke="#E6E3DB"/>')
        s.append(f'<text x="{L - 12}" y="{y + 4}" font-size="12" fill="#8A8F94" text-anchor="end">{k}%</text>')
    for lvl, x in xs.items():
        s.append(f'<text x="{x}" y="{T + ph + 26}" font-size="13" fill="#5E6368" text-anchor="middle">'
                 f'{lvl:+d} dB</text>')
    s.append(f'<text x="{L + pw / 2}" y="{T + ph + 50}" font-size="12" fill="#8A8F94" text-anchor="middle">'
             'quieter than the owner  →  as loud as the owner</text>')
    for i, (d, row) in enumerate(table.items()):
        pts = [(xs[lvl], T + ph - ph * (row["by_level"][lvl] or 0)) for lvl in LEVELS_DB
               if row["by_level"].get(lvl) is not None]
        if not pts:
            continue
        path = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f} {y:.1f}" for j, (x, y) in enumerate(pts))
        w = 3.5 if d == "shield" else 2.2
        s.append(f'<path d="{path}" fill="none" stroke="{COLOR[d]}" stroke-width="{w}"/>')
        s += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLOR[d]}"/>' for x, y in pts]
        ly = T + 18 + i * 30
        s.append(f'<rect x="{W - R + 40}" y="{ly - 10}" width="22" height="6" rx="3" fill="{COLOR[d]}"/>')
        s.append(f'<text x="{W - R + 72}" y="{ly - 2}" font-size="14" fill="#1A1C1E">{LABEL[d]}</text>')
    s.append("</g></svg>")
    return "\n".join(s)


def build_report(out: Path) -> str:
    rows = load_results(out)
    table = summarize(rows)
    cal = json.loads((out / "calibration.json").read_text()) if (out / "calibration.json").exists() else {}
    lines = ["| Defence | Hijacked (hidden command) | Hijacked (cloned voice) | Hijacked (TV only, owner silent) "
             "| Owner's request done (clean) | Owner's request done (under attack) | Hidden command transcribed |",
             "|---|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for d, r in table.items():
        lines.append(f"| {LABEL[d]} | **{_pct(r['hijack_attack'])}** | {_pct(r['hijack_clone'])} | "
                     f"{_pct(r['hijack_background'])} | {_pct(r['utility_clean'])} | "
                     f"{_pct(r['utility_under_attack'])} | {_pct(r['attack_heard'])} |")
    md = "\n".join(lines)
    if cal:
        md += (f"\n\nSpeaker gate threshold {cal['threshold']}: owner windows accepted {_pct(cal['owner_accept'])}, "
               f"unseen voices accepted {_pct(cal['other_accept'])}.")
    (out / "summary.md").write_text(md + "\n")
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    (out / "hijack_by_level.svg").write_text(chart_levels(table))
    return md
