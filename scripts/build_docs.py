"""Assemble documentation by pulling metrics JSON and plot paths into Markdown.

Expected metrics files:
- models/fraud/metrics.json
- models/taxi/metrics.json
- models/imdb/metrics.json

Outputs:
- docs/metrics_summary.md
"""
import json, pathlib
from datetime import datetime

BASE = pathlib.Path(__file__).resolve().parents[1]
OUT = BASE / 'docs' / 'metrics_summary.md'

pairs = {
    'fraud': BASE / 'models' / 'fraud' / 'metrics.json',
    'taxi': BASE / 'models' / 'taxi' / 'metrics.json',
    'imdb': BASE / 'models' / 'imdb' / 'metrics.json',
}

lines = [f"# Metrics Summary\n", f"_Generated: {datetime.utcnow().isoformat()}Z_\n\n"]
for name, p in pairs.items():
    if p.exists():
        try:
            js = json.loads(p.read_text())
        except Exception as e:
            js = {"error": str(e)}
        lines.append(f"## {name.title()}\n\n```json\n{json.dumps(js, indent=2)}\n```\n\n")
    else:
        lines.append(f"## {name.title()}\n\n_No metrics.json found at {p}_\n\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
