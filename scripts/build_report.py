"""Assemble EVALUATION_REPORT.md from the measured stage artefacts.

The narrative lives in reports/REPORT_TEMPLATE.md and the tables are injected from
reports/results/*.md, which are written by the ablation scripts themselves. Tables
are therefore never retyped, so the report cannot drift from the numbers that were
actually measured.

Placeholders are `{{stage1_parsing}}` etc., matching the artefact filenames. A
missing artefact is reported loudly rather than silently rendered blank.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
TEMPLATE = ROOT / "reports" / "REPORT_TEMPLATE.md"
OUTPUT = ROOT / "EVALUATION_REPORT.md"

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def stage_body(name: str) -> str:
    """Return a stage artefact with its H1 demoted, so it nests under the report."""
    path = RESULTS / f"{name}.md"
    if not path.exists():
        return f"> **MISSING ARTEFACT `{name}.md`** — run the corresponding script."
    lines = path.read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("# "):
            continue  # the template supplies its own heading
        if line.startswith("## "):
            line = "#" + line  # -> ###
        out.append(line)
    return "\n".join(out).strip()


def main() -> int:
    if not TEMPLATE.exists():
        print(f"missing template: {TEMPLATE}")
        return 1

    text = TEMPLATE.read_text()
    missing: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group(1)
        body = stage_body(name)
        if body.startswith("> **MISSING"):
            missing.append(name)
        return body

    rendered = PLACEHOLDER.sub(replace, text)
    OUTPUT.write_text(rendered)

    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(rendered.splitlines())} lines)")
    if missing:
        print(f"WARNING: {len(missing)} missing artefact(s): {sorted(set(missing))}")
        return 1
    print("all stage artefacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
