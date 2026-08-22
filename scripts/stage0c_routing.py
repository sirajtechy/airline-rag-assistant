"""Stage 0c — routing policy, measured.

Routing is 30% of this group's grade, so the policy gets its own ablation rather
than being asserted. Three policies are compared on route accuracy over the 30
labelled questions plus the 3 answer-expecting acceptance cases:

  top1        the single best-retrieved chunk decides
  vote        rank-weighted vote over the top 5 (1/rank)
  vote+prior  the vote, plus a prior from the documents' own defined vocabulary

The prior is extracted, not authored: both tariffs open with a definitions
section (Rule 3 in the passenger contract, G2 in the cargo tariff) in which each
document declares the terms it governs. Terms defined by both are discarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag.ablation import save_stage
from delta_rag.config import EVAL_DIR, SOURCE_DOCS
from delta_rag.evalset import load_eval_set
from delta_rag.pipeline import DeltaSupportBot
from delta_rag.routing import defined_terms, route_from_hits


def main() -> None:
    bot = DeltaSupportBot()
    questions = load_eval_set()

    spec = yaml.safe_load((EVAL_DIR / "acceptance_tests.yaml").read_text())
    extra = [
        (t["question"], t["expect_route"])
        for t in spec["tests"]
        if t["behaviour"] == "answer" and t.get("expect_route")
    ]
    cases = [(q.question, q.route) for q in questions] + extra

    terms = defined_terms()
    by_line: dict[str, int] = {}
    for line in terms.values():
        by_line[line] = by_line.get(line, 0) + 1
    print(f"extracted {len(terms)} uniquely-defined terms: {by_line}")
    for line in sorted(by_line):
        sample = [t for t, l in terms.items() if l == line][:8]
        print(f"   {line}: {sample}")

    retrieved = {}
    for question, _ in cases:
        retrieved[question] = bot.retriever.retrieve(
            question, k=15, mode=bot.config.mode, fusion=bot.config.fusion,
            alpha=bot.config.alpha,
        ).hits

    policies = {
        "top-1 chunk": lambda q, h: (
            SOURCE_DOCS[h[0].chunk.doc_id]["business_line"] if h else None
        ),
        "rank-weighted vote (top-5)": lambda q, h: route_from_hits(h).business_line,
        "vote + defined-term prior": lambda q, h: route_from_hits(
            h, question=q
        ).business_line,
    }

    rows, misses = [], {}
    for name, fn in policies.items():
        correct = eval_correct = 0
        wrong = []
        for i, (question, gold) in enumerate(cases):
            got = fn(question, retrieved[question])
            if got == gold:
                correct += 1
                if i < len(questions):
                    eval_correct += 1
            else:
                wrong.append(f"{question[:44]!r} -> {got} (want {gold})")
        rows.append({
            "Routing policy": name,
            "Route acc (30 eval questions)": round(eval_correct / len(questions), 4),
            "Route acc (all 33 cases)": round(correct / len(cases), 4),
            "Misroutes": len(cases) - correct,
        })
        misses[name] = wrong

    best = max(rows, key=lambda r: r["Route acc (all 33 cases)"])
    detail = "\n".join(
        f"**{name}** misroutes:\n" + ("\n".join(f"- {w}" for w in w_list) or "- none")
        for name, w_list in misses.items()
    )
    save_stage(
        "stage0c_routing", "Stage 0c — Routing policy", rows,
        notes=(
            f"Winner: **{best['Routing policy']}** — "
            f"{best['Route acc (all 33 cases)']:.4f} across all 33 cases.\n\n"
            f"The prior is built from **{len(terms)} terms each document uniquely "
            f"defines** ({by_line}), extracted from the definitions sections rather than "
            "hand-written. Terms both documents define — 'Carrier' most importantly — are "
            "discarded because they carry no signal.\n\n"
            "Why a prior is needed at all: retrieval evidence alone cannot separate "
            "\"What is Delta's liability limit for a damaged shipment?\" The passenger "
            "contract's baggage-liability clause is genuinely the most topically similar "
            "text in the corpus, and it even names a dollar figure ($4,700 per "
            "passenger), so a retriever is right to surface it. The distinguishing signal "
            "is not similarity but vocabulary: 'shipment' is a term the cargo tariff "
            "defines and the passenger contract does not.\n\n"
            "The prior is weighted at 0.35 of the retrieval evidence, so it breaks ties "
            "rather than overriding retrieval. Keyword routing *instead of* retrieval "
            "would fail the opposite case — R1, \"revenue from shipping cargo\", is "
            "saturated with cargo vocabulary but must route to the financial filings.\n\n"
            + detail
        ),
    )
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
