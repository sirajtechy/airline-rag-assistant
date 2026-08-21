# Requirement — Delta Air Lines Customer Support Assistant

## 1. Business context
A Delta customer (or a small business shipping cargo with Delta) needs quick answers about ticket rules, refund eligibility, and shipping policy — questions that today require reading dense tariff PDFs or waiting on hold. A RAG assistant answers instantly with the exact rule cited.

## 2. Objective
Build a RAG chatbot that answers a Delta customer's questions about **passenger ticket rules** (Contract of Carriage) and, separately, **cargo shipping rules** for a business shipper.

## 3. Data provided
| File | Format | Pages | Content |
|---|---|---|---|
| `Delta_contract_of_carriage.pdf` | PDF | 23 | Domestic passenger ticket rules (Jan 2025) |
| `Delta_Cargo_Shipping_Rules_Tariff.pdf` | PDF | 22 | U.S. domestic cargo shipping rules |
| `Delta_Financial_Data_XBRL.xml` + `_Labels_XBRL.xml` | XML | — | Real SEC financial facts (optional side-question source) |

## 4. Functional requirements
1. Chunk and index both PDFs, tagged by business line (passenger vs. cargo).
2. Retrieve and answer strictly from retrieved content, with citation.
3. Correctly route a passenger question to the Contract of Carriage and a shipping question to the Cargo tariff — don't cross-contaminate.

## 5. Guardrails
- Never invent a refund policy, fee, or rule not present in the documents.
- If asked about another airline (JetBlue, United, etc.), decline — out of scope.
- If asked something requiring live booking access ("what's my flight status"), explain the bot only answers general policy questions.

## 6. Acceptance test questions
| # | Question | Expected behavior |
|---|---|---|
| 1 | "What's Delta's policy on oversold flights?" | Answer from Contract of Carriage |
| 2 | "What are the rules for shipping cargo domestically with Delta?" | Answer from Cargo tariff |
| 3 | "Can I get a refund on a non-refundable Delta ticket?" | Answer from Contract of Carriage |
| 4 | "What's JetBlue's baggage policy?" | Bot declines — out of scope |
| 5 | "What's the status of my flight tomorrow?" | Bot explains it can't access live booking data |

## 7. Evaluation-Driven Design Justification (Mandatory)
Every pipeline decision below must be proven with metrics, not assumed. Follow the full methodology in [`../EVALUATION_METHODOLOGY.md`](../EVALUATION_METHODOLOGY.md). Since this group's core skill is passenger-vs-cargo routing, break your Stage 5 (dense/sparse/hybrid) results out by which document each query should hit — that breakdown is your strongest evidence for retrieval config choice.

**Seed evaluation questions** (expand to 20+ per the methodology):
| Question | Ground-truth source |
|---|---|
| "What's Delta's policy on involuntary denied boarding compensation?" | `Delta_contract_of_carriage.pdf` |
| "Can I cancel a non-refundable Delta ticket for a full refund?" | `Delta_contract_of_carriage.pdf` |
| "What are the packaging requirements for shipping cargo with Delta?" | `Delta_Cargo_Shipping_Rules_Tariff.pdf` |
| "What documentation is needed to ship domestically with Delta Cargo?" | `Delta_Cargo_Shipping_Rules_Tariff.pdf` |
| "What's Delta's total operating revenue in their latest 10-K?" | `Delta_Financial_Data_XBRL.xml` |

## 8. Deliverables
- Working chatbot
- **Evaluation report** with all 8 ablation tables from the shared methodology, filled with real measured numbers
- Acceptance test results
- End-to-end RAGAS + DeepEval scores on your final chosen pipeline

## 9. Evaluation criteria (grading weights — see methodology Part E)
| Component | Weight |
|---|---|
| Evaluation rigor: ablation tables complete, real numbers, winner justified per stage | 40% |
| Correct passenger-vs-cargo routing on the 5 acceptance tests | 30% |
| End-to-end RAGAS + DeepEval scores on final pipeline | 20% |
| Code quality / app usability | 10% |
