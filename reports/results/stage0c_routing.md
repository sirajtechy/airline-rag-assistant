# Stage 0c — Routing policy

| Routing policy | Route acc (30 eval questions) | Route acc (all 33 cases) | Misroutes |
|---|---|---|---|
| top-1 chunk | 1.0000 | 0.9714 | 1 |
| rank-weighted vote (top-5) | 0.9667 | 0.9143 | 3 |
| vote + defined-term prior | 0.9667 | 0.9143 | 3 |

## Reading of these numbers

Winner: **top-1 chunk** — 0.9714 across all 33 cases.

The prior is built from **68 terms each document uniquely defines** ({'passenger': 24, 'cargo': 44}), extracted from the definitions sections rather than hand-written. Terms both documents define — 'Carrier' most importantly — are discarded because they carry no signal.

Why a prior is needed at all: retrieval evidence alone cannot separate "What is Delta's liability limit for a damaged shipment?" The passenger contract's baggage-liability clause is genuinely the most topically similar text in the corpus, and it even names a dollar figure ($4,700 per passenger), so a retriever is right to surface it. The distinguishing signal is not similarity but vocabulary: 'shipment' is a term the cargo tariff defines and the passenger contract does not.

The prior is weighted at 0.35 of the retrieval evidence, so it breaks ties rather than overriding retrieval. Keyword routing *instead of* retrieval would fail the opposite case — R1, "revenue from shipping cargo", is saturated with cargo vocabulary but must route to the financial filings.

**top-1 chunk** misroutes:
- "What is Delta's liability limit for a damage" -> passenger (want cargo)
**rank-weighted vote (top-5)** misroutes:
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- "What is Delta's liability limit for a damage" -> passenger (want cargo)
**vote + defined-term prior** misroutes:
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- "What is Delta's liability limit for a damage" -> passenger (want cargo)
