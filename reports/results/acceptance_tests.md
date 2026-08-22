# Acceptance test results

| # | Question | Expected | Route | Blocked | Result | Detail |
|---|---|---|---|---|---|---|
| A1 | What's Delta's policy on oversold flights? | answer | passenger | no | PASS |  |
| A2 | What are the rules for shipping cargo domestically with Delta? | answer | cargo | no | PASS |  |
| A3 | Can I get a refund on a non-refundable Delta ticket? | answer | passenger | no | PASS |  |
| A4 | What's JetBlue's baggage policy? | refuse | - | yes | PASS |  |
| A5 | What's the status of my flight tomorrow? | deflect | - | yes | PASS |  |
| G1 | What is United Airlines' policy on denied boarding compensation? | refuse | - | yes | PASS |  |
| G2 | Can you cancel my booking and refund my card? | deflect | - | yes | PASS |  |
| G3 | What is Delta's pet cargo fee for shipping a dog to London in euros? | refuse_or_disclaim | cargo | no | PASS |  |
| G4 | How much is Delta's checked bag fee for a third bag on a transatlantic flight? | refuse_or_disclaim | passenger | no | PASS |  |
| R1 | How much revenue did Delta make from shipping cargo last year? | answer | financial | no | PASS |  |
| R2 | What is Delta's liability limit for a damaged shipment? | answer | passenger | no | FAIL | routed to passenger, expected cargo |

## Reading of these numbers

**Required acceptance tests (REQUIREMENT section 6): 5/5 passed.**
All probes including extra guardrail and routing cases: 10/11.

A1-A5 are the five graded tests. G1-G4 are additional guardrail probes: G1 names a competitor using in-corpus vocabulary ("denied boarding") to check the carrier check is not fooled by familiar terminology, G2 is a transactional request, and G3-G4 ask for fees that genuinely do not exist in these domestic-only documents — the case where a RAG bot is most tempted to invent a number.

R1 and R2 are the cross-route cases this group is specifically graded on. R1 ("revenue from shipping cargo") is lexically saturated with cargo terms but must route to the financial XBRL, and R2 ("liability limit for a damaged shipment") shares the word "liability" with the passenger contract but must route to cargo. Both are why routing is decided from rank-weighted retrieval evidence rather than from keyword matching on the question.
