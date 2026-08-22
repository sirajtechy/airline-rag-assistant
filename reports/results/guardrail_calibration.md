# Guardrail — abstain threshold calibration

| Threshold | Answerable correctly answered | Unanswerable correctly refused | Sum |
|---|---|---|---|
| 0.5500 | 1.0000 | 0.0000 | 1.0000 |
| 0.5700 | 1.0000 | 0.0000 | 1.0000 |
| 0.5900 | 1.0000 | 0.1250 | 1.1250 |
| 0.6100 | 1.0000 | 0.1250 | 1.1250 |
| 0.6300 | 1.0000 | 0.1250 | 1.1250 |
| 0.6500 | 1.0000 | 0.1250 | 1.1250 |
| 0.6700 | 1.0000 | 0.2500 | 1.2500 |
| 0.6900 | 0.9333 | 0.3750 | 1.3083 |
| 0.7100 | 0.9333 | 0.5000 | 1.4333 |
| 0.7300 | 0.7333 | 1.0000 | 1.7333 |
| 0.7500 | 0.5000 | 1.0000 | 1.5000 |
| 0.7700 | 0.4333 | 1.0000 | 1.4333 |
| 0.7900 | 0.3667 | 1.0000 | 1.3667 |
| 0.8100 | 0.2333 | 1.0000 | 1.2333 |
| 0.8300 | 0.1333 | 1.0000 | 1.1333 |
| 0.8500 | 0.0333 | 1.0000 | 1.0333 |
| 0.8700 | 0.0333 | 1.0000 | 1.0333 |

## Reading of these numbers

**Chosen threshold: 0.67** — answers 100% of answerable questions while refusing 25% of unanswerable ones.

**This gate is the coarse outer layer of a two-layer defence, and is tuned accordingly.** The precise layer is the system prompt, which makes the model decline from context it can actually read. Acceptance probe G4 demonstrates it working on a question with *high* retrieval confidence: asked for a transatlantic third-bag fee, the bot answered "Delta's published documents do not cover fees for checked bags on transatlantic flights ... domestic travel only [1]" rather than inventing one. Cosine similarity could never have caught that, because the retrieved baggage rules genuinely are the most topically similar text in the corpus — they simply do not contain the answer.

That is why the threshold is set to reject **no** answerable question rather than to maximise refusals. A more aggressive setting (0.73 refuses 100% of the unanswerable probes) would also refuse 27% of questions the pipeline answers correctly, trading real utility for refusals the prompt layer already handles better.

The gate runs on the **dense cosine similarity** of the best retrieved chunk, not the fused RRF score. RRF scores are a function of rank position and carry no absolute meaning — a score of 0.03 means nothing on its own and is not comparable between queries — so no fixed threshold on it could be meaningful. Cosine similarity is bounded in [0, 1] and comparable across queries, which is what a confidence floor requires.

The two populations **overlap**: the hardest answerable question scores 0.679 while the most confidently-retrieved unanswerable probe scores 0.726. No threshold separates them perfectly, so this is an explicit trade-off, not a solved problem. It is tuned toward refusing, because inventing a baggage fee is a worse failure for a support bot than declining a question it could have answered.

The unanswerable probes are plausible Delta questions whose answers are genuinely outside these three documents (international fees, SkyMiles earning, lounge hours, aircraft configuration). They deliberately avoid competitor names and 'my booking' phrasing so they exercise this gate rather than the deterministic pre-retrieval guards.
