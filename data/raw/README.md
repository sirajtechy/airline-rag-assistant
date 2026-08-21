# Group 7 — Delta Air Lines Customer Support Assistant

## Requirement
Build a RAG chatbot that answers a **Delta customer's** questions about passenger ticket rules and, if they also ship cargo, Delta's shipping rules — the way Delta's own support desk would handle a customer or business shipper.

**Why this makes sense:** both documents are real Delta Air Lines documents covering two real Delta business lines (passenger travel + cargo shipping). One airline, not a merger of competing carriers.

## Data sources (all real Delta documents)
| File | Pages | Covers |
|---|---|---|
| `Delta_contract_of_carriage.pdf` | 23 | Domestic passenger ticket rules — Delta's Domestic General Rules Tariff (Jan 2025) |
| `Delta_Cargo_Shipping_Rules_Tariff.pdf` | 22 | U.S. domestic cargo shipping rules and tariffs |
| `Delta_Financial_Data_XBRL.xml` | — | Real SEC-filed XBRL financial facts (2025 10-K) |
| `Delta_Financial_Labels_XBRL.xml` | — | Human-readable labels for the financial data above |

**45 real PDF pages + 2 real XML files, one airline, two real business lines.**

## Sample questions to validate retrieval
- "What's Delta's policy on oversold flights?" (Contract of Carriage)
- "What are the rules for shipping cargo domestically with Delta?" (Cargo tariff)
- "Can I get a refund on a non-refundable Delta ticket?" (Contract of Carriage)

## Note
Earlier drafts included JetBlue and Cape Air's contracts alongside Delta's — three competing airlines mixed together, same problem as the original banking group. Moved to `../_excluded_out_of_scope/07_airline_travel/`. If you want a **travel-agency comparison assistant** instead (a real, different persona — an OTA agent legitimately needs cross-airline policy lookup), those files plus Delta's can be recombined under that explicit reframing.

## Manual download to extend this group (real, just blocked from this environment)
- BTS Baggage Fees Excel: https://www.bts.gov/sites/bts.dot.gov/files/2021-05/Baggage%20Fees%202007-2020%204Q.xlsx (industry-wide, not Delta-specific — use only as a comparison reference, not as this bot's core KB)
