# ShopSense: what it can and cannot do (one page)

**What it is.** A question-answering tool over product reviews (Amazon All Beauty, 49,677 clean reviews after cleaning, 50,524 searchable passages). It runs entirely on a laptop: no data leaves the machine and there is no per-question cost.

**How it works.** The question is sent to the right tool.
- Averages and shares are **calculated by code**.
- "What do reviewers say about X" questions use search to find the most relevant reviews, then a small local model (3 billion parameters) summarises them with numbered citations.
- Comparisons run the summary for both products.
- Questions without a product chosen are not answered.

**What the tests show (small samples)**
- **Search:** vector search found a relevant review for all 19 test questions. Keyword search and the hybrid method found one for 16 (0.842). Hybrid clearly beat keyword search; hybrid versus vector is not statistically clear.
- **Refusals:** all 17 questions with no answer in the reviews (12 unanswerable, 5 out of scope) were refused. All 5 out-of-scope questions were also refused with a product selected.
- **Citations:** no answer cited a review number that did not exist. But a valid number is not always the right review. In a hand read of 10 answers, every claim could be found in the retrieved reviews in 8 (my loose reading); counting only claims found in the review that the answer actually cites, about 2 of 10 were fully correct (Claude's strict reading). A code check now fixes citations for quoted phrases only.
- **Numbers:** the share of low-star reviews mentioning shipping is a keyword count. In a 10-review check, 7 of 10 matches were real shipping complaints.
- **Safety:** 15 poisoned reviews (8 known, 7 new) were all ignored by the model, with and without the guard. The input guard caught 7 of 8 known attacks but only 1 of 8 new ones, so I do not rely on it alone, and after I narrowed its patterns it wrongly drops 2 of 50,524 real reviews (0.004%); it dropped 74 before.
- **Speed:** about 27 seconds per answered question on an 8 GB CPU laptop (see `cost_latency_table.md`).

**What it should not be used for yet**
- Decisions that need exact quotes or exact citation numbers: the number can point to the wrong review.
- Questions whose answer is not in the reviews: it usually says so, but 2 of 3 no-evidence test questions were answered anyway.
- Anything needing a read of negation or sarcasm in counts (the shipping share is a keyword match).

**Known limits.** 19 scorable test questions, one product category, relevance labels drafted by an AI and reviewed by the project owner, 10 hand-read answers, rules for routing that break on unusual wording. Full list: `failure_analysis.md`.

**Suggested next steps.** Add a bigger answer model or a second checking model; read 50 or more answers instead of 10; label the retrieval set by hand; add a trained injection detector; add a negation-aware way to count shipping complaints.
