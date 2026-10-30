# Model card: ShopSense

## What the system is
A review question-answering system. It uses two **pre-trained** models that I did not train or fine-tune:

| Model | Used for |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (384-dimension embeddings) | turning reviews and questions into vectors for search |
| `llama3.2:3b` through Ollama, temperature 0 | writing an answer from at most 6 review excerpts |

Everything else (cleaning, search, checks, safety, agent, tests) is my own code.

## Intended use
Exploring what reviewers say about a product, with citations a person can check. A learning and portfolio project.

## Not intended for
- Decisions where a wrong citation matters (legal, medical, safety, purchase decisions on their own).
- Products or categories I did not test (only Amazon All Beauty).
- Reading sarcasm, negation or exact quantities from text.

## Data
A 49,877-review sample of Amazon All Beauty reviews; 49,677 after cleaning (reviews under 5 words removed, HTML and media tags stripped, heavy reviewers and duplicates handled);
50,524 passages (reviews over 300 words split into 200-word chunks with 40-word overlap). Median 1 review per product; 551 products have 10 or more reviews.
Product-level answers are only given for products with at least 10 clean reviews.

## Evaluation
Test set: 67 questions (factual, numeric, aspect summary, unanswerable, comparison, out of scope), 19 scorable for search.
Relevance labels were drafted by an AI and reviewed by me; a 43-row spot check showed 44% agreement between my labels and the AI's.
Metrics, results and their limits are in `README.md` and `configs/kpi_dictionary.yaml`.

## Known limitations
See `reports/failure_analysis.md`. The most important:
- The model often attaches the wrong citation number to a correct fact (hand read: about 2 of 10 answers fully correct under a strict rule).
- It can invent a small inference that no review states.
- 2 of 3 test questions with no evidence were answered instead of refused.
- Share-of-reviews answers are keyword counts and cannot read negation.
- The input guard catches few unseen attack wordings; protection comes mainly from limiting what an attack can do (read-only tools, output cleaning, cited and checked answers).
- About 27 seconds per answer on a CPU laptop.

## Safety and privacy
Reviews are written by real people. The guard hides emails, phone numbers and links before text reaches the model and removes them from answers.
The model runs locally, so no review text is sent to a third party. In 15 prompt-injection tests the model followed none of the injected instructions;
this is a small sample and does not prove safety.

## Maintenance
No monitoring beyond a request log (`data/logs/requests.jsonl`, summarised by `src/shopsense/monitoring/log_report.py`). No retraining is involved.
