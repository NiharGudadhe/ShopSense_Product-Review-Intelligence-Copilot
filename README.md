# ShopSense: ask questions about product reviews, with citations

A question-answering tool over Amazon product reviews that runs **entirely on a laptop**.
Averages and shares are **calculated by code**. Opinion questions are answered by a small local language model from the most
relevant reviews, with numbered citations. The project is mostly about **measuring where such a system fails and making it safer**.

**Author:** NIHAR GUDADHE  | **Live demo:** link

> I did **not train a model** here. ShopSense is a retrieval-augmented generation (RAG) system built on two pre-trained models
> (MiniLM for search, `llama3.2:3b` for writing). My trained models are in my other project.

## What it does

```
question (+ product) -> agent -> rating   : average and star shares, calculated by code
                              -> share    : "what share of 1-2 star reviews mention shipping" (keyword count by code)
                              -> compare  : summary for each product + both average ratings
                              -> reviews  : search -> small local model -> answer with [1][2] citations -> code checks
                              -> refuse   : no product chosen
```

Search is vector search over 50,524 review passages (MiniLM embeddings in LanceDB). Hybrid (keyword + vector) search is also implemented
and was compared; the app uses vector search.

## Results

Data: a 49,877-review sample of Amazon All Beauty reviews; 49,677 after cleaning; 50,524 searchable passages.
Test set: 67 questions in 6 types, 19 of them scorable for search. Relevance labels were **drafted by an AI and reviewed by me**
(on a 43-row spot check my labels and the AI's agreed on 44%; I later accepted the AI labels).

| What | Result |
|---|---|
| Search, found a relevant review in the top 5 | vector 19 of 19, keyword and hybrid 16 of 19 |
| Search ranking quality (nDCG@5) | vector 0.673, hybrid 0.582, keyword 0.477. Hybrid beat keyword (interval +0.010 to +0.203); hybrid vs vector is **not statistically clear** (-0.219 to +0.044) |
| Questions with no answer in the reviews, refused | 17 of 17 (12 unanswerable, 5 out of scope); the 5 out-of-scope ones were also refused with a product selected |
| Wrong refusals (evidence existed) | 0 with vector search, 2 with hybrid |
| Citations to a review number that does not exist | 0 |
| Hand read of 10 answers | every claim present in the retrieved reviews: 8 of 10 (my reading); counting only claims found in the review the answer cites: about 2 of 10 (a stricter reading). Typical error: right fact, wrong citation number |
| Questions with no evidence that were still answered | 2 of 3 |
| Prompt-injection test: poisoned review in the prompt | 0 of 15 obeyed (guard off and on). Guard caught 7 of 8 known and 1 of 8 new attacks |
| Honest reviews wrongly dropped by the guard | 2 of 50,524 (0.004%); 74 before narrowing its patterns |
| Routing questions to the right tool | 67 of 67 on golden questions (rules were written around them), 14 of 18 on rephrased questions |
| Keyword share, matched reviews that are real shipping complaints | 7 of 10 (small sample) |
| Speed | about 27 s per answered question on CPU |

Full numbers and what each one means: `configs/kpi_dictionary.yaml`. Every known failure: `reports/failure_analysis.md`.

## What I fixed because of the measurements

- My first citation checker only read `[1]`, not `[1, 2]`. "12 of 24 answers uncited" became 6 of 19 after the fix.
- Hybrid search missed relevant reviews and caused 2 wrong refusals; vector search did not.
- A prompt change did **not** make the small model cite better. Code now removes any sentence without a citation.
- A check that quoted words appear in the cited review fixed wrong numbers in 2 of 2 visible cases. A similarity-based version made things worse, so I switched it off.
- The guard flagged 74 honest reviews; narrowing its patterns brought that to 2.

## Not built (planned at the start)

Aspect extraction, a suspicious-review model, an LLM judge, and a LoRA fine-tune. I chose to spend the effort on evaluation, safety and the agent.

## Demo

The online demo cannot run the language model, so it shows **answers saved from real runs on my laptop** (`app/demo_answers.json`).
Live mode needs the API and Ollama running locally.

## Project layout

```
configs/      settings and the metric dictionary
prompts/      answer prompt versions and a changelog
notebooks/    01-10, in build order (audit, cleaning, search, answers, golden set, evaluation, fixes, safety, agent)
src/shopsense/
  retrieval/  hybrid.py: BM25 + vector search
  llm/        answer.py (prompt, checks), attribute.py (quote check)
  safety/     guard.py: injection filter, personal data hiding, answer cleaning
  agent/      graph.py (LangGraph routes), tools.py (read-only tools)
  evaluation/ metrics.py (tested copies of the metric formulas)
  api/        FastAPI service
  monitoring/ log_report.py (request log summary)
app/          Streamlit app and saved demo answers
tests/        48 tests
reports/      failure analysis, cost and latency, stakeholder memo, model card, evaluation tables
docker/ and .github/workflows/   written
```

## Data

Amazon Reviews dataset (All Beauty category), McAuley Lab, University of California San Diego. Used for research and learning. The data is not
included in this repository. The demo file contains short excerpts of review text; check the dataset's terms before making the repository public.
