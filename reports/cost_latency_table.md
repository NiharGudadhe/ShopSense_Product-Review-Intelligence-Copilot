# Cost and latency

Hardware: laptop CPU only. Model: `llama3.2:3b` through Ollama (temperature 0, up to 250 answer tokens).

| Measure | Value | Source |
|---|---|---|
| Money cost per question | 0 (local model). Electricity and my own time are not counted | design |
| Average time over 39 questions | 27.8 s | notebook 06 |
| Average, new prompt, hybrid search, 22 aspect questions | 24.0 s | notebook 07 |
| Average, new prompt, vector search, 39 questions | 27.0 s | notebook 07 |
| Answered questions (24 in notebook 06) | median 32.2 s, range 16.3 to 102.1 s | notebook 06 |
| Refused questions (15 in notebook 06) | median 12.6 s, range 8.4 to 52.6 s | notebook 06 |
| First question after starting | 102 s (the model is loading) | notebook 06 |
| Rating and share questions (agent) | well under 1 s (pandas, no model) | notebook 10 (not timed separately) |
| Comparison question | about two answers long (two model calls) | notebook 10 |

What drives the time: reading the six review excerpts (prompt processing on CPU) and writing about 60 to 130 tokens.
Search itself takes a fraction of a second. A 3B model on CPU is slow; a hosted model would answer in a few seconds but costs money per question
and sends review text to a third party.

How to measure it from now on: run the API and then `python -m shopsense.monitoring.log_report`. It prints requests, refusal share and median and 95th percentile seconds per route.
