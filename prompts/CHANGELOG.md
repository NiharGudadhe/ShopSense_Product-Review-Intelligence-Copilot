# Prompt versions

The prompt the code uses is `SYSTEM_MESSAGE` in `src/shopsense/llm/answer.py`. The files here are copies for reading
and history. `tests/test_prompts.py` fails if `answer_v2.txt` stops matching the code, so the copy cannot drift.

| Version | When | What changed | Measured effect |
|---|---|---|---|
| v1 | notebooks 04 to 06 | first prompt: cite every sentence, refuse exactly, under 120 words | 12 of 24 answers looked uncited (half was a checker bug, see below) |
| v2 | notebook 07 | write citations as `[1][2]` never `[1, 2]`; cite an excerpt only if it is about the topic asked; do not guess what a reviewer meant; under 100 words | **no improvement** in rule-breaking (6 of 19 before, 8 of 19 hybrid, 11 of 21 vector). Code now removes uncited sentences instead of trusting the prompt |

Lesson: the prompt change did not fix citation discipline in a 3B model. The fixes that worked were in code
(checker fix, removing uncited sentences, quote check, vector search).

Prompts for extraction and for an LLM judge were in the first plan and were not built, so there are no files for them.
