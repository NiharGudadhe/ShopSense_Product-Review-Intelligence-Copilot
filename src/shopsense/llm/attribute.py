"""Put the right citation numbers on an answer, using code instead of trusting a small model.

Why: a 3B model writes mostly true sentences but often attaches the wrong [n]. We check each cited clause:
    1. Quoted words: if the clause quotes a phrase ("all day"), the cited review must contain that phrase.
       Otherwise the citation moves to the review that has it. A quote found in no review removes the sentence.
    2. Meaning: the clause is compared with each excerpt (sentence embeddings). A model-cited number is kept if it fits
       about as well as the best excerpt, otherwise it is replaced by the best one. A sentence in which no clause
       fits any excerpt is removed.
A sentence that uses a number inside the wording ("reviewers like [4] and [5] say...") is left untouched and counted
as not_checked, because moving that number would break the sentence.

Use:  new_answer, report = reattribute(answer, hits["text"].tolist(), embedder)
embedder is any object with .encode(list_of_text, normalize_embeddings=True) (the MiniLM model from load_index()).
"""
import re

import numpy as np

GROUP = re.compile(r"(?:\s*\[\d+\])+")                 # one or more citations in a row: [1][2]
QUOTE = re.compile(r"[\"“]([^\"”]{3,80})[\"”]")
NOUN_USE = re.compile(r"\b(?:like|reviewers?|and|or|by|from|as|of|to|with|that|says?|states?)\s*$", re.I)
FLOOR = 0.20      # a clause whose best excerpt scores below this is counted as weak (kept, but reported)
MARGIN = 0.10     # a model-cited number is kept if it scores within this of the best excerpt


def words_of(text):
    """Lower-case words only, so curly quotes, commas and spacing do not hide a real quote."""
    return " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'")))


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def cut_into_clauses(sentence):
    """Split a sentence into [(text, [numbers]), ...]. The last item may have no numbers (the tail of the sentence)."""
    clauses, position = [], 0
    for match in GROUP.finditer(sentence):
        clauses.append((sentence[position:match.start()], [int(n) for n in re.findall(r"\d+", match.group())]))
        position = match.end()
    if sentence[position:].strip():
        clauses.append((sentence[position:], []))
    return clauses


def reattribute(answer, passages, embedder=None, floor=FLOOR, margin=MARGIN, use_meaning=False):
    """Return (new answer, report). passages: list of excerpt texts in [1], [2], ... order.

    use_meaning=False (default): only the quoted-words check runs. Notebook 08 showed that the meaning check (embeddings)
    moved correct numbers to wrong reviews, because embeddings treat 'easy to use' and 'very hard to use' as similar.
    """
    report = {"sentences": 0, "not_checked": 0, "citations_moved": 0, "sentences_removed": 0, "fabricated_quotes": 0, "weak_clauses": 0}
    sentences = split_sentences(answer)
    if not sentences or not passages:
        return answer, report
    normalised = [words_of(p) for p in passages]

    # one embedding call for every clause and every passage
    plan = [cut_into_clauses(s) for s in sentences]
    clause_texts = [text.strip(" ,;") for clauses in plan for text, numbers in clauses if numbers]
    if clause_texts and use_meaning:
        vectors = np.asarray(embedder.encode(clause_texts + list(passages), normalize_embeddings=True))
        similarity = vectors[:len(clause_texts)] @ vectors[len(clause_texts):].T
    row = 0

    output = []
    for sentence, clauses in zip(sentences, plan):
        cited_clauses = [c for c in clauses if c[1]]
        if not cited_clauses:                              # refusal or uncited text: other checks handle it
            output.append(sentence)
            continue
        report["sentences"] += 1

        inline_noun = clauses[0][0].strip() == "" or any(NOUN_USE.search(text) for text, numbers in clauses if numbers)
        if inline_noun:                                    # the number is part of the wording: leave it
            report["not_checked"] += 1
            row += len(cited_clauses)
            output.append(sentence)
            continue

        pieces, moved, fabricated = [], False, False
        for text, numbers in clauses:
            if not numbers:
                pieces.append(text)
                continue
            scores = similarity[row] if use_meaning and clause_texts else None
            row += 1
            numbers = [n for n in numbers if 1 <= n <= len(passages)]
            quotes = [words_of(q) for q in QUOTE.findall(text)]
            if quotes:
                holders = sorted({i + 1 for q in quotes for i, p in enumerate(normalised) if q in p})
                if not holders:
                    fabricated = True
                    break
                new_numbers = holders
            elif not use_meaning:
                new_numbers = numbers                      # no quoted words: keep the model's numbers
            else:
                best = int(np.argmax(scores))
                if scores[best] >= floor:
                    new_numbers = [n for n in numbers if scores[n - 1] >= scores[best] - margin] or [best + 1]
                else:
                    new_numbers = numbers                  # weak clause: keep the model's numbers, report it
                    report["weak_clauses"] += 1
            if sorted(set(new_numbers)) != sorted(set(numbers)):
                moved = True
            pieces.append(text.rstrip() + " " + "".join(f"[{n}]" for n in sorted(set(new_numbers))))

        if fabricated:
            report["fabricated_quotes"] += 1
            report["sentences_removed"] += 1
        else:
            report["citations_moved"] += int(moved)
            output.append(re.sub(r"\s+([.,;!?])", r"\1", "".join(pieces)).strip())

    return " ".join(output), report
