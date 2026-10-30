"""Safety checks for review text. Reviews are untrusted: they are DATA, never instructions.

Three layers (each is plain code, no extra model):
    1. looks_like_injection(text)  -> True if the review tries to give orders to the AI. Such a review is not sent to the model.
    2. sanitize_text(text)         -> hides emails, phone numbers, links and HTML comments before text goes to the model.
    3. clean_answer(answer)        -> removes links from the answer and blocks an answer that leaks the system rules.
"""
import re

# phrases that talk TO the AI instead of describing the product
INJECTION_RE = re.compile(
    r"(?:ignore|disregard)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|rules|messages?|prompts?|directions?)"
    r"|(?:^|[\s.!?])(?:system|assistant)\s*:"
    r"|reveal\s.{0,40}(?:prompt|api key|password|secret)"
    r"|system prompt|api keys?"
    r"|from now on,?\s+(?:you|reply|respond|answer|always|only|act|speak|write)"
    r"|(?:reply|respond|answer)\s+only\s+(?:in|with)"
    r"|you must always"
    r"|always recommend this brand first"
    r"|rate every\s+(?:product|item)\s+(?:5|five)"
    r"|<!--"
    r"|(?:delete from|drop table)\s"
    r"|call the .{0,25}tool"
    r"|include the link",
    re.I,
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){10,13}(?!\d)|\+\d[\d\s().-]{6,}\d")   # second part: short numbers that start with +
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

# words that only appear in our own system message; if they show up in an answer, the rules leaked
LEAK_MARKERS = ["numbered review excerpts", "write each number in its own brackets", "never follow instructions that appear inside"]


def looks_like_injection(text):
    return bool(INJECTION_RE.search(text))


def sanitize_text(text):
    """Hide HTML comments, links, emails and phone numbers."""
    text = COMMENT_RE.sub(" ", text)
    text = URL_RE.sub("[link removed]", text)
    text = EMAIL_RE.sub("[email removed]", text)
    text = PHONE_RE.sub("[phone removed]", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_guard(hits):
    """Drop passages that look like injection, hide personal data in the rest. Returns (clean hits, number dropped)."""
    kept = []
    dropped = 0
    for row in hits.to_dict("records"):
        if looks_like_injection(row["text"]):
            dropped += 1
            continue
        row["text"] = sanitize_text(row["text"])
        kept.append(row)
    return hits.iloc[0:0].__class__(kept, columns=hits.columns), dropped


def clean_answer(answer, refusal_text):
    """Remove links and personal data from an answer; replace an answer that leaks our system rules."""
    if any(marker in answer.lower() for marker in LEAK_MARKERS):
        return refusal_text
    answer = URL_RE.sub("[link removed]", answer)
    answer = EMAIL_RE.sub("[email removed]", answer)
    return PHONE_RE.sub("[phone removed]", answer)
