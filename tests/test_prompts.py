from pathlib import Path

from shopsense.llm.answer import SYSTEM_MESSAGE

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def test_prompt_copy_matches_the_code():
    assert (PROMPTS / "answer_v2.txt").read_text().strip() == SYSTEM_MESSAGE.strip()
