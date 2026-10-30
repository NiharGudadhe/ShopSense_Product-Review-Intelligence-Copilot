from shopsense.llm.answer import NO_ANSWER, UNRELIABLE, check_citations, cited_numbers, is_refusal_text, tidy_answer
from shopsense.llm.attribute import reattribute


def test_citation_lists_are_read():
    assert cited_numbers("Good [1, 2, 3]. Bad [4].") == [1, 2, 3, 4]


def test_refusal_in_other_words_counts_even_with_a_citation():
    assert is_refusal_text("The reviewers do not mention the packaging [1].")
    assert is_refusal_text(NO_ANSWER)
    assert not is_refusal_text("Reviewers like the scent. Some do not mention price.")


def test_invalid_citation_numbers_are_found():
    cited, invalid, uncited, refused = check_citations("It works well [2]. It is very cheap [9].", n_passages=6)
    assert invalid == [9] and uncited == 0 and not refused


def test_tidy_rewrites_lists_and_drops_uncited_sentences():
    answer, dropped = tidy_answer("Reviewers like it a lot. Some say good [1, 2]. Others hate it [3].")
    assert answer == "Some say good [1][2]. Others hate it [3]." and dropped == 1


def test_citation_only_answer_is_not_an_answer():
    answer, _ = tidy_answer("[1], [2], [3].")
    assert answer == UNRELIABLE


def test_quote_check_moves_citation_to_the_review_that_has_the_words():
    passages = ["lasts quite a long time", "it lasts all day, really"]
    new, report = reattribute('One reviewer says it lasts "all day" [1].', passages)
    assert new == 'One reviewer says it lasts "all day" [2].' and report["citations_moved"] == 1


def test_quote_found_in_no_review_removes_the_sentence():
    passages = ["nice clean scent that isn't overpowering"]
    new, report = reattribute('Good. One says "nice and not overpowering" [1]. Fine overall [1].', passages)
    assert "nice and not overpowering" not in new and report["fabricated_quotes"] == 1


def test_number_inside_the_wording_is_left_alone():
    passages = ["a", "b"]
    sentence = "Reviewers like [1] and [2] say it works."
    new, report = reattribute(sentence, passages)
    assert new == sentence and report["not_checked"] == 1
