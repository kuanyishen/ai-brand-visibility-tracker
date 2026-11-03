import pytest

from brandvis.config import load_brands
from brandvis.extract import build_matcher, extract_dictionary, name_map_from


@pytest.fixture(scope="module")
def matcher():
    return build_matcher(load_brands())


def names(text, matcher):
    return [m.brand for m in extract_dictionary(text, matcher)]


def test_orders_brands_by_first_mention(matcher):
    ms = extract_dictionary("Try Zara first, then Aritzia. Zara again and Reformation.", matcher)
    assert [m.brand for m in ms] == ["Zara", "Aritzia", "Reformation"]
    assert [m.rank for m in ms] == [1, 2, 3]
    assert ms[0].count == 2


def test_aritzia_sublabels_count_as_aritzia(matcher):
    assert names("The Wilfred Sonnet dress and the Sunday Best Martine are great, plus the TNA Butter fabric.", matcher) == ["Aritzia"]
    ms = extract_dictionary("Wilfred, Babaton and Aritzia all make good blazers", matcher)
    assert ms[0].brand == "Aritzia" and ms[0].count == 3


def test_ambiguous_names_need_exact_case(matcher):
    assert names("we need a gap year, express shipping and some theory about it, a target audience", matcher) == []
    assert names("Shop at Gap, Express and Theory.", matcher) == ["Gap", "Express", "Theory"]


def test_common_words_do_not_match(matcher):
    assert names("it skims the surface and will revolve around a quince jam", matcher) == []
    assert names("SKIMS and Revolve have shapewear", matcher) == ["Skims", "Revolve"]


def test_punctuated_and_multiword_aliases(matcher):
    assert names("J.Crew, J. Crew, H&M, Abercrombie & Fitch and Rag & Bone", matcher) == ["J.Crew", "H&M", "Abercrombie & Fitch", "Rag & Bone"]
    assert names("Alo Yoga and Alo", matcher) == ["Alo Yoga"]


def test_word_boundaries(matcher):
    assert names("Zaraz is not a brand; neither is Nikeish", matcher) == []
    assert names("(Zara) and 'Madewell'.", matcher) == ["Zara", "Madewell"]


def test_name_map_is_case_insensitive_and_covers_aliases():
    m = name_map_from(load_brands())
    assert m["wilfred"] == "Aritzia" and m["aritzia"] == "Aritzia" and m["zara"] == "Zara"
