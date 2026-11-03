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


def test_longest_match_wins_over_shorter_brand(matcher):
    ms = extract_dictionary("Try Nordstrom Rack, or Nordstrom itself.", matcher)
    assert [(m.brand, m.count) for m in ms] == [("Nordstrom Rack", 1), ("Nordstrom", 1)]


def test_curly_apostrophes_are_normalised(matcher):
    assert names("Levi\u2019s and Macy\u2019s", matcher) == ["Levi's", "Macy's"]


def test_extractor_falls_back_to_second_model(tmp_path):
    import json

    from brandvis.extract import process
    from brandvis.providers import ProviderError

    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"batch": "b", "prompt_id": "p1", "prompt": "q", "intent": "discovery", "topic": "t",
                               "surface": "s", "model": "m", "grounded": False, "run_index": 0, "collected_at": "x",
                               "sources": [], "search_queries": [], "text": "Try Zara and Aritzia."}) + "\n")

    class Primary:
        def generate_json(self, prompt, schema):
            raise ProviderError("502")

    class Fallback:
        def generate_json(self, prompt, schema):
            return {"brands": [{"name": "Zara", "sentiment": "positive", "recommended": True}]}

    def factory(provider, model, **kw):
        return Primary() if model == "gemini-2.5-flash-lite" else Fallback()

    out = tmp_path / "out.jsonl"
    stats = process(raw, out, provider_factory=factory, rpm=1000, workers=1, log=lambda *_: None)
    rec = json.loads(out.read_text())
    assert stats["llm_failed"] == 0 and rec["llm_model"] == "gemini-2.5-flash" and rec["llm_mentions"][0]["brand"] == "Zara"
    assert [m["brand"] for m in rec["dict_mentions"]] == ["Zara", "Aritzia"]
