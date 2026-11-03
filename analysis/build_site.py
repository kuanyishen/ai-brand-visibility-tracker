"""Render docs/index.html from docs/assets/results.json (every number in the text comes from the analysis)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs" / "assets" / "results.json").read_text())
REPO = "https://github.com/kuanyishen/ai-brand-visibility-tracker"
F = R["meta"]["focus"]
S0, S1 = "gemini-flash", "gemini-flash-search"
pc = lambda x, d=0: f"{x * 100:.{d}f}%"
pp = lambda x, d=0: f"{x * 100:+.{d}f} pp"
ci = lambda lo, hi, f=pc: f"[{f(lo)}, {f(hi)}]"
ci_pp = lambda lo, hi: f"[{lo * 100:+.0f} pp, {hi * 100:+.0f} pp]"
ordinal = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

fi = lambda s, i: next(r for r in R["focus_by_intent"] if r["surface"] == s and r["intent"] == i)
disc = {s: fi(s, "discovery") for s in (S0, S1)}
cmpf, cmpb = R["surface_comparison"]["focus_mention"], R["surface_comparison"]["brands_per_answer"]
stab = lambda s, i: next(r for r in R["stability"] if r["surface"] == s and r["intent"] == i)
var = R["variance"]
cons = R["focus_consistency"]
ex = R["extractor"]
src = R["sources"]
min_p_holm = {s: min(r["p_holm"] for r in R["vs_competitors"][s]) for s in (S0, S1)}
lab = {S0: "Gemini (no search)", S1: "Gemini + Google Search"}
sent = lambda s, i: next(r for r in R["sentiment"] if r["surface"] == s and r["intent"] == i)


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def vis_table(s, n=10):
    rows = []
    for i, r in enumerate(R["visibility"][s][:n], 1):
        rows.append([i, ("<b>" + r["brand"] + "</b>") if r["brand"] == F else r["brand"], pc(r["mention_rate"]),
                     ci(r["mr_lo"], r["mr_hi"]), pc(r["share_of_voice"], 1), f"{r['mean_rank']:.1f}"])
    return table(["#", "Brand", "Mentioned in", "95% CI", "Share of voice", "Avg. position when mentioned"], rows)


def vs_table():
    rows = []
    for s in (S0, S1):
        for r in R["vs_competitors"][s][:6]:
            rows.append([lab[s], r["competitor"], pp(r["diff"]) , ci(r["lo"] * 100, r["hi"] * 100, lambda x: f"{x:+.0f} pp"), f"{r['p_holm']:.2f}"])
    return table(["Surface", "Competitor", f"{F} minus competitor", "95% CI", "p (Holm-adjusted)"], rows)


def stab_table():
    rows = []
    for i, name in (("discovery", "No brand named"), ("alternative", "Alternatives to Aritzia"), ("reputation", "About Aritzia"), ("versus", "Aritzia vs competitor")):
        rows.append([name, f"{stab(S0, i)['jaccard_all']:.2f}", f"{stab(S1, i)['jaccard_all']:.2f}"])
    return table(["Prompt type", lab[S0], lab[S1]], rows)


def sent_table():
    rows = []
    for i, name in (("discovery", "No brand named"), ("alternative", "Alternatives to Aritzia"), ("versus", "Aritzia vs competitor"), ("reputation", "About Aritzia")):
        for s in (S0, S1):
            r = sent(s, i)
            rows.append([name, lab[s], r["n"], f"{r['positive']} / {r['neutral']} / {r['mixed']} / {r['negative']}", pc(r["recommended"])])
    return table(["Prompt type", "Surface", "Answers mentioning " + F, "Positive / neutral / mixed / negative", "Recommended"], rows)


def src_table():
    return table(["Domain", "Share of citations", "Answers citing it", "Type"],
                 [[r["domain"], pc(r["share"], 1), r["answers"], r["category"]] for r in src["top"][:10]])


def topic_note():
    by = {}
    for r in R["focus_by_topic"]:
        by.setdefault(r["topic"], {})[r["surface"]] = r["rate"]
    top = sorted(by, key=lambda t: -(by[t][S0] + by[t][S1]))
    zero = [t for t in by if by[t][S0] == 0 and by[t][S1] == 0]
    return top[:4], zero


top_topics, zero_topics = topic_note()
ext = ex["per_brand"]

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Brand Visibility Tracker</title>
<meta name="description" content="How often, how prominently and how consistently do AI assistants bring up a brand? A measured study of women's fashion, with confidence intervals and statistical tests.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="top"><div class="wrap">
  <a class="brand" href="index.html">AI brand visibility tracker</a>
  <a href="#design">Design</a><a href="#visibility">Visibility</a><a href="#search">Search effect</a><a href="#stability">Stability</a><a href="#sources">Sources</a><a href="#audit">Audit</a>
  <a href="{REPO}">GitHub</a>
</div></nav>
<main class="wrap">
<header class="hero">
  <div class="eyebrow">Generative engine visibility · {R['meta']['category']}</div>
  <h1>Does AI recommend {F}? Measuring brand visibility in AI answers</h1>
  <p class="lead">When a shopper asks an AI assistant where to buy a blazer, which brands does it name, in what order, and would it say the same thing twice? I asked {R['meta']['n_prompts']} shopper questions {R['meta']['runs_per_prompt']} times each on two AI surfaces, extracted every brand mentioned, and measured visibility with proper confidence intervals and tests. {F} is the focus brand; {R['meta']['n_brands_tracked']} brands are tracked.</p>
  <p class="sub">Independent study; not affiliated with {F} or any brand mentioned. Results describe what one model said on one day ({R['meta']['collected_from']}), not the brands' products.</p>
  <div class="row"><a class="btn" href="#visibility">See the results</a><a class="btn ghost" href="{REPO}">View code</a></div>
</header>

<div class="kpis">
  <div class="kpi"><b>{pc(disc[S0]['rate'])} / {pc(disc[S1]['rate'])}</b><span>of answers mention {F} when the question never names a brand (no search / with search)</span></div>
  <div class="kpi"><b>#{R['focus_rank'][S0]['by_mention_rate']} / #{R['focus_rank'][S1]['by_mention_rate']}</b><span>{F}'s rank among {R['meta']['n_brands_tracked']} tracked brands by visibility, though gaps to rivals are within noise</span></div>
  <div class="kpi"><b>{stab(S0, 'discovery')['jaccard_all']:.2f} / {stab(S1, 'discovery')['jaccard_all']:.2f}</b><span>overlap between the brand lists of two answers to the same question (1.0 = identical)</span></div>
  <div class="kpi"><b>{var[S0]['prompts_for_se5_with_3_runs']}+</b><span>different prompts needed to pin the mention rate to ±5 pp; repeating the same prompts cannot do it</span></div>
</div>

<div class="callout"><p><b>TL;DR.</b> With no brand named, Gemini brings up {F} in about half of shopping answers ({pc(disc[S0]['rate'])} without search, {pc(disc[S1]['rate'])} with Google Search) and ranks it #{R['focus_rank'][S1]['by_mention_rate']} with search. But visibility depends heavily on the topic ({', '.join(top_topics[:3])} vs none in {', '.join(zero_topics[:3])}), and with only {R['meta']['intent_counts']['discovery']} unbranded prompts I can't statistically separate {F} from competitors such as Everlane or Uniqlo. Search makes answers shorter ({abs(cmpb['mean_diff_search_minus_plain']):.1f} fewer brands per answer, p = {cmpb['p_value']:.4f}) but the change in {F}'s mention rate ({pp(cmpf['mean_diff_search_minus_plain'])}, CI {ci_pp(cmpf['lo'], cmpf['hi'])}) is inconclusive. Repeating a question changes the brand list a lot, so single-shot "AI visibility scores" can mislead.</p></div>

<h2 id="design">1 · Study design</h2>
<p>Every prompt is something a shopper might type. Only the <b>unbranded</b> prompts measure visibility, because a brand named in the question will naturally appear in the answer.</p>
{table(['Prompt type', 'Prompts', 'Example', 'Used for'], [
    ['Unbranded ("discovery")', R['meta']['intent_counts']['discovery'], '"Where can I buy stylish women\'s workwear that isn\'t boring?"', 'Visibility, share of voice, rank'],
    ['Alternatives', R['meta']['intent_counts']['alternative'], '"What are good alternatives to Aritzia?"', 'Which competitors are named'],
    ['Head-to-head', R['meta']['intent_counts']['versus'], '"Aritzia vs Reformation: which is better for dresses?"', 'Positioning, sentiment'],
    ['About the brand', R['meta']['intent_counts']['reputation'], '"Is Aritzia worth the price?"', 'Sentiment'],
])}
<p>Two <b>AI surfaces</b> built on the same model (<code>{R['meta']['surfaces'][S0]['model']}</code>): plain, answering from the model's own knowledge, and with <b>Google Search grounding</b>, which looks up web pages and cites them (closer to what shoppers see in AI answers with sources). Each prompt was asked {R['meta']['runs_per_prompt']} times at temperature 1.0 on each surface: <b>{R['meta']['n_answers']} answers</b>, about {R['meta']['total_tokens'] // 1000}k tokens. Brands were extracted two independent ways (exact-name dictionary and an LLM reader) and audited against each other (section 7).</p>
<p class="sub">Statistics note: answers to the same prompt are far more alike than answers to different prompts, so every interval resamples <em>prompts</em>, not answers. Treating all {R['meta']['n_answers']} answers as independent would make the intervals look too tight.</p>

<h2 id="visibility">2 · How visible is {F}?</h2>
<figure><img src="assets/fig_visibility.png" alt="Brand mention rates on unbranded prompts for both surfaces"><figcaption>Share of answers to the 24 unbranded questions that mention each brand. Bars are 95% confidence intervals over prompts. {F}'s own labels (Wilfred, Babaton, Tna, Sunday Best, Super Puff) count as {F}.</figcaption></figure>
<h3>{lab[S0]}</h3>
{vis_table(S0)}
<h3>{lab[S1]}</h3>
{vis_table(S1)}
<p>{F} is the {ordinal(R['focus_rank'][S0]['by_mention_rate'])} most-mentioned brand without search and the {ordinal(R['focus_rank'][S1]['by_mention_rate'])} with search. Is it <em>significantly</em> ahead of anyone? No. Comparing {F} with each top competitor on the same prompts (paired by prompt, Holm-corrected for multiple comparisons), the smallest adjusted p-value is {min_p_holm[S0]:.2f} without search and {min_p_holm[S1]:.2f} with search:</p>
{vs_table()}
<p class="sub">Share of voice = a brand's mentions divided by all tracked-brand mentions. Rank = order of first mention within the answer (1 = first brand named). "Mentioned in" counts an answer once however often the brand appears.</p>

<h3>It depends on the topic</h3>
<figure><img src="assets/fig_topics.png" alt="Aritzia mention rate by topic" style="max-width:640px"><figcaption>Each topic has only 1-3 prompts (3-9 answers per surface), so read this as descriptive, not as precise estimates.</figcaption></figure>
<p>{F} shows up almost every time for Canada-related, capsule-wardrobe, basics, trend and workwear questions, and never for {', '.join(zero_topics)}. For a brand team the actionable view is the <em>gaps by topic</em>, not the single average.</p>

<h2 id="search">3 · Does adding search change the answer?</h2>
<p>Compared prompt by prompt (the same {cmpf['n_prompts']} unbranded questions on both surfaces):</p>
{table(['Metric', lab[S0], lab[S1], 'Difference (search − plain)', '95% CI over prompts', 'p (sign-flip test)'], [
    [f'{F} mention rate', pc(cmpf['plain_mean']), pc(cmpf['search_mean']), pp(cmpf['mean_diff_search_minus_plain']), ci_pp(cmpf['lo'], cmpf['hi']), f"{cmpf['p_value']:.2f}"],
    ['Tracked brands named per answer', f"{cmpb['plain_mean']:.1f}", f"{cmpb['search_mean']:.1f}", f"{cmpb['mean_diff_search_minus_plain']:+.1f}", ci(cmpb['lo'], cmpb['hi'], lambda x: f"{x:+.1f}"), f"{cmpb['p_value']:.4f}"],
])}
<p><b>Search makes answers more selective</b>: {cmpb['mean_diff_search_minus_plain']:.1f} brands per answer on average, a clear effect (the confidence interval excludes zero). <b>Its effect on {F}'s mention rate is inconclusive</b>: the estimate is {pp(cmpf['mean_diff_search_minus_plain'])}, but with {cmpf['n_prompts']} prompts this comparison can only reliably detect differences of about {pc(cmpf['mde_80pct'])} (80% power), so a real effect of 10 points would very often be missed. That is a power-analysis result about my design, not evidence of "no effect".</p>

<h2 id="stability">4 · Would the AI say the same thing twice?</h2>
{stab_table()}
<figure><img src="assets/fig_stability.png" alt="Brand list overlap between repeated answers" style="max-width:640px"><figcaption>Jaccard overlap of the brand lists in repeated answers to the same prompt (1 = identical, 0 = nothing in common).</figcaption></figure>
<p>For unbranded questions, two answers to the same prompt share only about {pc(stab(S0, 'discovery')['jaccard_all'])} of their brands without search and {pc(stab(S1, 'discovery')['jaccard_all'])} with search. In the plain surface, {cons[S0]['always']} of {cons[S0]['n_prompts']} prompts always mention {F}, {cons[S0]['never']} never do, and {cons[S0]['sometimes']} do so only <em>sometimes</em> (with search: {cons[S1]['always']} / {cons[S1]['never']} / {cons[S1]['sometimes']}). A single run of a prompt is therefore a noisy measurement.</p>
<h3>How many repeats and prompts do you need?</h3>
<p>I split the variance in "did the answer mention {F}?" into a part that differs <b>between prompts</b> and a part that differs <b>between repeats of the same prompt</b> (intra-class correlation, ICC): {var[S0]['icc']:.2f} without search, {var[S1]['icc']:.2f} with search. Prompt-to-prompt differences set a floor that repeating cannot remove:</p>
<figure><img src="assets/fig_design.png" alt="Standard error versus number of repeats and prompts"><figcaption>Standard error of the overall mention rate under the fitted variance components. Left: with 24 prompts, repeating each one 100 times still leaves SE ≈ {R['design'][S0]['se_by_runs'][-1] * 100:.0f} pp. Right: more prompts keep shrinking it.</figcaption></figure>
<p><b>Design rule:</b> to estimate a mention rate to ±5 percentage points (one SE) you need about {var[S0]['prompts_for_se5_with_3_runs']} prompts with 3 repeats each ({var[S0]['prompts_for_se5_with_30_runs']} even with 30 repeats). This is the same power logic as an A/B test, applied to prompt design, and it is why the intervals above are as wide as they are.</p>

<h2>5 · What do the answers say about {F}?</h2>
<p>Sentiment and "recommended" flags come from the LLM reader judging how the <em>answer</em> portrays the brand (not my opinion of the brand):</p>
{sent_table()}
<p>When {F} appears in an unbranded answer it is almost always portrayed positively and recommended ({pc(sent(S0, 'discovery')['recommended'])} without search, {pc(sent(S1, 'discovery')['recommended'])} with search). Asked directly whether it is worth the price, both surfaces give a <em>mixed</em> verdict every time ({sent(S0, 'reputation')['mixed']} of {sent(S0, 'reputation')['n']} and {sent(S1, 'reputation')['mixed']} of {sent(S1, 'reputation')['n']} answers) and recommend it less ({pc(sent(S0, 'reputation')['recommended'])} / {pc(sent(S1, 'reputation')['recommended'])}). Selection bias matters here: discovery answers list only brands the AI chooses to recommend, so they are positive almost by construction.</p>

<h2 id="sources">6 · Where does the search-grounded answer get its information?</h2>
<figure><img src="assets/fig_sources.png" alt="Most cited domains" style="max-width:640px"><figcaption>{src['n_grounded_answers']} search-grounded answers cited {src['n_domains']} different domains ({src['mean_citations_per_answer']:.0f} citations per answer on average).</figcaption></figure>
{src_table()}
<ul class="tight">
<li><b>Community and editorial content dominates.</b> Reddit alone is cited in {src['answers_citing_reddit']} of {src['n_grounded_answers']} answers ({pc(src['answers_citing_reddit'] / src['n_grounded_answers'])}); social/community sites are {pc(src['category_share']['social / community'])} of all citations and editorial/blogs {pc(src['category_share']['editorial / blogs'])}.</li>
<li><b>The brand's own site is a small share:</b> {F}'s website accounts for {pc(src['focus_brand_site_share'], 1)} of citations ({src['answers_citing_focus_site']} answers). What other people write about a brand matters at least as much as what the brand publishes.</li>
<li class="sub">Domain types are a rough automatic labelling (retailer/brand sites are matched against the brand list; everything else non-social, non-review is "editorial / blogs").</li>
</ul>

<h2 id="audit">7 · Auditing the measurement itself</h2>
<p>Both extractors agree on {pc(ex['agree'], 1)} of all answer × brand cells (Cohen's κ = {ex['kappa']:.2f}), and they give the <em>same</em> {F} mention rate on unbranded prompts ({pc(ex['focus_rate_dictionary'][S0]['rate'], 1)} and {pc(ex['focus_rate_dictionary'][S1]['rate'], 1)} by dictionary, {pc(ex['focus_rate_llm'][S0]['rate'], 1)} and {pc(ex['focus_rate_llm'][S1]['rate'], 1)} by LLM). I read the disagreements in context: the dictionary-only hits are mostly ownership mentions ("part of the H&amp;M group", "Gap Inc.", "an Amazon-owned retailer") and a few false positives such as the verb "Target"; the LLM never found a tracked brand that the dictionary missed. The dictionary is the primary measure; the LLM reader is the robustness check and the source of sentiment.</p>
<p>The LLM also reports brands outside my list: {pc(ex['untracked_share_of_llm_mentions'])} of its brand mentions are to untracked brands (Poshmark, Lulus, Boden and others), so share of voice here means share <em>among the {R['meta']['n_brands_tracked']} tracked brands</em>. After the first collection I added the 24 most frequent untracked names to the dictionary.</p>

<h2>Limitations</h2>
<ul class="tight">
<li><b>One model family, one day.</b> Both surfaces are Gemini 2.5 Flash. ChatGPT, Perplexity, Claude and Google's AI Overviews may differ, and answers drift over time. This is a snapshot, not a trend.</li>
<li><b>API is not the consumer product.</b> The apps add system prompts, personalization and location that an API call does not.</li>
<li><b>The prompt set is my design.</b> Several prompts relate to Canada and everyday basics, where {F} naturally fits, so the absolute mention rate depends on the prompt mix; relative comparisons across surfaces and brands are more reliable than the level.</li>
<li><b>Small samples per topic and per competitor comparison</b> (24 unbranded prompts): conclusions are about patterns and design, not precise market-share numbers.</li>
<li><b>What an AI says is not what is true.</b> Sentiment is how the answer portrays a brand, judged by another model.</li>
</ul>

<h2>Reproduce and extend</h2>
<pre><code>pip install -e ".[dev]"
export GEMINI_API_KEY=...                       # not needed if a proxy injects the credential
python -m brandvis.collect --batch 2026-11-03   # 240 answers, ~12 min
python -m brandvis.extract --raw data/raw/2026-11-03.jsonl --out data/processed/2026-11-03.jsonl
python analysis/analyze.py &amp;&amp; python analysis/build_site.py
pytest                                           # 28 tests</code></pre>
<p>Next steps: add more AI surfaces (the provider interface is one small class), repeat weekly to track drift, and grow the prompt set to ~70 for tighter intervals.</p>
<footer>Built by Sky Shen · <a href="{REPO}">source and raw data</a></footer>
</main>
</body>
</html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html")
