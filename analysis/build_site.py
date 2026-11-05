"""Render docs/index.html from docs/assets/results.json (every number in the text comes from the analysis)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs" / "assets" / "results.json").read_text())
REPO = "https://github.com/kuanyishen/ai-brand-visibility-tracker"
M = R["meta"]
F = M["focus"]
SS = R["surface_order"]
L = R["labels"]
G, GS, C, CS = "gemini-flash", "gemini-flash-search", "claude-sonnet", "claude-sonnet-search"
pc = lambda x, d=0: f"{x * 100:.{d}f}%"
pp = lambda x, d=0: f"{x * 100:+.{d}f} pp"
ci_pp = lambda lo, hi: f"[{lo * 100:+.0f}, {hi * 100:+.0f}] pp"
ordinal = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

disc = {s: next(r for r in R["focus_by_intent"] if r["surface"] == s and r["intent"] == "discovery") for s in SS}
pair = {(p["a"], p["b"]): p for p in R["pairs"]}
sp = {s: pair[(s, s.replace("-search", ""))] for s in (GS, CS)}                      # search effect
pv = {"plain": pair[(C, G)], "search": pair[(CS, GS)]}                                # Claude minus Gemini
stab = {(r["surface"], r["intent"]): r["jaccard_all"] for r in R["stability"]}
J = R["agreement"]["jaccard"]
off = [J[a][b] for a in SS for b in SS if a != b]
diag = [J[s][s] for s in SS]
src = R["sources"]
drift = {d["surface"]: d for d in R["drift"]}
ex = R["extractor"]
rates = [disc[s]["rate"] for s in SS]
ranks = [R["focus_rank"][s]["by_mention_rate"] for s in SS]
all_p = [m["p_value"] for p in R["pairs"] for k, m in p["metrics"].items() if k == "focus_mention"]
mdes = [m["mde_80pct"] for p in R["pairs"] for k, m in p["metrics"].items() if k == "focus_mention"]
sent = lambda s, i: next((r for r in R["sentiment"] if r["surface"] == s and r["intent"] == i), None)
var = R["variance"]


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def design_table():
    return table(["Surface", "Model", "Web search", "Repeats per prompt", "Answers"],
                 [[L[s], f"<code>{M['surfaces'][s]['model']}</code>", "yes" if M["surfaces"][s]["grounded"] else "no", M["surfaces"][s]["runs"], M["surfaces"][s]["n_answers"]] for s in SS])


def visibility_table(n=12):
    by = {s: {r["brand"]: r for r in R["visibility"][s]} for s in SS}
    brands = {b for s in SS for b in by[s]}
    avg = {b: sum(by[s][b]["mention_rate"] if b in by[s] else 0 for s in SS) / len(SS) for b in brands}
    top = sorted(avg, key=lambda b: -avg[b])[:n]
    if F not in top:
        top.append(F)
    rows = []
    for b in top:
        cells = [(pc(by[s][b]["mention_rate"]) if b in by[s] else "–") for s in SS]
        rows.append([f"<b>{b}</b>" if b == F else b, *[f"<b>{c}</b>" if b == F else c for c in cells]])
    rows.append([f"<i>{F}'s rank</i>", *[f"<i>{ordinal(R['focus_rank'][s]['by_mention_rate'])}</i>" for s in SS]])
    return table(["Brand (mentioned in % of unbranded answers)", *[L[s] for s in SS]], rows)


def pair_table():
    rows = []
    names = {"focus_mention": f"{F} mention rate", "brands_per_answer": "Tracked brands named per answer"}
    for p in R["pairs"]:
        for k, m in p["metrics"].items():
            f = (lambda x: pp(x)) if k == "focus_mention" else (lambda x: f"{x:+.1f}")
            ci = ci_pp(m["lo"], m["hi"]) if k == "focus_mention" else f"[{m['lo']:+.1f}, {m['hi']:+.1f}]"
            rows.append([p["name"], names[k], f(m["mean_diff"]), ci, f"{m['p_value']:.3f}" if m["p_value"] >= 0.001 else "< 0.001", pp(m["mde_80pct"]) if k == "focus_mention" else f"{m['mde_80pct']:.1f}"])
    return table(["Comparison (paired by prompt)", "Metric", "Difference", "95% CI over prompts", "p (sign-flip)", "Smallest detectable (80% power)"], rows)


def vs_table():
    rows = []
    for s in SS:
        worst = min(R["vs_competitors"][s], key=lambda r: r["p_holm"])
        rows.append([L[s], f"{worst['competitor']} ({pp(worst['diff'])})", f"{worst['p_holm']:.2f}"])
    return table(["Surface", f"Closest competitor to a significant difference ({F} minus competitor)", "Holm-adjusted p"], rows)


def stab_table():
    names = (("discovery", "No brand named"), ("alternative", f"Alternatives to {F}"), ("reputation", f"About {F}"), ("versus", f"{F} vs competitor"))
    return table(["Prompt type", *[L[s] for s in SS]], [[n, *[f"{stab[(s, i)]:.2f}" for s in SS]] for i, n in names])


def cons_table():
    rows = []
    for s in SS:
        c = R["focus_consistency"][s]
        rows.append([L[s], c["always"], c["never"], c["sometimes"], f"{var[s]['icc']:.2f}", var[s]["prompts_for_se5_with_3_runs"]])
    return table(["Surface", f"Prompts that always mention {F}", "Never", "Only sometimes", "Between-prompt share of variance (ICC)", "Prompts needed for ±5 pp (3 repeats)"], rows)


def drift_table():
    rows = []
    for s in (G, GS):
        d = drift.get(s)
        if d:
            rows.append([L[s], f"{pc(d['b_mean'])} → {pc(d['a_mean'])}", f"{pp(d['mean_diff'])} {ci_pp(d['lo'], d['hi'])}", f"{d['p_value']:.2f}",
                         f"{d['jaccard_within_old_day']:.2f} / {d['jaccard_within_new_day']:.2f}", f"{d['jaccard_across_days']:.2f}"])
    return table(["Surface", f"{F} mention rate, day 1 → day 2", "Change", "p", "Brand-list overlap within day 1 / day 2", "Across the two days"], rows)


def sent_table():
    rows = []
    for i, n in (("discovery", "No brand named"), ("reputation", f"About {F}")):
        for s in SS:
            r = sent(s, i)
            if r:
                rows.append([n, L[s], r["n"], f"{r['positive']} / {r['neutral']} / {r['mixed']} / {r['negative']}", pc(r["recommended"])])
    return table(["Prompt type", "Surface", f"Answers mentioning {F}", "Positive / neutral / mixed / negative", "Recommended"], rows)


def src_table():
    rows = []
    for s in (GS, CS):
        v = src[s]
        rows.append([L[s], v["n_grounded_answers"], f"{v['mean_citations_per_answer']:.1f}", v["n_domains"], f"{v['answers_citing_reddit']} of {v['n_grounded_answers']}",
                     pc(v["focus_brand_site_share"], 1), pc(v["category_share"].get("social / community", 0)), pc(v["category_share"].get("editorial / blogs", 0))])
    return table(["Surface", "Answers", "Cited pages per answer", "Distinct domains", "Answers citing Reddit", f"{F}'s own site, share of citations", "Social / community", "Editorial / blogs"], rows)


page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Brand Visibility Tracker</title>
<meta name="description" content="How often, how prominently and how consistently do Gemini and Claude bring up a brand? A measured study of women's fashion with confidence intervals and statistical tests.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="top"><div class="wrap">
  <a class="brand" href="index.html">AI brand visibility tracker</a>
  <a href="#design">Design</a><a href="#visibility">Visibility</a><a href="#agree">Agreement</a><a href="#search">Search</a><a href="#stability">Stability</a><a href="#sources">Sources</a><a href="#audit">Audit</a>
  <a href="{REPO}">GitHub</a>
</div></nav>
<main class="wrap">
<header class="hero">
  <div class="eyebrow">Generative engine visibility · {M['category']}</div>
  <h1>Does AI recommend {F}? Measuring brand visibility in Gemini and Claude</h1>
  <p class="lead">When a shopper asks an AI assistant where to buy a blazer, which brands does it name, in what order, and would it say the same thing twice? I asked {M['n_prompts']} shopper questions on four AI surfaces (Gemini 2.5 Flash and Claude Sonnet 5.5, each with and without web search), extracted every brand mentioned, and measured visibility with confidence intervals and statistical tests. {F} is the focus brand; {M['n_brands_tracked']} brands are tracked.</p>
  <p class="sub">Independent study; not affiliated with {F} or any brand mentioned. Results describe what these models said on {M['collected_on']}, not the brands' products.</p>
  <div class="row"><a class="btn" href="#visibility">See the results</a><a class="btn ghost" href="{REPO}">View code</a></div>
</header>

<div class="kpis">
  <div class="kpi"><b>{pc(min(rates))}–{pc(max(rates))}</b><span>of answers mention {F} when the question never names a brand, across the four surfaces. No surface differs significantly from another</span></div>
  <div class="kpi"><b>{min(off):.2f}–{max(off):.2f}</b><span>brand-list overlap between <em>different</em> AI surfaces, vs {min(diag):.2f}–{max(diag):.2f} when the <em>same</em> surface is asked again</span></div>
  <div class="kpi"><b>{abs(sp[GS]['metrics']['brands_per_answer']['mean_diff']):.1f} / {abs(sp[CS]['metrics']['brands_per_answer']['mean_diff']):.1f}</b><span>fewer brands per answer when Gemini / Claude use web search</span></div>
  <div class="kpi"><b>{src[GS]['answers_citing_reddit']} vs {src[CS]['answers_citing_reddit']}</b><span>search answers citing Reddit: Gemini ({src[GS]['n_grounded_answers']} answers) vs Claude ({src[CS]['n_grounded_answers']})</span></div>
</div>

<div class="callout"><p><b>TL;DR.</b> With no brand named, {F} is mentioned in {pc(min(rates))}–{pc(max(rates))} of shopping answers on every surface (ranked {ordinal(min(ranks))}–{ordinal(max(ranks))} of {M['n_brands_tracked']} brands). I can't separate the surfaces from each other or {F} from its closest competitors (all paired p ≥ {min(all_p):.2f}; with {M['intent_counts']['discovery']} prompts each comparison only detects gaps of about {pc(min(mdes))}–{pc(max(mdes))}). What <em>is</em> clear: web search makes answers more selective on both providers; different AIs name noticeably different brands for the same question, less overlap than one AI repeating itself; and Gemini's and Claude's search tools lean on very different sources, with Reddit appearing in {src[GS]['answers_citing_reddit']} of {src[GS]['n_grounded_answers']} Gemini search answers and in none of Claude's {src[CS]['n_grounded_answers']}. Because repeating the same question changes the brand list so much, single-shot "AI visibility scores" can mislead.</p></div>

<h2 id="design">1 · Study design</h2>
<p>Every prompt is something a shopper might type. Only the <b>unbranded</b> prompts measure visibility, because a brand named in the question naturally appears in the answer.</p>
{table(['Prompt type', 'Prompts', 'Example', 'Used for'], [
    ['Unbranded ("discovery")', M['intent_counts']['discovery'], '"Where can I buy stylish women\'s workwear that isn\'t boring?"', 'Visibility, share of voice, rank'],
    ['Alternatives', M['intent_counts']['alternative'], f'"What are good alternatives to {F}?"', 'Which competitors are named'],
    ['Head-to-head', M['intent_counts']['versus'], f'"{F} vs Reformation: which is better for dresses?"', 'Positioning, sentiment'],
    ['About the brand', M['intent_counts']['reputation'], f'"Is {F} worth the price?"', 'Sentiment'],
])}
{design_table()}
<p>All {M['n_answers']} answers were collected on the same day at temperature 1.0 (Claude's search surface got 2 repeats per prompt instead of 3 to control cost: a search answer costs about 15 times a plain one because the retrieved pages count as input). Brands were extracted two independent ways (exact-name dictionary and an LLM reader) and audited against each other (section 8). A first collection of the Gemini surfaces two days earlier is used to look for drift (section 6).</p>
<p class="sub">Statistics note: answers to the same prompt are far more alike than answers to different prompts, so every interval resamples <em>prompts</em>, not answers. Treating all answers as independent would make the intervals look too tight.</p>

<h2 id="visibility">2 · How visible is {F}?</h2>
<figure><img src="assets/fig_overview.png" alt="Aritzia mention rate on each surface" style="max-width:700px"><figcaption>Share of answers to the {M['intent_counts']['discovery']} unbranded questions that mention {F} (95% CI over prompts). {F}'s own labels (Wilfred, Babaton, Tna, Sunday Best, Super Puff) count as {F}.</figcaption></figure>
{visibility_table()}
<figure><img src="assets/fig_visibility.png" alt="Most-mentioned brands per surface"><figcaption>The ten most-mentioned brands on each surface, with 95% confidence intervals over prompts ({F} in red).</figcaption></figure>
<p>The intervals overlap broadly everywhere: each surface's range for {F} is roughly {pc(min(d['lo'] for d in disc.values()))} to {pc(max(d['hi'] for d in disc.values()))}. Comparing {F} with each of the ten most-mentioned competitors on the same prompts (paired, Holm-corrected for multiple comparisons) finds no significant difference on any surface:</p>
{vs_table()}
<h3>It depends on the topic</h3>
<figure><img src="assets/fig_topics.png" alt="Aritzia mention rate by topic" style="max-width:700px"><figcaption>Each topic has only 1-3 prompts, so read this as descriptive, not as precise estimates.</figcaption></figure>
<p>For a brand team the actionable view is the <em>gaps by topic</em>, not the single average: some questions bring {F} up almost every time on every surface, and others never do.</p>

<h2 id="agree">3 · Do the AIs agree with each other?</h2>
<figure><img src="assets/fig_agreement.png" alt="Brand-list overlap between surfaces" style="max-width:640px"><figcaption>For each prompt, the average Jaccard overlap between the brand lists of answers from two surfaces (1 = identical lists). The diagonal compares repeated answers from the same surface.</figcaption></figure>
<p>Different surfaces share {min(off):.2f}–{max(off):.2f} of their brands on the same question, while one surface asked twice shares {min(diag):.2f}–{max(diag):.2f}. The most similar pair is Gemini and Claude <em>without</em> search ({J[G][C]:.2f}); Gemini with search is the most different from the others ({min(J[GS][b] for b in SS if b != GS):.2f}–{max(J[GS][b] for b in SS if b != GS):.2f}). At the level of the whole brand universe the AIs agree more: the rank correlation of brand mention rates is {min(R['agreement']['spearman_mention_rates'][a][b] for a in SS for b in SS if a != b):.2f}–{max(R['agreement']['spearman_mention_rates'][a][b] for a in SS for b in SS if a != b):.2f}. They broadly agree on which brands are popular, but not on which ones to name for a particular question.</p>
<p>On {F}'s own visibility, Claude minus Gemini is {pp(pv['plain']['metrics']['focus_mention']['mean_diff'])} without search and {pp(pv['search']['metrics']['focus_mention']['mean_diff'])} with search, neither distinguishable from zero:</p>
{pair_table()}

<h2 id="search">4 · Does adding search change the answer?</h2>
<p><b>Yes for breadth, inconclusive for {F}.</b> On both providers, web search makes answers more selective: {abs(sp[GS]['metrics']['brands_per_answer']['mean_diff']):.1f} fewer tracked brands per answer for Gemini (p = {sp[GS]['metrics']['brands_per_answer']['p_value']:.4f}) and {abs(sp[CS]['metrics']['brands_per_answer']['mean_diff']):.1f} fewer for Claude (p {'&lt; 0.001' if sp[CS]['metrics']['brands_per_answer']['p_value'] < 0.001 else '= ' + format(sp[CS]['metrics']['brands_per_answer']['p_value'], '.3f')}). The effect on {F}'s mention rate is {pp(sp[GS]['metrics']['focus_mention']['mean_diff'])} for Gemini and {pp(sp[CS]['metrics']['focus_mention']['mean_diff'])} for Claude, both within noise: with {M['intent_counts']['discovery']} prompts this comparison can only detect differences of about {pc(sp[GS]['metrics']['focus_mention']['mde_80pct'])}–{pc(sp[CS]['metrics']['focus_mention']['mde_80pct'])} (80% power). That is a statement about my design's power, not evidence of "no effect".</p>

<h2 id="stability">5 · Would the AI say the same thing twice?</h2>
{stab_table()}
<figure><img src="assets/fig_stability.png" alt="Brand list overlap between repeated answers"><figcaption>Jaccard overlap of the brand lists in repeated answers to the same prompt (1 = identical, 0 = nothing in common).</figcaption></figure>
<p>For unbranded questions, two answers to the same prompt share only {pc(min(stab[(s, 'discovery')] for s in SS))}–{pc(max(stab[(s, 'discovery')] for s in SS))} of their brands. Gemini with search is the least consistent ({stab[(GS, 'discovery')]:.2f}); Claude with search is the most ({stab[(CS, 'discovery')]:.2f}). Whether a prompt brings up {F} is also unreliable: {pc(min(R['focus_consistency'][s]['sometimes'] / R['focus_consistency'][s]['n_prompts'] for s in SS))}–{pc(max(R['focus_consistency'][s]['sometimes'] / R['focus_consistency'][s]['n_prompts'] for s in SS))} of prompts mention {F} in some repeats but not others (Claude's search surface, with only 2 repeats, has fewer chances to show this).</p>
{cons_table()}
<p>The ICC is the share of variance in "did the answer mention {F}?" that lies between prompts rather than between repeats of one prompt. Prompt-to-prompt differences set a floor that repeating cannot remove:</p>
<figure><img src="assets/fig_design.png" alt="Standard error versus number of repeats and prompts"><figcaption>Standard error of the overall mention rate under the fitted variance components. Left: with {M['intent_counts']['discovery']} prompts, even 100 repeats of each leaves a large error. Right: more prompts keep shrinking it.</figcaption></figure>
<p><b>Design rule:</b> to estimate a mention rate to ±5 percentage points (one SE) you need about {min(v['prompts_for_se5_with_3_runs'] for v in var.values())}–{max(v['prompts_for_se5_with_3_runs'] for v in var.values())} prompts with 3 repeats each, depending on the surface. This is the same power logic as an A/B test, applied to prompt design, and it is why the intervals above are as wide as they are.</p>

<h2>6 · Is there drift between days?</h2>
<p>The two Gemini surfaces were also collected two days earlier with the same prompts. If the model's behaviour shifted, answers from different days should overlap less than repeats on the same day:</p>
{drift_table()}
<p>The change in {F}'s mention rate is small and not significant, and brand lists from different days overlap about as much as repeats from the same day. Two days is a short window, so this says the measurement is stable at that scale, not that models never change. The weekly workflow is built to extend this.</p>

<h2>7 · What do the answers say about {F}?</h2>
<p>Sentiment and "recommended" flags come from the LLM reader judging how the <em>answer</em> portrays the brand (not my opinion of the brand):</p>
{sent_table()}
<p>When {F} appears in an unbranded answer it is portrayed positively almost every time and usually recommended. Asked directly whether it is worth the price, every surface gives a mostly <em>mixed</em> verdict and recommends it far less. Discovery answers list only brands the AI chooses to recommend, so they are positive almost by construction.</p>

<h2 id="sources">8 · Where do the search answers get their information?</h2>
<figure><img src="assets/fig_sources.png" alt="Most cited domains by provider"><figcaption>The ten most cited domains per search surface (share of that surface's citations; the number is how many answers cite the domain).</figcaption></figure>
{src_table()}
<ul class="tight">
<li><b>The two search tools cite very different pages.</b> Only {R['source_overlap']['shared_domains']} domains are cited by both (Jaccard {R['source_overlap']['jaccard']:.2f}). Gemini cites Reddit in {src[GS]['answers_citing_reddit']} of {src[GS]['n_grounded_answers']} answers; Claude never did in {src[CS]['n_grounded_answers']}. I did not test why (it may reflect which sites each tool can read).</li>
<li><b>The brand's own site is a small share</b> for Gemini ({pc(src[GS]['focus_brand_site_share'], 1)} of citations, {src[GS]['answers_citing_focus_site']} answers) and absent for Claude. Third-party content matters at least as much as what the brand publishes.</li>
<li class="sub">Counts are not strictly comparable: Gemini's list is the pages its answer was grounded on, Claude's is the unique pages cited in its text, and the domain types are a rough automatic labelling.</li>
</ul>

<h2 id="audit">9 · Auditing the measurement itself</h2>
<p>Both extractors agree on {pc(ex['agree'], 1)} of all answer × brand cells (Cohen's κ = {ex['kappa']:.2f}), and they give nearly the same {F} mention rate on every surface ({', '.join(f"{pc(ex['focus_rate_dictionary'][s]['rate'])} vs {pc(ex['focus_rate_llm'][s]['rate'])}" for s in SS)} by dictionary vs LLM). Reading disagreements in context, the dictionary-only hits are mostly ownership mentions ("part of the H&amp;M group", "Gap Inc.") and a few false positives; the LLM reader found no tracked brand the dictionary missed. The dictionary is the primary measure; the LLM is the robustness check and the source of sentiment. The LLM also finds brands outside my list: {pc(ex['untracked_share_of_llm_mentions'])} of its brand mentions are untracked, so share of voice here means share <em>among the {M['n_brands_tracked']} tracked brands</em>.</p>

<h2>Limitations</h2>
<ul class="tight">
<li><b>Two model families, one day.</b> ChatGPT, Perplexity, Google's AI Overviews and others may differ, and answers change over time. Claude was run at 2 repeats on its search surface.</li>
<li><b>API is not the consumer product.</b> The apps add system prompts, personalization and location that an API call does not.</li>
<li><b>The prompt set is my design.</b> Several prompts relate to Canada and everyday basics, where {F} naturally fits, so the absolute mention rate depends on the prompt mix; comparisons across surfaces and brands are more reliable than the level.</li>
<li><b>Small samples:</b> {M['intent_counts']['discovery']} unbranded prompts means wide intervals; conclusions are about patterns and design, not precise market-share numbers.</li>
<li><b>What an AI says is not what is true.</b> Sentiment is how the answer portrays a brand, judged by another model.</li>
</ul>

<h2>Reproduce and extend</h2>
<pre><code>pip install -e ".[dev]"
export GEMINI_API_KEY=... ANTHROPIC_API_KEY=...   # never commit keys
python -m brandvis.check_keys                      # free: keys valid? models available?
python -m brandvis.collect --batch 2026-11-05
python -m brandvis.extract --raw data/raw/2026-11-05.jsonl --out data/processed/2026-11-05.jsonl
python analysis/analyze.py &amp;&amp; python analysis/build_site.py
pytest                                              # 42 tests</code></pre>
<p>Collection also runs on demand from GitHub Actions. Adding another AI surface is one small provider class plus a line in <code>config/study.yaml</code>.</p>
<footer>Built by Sky Shen · <a href="{REPO}">source and raw data</a></footer>
</main>
</body>
</html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html")
