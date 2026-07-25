# Writing style for this paper (human, not AI-sounding)

Notes for keeping `writeup/volatility_persistence_paper.tex` reading as
human-authored. Specific to this paper and its target venue (an ICAIF workshop
short paper). Calibrated against a 2026 read of five accepted ICAIF / arXiv finance
papers (statistical arbitrage, LightGCN recommenders, MVECF, FinRL ensembles,
FINSABER).

The tells that make prose look LLM-generated are not the *presence* of rhetorical
devices. Real authors use them too. The tells are **repetition, decoration, and
uniformity**. A human uses each device once, where it does real work, and varies
sentence length hard. An LLM uses every device three times and writes uniformly
medium-long sentences. Aim for restraint and rhythm variance.

## Punctuation

- **Em-dashes (`---`): do not use.** Two of the five papers use zero; the rest use a
  handful only to append a concrete appositive, never for drama or rhythm. Convert
  every em-dash to a period, comma, parentheses, or a restructured sentence.
  Leave en-dashes (`--`) alone in numeric ranges (`0.32--0.34`) and hyphenated
  names (`Benjamini--Hochberg`).
- **Colons: functional only, never a dramatic reveal.** The AI-tell is
  `... unsurprising: no metric beats volatility` and `A clean law: magnitude, not
  shape`. Real papers use colons only to introduce a list, an equation, a section
  roadmap (`organized as follows:`), or a single setup-question. Rewrite reveal
  colons into two sentences or a comma clause.
  - KEEP: the title colon (`Rolling Forward: Enhancing LightGCN`), section-heading
    colons, figure-caption `Label: description`, bibliography `volume(issue):pages`.
    Stripping these makes the paper look *less* like a real ICAIF paper.
- **Semicolons: only to separate list items that themselves contain commas**, or to
  join two tightly parallel independent clauses. Not as a rhythm crutch.
- **Parentheses: technical only** (citations, acronym definitions, numeric
  qualifiers, `e.g.` examples). Not for author-voice asides.

## Openings

- Name the object in the first few words. `Statistical arbitrage exploits temporal
  price differences between similar assets.` Not `In an era of...`, not an aphorism.
- State the gap fast. Four of five papers pivot to the problem by **sentence 2**,
  usually with **However** or **yet** (`..., yet stock recommender systems have
  received limited attention`).
- No grand stage-setting. No rhetorical question as the first sentence.

## Sentence rhythm (the strongest human signal)

- **Interleave genuinely short declaratives.** Real papers drop 3--8 word sentences
  between longer ones: `The trick is quite simple.` / `LLM strategies perform
  poorly.` / `This limitation is especially important in finance.`
- Vary length hard. A run of terse claims, then one long qualified sentence. Uniform
  medium-long sentences are the clearest LLM tell.

## Contributions

- Use **varied verbs**: propose / show / reassess / conduct / reveal / offer. Do not
  write four consecutive `We propose`. Verb monotony is a tell.
- State quantified results flatly, no `surprisingly` / `remarkably`.

## Transitions

- Workhorses: **However, yet, In contrast, First/Second/Finally, Hence, Thus.**
- Prefer **Thus / Hence** over `Therefore`.
- Avoid `Moreover`/`Furthermore` pile-ups and exotic connectives. Often the cleanest
  transition is none: just start with the content.

## AI-tell budget (each at most once, load-bearing)

- `not X, but Y` and `not only ... but also`
- `crucially`, `importantly`, `notably`, `it is worth noting` (best avoided entirely)
- a rhetorical question
- a decorative triad (three parallel items purely for cadence). Factual triads that
  enumerate real things (`bull, bear, and sideways`) are fine and unlimited.

## Self-audit (run before declaring a draft done)

```bash
cd writeup
grep -c -- "---" volatility_persistence_paper.tex                 # em-dashes: want 0
grep -nE ": [a-z]" volatility_persistence_paper.tex | grep -v cite  # reveal colons in prose
grep -onE "not [^,]*, but|not only" volatility_persistence_paper.tex | wc -l   # want <= 1
grep -oiE "moreover|furthermore|crucially|it is worth noting" volatility_persistence_paper.tex
```

Then read the abstract and intro aloud. If every sentence is the same length, break
some. If a colon sets up a punchline, split it into two sentences.
