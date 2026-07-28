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

## Nine basic ways to improve academic style (UC Berkeley SLC)

From the UC Berkeley Student Learning Center worksheet
(slc.berkeley.edu/.../nine-basic-ways-improve-your-style-academic-writing). Where a
point conflicts with the punctuation rules above, the house rules win (flagged inline).

1. **Use active voice.** Prefer active over passive constructions. Reserve passive for
   when the actor is unknown or irrelevant.
2. **Mix up punctuation.** Use it correctly and with variety: semicolons join two
   complete, complementary sentences; colons introduce a list; dashes bracket an aside.
   *House override:* do not use em-dashes at all, and never use a colon for a dramatic
   reveal (functional list/equation colons only) — rephrase instead.
3. **Vary sentence structure.** Avoid a monotonous run of short, similar sentences;
   combine and restructure for flow (reinforces "Sentence rhythm" above).
4. **Avoid choppiness.** Merge fragmented, choppy sentences into smoother connected ones,
   while still keeping the occasional deliberate short declarative for punch.
5. **Avoid repetition.** Do not pair redundant synonyms ("jealous and envious"); pick one.
6. **Be concise.** Trim wordy constructions and state ideas directly.
7. **Use the vocabulary you know.** Favor clear, simple words over flashy ones you might
   misuse, but avoid weak choices such as "bad", "big", or "mad".
8. **Expand your vocabulary.** Look up unfamiliar words while reading and adopt them once
   they are comfortable and appropriate.
9. **Keep language formal.** Avoid casual phrasing; choose precise formal wording
   ("mild-mannered and kind", not "mellow and good"). Matches the earlier fix that
   replaced "not decorative" with a formal phrasing.

Guiding line: **write to express, not to impress.**

## Eleven essential academic-writing skills (EIKI guide)

From the EIKI "Complete Guide to Academic Writing" (eikipub.com). Broader than the prose
rules above; the first ones bear directly on drafting, the later ones are research-workflow
and soft skills. Kept here for completeness.

1. **Clarity and coherence.** Communicate ideas clearly and concisely with sound grammar
   and structure so the writing flows from one point to the next.
2. **Critical thinking.** Evaluate and analyze evidence, weighing the strengths and
   weaknesses of competing perspectives to support the argument.
3. **Research skills.** Locate reliable sources across journals, books, and databases and
   judge their credibility.
4. **Attention to detail.** Eliminate errors in grammar, spelling, and citation; apply the
   referencing style consistently; edit and proofread.
5. **Time management.** Prioritize, schedule, and allocate time so drafts and deadlines are
   met.
6. **Writing style.** Develop a distinctive, consistent voice and adapt language to the
   audience, purpose, and genre.
7. **Interpersonal skills.** Collaborate with peers and reviewers; listen and respond to
   feedback when revising.
8. **Adaptability.** Adjust to new genres, styles, and tools across the writing process.
9. **Organization.** Arrange ideas and present them in a clear, logical order.
10. **Independence.** Work autonomously, take initiative, and make decisions to finish the
    work.
11. **Curiosity.** Keep an inquisitive mindset and pursue meaningful, original questions.

For this paper the load-bearing four are #1 clarity, #4 attention to detail, #6 style, and
#9 organization; they operationalize the same discipline as the punctuation and rhythm
rules above.

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
