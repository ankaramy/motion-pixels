# 05 — Style Guide (from the author's edited text)

This guide is inferred **primarily from the author's manually edited calibration passage**, not from generic
academic-writing rules. When stylistic choices conflict, the order of authority is:

**author's edited text  >  previously generated text  >  generic academic convention.**

Reference sample (LOCKED): `THESIS/STYLE_CALIBRATION_horizon_rollouts.md`, paragraphs 1–4.

**LOCKED convention:** any section the author marks LOCKED is frozen. Do not rewrite, re-phrase, "polish,"
or correct it (including spelling) unless the author explicitly asks. New sections should match the locked
material's voice without touching it.

---

## What the author changed (generated draft → edited text)

Small edits, but they define the voice. Each lesson below is tied to an actual change.

1. **No clipped aphoristic fragments.** My "A horizon is just that. It is how…" became
   "And that's exactly what a horizon is: It is how…". The two-word fragment was smoothed into one
   connected sentence. → Avoid the punchy sentence-fragment tic. Short sentences are fine; *fragments that
   exist for rhythm* are not.
2. **First person is selective, not constant.** My "I kept the horizons separate on purpose" became
   "The horizons were kept separated on purpose." The author will use passive voice when the subject is the
   method or the horizons. First person is reserved for things the author actually did or judges
   ("I trained and evaluated five of them", "That is the number I watch most"). → Don't force "I" into
   every process sentence.
3. **Fuse the reason, then break out the contrast.** My "Each one is its own… , because asking a network…"
   became "…because each one is its own trained model with its own checkpoint. Asking a network to commit
   to twenty metres is a different problem…". The cause is attached with "because"; the contrasting idea
   gets its own sentence. → Prefer this over long subordinate chains.
4. **Plain verbs over idioms.** "hold up well" → "perform well". → Choose the neutral technical verb.
5. **Conversational connectives are allowed.** The author starts a sentence with "And" and uses a colon to
   introduce an elaboration. Contractions are acceptable ("that's"). → The register is relaxed, not stiff.
6. **Mild hypothetical softening.** "covers that ground" → "would cover that ground". → Use "would" for
   hypothetical/illustrative statements.
7. **Natural, not over-polished.** The edit left a spelling slip ("seperated") in place. The author is not
   chasing a glossy surface. → Don't over-refine. (Do not *introduce* errors either; a later proofread pass
   handles spelling. Locked text is left exactly as written.)

---

## Sentence rhythm

Measured from the edited paragraphs. This is the target rhythm.

- **High variation within a paragraph.** Word counts per sentence run, for example: 16, 28, 7, 30, 6, 30
  (P1) and 5, 25, 8, 19, 21 (P3). Long evidence sentences sit next to short declaratives.
- **Average sentence length ≈ 18–20 words.**
- **Short sentences are complete declaratives**, not fragments: "The numbers are frames, not minutes."
  (6) · "For an architect that is a usable signal." (8) · "The short horizons perform well." (5).
- **Longest sentences (28–30 words)** carry the technical payload (the horizon list, the metre mapping) and
  are still one clean clause chain, not nested subclauses.
- **Paragraph length:** 3 to 6 sentences, roughly **60–110 words**. Paragraphs vary; do not standardise.

## Paragraph shape

- Paragraphs often **open with a short topic line** (5–8 words) and then expand into the evidence.
  "The short horizons perform well." → numbers → interpretation.
- **No fixed topic-sentence → evidence → conclusion template.** Some paragraphs end on the evidence; some
  end on a quiet reflection. Do not add a mini-conclusion to every paragraph.

## First person and distance

- Use **"I"** for author actions and judgements: trained, evaluated, kept (sometimes), watch, would rather.
- Use **passive / method-as-subject** for process facts: "The horizons were kept separated", "the model is
  asked to predict". Both appear within a few lines of each other. Alternate naturally.
- Academic distance is **moderate**. The author speaks plainly to the reader, occasionally addresses the
  architect ("For an architect that is a usable signal").

## Moving between technical and architectural language

- State the metric plainly and specifically, then translate it into what it means for reading space.
  Pattern: *number → "For an architect that is a usable signal" → what it says about intention.*
- Keep numbers **inline and concrete** (0.45 m, 83%, H100). Round with "roughly", "about", "around".
- Architectural reflection is **short and embedded**, not a separate lyrical passage. One or two sentences,
  then back to the evidence.

## Introducing evidence

- Numbers arrive directly, without throat-clearing. No "the results demonstrate that". Just "At H20 the
  average displacement error sits around 0.45 m".
- Comparison and trend are described in plain terms: "falls along the same line", "runs higher still".

## Qualifying claims

- Hedge with **"roughly / about / around"** on numbers, and with **"would"** on hypotheticals.
- Honesty is explicit and calm: "where the method holds and where it starts to fail." The author states
  limits without drama and without over-apologising.

## Conclusions and closings

- Phrased quietly and concretely. Endings state a consequence, not a grand claim: "Reporting them as a
  single number would have hidden where the method holds and where it starts to fail."
- Avoid summarising flourishes and "in conclusion" gestures inside a section.

## Vocabulary

**Natural to the author:** perform, sits (around), lands, holds, falls, glimpse, usable signal, legible,
intention, tendency, commit (to a distance), agrees with, watch (a number), plain and specific number words.

**Avoid** (generated-writing habits the author does not use): Furthermore, Moreover, Additionally,
Ultimately, Importantly, Interestingly, This highlights / demonstrates / underscores, It is important /
worth noting, By leveraging, valuable insights, innovative approach, significant advancement, robust
framework, holistic, multifaceted, paradigm, "not only… but also…".

**Also avoid:** em dashes (zero, always), excessive three-part lists, rhetorical questions, over-neat
paragraph symmetry, generic filler, restating the same idea in new words, announcing what you are about to
say, inflated vocabulary, unnecessary passive voice (passive is fine when the method is the subject; not as
a default).

## Punctuation and mechanics

- **Zero em dashes.** Use commas, colons, or full stops. The author uses a colon to introduce an
  elaboration and commas to extend a sentence.
- Contractions allowed, used sparingly ("that's").
- Starting a sentence with "And", "so", "because" is acceptable.
- Units: metres as "m", horizons as H20/H60/H100/H200/H400, distances "roughly 1, 3, 5, 10 and 20 metres".
  Horizons are frames/steps, never minutes.

## Content discipline (unchanged, carries into every section)

- Only claims supported by project material. No invented results, numbers, citations, experiments, or
  capabilities. Missing fact → `[AUTHOR INPUT REQUIRED]`. Unclear citation → `[CITATION REQUIRED]`.
- Be transparent about which model produced which figure (baseline MODEL_X vs final MODEL_XC).
- This is a thesis in AI for Architecture and the Built Environment, not a software report. Explain code
  only where it matters to the research argument.

---

## Quick pre-flight checklist for each new section

- [ ] Sentence lengths vary; average ~18–20 words; at least one short declarative per paragraph.
- [ ] No sentence fragments used for effect.
- [ ] First person only for real author actions/judgements; method-as-subject otherwise.
- [ ] Metric stated → translated into architectural meaning, briefly.
- [ ] Numbers inline, hedged with roughly/about/around.
- [ ] No banned connectives or inflated vocabulary.
- [ ] Zero em dashes.
- [ ] No mini-conclusion tacked onto every paragraph.
- [ ] Any unsupported fact marked `[AUTHOR INPUT REQUIRED]`.
- [ ] Word count within ±5% of the subsection target.
