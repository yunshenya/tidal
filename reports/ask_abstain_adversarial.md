# Adversarial ask abstain

Flat-row `ask` on a held-out English and Chinese set written apart from the cue list, the probe, and the surface fit/dev phrasing. The test strings were frozen before any abstain score (question sha256 `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327). A separate dev split (sha256 `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, n=327) is the only generated bank the rule change consulted. The test was scored as shipped, then once after the change. Wording was not edited after the baseline.

Abstained: `ask` returned no fact. Still answered: a stored fact was returned. Ask hits: answerable and the decode is a gold value. False abstain: answerable and abstained. On an unanswerable row, still answered is the missed abstain.

## Disjointness

Literal collisions with the probe, the surface corpus, and the sibling split are 0. Token Jaccard uses the probe tokenizer (CJK per character, Latin words lowercased). Maxima are at or under 0.50, under the probe bar of 0.65. Sentence cosine is frozen `multilingual-e5-small`, mean pool, L2, maxlen 64, no query prefix.

| split | against | literal | max token Jaccard | max sentence cosine |
|---|---|---:|---:|---:|
| test | probe | 0 | 0.455 | 0.939 |
| test | surface | 0 | 0.417 | 0.941 |
| test | dev | 0 | 0.500 | 0.956 |
| dev | probe | 0 | 0.400 | 0.924 |
| dev | surface | 0 | 0.500 | 0.929 |
| dev | test | 0 | 0.500 | 0.956 |

Closest Jaccard pair, test vs probe: "The birthday of mine, which date is kept?" ‖ "Past the decoys, which date is Nico's birthday?"

## Sizes (test)

| slice | n | en | zh | what it is |
|---|---:|---:|---:|---|
| relation_paraphrase | 48 | 24 | 24 | cue-free indirect wording and idioms, answerable and missing_rel |
| compact_form | 26 | 14 | 12 | short forms, including `how do I reach`, `几号出生`, `住在哪边`, `hang out` |
| bare_cue | 16 | 8 | 8 | `place` / `where` / `哪里` / `位置` where they are not relation cues |
| misleading_cue | 24 | 12 | 12 | `call` / `birthday` / `place` (and the Chinese cue words) used for a different relation |
| name_variant | 75 | 32 | 43 | nickname, case, space, honorific, possessive, no surname, pinyin, 1-char typo |
| near_dup_absent | 30 | 16 | 14 | absent people whose names are near a stored name |
| two_names | 36 | 18 | 18 | both have the relation, only the other has it, or an absent name sits next to a stored one |
| deixis | 72 | 36 | 36 | second person, reflexive, indirect, first person, unique and ambiguous pronouns |

Labels: answerable 216, missing_rel 69, absent_person 42. Languages: en 160, zh 167.

## Headline, test set

| | before | after |
|---|---:|---:|
| answerable false abstain | 135/216 | 74/216 |
| answerable ask hits | 71/216 | 127/216 |
| missing_rel abstained | 37/69 | 53/69 |
| absent_person abstained | 16/42 | 26/42 |

Cue-free paraphrases were already answered (24/24 hits, 0 false abstain) and stayed there. The movement is pronouns, name edges, a cue word used for another relation, and two stored names.

### Test, per slice and language

| slice | lang | label | n | abstained before → after | still answered | ask hits | false abstain |
|---|---|---|---:|---:|---:|---:|---:|
| bare_cue | en | answerable | 4 | 1 → 1 | 3 → 3 | 3 → 3 | 1 → 1 |
| bare_cue | en | missing_rel | 4 | 3 → 3 | 1 → 1 | 0 → 0 | 0 → 0 |
| bare_cue | zh | answerable | 4 | 1 → 1 | 3 → 3 | 3 → 3 | 1 → 1 |
| bare_cue | zh | missing_rel | 4 | 2 → 2 | 2 → 2 | 0 → 0 | 0 → 0 |
| compact_form | en | answerable | 7 | 2 → 2 | 5 → 5 | 5 → 5 | 2 → 2 |
| compact_form | en | missing_rel | 7 | 5 → 5 | 2 → 2 | 0 → 0 | 0 → 0 |
| compact_form | zh | answerable | 6 | 0 → 0 | 6 → 6 | 6 → 6 | 0 → 0 |
| compact_form | zh | missing_rel | 6 | 5 → 5 | 1 → 1 | 0 → 0 | 0 → 0 |
| deixis | en | answerable | 36 | 30 → 6 | 6 → 30 | 6 → 30 | 30 → 6 |
| deixis | zh | answerable | 36 | 30 → 6 | 6 → 30 | 2 → 21 | 30 → 6 |
| misleading_cue | en | answerable | 6 | 6 → 3 | 0 → 3 | 0 → 3 | 6 → 3 |
| misleading_cue | en | missing_rel | 6 | 0 → 4 | 6 → 2 | 0 → 0 | 0 → 0 |
| misleading_cue | zh | answerable | 6 | 6 → 4 | 0 → 2 | 0 → 2 | 6 → 4 |
| misleading_cue | zh | missing_rel | 6 | 0 → 3 | 6 → 3 | 0 → 0 | 0 → 0 |
| name_variant | en | answerable | 32 | 21 → 17 | 11 → 15 | 11 → 15 | 21 → 17 |
| name_variant | zh | answerable | 43 | 38 → 34 | 5 → 9 | 5 → 9 | 38 → 34 |
| near_dup_absent | en | absent_person | 16 | 10 → 16 | 6 → 0 | 0 → 0 | 0 → 0 |
| near_dup_absent | zh | absent_person | 14 | 6 → 10 | 8 → 4 | 0 → 0 | 0 → 0 |
| relation_paraphrase | en | answerable | 12 | 0 → 0 | 12 → 12 | 12 → 12 | 0 → 0 |
| relation_paraphrase | en | missing_rel | 12 | 10 → 10 | 2 → 2 | 0 → 0 | 0 → 0 |
| relation_paraphrase | zh | answerable | 12 | 0 → 0 | 12 → 12 | 12 → 12 | 0 → 0 |
| relation_paraphrase | zh | missing_rel | 12 | 12 → 12 | 0 → 0 | 0 → 0 | 0 → 0 |
| two_names | en | absent_person | 6 | 0 → 0 | 6 → 6 | 0 → 0 | 0 → 0 |
| two_names | en | answerable | 6 | 0 → 0 | 6 → 6 | 3 → 3 | 0 → 0 |
| two_names | en | missing_rel | 6 | 0 → 3 | 6 → 3 | 0 → 0 | 0 → 0 |
| two_names | zh | absent_person | 6 | 0 → 0 | 6 → 6 | 0 → 0 | 0 → 0 |
| two_names | zh | answerable | 6 | 0 → 0 | 6 → 6 | 3 → 3 | 0 → 0 |
| two_names | zh | missing_rel | 6 | 0 → 6 | 6 → 0 | 0 → 0 | 0 → 0 |

## What changed

Always on. No flag. Ranking `W` and `alpha` (0.3) are the same bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`).

1. **Deixis.** A question with no stored name is answered when it is second person (`you` / `your` / `yourself`, `你` / `您`), first person (`I` / `me` / `my` / `mine`, `我`), or a pronoun (`he` / `she` / `they` / `his` / `her` / `their`, `他` / `她`). `我们` stays out, because `我` is followed by `们`. `其他` stays out. `them` stays out: the probe's absent questions say "if we had heard them" about a fact, and treating `them` as a person would stop those abstains.

2. **Name edges.** Latin: case-insensitive, optional space inside a token (`Ro bert`), flexible space between the tokens of a multi-token name (`Mary  Jane`, `MaryJane`). A hyphen or another capitalized word blocks a shorter name (`Ann-Marie`, `Ann Marie`, `Nellie Jane`). `Ada's` still matches. CJK: optional space between characters (`王 伟`, including an ideographic space). `老` / `小` / `大` / `阿` glued to the front is a different person. One extra character closed off by punctuation is a different person (`孙悦然，` on the dev split). A following verb or title still matches (`孙悦添岁`, `刘洋先生`, `姜北对应`, `唐宁选的`).

3. **Cue vs ridge.** A unique lexical cue yields when the abstain ridge disagrees by a top-2 gap of at least 0.15. That cutoff sits above the one probe question the ridge misreads ("What places did Ravi store?", gap about 0.12) and above the one surface-dev cue it misreads (gap about 0.02). The dev misleading rows are where this was checked. Below 0.15 the cue still wins, and it still ignores the 0.1 margin.

4. **Two stored names.** With one lexical cue, only the stored names nearest that cue are the asked people. The dev check: Chinese missing-relation rows go from 0/6 abstained to 6/6; English from 0/6 to 3/6, on the sentences where the asked person is the one next to the cue. Answerable rows, where both people have the relation, do not pick up a false abstain.

## Existing probe

Re-run after the change, same cache:

| check | result |
|---|---|
| answerable ask ret@1 | 360/370 |
| unanswerable abstained | 72/72 |
| false abstain | 0/370 |
| gold lost at overwrite 0.9 | 0/370 |
| co-resident pairs with cosine > 0.9 | 0 (max 0.893) |
| abstain margin / ridge | 0.1 / 1.0 |
| relation alpha | 0.3 |

The surface dev curve is unchanged (false abstain 0/192 per language, missing 188 en / 189 zh, absent 36/36, deictic false 0/6). Pytest is green, including the held-out probe, the question head, and `LLMMemory`. `LLMMemory` is untouched.

## What remains, and why

**Indirect person, 12/12 false abstain on the test (6 en, 6 zh).** `addressee`, `the person being addressed`, `interlocutor`, `跟前这位`, `对面那位`. These are open descriptions. The two splits use different words on purpose, so a word list would be the lists themselves. Closed-class pronouns are what shipped.

**Nicknames, pinyin, given name without the surname, short honorifics, 1-character typos.** Still false abstains: en nickname 6, honorific short 3, typo 8; zh nickname 8, pinyin 8, no surname 6, honorific short 4, typo 8. There is no alias table. Edit distance 1 cannot both accept a typo and reject the same edit as an absent person: the dev split uses `Harols`, `Juduth`, `Marcas`, and `Anna Louize` on both sides, with opposite labels. Lowercase, possessive, full honorific (`Mr. Robert`, `刘洋先生`), and internal spaces do match.

**Chinese name plus one character, then `的`.** At this measurement, `名下是王伟强的话` matches `王伟` (4/8 test superstrings: 王伟强, 李娜儿, 张强生, 刘洋子). That shape is also how the probe attaches a verb: `唐宁选的`, `祁安报过`, `姜北对应`. A superstring closed by punctuation does abstain (`孙悦然，` on the dev split, 8/8). Prefixes do abstain on the test too (`老王伟`, `小李娜`, `大张强`, `老赵敏`). Latin hyphen and `Ann Marie` abstain (16/16 English near-duplicates). The later split is `reports/name_superstring.md`: those four longer names abstain, and `唐宁选的` still matches.

**An absent name next to a stored name, 12/12 still answered.** The stored name is in the question, so the rule treats the person as present. Spotting an unstored person as "any capitalized word" or "any two characters before `的`" abstained real probe and dev questions (`Green-diary`, `Set the birthday…`, `林晓列出的地方`). That spotter was not shipped.

**English two-name missing relation, 3/6 still answered.** `Leave Robert aside. The telephone to read belongs with Margaret.` `Robert` is closer to `telephone` than `Margaret`, so the union-narrowing keeps the person who has the number. The other English frame, and the Chinese frames, put the asked person next to the cue and abstain (3/6 en, 6/6 zh).

**Misleading cue when the ridge agrees with the cue word, or the gap is under 0.15.** Test answerable false abstain 6→3 en and 6→4 zh. Missing-relation caught 0→4 en and 0→3 zh. The sentence contains the cue, and the ridge often repeats it. The 0.15 floor is the probe question above.

**Bare `where` / `place` / `哪里` and a few cue-free lines.** Unchanged, and small: answerable false abstain 1/4 en and 1/4 zh on bare cues, 2/7 en on cue-free compact forms. The label is confident and wrong, or the gap is under 0.1. Margins 0.0 and 0.05 each false-abstain one English surface-dev question, so the shipped margin stays 0.1. The abstain ridge was not refit. Relation paraphrases with no cue word were already 24/24 hits.

**Ask hits after the abstain lets go.** Some of these are the ranker. Chinese second person was already not abstaining and hits 2/6. Chinese `您`, once answered, hits 2/6. Two-name answerable hits stay 3/6 en and 3/6 zh because both people store the relation and the other person's fact sometimes wins. Ranking weights were left as they are.
