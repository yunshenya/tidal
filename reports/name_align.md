# Name alignment head

Flat-row ask decides `absent_person` by who the question names. The string matcher from #18 stays the lexical hit. A 10-feature logistic head (11 parameters) adds a nickname, cross-script pinyin, a dropped surname, or a short honorific, and clears the match when an unmatched name sits in the cue's name slot.

Always on. No flag. The bundle is `tidal/question_name_align.npz`. Ranking `W` and `alpha` (0.3) are the same surface bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`). `LLMMemory` is untouched. The relation path is the #18 cue-yield rule.

`multilingual-e5-small` is not a feature and is not loaded on this path. On the dev bank its cosine sits in one band for a real two-letter nickname (`Immy` / `Imogen` about 0.86), a two-letter stub, and an absent name next to a stored one (`Daphne` / `Lorraine` about 0.81).

## Fit

Abstain dev bank only. Question sha256 `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, n=327. The frozen adversarial test (sha256 `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327) is not an input. Threshold 0.8 and logit margin 4.0 are the dev-grid pick that keeps non-name dev questions unchanged (guard 0). These weights score **126/143** name-slice points; the string matcher scores **78/143**. The grid was not re-run after the frozen test.

A name-slice point is: the canonical person is matched on `name_variant`; the question abstains on `near_dup_absent` and on `two_names` absent; the focus person stays matched on the other `two_names` rows.

One same-script edit is a hard non-match. The dev bank uses `Harols`, `Juduth`, `Marcas`, `Anna Louize`, `吴笛`, `郑恺`, and `徐锋` on both sides, with opposite labels. A longer span that still contains the stored name in order is not a new match (`Harrold`, `老孙悦`, `孙悦然`). A prefix of exactly two letters is not a nickname (`Immy`); it is the same feature as a two-letter stub.

## Disjointness

Literal collisions are 0. Token Jaccard uses the probe tokenizer (CJK per character, Latin words lowercased). Maxima are at or under 0.50. Sentence cosine is frozen `multilingual-e5-small`, mean pool, L2, maxlen 64, no query prefix. This scoring run reproduced test vs probe and test vs surface. The other rows are the same banks as #18.

| split | against | literal | max token Jaccard | max sentence cosine |
|---|---|---:|---:|---:|
| test | probe | 0 | 0.455 | 0.939 |
| test | surface | 0 | 0.417 | 0.941 |
| test | dev | 0 | 0.500 | 0.956 |
| dev | probe | 0 | 0.400 | 0.924 |
| dev | surface | 0 | 0.500 | 0.929 |
| dev | test | 0 | 0.500 | 0.956 |

## (a) Frozen test, name slices

String is the #18 matcher. Head is this bundle, scored once. n is the same in both columns.

| slice | lang | label | n | abstained string → head | ask hits string → head | false abstain string → head |
|---|---|---|---:|---:|---:|---:|
| name_variant | en | answerable | 32 | 17 → 13 | 15 → 19 | 17 → 13 |
| name_variant | zh | answerable | 43 | 34 → 8 | 9 → 32 | 34 → 8 |
| near_dup_absent | en | absent_person | 16 | 16 → 16 | 0 → 0 | 0 → 0 |
| near_dup_absent | zh | absent_person | 14 | 10 → 10 | 0 → 0 | 0 → 0 |
| two_names | en | absent_person | 6 | 0 → 6 | 0 → 0 | 0 → 0 |
| two_names | zh | absent_person | 6 | 0 → 6 | 0 → 0 | 0 → 0 |
| two_names | en | answerable | 6 | 0 → 0 | 3 → 3 | 0 → 0 |
| two_names | zh | answerable | 6 | 0 → 0 | 3 → 3 | 0 → 0 |
| two_names | en | missing_rel | 6 | 3 → 3 | 0 → 0 | 0 → 0 |
| two_names | zh | missing_rel | 6 | 6 → 6 | 0 → 0 | 0 → 0 |

On these three slices together: answerable false abstain 51 → 21 (of 87), ask hits 30 → 57, absent-person abstain 26 → 38 (of 42). That is the keep check. The head stays.

### Name subtypes

| slice | lang | subtype | n | false abstain or missed abstain, string → head | ask hits string → head |
|---|---|---|---:|---:|---:|
| name_variant | en | nickname | 6 | 6 → 3 | 0 → 3 |
| name_variant | en | honorific_short | 3 | 3 → 2 | 0 → 1 |
| name_variant | en | typo | 8 | 8 → 8 | 0 → 0 |
| name_variant | zh | nickname | 8 | 8 → 0 | 0 → 8 |
| name_variant | zh | no_surname | 6 | 6 → 0 | 0 → 6 |
| name_variant | zh | honorific_short | 4 | 4 → 0 | 0 → 4 |
| name_variant | zh | pinyin | 8 | 8 → 0 | 0 → 5 |
| name_variant | zh | typo | 8 | 8 → 8 | 0 → 0 |
| near_dup_absent | zh | cjk_superstring | 8 | 4 → 4 still answered | 0 → 0 |
| two_names | en | absent_mentioned_with_stored | 6 | 0 → 6 abstained | 0 → 0 |
| two_names | zh | absent_mentioned_with_stored | 6 | 0 → 6 abstained | 0 → 0 |

English case, internal space, possessive, and a full honorific (`Mr. Robert`) were already hits and stay hits. English near-duplicates stay 16/16 abstained (edit distance 1, hyphen, extra letter, `Ann Marie`). Chinese edit-1 and middle-character near-duplicates stay abstained. Prefixes `老` / `小` / `大` stay abstained (`老王伟`, `小李娜`, `大张强`, `老赵敏`). The `cjk_superstring` row is this head on the previous string matcher. Current ask numbers are in `reports/name_superstring.md` (8/8 abstained).

## (b) Frozen test, full set

| | string (#18) | head |
|---|---:|---:|
| answerable false abstain | 74/216 | 44/216 |
| answerable ask hits | 127/216 | 154/216 |
| missing_rel abstained | 53/69 | 53/69 |
| absent_person abstained | 26/42 | 38/42 |

Every non-name cell matches #18: `relation_paraphrase`, `compact_form`, `bare_cue`, `misleading_cue`, and `deixis` (including indirect descriptions, 12/12 false abstain). The headline move is the name table above. False abstain drops by 30 and ask hits rise by 27, which is the name-variant change. Absent-person abstain rises by 12, which is the absent name next to a stored name.

## (c) Existing probe

Re-run on this head, same cache:

| check | result |
|---|---|
| answerable ask ret@1 | 360/370 |
| unanswerable abstained | 72/72 |
| false abstain | 0/370 |
| gold lost at overwrite 0.9 | 0/370 |
| co-resident pairs with cosine > 0.9 | 0 (max 0.8927) |
| abstain margin / ridge | 0.1 / 1.0 |
| relation alpha | 0.3 |

The surface dev curve at margin 0.1 is unchanged: false abstain 0/192 per language, missing 188/192 en and 189/192 zh, absent 36/36, deictic false abstain 0/6.

## What remains

**One-character typos, 16/16 still false abstain** (8 English, 8 Chinese). `Robart`, `Elizaboth`, `Willian`, `Catharine`, `Margoret`, `Patricis`, `Anm`, `Mary Jame`, `王玮`, `李那`, `张疆`, `刘样`, `陈靖`, `赵悯`, `黄蕾`, `周滔`. Edit distance 1 is the same surface the dev bank labels as an absent person.

**English nicknames that are not a prefix of length 3 or more.** `Bob` / `Robert`, `Bill` / `William`, and `Maggie` / `Margaret` still false-abstain. `Maggie` shares only `Ma` with `Margaret`. `Liz` / `Elizabeth`, `Cathy` / `Catherine`, and `Patty` / `Patricia` match. `Mr. Bob` and `Mr. Bill` miss with the nickname; `Ms. Jane` matches `Mary Jane`.

**Chinese longer name before a particle.** On this head, `名下是王伟强的话` matched `王伟`, and the same shape matched `李娜儿`, `张强生`, and `刘洋子` (4/8 still answered). `reports/name_superstring.md` changes the string matcher on a separate dev bank: those four abstain as `absent_person`, the four prefixes stay abstained, and `唐宁选的` still matches. The subtype row above is this head on the previous matcher.

**Pinyin matches, and three asks return the other person's fact.** All eight pinyin strings match the stored person (`Wang Wei`, `Li Na`, `Zhang Qiang`, `liuyang`, `Chen Jing`, `zhao min`, `Huang Lei`, `Zhou Tao`). Ask hits are 5/8. The three misses are a different stored value for the same relation. Ranking weights were left as they are: the probe is 360/370, and this gap is not a reason to change `W`.

**Indirect person, 12/12 still false abstain.** `addressee`, `the person being addressed`, `interlocutor`, `跟前这位`, `对面那位`. No name head is in those strings. Word lists for them were not added.

**English two-name missing relation, 3/6 still answered.** `Leave Robert aside. The telephone to read belongs with Margaret.` `Robert` is closer to the cue than `Margaret`, so the asked person is the one who has the number. The other English frame and the Chinese frames put the asked person next to the cue and abstain (3/6 en, 6/6 zh). Who wins among stored names was not changed.

**Misleading cue, bare `where` / `place` / `哪里`, and the ranker leftovers from #18.** Unchanged.
