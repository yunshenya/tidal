# Last-token name match

**Measurement only.** The shipped name head still reads `Ms. Jane` and `Jane` as stored `Mary Jane`. A hard reject of an exact final Latin token was scored on a new dev bank and on the frozen adversarial file, then left out of `_featurize`.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched (`tidal/llm_inject_smollm2_135m.pt` sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`). The name-head bundle is unchanged (`tidal/question_name_align.npz` sha256 `5a254854c3ebd3c03ebdfc4054d8f92b3fe1512e3cf0a0db1c93b69502e42749`, threshold 0.8, logit margin 4.0). Ranking `W` and `alpha` (0.3) are the same surface bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`).

The frozen questions were scored as shipped, then the reject was applied on top of that checkout. Wording of the frozen file was not edited. The reject was not an input to any fit.

## What the head does

`Ms. Jane` strips to `Jane`. Against stored `Mary Jane` the active features are `substr_len` 4, `token_frac` 0.5, and `honorific` 1. Logit 3.270, probability 0.963. Bare `Jane` drops the honorific penalty: logit 3.505, probability 0.971. Both clear 0.8. Jane is absent from the #22 alias list. The same arithmetic accepts the fit's positive, `Ms. Louise` against `Anna Louise` (logit 4.379, probability 0.988), and bare `Louise` (logit 4.614, probability 0.990).

A first token stays a prefix. `Mary` and `Ms. Mary` against `Mary Jane` keep `prefix_ge3` (logits 3.874 and 3.639). `Lou` against `Anna Louise` is a prefix of the second token, not the whole token (logit 2.950). `Gwendolyn` against `Gwendolyn Hesketh` is the same prefix. `Bob` / `Mr. Bob` against `Robert` stay on the alias list. `王先生`, `小岐`, and a CJK superstring (`王伟强` against `王伟`) stay on the CJK rules.

When the short name is also stored, the head goes quiet. `Ms. Jane` with both `Jane` and `Mary Jane` scores Jane at logit 4.284 and Mary Jane at 3.270. The gap is 1.015, under the margin 4.0, so `best_name` returns nothing. The string matcher still has `Jane`, because the full `Mary Jane` is not in the question. The ask target is Jane. That case is already the short person.

## The reject that was measured

The function below was applied by wrapping `_featurize` for the scores in this note. `tidal/question_name_align.py` on this branch is the shipped head.

Return no row when the stored name has two or more Latin tokens and the honorific-stripped span is exactly the final token. A one-token stored name is untouched, so a real `Crowe` still matches `Crowe`. A prefix (`Lou`, `Mary`, `Gwendolyn`), an alias, and every CJK feature stay. The reject is the same kind of hard drop as a longer span that still contains the name, or one same-script edit. It is not an 11th feature and the head was not refit.

On the dev bank below, that drop abstains the 33 questions whose asked person is absent and whose only hit was the final token of a longer stored name. It leaves the six questions where the full name and the last name are both stored and the last name's sentence is closer.

## Why it stays out

The same orthography is a positive on the fit and on the frozen test.

| span | stored | where | label |
|---|---|---|---|
| `Ms. Louise` | `Anna Louise` | abstain dev `honorific_short` | match, and `best_name` is `Anna Louise` |
| `Ms. Jane` | `Mary Jane` | frozen `name_variant` / `honorific_short` | answerable, gold is Mary Jane's places |

Abstain-dev name-slice points go from 126/143 to 125/143. Guard stays 0/184. The only cell that moves is `name_variant|en|honorific_short|answerable`, 3/3 to 2/3, which is `Ms. Louise`. Frozen `matched_people` changes on 1/327 questions, the row below. That ask becomes `absent_person`. Answerable hits go from 188/216 to 187/216. False abstain goes from 19/216 to 20/216. The floor on this checkout is 188/216 with false abstain held at 19. Both move the wrong way.

This is the same bind as the one-edit pairs in #24: one surface, two labels. There the conflicting pairs were dropped from the fit and the edit became a hard non-match, which kept the frozen typos abstaining. Here the positive label is the fit point and the frozen gold. Dropping it misses both.

## Dev bank

`tidal/question_lastname_dev_data.py`. The matcher does not import it. Question sha256 `064d7052e2caad6db71272315fb8af2bbfbb246a1c474970548bf424217e8920`, n=101 (85 English, 16 Chinese). The bank is 309 strings once questions and statements are deduped.

| kind | n | what the row requires |
|---|---:|---|
| honor_absent | 24 | `Ms.` / `Mr.` / `Mrs.` / `Miss` plus the last name, only the multiword person stores it: the asked person is absent |
| bare_absent | 9 | bare last name, same store: the asked person is absent |
| honor_sep | 12 | honorific plus last name, and that last name is also stored; the multiword sentence is closer: answer the last name |
| bare_sep | 6 | bare last name, same separate store: answer the last name |
| full_only | 6 | the question uses the full multiword name, only that person is stored: answer them |
| full_both | 6 | the question uses the full name, the last name is also stored and closer: answer the full name |
| single_hit | 12 | bare last name or `Ms.` plus last name, and only that one-token person is stored: answer them |
| given_keep | 3 | the first token `Gwendolyn` against `Gwendolyn Hesketh`: answer the full name |
| nick_hit | 5 | #22 diminutives `Abby`, `Mr. Mike`, `Bob`, `Ms. Ginny`, `Kate`: answer the anchor |
| nick_miss | 2 | `Dot` / `Dorothy Quinn`, `Nell` / `Helen Grace`: absent |
| cjk_super | 8 | `{name}生的` and `老{name}`: absent |
| cjk_hit | 8 | surname plus `先生` / `女士`, or `小{given}` / reduplication: answer the person |

Names: Philomena Crowe, Thaddeus Brabazon, Gwendolyn Hesketh, and the one-token last names Crowe, Brabazon, Hesketh. Distractors Lysander, Jocasta, Iphigenia never appear in a question. Chinese names 傅岐, 钟岫, 宋湄, 程岢. Anchors are `quern` / `nacre` / `skerry` and `齂` / `齃`. Diminutive controls reuse #22 anchors. The question and the stored sentence name the diminutive, so the full anchor is not a string hit in this bank. `Dolly` is a stored roster name on the nickname bank, so the compound miss is `Dot`.

Literal overlap with every reference bank is 0. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased). The cap is 0.50. Max across the 309 strings is 0.294.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 strings | 0 | 0.286 |
| surface cue corpus | — | 1124 strings | 0 | 0.278 |
| abstain dev | `26889048…` | 381 strings | 0 | 0.261 |
| frozen adversarial | `cce20b6b…` | 381 strings | 0 | 0.273 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.208 |
| nickname dev | `a766f748…` | 43 | 0 | 0.214 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.179 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.179 |
| typo dev | `68260c01…` | 68 | 0 | 0.138 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.217 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.160 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.227 |
| two-name dev | `9c72a8b3…` | 120 strings | 0 | 0.294 |
| focus dev | `03dea71d…` | 132 strings | 0 | 0.182 |

Closest pair, against the two-name bank: "On skerry tray 00b, nacre holds the birthday entered for Lysander." ‖ "No birthday was entered for Honoria on the cedar slip, brinekeel46." Closest probe pair: that same Lysander sentence ‖ "What's the birthday on file for Jules?" Closest frozen pair: "齂匣点到宋先生，要查的是生日，齃95。" ‖ "先放下周涛。要回读的生日是刘洋的。" Closest nickname pair: "Nacreous orrery records Ms. Ginny beside the telephone, quern81." ‖ "quinzaine xeraphim slip lists Ms. Ginny beside mark27."

Ask on this bank uses orthogonal sentence axes, not a new embedder. The query is a unit vector the shipped abstain ridge agrees with (phone axis 121, birthday 261, places 13; each gap is at least 0.15). The closer person's sentence is that axis plus a 0.05 tilt on axis 2, cosine 0.9988, under the 0.999 exact-sentence cancel. Everyone else's sentence is axis 2. The toy encoder is width 64 so the place strings fit in the value code. Before is the shipped head. After wraps `_featurize` with the reject and restores it.

| kind | n | before correct | after correct |
|---|---:|---:|---:|
| honor_absent | 24 | 0 | 24 |
| bare_absent | 9 | 0 | 9 |
| honor_sep | 12 | 12 | 12 |
| bare_sep | 6 | 6 | 6 |
| full_only | 6 | 6 | 6 |
| full_both | 6 | 0 | 0 |
| single_hit | 12 | 12 | 12 |
| given_keep | 3 | 3 | 3 |
| nick_hit | 5 | 5 | 5 |
| nick_miss | 2 | 2 | 2 |
| cjk_super | 8 | 8 | 8 |
| cjk_hit | 8 | 8 | 8 |
| total | 101 | 62 | 95 |

The 33 before-failures are the absent rows, and each one returns the multiword person's fact. After the reject those 33 abstain `absent_person`. Separate last names, one-token people, the given-name prefix, the diminutives, and the CJK controls stay. `full_both` returns the last name's value both times. The question contains the whole multiword name, so the string matcher hits the last token too: a capitalized word after a shorter name blocks `Ann` inside `Ann Marie`, and the last token has nothing after it. The head already refuses to score that short name from the longer span (`_extension`). The string hit remains. That is a different leak from `Ms. Jane` against a store that has no `Jane`. The reject measured here leaves it in place. A string-matcher change aimed at it would still leave `Ms. Jane` → `Mary Jane` when Jane is absent, which is the residual above.

## Frozen adversarial

Question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Scored with the committed cache on this checkout (`f7f13b1`, the #30 squash). Before is the shipped head. The name slices have one gold miss class: the 16 one-character typos, all `absent_person`, matched empty. `near_dup_absent` is 30/30 abstained. `two_names` is 36/36 on the name match and on the ask. `Ms. Jane` is a hit.

| | before | reject applied |
|---|---:|---:|
| answerable hits | 188/216 | 187/216 |
| answerable false abstain | 19/216 | 20/216 |
| missing_rel abstained | 60/69 | 60/69 |
| missing_rel still answered | 9 | 9 |
| absent_person | 42/42 | 42/42 |
| English answerable hits | 92/103 | 91/103 |
| Chinese answerable hits | 96/113 | 96/113 |
| English answerable false abstain | 11/103 | 12/103 |
| Chinese answerable false abstain | 8/113 | 8/113 |

`matched_people` under the reject differs on 1/327 questions. Ask targets are that list, so the other 326 asks stay. The one question was re-scored with the reject installed. It abstains.

| question | before | reject applied | ridge (gap) | slot scores |
|---|---|---|---|---|
| Under Ms. Jane, which ordered locations is on file? | Mary Jane `north271-ridge271-ford271` (1.066) | `absent_person`, matched `[]` | places 0.538 | Mary Jane places 1.066, Catherine places 1.022 (limited to 0.066), Margaret phone 0.817, Patricia birthday 0.753 |

Matched before is `['Mary Jane']`. Catherine's places sentence is the next raw score and is outside the ask target, so the ranker keeps Mary Jane. Scores are sentence cosine plus the relation logit (`alpha` 0.3).

Name slices:

| | en hits | en false abstain | zh hits | zh false abstain | near_dup abstain | en honorific_short hits |
|---|---:|---:|---:|---:|---:|---:|
| before | 24/32 | 8/32 | 35/43 | 8/43 | 30/30 | 3/3 |
| reject applied | 23/32 | 9/32 | 35/43 | 8/43 | 30/30 | 2/3 |

The eight English and eight Chinese false abstains are the typos (`Robart`, `Elizaboth`, `Willian`, `Catharine`, `Margoret`, `Patricis`, `Anm`, `Mary Jame`, `王玮`, `李那`, `张疆`, `刘样`, `陈靖`, `赵悯`, `黄蕾`, `周滔`). The reject adds `Ms. Jane` beside them. `Mr. Bob` and `Mr. Bill` stay hits. Chinese `王先生` / `李女士` / `张先生` / `陈小姐` stay hits.

`two_names` is unchanged, and the #29 / #30 floors hold.

| | en missing_rel abstained | zh missing_rel abstained | en answerable hits | zh answerable hits | absent both languages |
|---|---:|---:|---:|---:|---:|
| before | 6/6 | 6/6 | 6/6 | 6/6 | 12/12 |
| reject applied | 6/6 | 6/6 | 6/6 | 6/6 | 12/12 |

The other 19 false abstains and the nine `missing_rel` rows that are still answered are the same rows as #30. None of them is this name shape. False abstain: the two English compact-form phone rows ("How can Robert be reached after dusk?", "How do I reach William from the quay after dark?"), the Mary Jane locations-file row, and the 16 typos. Still answered: "How do I get through to Elizabeth once the office is dark?" (places gap 0.031), "How can Elizabeth be reached after dusk?" (places gap 0.247), "How do I reach Catherine from the quay after dark?" (places gap 0.519), "When does Patricia mark another trip around the sun?" (birthday gap 0.095), "Where should the year-older mark for Ann be placed?" (places gap 0.022), "Ignore the birthday feast and say how to get through to Catherine." (birthday gap 0.258), "怎样才能够着周涛？" (birthday gap 0.072), "电话簿先合上，赵敏常去的几处按先后是什么？" (places gap 0.070), "地方名单先放下，怎么用声音找到刘洋？" (phone gap 0.089).

## Probe

`eval_heldout` on this checkout, shipped head, cache sha256 `bb994f4e85a3c06453d43af8c59eb3e6b352035bef14f2f208e5140b6b553f8b`:

| check | result |
|---|---|
| answerable ask ret@1 | 365/370 |
| false abstain | 0/370 |
| English / Chinese answerable | 183/185, 182/185 |
| phone / birthday / places | 122/124, 122/124, 121/122 |
| missing_rel | 36/36 |
| absent_person | 36/36 |
| unanswerable | 72/72 |
| gold lost at overwrite 0.9 | 0/370 |
| co-resident pairs, cosine > 0.9 | 0 of 8390, max 0.8927 |

Surface dev at margin 0.1, from that same run: false abstain 0/192 English and 0/192 Chinese, missing 188/192 and 189/192, absent 36/36, deictic false abstain 0/6.

`matched_people` under the reject differs on 0/442 probe questions and on 0/43 nickname-dev questions. The probe's stored names are one token, and the nickname rows resolve through the alias list or a prefix. The reject has nothing to fire on, so those asks stay at the table above. The shipped head is what this checkout runs.

## Tests

`python3 -m pytest -q`: 285 passed, 1 skipped. The skip is the existing `tests/test_llm_memory.py` case that imports `transformers`, which this environment does not have. The new file locks the dev sha, the Jaccard cap, literal disjointness, the shipped `Ms. Jane` / `Ms. Louise` hits, and the dev ask before and after the reject (62/101, then 95/101) with the reject removed again before the test returns. Frozen numeric scores are not a pytest gate.
