# Name typos at edit distance 1

**Measurement only.** No matcher change. A one-character typo and an absent near-duplicate are the same name decision on this data, and the new dev bank labels that pair both ways. Shipping a gate that recovers a frozen typo also answers a row that must stay absent.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched (`tidal/llm_inject_smollm2_135m.pt` sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`). The name-head bundle is unchanged (`tidal/question_name_align.npz` sha256 `5a254854c3ebd3c03ebdfc4054d8f92b3fe1512e3cf0a0db1c93b69502e42749`, dev sha `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, 126/143 name-slice points, threshold 0.8, logit margin 4.0). Ranking `alpha` stays 0.3 on the surface bundle `820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`. The frozen adversarial questions were scored after the decision to leave the matcher alone. They were not an input to the wording.

Closed PR #24 reached the same separability conclusion on an earlier bundle (false abstain 39/216). Current main, after #25–#30, is 188/216 hits and 19/216 false abstain. The name decision is the same one. Closed PR #31 showed `Ms. Jane` → `Mary Jane` is gold-labeled answerable on the frozen file. That hit stays. The #19 head, the #21 CJK superstring reject, and the #22 diminutive list are untouched.

## What the 16 misses are

Frozen question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. The remaining name-slice gold misses are the typo subtype: `name_variant` English 8/32 false abstain, Chinese 8/43 false abstain. Every other name-variant subtype on current main is a hit (nickname, pinyin, dropped surname, short honorific, lowercase, possessive, full honorific, spacing). `near_dup_absent` is 16/16 English and 14/14 Chinese.

The head never scores an equal-length substitution. `_ambiguous_edit` drops one same-script edit whose length differs by at most 1, and `_featurize` returns nothing on that drop. With the span and the name the same length, every cheap feature is also zero: a prefix requires a shorter span. Deletions are the exception (`Hartly` / `Hartley` lights `prefix_ge3`) and the edit gate drops them before the vote. Coefficients on the shipped head: `prefix_ge3` 0.369, `prefix_eq2` −3.5064, `substr_len` 0.5547, `token_frac` 0.1174, `pinyin_equal` 0.6238, `given_nick` 0.6238, `given_only` 0.5138, `surname_title` 0.3823, `exact_fold` 0.9558, `honorific` −0.2354. There is no positive substitution example left to fit. The abstain-dev both-label pairs (`Harols` / `Harold`, `Juduth` / `Judith`, `Marcas` / `Marcus`, `Anna Louize` / `Anna Louise`, `吴笛` / `吴迪`, `郑恺` / `郑凯`, `徐锋` / `徐峰`) are dropped in `_pairs_from_dev`.

On each frozen typo row, `diagnose` reports the typo span as unknown, `matched` empty, and `force_absent` false. `force_absent` needs a known stored name beside the unknown one. `surface_abstain` still returns `absent_person`: no stored name was selected, and the question is not deictic. The gold fact is already rank 1 among the stored sentences (sentence cosine plus `alpha` 0.3 times the relation logit). The ridge gap on every typo row is above 0.15, and the top ridge label is the asked relation. The miss is the name.

Scores below are that cache ranking on the committed vectors. The row roster is who the question stores. The full-list distance is the next name on the eight-person frozen roster.

| surface | stored | edit | keyboard | row next | full-list next | ridge (gap) | top two scores |
|---|---|---|---|---|---|---|---|
| Robart | Robert | a→e | no | Margaret 5 | Ann 5 | phone 1.0154 | Robert phone `+1-917-555-1299` 1.0984, William phone 1.0708 |
| Elizaboth | Elizabeth | o→e | no | Margaret 7 | Margaret 7 | birthday 0.5249 | Elizabeth birthday `1953-04-07` 0.9913, Catherine birthday 0.9706 |
| Willian | William | n→m | yes | Mary Jane 6 | Ann 6 | places 0.6245 | William places `north307-…` 1.0248, Margaret places 0.9981 |
| Catharine | Catherine | a→e | no | Mary Jane 6 | Margaret 6 | phone 0.8216 | Catherine phone `+1-917-555-1311` 1.0173, Patricia phone 0.9904 |
| Margoret | Margaret | o→a | no | Mary Jane 5 | Mary Jane 5 | birthday 0.5546 | Margaret birthday `1965-04-19` 0.9907, Mary Jane birthday 0.9757 |
| Patricis | Patricia | s→a | yes | Ann 7 | Catherine 6 | places 0.7103 | Patricia places `north319-…` 1.0686, Ann places 1.0234 |
| Anm | Ann | m→n | yes | Robert 6 | Mary Jane 6 | phone 1.0720 | Ann phone `+1-917-555-1323` 1.0840, Robert phone 1.0440 |
| Mary Jame | Mary Jane | m→n | yes | William 6 | Margaret 4 | birthday 0.4142 | Mary Jane birthday `1977-04-04` 0.9572, Elizabeth birthday 0.9021 |
| 王玮 | 王伟 | 玮→伟 | — | 刘洋 2 | 刘洋 2 | places 0.3109 | 王伟 places `north901-…` 1.0265, 李娜 places 1.0103 |
| 李那 | 李娜 | 那→娜 | — | 刘洋 2 | 刘洋 2 | phone 0.4960 | 李娜 phone `+1-917-555-1905` 1.0977, 张强 phone 1.0786 |
| 张疆 | 张强 | 疆→强 | — | 刘洋 2 | 刘洋 2 | birthday 0.3521 | 张强 birthday `1999-10-19` 1.0779, 刘洋 birthday 1.0574 |
| 刘样 | 刘洋 | 样→洋 | — | 赵敏 2 | 周涛 2 | places 0.4666 | 刘洋 places `north913-…` 1.0553, 陈静 places 1.0432 |
| 陈靖 | 陈静 | 靖→静 | — | 周涛 2 | 刘洋 2 | phone 0.6459 | 陈静 phone `+1-917-555-1917` 1.1013, 赵敏 phone 1.0802 |
| 赵悯 | 赵敏 | 悯→敏 | — | 周涛 2 | 刘洋 2 | birthday 0.4941 | 赵敏 birthday `1941-10-04` 1.0805, 黄磊 birthday 1.0644 |
| 黄蕾 | 黄磊 | 蕾→磊 | — | 周涛 2 | 刘洋 2 | places 0.2398 | 黄磊 places `north925-…` 1.0111, 周涛 places 0.9809 |
| 周滔 | 周涛 | 滔→涛 | — | 张强 2 | 刘洋 2 | phone 0.6512 | 周涛 phone `+1-917-555-1929` 1.1188, 王伟 phone 1.1031 |

Keyboard adjacency is unstaggered QWERTY, Chebyshev distance 1, diagonals included. `a`/`e` is not adjacent. `m`/`n`, `s`/`a`, and `s`/`d` are. `y`/`i` is not.

Chinese shape, direct IDS operands only (cjkvi-ids, the bracketed source tag stripped). A recursive walk shares `一` for `蕾`/`磊` and is not used.

| pair | pinyin | direct operands | shape |
|---|---|---|---|
| 玮 / 伟 | wei = wei | share 韦 | same pinyin, shared operand |
| 那 / 娜 | na = na | 娜 is ⿰女那 | same pinyin, containment |
| 疆 / 强 | jiang ≠ qiang | share 弓 | different pinyin, shared operand |
| 样 / 洋 | yang = yang | share 羊 | same pinyin, shared operand |
| 靖 / 静 | jing = jing | share 青 | same pinyin, shared operand |
| 悯 / 敏 | min = min | no shared operand | same pinyin, nothing shared |
| 蕾 / 磊 | lei = lei | 艹雷 vs 石砳 | same pinyin, nothing shared |
| 滔 / 涛 | tao = tao | share 氵 | same pinyin, shared operand |

`黄蕾`, `周滔`, and `陈靖` are also the absent near-duplicates. The answerable typo and the absent row use the same surface and the same edit-1 stored name. Distractors differ. A function of (surface, that stored name) cannot return both labels. The abstain dev bank already has this collision for `Harols`, `Juduth`, `Marcas`, `Anna Louize`, `吴笛`, `郑恺`, and `徐锋`. `tests/test_question_name_align.py` locks those as non-matches.

English near-duplicates that must stay absent, row roster next to the edit-1 name: `Robern`→Robert (Margaret 7), `Williem`→William (Patricia 7), `Catherina`→Catherine (Ann 7), `Mary Jone`→Mary Jane (Ann 6), `Patrici`→Patricia deletion (Robert 7), `Roberrt`→Robert insertion and an extension. On the full eight-person roster the next distances are Ann 5, Ann 7, Patricia 5, Margaret 5, and Catherine 5. Hyphen and space forms (`Ann-Marie`, `Ann Marie`) are extensions at distance 4–6. Chinese superstrings and middles (`王伟强`, `王小伟`) are extensions. The shipped matcher abstains all 30.

Row-roster next distances for the English typos are {5, 6, 7}. The English near-substitutions on their own rows are {6, 7}. A cutoff that accepts a typo at 5 accepts a near-duplicate at 6 and at 7. On the full roster `Mary Jame` is 4 from Margaret and `Robern` is 5 from Ann, so the bands overlap there too. Every Chinese second neighbor, typo or near, is 2.

## Gates that were measured

Applied only when the shipped `best_name` abstains. A hit has to be unique. Extensions are dropped. Direct component share uses the operands above, not the recursive walk. Moving `matched_people` is what `ask` would see.

| gate | frozen typos recovered | near still abstain | probe / nick / cjk | abstain-dev near that starts matching | why it stops |
|---|---|---|---|---|---|
| any unique edit, sub or indel | 16/16 | 11/16 en, 11/14 zh | 1/442, 1/43, 0/201 | Harols, Juduth, Marcas, Anna Louize, Kennth, 吴笛, 郑恺, 徐锋 | also accepts Robern, Williem, Catherina, Mary Jone, Patrici, 黄蕾, 周滔, 陈靖. Probe: "Where did Ada tell me to show up?" gains Mei (`me` deletes one letter). Nickname: Charls→Charles |
| unique substitution | 16/16 | 12/16 en, 11/14 zh | 0, 0, 0 | the four English subs and the three Chinese | Patrici stays absent. The four English near-subs and 黄蕾 / 周滔 / 陈靖 match |
| unique sub, next neighbor ≥ 3 | 8/16, English only | 12/16 en, 14/14 zh | — | Harols, Juduth, Marcas, Anna Louize | Chinese seconds are 2, so all eight Chinese typos stay missed, and the four English near-subs (seconds 6–7) still match |
| keyboard-adjacent sub | 4/16 (Willian, Patricis, Anm, Mary Jame) | 16/16, 14/14 | 0, 0, 0 | Harols, Juduth, Anna Louize | Marcas (`u`/`a`) is not adjacent. The abstain-dev adjacent pairs are both labels |
| adjacent, minus the dev both-label pairs | 4/16, same four | 16/16, 14/14 | 0, 0, 0 | none | frozen near stays 30/30. The denylist is `ds`, `iu`, `ua`, `zs`, 笛迪, 恺凯, 锋峰. The new bank labels `Femwick`/`Fenwick` (`m`/`n`) and `Danils`/`Danila` (`s`/`a`) both ways, and those pairs are not on the list. `Tressis`/`Tressid` (`s`/`d`) is on the list, so both of its labels stay abstained and the typo side is a miss too |
| same-pinyin substitution | 7/16 (all but 张疆) | 16/16, 11/14 | 0, 0, 0 | 吴笛, 郑恺, 徐锋 | also accepts the absent rows 黄蕾, 周滔, 陈靖 |
| direct IDS share | 5/16 (王玮, 张疆, 刘样, 陈靖, 周滔) | 16/16, 12/14 | 0, 0, 0 | 吴笛, 郑恺, 徐锋 | 黄蕾 and 赵悯 share nothing directly. 李那 is containment, not a shared operand. 周滔 and 陈靖 are also absent |
| containment (surface character is an operand of the stored character) | 1/16 (李那→李娜) | 16/16, 14/14 | 0, 0, 0 | none | the narrowest frozen move. The new bank labels 逯爰 / 逯媛 both ways (爰 is an operand of 媛, ⿰女爰). Same pinyin `yuan` |

Frozen `matched_people` moves under the wide edit gate are those 24 rows and no others (16 typos + 8 near). Substitution drops only Patrici, 23 rows. Adjacent and containment do not touch `near_dup_absent`. Each of them still accepts an abstain twin on the new bank, or an absent row on the frozen file, or both.

## Dev bank

`tidal/question_typo_edit_dev_data.py`. The matcher does not import it. Question sha256 `3447aaaa0b649dc602f139cedb660bb92bace99f9de3f22d1aa3ad0f261b70fd`, n=96 (48 English, 48 Chinese). This is not the closed PR #24 bank (`68260c01…`, n=68).

Each collision kind is one surface and one roster, emitted once as `match` and once as `abstain`. The closer is not a feature of the name. Exactly one roster name is one edit away, and it is the first name. The shipped `best_name` is `None` on every profile, including the ties and the extensions.

| kind | n | labels | what it collides |
|---|---:|---|---|
| collision_adj_sd | 6 | match + abstain | Tressis / Tressid, `s`↔`d`, adjacent, on the denylist |
| collision_adj_mn | 6 | match + abstain | Femwick / Fenwick, `m`↔`n`, adjacent, not on the denylist |
| collision_adj_as | 6 | match + abstain | Danils / Danila, `s`↔`a`, adjacent, not on the denylist |
| collision_vowel_ae | 6 | match + abstain | Baltram / Beltram, `a`↔`e`, the Robart / Williem class |
| collision_vowel_oa | 6 | match + abstain | Rasmund / Rosmund, `a`↔`o`, the Margoret / Mary Jone class |
| collision_del | 6 | match + abstain | Hartly / Hartley, deletion. `prefix_ge3` is 1 and `_featurize` is still `None` |
| collision_homo | 8 | match + abstain | 郗潞 / 郗璐, same pinyin, share 路 |
| collision_nose | 8 | match + abstain | 盖虹 / 盖宏, same pinyin, no shared operand |
| collision_diffpy | 8 | match + abstain | 蒯绵 / 蒯锦, pinyin differs, share 帛 |
| collision_contain | 8 | match + abstain | 逯爰 / 逯媛, 爰 is an operand of 媛 |
| tie_en | 3 | abstain | Rasmund is one edit from both Rosmund and Resmund |
| tie_zh | 4 | abstain | 郗潞 is one edit from both 郗璐 and 郗露 |
| hyphen, space, extension | 3 each | abstain | Fenwick-Bryn, Fenwick Bryn, Fenwicko. Extensions. The shipped matcher already abstains |
| prefix, middle, superstring | 4 each | abstain | 老郗璐, 郗小璐, 郗璐然. Extensions. Already abstained |

English closers are `docket`, `folio`, `quire`. Chinese closers are `deng`, `ruo`, `zhi`, `ye`. Anchors are `spikenard` / `cimbalom` and `蠃` / `鼏` / `爟`. Names: Tressis, Tressid, Quenby, Islwyn, Femwick, Fenwick, Ostara, Bryony, Danils, Danila, Torquil, Morwen, Baltram, Beltram, Pryderi, Elspeth, Rasmund, Rosmund, Resmund, Idris, Lowri, Tanwen, Hartly, Hartley, Ciaran, Bronwen; 郗潞, 郗璐, 郗露, 盖宏, 盖虹, 蒯锦, 蒯绵, 逯媛, 逯爰. None of those names are on the probe, surface, abstain, adversarial, CJK, nickname, indirect, cue, bare-locative, residual, two-name, focus, lastname, or prior typo rosters.

Before and after are the same name decision, because nothing shipped. `best_name` is `None` on all 96 profiles: 0/34 match rows recovered, 62/62 abstain rows still abstain. Every gate in the table that recovers a match row returns that same stored name on the abstain twin.

Literal overlap with every reference bank is 0. Token Jaccard is the probe tokenizer. The cap is 0.50. Max across the 96 questions is 0.231.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 strings | 0 | 0.231 |
| surface cue corpus | — | 1124 strings | 0 | 0.200 |
| abstain dev | `26889048…` | 381 strings | 0 | 0.154 |
| frozen adversarial | `cce20b6b…` | 381 strings | 0 | 0.214 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.120 |
| nickname dev | `a766f748…` | 43 | 0 | 0.083 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.118 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.118 |
| typo dev (PR #24) | `68260c01…` | 68 | 0 | 0.150 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.118 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.167 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.150 |
| two-name dev | `9c72a8b3…` | 120 strings | 0 | 0.176 |
| focus dev | `03dea71d…` | 132 strings | 0 | 0.118 |
| lastname bank (PR #31) | questions `064d7052…` | 309 strings (101 questions) | 0 | 0.176 |

Closest probe pair: "spikenard quire for Tressis sits on the cimbalom glyph03." ‖ "What's the birthday on file for Jules?" Closest frozen pair: the same left side ‖ "Read the ordered haunts on file for Ann." Closest focus pair: "The spikenard cimbalom folio keeps Tressis at glyph02." ‖ "Skip Roderick outright. The birch tally keeps the telephone under Cunigunde, quillon60." Closest two-name pair: the quire line ‖ "No telephone was entered for Balthasar on the cedar slip, brinekeel45." Closest lastname pair: the quire line ‖ "On skerry tray 00b, nacre holds the birthday entered for Lysander."

Sentence cosine was not a cap on this bank. The embedder is not what separates these pairs: the collision is the label on one surface.

## Frozen adversarial and the probe

Before equals after. The matcher at `f7f13b1` is the matcher on this branch.

Full ask is the shipped `score_split` path (real `ask`, committed cache). Name-slice cells were re-scored here with `surface_abstain` on those vectors; they match the full-ask cells. A cache-only pass over every slice scores 176/216 because deixis and indirect rows resolve through `embed_event`. That gap is outside the name slice. The headline is the full ask.

| | before | after |
|---|---:|---:|
| answerable hits | 188/216 | 188/216 |
| answerable false abstain | 19/216 | 19/216 |
| missing_rel abstained | 60/69 | 60/69 |
| missing_rel still answered | 9 | 9 |
| absent_person | 42/42 | 42/42 |
| name_variant en hits (false abstain) | 24/32 (8/32) | 24/32 (8/32) |
| name_variant zh hits (false abstain) | 35/43 (8/43) | 35/43 (8/43) |
| typo false abstain, en + zh | 8/8 and 8/8 | 8/8 and 8/8 |
| near_dup_absent | 16/16 and 14/14 | 16/16 and 14/14 |
| two_names answerable, en + zh | 6/6 and 6/6 | 6/6 and 6/6 |
| two_names missing_rel, en + zh | 6/6 and 6/6 | 6/6 and 6/6 |
| two_names absent, both languages | 12/12 | 12/12 |
| cjk_superstring abstained | 8/8 | 8/8 |
| pinyin hits | 8/8 | 8/8 |
| nickname hits, en + zh | 6/6 and 8/8 | 6/6 and 8/8 |

Of the 19 answerable false abstains, 16 are these typos. The other three are the cells that #29 left in place and #30 did not move: `compact_form` English answerable 2/7, `misleading_cue` English answerable 1/6.

`eval_heldout` on this checkout, matcher unchanged:

| category | n | ask ret@1 | false abstain |
|---|---:|---|---|
| answerable | 370 | 365/370 | 0/370 |
| en answerable | 185 | 183/185 | 0 |
| zh answerable | 185 | 182/185 | 0 |
| unanswerable | 72 | — | 72/72 abstained |
| missing_rel | 36 | — | 36/36 abstained |
| absent_person | 36 | — | 36/36 abstained |

Phone 122/124, birthday 122/124, places 121/122. Gold lost at overwrite 0.9 is 0/370. Co-resident pairs 8390, cosine above 0.9 is 0, max 0.8926955194738055. n=442.

## Tests

`tests/test_question_typo_edit.py` locks the dev sha, the kind and label counts, that each collision's accepting gate returns one stored name for both labels, that the deletion's `prefix_ge3` still featurizes to nothing, that ties and extensions are not a unique edit, literal disjointness, and token Jaccard under 0.50. The lastname bank strings are embedded because that module is not on main. Frozen numeric scores are not a pytest gate.

`python3 -m pytest -q`: count filled after the suite run.
