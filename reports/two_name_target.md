# Two-name ask target

**Fix shipped.** A flat-row `ask` that names two stored people now decides who the question is about before it checks whether that person stores the asked relation. The other person's fact no longer fills a missing relation, and it no longer wins the ranker when the target does store the relation.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen adversarial questions were scored after the rule was fixed. They were not an input to the wording.

## Rule

`ask_people` is still the match (a written name, a nickname, pinyin, or the unique first-person hearer). `ask_targets` narrows that list only when two or more people matched:

1. Drop a name whose every lexical span sits inside a dismissed clause, when another matched name remains. The clause is the same closed class as `cue_set_aside` (`ignore` / `skip` / `forget` / `exclude`, `aside`, a decoy noun, and the Chinese marks). A name with no lexical span stays. If every matched name is inside the clause, the list stays, so a question about one dismissed person is still about that person.
2. If exactly one remaining name possessively governs the unique live cue, that name is the only target. The cue can follow the marker in the same clause (`Name's cue`, `Name的cue`) or the possessive can close the clause (`is Name's`, `是Name的`). A partner joined by `and` / `和` / `与` / `跟` (`X and Y's cue`, `X和Y的cue`) is included, so the pair is not narrowed to the second name. A dismissed unique cue is not a live cue.

Otherwise every remaining name stays. Cue-nearest (`_focus_names`) still runs only inside `selected_fact_rows`, and only among whoever remains. The ranker uses `ask_targets` through `matched_fact_rows` and does not apply cue-nearest. One match is unchanged.

The failure this catches: on "Leave Robert aside. The telephone to read belongs with Margaret." the dismissed name sits next to "telephone", so cue-nearest treated Robert as the ask. Robert stores the telephone, the availability check passed, and `ask` returned Robert's number. Margaret, who has no telephone, was the person the question named.

## Dev bank

`tidal/question_two_name_dev_data.py`. The matcher does not import it. Question sha256 `9c72a8b3f65eff7e6fdc436e906f769de6f9e85451284464f23eb338296d02d8`, n=96 (57 English, 39 Chinese).

| kind | n | what the row requires |
|---|---:|---|
| dismiss_miss | 18 | target lacks the relation, the other person has it, the other name is only in a dismissed clause: abstain |
| dismiss_hit | 18 | both store the relation, same shape: answer the target's value |
| poss_miss | 12 | target lacks it, the other has it, only the target possessively governs the cue: abstain |
| poss_hit | 12 | both store it, same shape: answer the target's value |
| one_hit | 9 | one name, that person stores it: answer |
| one_miss | 9 | one name, that person lacks it: abstain |
| one_dismiss_hit | 6 | the only name sits inside a dismiss clause and stores the relation: answer |
| absent | 6 | the asked person is not stored: `absent_person` |
| open_either | 6 | both names stay (no dismiss, no unique possessor) and both store the relation: answer either value, do not abstain |

Names: Cressida, Balthasar, Honoria, Peregrine, absent Winifred; 嵇沅, 逯荞, 酆棠, 禚岚, absent 扈茗. Frames cover Skip/Ignore/Forget/Exclude, "is Name's", "for Y … as Name's", 别提/跳过, 跟/和 with 是Name的, and one-name controls. `open_either` is the comparison shape the rule deliberately leaves open.

Literal overlap with every reference bank is 0. Token Jaccard is the probe tokenizer. The cap is 0.50. Max across the bank (questions and statements) is 0.357.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | — | 0 | 0.318 |
| surface cue corpus | — | 1124 strings | 0 | 0.357 |
| abstain dev | `26889048…` | 327 | 0 | 0.357 |
| frozen adversarial | `cce20b6b…` | 327 | 0 | 0.357 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.194 |
| nickname dev | `a766f748…` | 43 | 0 | 0.111 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.278 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.278 |
| typo dev | `68260c01…` | 68 | 0 | 0.125 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.278 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.240 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.280 |

Closest surface pair: "Which birthday was written down for Winifred rather than Cressida, brinekeel52." ‖ "Was a birthday ever written down for Quentin?" Closest frozen pair: "Forget Balthasar. The telephone on the cedar slip is the one under Cressida, brinekeel6." ‖ "Elizabeth's telephone is the one on file."

Sentence cosine of the 120 dev strings against the frozen caches, same embedder (`intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix): 0.917 vs the probe ("电话记给扈茗，不是记给嵇沅，廌廑90。" ‖ "江禾提过打电话用的数字，我们就去聊别的了。") and 0.918 vs the surface corpus ("生日记给扈茗，不是记给嵇沅，廌廑91。" ‖ "节庆不算，留下谢迟认的生日。"). The strings are different questions. The vectors are not committed.

Ask was scored with that embedder on the shipped bundle (`alpha` 0.3, abstain ridge, margin 0.1). Before replaces `ask_targets` with `ask_people`, which is the status quo (cue-nearest still applied inside abstain). After is the shipped function. No dev row had the abstain ridge disagree with the unique cue by 0.15 or more, so the cue stood on every row.

| kind | n | before correct | after correct |
|---|---:|---:|---:|
| dismiss_miss | 18 | 0 | 18 |
| dismiss_hit | 18 | 3 | 18 |
| poss_miss | 12 | 0 | 12 |
| poss_hit | 12 | 6 | 12 |
| one_hit | 9 | 9 | 9 |
| one_miss | 9 | 9 | 9 |
| one_dismiss_hit | 6 | 6 | 6 |
| absent | 6 | 6 | 6 |
| open_either | 6 | 6 | 6 |
| total | 96 | 45 | 96 |

The 51 before-failures are the leak: 30 missed abstains (`dismiss_miss` 18, `poss_miss` 12), each answered with the other person's fact, and 21 wrong-person answers (`dismiss_hit` 15, `poss_hit` 6). One-name rows, the absent rows, the single name inside a dismiss clause, and the open comparison all stayed correct. After: 96/96, wrong person 0, false abstain 0.

## Frozen adversarial

Question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Wording was not edited. Scores use the committed cache. Before is `ask_targets` replaced by `ask_people`. Eighteen `two_names` questions change their target list, all of them narrowing to the labeled focus. Eight of those change the returned fact. Every other slice is identical.

| | before | after |
|---|---:|---:|
| answerable hits | 182/216 | 187/216 |
| answerable false abstain | 19/216 | 19/216 |
| missing_rel abstained | 57/69 | 60/69 |
| missing_rel still answered | 12 | 9 |
| absent_person | 42/42 | 42/42 |
| English missing_rel abstained | 26/35 | 29/35 |
| Chinese missing_rel abstained | 31/34 | 31/34 |
| English answerable hits | 89/103 | 91/103 |
| Chinese answerable hits | 93/113 | 96/113 |

`two_names` by itself:

| | en missing_rel abstained | zh missing_rel abstained | en answerable hits | zh answerable hits | absent both languages |
|---|---:|---:|---:|---:|---:|
| before | 3/6 | 6/6 | 3/6 | 3/6 | 12/12 |
| after | 6/6 | 6/6 | 5/6 | 6/6 | 12/12 |

The three English missing-relation rows that were answered are all "Leave {other} aside. The {cue} to read belongs with {focus}." Cue-nearest selected the dismissed person, who stores the relation. After, the target is the focus, who does not store it, and `ask` abstains. Slot scores are sentence cosine plus the relation logit. A name outside the target set is placed one below the worst in-scope score.

| question | before | after | ridge (gap) | in-scope scores after |
|---|---|---|---|---|
| Leave Robert aside. The telephone to read belongs with Margaret. | Robert `+1-917-555-1384` (1.118) | `missing_rel` | phone 0.965 (0.887) | Margaret places 0.916, birthday 0.795 |
| Leave Margaret aside. The birthday to read belongs with Robert. | Margaret `1982-07-25` (1.027) | `missing_rel` | birthday 0.711 (0.530) | Robert phone 0.904, places 0.878 |
| Leave Robert aside. The ordered locations to read belongs with Margaret. | Robert `north420-…` (1.077) | `missing_rel` | places 0.753 (0.518) | Margaret phone 0.935, birthday 0.810 |

The English answerable misses that move are the same frame. Mary Jane's sentence was ahead by about 0.002. After, William is the only target.

| question | before | after |
|---|---|---|
| Leave Mary Jane aside. The telephone to read belongs with William. | Mary Jane `+1-917-555-1380` (1.101) over William `+1-917-555-1379` (1.099) | William's number (1.099) |
| Leave Mary Jane aside. The ordered locations to read belongs with William. | Mary Jane `north416-…` (1.092) over William `north415-…` (1.090) | William's places (1.090) |
| Leave William aside. The birthday to read belongs with Mary Jane. | Mary Jane `1977-02-20` (1.028) already ahead of William (1.017) | same answer, target list is Mary Jane only |

Chinese `先放下{other}。要回读的{cue}是{focus}的。` answerable rows had the same leak (the dismissed person won by 0.007 to 0.012) and now return the focus: 周涛's telephone, 刘洋's birthday `1943-10-22`, 周涛's places. On the places row the ridge gap is 0.114, under 0.15, and the top label is places, so the cue stands. The ranker scope is what changes the answer. Chinese missing-relation `先放下` rows already abstained, because the focus sits next to the cue; they still abstain, and the target list is now the focus only. `只问{focus}的{cue}` already returned the focus (the possessive puts that name on the cue); the target list narrows and the fact does not change.

The three English "Between {other} and {focus}, ask only about {focus}" missing-relation rows already abstained and still do. Cue-nearest was already the focus.

Full frozen cells. False abstain is unchanged in every cell, so it is one column.

| slice | lang | label | n | abstained before | abstained after | hits before | hits after | false abstain |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bare_cue | en | answerable | 4 | 0 | 0 | 4 | 4 | 0 |
| bare_cue | en | missing_rel | 4 | 3 | 3 | 0 | 0 | 0 |
| bare_cue | zh | answerable | 4 | 0 | 0 | 4 | 4 | 0 |
| bare_cue | zh | missing_rel | 4 | 4 | 4 | 0 | 0 | 0 |
| compact_form | en | answerable | 7 | 2 | 2 | 5 | 5 | 2 |
| compact_form | en | missing_rel | 7 | 5 | 5 | 0 | 0 | 0 |
| compact_form | zh | answerable | 6 | 0 | 0 | 6 | 6 | 0 |
| compact_form | zh | missing_rel | 6 | 5 | 5 | 0 | 0 | 0 |
| deixis | en | answerable | 36 | 0 | 0 | 36 | 36 | 0 |
| deixis | zh | answerable | 36 | 0 | 0 | 27 | 27 | 0 |
| misleading_cue | en | answerable | 6 | 1 | 1 | 5 | 5 | 1 |
| misleading_cue | en | missing_rel | 6 | 5 | 5 | 0 | 0 | 0 |
| misleading_cue | zh | answerable | 6 | 0 | 0 | 6 | 6 | 0 |
| misleading_cue | zh | missing_rel | 6 | 4 | 4 | 0 | 0 | 0 |
| name_variant | en | answerable | 32 | 8 | 8 | 24 | 24 | 8 |
| name_variant | zh | answerable | 43 | 8 | 8 | 35 | 35 | 8 |
| near_dup_absent | en | absent_person | 16 | 16 | 16 | 0 | 0 | 0 |
| near_dup_absent | zh | absent_person | 14 | 14 | 14 | 0 | 0 | 0 |
| relation_paraphrase | en | answerable | 12 | 0 | 0 | 12 | 12 | 0 |
| relation_paraphrase | en | missing_rel | 12 | 10 | 10 | 0 | 0 | 0 |
| relation_paraphrase | zh | answerable | 12 | 0 | 0 | 12 | 12 | 0 |
| relation_paraphrase | zh | missing_rel | 12 | 12 | 12 | 0 | 0 | 0 |
| two_names | en | absent_person | 6 | 6 | 6 | 0 | 0 | 0 |
| two_names | en | answerable | 6 | 0 | 0 | 3 | 5 | 0 |
| two_names | en | missing_rel | 6 | 3 | 6 | 0 | 0 | 0 |
| two_names | zh | absent_person | 6 | 6 | 6 | 0 | 0 | 0 |
| two_names | zh | answerable | 6 | 0 | 0 | 3 | 6 | 0 |
| two_names | zh | missing_rel | 6 | 6 | 6 | 0 | 0 | 0 |

Nine `missing_rel` rows are still answered. None of them is `two_names`. They are the same rows as on main `d54aa6b` outside this slice: "How do I get through to Elizabeth once the office is dark?" (places gap 0.031), "How can Elizabeth be reached after dusk?" (places gap 0.247), "How do I reach Catherine from the quay after dark?" (places gap 0.519), "When does Patricia mark another trip around the sun?" (birthday gap 0.095, stores places), "Where should the year-older mark for Ann be placed?" (places gap 0.022), "Ignore the birthday feast and say how to get through to Catherine." (birthday gap 0.258), "怎样才能够着周涛？" (birthday gap 0.072), "电话簿先合上，赵敏常去的几处按先后是什么？" (places gap 0.070), "地方名单先放下，怎么用声音找到刘洋？" (phone gap 0.089). False abstains stay the two English compact-form phone rows, the Mary Jane locations-file row, and the 8+8 name-variant typos.

The abstain-dev bank (`26889048…`, n=327) moves only its `two_names` cells, the same way: English missing_rel 3/6 → 6/6, English answerable 4/6 → 6/6, Chinese answerable 1/6 → 6/6, Chinese missing_rel stays 6/6. Other cells are unchanged. Rollup: answerable hits 188/217 → 195/217 with false abstain 22/217 both times; missing_rel abstained 57/68 → 60/68; absent 42/42.

## Probe and surface margin

Probe questions: `ask_targets` equals `ask_people` on all 442 rows, and on all 1124 surface-corpus strings, and on every indirect-dev row. The measured probe is therefore the same path as main. `eval_heldout` after the change:

| category | n | cosine ret@1 | ask ret@1 | phone cos·ask | date cos·ask | tokens cos·ask | false cos·ask | gold lost @0.9 |
|---|---:|---|---|---|---|---|---|---|
| en_direct | 36 | 12/36 | 36/36 | 2/12 · 12/12 | 4/12 · 12/12 | 6/12 · 12/12 | — | 0/36 |
| zh_direct | 36 | 9/36 | 36/36 | 4/12 · 12/12 | 2/12 · 12/12 | 3/12 · 12/12 | — | 0/36 |
| en_indirect | 40 | 14/40 | 39/40 | 3/13 · 13/13 | 7/14 · 13/14 | 4/13 · 13/13 | — | 0/40 |
| zh_indirect | 40 | 10/40 | 38/40 | 3/13 · 12/13 | 4/14 · 14/14 | 3/13 · 12/13 | — | 0/40 |
| en_cross_turn | 37 | 8/37 | 37/37 | 3/13 · 13/13 | 4/12 · 12/12 | 1/12 · 12/12 | — | 0/37 |
| zh_cross_turn | 37 | 6/37 | 36/37 | 2/13 · 12/13 | 1/12 · 12/12 | 3/12 · 12/12 | — | 0/37 |
| en_after_distractors | 36 | 5/36 | 35/36 | 0/12 · 12/12 | 1/12 · 11/12 | 4/12 · 12/12 | — | 0/36 |
| zh_after_distractors | 36 | 6/36 | 36/36 | 0/12 · 12/12 | 2/12 · 12/12 | 4/12 · 12/12 | — | 0/36 |
| en_after_update | 36 | 12/36 | 36/36 | 6/12 · 12/12 | 2/12 · 12/12 | 4/12 · 12/12 | — | 0/36 |
| zh_after_update | 36 | 2/36 | 36/36 | 0/12 · 12/12 | 2/12 · 12/12 | 0/12 · 12/12 | — | 0/36 |
| en_unanswerable | 36 | — | — | — | — | — | 36/36 · 0/36 | — |
| zh_unanswerable | 36 | — | — | — | — | — | 36/36 · 0/36 | — |
| answerable | 370 | 84/370 | 365/370 | 23/124 · 122/124 | 29/124 · 122/124 | 32/122 · 121/122 | — | 0/370 |
| unanswerable | 72 | — | — | — | — | — | 72/72 · 0/72 | — |

Ask abstain on that run: answerable 365/370 with false abstain 0/370 (en 183/185, zh 182/185; phone 122/124, birthday 122/124, places 121/122); missing_rel 36/36; absent_person 36/36; unanswerable 72/72. Co-resident pairs 8390, cosine above 0.9 is 0, max 0.8927. Gold lost at overwrite 0.9 is 0/370.

Surface dev at margin 0.1: false abstain 0/192 English and 0/192 Chinese, missing 188/192 and 189/192, absent 36/36 both, deictic false 0/6 both. Margins 0.0 and 0.05 still false-abstain 1/192 English.

## What stayed

`name_variant`, `misleading_cue`, `bare_cue`, `deixis`, `compact_form`, `relation_paraphrase`, and `near_dup_absent` are cell-identical to the measurement above their "before" column, which is main `d54aa6b` (#27). That is the name-align / CJK / diminutive / person-scoped ranking / indirect-hearer / misleading-cue / bare-locative behavior on this frozen set. Person-scoped ranking (#23) still keeps both names when neither is only in a dismissed clause and neither is the unique possessor; cue-nearest is still abstain-only. "William birthday Mary Jane" still ranks both stored people. A single name inside "Ignore Cressida's supper" is still that person.

## Residual

"Between Catherine and Ann, ask only about Ann: which birthday is filed?" still returns Catherine's birthday `1984-09-27` (0.973) ahead of Ann's `1983-08-26` (0.964). Targets stay `['Catherine', 'Ann']`. There is no dismissed clause and no possessive, so the rule does not choose Ann. The phone and places rows of that frame already returned the focus and still do. A string list for "only about" / "只问" would be fit to this frozen template. Chinese `只问` is already handled when the focus possessively governs the cue. This one English answerable row is the leftover inside `two_names`.

## Tests

`python3 -m pytest -q`: 276 passed. The new file locks the dev sha, the Jaccard cap, the zero-query abstain decisions, coordination (`Ann and Catherine's birthday` keeps both), probe/surface/indirect target identity, and a phone ridge on axis 121 where the other person's sentence is an exact copy of the query. Frozen numeric scores are not a pytest gate.
