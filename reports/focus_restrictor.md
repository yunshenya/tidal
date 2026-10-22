# Focus restrictor

**Fix shipped.** A two-name `ask` that says who it is about (`only about Ann`, `asks only Kenneth`, `只问仉禾`) now keeps that name when another stored name is still in the sentence. The other person's fact no longer wins the ranker, and it no longer fills the relation when the focused person does not store it.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The rule was fixed on the dev bank below. The frozen adversarial questions were scored after that, and they were not an input to the wording.

## Rule

`ask_people` is still the match. `ask_targets` narrows that list only when two or more people matched, in this order:

1. Drop a name whose every lexical span sits inside a dismissed clause, when another matched name remains. Same closed class as #29.
2. If exactly one remaining name possessively governs the unique live cue, that name is the only target. A coordinated partner (`and` / `和` / `与` / `跟`) is included, so `X and Y's cue` stays both. Same as #29. This step still runs first, so `只问{focus}的{cue}` is the possessor, not the restrictor.
3. If a focus restrictor governs a non-empty proper subset of whoever remains, that subset is the target. The operator is a closed class. English is `only` / `just` / `solely`, with an optional `about`, and the governed name is the matched name that follows. Chinese is `单独问` / `只问` / `只看`, the same way. `not only` and a preceding `不` (`不只问`) are not restrictors. A partner joined by `and` / `和` / `与` / `跟`, before or after the name, is governed too, so the pair is not narrowed. Nobody governed, or everybody governed, leaves the list as it was.

Otherwise every remaining name stays. Cue-nearest (`_focus_names`) still runs only inside `selected_fact_rows`, and only among whoever remains. The ranker uses `ask_targets` through `matched_fact_rows` and does not apply cue-nearest. One match is unchanged, so a restrictor on a single name does not alter that ask.

The failure this catches: "Between Catherine and Ann, ask only about Ann: which birthday is filed?" has no dismissed clause and no possessive. Both names stayed, Catherine's birthday scored 0.973 against Ann's 0.964, and `ask` returned Catherine.

## Dev bank

`tidal/question_focus_dev_data.py`. The matcher does not import it. Question sha256 `03dea71d6a47343fc6410c21df19868e824818e83b48e61ae2aaffc45e188c7c`, n=96 (51 English, 45 Chinese). The bank is 132 strings once questions and statements are deduped.

| kind | n | what the row requires |
|---|---:|---|
| focus_hit | 21 | both store the relation, the other name is nearer the cue, a restrictor names the target: answer the target |
| focus_miss | 21 | the target lacks the relation, the other has it and is nearer the cue, a restrictor names the target: abstain |
| open_either | 12 | `between` / `among` / `和` / `与`, no restrictor, both store it: answer either value |
| open_miss | 6 | same open shape, neither stores the asked relation: abstain |
| dismiss_hit | 6 | #29 dismiss control, both store it: answer the target |
| dismiss_miss | 6 | #29 dismiss control, the dismissed person stores it and the target does not: abstain |
| coord_either | 6 | the restrictor governs both names (`only about X and Y`, `只问X和Y`): answer either |
| not_only_either | 6 | `not only` / `不只问` is not a restrictor: answer either |
| one_hit | 6 | one name under a restrictor, that person stores it: answer |
| one_miss | 6 | one name under a restrictor, that person lacks it: abstain |

Focus frames are `only about`, `just about`, `solely`, `just`, `只问`, `只看`, and `单独问`, each on phone, birthday, and places. The question starts with the other person's stored clause, so sentence cosine prefers that person until the restrictor drops them. The target's sentence does not share the clause.

Names: Griselda, Hildebrand, Cunigunde, Roderick; 夔昭, 厍楠, 仉禾, 爨书. No absent-person rows. Anchors are `quillon` / `黿鼉`. None of those names are on the probe, surface, abstain, adversarial, CJK, nickname, indirect, cue, bare-locative, residual, or two-name rosters.

Literal overlap with every reference bank is 0. Token Jaccard is the probe tokenizer. The cap is 0.50. Max across the bank (questions and statements) is 0.333.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 strings | 0 | 0.250 |
| surface cue corpus | — | 1124 strings | 0 | 0.286 |
| abstain dev | `26889048…` | 381 strings | 0 | 0.227 |
| frozen adversarial | `cce20b6b…` | 381 strings | 0 | 0.231 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.160 |
| nickname dev | `a766f748…` | 43 | 0 | 0.056 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.241 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.241 |
| typo dev | `68260c01…` | 68 | 0 | 0.150 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.240 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.200 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.261 |
| two-name dev | `9c72a8b3…` | 120 strings | 0 | 0.333 |

Closest two-name pair: "Between Cunigunde and Roderick, the tin chit holds a telephone for each of them, quillon42." ‖ "Peregrine alongside Honoria, and the cedar book holds a telephone for each, brinekeel54." Closest frozen pair: "只问仉禾和爨书，锡笺上的电话记了两条，黿鼉75。" ‖ "王伟和陈静里面，只问王伟的电话。" Closest probe pair: "南厢桦笺单记仉禾留的地点底账。" ‖ "江禾的新地点清单是什么？"

Sentence cosine of the 132 dev strings against the frozen caches, same embedder (`intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix): 0.928 vs the probe ("别提爨书。桦笺把电话留给仉禾，黿鼉66。" ‖ "江禾提过打电话用的数字，我们就去聊别的了。"), 0.919 vs the surface corpus ("不只问仉禾，爨书也在册，锡笺上的电话记了两条，黿鼉81。" ‖ "把蔡禾叫起来通话，用哪部电话？"), and 0.913 vs the adversarial cache ("只问仉禾，桦笺把生日留在哪一条，黿鼉94。" ‖ "黄磊和张强里面，只问黄磊的生日。"). The strings are different questions that share a relation word. The vectors are not committed. No cosine cap was required, and the wording was not edited to chase it.

Ask was scored with that embedder on the shipped bundle (`alpha` 0.3, abstain ridge, margin 0.1). Before replaces `_restrict_targets` with a function that returns nothing, which is the #29 rule (dismiss, then a unique possessor). After is the shipped function.

| kind | n | before correct | after correct |
|---|---:|---:|---:|
| focus_hit | 21 | 0 | 21 |
| focus_miss | 21 | 0 | 21 |
| open_either | 12 | 12 | 12 |
| open_miss | 6 | 6 | 6 |
| dismiss_hit | 6 | 6 | 6 |
| dismiss_miss | 6 | 6 | 6 |
| coord_either | 6 | 6 | 6 |
| not_only_either | 6 | 6 | 6 |
| one_hit | 6 | 6 | 6 |
| one_miss | 6 | 6 | 6 |
| total | 96 | 54 | 96 |

The 42 before-failures are the leak, and they are only the focus rows. Each of the seven operators was 0/3 before and 3/3 after, on both the hit and the miss, in both languages. `focus_hit` answered the other person's value (21 wrong-person answers). `focus_miss` did the same instead of abstaining (21 missed abstains). Open comparisons, the dismiss controls, coordinated pairs, `not only` / `不只问`, and the one-name rows stayed correct. After: 96/96, wrong person 0, missed abstain 0, false abstain 0.

On a focus row the cue-nearest name, before the restrictor, is the other person. A zero-query abstain (the ridge gap stays under 0.15, so the unique cue stands) agrees with the label on all 96 rows after the rule. It does not abstain a focus miss before the restrictor, because the other person stores the relation.

## Frozen adversarial

Question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Wording was not edited. Scores use the committed cache. Before is `_restrict_targets` replaced by a function that returns nothing. One question changes its returned fact. Six English `two_names` questions change their target list, all of them narrowing to the labeled focus. Every other slice is cell-identical, and false abstain does not move in any cell.

| | before | after |
|---|---:|---:|
| answerable hits | 187/216 | 188/216 |
| answerable false abstain | 19/216 | 19/216 |
| missing_rel abstained | 60/69 | 60/69 |
| missing_rel still answered | 9 | 9 |
| absent_person | 42/42 | 42/42 |
| English missing_rel abstained | 29/35 | 29/35 |
| Chinese missing_rel abstained | 31/34 | 31/34 |
| English answerable hits | 91/103 | 92/103 |
| Chinese answerable hits | 96/113 | 96/113 |

`two_names` by itself. The #29 floors hold, and the leftover English answerable row is now a hit.

| | en missing_rel abstained | zh missing_rel abstained | en answerable hits | zh answerable hits | absent both languages |
|---|---:|---:|---:|---:|---:|
| before | 6/6 | 6/6 | 5/6 | 6/6 | 12/12 |
| after | 6/6 | 6/6 | 6/6 | 6/6 | 12/12 |

The one fact that moves:

| question | before | after | ridge (gap) | slot scores |
|---|---|---|---|---|
| Between Catherine and Ann, ask only about Ann: which birthday is filed? | Catherine `1984-09-27` (0.973) | Ann `1983-08-26` (0.964) | birthday 0.556 | Ann birthday 0.964, Catherine birthday 0.973, William places `north405-…` 0.832 |

Targets go from `['Catherine', 'Ann']` to `['Ann']`. `ask_people` is still both names; that is the match, and the ranker uses `ask_targets`. The birthday gap is 0.556, so the cue stands. Catherine is then out of ranker scope. Scores are sentence cosine plus the relation logit (`alpha` 0.3).

The other five English rows narrow the same way and already returned the right fact, so the fact does not change:

| question | targets before | targets after |
|---|---|---|
| Between Ann and Catherine, ask only about Catherine: which telephone is filed? | Catherine, Ann | Catherine |
| Between Elizabeth and Patricia, ask only about Patricia: which telephone is filed? | Elizabeth, Patricia | Patricia |
| Between Patricia and Elizabeth, ask only about Elizabeth: which birthday is filed? | Elizabeth, Patricia | Elizabeth |
| Between Ann and Catherine, ask only about Catherine: which ordered locations is filed? | Catherine, Ann | Catherine |
| Between Elizabeth and Patricia, ask only about Patricia: which ordered locations is filed? | Elizabeth, Patricia | Patricia |

The three missing-relation rows in that list already abstained and still do. Chinese `只问{focus}的{cue}` was already one target under the possessor step, so those rows do not move, and Chinese answerable stays 6/6. The nine `missing_rel` rows that are still answered are the same rows as on #29, and none of them is `two_names`.

The abstain-dev bank (`26889048…`, n=327) changes no fact. Rollup stays answerable hits 195/217 with false abstain 22/217, missing_rel abstained 60/68, absent 42/42. Six English rows narrow to the focus and were already correct: "Of {other} and {focus}, the porter asks only {focus}". That is the same operator without the word `about`. `只查{focus}的{cue}` is not in the operator class; those Chinese rows were already the possessor.

## Probe and surface margin

`ask_targets` equals `ask_people` on all 442 probe rows, all 1124 surface-corpus strings, and all 50 indirect-dev questions. The restrictor does not fire on those sets, so the measured probe is the same path as #29. `eval_heldout` after the change:

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

Surface dev at margin 0.1: false abstain 0/192 English and 0/192 Chinese, missing 188/192 and 189/192, absent 36/36 both, deictic false 0/6 both. Those rows are one name, so the restrictor cannot run.

## What stayed

Dismiss and the unique possessor still decide before the restrictor. #29's two-name wins are intact: English and Chinese missing_rel stay 6/6, Chinese answerable stays 6/6, English answerable moves from 5/6 to 6/6. Full adversarial hits go from 187/216 to 188/216, above the 187 floor, and false abstain stays 19/216. `name_variant`, `misleading_cue`, `bare_cue`, `deixis`, `compact_form`, `relation_paraphrase`, and `near_dup_absent` are cell-identical to the before column. An open `between` / `among` / `和` still keeps both names. `not only` and a coordinated pair still keep both. A single name inside a dismiss clause is still that person.

## Tests

`python3 -m pytest -q`: 281 passed. The new file locks the dev sha, the Jaccard cap, literal disjointness, the zero-query abstain decisions, that only the restrictor moves the focus rows, and a phone ridge on axis 121 where the other person's sentence is the query vector: without the restrictor that fact wins, with it the focus's fact wins, and a focus who lacks the relation abstains. Frozen numeric scores are not a pytest gate.
