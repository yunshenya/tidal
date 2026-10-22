# Misleading cue versus the abstain ridge

Flat-row `ask` still takes a unique lexical cue (`call` / `birthday` / `place`, and the Chinese cue words) unless the abstain ridge disagrees by at least `LEX_RIDGE_GAP` 0.15. After that yield, the cue still wins when the ridge agrees with it, or when the gap is under 0.15. Several of those sentences are not asking the relation the cue names. The cue sits in a clause the sentence sets aside, and the ask is the next clause.

The change drops that unique cue. It does not lower 0.15 or the 0.1 margin, and it does not change ranking `W` or `alpha` (0.3). There is no enable flag and no phrase list. A cue after the clause break stays the ask, which is what keeps "Ignore the side topics and read Owen's birthday" and "What places did Ravi store?" on the old path. The probe ridge calls Ravi's question phone at about 0.12.

Frozen adversarial question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Scored before the change, then once after. Wording was not edited after the baseline.

## What changed

A dismissed clause is a closed class:

- English verbs: ignore / ignoring, skip / skipping, forget / forgetting, exclude / excluding. The clause is the complement, from the verb to the next break.
- `aside`, and the decoy nouns `distraction` and `wrong drawer`. The clause runs from the previous break through the next one, so the cue may sit before the mark ("birthday aside", "地点以后再说").
- Chinese marks: `别管` / `别提` / `别看` / `别查`, `先别`, `先不`, `跳过`, `放一边`, `一边去`, `以后再`, `先合上`, `先放下`, `先收`. Same whole-clause span.

Breaks are `and`, `并且`, `然后`, and `， , ; ； : ： . 。`. A period after `Mr.` / `Ms.` / a single initial is not a break. The clause counts only when text follows the break. Every span of the cue has to sit inside one such clause. One span after the break keeps the cue. `lex_labels` is unchanged, so a two-name question still focuses the name nearest the cue. Only the abstain decision drops it.

If the abstain ridge's top label is the dropped cue, the next label is used and its gap is against the remaining label, then the ordinary 0.1 margin. A confident wrong cue must not become a confident wrong abstain. If the ridge already preferred a different label, that label and its existing gap go through the same margin. No ridge, and the dismissed cue does not abstain on its own.

`LEX_RIDGE_GAP` stays 0.15. `ABSTAIN_MARGIN` stays 0.1. The surface dev curve is unchanged: false abstain 0/192 per language, missing 188 en / 189 zh, absent 36/36, deictic false 0/6. Margins 0.0 and 0.05 still false-abstain one English surface-dev question.

## Dev split

New bank `tidal/question_cue_dev_data.py`. Question sha256 `ed3bd6aadfa92ed4d9843260ef1788875c83019fcb0046491d3928394c08628f`, n=44 (set aside 20, cue after the break 12, genuine 6, polysemy 6). English names Eudora, Barnaby, Winslow, Petronella. Chinese names 闵舟, 濮兰, 隗衡, 郏棠. Not an input to the frozen test. The matcher does not import it.

Literal collisions are 0. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased). Sentence cosine was not measured: `multilingual-e5-small` is not in this environment, and the new strings are not in the frozen caches.

| against | literal | max token Jaccard |
|---|---:|---:|
| probe | 0 | 0.417 |
| surface cue corpus | 0 | 0.385 |
| abstain dev `26889048…` | 0 | 0.333 |
| frozen adversarial `cce20b6b…` | 0 | 0.385 |
| CJK dev `5e8b86f6…` | 0 | 0.185 |
| nickname dev `a766f748…` | 0 | 0.059 |
| indirect dev questions `daf4619f…` / bank `18b5e2d1…` | 0 | 0.286 |
| typo dev `68260c01…` | 0 | 0.158 |

Closest pair, against the probe: "Ignore the side chatter and read Eudora's telephone, ostensory." ‖ "Ignore the side topics and read Owen's birthday." The cue is after `and` in both, so neither is set aside.

Polysemy rows (`locations file`, `call round`, `打电话`) are labeled not set aside. The class does not claim them.

## Baseline, frozen test, before the change

`misleading_cue` is 6 answerable and 6 `missing_rel` per language. A unique cue with no disagreeing ridge is `cue_wins`. `missing_rel` stores the misleading relation, so a cue win there is a missed abstain. Answerable stores only the true relation, so a cue win there is a false abstain.

| lang | label | n | ridge wins | cue wins, ridge agrees | cue wins, gap under 0.15 | abstained | false abstain |
|---|---|---:|---:|---:|---:|---:|---:|
| en | answerable | 6 | 3 | 3 | 0 | 3 | 3 |
| en | missing_rel | 6 | 4 | 2 | 0 | 4 | 0 |
| zh | answerable | 6 | 2 | 2 | 2 | 4 | 4 |
| zh | missing_rel | 6 | 3 | 1 | 2 | 3 | 0 |

Answerable cue wins, with the person's stored fact closer to the question than the cue relation:

| question | cue | ridge | gap | second label | second lead |
|---|---|---|---:|---|---:|
| Ignore the birthday feast and say how to get through to William. | birthday | birthday | 0.308 | places | 0.083 |
| Skip Margaret's birthday feast and keep the spoken order of haunts. | birthday | birthday | 0.350 | places | 0.098 |
| Which locations file holds the way to raise Mary Jane by voice? | places | places | 0.027 | phone | 0.552 |
| 电话簿先合上，陈静常去的几处按先后是什么？ | phone | phone | 0.068 | places | 0.414 |
| 生日先放一边，黄磊天黑后怎么联络得上？ | birthday | birthday | 0.009 | phone | 0.190 |
| 别管王伟的生日，常去的几处按先后是哪些？ | birthday | places | 0.050 | birthday | 0.353 |
| 地方名单先放下，怎么用声音找到张强？ | places | phone | 0.144 | places | 0.658 |

`bare_cue` has no unique lexical cue (`where` / `place` / `哪里` / `位置` are not cues). It was already 1 false abstain per language on the answerable rows, and 1 en / 2 zh missed `missing_rel`. Yield never applied. Counts below are unchanged.

## After

Only `misleading_cue` moved. Every other frozen slice had zero unique cues set aside, so abstain reasons and ask hits stay put: near-duplicate absent, two names, cjk superstring, nicknames, Maggie, pinyin, deixis, `second_indirect`.

Ask hits are the shipped ranker on the frozen sentence cache (cosine plus `alpha` 0.3). Before-hits match the table from the cue-yield change (en 3, zh 2). An answerable person stores only the true relation, so a row that stops abstaining is a hit.

| slice | lang | label | n | abstained before → after | false abstain | ask hits | missed abstain |
|---|---|---|---:|---:|---:|---:|---:|
| misleading_cue | en | answerable | 6 | 3 → 1 | 3 → 1 | 3 → 5 | 0 |
| misleading_cue | en | missing_rel | 6 | 4 → 5 | 0 | 0 | 2 → 1 |
| misleading_cue | zh | answerable | 6 | 4 → 0 | 4 → 0 | 2 → 6 | 0 |
| misleading_cue | zh | missing_rel | 6 | 3 → 4 | 0 | 0 | 3 → 2 |
| bare_cue | en | answerable | 4 | 1 → 1 | 1 → 1 | 3 → 3 | 0 |
| bare_cue | en | missing_rel | 4 | 3 → 3 | 0 | 0 | 1 → 1 |
| bare_cue | zh | answerable | 4 | 1 → 1 | 1 → 1 | 3 → 3 | 0 |
| bare_cue | zh | missing_rel | 4 | 2 → 2 | 0 | 0 | 2 → 2 |

Eight questions changed, all on `misleading_cue`:

| question | before | after |
|---|---|---|
| Ignore the birthday feast and say how to get through to William. | false abstain | hit (second label places leads by 0.083, under 0.1, so the only stored fact, phone, is returned) |
| Skip Margaret's birthday feast and keep the spoken order of haunts. | false abstain | hit (places leads by 0.098 and is stored) |
| Skip Patricia's birthday feast and keep the spoken order of haunts. | missed `missing_rel` | abstain (places leads by 0.130 and is not stored) |
| 电话簿先合上，陈静常去的几处按先后是什么？ | false abstain | hit |
| 生日先放一边，黄磊天黑后怎么联络得上？ | false abstain | hit |
| 生日先放一边，周涛天黑后怎么联络得上？ | missed `missing_rel` | abstain (phone leads by 0.180 and is not stored) |
| 别管王伟的生日，常去的几处按先后是哪些？ | false abstain | hit (ridge already said places, gap 0.050, places stored) |
| 地方名单先放下，怎么用声音找到张强？ | false abstain | hit (ridge already said phone, gap 0.144, phone stored) |

## What is still wrong

These four are the same mechanism's floor, not a second list of phrases.

- "Which locations file holds the way to raise Mary Jane by voice?" The places cue is the ask's container, not a dismissed clause. The ridge agrees (places, gap 0.027). False abstain stays. `call round` / "folks call" rows were already ridge-yield hits and are not dismissed.
- "Ignore the birthday feast and say how to get through to Catherine." The birthday cue is dismissed. The next ridge label is places, lead 0.244, and places is stored. The missing relation is phone. The row still answers.
- "电话簿先合上，赵敏常去的几处按先后是什么？" Dismissed. The ridge already says places, gap 0.070, under 0.1, and places is the missing relation. Still answered.
- "地方名单先放下，怎么用声音找到刘洋？" Dismissed. The ridge says phone, gap 0.089, under 0.1. Still answered.

Lowering the margin to 0.05, or the yield gap toward the Ravi disagreement at 0.12, is the move that false-abstains the surface dev curve or the probe. It was not taken. Bare `where` / `place` / `哪里` have no unique cue, so this clause rule does not touch them. "想用声音找到赵敏，位置先别查，该看哪一条记录？" contains `先别`, and `位置` is not a cue.

## Probe

`python -m` held-out eval, full Continuum path, after the change. No probe question sets a unique cue aside.

| check | result |
|---|---|
| answerable ask hits | 365/370 |
| answerable false abstain | 0/370 |
| unanswerable abstained | 72/72 |
| gold lost at overwrite 0.9 | 0/370 |
| co-resident pairs with cosine above 0.9 | 0 (max 0.893, 8390 pairs) |
| surface dev false abstain | 0/192 en, 0/192 zh |
| `LLMMemory` sources | untouched |

Pytest: 257 passed, 1 skipped. That includes the held-out probe, the abstain dev curve, indirect hearer, nickname, and CJK.
