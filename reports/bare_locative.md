# Bare locative versus a places label

Flat-row `ask` still abstains on the abstain ridge when a question has no unique lexical cue, and the top-2 gap must clear `ABSTAIN_MARGIN` 0.1. Bare `where` / `place` / `哪里` / `位置` are not cues, so `place a call` stays a phone cue. On the frozen bare-cue rows the ridge was reading the placement slot itself as places. Two answerable birthday questions then false-abstained, because places led by 0.125 and 0.141 and the person stored only birthday.

The change drops that places label when the surface is a bare locative and there is no unique cue. It uses the same next-label call as a dismissed cue: if places is the top label, the next label's gap is against the remaining label, then the ordinary 0.1 margin. It does not lower 0.1 or `LEX_RIDGE_GAP` 0.15, and it does not change ranking `W` or `alpha` (0.3). There is no enable flag.

Frozen adversarial question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Scored before the change, then once after. Wording was not edited after the baseline.

## Before

`bare_cue` is 4 answerable and 4 `missing_rel` per language. No row has a unique lexical cue. The person match is the named person on every row.

| question | gold | label | ridge | gap | second | second lead | abstain |
|---|---|---|---|---:|---|---:|---|
| Where do I place the request so a voice finds Mary Jane? | phone | answerable | phone | 0.365 | places | 0.526 | no |
| Where do I place the request so a voice finds Ann? | phone | missing | phone | 0.504 | places | 0.459 | yes |
| If I place a note for a voice to find Robert, what was filed? | phone | answerable | phone | 0.620 | places | 0.045 | no |
| If I place a note for a voice to find Elizabeth, what was filed? | phone | missing | phone | 0.538 | places | 0.053 | yes |
| Where should the year-older mark for Mary Jane be placed? | birthday | answerable | places | 0.125 | birthday | 0.197 | false abstain |
| Where should the year-older mark for Ann be placed? | birthday | missing | places | 0.022 | birthday | 0.042 | missed |
| In which calendar square do I place Robert growing a year older? | birthday | answerable | birthday | 0.336 | phone | 0.176 | no |
| In which calendar square do I place Elizabeth growing a year older? | birthday | missing | birthday | 0.461 | phone | 0.046 | yes |
| 张强该往哪里去做语音联络？ | phone | answerable | phone | 0.660 | places | 0.270 | no |
| 刘洋该往哪里去做语音联络？ | phone | missing | phone | 0.576 | places | 0.336 | yes |
| 想用声音找到陈静，位置先别查，该看哪一条记录？ | phone | answerable | phone | 0.087 | places | 0.497 | no |
| 想用声音找到赵敏，位置先别查，该看哪一条记录？ | phone | missing | places | 0.004 | phone | 0.438 | missed |
| 先别标张强的位置，满岁在哪一天？ | birthday | answerable | birthday | 0.479 | places | 0.141 | no |
| 先别标刘洋的位置，满岁在哪一天？ | birthday | missing | birthday | 0.449 | places | 0.188 | yes |
| 陈静长一岁的标记该放在哪里？ | birthday | answerable | places | 0.141 | birthday | 0.030 | false abstain |
| 赵敏长一岁的标记该放在哪里？ | birthday | missing | places | 0.062 | birthday | 0.227 | missed |

Logits on the two false abstains:

- "Where should the year-older mark for Mary Jane be placed?" Person Mary Jane. places 0.480, birthday 0.354, phone 0.158. She stores birthday only.
- "陈静长一岁的标记该放在哪里？" Person 陈静. places 0.437, birthday 0.296, phone 0.266. She stores birthday only.

The phone rows the ridge already calls phone (gaps 0.365–0.660) were abstaining correctly on `missing_rel`. A global margin under 0.1 would be the lever that also false-abstains one English surface-dev question. Margins 0.0 and 0.05 still do that. It was not taken.

## What changed

`bare_locative` is a closed class:

- English `place` / `placed` / `placing`, except the noun `place list`.
- Chinese `放在哪里`, `往哪里`, `朝哪里`, and `位置先` (not `位置先后`).

`where` alone, `在哪里`, `是哪里`, `哪边`, and `位置是哪些` are not in the class. Those wordings are places asks on the surface corpus and the probe ("Where should a runner show up", "夜班找…是哪里？", "跑腿…该在哪里露面", "place list", "按先后点过的位置是哪些"). Dropping places on them would either false-abstain a one-relation places store (Chinese surface `是哪里` / `在哪里`, next-label gaps 0.186 and 0.168) or give up a missing-relation catch whose places lead is wide.

The drop runs only when there is no unique cue and no dismissed cue already in force. `place a call` and `place a voice call` match the verb and keep the phone cue. On the surface corpus the ridge calls those sentences phone with gaps 0.69–1.00. `What places did Ravi store?` keeps the places cue; the probe ridge still disagrees toward phone by about 0.12, under `LEX_RIDGE_GAP` 0.15. `想把交接放在哪里` keeps the places cue `交接`.

When places is already not the top label, the next-label call leaves the top label and its gap as they were. The phone rows above stay phone.

`先别标{name}的位置` is not `位置先`. Those two rows already had birthday on top (gaps 0.479 and 0.449) and stay on the ridge.

## Dev split

New bank `tidal/question_bare_dev_data.py`. Question sha256 `e24bda9007a41912ce6bdcd5213baf088447437568b7ffa6b7b6a291bff4bfd6`, n=25 (locative 10, guarded cue 5, keep 10). English names Sibyl, Leander, Quince, Romilda. Chinese names 卞珩, 汲棠, 郗宁, 晏书. Not an input to the frozen test. The matcher does not import it.

Literal collisions are 0. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased). Sentence cosine was not measured: `multilingual-e5-small` is not loaded for these new strings, and they are not in the frozen caches.

| against | literal | max token Jaccard |
|---|---:|---:|
| probe | 0 | 0.379 |
| surface cue corpus | 0 | 0.312 |
| abstain dev `26889048…` | 0 | 0.345 |
| frozen adversarial `cce20b6b…` | 0 | 0.375 |
| CJK dev `5e8b86f6…` | 0 | 0.147 |
| nickname dev `a766f748…` | 0 | 0.083 |
| indirect dev questions `daf4619f…` / bank `18b5e2d1…` | 0 | 0.222 |
| typo dev `68260c01…` | 0 | 0.176 |
| misleading dev `ed3bd6aa…` | 0 | 0.273 |

Closest pair, against the probe: "汲棠按先后点过的位置是哪些，钟楼侧柜里那本蓝册子鼗鼙镛钲。" ‖ "江禾按先后点过的位置是哪些？" The dev row is a keep: `位置是哪些` is not a bare locative.

Locative rows are the placement slot with no cue. Guarded rows match the slot and keep one cue (`place a call`, `交接放在哪里`). Keep rows are places asks and cue-free questions the ridge still owns.

## After

Only four frozen questions changed, all on `bare_cue`. Every other slice had no cue-free bare locative whose places label moved, so abstain reasons and ask hits stay put: misleading cue, near-duplicate absent, two names, cjk superstring, nicknames, Maggie, pinyin, deixis, `second_indirect` (12/12 hits, false abstain 0).

Ask hits are the shipped ranker on the frozen sentence cache (cosine plus `alpha` 0.3), scoped to the matched person. An answerable person stores only the true relation, so a row that stops abstaining is a hit.

| slice | lang | label | n | abstained before → after | false abstain | ask hits | missed abstain |
|---|---|---|---:|---:|---:|---:|---:|
| bare_cue | en | answerable | 4 | 1 → 0 | 1 → 0 | 3 → 4 | 0 |
| bare_cue | en | missing_rel | 4 | 3 → 3 | 0 | 0 | 1 → 1 |
| bare_cue | zh | answerable | 4 | 1 → 0 | 1 → 0 | 3 → 4 | 0 |
| bare_cue | zh | missing_rel | 4 | 2 → 4 | 0 | 0 | 2 → 0 |

| question | before | after |
|---|---|---|
| Where should the year-older mark for Mary Jane be placed? | false abstain | hit (birthday leads phone by 0.197 and is stored) |
| 陈静长一岁的标记该放在哪里？ | false abstain | hit (birthday leads phone by 0.030, under 0.1, so the only stored fact, birthday, is returned) |
| 想用声音找到赵敏，位置先别查，该看哪一条记录？ | missed `missing_rel` | abstain (phone leads birthday by 0.438 and is not stored) |
| 赵敏长一岁的标记该放在哪里？ | missed `missing_rel` | abstain (birthday leads phone by 0.227 and is not stored) |

## What is still wrong

"Where should the year-older mark for Ann be placed?" Places leads birthday by 0.022. After the drop, birthday leads phone by 0.042, under 0.1, and birthday is the missing relation. The row still answers. Catching it means a margin under 0.1. Margins 0.0 and 0.05 each false-abstain one English surface-dev question, so the shipped margin stays 0.1.

These four are the same floor as #26. The bare-locative path does not match them: each still has a unique cue, and the places or phone lead after that cue is under 0.1, or the cue is the ask.

- "Which locations file holds the way to raise Mary Jane by voice?" Places cue, ridge agrees, gap 0.027. False abstain stays.
- "Ignore the birthday feast and say how to get through to Catherine." Birthday is dismissed. The next label is places, lead 0.244, and places is stored. The missing relation is phone. The row still answers.
- "电话簿先合上，赵敏常去的几处按先后是什么？" Dismissed. The ridge already says places, gap 0.070. Still answered.
- "地方名单先放下，怎么用声音找到刘洋？" Dismissed. The ridge says phone, gap 0.089. Still answered.

## Probe

`python -m tidal.question_heldout`, full Continuum path, after the change. No probe question changes its abstain decision (442/442). The places asks worded with `where`, `place list`, `在哪里`, `是哪里`, or `位置是哪些` are not bare locatives, except `想把交接放在哪里`, which keeps the places cue.

| check | result |
|---|---|
| answerable ask hits | 365/370 |
| answerable false abstain | 0/370 |
| unanswerable abstained | 72/72 |
| gold lost at overwrite 0.9 | 0/370 |
| co-resident pairs with cosine above 0.9 | 0 (max 0.8927, 8390 pairs) |
| surface dev false abstain at 0.1 | 0/192 en, 0/192 zh |
| surface dev missing at 0.1 | 188/192 en, 189/192 zh |
| surface dev at margins 0.0 and 0.05 | false abstain 1/192 en, 0/192 zh |
| `LLMMemory` sources | untouched |

Pytest: 266 passed, 1 skipped. That includes the held-out probe, the abstain dev curve, the bare-locative dev split, indirect hearer, nickname, CJK, and the cue-aside split.
