# Bare locative versus a places label

Flat-row `ask` abstains on the abstain ridge when a question has no unique lexical cue and the top-2 gap clears `ABSTAIN_MARGIN` 0.1. Bare `where` / `place` / `哪里` / `位置` are not cues, so `place a call` stays phone. On the frozen bare-cue rows the ridge was reading a record's placement slot as places. Two answerable birthday questions then false-abstained.

The first cut of this change dropped `places` on every cue-free `place` / `placed` / `placing` (not `place list`) and on `放在哪里` / `往哪里` / `朝哪里` / `位置先`. That fixed the frozen rows. An expanded dev bank, built after that cut and not used to word the frozen set, shows the same pattern also matches places asks whose object is the person (`place Sibyl`, `where was Leander placed`, `把卞珩放在哪里`). Those stay on the ridge. A record slot still drops `places`. The margin and `LEX_RIDGE_GAP` 0.15 are unchanged. There is no enable flag.

The unrestricted pattern was too wide, not a rule that only fires on the two frozen templates. Those templates place a mark, not the person, and they still take the drop. Wider verbs (`put`, `file`, `deposit`, `搁在`) do not. `on file` is a noun, and treating it as a slot false-abstains a frozen places paraphrase.

Frozen adversarial question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Wording was not edited. Dev question sha `a0d7309555ecb504c1f44bc382786c0ccff35f9a339d95b685b7a148cb465d23`, n=100.

## Frozen bare_cue before any drop

`bare_cue` is 4 answerable and 4 `missing_rel` per language. No row has a unique lexical cue. The person match is the named person on every row.

| question | gold | label | ridge | gap | second lead | abstain |
|---|---|---|---|---:|---:|---|
| Where do I place the request so a voice finds Mary Jane? | phone | answerable | phone | 0.365 | 0.526 | no |
| Where do I place the request so a voice finds Ann? | phone | missing | phone | 0.504 | 0.459 | yes |
| If I place a note for a voice to find Robert, what was filed? | phone | answerable | phone | 0.620 | 0.045 | no |
| If I place a note for a voice to find Elizabeth, what was filed? | phone | missing | phone | 0.538 | 0.053 | yes |
| Where should the year-older mark for Mary Jane be placed? | birthday | answerable | places | 0.125 | 0.197 | false abstain |
| Where should the year-older mark for Ann be placed? | birthday | missing | places | 0.022 | 0.042 | missed |
| In which calendar square do I place Robert growing a year older? | birthday | answerable | birthday | 0.336 | 0.176 | no |
| In which calendar square do I place Elizabeth growing a year older? | birthday | missing | birthday | 0.461 | 0.046 | yes |
| 张强该往哪里去做语音联络？ | phone | answerable | phone | 0.660 | 0.270 | no |
| 刘洋该往哪里去做语音联络？ | phone | missing | phone | 0.576 | 0.336 | yes |
| 想用声音找到陈静，位置先别查，该看哪一条记录？ | phone | answerable | phone | 0.087 | 0.497 | no |
| 想用声音找到赵敏，位置先别查，该看哪一条记录？ | phone | missing | places | 0.004 | 0.438 | missed |
| 先别标张强的位置，满岁在哪一天？ | birthday | answerable | birthday | 0.479 | 0.141 | no |
| 先别标刘洋的位置，满岁在哪一天？ | birthday | missing | birthday | 0.449 | 0.188 | yes |
| 陈静长一岁的标记该放在哪里？ | birthday | answerable | places | 0.141 | 0.030 | false abstain |
| 赵敏长一岁的标记该放在哪里？ | birthday | missing | places | 0.062 | 0.227 | missed |

Logits on the two false abstains: Mary Jane places 0.480 / birthday 0.354 / phone 0.158, birthday only stored. 陈静 places 0.437 / birthday 0.296 / phone 0.266, birthday only stored.

## Dev bank

`tidal/question_bare_dev_data.py`. Not an input to the frozen test. The matcher does not import it.

n=100, 50 English and 50 Chinese. `answerable` 32, `missing_rel` 28, `keep_places` 28, `guarded` 12. Names: Sibyl, Leander, Quince, Romilda, 卞珩, 汲棠, 郗宁, 晏书.

- `answerable`: cue-free slot, gold phone or birthday, store keeps only that relation. A places label must not false-abstain.
- `missing_rel`: same questions' class, store keeps the other two relations.
- `keep_places`: a places ask. Bare `where` / `哪里`, `where does X dwell`, `place list`, `位置是哪些`, `在哪里`, `哪一带`, and a places cue. Four rows put the person in the slot (`place Sibyl`, `where was Leander placed`, `把卞珩放在哪里`, `这个人该往哪里`).
- `guarded`: the slot is real and so is one cue (`place a call`, `交接放在哪里`, `蜡烛`, `地点`).

Families wider than the frozen templates are in the bank on purpose: `put`, `lodge`, `tuck`, `file`, `deposit`, `slot`, `pin`, `搁在`, `摆在`, `填在`, `记在`, `写在`, `塞到`, `安放`, `置于`, and a drawer with no verb.

Literal collisions are 0. Token Jaccard is the probe tokenizer. Sentence cosine of these strings was measured with frozen `multilingual-e5-small` (revision `614241f6…`, mean pool, L2, maxlen 64, no query prefix) when the rule was chosen. The vectors are not committed.

| against | literal | max token Jaccard |
|---|---:|---:|
| probe | 0 | 0.304 |
| surface cue corpus | 0 | 0.333 |
| abstain dev `26889048…` | 0 | 0.333 |
| frozen adversarial `cce20b6b…` | 0 | 0.333 |
| CJK dev `5e8b86f6…` | 0 | 0.174 |
| nickname dev `a766f748…` | 0 | 0.133 |
| indirect dev questions `daf4619f…` / bank `18b5e2d1…` | 0 | 0.222 |
| typo dev `68260c01…` | 0 | 0.214 |
| misleading dev `ed3bd6aa…` | 0 | 0.333 |

Closest pair, against the surface corpus: "Which places does Quince keep on the tin card, takahe weka." ‖ "Which places does Ingrid keep for the handoff roster?" The dev row is a keep: it has the places cue.

## What the dev ridge does

Scored with the shipped abstain ridge, harsh stores, margin 0.1. `none` is no places drop. `unrestricted` is the first cut (every cue-free placement slot). `shipped` is that cut except when the object is a person.

| rule | answerable false abstain | missing_rel caught | keep_places diverged from none | guarded diverged |
|---|---:|---:|---:|---:|
| none | 5/32 | 16/28 | 0 | 0 |
| unrestricted | 5/32 | 16/28 | 5 | 0 |
| shipped (person excluded) | 5/32 | 16/28 | 0 | 0 |
| also drop `put` / `file` / `deposit` / `搁在` / …, still excluding a person | 5/32 | 17/28 | 0 | 0 |
| also drop every `where` / `哪里` | 5/32 | 17/28 | 15 | 0 |

The five answerable false abstains are the same under every rule. The ridge says phone, not places, and the gap is already over 0.1:

| question | phone | birthday | places | gap |
|---|---:|---:|---:|---:|
| Sibyl's added-year tick stays where the clerk has it placed | 0.455 | 0.217 | 0.313 | 0.141 |
| I place Quince's added-year tick on which square of the slate | 0.475 | 0.252 | 0.260 | 0.215 |
| Romilda's added-year tick, where ought the clerk put it | 0.572 | 0.248 | 0.175 | 0.324 |
| Which square slots Quince's added-year tick | 0.550 | 0.270 | 0.170 | 0.279 |
| Romilda's added-year tick is pinned to which square | 0.492 | 0.290 | 0.207 | 0.202 |

Dropping places does not move a phone-top row. Catching them means a margin under 0.1, or a new birthday cue. Margins 0.0 and 0.05 each false-abstain one English surface-dev question. Neither was taken.

The unrestricted drop does not reduce those five, and it breaks three keep rows:

| question | ridge | gap | what the drop does |
|---|---|---:|---|
| Where does the parish place Sibyl for the evening halt | places 0.619 / birthday 0.306 / phone 0.061 | 0.313 | false abstain on a places-only store (next lead 0.245); loses the missing-places catch |
| Where was Leander placed for the evening halt | places 0.684 / birthday 0.315 / phone 0.003 | 0.369 | same (next lead 0.312) |
| 夜巡要把卞珩放在哪里 | places 0.514 / phone 0.267 / birthday 0.225 | 0.247 | loses the missing-places catch (next lead 0.042, under 0.1, so the places-only store does not false-abstain) |

`汲棠这个人该往哪里去歇` matches `往哪里` and is a places ask. The ridge already says phone with gap 0.056, so the drop would not move the decision. The exclusion covers it anyway. `place Sibyl's tick` is a record (the possessive), and the drop still applies.

The extra missing catch from a wider verb list is one sentence: "The added-year seal for Romilda is deposited where" (places 0.470, birthday then leads phone by 0.131). The same list matches `\bfile\b`. On the frozen set that false-abstains "Read the ordered haunts on file for Mary Jane" (places gap 0.232) and drops the missing catch for Ann. The wider list was not shipped. `deposit` alone would have caught the one dev row and touched nothing in the probe, the surface curve, or the frozen set. It is one authored sentence. It was not added.

Dropping every `where` / `哪里` moves 15 keep-places decisions, including "Where does Leander stay" (places gap 0.603) and "Where, for Romilda, is the usual halt" (places gap 0.342). Not shipped.

Guarded rows (`place a call`, `交接放在哪里`, the candle, `地点`) do not move under any of these drops. The cue blocks the drop. On the surface corpus, "To place a call through to Ingrid, what digits leave the booth?" is phone with a gap above 0.15. "What places did Ravi store?" keeps the places cue; the probe ridge still leans phone by about 0.12, under 0.15.

## Shipped rule

`bare_locative` is unchanged as a surface test: `place` / `placed` / `placing` except `place list`, and `放在哪里` / `往哪里` / `朝哪里` / `位置先` (not `位置先后`). The drop runs only when there is no unique cue, no dismissed cue, and `_placement_object_is_person` is false.

The object is a person when the text says `place` / `placed` / `placing` plus `him` / `her` / `them` or a stored name that is not possessive, or `where was he/she/they/{name} placed`, or `把{name}放在哪里`, or `这个人该往哪里` / `人放在哪里`. `place Sibyl's tick`, `把它放在哪里`, and `标记该放在哪里` are records. `{name}该往哪里去做语音联络` is not the person-object test: that frozen phone row stays a slot, and the ridge already calls it phone.

## Frozen after

Abstain on the frozen cache differs from no drop on exactly four questions, all `bare_cue`, and differs from the unrestricted drop on none. Ask hits below are that abstain gate: an answerable person stores only the true relation, so a row that stops abstaining is a hit. Misleading cue, near-duplicate absent, two names, cjk superstring, nicknames, Maggie, pinyin, deixis, and `second_indirect` do not move.

| slice | lang | label | n | abstained before → after | false abstain | ask hits | missed abstain |
|---|---|---|---:|---:|---:|---:|---:|
| bare_cue | en | answerable | 4 | 1 → 0 | 1 → 0 | 3 → 4 | 0 |
| bare_cue | en | missing_rel | 4 | 3 → 3 | 0 | 0 | 1 → 1 |
| bare_cue | zh | answerable | 4 | 1 → 0 | 1 → 0 | 3 → 4 | 0 |
| bare_cue | zh | missing_rel | 4 | 2 → 4 | 0 | 0 | 2 → 0 |

| question | before | after |
|---|---|---|
| Where should the year-older mark for Mary Jane be placed? | false abstain | hit (birthday leads phone by 0.197 and is stored) |
| 陈静长一岁的标记该放在哪里？ | false abstain | hit (birthday leads phone by 0.030, under 0.1, so the only stored fact is returned) |
| 想用声音找到赵敏，位置先别查，该看哪一条记录？ | missed `missing_rel` | abstain (phone leads birthday by 0.438 and is not stored) |
| 赵敏长一岁的标记该放在哪里？ | missed `missing_rel` | abstain (birthday leads phone by 0.227 and is not stored) |

"Where should the year-older mark for Ann be placed?" still answers. After the drop, birthday leads phone by 0.042. Catching it needs a margin under 0.1.

These four from #26 are still open. Each has a unique cue, so this path does not match them.

- "Which locations file holds the way to raise Mary Jane by voice?" Places cue, ridge agrees, gap 0.027.
- "Ignore the birthday feast and say how to get through to Catherine." Birthday is dismissed. Next label places, lead 0.244, and places is stored. The missing relation is phone.
- "电话簿先合上，赵敏常去的几处按先后是什么？" Places gap 0.070.
- "地方名单先放下，怎么用声音找到刘洋？" Phone gap 0.089.

## Probe

`python -m tidal.question_heldout` after the person exclusion. No probe question changes its abstain decision relative to the unrestricted drop (442/442), and the full path matches the previous cut.

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

Pytest: 269 passed. That includes the held-out probe, the abstain dev curve, the expanded bare-locative split, indirect hearer, nickname, CJK, and the cue-aside split. The tiny-llama test runs in this environment because `transformers` is installed. It was not edited.
