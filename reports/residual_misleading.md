# Residual misleading cues

Flat-row `ask` still takes a unique lexical cue unless the abstain ridge disagrees by at least `LEX_RIDGE_GAP` 0.15. A cue inside a dismissed clause is dropped, and the next label then faces `ABSTAIN_MARGIN` 0.1. After #26 and #27, four frozen `misleading_cue` rows are still wrong. Each still has a unique cue, so the record-slot drop does not see them.

This change is measurement-only. `question_surface.py` is untouched. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen adversarial wording was not edited, and it was not an input to a threshold.

Frozen adversarial question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Dev question sha `b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5`, n=100. Scored on current main (`d54aa6b`). Before and after are the same numbers, because the matcher did not change.

## The four leftovers

`misleading_cue` is 6 answerable and 6 `missing_rel` per language. Answerable stores only the true relation, so a cue win is a false abstain. `missing_rel` stores every other relation, including the misleading cue, so a cue win there is a missed abstain. Logits are the shipped abstain ridge (`Wa`, ridge 1.0).

| question | gold | cue | aside | ridge | gap | decision |
|---|---|---|---|---|---:|---|
| Which locations file holds the way to raise Mary Jane by voice? | phone | places | no | places | 0.027 | false abstain |
| Which locations file holds the way to raise Ann by voice? | phone | places | no | phone | 0.187 | abstain |
| Ignore the birthday feast and say how to get through to William. | phone | birthday | yes | places, after the drop | 0.083 | hit |
| Ignore the birthday feast and say how to get through to Catherine. | phone | birthday | yes | places, after the drop | 0.244 | missed abstain |
| 电话簿先合上，陈静常去的几处按先后是什么？ | places | phone | yes | places | 0.414 | hit |
| 电话簿先合上，赵敏常去的几处按先后是什么？ | places | phone | yes | places | 0.070 | missed abstain |
| 地方名单先放下，怎么用声音找到张强？ | phone | places | yes | phone | 0.144 | hit |
| 地方名单先放下，怎么用声音找到刘洋？ | phone | places | yes | phone | 0.089 | missed abstain |

Mary Jane logits: phone +0.501, birthday −0.051, places +0.529. The places cue agrees, and the gap is under 0.1, so the margin is never consulted. Ann is the same template. The name moves the ridge to phone with gap 0.187, which clears 0.15, and phone is absent, so the row abstains.

Catherine logits: birthday +0.587, places +0.329, phone +0.085. The birthday cue is dismissed. Places then leads phone by 0.244, and places is stored. The asked relation is phone. William's twin, after the same drop, has places leading phone by 0.083. Only phone is stored, so the row answers. That 0.083 is the answerable row a lower set-aside margin would false-abstain.

赵敏 logits: phone +0.407, birthday +0.114, places +0.477. The phone cue is dismissed. The ridge already says places, gap 0.070, and places is the missing relation. 刘洋 logits: phone +0.615, birthday −0.147, places +0.526. The places cue is dismissed. Phone leads by 0.089. 陈静 and 张强 are the answerable twins (gaps 0.414 and 0.144) and already hit.

The three gaps that a single set-aside margin would have to order are 赵敏 0.070047 (must abstain), William 0.082638 (must answer), 刘洋 0.088745 (must abstain). The table rounds those to 0.070, 0.083, and 0.089. William sits between the two missed abstains. A cutoff low enough to catch 赵敏 also abstains William. A cutoff high enough to spare William misses 赵敏, and the only band that catches 刘洋 while sparing William misses 赵敏. That band is a fit to this frozen pair.

## Why a global margin cut fails

Surface-dev curve, same bundle, harsh one-relation store. A margin is eligible only when false abstain and deictic false abstain are 0 in both languages.

| margin | false en | false zh | missing en | missing zh | deictic false en | deictic false zh |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1/192 | 0/192 | 191/192 | 192/192 | 0/6 | 0/6 |
| 0.05 | 1/192 | 0/192 | 191/192 | 192/192 | 0/6 | 0/6 |
| 0.10 | 0/192 | 0/192 | 188/192 | 189/192 | 0/6 | 0/6 |
| 0.15 | 0/192 | 0/192 | 186/192 | 183/192 | 0/6 | 0/6 |

The English false abstain at 0.00 and 0.05 is "Which cradle should the night clerk try for Maren?" No lexical cue. Ridge places, gap 0.063 (phone +0.399, birthday +0.137, places +0.463). Phone is the stored relation. The same cut is what would be required to catch 赵敏 at 0.070 and 刘洋 at 0.089, and it also false-abstains William.

Raising the yield so an agreeing cue abstains only when the gap clears a bar has the same collision on true catches:

| rule | what it changes outside the new dev |
|---|---|
| Agreeing unique cue with gap under 0.10 does not abstain | Fixes Mary Jane. Loses the abstain-dev two-name catch "先别看郑凯。值班要念的地方是胡斌的。" (places agrees, gap 0.079). Frozen two-name and surface-dev stay put. Catherine, 赵敏, and 刘洋 stay wrong. |
| Same bar at 0.15 | Also loses frozen "黄磊和张强里面，只问黄磊的生日。" (birthday agrees, gap 0.145) and surface "Which extension raises Keiko on the night line?" (phone agrees, gap 0.124). |

On the abstain-dev and surface banks, true catches with an agreeing cue sit on both sides of 0.10 and 0.15. The wrong-cue agreements do too: Mary Jane 0.027, Lorraine "Which locations slip lets a voice land on Lorraine?" places gap 0.284 (still a false abstain). Marcus, the missing twin, is places gap 0.019 with places stored. The gap follows the name. A threshold does not name the class.

## Dev bank

`tidal/question_residual_dev_data.py`. The matcher does not import it. The frozen test is not an input.

n=100, 50 English and 50 Chinese. Names: Godfrey, Linnea, Hadrian, Odette, 宓珩, 邹岚, 庞屹, 岑书.

| kind | n | what it holds |
|---|---:|---|
| `container_ans` / `container_miss` | 16 / 12 | Places cue is a record compound (`locations folder`, `地点卷宗`). The ask is phone or birthday and has no cue of its own. |
| `keep_ans` / `keep_miss` | 8 / 8 | The same compounds are the places ask. |
| `manner_ans` / `manner_miss` | 4 / 4 | A places ask that also says `voice` / `声音` as how to read it. |
| `novoice_ans` / `novoice_miss` | 4 / 4 | Phone ask inside a record compound, with no `voice` / `声音`. |
| `aside_ans` / `aside_miss` | 8 / 8 | The unique cue sits in a dismissed clause. The ask has no cue word. |
| `genuine_ans` / `genuine_miss` | 8 / 8 | The unique cue is the ask. |
| `outer_ans` | 8 | A dismiss word is present and the cue is after the break. |

Answerable stores only the asked relation. Missing stores the other two. Control rows are `keep`, `manner`, `genuine`, and `outer` answerable, plus `keep`, `manner`, and `genuine` missing. A rule that moves those is fitting the container wording.

Literal collisions are 0. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased, an apostrophe word is one token). Sentence cosine uses frozen `multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix). The vectors are not committed. Cosine is high because the questions are the same shape as other person-and-relation asks. The disjointness gate is literal equality and token Jaccard under 0.50.

| against | literal | max token Jaccard | max sentence cosine |
|---|---:|---:|---:|
| probe | 0 | 0.353 | 0.918 |
| surface cue corpus | 0 | 0.385 | 0.921 |
| abstain dev `26889048…` | 0 | 0.389 | 0.911 |
| frozen adversarial `cce20b6b…` | 0 | 0.421 | 0.920 |
| CJK dev `5e8b86f6…` | 0 | 0.227 | 0.927 |
| nickname dev `a766f748…` | 0 | 0.133 | 0.911 |
| indirect questions `daf4619f…` | 0 | 0.333 | 0.921 |
| indirect bank `18b5e2d1…` | 0 | 0.333 | 0.923 |
| typo dev `68260c01…` | 0 | 0.200 | 0.931 |
| misleading dev `ed3bd6aa…` | 0 | 0.429 | 0.954 |
| bare-locative dev `a0d73095…` | 0 | 0.412 | 0.955 |

Closest Jaccard pair, against the misleading dev: "地点清册一边去，庞屹入夜后该把声音送去哪一条，齰齻。" ‖ "隗衡的地点清册一边去，声音该落在哪一条，飝鼽。" The cue dev row is already set aside. The residual row is too.

## What was measured

Scored with the shipped abstain ridge and the shipped decision, including the record-slot drop. Harsh stores. Every alternative below is applied on top of that path. `file` and `名单` are absent from the dev compounds on purpose: those words are the frozen templates.

| rule | answerable false abstain | missing caught | control false abstain | control missed |
|---|---:|---:|---:|---:|
| shipped | 14/56 | 30/44 | 0/28 | 1/20 |
| agreeing cue, gap under 0.10, do not abstain | 10/56 | 30/44 | 0/28 | 1/20 |
| agreeing cue, gap under 0.15, do not abstain | 7/56 | 30/44 | 0/28 | 1/20 |
| drop a live places cue when it is a record compound, then the 0.1 margin | 6/56 | 30/44 | 0/28 | 1/20 |
| same, and also `locations file` / `名单` | 6/56 | 30/44 | 0/28 | 1/20 |
| drop the places label on that compound | 9/56 | 26/44 | 8/28 | 12/20 |
| add `voice` / `声音` as a phone cue | 8/56 | 30/44 | 0/28 | 1/20 |
| `voice` / `声音` forces phone | 10/56 | 32/44 | 4/28 | 4/20 |
| add that cue and drop dismissed cues | 9/56 | 28/44 | 0/28 | 1/20 |
| set-aside margin 0.05 | 15/56 | 30/44 | 0/28 | 1/20 |
| global margin 0.05 | 15/56 | 30/44 | 0/28 | 1/20 |
| embed the clause after the dismiss mark | 14/56 | 31/44 | 0/28 | 1/20 |

The one control miss under the shipped path is "庞屹的地方簿册用声音按落脚先后念，齰齳。" Places is the cue and the missing relation. The ridge says phone, gap 0.241, and phone is stored. The cue yields. The same miss remains under every rule except the label drop, which also false-abstains keep and manner rows.

Shipped false abstains on this bank, and which rules clear them:

| question | gold | ridge | gap | cleared by |
|---|---|---|---:|---|
| Godfrey's voice route sits inside a locations folder | phone | places | 0.144 | agree-at-0.15, label drop, voice-forces-phone |
| A locations ledger … Linnea is reached by voice | phone | places | 0.076 | tiny agree, compound margin, voice cue |
| Hadrian is reached by voice from the locations slip | phone | phone | 0.115 | compound margin, voice cue |
| The locations chit that lets a voice find Odette | phone | places | 0.008 | tiny agree, compound margin, voice cue |
| Inside the locations dossier, the voice path for Godfrey | phone | places | 0.066 | tiny agree, compound margin, voice cue |
| Linnea's voice path was copied onto a locations binder | phone | places | 0.136 | agree-at-0.15, label drop, voice-forces-phone |
| Odette adds a year on the date a locations logbook still holds | birthday | birthday | 0.146 | compound margin |
| 宓珩的声音路径夹在地点卷宗里 | phone | places | 0.062 | tiny agree, compound margin, voice cue |
| 地方案卷里留着岑书的声音去向 | phone | phone | 0.021 | compound margin, voice cue |
| The way to get through to Godfrey is the entry a locations folder still holds | phone | places | 0.702 | label drop |
| How Linnea is reached after dusk is written on a locations slip | phone | places | 0.469 | label drop |
| 地点卷宗里写着怎么联络上宓珩 | phone | places | 0.124 | agree-at-0.15, label drop |
| 哪一本地方簿册记着庞屹入夜后怎么找得着 | phone | phone | 0.035 | compound margin |
| Ignore the birthday supper and leave it; how is Godfrey reached after dusk | phone | birthday, after the drop | 0.385 | none |

The compound-margin rule is the only candidate that cuts dev false abstains (14 to 6) with control false abstains still 0 and the control miss unchanged. It leaves every confident places lead, including the two novoice rows at 0.702 and 0.469, and the dismissed-birthday row at 0.385. It does not improve missing catches. The label drop clears the confident rows by deleting `places`, and it false-abstains 5/8 keep-answerable rows and 3/4 manner-answerable rows, and it drops keep-missing catches from 8/8 to 0/8.

`voice` as a phone cue looks clean on this bank and is not. On the abstain dev it loses two catches the shipped path holds: "Set the birthday cake aside and say how a voice lands on Imogen." and "地点清单一边去，怎么用声音找着高远？". Both are dismissed cues. Adding `声音` makes a second cue, the unique-cue drop does not fire, and the stored dismissed relation wins the margin. Forcing phone false-abstains all 4 manner answerable rows. Dropping dismissed cues after adding `voice` false-abstains William and Margaret on the frozen set (the #26 hits) and loses the Patricia and 周涛 catches.

## Frozen and abstain-dev deltas

Same harsh stores as `question_adversarial`. A row counts only when the person match succeeds, so typo `absent_person` rows stay out of the relation tally. Probe and surface questions contain none of the record-compound pattern (`locations` plus folder/ledger/slip/chit/dossier/binder/folio/logbook, or `地点`/`地方` plus 卷宗/档案/簿册/案卷/档册/卷夹/册页/清册).

| rule | frozen new false abstain | frozen lost catch | frozen fixed false abstain | abstain-dev lost catch | surface-dev |
|---|---:|---:|---|---:|---|
| agreeing gap under 0.10 | 0 | 0 | Mary Jane | 郑凯/胡斌, gap 0.079 | unchanged |
| agreeing gap under 0.15 | 0 | 黄磊 birthday, gap 0.145 | Mary Jane | 郑凯/胡斌 | loses Keiko, gap 0.124 |
| compound margin, no `file` | 0 | 0 | none | 0 | unchanged |
| compound margin plus `file` / `名单` | 0 | 0 | Mary Jane | 0 | unchanged |
| places-label drop | 0 | 0 | Mary Jane, and Lorraine on the abstain dev | 0 | unchanged on the surface curve; destroys keep/manner on the new dev |
| `voice` as a phone cue | 0 | 0 | Mary Jane | Imogen, 高远 | unchanged |
| `voice` and drop dismissed cues | William, Margaret | Patricia, 周涛 | Mary Jane | Judith false abstain, 吴迪 false abstain, two lost catches | unchanged |
| set-aside margin 0.05 | William | 0 | none; 赵敏 and 刘洋 would be caught | one new catch | unchanged |
| global margin 0.05 | William | 0 | none | three new catches | Maren false abstain, missing catch 188→191 en and 189→192 zh |

The compound pattern matches, and the decision does not move, on the cue-dev polysemy rows "Eudora's voice path sits in a locations file, baldachin." (places gap 0.155) and "哪一份地点档案能用声音找到闵舟，夔鼽。" (phone gap 0.331, the ridge already overruled the cue) and on the bare-dev guarded row "晏书的地点清册在柜侧，位置先搁着，问落脚次序，鼲鼶鼛。" (places gap 0.374). Dropping the places label on Eudora false-abstains a places-only store and loses the missing-places catch. `file` is what makes the pattern hit Mary Jane. The dev bank has no `file`. Adding it to catch that one frozen sentence is the frozen template.

Ann stays an abstain under the `file` variant: after the cue is dropped, phone still leads by 0.187 and phone is absent. Lorraine stays a false abstain under the compound margin: places leads by 0.284, which clears 0.1, and places is absent. The same pattern's missing twin, Marcus, is places gap 0.019 with places stored, so the margin already answers. The two gaps are the same template.

## Ask-clause embeddings

Re-embedding the clause after the dismiss mark, or the words after "locations file", separates some of the twins. The ask path does not load the encoder. It reads `question_surface.npz` only. These tails are an oracle.

| tail | ridge | gap | decision on the harsh store |
|---|---|---:|---|
| the way to raise Mary Jane by voice | phone | 0.271 | would answer |
| the way to raise Ann by voice | phone | 0.425 | would still abstain |
| say how to get through to Catherine. | places | 0.229 | would still answer |
| say how to get through to William. | phone tied with places | 0.000 | would still answer |
| 赵敏常去的几处按先后是什么？ | places | 0.557 | would abstain |
| 陈静常去的几处按先后是什么？ | places | 0.652 | would still answer |
| 怎么用声音找到刘洋？ | phone | 0.540 | would abstain |
| 怎么用声音找到张强？ | phone | 0.635 | would still answer |

Catherine's tail is still places by 0.229 (phone +0.273, birthday +0.227, places +0.501). "Get through" is not a cue, and the ridge does not read the isolated clause as phone. A tail embedding would fix 赵敏, 刘洋, and Mary Jane, and it would leave Catherine. It is not a change the runtime can make.

On the new dev, the same tail embedding fixes one missed abstain ("生日册先放一边，邹岚的落脚顺序怎么念") and leaves the Godfrey birthday-supper false abstain (birthday still leads by 0.385 after the drop).

## Frozen adversarial, before and after

No row moved. Ask hits are the shipped ranker on the frozen sentence cache. False abstain is an answerable question that returned no fact. On an unanswerable row, "still answered" is the missed abstain.

| slice | lang | label | n | abstained | still answered | ask hits | false abstain |
|---|---|---|---:|---:|---:|---:|---:|
| bare_cue | en | answerable | 4 | 0 | 4 | 4 | 0 |
| bare_cue | en | missing_rel | 4 | 3 | 1 | 0 | 0 |
| bare_cue | zh | answerable | 4 | 0 | 4 | 4 | 0 |
| bare_cue | zh | missing_rel | 4 | 4 | 0 | 0 | 0 |
| compact_form | en | answerable | 7 | 2 | 5 | 5 | 2 |
| compact_form | en | missing_rel | 7 | 5 | 2 | 0 | 0 |
| compact_form | zh | answerable | 6 | 0 | 6 | 6 | 0 |
| compact_form | zh | missing_rel | 6 | 5 | 1 | 0 | 0 |
| deixis | en | answerable | 36 | 0 | 36 | 36 | 0 |
| deixis | zh | answerable | 36 | 0 | 36 | 27 | 0 |
| misleading_cue | en | answerable | 6 | 1 | 5 | 5 | 1 |
| misleading_cue | en | missing_rel | 6 | 5 | 1 | 0 | 0 |
| misleading_cue | zh | answerable | 6 | 0 | 6 | 6 | 0 |
| misleading_cue | zh | missing_rel | 6 | 4 | 2 | 0 | 0 |
| name_variant | en | answerable | 32 | 8 | 24 | 24 | 8 |
| name_variant | zh | answerable | 43 | 8 | 35 | 35 | 8 |
| near_dup_absent | en | absent_person | 16 | 16 | 0 | 0 | 0 |
| near_dup_absent | zh | absent_person | 14 | 14 | 0 | 0 | 0 |
| relation_paraphrase | en | answerable | 12 | 0 | 12 | 12 | 0 |
| relation_paraphrase | en | missing_rel | 12 | 10 | 2 | 0 | 0 |
| relation_paraphrase | zh | answerable | 12 | 0 | 12 | 12 | 0 |
| relation_paraphrase | zh | missing_rel | 12 | 12 | 0 | 0 | 0 |
| two_names | en | absent_person | 6 | 6 | 0 | 0 | 0 |
| two_names | en | answerable | 6 | 0 | 6 | 3 | 0 |
| two_names | en | missing_rel | 6 | 3 | 3 | 0 | 0 |
| two_names | zh | absent_person | 6 | 6 | 0 | 0 | 0 |
| two_names | zh | answerable | 6 | 0 | 6 | 3 | 0 |
| two_names | zh | missing_rel | 6 | 6 | 0 | 0 | 0 |

`misleading_cue` by the misleading relation: English places-answerable is the one false abstain (Mary Jane). English birthday-missing is the one English missed abstain (Catherine). Chinese phone-missing and places-missing are the two Chinese missed abstains (赵敏, 刘洋). Chinese answerable false abstain stays 0. `bare_cue` answerable false abstain stays 0/4 and 0/4. Deixis, including `second_indirect`, false abstain stays 0. Two-name answerable false abstain stays 0. Near-duplicate absent, including CJK superstring, stays fully abstained. Nickname subtypes stay at false abstain 0 (the name-variant false abstains are the typo rows, 8 and 8, already absent-person).

## Probe

`python -m tidal.question_heldout` on this main. No probe question changes, because the matcher did not change. Cache sha256 `bb994f4e85a3c06453d43af8c59eb3e6b352035bef14f2f208e5140b6b553f8b`.

| category | n | jaccard | cosine ret@1 | ask ret@1 | ask only | cosine only | phone cos·ask | date cos·ask | tokens cos·ask | stale cos·ask | false cos·ask | gold lost @0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en_direct | 36 | 0.12 | 12/36 | 36/36 | 24/36 | 0/36 | 2/12 · 12/12 | 4/12 · 12/12 | 6/12 · 12/12 | — | — | 0/36 |
| zh_direct | 36 | 0.28 | 9/36 | 36/36 | 27/36 | 0/36 | 4/12 · 12/12 | 2/12 · 12/12 | 3/12 · 12/12 | — | — | 0/36 |
| en_indirect | 40 | 0.10 | 14/40 | 39/40 | 25/40 | 0/40 | 3/13 · 13/13 | 7/14 · 13/14 | 4/13 · 13/13 | — | — | 0/40 |
| zh_indirect | 40 | 0.19 | 10/40 | 38/40 | 28/40 | 0/40 | 3/13 · 12/13 | 4/14 · 14/14 | 3/13 · 12/13 | — | — | 0/40 |
| en_cross_turn | 37 | 0.17 | 8/37 | 37/37 | 29/37 | 0/37 | 3/13 · 13/13 | 4/12 · 12/12 | 1/12 · 12/12 | — | — | 0/37 |
| zh_cross_turn | 37 | 0.22 | 6/37 | 36/37 | 30/37 | 0/37 | 2/13 · 12/13 | 1/12 · 12/12 | 3/12 · 12/12 | — | — | 0/37 |
| en_after_distractors | 36 | 0.12 | 5/36 | 35/36 | 30/36 | 0/36 | 0/12 · 12/12 | 1/12 · 11/12 | 4/12 · 12/12 | — | — | 0/36 |
| zh_after_distractors | 36 | 0.24 | 6/36 | 36/36 | 30/36 | 0/36 | 0/12 · 12/12 | 2/12 · 12/12 | 4/12 · 12/12 | — | — | 0/36 |
| en_after_update | 36 | 0.17 | 12/36 | 36/36 | 24/36 | 0/36 | 6/12 · 12/12 | 2/12 · 12/12 | 4/12 · 12/12 | 0/36 · 0/36 | — | 0/36 |
| zh_after_update | 36 | 0.29 | 2/36 | 36/36 | 34/36 | 0/36 | 0/12 · 12/12 | 2/12 · 12/12 | 0/12 · 12/12 | 0/36 · 0/36 | — | 0/36 |
| en_unanswerable | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 0/36 | — |
| zh_unanswerable | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 0/36 | — |
| answerable | 370 | 0.19 | 84/370 | 365/370 | 281/370 | 0/370 | 23/124 · 122/124 | 29/124 · 122/124 | 32/122 · 121/122 | 0/72 · 0/72 | — | 0/370 |
| unanswerable | 72 | — | — | — | — | — | — | — | — | — | 72/72 · 0/72 | — |
| all | 442 | 0.19 | 84/370 | 365/370 | 281/370 | 0/370 | 23/124 · 122/124 | 29/124 · 122/124 | 32/122 · 121/122 | 0/72 · 0/72 | 72/72 · 0/72 | 0/370 |

| slice | n | abstained | still answered | ask hits | false abstain |
|---|---:|---:|---:|---:|---:|
| missing_rel | 36 | 36 | 0 | 0 | 0 |
| absent_person | 36 | 36 | 0 | 0 | 0 |
| unanswerable | 72 | 72 | 0 | 0 | 0 |
| en/answerable | 185 | 0 | 185 | 183 | 0 |
| zh/answerable | 185 | 0 | 185 | 182 | 0 |
| answerable | 370 | 0 | 370 | 365 | 0 |

| check | result |
|---|---|
| answerable ask hits | 365/370 |
| answerable false abstain | 0/370 |
| unanswerable abstained | 72/72 |
| gold lost at overwrite 0.9 | 0/370 |
| co-resident pairs with cosine above 0.9 | 0 (max 0.8927, 8390 pairs) |
| surface dev false abstain at 0.1 | 0/192 en, 0/192 zh |
| surface dev missing at 0.1 | 188/192 en, 189/192 zh |
| `LLMMemory` sources | untouched |

## What this leaves in place

The four frozen rows stay wrong, for three different reasons that share no cutoff:

- Mary Jane is a live places cue the ridge agrees with, gap 0.027. The same template on Ann disagrees by 0.187. Lorraine, the abstain-dev twin, agrees by 0.284. An agree-bar that fixes Mary Jane loses a real two-name catch at 0.079.
- Catherine is a dismissed cue whose replacement label is places, lead 0.244, and places is stored. The tail embedding is still places. William, the answerable twin, sits at 0.083.
- 赵敏 and 刘洋 are dismissed cues whose ridge already names the missing relation, at 0.070 and 0.089. Those two numbers bracket William.

A lower global margin false-abstains Maren. A lower set-aside margin false-abstains William. Adding `voice` / `声音` as a phone cue loses abstain-dev catches or, once dismissed cues are dropped, re-breaks the William and Margaret hits from #26. The isolated Catherine clause is still places by 0.229, so "get through" was left off the cue list: it is the frozen wording, and the dev novoice rows use that same wording. A record-compound margin that avoids the frozen word `file` moves none of the four. Adding `file` fixes only Mary Jane and is the frozen sentence. The places-label drop fixes confident container rows by breaking keep and manner asks.

Pytest: 274 passed, 14 warnings. That includes this split, the held-out probe, the abstain curve, the bare-locative split, the cue-aside split, indirect hearer, nickname, and CJK. The tiny-llama test runs in this environment because `transformers` is installed. It was not edited.
