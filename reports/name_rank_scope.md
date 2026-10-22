# Rank the matched person's facts

`Under Maggie, which birthday is on file?` matches only Margaret. The alias list is what does that (`Maggie` → Margaret). Before this change, `ask` still returned Mary Jane's birthday `1940-05-05`. Margaret's is `1939-04-04`. The same gap was the three Chinese pinyin rows: the person matched, and the ranker still returned another stored fact of the same relation.

The ranker is sentence cosine plus the relation logit (`alpha` 0.3). `query` called `surface_scores` on every visible slot after `matched_people` had already succeeded. Maggie and Mary Jane share the sentence `The notebook under {name} marks the morning they grow a year older.`, so Mary Jane's row won. Confirmed on the frozen adversarial file before any edit.

Always on. No flag. No new threshold. `W` and `alpha` are the same surface bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`). The name-head bundle is unchanged (`question_name_align.npz`, dev sha `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, 126/143, guard 0, threshold 0.8, logit margin 4.0). `LLMMemory` is untouched (`tidal/llm_inject_smollm2_135m.pt` sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`).

Before is `23c6c5b`. After is the same two calls (`evaluate("test")` and a probe replay of `eval_heldout`'s ask path) with the scope below. The frozen adversarial question sha256 is `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`. It was not used to choose the rule.

## Rule

`matched_fact_rows` is the people `matched_people` returned, before the cue-nearest narrowing. `ask`, `ask_tokens`, `query`, and `rank_question_scores` rank only those rows. A two-name question keeps every matched person. A string that matches nobody (an absent person, or a deictic question) still ranks the whole store. `recall` stays raw cosine and does not abstain.

Out-of-scope slots stay in the score vector, one below the worst in-scope score. In-scope values are the `surface_scores` values, so `W` and `alpha` are not touched. The held-out wire check still sees one score per visible slot.

`selected_fact_rows` is that set after `_focus_names`. It is the set `missing_rel` already used. The exact-cosine gate (cosine ≥ 0.999) now reads only those sentences, so another person's copy of the question does not cancel `missing_rel`. Cue-nearest is not applied to the ranker. Applying it there turns a correct two-name answer into the wrong person: `Leave William aside. The birthday to read belongs with Mary Jane.` is a hit for Mary Jane (`1977-02-20`). William is nearer the cue, and ranking only him would return William's birthday (`1978-03-21`). Both names stay in the ranker. The cosine contest between them is unchanged.

## Before / after

### Maggie, pinyin, and the nickname asks

| check | before | after |
|---|---:|---:|
| `Under Maggie, which birthday is on file?` | Mary Jane `1940-05-05` | Margaret `1939-04-04` |
| en nickname, person matched | 6/6 | 6/6 |
| en nickname, ask hits | 5/6 | 6/6 |
| en nickname, false abstain | 0/6 | 0/6 |
| pinyin, person matched | 8/8 | 8/8 |
| pinyin, ask hits | 5/8 | 8/8 |
| `Ms. Jane` → Mary Jane places | `north271-ridge271-ford271` | same |
| `Bob`, `Bill`, `Mr. Bob`, `Mr. Bill` | hits | hits |

The three pinyin misses were the same shape. Each matched the stored person and returned another person's fact of the same relation:

| question | before | after |
|---|---|---|
| 名册写着Wang Wei时，电话该回读什么？ | 李娜 `+1-917-555-1834` | 王伟 `+1-917-555-1833` |
| 名册写着Zhang Qiang时，地方该回读什么？ | 刘洋 `north842-ridge842-ford842` | 张强 `north841-ridge841-ford841` |
| 名册写着Chen Jing时，生日该回读什么？ | 赵敏 `1940-11-14` | 陈静 `1939-10-13` |

`Ms. Jane` is the substring head. `Jane` is not on the alias list. The frozen row matches only Mary Jane, and after the scope it still returns her places. A unit test stores a closer Margaret birthday sentence under `Under Ms. Jane, which birthday is on file?` and the ask still returns Mary Jane.

### Slices that had to stay

| check | before | after |
|---|---:|---:|
| near_dup absent, en | 16/16 | 16/16 |
| near_dup absent, zh | 14/14 | 14/14 |
| two_names absent | 12/12 | 12/12 |
| two_names en answerable | 3/6 | 3/6 |
| two_names zh answerable | 3/6 | 3/6 |
| en two-name missing relation, still answered | 3/6 | 3/6 |
| zh two-name missing relation, abstained | 6/6 | 6/6 |
| zh nickname + dropped surname + short honorific | 18/18 hits, 0 false abstain | 18/18 hits, 0 false abstain |
| cjk_superstring abstained | 8/8 | 8/8 |
| typos, en + zh | 16/16 false abstain | 16/16 false abstain |
| indirect person (`second_indirect`) | 12/12 false abstain | 12/12 false abstain |
| absent_person | 42/42 | 42/42 |

The English two-name rows did not move. `Leave William aside. The birthday to read belongs with Mary Jane.` stays a hit. The three English misses stay the other matched person's fact of the same relation (Mary Jane over William, Catherine over Ann).

One Chinese two-name miss changed its wrong answer and stayed a miss. `先放下刘洋。要回读的地方是周涛的。` matched 周涛 and 刘洋. Before, 张强's phone `+1-917-555-2013` won. 张强 is outside the matched set, so that row can no longer win. 刘洋's places `north1012-ridge1012-ford1012` now beats 周涛's `north1011-ridge1011-ford1011`. The gold is still 周涛. No new tie-break was added.

### Rows that were already wrong, and now name the matched person

These four were `missing_rel` rows the abstain rule did not catch. The asked person matched. Another person's fact won. After the scope the winner is a different relation of the matched person. Abstain counts do not move: `missing_rel` is still 53/69, and these four stay in the 16 that are still answered.

| question | before | after |
|---|---|---|
| When does Patricia mark another trip around the sun? | Mary Jane birthday `1936-05-23` | Patricia places `north75-ridge75-ford75` |
| 赵敏长一岁的标记该放在哪里？ | 黄磊 birthday `1976-05-02` | 赵敏 places `north675-ridge675-ford675` |
| 电话簿先合上，赵敏常去的几处按先后是什么？ | 黄磊 places `north732-ridge732-ford732` | 赵敏 phone `+1-917-555-1730` |
| 地方名单先放下，怎么用声音找到刘洋？ | 陈静 phone `+1-917-555-1753` | 刘洋 places `north752-ridge752-ford752` |

The exact-cosine gate, now limited to the cue-nearest person, did not change any abstain decision on this file. No adversarial pair sits at cosine ≥ 0.999.

### Full adversarial test

| check | before | after |
|---|---:|---:|
| answerable false abstain | 39/216 | 39/216 |
| answerable ask hits | 158/216 | 162/216 |
| answerable still answered | 177/216 | 177/216 |
| missing_rel abstained | 53/69 | 53/69 |
| absent_person abstained | 42/42 | 42/42 |
| `name_variant\|en\|answerable` abstained | 8 | 8 |
| `name_variant\|en\|answerable` ask hits | 23 | 24 |
| `name_variant\|en\|answerable` still answered | 24 | 24 |

Nine rows changed an answer. Four of those are the new hits (Maggie and the three pinyin rows). The other five are the four rows above and the 周涛 places miss.

### Probe

The unscoped ranker on this checkout is 360/370. The scoped ranker is 365/370. False abstain stays 0/370. Unanswerable abstain stays 72/72. Gold lost at overwrite 0.9 stays 0/370. Co-resident pairs with cosine > 0.9 stay 0 (8390 pairs, max 0.8927). Abstain margin 0.1, abstain ridge 1.0, relation alpha 0.3.

| category | before | after |
|---|---:|---:|
| zh_direct | 33/36 | 36/36 |
| en_after_distractors | 34/36 | 35/36 |
| zh_after_distractors | 35/36 | 36/36 |
| every other answerable category | unchanged | unchanged |
| answerable ask ret@1 | 360/370 | 365/370 |
| phone / birthday / places | 121/124, 122/124, 117/122 | 122/124, 122/124, 121/122 |
| en / zh | 182/185, 178/185 | 183/185, 182/185 |

The five new hits:

| category | question | before | after |
|---|---|---|---|
| en_after_distractors phone | I still need Mei's telephone after all of that. | Lena `+44-348-1048-4144` | Mei `+44-312-1012-4036` |
| zh_after_distractors places | 说了这么多，我还是要周宁的地点。 | 苏晚 `+44-492-1192-4576` | 周宁 `梨沟 葛岭 青冈` |
| zh_direct places | 按顺序说出陈可存下的地方。 | 赵一楠 `梨沟 榆关 青冈` | 陈可 `柿岸 榆关 青冈` |
| zh_direct places | 告诉我何安的地方。 | 江禾 `+44-540-1240-4720` | 何安 `柿岸 樟埠 青冈` |
| zh_direct places | 把许迟口述的地点给我。 | 林晓 `+44-444-1144-4432` | 许迟 `柿岸 杉径 青冈` |

One probe answer moved and stayed a miss. `Ignore the side topics and read Owen's birthday.` The gold is Owen's birthday token triple (`2 May 1948`). Before, Priya's birthday `1960-05-14` won. After, Owen's places triple (`cinder jasper vellum`) wins. Priya is outside the matched set. Inside Owen's facts the places row still outscores the birthday. That is the relation term, and `alpha` was not retuned.

## What stays open

One-character typos, 16/16 still false abstain. Indirect descriptions, 12/12 still false abstain. Misleading cues are unchanged. English two-name answerable asks are still 3/6, and the misses are the other matched person, which the ranker is supposed to keep. Chinese two-name answerable asks are still 3/6; the 周涛 places row now loses to 刘洋 rather than to 张强. Four unique-person `missing_rel` rows still return a fact, now the matched person's other relation. Owen's birthday still loses to his own places. None of those are a reason to change `W`.
