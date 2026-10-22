# English diminutives

`Bob` / `Robert`, `Bill` / `William`, and `Maggie` / `Margaret` share no prefix of length 3, so the name head abstains. `Liz` / `Elizabeth`, `Cathy` / `Catherine`, and `Patty` / `Patricia` already clear `prefix_ge3`. `Mr. Bob` and `Mr. Bill` miss with the nickname. `Ms. Jane` matches `Mary Jane` through the substring head. That hit stays. Jane is not an alias of Mary.

Always on. No flag. No new embedder. The list is not an 11th feature and the head is not refit. Ranking `W` and `alpha` (0.3) are the same surface bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`). The name-head bundle is unchanged (`5a254854c3ebd3c03ebdfc4054d8f92b3fe1512e3cf0a0db1c93b69502e42749`, dev sha `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, 126/143 name-slice points, guard 0, threshold 0.8, logit margin 4.0). `LLMMemory` is untouched (`tidal/llm_inject_smollm2_135m.pt` sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`).

The list is the filter output of the dev bank below. The frozen adversarial test (sha256 `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`) is scored after the list is fixed. It is not a row in the bank, and it is not used to add or drop a form.

## Dev bank

`tidal/question_nickname_dev_data.py`. The matcher does not import it. Question sha256 `a766f748cfc2bcc28f8c9b468c9822003798b3de377df15daa8a0633eb1aec98`, n=43.

The proposed dictionary is a short list of common full given names and their diminutives. The filter drops a diminutive shorter than 3 letters, a diminutive that names a person on another bank and is not its own anchor, or one edit of its own anchor. `ALIAS_GROUPS` is that filter's output. The unit test asserts equality. Question rosters are not people from the probe, the surface cue corpus, the abstain dev bank, the CJK superstring bank, or the frozen test. Anchors that are frozen-test people (`Robert`, `William`, `Margaret`, `Catherine`, `Elizabeth`, `Patricia`) stay in the dictionary and are checked as pairs, not as questions. The questions that choose the rule are the analogous pairs (`Ted` / `Edward`, `Dick` / `Richard`, `Dolly` / `Dorothy`, `Joe` / `Joseph`).

| kind | n | what it locks |
|---|---:|---|
| positive | 20 | diminutive → stored full name, with distractors |
| reverse | 1 | full name → stored diminutive |
| honorific | 2 | `Mr.` / `Ms.` plus the diminutive |
| possessive | 2 | `Becky's`, `Tony's` |
| prefix_kept | 3 | `Kate` / `Kathy` / `Katie` → `Katherine` (the head already matches) |
| negative | 6 | `Ed`, `Anne`, `Edwardo`, `Charls`, `Dick` against the wrong name, lowercase `ted` |
| ambiguous | 1 | `Ted` with both `Edward` and `Theodore` stored |
| compound | 3 | `Dolly` / `Dot` vs `Dorothy Quinn`, `Nell` vs `Helen Grace` |
| zh_keep | 3 | `小宁`, `宁宁`, `阿砚` already match on the head |
| zh_drop | 2 | `小侯` with two 侯 surnames, `老侯宁` |

Dropped before shipping:

| form | anchor | reason |
|---|---|---|
| greta | margaret | known person (surface roster) |
| ed | edward | shorter than 3 |
| anne | ann | edit-1 of ann |
| al | albert | shorter than 3 |
| nellie | helen | known person (abstain dev roster) |

Chinese given-name nicknames stay on the head (`小` / `阿` / `大` and reduplication). A surname diminutive is not on the list: `小侯` is ambiguous between 侯宁 and 侯柏, and `老侯宁` stays unmatched. No Chinese alias shipped.

`quinzaine` / `xeraphim` / `markNN` and `鹾` / `齾` are anchors so token Jaccard is not carried by shared function words. They are not people.

## Disjointness

Literal equality is 0 against the probe, the surface cue corpus, the abstain dev bank, the frozen test, and the CJK superstring dev bank. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased). The house bar used for the stress banks is 0.50. Sentence cosine is frozen `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix. No stored name from those banks is a lexical hit in these questions, and no question roster intersects those names.

| against | literal | max token Jaccard | closest Jaccard pair | max sentence cosine | closest cosine pair |
|---|---:|---:|---|---:|---|
| probe | 0 | 0.077 | 宁宁。鹾齾页 ‖ 周宁的手机号是哪一个？ | 0.904 | 宁宁。鹾齾页 ‖ 用唐宁的新电话，就是今天早上发过来的那部。 |
| surface | 0 | 0.071 | quinzaine xeraphim slip lists Ted beside mark01. ‖ The route card for Arthur lists which stops? | 0.897 | 宁宁。鹾齾页 ‖ 吸墨纸下的条子是姜北的站序，念出来。 |
| dev | 0 | 0.111 | 阿砚。鹾齾条 ‖ 值班簿写着阿斌时，生日该念哪一条？ | 0.886 | 老侯宁。鹾齾夹 ‖ 胡斌的蓝文件夹里夹着值班要念的声音联络。 |
| test | 0 | 0.118 | 小宁。鹾齾册 ‖ 名册写着小伟时，电话该回读什么？ | 0.897 | 小宁。鹾齾册 ‖ 名册写着小洋时，电话该回读什么？ |
| cjk | 0 | 0.105 | 宁宁。鹾齾页 ‖ 若册页写的是汪柘宁的话，先翻碘铪再开口。 | 0.913 | 宁宁。鹾齾页 ‖ 若册页写的是邵岚文的话，先翻钬镍再开口。 |

No pair exceeds Jaccard 0.50. Existing maxima in `reports/name_align.md` top out at Jaccard 0.500 and sentence cosine 0.956. The CJK bank's sentence cosine tops out at 0.921. This bank sits under both.

## Rule

`ALIAS_GROUPS` in `tidal/question_name_align.py`. Each item is `(full given names, diminutives)`. A surface form matches a stored form when one side is a full name and the other is a diminutive in the same item.

- The whole stored name has to be the member. A multi-token name folds to empty, so `Dolly` and `Dot` do not attach to `Dorothy Quinn`, and `Nell` does not attach to `Helen Grace`.
- A diminutive shared by two full names (`Ted` = `Edward` and `Theodore`) matches only when exactly one of those names is stored.
- A stored name that already is the surface form does not pull a sibling. `Bob` with `Bob` stored is the lexical hit.
- The head's decision stands when the probability clears 0.8 and the logit gap clears 4.0. The list is the fallback when that decision abstains.
- The surface form is title case, or an honorific (`Mr` / `Mrs` / `Ms` / `Miss` / `Dr`) plus title case, with an optional `'s`. A lowercase word (`ted`, `will`, `drew breath`) is not a name. A token shorter than three letters (`Ed`, `Al`) is not a name. Stopwords are rejected.
- Diminutive against diminutive does not match (`Teddy` vs stored `Ted`). Full name against full name does not match (`Catherine` vs `Katherine`).
- The head's extension and one-edit gates stay on the logistic path. They are not reapplied here: `William` contains the letters of `Liam` in order, and that pair is a real alias.

`Kate` is a prefix of `Katherine`, so the head wins. `Kate` → `Catherine` is the alias (`k` vs `c`). When both spellings are stored, `Kate` follows the head to `Katherine`.

| full given name | diminutives |
|---|---|
| Robert | Bob, Bobby |
| William | Bill, Billy, Liam |
| Margaret | Maggie, Meg, Peggy |
| Richard | Dick, Rick |
| James | Jim, Jimmy |
| John | Jack |
| Edward | Ted, Teddy, Ned |
| Theodore | Ted, Teddy |
| Charles | Chuck, Charlie |
| Anthony | Tony |
| Michael | Mike |
| Joseph | Joe |
| Catherine, Katherine | Kate, Katie, Kathy |
| Elizabeth | Beth, Betty |
| Rebecca | Becky |
| Abigail | Abby |
| Dorothy | Dolly, Dot |
| Virginia | Ginny |
| Barbara | Babs |
| Patricia | Tricia |
| Andrew | Drew |
| Susan | Susie |

Prefix nicknames the head already handles (`Liz`, `Cathy`, `Patty`, `Harry`, `Will`, `Rob`) are not repeated. A scan of every stored name in the probe, the surface corpus, the abstain dev bank, and the CJK bank found zero strings where the list fires and the head abstains. On the frozen test the same scan fires on exactly five spans: `Bob` → Robert, `Bill` → William, `Maggie` → Margaret, `Mr. Bob` → Robert, `Mr. Bill` → William.

## Before / after

Before is `c3fe5fa` (`reports/name_superstring.md`): `evaluate("test")` and `eval_heldout` on the same caches. After is the same two calls with the list above. The only adversarial cell that moves is `name_variant|en|answerable`.

### Nickname and short honorific

| check | before | after |
|---|---:|---:|
| en nickname, false abstain | 3/6 | 0/6 |
| en nickname, ask hits | 3/6 | 5/6 |
| en nickname, person matched | 3/6 (`Liz`, `Cathy`, `Patty`) | 6/6 |
| en honorific_short, false abstain | 2/3 | 0/3 |
| en honorific_short, ask hits | 1/3 (`Ms. Jane`) | 3/3 |
| `Ms. Jane` → Mary Jane | head substring | head substring (unchanged) |
| `Jane` on the alias list | absent | absent |

`Under Maggie, which birthday is on file?` matches only Margaret. Ask returns Mary Jane's birthday `1940-05-05`; the gold is Margaret's `1939-04-04`. Both facts use `The notebook under {name} marks the morning they grow a year older.` `query` ranks every fact after abstain is `None`, and the two sentences differ by the name. This is the same gap as the pinyin misses (ask hits 5/8, person match 8/8). Ranking `W` and `alpha` stay. `Maggie` does not match Mary Jane: with both stored, the list picks Margaret, and the head's two-letter prefix tie stays under the threshold.

The four new ask hits are `Bob`, `Bill`, `Mr. Bob`, and `Mr. Bill`. Maggie is the extra answer that is not a hit. False abstain on that row is 0.

### Slices that had to stay

| check | before | after |
|---|---:|---:|
| near_dup absent, en | 16/16 | 16/16 |
| two_names absent | 12/12 | 12/12 |
| zh nickname + dropped surname + short honorific | 18/18 hits, 0 false abstain | 18/18 hits, 0 false abstain |
| cjk_superstring abstained | 8/8 | 8/8 |
| typos, en + zh | 16/16 false abstain | 16/16 false abstain |
| near_dup absent, zh | 14/14 | 14/14 |
| pinyin ask hits | 5/8 | 5/8 |
| indirect person false abstain | 12/12 | 12/12 |
| en two-name missing relation, still answered | 3/6 | 3/6 |

### Full adversarial test

| check | before | after |
|---|---:|---:|
| answerable false abstain | 44/216 | 39/216 |
| answerable ask hits | 154/216 | 158/216 |
| missing_rel abstained | 53/69 | 53/69 |
| absent_person abstained | 42/42 | 42/42 |
| `name_variant\|en\|answerable` abstained | 13 | 8 |
| `name_variant\|en\|answerable` ask hits | 19 | 23 |
| `name_variant\|en\|answerable` still answered | 19 | 24 |

### Probe

| check | before | after |
|---|---:|---:|
| answerable ask ret@1 | 360/370 | 360/370 |
| unanswerable abstained | 72/72 | 72/72 |
| false abstain | 0/370 | 0/370 |
| gold lost at overwrite 0.9 | 0/370 | 0/370 |
| co-resident pairs with cosine > 0.9 | 0 (max 0.8927) | 0 (max 0.8927) |
| abstain margin / ridge | 0.1 / 1.0 | 0.1 / 1.0 |
| relation alpha | 0.3 | 0.3 |

The abstain-dev name score stays 126/143, guard 0. The surface dev curve at margin 0.1 is unchanged: false abstain 0/192 per language, missing 188/192 en and 189/192 zh, absent 36/36, deictic false abstain 0/6. Pytest is green (242 passed), including the held-out probe, the question head, the nickname dev split, and `LLMMemory`.

## What stays open

One-character typos, 16/16 still false abstain. Indirect descriptions (`addressee`, `跟前这位`, `对面那位`), 12/12 still false abstain. Misleading cues are unchanged. Pinyin still matches the person and still returns the other person's fact on 3/8 asks. `Maggie` now matches Margaret and still loses the ask to Mary Jane's birthday for the same ranker reason. English two-name missing relation is still answered on 3/6. None of those are a reason to change `W`.
