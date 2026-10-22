# Chinese superstring measurement

The matcher is unchanged. `tidal/question_name_align.py` and `tidal/question_surface.py` are the #19 head. This note measures the Chinese superstring failures and the rules that could separate them from a verb glued on with `的` / `过` / `了`. None of those rules can be chosen on the abstain dev bank.

Dev question sha256 `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, n=327. Frozen adversarial sha256 `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Literal collisions are 0. Token Jaccard, recomputed with the probe tokenizer: test vs probe 0.455, test vs surface 0.417, test vs dev 0.500, dev vs probe 0.400, dev vs surface 0.500. Sentence cosine was not re-embedded. The question strings are the same banks as #19, so the #19 maxima still apply (test vs probe 0.939, test vs dev 0.956). Ranking `W` / `alpha` live in `tidal/question_surface.npz` sha256 `820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`. `LLMMemory` is untouched.

Abstain below is `surface_abstain` on the shipped caches (`question_adversarial_emb.npz`, `question_heldout_emb.npz`) with the shipped name head. The ranker was not re-decoded. Ask hits are the #19 figures: the abstain decision matches that run, and the ranker bundle is the same file.

## How a two-character name stays inside a longer string

`_cjk_span_ok` keeps a stored name when the next character is a particle (`的了着过吗呢吧啊时`), or when two or more CJK characters follow (`孙悦添岁`, `姜北对应`, `刘洋先生`). It rejects a prefix `老` / `小` / `大` / `阿`, and it rejects exactly one extra CJK character closed by punctuation or the end of the string (`孙悦然，`).

The four failures are the third shape. `名下是王伟强的话` has one content character (`强`) and then `的`, and `的` is CJK, so the “another character follows” branch keeps `王伟`. The same branch keeps the probe’s verb glue (`唐宁选的`, `祁安报过`).

The head does not clear that lexical hit. A span that still contains the stored name in order (`王伟强`) is `_covers_stored`, so it is not an unmatched person, and `force_absent` does not fire.

## Frozen test, the eight superstrings

Template `名下是{who}的话，{cue}存了没有？`. Label `absent_person`.

| focus | stored name that matches | abstain |
|---|---|---|
| 王伟强 | 王伟 | answered |
| 李娜儿 | 李娜 | answered |
| 张强生 | 张强 | answered |
| 刘洋子 | 刘洋 | answered |
| 老王伟 | none | absent_person |
| 小李娜 | none | absent_person |
| 大张强 | none | absent_person |
| 老赵敏 | none | absent_person |

4/8 abstain. The four that answer are the longer given names. The four prefixes already abstain.

## Near misses on the same banks

| cluster | split | n | abstain |
|---|---|---:|---:|
| `cjk_superstring` suffix, closed by `，` | dev | 4 | 4 |
| `cjk_superstring` prefix `老` / `小` / `大` | dev | 4 | 4 |
| `cjk_edit1` (`黄蕾`, `周滔`, `陈靖`) | test | 3 | 3 |
| `cjk_middle` (`王小伟`, `李小娜`, `刘小洋`) | test | 3 | 3 |
| English `near_dup_absent` (hyphen, space, extra letter, edit 1) | test | 16 | 16 |
| `王伟平时窝在哪一带？` | test, answerable idiom | 1 | answered, and it is a hit in the #19 decode |
| `李娜平时窝在哪一带？` | test, `missing_rel` idiom | 1 | `missing_rel` |

`平时` is the same orthographic shape as the four failures: one content character, then a particle (`时`). The label is the opposite. A rule that drops every one-character extension before a particle drops `王伟` here.

Dev superstrings are `值班簿若写的是{who}，`. The extra character is closed by `，`, which already abstains. Dev has no `的话`.

## The shape that is actually ambiguous

Stored name, then exactly one CJK character outside the particle list, then a particle.

| split | rows of this shape | extra characters |
|---|---:|---|
| abstain dev, questions and stored statements | 0 | — |
| probe, strings whose only name hit is this glue | 36 strings; 15 of them are asks | 报 说 列 提 认 换 给 改 存 要 留 选 |
| surface / cue corpus, named questions whose only hit is this glue | 46 | 报 认 存 留 要 |
| frozen test | 6 asks | 平 (keep), 强 儿 生 子 (the four failures) |

Dev verb attachments that should stay the stored person are two content characters (`添岁`, `老待`, `常歇`, `歇脚`, `女士`), not one character plus a particle. The dev bank therefore has both labels for “a longer string that contains the name”, and zero rows of the shape the test fails on.

The 15 probe asks whose only hit is the glue:

| question | if the glue is dropped |
|---|---|
| 苏晚说过又放下的是哪一天？ | false abstain |
| 把苏晚给过的地点读出来。 | false abstain |
| 苏晚说过又放下的站点是哪些？ | false abstain |
| 再说一遍何安提过又翻篇的那个号码。 | false abstain |
| 几轮之前沈予给过一个联系办法，是什么？ | false abstain |
| 几轮之前沈予给过一个生日，是哪一天？ | false abstain |
| 几轮之前沈予给过地点，是哪些？ | false abstain |
| 不要用别人的履历——沈予给过哪些地方？ | false abstain |
| 唐宁选的碰面点是哪里？ | false abstain |
| 许迟提过怎么打给自己，后来被别的话盖住了，是什么？ | false abstain |
| 许迟提过某一天我们就翻篇了，是哪一天？ | false abstain |
| 许迟提过一些站点我们就翻篇了，按顺序是哪些？ | false abstain |
| 提醒我赵一楠说过该拨的那一串。 | deictic (`我`); stays answered with no stored name |
| 其他记录不是我要的。唐宁存了哪些地方？ | deictic (`我`); stays answered with no stored name |
| 我们为顾清留了哪些位置？ | unanswerable; reason flips from `missing_rel` to `absent_person` |

`想找唐宁的话该往哪里打？` is a full name immediately followed by `的话`. The particle branch already keeps it. It is not this shape.

Eight surface dev-curve questions are the same glue and would become false abstains: `把姜北要的地点列出来，不要重排。` and the same frame for 罗屿, 蔡禾, 潘夏, 杜衡, 魏迟, 任潇, 卢卡. The #19 curve is false abstain 0/192 per language.

## Rules that were scored and not shipped

No threshold was fit. The dev bank has no positive and no negative row of the ambiguous shape, so a grid on that bank cannot see the contrast.

**Drop every one-character extension before a particle.** Dev abstain bank: 0 questions change. Dev name-slice score stays 126/143, guard 0. Frozen test: the four longer names abstain, and `王伟平时窝在哪一带？` false-abstains. `李娜平时窝` flips from `missing_rel` to `absent_person`. Probe: 12 new false abstains, plus two deictic asks ranked with the name removed. Surface dev curve: 8 new false abstains. This is the rule #18 and #19 declined. It is still declined.

**Drop the extension only when the next two characters are `的话`.** Frozen test: the four longer names abstain. `平时` stays. Probe glue stays, including `唐宁选的` and `想找唐宁的话`. Surface glue stays. Dev bank: 0 rows contain `的话` after an extra character, and 0 rows contain `的话` at all. The four negative examples are the frozen template `名下是{who}的话`. Dev writes the same longer-name idea as `值班簿若写的是孙悦然，`, which already abstains. Choosing `的话` as the closer would be a fit to that frozen template. It was not shipped.

**Block the dev superstring endings `然` / `子` / `生` when they are the single extra character.** Those three characters are the dev suffix names (`孙悦然`, `胡斌子`, `郭燕子`, `高远生`). On the frozen test, `子` and `生` would abstain (`刘洋子`, `张强生`). `王伟强` and `李娜儿` would still answer. `然` does not occur on the test. That is the overlap of the two name lists, and it leaves two of the four failures in place. It was not shipped.

A character class wide enough to cover `强` and `儿` as name characters, and narrow enough to spare `选` `报` `说` `提` `给` `存` `留` `要` `认` and `平`, is not determined by any dev row. The probe and the cue corpus are the only places those verb characters appear, and they are held out.

## Before and after

The matcher, the head weights, and the ranker bundle are the files at `0ca8e9f`. Before and after are the same run.

### Superstring slice

| | n | abstained | still answered |
|---|---:|---:|---:|
| before (`cjk_superstring`, zh, `absent_person`) | 8 | 4 | 4 |
| after | 8 | 4 | 4 |

The four that still answer are 王伟强, 李娜儿, 张强生, 刘洋子.

### Name slices the head is required to keep

| slice | n | abstained before = after | false abstain before = after |
|---|---:|---:|---:|
| zh `name_variant` nickname | 8 | 0 | 0 |
| zh `name_variant` `no_surname` | 6 | 0 | 0 |
| zh `name_variant` `honorific_short` | 4 | 0 | 0 |
| zh `name_variant` together (the #19 18/18) | 18 | 0 | 0 |
| en `near_dup_absent` | 16 | 16 | 0 |
| `two_names` `absent_person` (en + zh) | 12 | 12 | 0 |
| zh `cjk_edit1` + `cjk_middle` | 6 | 6 | 0 |

zh `name_variant` overall stays false abstain 8/43, all eight of them typos. en `name_variant` stays false abstain 13/32.

### Full frozen test, abstain

| | n | abstained | still answered | false abstain |
|---|---:|---:|---:|---:|
| answerable, before = after | 216 | 44 | 172 | 44 |
| `missing_rel`, before = after | 69 | 53 | 16 | 0 |
| `absent_person`, before = after | 42 | 38 | 4 | 0 |

The four `absent_person` rows that still answer are the four longer names above. #19 ask hits on this abstain pattern were 154/216 answerable. This run did not re-decode them.

Non-name cells are the same abstain counts as #19: `relation_paraphrase` answerable 0/24 false abstain, `compact_form` false abstain 2/7 en and 0/6 zh, `bare_cue` 1/4 per language, `misleading_cue` 3/6 en and 4/6 zh, `deixis` 6/36 per language.

### Probe

| check | this run | #19 ranker figure, bundle unchanged |
|---|---|---|
| answerable false abstain | 0/370 | |
| unanswerable abstained | 72/72 | |
| answerable ask ret@1 | not re-decoded | 360/370 |
| gold lost at overwrite 0.9 | not re-decoded | 0/370 |
| co-resident pairs with cosine > 0.9 | not re-decoded | 0 (max 0.8927) |

Dev name-slice score of the shipped head is 126/143, guard 0 on the 184 non-name dev questions. Baseline string matcher on that slice is 78/143.

`tests/test_question_name_align.py` and `tests/test_question_adversarial.py`: 10 passed. `tests/test_question_head.py` was not collected here; this environment has no torch. Those tests do not import the matcher change, because there is no matcher change.
