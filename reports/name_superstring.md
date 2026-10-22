# CJK superstring versus verb attachment

A stored two-character name plus one more syllable, then a particle, was one branch in `_cjk_span_ok`. `的` is a CJK character, so `王伟强的话` and `唐宁选的` took the same path, and the name head kept the lexical hit. Prefixes `老` / `小` / `大` already abstained.

Always on. No flag. No new embedder. Ranking `W` and `alpha` (0.3) are the same surface bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`). The name-head bundle is unchanged (`5a254854c3ebd3c03ebdfc4054d8f92b3fe1512e3cf0a0db1c93b69502e42749`, dev sha `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`, 126/143 name-slice points, guard 0). `LLMMemory` is untouched (`tidal/llm_inject_smollm2_135m.pt` sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`).

The rule is fit on the dev bank below. The frozen adversarial test (sha256 `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`) is scored after the rule is fixed. It is not a row in the bank, and it is not used to choose a character. The abstain dev bank has zero rows of the shape "stored name + one content character + particle".

## Dev bank

`tidal/question_cjk_dev_data.py`. The matcher does not import it. Question sha256 `5e8b86f694d8a26a3505230fd3fd8f1e80e5261751026c57203dcf2e5ad508ee`, n=201.

Roster: 孟峤, 邵岚, 姚岑, 汪柘, 戴嵊, 夏芮, 田栖, 董馥. Two mark characters from a list absent from the other banks are appended to every question.

| class | label | n |
|---|---|---:|
| longer_name | drop | 81 |
| verb | keep | 68 |
| near_miss | keep | 52 |

Closers. `的话` is one closer. A compound whose second character is not a particle (`列出的`) is reported as `出…`.

| class | closers |
|---|---|
| longer_name | 的 12, 的话 12, 呢 13, 吧 12, 吗 13, 啊 11, 时 8 |
| verb | 的 23, 过 23, 了 12, 呢 1, 吧 1, 吗 1, 啊 1, 出… 3, 下… 2, 过… 1 |
| near_miss | 过 10, 了 11, 着 4, 时 15, 的 11, 呢 1 |

Longer names are a name-final syllable (然, 子, 儿, 伟, 芳, 杰, 宇, 婷, 博, 轩, 玲, 娜, 洋, 涛, 凯, 燕, 斌, 迪, 悦, 宁, 峰, 磊, 敏, 静, 丽, 文, 华, 军, 安, and 强, 生, 平) plus a nominal closer. Verbs are the probe glue (选, 报, 说, 列, 提, 认, 换, 给, 改, 存, 要, 留) and further one-character predicates, plus compounds (`列出的`, `写下的`, `说出的`, `找出的`, `存下的`, `给过的`). Near-misses are the same name-final syllables used as verbs or time words: aspect on 生 / 强 (`生过`, `生了`, `生着`, `强过`, `强了`, `强着`), the time words 平时 / 当时 / 临时 / 有时 / 准时 / 这时 / 那时, and name-like predicates that stay verbs before 的 (`定`, `立`, `成`, `明`, `记`, `念`, `开`, `正`, `清`, `发`, `通`). 田栖 has both `田栖平的` (drop) and `田栖平时` (keep).

`名下是` occurs in 0 questions. Every row separates: `name_in_text` matches the label, and `matched_people` on the roster is `[name]` on a keep and `[]` on a drop. The head does not put the short name back.

## Disjointness

Literal equality is 0 against the probe, the surface cue corpus, the abstain dev bank, and the frozen test. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased). The house bar used for the stress banks is 0.50. Sentence cosine is frozen `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix. No stored name from the probe, the surface roster, or either abstain bank is a lexical hit in these questions.

| against | literal | max token Jaccard | closest Jaccard pair | max sentence cosine | closest cosine pair |
|---|---:|---:|---|---:|---|
| probe | 0 | 0.300 | 把夏芮给过的站序从藤铌侧取出来。 ‖ 把苏晚给过的地点读出来。 | 0.921 | 汪柘说过一次呼号，夹在氚钡页下。 ‖ 还没跑题的时候，赵一楠说过一次电话号码。 |
| surface | 0 | 0.318 | 把夏芮念的站序从铱砂侧取出来。 ‖ 吸墨纸下的条子是潘夏的站序，念出来。 | 0.914 | 门口条子写孟峤报吧，参见釉锗。 ‖ 吸墨纸下的条子是姜北的站序，念出来。 |
| dev | 0 | 0.273 | 把夏芮念的站序从铱砂侧取出来。 ‖ 把孙悦的地方从值班簿里念出来。 | 0.918 | 广播点到邵岚斌啊，值日去锶钬旁取那一串。 ‖ 先别看郑凯。值班要念的地方是胡斌的。 |
| test | 0 | 0.208 | 轮到姚岑生时，只查釉锗，别的先放下。 ‖ 先放下周涛。要回读的生日是刘洋的。 | 0.916 | 戴嵊清了呼号就封存，见硼藤。 ‖ 名下是周滔的话，电话存了没有？ |

Existing maxima in `reports/name_align.md` top out at Jaccard 0.500 and sentence cosine 0.956. This bank sits under both.

## Rule

One content syllable and then a particle is the ambiguous shape. The sets are the labels of this bank, and the unit test derives them again from those labels and checks equality. No pair `(extra, particle)` carries both labels.

- A time-word bigram keeps the stored name. The keep on `时` is 平时, 当时, 临时, 有时, 准时, 这时, 那时. `田栖平时` keeps 田栖. `田栖平的` drops it.
- A predicate keeps the stored name before any particle, including 的 / 过 / 了 / 呢 / 吧 / 吗 / 啊. The set is 写, 做, 列, 去, 发, 存, 定, 开, 念, 想, 成, 打, 找, 报, 拿, 换, 提, 改, 明, 来, 正, 清, 用, 留, 看, 立, 给, 要, 认, 记, 说, 选, 通, 问.
- 生 and 强 have both labels. Aspect (过 / 了 / 着) keeps the short name (`姚岑生过`, `孟峤强了`). A nominal closer (的 / 的话 / 呢 / 吧 / 吗 / 啊 / 时) is the longer name (`姚岑生的话`, `孟峤强呢`). `的话` is this case because the particle is `的`.
- Every other content syllable before a particle is the rest of a longer personal name (`王伟强的`, `李娜儿呢`, `刘洋子吗`) and the short name does not match.

Unchanged branches: a prefix 老 / 小 / 大 / 阿 rejects; the name at the end of the string, or before punctuation, matches; an immediate particle matches (`王伟的`, `想找唐宁的话`, `写着孙悦时`); one extra syllable closed by punctuation rejects (`孙悦然，`); a following character that is CJK and not a particle matches (`刘洋先生`, `孙悦添岁`, `姜北对应`, `唐宁列出的`).

The name head is the same weights. A lexical miss is `absent_person` when the question is not deictic, and the head does not restore the short name.

A relative-clause reading of `生的` ("the one X bore") has the longer-name label in this bank. Aspect on 生 / 强 is the verb row. The bank does not assign both labels to that pair.

## Before / after

Before is the name head at `0ca8e9f` (`reports/name_align.md`). After is `evaluate("test")` and `eval_heldout` on the same caches. A scan of every stored name in the probe, the surface corpus, the abstain dev bank, and the frozen test found four lexical flips, and they are the four longer names below. Every other slice cell is the head column from that report.

| check | before | after |
|---|---:|---:|
| cjk_superstring abstained | 4/8 | 8/8 |
| 王伟强, 李娜儿, 张强生, 刘洋子 | answered, matched the short name | `absent_person`, matched nobody |
| prefixes 老王伟, 小李娜, 大张强, 老赵敏 | `absent_person` | `absent_person` |
| 王伟平时窝在哪一带？ (answerable) | hit | hit |
| 李娜平时窝在哪一带？ (missing_rel) | `missing_rel` | `missing_rel` |
| near_dup absent, zh | 10/14 | 14/14 |
| near_dup absent, en | 16/16 | 16/16 |
| two_names absent | 12/12 | 12/12 |
| zh nickname + dropped surname + short honorific | 18/18 hits, 0 false abstain | 18/18 hits, 0 false abstain |
| answerable false abstain | 44/216 | 44/216 |
| answerable ask hits | 154/216 | 154/216 |
| missing_rel abstained | 53/69 | 53/69 |
| absent_person abstained | 38/42 | 42/42 |

The four new abstentions are unanswerable, so ask hits do not move. The +4 on absent-person is those four rows.

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

The surface dev curve at margin 0.1 is unchanged: false abstain 0/192 per language, missing 188/192 en and 189/192 zh, absent 36/36, deictic false abstain 0/6. Probe and surface strings of the shape "name + content syllable + particle" still match, including 选, 报, 要, 给, 留, 说. Pytest is green (234 passed), including the held-out probe, the question head, and `LLMMemory`.

## What stays open

One-character typos, English `Bob` / `Robert` aliases, indirect descriptions (`addressee`, `跟前这位`), and misleading cues are out of scope. Their counts are the name-head report: typos 16/16 still false abstain, indirect person 12/12 still false abstain, English two-name missing relation 3/6 still answered.
