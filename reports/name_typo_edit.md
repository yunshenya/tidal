# One-character typos stay abstained

The frozen adversarial file still false-abstains all 16 one-character typos (8 English, 8 Chinese). A store-context edit gate was measured against that file and against a new dev split that labels the same edit both ways. The gate accepts a typo only by also accepting an absent near-duplicate. The matcher is unchanged.

Re-measured on this checkout with shipped `ask` and the frozen sentence cache. Question sha256 `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`. The name head, the alias list, ranking `W`, and `alpha` (0.3) are the bundles from #23 (`question_name_align.npz` `5a254854c3ebd3c03ebdfc4054d8f92b3fe1512e3cf0a0db1c93b69502e42749`, surface `820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`). `LLMMemory` is untouched (`tidal/llm_inject_smollm2_135m.pt` sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`).

## Baseline

Every typo is one substitution. Exactly one stored name in that question's roster is at distance 1, and the span is not a longer copy of the name. The second stored name is 5 to 7 letters away in English and 2 characters away in Chinese. `ask` abstains `absent_person`. The lexical cue and the abstain ridge name the same relation on all 16, so the abstain is the name decision. The canonical person stores only that relation, and the value is the gold. A match of that one person would be an ask hit under the person-scoped ranker. The match was not added.

| surface | stored | edit | second | abstain |
|---|---|---|---:|---|
| Robart | Robert | a→e | 5 | absent_person |
| Elizaboth | Elizabeth | o→e | 7 | absent_person |
| Willian | William | n→m, adjacent keys | 6 | absent_person |
| Catharine | Catherine | a→e | 6 | absent_person |
| Margoret | Margaret | o→a | 5 | absent_person |
| Patricis | Patricia | s→a, adjacent keys | 7 | absent_person |
| Anm | Ann | m→n, adjacent keys | 6 | absent_person |
| Mary Jame | Mary Jane | m→n, adjacent keys | 6 | absent_person |
| 王玮 | 王伟 | 玮→伟, same pinyin | 2 | absent_person |
| 李那 | 李娜 | 那→娜, same pinyin | 2 | absent_person |
| 张疆 | 张强 | 疆→强, pinyin differs | 2 | absent_person |
| 刘样 | 刘洋 | 样→洋, same pinyin | 2 | absent_person |
| 陈靖 | 陈静 | 靖→静, same pinyin | 2 | absent_person |
| 赵悯 | 赵敏 | 悯→敏, same pinyin | 2 | absent_person |
| 黄蕾 | 黄磊 | 蕾→磊, same pinyin | 2 | absent_person |
| 周滔 | 周涛 | 滔→涛, same pinyin | 2 | absent_person |

English near-duplicates stay 16/16 abstained. Chinese near-duplicates stay 14/14. Two-name absent stays 12/12. Headline abstain, from shipped `ask` on all 327 frozen rows, is 39/216 false abstain, 53/69 missing-relation, 42/42 absent-person. Those are the #23 counts.

## Siblings that share the shape

**Same surface, both labels.** The roster on each side has a different distractor, and the distance profile does not. One stored name at distance 1, nothing else at distance 1, second neighbor at distance 2, same pinyin.

| surface | stored | typo question | absent question |
|---|---|---|---|
| 黄蕾 | 黄磊 | 名册写着黄蕾时，地方该回读什么？ | 名下是黄蕾的话，地方存了没有？ |
| 周滔 | 周涛 | 名册写着周滔时，电话该回读什么？ | 名下是周滔的话，电话存了没有？ |
| 陈靖 | 陈静 | 名册写着陈靖时，电话该回读什么？ | 名下是陈靖的话，生日存了没有？ |

The abstain dev bank (sha256 `26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a`) is the same construction. Both labels, one unique edit:

| surface | stored | edit |
|---|---|---|
| Harols | Harold | s↔d, adjacent keys. Typo second neighbor 6, absent second neighbor 3 (Marcus) |
| Juduth | Judith | i↔u, adjacent keys. Second neighbor 6 on both labels |
| Marcas | Marcus | u↔a. Typo second neighbor 4, absent second neighbor 6 |
| Anna Louize | Anna Louise | z↔s, adjacent keys. Second neighbor 9 and 8 |
| 吴笛 | 吴迪 | same pinyin, second neighbor 2 |
| 郑恺 | 郑凯 | same pinyin, second neighbor 2 |
| 徐锋 | 徐峰 | same pinyin, second neighbor 2 |

Harols is the only pair whose second neighbor moves (6 versus 3), and it moves because the absent row stored Marcus. Juduth stays at 6 on both rows. A cutoff fit to Harols does not travel.

**Same edit, different string.** These absent names are also one non-extension edit from exactly one stored name. English typo seconds are {5, 6, 7}. These seconds are {6, 7}. Any cutoff that accepts a typo at 5 accepts a near-duplicate at 6 and at 7.

| surface | stored | edit | second |
|---|---|---|---:|
| Robern | Robert | n→t | 7 |
| Williem | William | e→a | 7 |
| Catherina | Catherine | a→e | 7 |
| Mary Jone | Mary Jane | o→a | 6 |
| Patrici | Patricia | delete a | 7 |

`Catherina` is the same vowel pair as dev `Imogan` / `Imogen`, which the abstain dev labels only as a typo. Accepting the dev's typo-only pairs accepts this absent name. `Mary Jone` is the same o↔a pair as `Margoret`.

The other 11 English near-duplicates and the other 11 Chinese near-duplicates are a longer span that still contains the stored name (hyphen, extra word, extra letter, `老` / `小` / `大`, a syllable in the middle, a syllable on the end). Distance 1 and an extension are the same fact for `Roberto`, `Anne`, `Margarett`, `Roberrt`, and the Chinese superstrings. The shipped matcher already abstains on all 22. An edit rule that still rejects an extension leaves them abstained and does not touch the 16 typos.

## Gates, scored and not shipped

A gate returns the one stored name at distance 1 when the extra condition holds, and abstains otherwise. "Adjacent" is a QWERTY neighbor, diagonals included. `y` and `i` are two columns apart and are not neighbors. The denylist is the substitution pairs the abstain dev bank labels both ways: `d↔s`, `i↔u`, `u↔a`, `z↔s`, `笛↔迪`, `凯↔恺`, `峰↔锋`.

On a row the gate accepts, the abstain clears. The ridge agrees with the cue on every typo and every edit-1 near-duplicate (gap above 0.15 is not required, because the ridge names the cue). The person stores the asked relation. Person-scoped ranking then sees one fact, the gold, on a typo, and the near person's fact on a near-duplicate.

| gate | typo false abstain | en near abstain | zh near abstain | what it also accepts |
|---|---:|---:|---:|---|
| shipped, edit distance 1 is a non-match | 16/16 | 16/16 | 14/14 | nothing in this slice |
| unique edit-1, drop extensions | 0/16 | 11/16 | 11/14 | Robern, Williem, Catherina, Mary Jone, Patrici, 黄蕾, 周滔, 陈靖, and the seven dev surfaces above plus `Kennth` |
| unique substitution | 0/16 | 12/16 | 11/14 | the same set without `Patrici` and `Kennth` |
| unique substitution, second neighbor at least 3 | 8/16 English cleared, 8/8 Chinese still abstain | 12/16 | 14/14 | Chinese seconds are 2, so the margin drops every Chinese typo and still takes the four English near substitutions |
| adjacent-key substitution | 12/16 | 16/16 | 14/14 | Willian, Patricis, Anm, Mary Jame. Also dev Harols, Juduth, Anna Louize |
| adjacent-key, minus the dev's both-label pairs | 12/16 | 16/16 | 14/14 | the same four frozen typos. Dev Harols, Juduth, Anna Louize stay abstained. `m↔n` is not on that list |
| unique same-pinyin substitution | 9/16 (张疆 still abstains, English untouched) | 16/16 | 11/14 | 黄蕾, 周滔, 陈靖, and dev 吴笛, 郑恺, 徐锋 |

`m↔n` is Willian, Anm, and Mary Jame. It is also the new dev pair `Branwell` / `Bramwell`, labeled both ways. Dropping the pairs the old dev bank happened to use does not make the next adjacent pair a typo.

Two-name absent questions have no stored name at distance 1. That slice stays 12/12 under every gate above. The eight Chinese superstrings stay extensions. The probe has one question-level deletion: `me` against `Mei` in `Where did Ada tell me to show up?`. A gate that accepts deletions attaches Mei beside Ada. A substitution-only gate does not. Neither substitution gate separates the absent near-duplicates above.

## Dev split

`tidal/question_typo_dev_data.py`. The matcher does not import it. Question sha256 `68260c01d5f130c46285348f1ef9886f14e9aa2b47f5606c9357ecdc521f5f99`, n=68. Twenty rows are labeled match. Forty-eight are labeled abstain. The shipped matcher abstains on all 68.

Each collision is one surface and one roster, emitted as both labels. English closers are `slip`, `ledger`, and `chit`. Chinese closers are `时`, `的话`, `，`, and `呢`. The closer is not a feature of the name decision: the edit profile is one value per collision.

| kind | n | labels | pair |
|---|---:|---|---|
| collision_adj | 6 | both | Hollid / Hollis, s↔d, adjacent |
| collision_mn | 6 | both | Branwell / Bramwell, m↔n, adjacent |
| collision_vowel | 6 | both | Bramwall / Bramwell, e↔a |
| collision_del | 6 | both | Pembrok / Pembroke |
| collision_homo | 8 | both | 阮峥 / 阮铮, same pinyin |
| collision_diffpy | 8 | both | 庞琨 / 庞岚, pinyin differs |
| tie | 7 | abstain | Hollid with Hollis and Hollin both at distance 1; 阮峥 with 阮铮 and 阮崎 |
| hyphen, space, extension | 3 each | abstain | Hollis-Jane, Hollis Jane, Bramwello |
| prefix, middle, superstring | 4 each | abstain | 老阮铮, 阮小铮, 阮铮然 |

The six gates above each miss at least one collision row. The tie and the longer spans are the cases a uniqueness-plus-extension rule does separate, and the shipped matcher already abstains on them. They are not the 16 frozen typos.

### Disjointness

Literal collisions are 0. Token Jaccard is the probe tokenizer (CJK per character, Latin words lowercased). The house bar used for these banks is 0.50. Sentence cosine is frozen `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix.

| against | literal | max token Jaccard | closest Jaccard pair | max sentence cosine | closest cosine pair |
|---|---:|---:|---|---:|---|
| probe | 0 | 0.250 | The brumaire chit for Hollid is xylograph q03. ‖ What is the new telephone for Hana? | 0.890 | 阮铮然呢，鼷页先翻鼢，参见鼩鼹q68。 ‖ 再说一遍何安提过又翻篇的那个号码。 |
| surface | 0 | 0.200 | The brumaire chit for Hollid is xylograph q03. ‖ The chit under the blotter is Ingrid's stand order. Read it. | 0.899 | 阮铮然呢，鼷页先翻鼢，参见鼩鼹q68。 ‖ 罗屿按口头顺序报了站点，还不许重排。 |
| abstain dev | 0 | 0.214 | 若鼷册是阮小铮的话，鼢页没抄，参见鼩鼹q62。 ‖ 值班簿若写的是马小丽，电话记过没有？ | 0.890 | 老阮铮呢，鼷页先翻鼢，参见鼩鼹q60。 ‖ 声音要找到郑凯，位置表先别翻，看哪一页？ |
| frozen test | 0 | 0.214 | The brumaire chit for Hollid is xylograph q03. ‖ Your telephone, the one kept for you, is which? | 0.893 | 阮小铮，鼷鼢页仍空，参见鼩鼹q63。 ‖ 名下是王小伟的话，地方存了没有？ |
| cjk dev `5e8b86f…` | 0 | 0.250 | 若鼷册是阮铮然的话，鼢页没抄，参见鼩鼹q66。 ‖ 若册页写的是邵岚然的话，先翻硼藤再开口。 | 0.923 | 若鼷册是阮铮然的话，鼢页没抄，参见鼩鼹q66。 ‖ 若册页写的是邵岚强的话，先翻铱砂再开口。 |
| nickname dev `a766f748…` | 0 | 0.273 | brumaire xylograph slip lists Hollid beside q01. ‖ quinzaine xeraphim slip lists Ted beside mark01. | 0.923 | brumaire xylograph slip lists Hollid beside q04. ‖ quinzaine xeraphim slip lists Edward beside mark04. |

No stored name from those banks is a lexical hit in these questions, and no question roster intersects those names.

## Before / after

The matcher, the ranker, `W`, and `alpha` are the #23 code. Before and after are the same measurement. The after column was scored again on this checkout: shipped `ask` on the frozen file, and `eval_heldout` on the probe.

| check | before | after |
|---|---:|---:|
| typos, en + zh, false abstain | 16/16 | 16/16 |
| near_dup absent, en | 16/16 | 16/16 |
| near_dup absent, zh | 14/14 | 14/14 |
| two_names absent | 12/12 | 12/12 |
| cjk_superstring abstained | 8/8 | 8/8 |
| answerable false abstain | 39/216 | 39/216 |
| missing_rel abstained | 53/69 | 53/69 |
| absent_person abstained | 42/42 | 42/42 |
| en nickname ask, Maggie birthday | Margaret `1939-04-04` | same |
| pinyin ask hits | 8/8 | 8/8 |
| probe answerable ask ret@1 | 365/370 | 365/370 |
| probe unanswerable abstained | 72/72 | 72/72 |
| probe false abstain | 0/370 | 0/370 |
| gold lost at overwrite 0.9 | 0/370 | 0/370 |
| co-resident cosine > 0.9 | 0 (max 0.8927) | 0 (max 0.8927) |

Maggie returns Margaret `1939-04-04`. The eight pinyin asks hit. `tidal/question_name_align.py`, `tidal/question_surface.py`, both npz bundles, and `tidal/llm_memory.py` are the #23 bytes. A deletion gate is the one change that would touch a probe question (`me` against `Mei`), and it was not shipped.

`python3 -m pytest` on this revision: 250 passed. The held-out probe invariants in that suite are green: gold lost at overwrite 0.9 is 0, and no co-resident pair has cosine above 0.9 (max 0.8927).

## What stays open

One-character typos, 16/16 still false abstain. The edit that would clear them is the edit that answers Robern, Williem, Catherina, Mary Jone, Patrici, 黄蕾, 周滔, and 陈靖. Indirect descriptions, 12/12 still false abstain. Misleading cues are unchanged. English two-name answerable asks are still 3/6. None of those are a reason to change `W`.
