# Real multi-turn dialogue

Measurement of the shipped ask path on natural multi-turn episodes. No threshold, window, or name-head weight was fit. `TURN_WINDOW` stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input.

Episodes 104. Scored asks 142.
Questions `117ef589bb02bd883db44a4d2d5859b17c271c00a5df324574c8095aced5b704`.
Bank `0be42845836ec2a19c961d5a415ccd0d9d0dcb1f8adf5bb632335cd26871cec5` (1032 strings).

## Result

State rule 142/142. Empty rule 142/142. False accepts 0.
Gated rows, where the rule and the listener already agree: 106/106 with the turns, 106/106 on the empty log, human 106/106. Ordinary talk is 52/52.
The listener's reading on the whole bank is 106/142. Experience gaps are 0/21 against the listener and 21/21 against the rule. Open addressee forms are 0/8 against the listener. Empty-log controls are 0/4 against the listener. Name-head observations are 0/3 against the listener. A miss in those four columns is not a regression.

Rule labels are the replay of `DialogueState`, `ask_people`, and the deixis row selectors. Human labels are the conversational reading written with the episode. State mode walks `note_turn` and `query` in order. Empty mode asks each question on a fresh store of the same facts. The encoder is the tied relation instrument from the deixis dev. Same-relation slots share one direction, the decoy is stored first, and a 1e-6 insertion-order slope keeps that earliest slot when a batched cosine would otherwise move the tie by one ulp. A hit is who was selected, not an e5 score. A false accept is a rule abstain (`absent_person` or `missing_rel`) that returned a value.

## Disjointness

Literal overlap 0 and max token Jaccard under 0.50. Tokenizer is `tidal.question_heldout_data._tokens`. Strings are questions, turns, and fact sentences.

| against | sha | n | literal | max Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 | 0 | 0.467 |
| surface | — | 1124 | 0 | 0.444 |
| abstain_dev | 26889048ad93… | 381 | 0 | 0.400 |
| frozen | cce20b6bf29f… | 381 | 0 | 0.467 |
| cjk | 5e8b86f694d8… | 201 | 0 | 0.238 |
| nickname | a766f748cfc2… | 43 | 0 | 0.143 |
| typo | 68260c01d5f1… | 68 | 0 | 0.214 |
| indirect_questions | daf4619f2d24… | 50 | 0 | 0.393 |
| indirect_bank | 18b5e2d128fb… | 72 | 0 | 0.393 |
| cue | ed3bd6aadfa9… | 44 | 0 | 0.400 |
| bare | a0d7309555ec… | 100 | 0 | 0.375 |
| residual | b35a4c42560a… | 100 | 0 | 0.471 |
| two_name | 9c72a8b3f65e… | 120 | 0 | 0.400 |
| focus | 03dea71d6a47… | 132 | 0 | 0.292 |
| typo_edit | 3447aaaa0b64… | 96 | 0 | 0.200 |
| lastname_questions | 064d7052e2ca… | 101 | 0 | 0.217 |
| lastname_bank | 626514af8f1f… | 309 | 0 | 0.333 |
| dialogue_dev_questions | 11bc04125694… | 102 | 0 | 0.214 |
| dialogue_dev_bank | 69bd3ed63276… | 340 | 0 | 0.333 |
| dialogue_heldout_questions | 99acea85195e… | 118 | 0 | 0.316 |
| dialogue_heldout_bank | e09fbcc043fd… | 538 | 0 | 0.400 |
| deixis_dev_questions | 4c582b14b181… | 48 | 0 | 0.429 |
| deixis_dev_bank | 1f9674cb2323… | 216 | 0 | 0.429 |
| deixis_heldout_questions | d414c44654db… | 136 | 0 | 0.222 |
| deixis_heldout_bank | 901390f59988… | 670 | 0 | 0.222 |

Closest pair is 0.471 against residual.
`你的生日写在蓝纸封的哪一页？`
`邹岚的出生日写在哪一页，龥齯。`

## Scope

| | n | state rule | empty rule | state human | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| gated | 106 | 106/106 | 106/106 | 106/106 | 0 | 0 |
| gap | 21 | 21/21 | 21/21 | 0/21 | 0 | 0 |
| oos | 8 | 8/8 | 8/8 | 0/8 | 0 | 0 |
| control | 4 | 4/4 | 4/4 | 0/4 | 0 | 0 |
| observe | 3 | 3/3 | 3/3 | 0/3 | 0 | 0 |
| total | 142 | 142/142 | 142/142 | 106/142 | 0 | 0 |

## Family

| | n | state rule | empty rule | state human | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| you | 33 | 33/33 | 33/33 | 30/33 | 0 | 0 |
| third | 35 | 35/35 | 35/35 | 27/35 | 0 | 0 |
| name | 22 | 22/22 | 22/22 | 19/22 | 0 | 0 |
| window | 6 | 6/6 | 6/6 | 2/6 | 0 | 0 |
| edit | 22 | 22/22 | 22/22 | 16/22 | 0 | 0 |
| abstain | 6 | 6/6 | 6/6 | 6/6 | 0 | 0 |
| indirect | 6 | 6/6 | 6/6 | 6/6 | 0 | 0 |
| oos | 8 | 8/8 | 8/8 | 0/8 | 0 | 0 |
| empty | 4 | 4/4 | 4/4 | 0/4 | 0 | 0 |
| total | 142 | 142/142 | 142/142 | 106/142 | 0 | 0 |

## Language

| | n | state rule | empty rule | state human | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| en | 70 | 70/70 | 70/70 | 52/70 | 0 | 0 |
| zh | 70 | 70/70 | 70/70 | 52/70 | 0 | 0 |
| bi | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| total | 142 | 142/142 | 142/142 | 106/142 | 0 | 0 |

## Band

| | n | state rule | empty rule | state human | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 52 | 52/52 | 52/52 | 52/52 | 0 | 0 |
| boundary | 54 | 54/54 | 54/54 | 54/54 | 0 | 0 |
| gap | 21 | 21/21 | 21/21 | 0/21 | 0 | 0 |
| oos | 8 | 8/8 | 8/8 | 0/8 | 0 | 0 |
| control | 4 | 4/4 | 4/4 | 0/4 | 0 | 0 |
| observe | 3 | 3/3 | 3/3 | 0/3 | 0 | 0 |
| total | 142 | 142/142 | 142/142 | 106/142 | 0 | 0 |

## Phenomenon

| | n | state rule | empty rule | state human | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| you_hearer | 16 | 16/16 | 16/16 | 16/16 | 0 | 0 |
| third_unique | 14 | 14/14 | 14/14 | 14/14 | 0 | 0 |
| name_exact | 14 | 14/14 | 14/14 | 14/14 | 0 | 0 |
| third_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| you_followup | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| code_switch | 4 | 4/4 | 4/4 | 4/4 | 0 | 0 |
| third_hold | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| name_aged | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pronoun_hold | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit1_sub | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit1_del | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit1_silent | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit1_other | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit1_tie | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit_extension | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| edit_far | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| missing_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| missing_third | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| absent_name | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| hearer_unset | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| hearer_offroster | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| hearer_switch | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| former_hearer | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| indirect_self | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| indirect_hearer | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| indirect_noself | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| prior_ask_name | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| prior_ask_edit | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| mixed_pronoun | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| not_pronoun_them | 1 | 1/1 | 1/1 | 1/1 | 0 | 0 |
| ta_hit | 1 | 1/1 | 1/1 | 1/1 | 0 | 0 |
| you_named | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| vocative_question | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| third_tie | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| window_backchannel | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| pronoun_chain_age | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| vocative_unpassed | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| gender_tie | 4 | 4/4 | 4/4 | 0/4 | 0 | 0 |
| recency_clash | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| edit_aged | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| edit_insert | 1 | 1/1 | 1/1 | 0/1 | 0 | 0 |
| edit_transpose | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| cjk_prefix | 1 | 1/1 | 1/1 | 0/1 | 0 | 0 |
| speaker_as_third | 2 | 2/2 | 2/2 | 0/2 | 0 | 0 |
| plural_you | 1 | 1/1 | 1/1 | 0/1 | 0 | 0 |
| oos_addressee | 8 | 8/8 | 8/8 | 0/8 | 0 | 0 |
| empty_deictic | 4 | 4/4 | 4/4 | 0/4 | 0 | 0 |
| short_name | 1 | 1/1 | 1/1 | 0/1 | 0 | 0 |
| given_nick | 1 | 1/1 | 1/1 | 0/1 | 0 | 0 |
| pinyin_name | 1 | 1/1 | 1/1 | 0/1 | 0 | 0 |
| total | 142 | 142/142 | 142/142 | 106/142 | 0 | 0 |

## Where the listener and the rule disagree

These rows were marked gap before the score. A human miss with a rule hit is the experience gap. It is not a regression.

### window_backchannel

Four short acknowledgements open four turns. The name is still what he means, and TURN_WINDOW has dropped it.

- `en-age` window_backchannel (en): rule absent_person / human Torven; ask returned absent_person. `Which telephone did he leave on the chalk slate after the nods?` Four short acknowledgements open four turns. The name is still what he means, and TURN_WINDOW has dropped it.
- `zh-age` window_backchannel (zh): rule absent_person / human 喻棠; ask returned absent_person. `他留在后面那块粉笔板上的电话是哪一页？` 四句很短的应答各算一轮。听的人还是在问他，窗口已经把名字推出去了。

### pronoun_chain_age

Later turns only say he, and a pronoun is not written back onto the log, so the name ages out.

- `en-chain` pronoun_chain_age (en): rule absent_person / human Torven; ask returned absent_person. `Which telephone did he leave with the horn cup after dusk?` Later turns only say he, and a pronoun is not written back onto the log, so the name ages out.
- `zh-chain` pronoun_chain_age (zh): rule absent_person / human 喻棠; ask returned absent_person. `他留在黄昏后那只角杯上的电话是哪一页？` 后面几轮只说他。代词不会写回人名，名字就滑出了窗口。

### vocative_unpassed

The line addresses Briony, but note_turn was not given a hearer. you does not recover a vocative.

- `en-vocative-miss` vocative_unpassed (en): rule absent_person / human Briony; ask returned absent_person. `Which telephone should the closer read for you off the green tin by the elbow?` The line addresses Briony, but note_turn was not given a hearer. you does not recover a vocative.
- `zh-vocative-miss` vocative_unpassed (zh): rule absent_person / human 闵昭; ask returned absent_person. `值夜要回读给你的电话，看手边青瓷上的哪一页？` 这句话点了闵昭，但 note_turn 没有传入听者。你不会从呼语里补出来。

### gender_tie

she would pick Elowen. Two names on the newest turn abstain, and gender is not a feature.

- `en-gender` gender_tie (en): rule absent_person / human Elowen; ask returned absent_person. `Which telephone did she leave on the ferry chit for the pair?` she would pick Elowen. Two names on the newest turn abstain, and gender is not a feature.
- `en-gender` gender_tie (en): rule absent_person / human Odhran; ask returned absent_person. `Which birthday did he leave on the ferry chit for the pair?` he would pick Odhran. The same tie abstains.
- `zh-gender` gender_tie (zh): rule absent_person / human 邢渚; ask returned absent_person. `她留在成对渡轮条上的电话是哪一页？` 她会指邢渚。最新一轮上有两个人就弃权，性别不是特征。
- `zh-gender` gender_tie (zh): rule absent_person / human 喻棠; ask returned absent_person. `他留在成对渡轮条上的生日是哪一页？` 他会指喻棠。同样的并列仍然弃权。

### recency_clash

she, going back, is Elowen. eligible_latest keeps Odhran, the newest name, and ignores gender.

- `en-recency` recency_clash (en): rule Odhran / human Elowen; ask returned Odhran. `Going back to the start, which birthday did she mention on the ferry chit from the start?` she, going back, is Elowen. eligible_latest keeps Odhran, the newest name, and ignores gender.
- `zh-recency` recency_clash (zh): rule 喻棠 / human 邢渚; ask returned 喻棠. `回到一开始，她在渡轮条上说的生日是哪一页？` 回到她，指的是邢渚。最新的合格名字是喻棠，性别不参与。

### edit_aged

Pellum is still Pellam. Four later turns have taken the mention out of the window, so the edit is no longer licensed.

- `en-edit-aged` edit_aged (en): rule absent_person / human Pellam; ask returned absent_person. `Which telephone did Pellum leave on the oat sack after the nods?` Pellum is still Pellam. Four later turns have taken the mention out of the window, so the edit is no longer licensed.
- `zh-edit-aged` edit_aged (zh): rule absent_person / human 阮澄; ask returned absent_person. `阮橙的电话记在点头后那条纸带的哪一页？` 阮橙离阮澄一笔。四轮之后窗口不再准许这一笔。

### edit_insert

An extra letter still contains Pellam, and a longer span that contains the name is rejected as a different person.

- `en-insert` edit_insert (en): rule absent_person / human Pellam; ask returned absent_person. `Which telephone did Pellamm leave on the hemp coil with an extra loop?` An extra letter still contains Pellam, and a longer span that contains the name is rejected as a different person.

### edit_transpose

Pellma is Pellam with the last two letters swapped. That distance is 2, so the edit is not licensed.

- `en-swap` edit_transpose (en): rule absent_person / human Pellam; ask returned absent_person. `Which telephone did Pellma leave on the paper band typed fast?` Pellma is Pellam with the last two letters swapped. That distance is 2, so the edit is not licensed.
- `zh-swap` edit_transpose (zh): rule absent_person / human 闵昭; ask returned absent_person. `昭闵的电话记在写反的软木的哪一页？` 昭闵是闵昭两个字对调，距离是 2，编辑许可不会接。

### cjk_prefix

老加全名在口头上还是闵昭。前缀规则把它当成另一个人。

- `zh-prefix` cjk_prefix (zh): rule absent_person / human 闵昭; ask returned absent_person. `老闵昭的电话记在藤圈上的哪一页？` 老加全名在口头上还是闵昭。前缀规则把它当成另一个人。

### speaker_as_third

he is the man who just spoke. The turn names nobody, and a speaker is not eligible even if we knew gender.

- `en-speaker` speaker_as_third (en): rule absent_person / human Torven; ask returned absent_person. `Which telephone did he leave in the wool ledger on the stool?` he is the man who just spoke. The turn names nobody, and a speaker is not eligible even if we knew gender.
- `zh-speaker` speaker_as_third (zh): rule absent_person / human 喻棠; ask returned absent_person. `他放在凳子上那本账里的电话是哪一页？` 他就是刚说完话的喻棠。这一轮没有点名，说话人不会变成第三人。

### plural_you

你们不是单数的你，不该落到听者，也不该落到名册里第一页电话。

- `zh-plural` plural_you (zh): rule DECOY / human ABSENT; ask returned 靳晚. `你们的电话写在柜上那只青瓷的哪一页？` 你们不是单数的你，不该落到听者，也不该落到名册里第一页电话。

## Out of scope

Open addressee forms left out of #34. A failure here is not a regression.

- `en-oos-party` oos_addressee (en): rule absent_person / human Briony; ask returned absent_person. `Which telephone does the other party keep on the green tin for the other party?` other party was left out of the closed class in #34.
- `en-oos-one` oos_addressee (en): rule absent_person / human Tamsin; ask returned absent_person. `Which telephone does this one keep on the chalk slate for this one?` this one was left out of the closed class in #34.
- `en-oos-person` oos_addressee (en): rule absent_person / human Nessa; ask returned absent_person. `Which birthday does that person keep on the moss folder for that person?` that person was left out of the closed class in #34.
- `en-oos-guest` oos_addressee (en): rule absent_person / human Elowen; ask returned absent_person. `Which places does our guest keep on the copper bowl for our guest?` our guest was left out of the closed class in #34.
- `zh-oos-duifang` oos_addressee (zh): rule absent_person / human 闵昭; ask returned absent_person. `对方的电话记在这只后放的青瓷的哪一页？` 对方不在 #34 的封闭类里。
- `zh-oos-zhewei` oos_addressee (zh): rule absent_person / human 霍涟; ask returned absent_person. `这位的电话记在后门那块粉笔的哪一页？` 这位不在 #34 的封闭类里。
- `zh-oos-nawei` oos_addressee (zh): rule absent_person / human 邢渚; ask returned absent_person. `那位的生日记在排练用的苔藓夹的哪一页？` 那位不在 #34 的封闭类里。
- `zh-oos-renjia` oos_addressee (zh): rule absent_person / human 闵昭; ask returned absent_person. `人家的地点记在矮架那只铜碗的哪一页？` 人家不在 #34 的封闭类里。

## Empty log

No turns. The human column abstains. The rule leaves a deictic pronoun on the ranker, which returns the decoy. That is current main behavior.

- `en-empty-you` empty_deictic (en): rule DECOY / human ABSENT; ask returned Sable. `What telephone should the closer use for you off the pebble jar with no talk?` No turn and no hearer. you stays on the ranker.
- `en-empty-he` empty_deictic (en): rule DECOY / human ABSENT; ask returned Sable. `Which telephone did he leave on the reed basket with no talk?` No turn and no hearer. he stays on the ranker.
- `zh-empty-you` empty_deictic (zh): rule DECOY / human ABSENT; ask returned 靳晚. `还没人开口时，你的电话要看没说过的窗台罐上的哪一页？` 没有轮次也没有听者。你留在排序器上。
- `zh-empty-he` empty_deictic (zh): rule DECOY / human ABSENT; ask returned 靳晚. `他还没被提到，电话要看没说过的蓝票上的哪一页？` 没有轮次也没有听者。他留在排序器上。

## Observed, not gated

Short name, given-name nickname, and pinyin. The head may accept or abstain. Neither result is a gate.

- `en-observe-pell` short_name (en): rule absent_person / human Pellam; ask returned absent_person. `Which telephone did Pell leave on the horn cup for the short name?` Pell is a short form, not an edit of one. The name head may or may not take it.
- `zh-observe-nick` given_nick (zh): rule absent_person / human 闵昭; ask returned absent_person. `小昭的电话记在角杯上的小称那一页吗？` 小昭是名的小称。头接不接，不作为本门的通过条件。
- `zh-observe-py` pinyin_name (en): rule absent_person / human 闵昭; ask returned absent_person. `Which telephone did Minzhao leave on the horn cup for the pinyin?` The surface is the stored name's pinyin. The head may or may not take it. Not a gate.

## Spec misses

None. Every ask matched its rule label on the state pass and on the empty pass.

## Floors

The ask path is unchanged, so the floors already recorded in `reports/deixis.md` still stand. They were not re-run for this measurement. Probe answerable 365/370, false abstain 0/370, unanswerable 72/72, gold lost at overwrite 0.9 is 0, co-resident cosine above 0.9 is 0 (max 0.8927). Frozen hits 188/216, false abstain 19/216, near_dup_absent 30/30. Dialogue dev 102/102 and 102/102. Dialogue held-out at window 4, gated 108/108 and 108/108. Deixis dev 48/48 and 48/48. Deixis held-out 128/136 with the turns (the miss is `addressee_extra` 0/8) and 136/136 on the empty log.

## Recommendation

Ship the caller contract, and do not widen the mechanism in this change. Gated rows match the shipped rule on both passes, and a rule abstain never returned a value. The experience gaps are the ones already outside the contract: four quiet turns age a name out, a pronoun does not refresh the log, a vocative is not a hearer unless the caller passes it, two people on the newest turn abstain with no gender, an insertion or a swap or a 老/小 prefix is not edit-1, and the speaker is not he. Open addressee forms stay out of scope. The next product step is to call note_turn with the hearer every time, and to treat the window and the closed class as the behavior, not as a full coreference model. A richer dialogue state would be a new mechanism, not a patch on this bank.
