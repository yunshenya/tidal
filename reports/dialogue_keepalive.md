# Pronoun keep-alive and leading vocatives

Two always-on updates on the utterance path. No threshold, window, or name-head weight was fit. `TURN_WINDOW` stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input and its labels were not edited.

Pre-registration, written before the code: `reports/dialogue_keepalive_prereg.md`.

## What the code does

`DialogueState.refresh` moves one person's latest utterance or ask mention onto the current turn when that mention is still inside `TURN_WINDOW`. Source and surface stay. Nothing is appended. A mention already outside the window is left where it is.

`refresh_resolved` calls that only when `resolve_deixis` returns exactly one person. `None` and `()` do nothing: a stored name in the same string, a tie, a mixed second-and-third person, `them`, `你们`, and a third person who has already aged out. The pronoun text is not stored.

`vocative_hearer` reads a hearer only when `note_turn` was not given one. The utterance, after a left strip, has to start with exactly one exact roster name, and the next character has to be address punctuation (`,` `，` `、` `:` `：` `!` `！`). A second name in that same position after the punctuation leaves the hearer unset. A subject, an object, a name later in the sentence, an edit, a nickname, pinyin, `Hey Name`, and a period do not qualify. A hearer string the caller passed, including one that is not on the roster, is kept.

`Continuum.note_turn` and the #35 bank replay both go through `apply_utterance`, which notes the names `ask_people` found and then restamps. `query` restamps again after it records the ask, so a chain of distinct pronoun questions keeps the person in the window. The names returned by `note_turn` are still the names in the string, not the pronoun. An empty log, with no utterance and no stored hearer, is unchanged.

## Development (#35 bank)

Same questions and same bank strings as the merged measurement. Questions `117ef589bb02bd883db44a4d2d5859b17c271c00a5df324574c8095aced5b704`. Bank `0be42845836ec2a19c961d5a415ccd0d9d0dcb1f8adf5bb632335cd26871cec5` (1032 strings). The four former gap rows are scope `dev`. They are not folded into the gated 106 or the ordinary 52.

State mode walks `note_turn` and `query`. Empty mode asks each question on a fresh store. False accepts 0 on every slice.

| slice | n | state rule | empty rule | state human |
|---|---:|---:|---:|---:|
| gated | 106 | 106/106 | 106/106 | 106/106 |
| ordinary talk | 52 | 52/52 | 52/52 | 52/52 |
| `pronoun_chain_age` | 2 | 2/2 | 2/2 | 2/2 |
| `vocative_unpassed` | 2 | 2/2 | 2/2 | 2/2 |
| other gap rows | 17 | 17/17 | 17/17 | 0/17 |
| open addressee | 8 | 8/8 | 8/8 | 0/8 |

Whole bank: state rule 142/142, empty rule 142/142, listener 110/142, false accepts 0. The listener misses that remain are four quiet turns with no pronoun, gender ties, recency, insertions, swaps, `老` plus the full name, the speaker read as `he`, and `你们`. Detail is in `reports/dialogue_real.md`.

## Held-out

`tidal/question_keepalive_heldout_data.py`. 34 episodes, 42 asks. Questions `092252c79252651ad6cdee2c63783a445ca1a2e4e15804732c955a8407de1a03`. Bank `694fed5e12b4e4dbbcd831c8394e7101cbcab908d6406dd5ac5611a4e2c85b25` (509 strings).

People: Calder, Mirren, Solway, Anwick, Berrick, Yarrow; 卞茗, 苟宁, 裴涟, 栾昭, 童屿. Props: lichen board, kelp ticket, amber spindle, salt-wick spool; 青苔板, 海带票, 琥珀轴, 盐芯. The edit surfaces are Caldar and 苟柠. Both are distance 1, neither is an extension, and neither matches on an empty log.

Rule labels were checked against the pre-registered human labels when the bank was built. A disagreement would have failed the build. State and empty then each matched that rule label. False accepts 0. Listener column on the state pass is 42/42, because the pre-registered label for the `empty` rows is the decoy the ranker returns.

| kind | n | state rule | empty rule | state human |
|---|---:|---:|---:|---:|
| `keep_he` | 1 | 1/1 | 1/1 | 1/1 |
| `keep_she` | 1 | 1/1 | 1/1 | 1/1 |
| `keep_ta` | 1 | 1/1 | 1/1 | 1/1 |
| `keep_zhta` | 2 | 2/2 | 2/2 | 2/2 |
| `keep_edit` | 2 | 2/2 | 2/2 | 2/2 |
| `keep_ask` | 10 | 10/10 | 10/10 | 10/10 |
| `age_quiet` | 2 | 2/2 | 2/2 | 2/2 |
| `age_resurrect` | 2 | 2/2 | 2/2 | 2/2 |
| `age_tie` | 2 | 2/2 | 2/2 | 2/2 |
| `age_edit` | 2 | 2/2 | 2/2 | 2/2 |
| `voc_lead` | 5 | 5/5 | 5/5 | 5/5 |
| `voc_subject` | 2 | 2/2 | 2/2 | 2/2 |
| `voc_object` | 2 | 2/2 | 2/2 | 2/2 |
| `voc_medial` | 2 | 2/2 | 2/2 | 2/2 |
| `voc_two` | 2 | 2/2 | 2/2 | 2/2 |
| `voc_explicit` | 2 | 2/2 | 2/2 | 2/2 |
| `empty` | 2 | 2/2 | 2/2 | 2/2 |
| total | 42 | 42/42 | 42/42 | 42/42 |

`keep_ask` is five distinct pronoun questions after the name, in English and in Chinese. The fifth still names that person. `age_quiet` is four acknowledgements and then the pronoun. `age_resurrect` adds one pronoun utterance after those four and still abstains. `age_edit` is Caldar / 苟柠 after the same quiet stretch. `voc_lead` covers comma, exclamation, and colon. `voc_explicit` keeps the caller hearer when the line starts with someone else.

On an empty log, `you` / `he` / `她` stay on the ranker and return the decoy. `ta` abstains, because `is_deictic` does not include `ta`. That is the same empty-log rule as the deixis bank. The pre-registered `empty` rows are `you` and `他`, and both return the decoy.

Literal overlap with every earlier bank, including the #35 questions `117ef589…` and bank `0be42845…`, is 0. Max token Jaccard is under 0.50. Tokenizer is `tidal.question_heldout_data._tokens`. Closest pair is 0.389 against the #35 bank:

`她在对着琥珀轴的门口挥了下手。 bracken scrip 028`

`他在门口挥了下手。`

## Floors

Re-run because the ask path changed.

| bank | result |
|---|---|
| probe answerable | 365/370 (en 183/185, zh 182/185; phone 122/124, birthday 122/124, places 121/122) |
| probe false abstain | 0/370 |
| probe unanswerable | 72/72 (missing_rel 36/36, absent_person 36/36) |
| probe gold lost @0.9 | 0/370 |
| probe co-resident cosine above 0.9 | 0 (8390 pairs, max 0.8927) |
| frozen answerable hits | 188/216 |
| frozen false abstain | 19/216 |
| frozen missing_rel abstained | 60/69 |
| frozen absent_person | 42/42 |
| frozen near_dup_absent | 30/30 |
| dialogue dev, with the log | 102/102 |
| dialogue dev, empty log | 102/102 |
| dialogue held-out, W=4, with the log | 108/108 gated, misses 0 (lag_observe 4/10, not gated) |
| dialogue held-out, empty log | 108/108 gated, misses 0 |
| deixis dev, with the log | 48/48 |
| deixis dev, empty log | 48/48 |
| deixis held-out, with the log | 128/136 (the miss is `addressee_extra` 0/8) |
| deixis held-out, empty log | 136/136 |

The ask path was not edited for the expansion, or for the positive slice below. The same floors were run again after each, and did not move. `python3 -m pytest -q`: 323 passed, 1 skipped. The skip is `test_tiny_llama_empty_store_matches_and_script_runs`, which imports `transformers`. Four of the passing tests are the expansion bank. Four more are this positive slice.

## Expanded held-out

Pre-registration, written before this score: `reports/dialogue_keepalive_expand_prereg.md`. The rule was not edited. The 42-ask bank above is unchanged (questions `092252c7…`, bank `694fed5e…`).

`tidal/question_keepalive_expand_data.py`. 54 episodes, 94 asks. English 47, Chinese 47. Questions `9af36973b625ce4ae30f516fafdc3a4aec5c37fbfcaf2a8fc31ec423025ee19c`. Bank `30af4bb72c1fbe346dd6b5572ebb60d811b860b71ad060385654f3a7d35f1604` (1176 strings). People: Phelan, Idony, Hawise, Thurstan, Rohese, Osbern; 庞荞, 倪蒲, 滕岚, 邹蒲, 雷桐, 傅苔. The stamp is `marl docket`.

72 asks are listener abstentions. 22 are listener hits: 8 pronoun chains, 8 leading vocatives, and 6 explicit roster hearers that override a different leading name. The other 4 explicit rows pass a hearer that is not on the roster and abstain.

The replay matched the pre-registered listener label on every row (0 gaps). State and empty then matched that rule. False accepts 0.

| kind | n | state rule | empty rule | state human | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| `neg_subject` | 12 | 12/12 | 12/12 | 12/12 | 0 | 0 |
| `neg_object` | 12 | 12/12 | 12/12 | 12/12 | 0 | 0 |
| `neg_quote` | 12 | 12/12 | 12/12 | 12/12 | 0 | 0 |
| `neg_ambig` | 12 | 12/12 | 12/12 | 12/12 | 0 | 0 |
| `neg_unknown` | 8 | 8/8 | 8/8 | 8/8 | 0 | 0 |
| `neg_explicit` | 10 | 10/10 | 10/10 | 10/10 | 0 | 0 |
| `neg_age` | 12 | 12/12 | 12/12 | 12/12 | 0 | 0 |
| `pos_keep` | 8 | 8/8 | 8/8 | 8/8 | 0 | 0 |
| `pos_voc` | 8 | 8/8 | 8/8 | 8/8 | 0 | 0 |
| total | 94 | 94/94 | 94/94 | 94/94 | 0 | 0 |

`neg_subject` and `neg_object` ask `you` / `你` about a name that is not a leading vocative. `neg_quote` puts the name in quotation or after a reporting verb. `neg_ambig` is two people on the newest turn (`he` / `she` / `他` / `她` do not pick a gender), two leading names, and an ambiguous pronoun followed by three quiet turns (a restamp of either person would still have been inside the window). `neg_unknown` is `Sigrid` / `Phelam` / `沈禾` / `牟芷`. `neg_explicit` keeps the caller hearer. `neg_age` is four acknowledgements, four `them` or `你们` turns, or four `it` / `它` turns, then the pronoun. `pos_keep` and `pos_voc` are the controls.

Literal overlap with every earlier bank, including the 42-ask bank and the #35 bank, is 0. Max token Jaccard is 0.394, against the 42-ask bank:

`他朝榫钉外侧的窗台扬了扬手，桦皮条还靠在那儿。 marl docket 188`

`他朝夜闩对面的窗台扬了扬手，青苔板还靠在那儿。 bracken scrip 022`

Union with the frozen 42: 136 scored asks. State rule 136/136, empty rule 136/136, listener 136/136, false accepts 0. No miss to leave unfixed.

## Positive held-out

Pre-registration, committed before this score: `reports/dialogue_keepalive_pos_prereg.md` (`56634f5`). The rule was not edited. Listener labels were read off the dialogue text and were not changed after the score.

`tidal/question_keepalive_pos_data.py`. 116 episodes, 116 asks. English 58, Chinese 58. Questions `322afe2bba895dae80beda74db92272356732a771e561e6bb2d2b5b983c47038`. Bank `b32f13929119c17b1cc4dfcc81b534f21e7c953a6c5904eba87a7062a5f064af` (2546 strings). People: Ranulf, Isolde, Petronilla, Wymond, Mahel, Goscelin; 祁松, 康梧, 钮芮, 费桐, 薛蔓, 穆宁. The stamp is `gallnut tally`. Goscelin and 穆宁 are stored first. `pos_keep` is 48 asks, `pos_voc` is 48, `neg_near` is 20.

State pass is the state-mode answer against the replayed rule. Empty pass is the fresh store against that same rule. Listener is the state-mode answer against the pre-registered label. A false accept is a rule abstain that returned a value. The encoder matched the rule on every ask. False accepts 0.

| kind | n | state rule | empty rule | listener | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| `pos_keep` | 48 | 48/48 | 48/48 | 44/48 | 0 | 0 |
| `pos_voc` | 48 | 48/48 | 48/48 | 22/48 | 0 | 0 |
| `neg_near` | 20 | 20/20 | 20/20 | 20/20 | 0 | 0 |
| total | 116 | 116/116 | 116/116 | 86/116 | 0 | 0 |

| lang | n | state rule | empty rule | listener | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| en | 58 | 58/58 | 58/58 | 46/58 | 0 | 0 |
| zh | 58 | 58/58 | 58/58 | 40/58 | 0 | 0 |

`pos_keep` hits the chains of 2, 3, 4, and 5 intervening turns, and the mixed gap of 6 where a pronoun still sits inside the window. That covers `he` / `she` / `you` and `他` / `她` / `你` / `您`. The four misses are the third-person pronoun after five acknowledgements (`he6a`, `ta6a`, telephone and birthday). The `you` / `您` twins of that quiet gap still hit: the comma address had set the hearer, and the later pronoun does not need the name to stay inside the window.

`pos_voc` hits a leading roster name plus a comma or a dunhao, including `Ranulf, you` / `祁松，你` and the same shapes followed by one acknowledgement that does not repeat the address. The 26 misses are `Dr.`, `啊`, `呀`, `老祁` / `老康`, `祁老师` / `康老师`, and a name followed by `you` / `你` with no comma. Each of those abstains. `neg_near` abstains on all 20, so a degenerate always-accept does not pass. On an empty log every deictic ask in this slice returns the decoy, Goscelin or 穆宁.

Literal overlap with every earlier bank, including the 42-ask bank, the 94-ask expansion, and #35, is 0. Max token Jaccard is 0.412, against the frozen abstain file:

`Which telephone is kept for you with the gallnut tag near the quern pin? gallnut tally 0406`

`Your telephone, the one kept for you, is which?`

Union of the three keep-alive banks: 252 scored asks. State rule 252/252, empty rule 252/252, listener 222/252, false accepts 0. The 30 listener misses are only in this slice. None of them is patched, and no label was edited after the score.

### Listener misses

- `pos_keep_en_he6a_phone` pos_keep en phone: listener Ranulf (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0241.
  - Ranulf rested the walnut hull by the dye pot after the bench was wiped. gallnut tally 0259
  - Leave the dye pot untouched and keep the aisle open. gallnut tally 0260
  - Quiet around the walnut hull for a moment, with the lid left shut. gallnut tally 0261
  - The dye pot does not need a hand yet, so the bench stays clear. gallnut tally 0262
  - A nod over the walnut hull, and nothing else was moved. gallnut tally 0263
  - The dye pot can stay as it is until the next check. gallnut tally 0264
  - He pointed at the walnut hull near the dye pot. gallnut tally 0265
  - Which telephone did he leave with the walnut hull near the dye pot? gallnut tally 0266
- `pos_keep_en_he6a_birthday` pos_keep en birthday: listener Ranulf (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0273.
  - Ranulf set the clover tow beside the heckle pin and stepped back. gallnut tally 0285
  - Quiet around the clover tow for a moment, with the lid left shut. gallnut tally 0286
  - The heckle pin does not need a hand yet, so the bench stays clear. gallnut tally 0287
  - A nod over the clover tow, and nothing else was moved. gallnut tally 0288
  - The heckle pin can stay as it is until the next check. gallnut tally 0289
  - No fresh mark on the clover tow after that pause. gallnut tally 0290
  - He folded the clover tow once the heckle pin cooled. gallnut tally 0291
  - Which birthday did he mark beside the clover tow and the heckle pin? gallnut tally 0292
- `pos_voc_en_dr_ranulf_phone` pos_voc en phone: listener Ranulf (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0665.
  - Dr. Ranulf, the broom corn rests near the besom peg. gallnut tally 0683
  - Which telephone is kept for you with the broom corn near the besom peg? gallnut tally 0684
- `pos_voc_en_dr_ranulf_birthday` pos_voc en birthday: listener Ranulf (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0691.
  - Dr. Ranulf, the ragwort bundle rests near the ditch stake. gallnut tally 0703
  - Which birthday stays recorded for you, the ragwort bundle by the ditch stake? gallnut tally 0704
- `pos_voc_en_dr_isolde_phone` pos_voc en phone: listener Isolde (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0705.
  - Dr. Isolde, the mallow pad rests near the trough lip. gallnut tally 0723
  - Which telephone is kept for you with the mallow pad near the trough lip? gallnut tally 0724
- `pos_voc_en_dr_isolde_birthday` pos_voc en birthday: listener Isolde (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0731.
  - Dr. Isolde, the weld seed rests near the copper pan. gallnut tally 0743
  - Which birthday stays recorded for you, the weld seed by the copper pan? gallnut tally 0744
- `pos_voc_en_dr_later_ranulf_phone` pos_voc en phone: listener Ranulf (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0867.
  - Dr. Ranulf, the weld scrap was left near the pot hook. gallnut tally 0885
  - The pot hook does not need a hand yet, so the bench stays clear. gallnut tally 0886
  - Which telephone is kept for you with the weld scrap near the pot hook? gallnut tally 0887
- `pos_voc_en_dr_later_ranulf_birthday` pos_voc en birthday: listener Ranulf (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0894.
  - Dr. Ranulf, the indigo cake was left near the vat rim. gallnut tally 0906
  - The vat rim does not need a hand yet, so the bench stays clear. gallnut tally 0907
  - Which birthday stays recorded for you, the indigo cake by the vat rim? gallnut tally 0908
- `pos_voc_en_bare_you_later_isolde_phone` pos_voc en phone: listener Isolde (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0909.
  - Isolde you left the cochineal bit near the bowl lip. gallnut tally 0927
  - The bowl lip does not need a hand yet, so the bench stays clear. gallnut tally 0928
  - Which telephone is kept for you with the cochineal bit near the bowl lip? gallnut tally 0929
- `pos_voc_en_bare_you_later_isolde_birthday` pos_voc en birthday: listener Isolde (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0936.
  - Isolde you left the sumac chip near the tan pit. gallnut tally 0948
  - The tan pit does not need a hand yet, so the bench stays clear. gallnut tally 0949
  - Which birthday stays recorded for you, the sumac chip by the tan pit? gallnut tally 0950
- `pos_voc_en_dr_later_petronilla_phone` pos_voc en phone: listener Petronilla (answerable), rule absent_person. State absent_person. Empty Goscelin gnq0991.
  - Dr. Petronilla, the broom corn was left near the besom peg. gallnut tally 1009
  - The besom peg does not need a hand yet, so the bench stays clear. gallnut tally 1010
  - Which telephone is kept for you with the broom corn near the besom peg? gallnut tally 1011
- `pos_voc_en_dr_later_petronilla_birthday` pos_voc en birthday: listener Petronilla (answerable), rule absent_person. State absent_person. Empty Goscelin gnq1018.
  - Dr. Petronilla, the ragwort bundle was left near the ditch stake. gallnut tally 1030
  - The ditch stake does not need a hand yet, so the bench stays clear. gallnut tally 1031
  - Which birthday stays recorded for you, the ragwort bundle by the ditch stake? gallnut tally 1032
- `pos_keep_zh_ta6a_phone` pos_keep zh phone: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq1515.
  - 祁松把核桃壳挪到染锅边上，擦了擦手就停住。 gallnut tally 1533
  - 染锅先别挪，过道留出来。 gallnut tally 1534
  - 先静一静，核桃壳放着就好，盖子别揭。 gallnut tally 1535
  - 染锅暂时不用动手，案边保持空着。 gallnut tally 1536
  - 点一下头，核桃壳先这样，别的都别动。 gallnut tally 1537
  - 染锅保持原样即可，下一回再看。 gallnut tally 1538
  - 他指了指染锅旁边的核桃壳。 gallnut tally 1539
  - 他的电话落在核桃壳挨着染锅的哪一栏？ gallnut tally 1540
- `pos_keep_zh_ta6a_birthday` pos_keep zh birthday: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq1547.
  - 祁松把苜蓿缕搁到栉钉旁边，然后退开半步。 gallnut tally 1559
  - 先静一静，苜蓿缕放着就好，盖子别揭。 gallnut tally 1560
  - 栉钉暂时不用动手，案边保持空着。 gallnut tally 1561
  - 点一下头，苜蓿缕先这样，别的都别动。 gallnut tally 1562
  - 栉钉保持原样即可，下一回再看。 gallnut tally 1563
  - 苜蓿缕上没有新痕，这一停就到此。 gallnut tally 1564
  - 他把苜蓿缕折好，栉钉已经凉了。 gallnut tally 1565
  - 他的生日收在苜蓿缕同栉钉一块的哪一栏？ gallnut tally 1566
- `pos_voc_zh_a_qi_phone` pos_voc zh phone: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq1939.
  - 祁松啊，扫帚草靠在帚钉旁边。 gallnut tally 1957
  - 你的电话落在扫帚草挨着帚钉的哪一栏？ gallnut tally 1958
- `pos_voc_zh_a_qi_birthday` pos_voc zh birthday: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq1965.
  - 祁松啊，艾捆靠在沟桩旁边。 gallnut tally 1977
  - 你的生日收在艾捆同沟桩一块的哪一栏？ gallnut tally 1978
- `pos_voc_zh_ya_kang_phone` pos_voc zh phone: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq1979.
  - 康梧呀，锦葵垫靠在槽沿旁边。 gallnut tally 1997
  - 您的电话落在锦葵垫挨着槽沿的哪一栏？ gallnut tally 1998
- `pos_voc_zh_ya_kang_birthday` pos_voc zh birthday: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2005.
  - 康梧呀，皂角渣靠在磨槽旁边。 gallnut tally 2017
  - 您的生日收在皂角渣同磨槽一块的哪一栏？ gallnut tally 2018
- `pos_voc_zh_lao_qi_phone` pos_voc zh phone: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2019.
  - 老祁，皂荚牌靠在石磨钉旁边。 gallnut tally 2037
  - 你的电话落在皂荚牌挨着石磨钉的哪一栏？ gallnut tally 2038
- `pos_voc_zh_lao_qi_birthday` pos_voc zh birthday: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2045.
  - 老祁，蓝靛条靠在染缸桩旁边。 gallnut tally 2057
  - 你的生日收在蓝靛条同染缸桩一块的哪一栏？ gallnut tally 2058
- `pos_voc_zh_teacher_qi_phone` pos_voc zh phone: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2059.
  - 祁老师，茜草片靠在染槽旁边。 gallnut tally 2077
  - 您的电话落在茜草片挨着染槽的哪一栏？ gallnut tally 2078
- `pos_voc_zh_teacher_qi_birthday` pos_voc zh birthday: listener 祁松 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2085.
  - 祁老师，起绒牌靠在漂洗钩旁边。 gallnut tally 2097
  - 您的生日收在起绒牌同漂洗钩一块的哪一栏？ gallnut tally 2098
- `pos_voc_zh_bare_ni_kang_phone` pos_voc zh phone: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2099.
  - 康梧你把亚麻饼挪到榨布旁边。 gallnut tally 2117
  - 你的电话落在亚麻饼挨着榨布的哪一栏？ gallnut tally 2118
- `pos_voc_zh_bare_ni_kang_birthday` pos_voc zh birthday: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2125.
  - 康梧你把橡瘿挪到墨角旁边。 gallnut tally 2137
  - 你的生日收在橡瘿同墨角一块的哪一栏？ gallnut tally 2138
- `pos_voc_zh_lao_kang_later_phone` pos_voc zh phone: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2179.
  - 老康，胭脂屑靠在碗沿旁边。 gallnut tally 2197
  - 碗沿暂时不用动手，案边保持空着。 gallnut tally 2198
  - 你的电话落在胭脂屑挨着碗沿的哪一栏？ gallnut tally 2199
- `pos_voc_zh_lao_kang_later_birthday` pos_voc zh birthday: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2206.
  - 老康，五倍子靠在鞣坑旁边。 gallnut tally 2218
  - 鞣坑暂时不用动手，案边保持空着。 gallnut tally 2219
  - 你的生日收在五倍子同鞣坑一块的哪一栏？ gallnut tally 2220
- `pos_voc_zh_a_niu_later_phone` pos_voc zh phone: listener 钮芮 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2221.
  - 钮芮啊，核桃壳靠在染锅旁边。 gallnut tally 2239
  - 染锅暂时不用动手，案边保持空着。 gallnut tally 2240
  - 您的电话落在核桃壳挨着染锅的哪一栏？ gallnut tally 2241
- `pos_voc_zh_a_niu_later_birthday` pos_voc zh birthday: listener 钮芮 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2248.
  - 钮芮啊，苜蓿缕靠在栉钉旁边。 gallnut tally 2260
  - 栉钉暂时不用动手，案边保持空着。 gallnut tally 2261
  - 您的生日收在苜蓿缕同栉钉一块的哪一栏？ gallnut tally 2262
- `pos_voc_zh_teacher_kang_later_phone` pos_voc zh phone: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2263.
  - 康老师，扫帚草靠在帚钉旁边。 gallnut tally 2281
  - 帚钉暂时不用动手，案边保持空着。 gallnut tally 2282
  - 你的电话落在扫帚草挨着帚钉的哪一栏？ gallnut tally 2283
- `pos_voc_zh_teacher_kang_later_birthday` pos_voc zh birthday: listener 康梧 (answerable), rule absent_person. State absent_person. Empty 穆宁 gnq2290.
  - 康老师，艾捆靠在沟桩旁边。 gallnut tally 2302
  - 沟桩暂时不用动手，案边保持空着。 gallnut tally 2303
  - 你的生日收在艾捆同沟桩一块的哪一栏？ gallnut tally 2304

No encoder miss. State and empty matched the replayed rule on all 116 asks.

## Recommendation

Keep both updates on the evidence of the 42-ask bank and the 94-ask expansion. A pronoun chain keeps a person alive only while some mention is still inside the window, and it does not write the pronoun out as a name. A leading exact name plus address punctuation is the only vocative. The positive slice was not used to change that. Its 30 listener misses stay as scored: five quiet turns before `he` / `他`, and addresses the frozen vocative does not read (`Dr.`, `啊`, `呀`, `老` plus a surname, surname plus `老师`, and a name then `you` / `你` with no comma). Gender, open addressee forms, insertions, swaps, the speaker read as `he`, and `你们` stay as they are.
