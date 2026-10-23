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

The ask path was not edited for the expansion. The same floors were run again afterwards and did not move. `python3 -m pytest -q`: 319 passed, 1 skipped. The skip is `test_tiny_llama_empty_store_matches_and_script_runs`, which imports `transformers`. Four of the passing tests are the expansion bank.

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

## Recommendation

Keep both updates. The expanded held-out did not disagree with them, and it did not justify a wider rule. A pronoun chain keeps a person alive only while some mention is still inside the window, and it does not write the pronoun out as a name. A leading exact name is the only vocative. Four quiet turns with no pronoun still age a name out. A name in subject, object, or quotation position does not become the hearer. Two candidates, an unknown vocative, and an explicit hearer stay as the frozen rule reads them. Gender, open addressee forms, insertions, swaps, `老` plus the full name, the speaker read as `he`, and `你们` stay as they are.
