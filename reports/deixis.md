# Deixis

**48/48 with the turns, 48/48 on the empty log.** Probe answerable **365/370**, false abstain **0/370**, unanswerable **72/72**. Frozen hits **188/216**, false abstain **19/216**, near_dup_absent **30/30**. Dialogue dev **102/102** and **102/102**. Held-out at window 4, gated **108/108** and **108/108**, misses 0.

The 136-episode deixis held-out does not clear its gate. With the turns **128/136**, empty log **136/136**, false accepts **0**. The miss is `addressee_extra` **0/8**: `other party`, `this one`, `that person`, `our guest`, `人家`, `对方`, `这位`, and `那位` abstain `absent_person` while a hearer is stored. Every other kind is complete on both passes. The rule was not changed.

`you` / `你` / `您` and a hearer-role cue name the hearer `note_turn` stored. `he` / `she` / `they` / `他` / `她` / `ta` name the one person on the newest conversational turn who is neither speaker nor hearer. An empty log does not take that branch.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen file was not an input. The rule was fixed before this run.

## Gate

Fixed before the scored run. A miss stops the change. No second rule.

With the turns played:

| kind | pass |
|---|---|
| hearer_set | answer the stored hearer, including a hearer-role noun when that hearer is not the first-person self |
| hearer_unset | abstain `absent_person` |
| hearer_ambiguous | abstain `absent_person` (mixed second and third person, or a hearer who is not on the roster) |
| third_unique | answer the one person on the newest eligible turn, including when an older eligible name is still inside the window |
| third_tie | abstain `absent_person` |
| third_aged | abstain `absent_person` |
| third_silent | abstain `absent_person` |
| third_wrong_role | abstain `absent_person` |
| control_named | answer the named person |
| control_you_named | answer the named person, not the hearer |
| control_indirect | answer the unique first-person person |
| control_indirect_noself | abstain `absent_person` |
| control_missing | abstain `missing_rel` |

Empty log (facts written, no `note_turn`): every episode meets its empty label. For a pronoun `is_deictic` already accepts, that label is the current ranker (the first stored fact of the asked relation; same-relation sentence vectors are tied). `ta` is not in `is_deictic`, so the empty log abstains `absent_person`. A hearer-role noun with no stored hearer is the unique first-person person. A question that already names someone is that person.

Literal overlap with every earlier bank, including dialogue dev questions `11bc0412…` and held-out questions `99acea85…`, is 0. Max token Jaccard is under 0.50.

Floors, empty log: probe answerable at least 365/370, false abstain 0/370, unanswerable 72/72. Frozen hits 188/216, false abstain 19/216, near_dup_absent 30/30. Dialogue dev 102/102 and held-out gated 108/108 at window 4, both passes.

## Rule

`resolve_deixis` runs from `matched_fact_rows` and `selected_fact_rows`. `ask_people` does not call it, so `note_turn` and the ask log still record names only. A pronoun is not a later mention.

The branch is taken only when `utterances > 0` or a hearer string is already stored. A fact write does not increment `utterances`. Recording the ask does not either, so asking one string twice stays on the empty slice. No utterance and no hearer is the current path: `you` / `他` stay on the ranker, and a hearer-role noun stays on the unique first-person fact.

A stored name in the question, including an edit the active list already licenses, keeps the name path. `can you read Godgifu telephone` stays Godgifu.

`you` / `your` / `你` / `您` (not `你们`) and a hearer-role cue (`addressee`, `对面那人`, the same class as `is_indirect_hearer`) are the hearer when that string is one roster name. Unset after an utterance, or a hearer who is not on the roster, abstains `absent_person`. A hearer-role cue with no stored hearer does not abstain here. A second-person cue and a third-person pronoun in the same question abstain.

`he` / `she` / `they` / `他` / `她` / `ta` (`them` is not in this class) are `eligible_latest`: utterance and ask mentions inside `TURN_WINDOW` (4), excluding the speaker and the hearer, restricted to the newest turn those mentions occupy. One name is that person. Two names on that turn abstain. Zero names abstain: nobody was said, the only record is a write, the mention has aged out, or the only people said are the speaker and the hearer. An older eligible name still inside the window does not block a later one. The fact store is what `ask` reads, scoped to that one person.

## Dev bank

`tidal/question_deixis_dev_data.py`. The matcher does not import it. Question sha256 `4c582b14b181fcead6b256e1bb031af1c59ffece6047c7eaebc5c814de183734`, n=48 (24 English, 24 Chinese). Bank sha256 `1f9674cb23237ba3879a8c4938be323024d3dcc590e982da55868f15cbe53338`, 216 strings.

Names: Godgifu, Waerburh, Eanswith, Osthryth, Cyneswith, Leofwaru; 郗泠, 逄葭, 晁蘅, 蒯茗, 宓岫, 隗岑. Anchors are `kibble` / `rowen` / `swarfNNN` / `㐆㐇`.

Literal overlap with every reference bank is 0. Token Jaccard is the probe tokenizer. The cap is 0.50. Max across the bank is 0.364.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 | 0 | 0.364 |
| surface cue corpus | — | 1124 | 0 | 0.286 |
| abstain dev | `26889048…` | 381 | 0 | 0.316 |
| frozen adversarial | `cce20b6b…` | 381 | 0 | 0.333 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.208 |
| nickname dev | `a766f748…` | 43 | 0 | 0.071 |
| typo dev | `68260c01…` | 68 | 0 | 0.120 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.214 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.214 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.188 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.190 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.200 |
| two-name dev | `9c72a8b3…` | 120 | 0 | 0.200 |
| focus dev | `03dea71d…` | 132 | 0 | 0.200 |
| typo-edit dev | `3447aaaa…` | 96 | 0 | 0.222 |
| lastname questions | `064d7052…` | 101 | 0 | 0.227 |
| lastname bank | `626514af…` | 309 | 0 | 0.227 |
| dialogue dev questions | `11bc0412…` | 102 | 0 | 0.200 |
| dialogue dev bank | `69bd3ed6…` | 340 | 0 | 0.273 |
| dialogue held-out questions | `99acea85…` | 118 | 0 | 0.231 |
| dialogue held-out bank | `e09fbcc0…` | 538 | 0 | 0.333 |

Closest pair: "kibble rowen 你的电话 swarf111 㐆㐇" ‖ "你有沈予的电话吗？"

Ask was scored with the relation-direction encoder the dialogue pytest uses (the abstain-ridge column, plus a fixed orthogonal nudge, cosine about 0.999 below the exact-copy cutoff). Same-relation slots tie, so an unscoped deictic ask returns the first fact of that relation. The person is chosen by the string rule, not by the sentence vector. "State" plays the turns. "Empty" writes the same facts and asks once.

| kind | n | state | empty |
|---|---:|---:|---:|
| hearer_set | 6 | 6/6 | 6/6 |
| hearer_unset | 4 | 4/4 | 4/4 |
| hearer_ambiguous | 4 | 4/4 | 4/4 |
| third_unique | 8 | 8/8 | 8/8 |
| third_tie | 4 | 4/4 | 4/4 |
| third_aged | 4 | 4/4 | 4/4 |
| third_silent | 4 | 4/4 | 4/4 |
| third_wrong_role | 4 | 4/4 | 4/4 |
| control_named | 2 | 2/2 | 2/2 |
| control_you_named | 2 | 2/2 | 2/2 |
| control_indirect | 2 | 2/2 | 2/2 |
| control_indirect_noself | 2 | 2/2 | 2/2 |
| control_missing | 2 | 2/2 | 2/2 |
| total | 48 | 48/48 | 48/48 |

Failures 0. The 6 hearer-set rows return the hearer's value, including the role noun when the unique first-person fact belongs to someone else. Empty, those pronoun rows return the decoy and the role noun returns the first-person self. The 4 unset rows and the 4 ambiguous rows abstain with the turns and follow the ranker, or abstain, when the log is empty. The 8 third-person hits return the newest eligible person (4 are the only eligible name, 4 are the later of two still inside the window) and, empty, follow `is_deictic` (`ta` abstains, the others return the decoy). Ties, four later turns, a write plus a turn that names nobody, and a mention that is only the speaker or only the hearer abstain with the turns. A named question, including one that also says `you`, stays on that name while a hearer is set. The indirect hearer with no stored hearer still answers the unique first person. A hearer who does not store the asked relation abstains `missing_rel` and does not return the other person's fact.

## Floors

Empty conversational log. Same commands as the shipped reports. The frozen file was scored after the rule was fixed, and it was not an input to the rule.

| bank | result |
|---|---|
| probe answerable | 365/370 (en 183/185, zh 182/185; phone 122/124, birthday 122/124, places 121/122) |
| probe false abstain | 0/370 |
| probe unanswerable | 72/72 (missing_rel 36/36, absent_person 36/36) |
| probe gold lost @0.9 | 0/370 |
| frozen answerable hits | 188/216 (en 92/103, zh 96/113) |
| frozen false abstain | 19/216 (en 11/103, zh 8/113) |
| frozen missing_rel abstained | 60/69 |
| frozen absent_person | 42/42 |
| frozen near_dup_absent | 30/30 |
| dialogue dev, with the log | 102/102 |
| dialogue dev, empty log | 102/102 |
| dialogue held-out, W=4, with the log | 108/108 gated, misses 0 (lag_observe 4/10, not gated) |
| dialogue held-out, empty log | 108/108 gated, misses 0 |

The probe category rows match `reports/dialogue_state.md`: direct 36/36 and 36/36, indirect 39/40 and 38/40, cross-turn 37/37 and 36/37, after distractors 35/36 and 36/36, after update 36/36 and 36/36. Co-resident pairs 8390, cosine above 0.9 is 0, max 0.8927. Frozen `name_variant` typos stay 8/8 English and 8/8 Chinese false abstains. The held-out kind rows match that report's window-4 table, including the indirect hearer.

## What stayed

Lexical hits, the name head, the alias list, CJK verb attachment, and the longer-name rejection are unchanged when no utterance has been noted and no hearer is stored. The indirect hearer still resolves only to the unique first-person fact in that case. `TURN_WINDOW` stays 4.

## Tests

`tests/test_question_deixis.py` locks the sha, literal disjointness, the Jaccard cap, the resolver, and the 48/48 ask gate. `python3 -m pytest -q`: 299 passed, 1 skipped. The skip is `test_tiny_llama_empty_store_matches_and_script_runs`, which imports `transformers`. That package is not installed here. The other `LLMMemory` tests passed. `tidal/llm_memory.py` is unchanged.

## Held-out gate

Fixed before the scored run. The rule is not changed on this bank. A miss is reported as a miss. It is not repaired here, and a repair would need a new slice.

`tidal/question_deixis_heldout_data.py`. 136 gated episodes, 68 English and 68 Chinese. Question sha256 `d414c44654db45eb0df85e0220045a28a44441e65b539034545ce8bade0d0dd6`. Bank sha256 `901390f59988cf68bebfbc352a2c0f8f768262ab13a6057cb748a144e85ef770`, 670 strings. Names: Sunngifu, Tidfrith, Wynflaed, Beorngyth, Quenburh, Aelflaed; 鄢姝, 濮苓, 缑蔓, 佘芷, 卞琦, 邬岚. Carriers: `sklent` / `mundle` / `criftNNN` / `plorq` / `龗龞`.

Literal overlap with every earlier bank is 0, including deixis dev questions `4c582b14b181fcead6b256e1bb031af1c59ffece6047c7eaebc5c814de183734` and bank `1f9674cb23237ba3879a8c4938be323024d3dcc590e982da55868f15cbe53338`, dialogue dev questions `11bc041256941969c1055bb5bfc3cd491407718131689aa2f253a2d38e669f2a`, and dialogue held-out questions `99acea85195eca1f0e680f287b040646c321853c7312168801c46d38175f8d5d`. Max token Jaccard is 0.312. The cap is 0.50.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 | 0 | 0.286 |
| surface cue corpus | — | 1124 | 0 | 0.235 |
| abstain dev | `26889048…` | 381 | 0 | 0.200 |
| frozen adversarial | `cce20b6b…` | 381 | 0 | 0.250 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.125 |
| nickname dev | `a766f748…` | 43 | 0 | 0.000 |
| typo dev | `68260c01…` | 68 | 0 | 0.080 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.222 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.222 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.125 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.136 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.167 |
| two-name dev | `9c72a8b3…` | 120 | 0 | 0.143 |
| focus dev | `03dea71d…` | 132 | 0 | 0.143 |
| typo-edit dev | `3447aaaa…` | 96 | 0 | 0.000 |
| lastname questions | `064d7052…` | 101 | 0 | 0.150 |
| lastname bank | `626514af…` | 309 | 0 | 0.150 |
| dialogue dev questions | `11bc0412…` | 102 | 0 | 0.214 |
| dialogue dev bank | `69bd3ed6…` | 340 | 0 | 0.214 |
| dialogue held-out questions | `99acea85…` | 118 | 0 | 0.312 |
| dialogue held-out bank | `e09fbcc0…` | 538 | 0 | 0.312 |
| deixis dev questions | `4c582b14…` | 48 | 0 | 0.308 |
| deixis dev bank | `1f9674cb…` | 216 | 0 | 0.308 |

Closest pair: "sklent mundle 面前这位电话 crift043 plorq 龗龞" ‖ "yelm stook 跟前这位 电话 rickle532 㐂㐃".

`TURN_WINDOW` is 4. A mention followed by 3 quiet turns is still inside the window and is answered. Four quiet turns, and five, are outside it and abstain. The current speaker and the current hearer are the ones excluded. A hearer who was replaced is not excluded. Gender is not a feature: `his` names Wynflaed on that row. `them` is not a third-person cue. `你们` is not singular `你`. A fact write is not a mention. Silent is a write plus one turn that names nobody. A store with no turn is the empty column, not the silent kind.

Empty log: the same facts, no `note_turn`, and no earlier ask. A deictic pronoun returns the first stored fact of the asked relation. `ta` abstains `absent_person`. A stored name returns that person. A hearer-role noun with no stored hearer returns the unique first-person person, or abstains `absent_person` when there is none. `人家` / `对方` / `这位` / `那位` and the English paraphrases `other party`, `this one`, `that person`, `our guest` are not deictic, so the empty log abstains `absent_person`.

With the turns, every row is gated:

| kind | n | pass |
|---|---:|---|
| hearer_set_you | 8 | the stored hearer (`you` / `your` / `yours` / `yourself` / `你` / `您`) |
| hearer_set_role | 4 | the stored hearer, who is not the first-person self (`listener`, `hearer`, `面前这位`, `身边那人`) |
| hearer_unset | 4 | abstain `absent_person` |
| hearer_switch | 4 | the later hearer |
| hearer_ambiguous_mixed | 4 | abstain `absent_person` |
| hearer_ambiguous_roster | 4 | abstain `absent_person` |
| addressee_extra | 8 | the stored hearer (`other party`, `this one`, `that person`, `our guest`, `人家`, `对方`, `这位`, `那位`) |
| addressee_extra_unset | 4 | abstain `absent_person` |
| third_unique | 8 | the one eligible person (`he` / `she` / `they` / `他` / `她` / `ta` / `他们`) |
| third_oblique | 10 | that person (`him` / `her` / `his` / `hers` / `their` / `他的` / `她的`) |
| third_later | 8 | the newest eligible person in a 3-person or 4-person window |
| third_after_ask | 8 | the person named by the earlier ask, who is neither speaker nor hearer |
| third_tie | 6 | abstain `absent_person`, including when an older eligible name is still inside the window |
| window_filler_3 | 4 | the mentioned person |
| window_filler_4 | 4 | abstain `absent_person` |
| window_filler_5 | 4 | abstain `absent_person` |
| third_silent | 4 | abstain `absent_person` |
| third_wrong_role | 8 | abstain `absent_person` |
| third_former_hearer | 4 | the mentioned person, after the hearer has changed away from them |
| third_ask_hearer | 4 | abstain `absent_person` |
| control_named | 4 | the named person |
| control_you_named | 4 | the named person, not the hearer |
| control_indirect | 4 | the unique first-person person |
| control_indirect_noself | 4 | abstain `absent_person` |
| control_missing | 4 | abstain `missing_rel` |
| control_not_pronoun | 2 | abstain `absent_person` (`them`) |
| control_plural | 2 | the ranker's first fact, not the hearer (`你们`) |

A false accept is an `absent_person` or `missing_rel` row that returned a fact. The held-out confirms the rule only when every kind is complete on both passes and false accepts are 0.

## Held-out result

Scored once, with the rule left as it was when the gate was committed. Tiny encoder, same tie break as the 48-episode dev: an unscoped deictic ask returns the first fact of that relation.

| kind | n | state | empty | state false accept | empty false accept |
|---|---:|---:|---:|---:|---:|
| hearer_set_you | 8 | 8/8 | 8/8 | 0 | 0 |
| hearer_set_role | 4 | 4/4 | 4/4 | 0 | 0 |
| hearer_unset | 4 | 4/4 | 4/4 | 0 | 0 |
| hearer_switch | 4 | 4/4 | 4/4 | 0 | 0 |
| hearer_ambiguous_mixed | 4 | 4/4 | 4/4 | 0 | 0 |
| hearer_ambiguous_roster | 4 | 4/4 | 4/4 | 0 | 0 |
| addressee_extra | 8 | 0/8 | 8/8 | 0 | 0 |
| addressee_extra_unset | 4 | 4/4 | 4/4 | 0 | 0 |
| third_unique | 8 | 8/8 | 8/8 | 0 | 0 |
| third_oblique | 10 | 10/10 | 10/10 | 0 | 0 |
| third_later | 8 | 8/8 | 8/8 | 0 | 0 |
| third_after_ask | 8 | 8/8 | 8/8 | 0 | 0 |
| third_tie | 6 | 6/6 | 6/6 | 0 | 0 |
| window_filler_3 | 4 | 4/4 | 4/4 | 0 | 0 |
| window_filler_4 | 4 | 4/4 | 4/4 | 0 | 0 |
| window_filler_5 | 4 | 4/4 | 4/4 | 0 | 0 |
| third_silent | 4 | 4/4 | 4/4 | 0 | 0 |
| third_wrong_role | 8 | 8/8 | 8/8 | 0 | 0 |
| third_former_hearer | 4 | 4/4 | 4/4 | 0 | 0 |
| third_ask_hearer | 4 | 4/4 | 4/4 | 0 | 0 |
| control_named | 4 | 4/4 | 4/4 | 0 | 0 |
| control_you_named | 4 | 4/4 | 4/4 | 0 | 0 |
| control_indirect | 4 | 4/4 | 4/4 | 0 | 0 |
| control_indirect_noself | 4 | 4/4 | 4/4 | 0 | 0 |
| control_missing | 4 | 4/4 | 4/4 | 0 | 0 |
| control_not_pronoun | 2 | 2/2 | 2/2 | 0 | 0 |
| control_plural | 2 | 2/2 | 2/2 | 0 | 0 |
| total | 136 | 128/136 | 136/136 | 0 | 0 |

Eight state misses, all in `addressee_extra`, all `absent_person` rather than the stored hearer's value. No fact was returned on a row whose label was an abstention. Empty is 136/136.

`you` / `your` / `yours` / `yourself` / `你` / `您`, the closed hearer nouns, `he` / `she` / `they` / `him` / `her` / `his` / `hers` / `their` / `他` / `她` / `ta` / `他的` / `她的` / `他们`, a 3-person and 4-person window, a pronoun after a name-ask, filler 3 inside the window, fillers 4 and 5 aged out, a tie, a silent write, a speaker-only or hearer-only mention, a hearer change, and the named and indirect controls all met the gate. `them` did not select the mentioned person. `你们` did not select the hearer.

The eight forms are outside that closed class, and the current rule leaves them on the no-name path, which abstains. That is a real miss of the pre-registered reading. The rule was not edited to cover them. A later change would need a fresh slice.
