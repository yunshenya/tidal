# Positive keep-alive and vocative slice — pre-registration

Written before this bank is scored, and before the frozen rule is applied to it. The rule at the current HEAD is not edited. `TURN_WINDOW`, `ABSTAIN_MARGIN`, and `LEX_RIDGE_GAP` stay as they are. No enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input.

Listener labels below were read off the dialogue text and the roster. They were not taken from `resolve_deixis`, `vocative_hearer`, or a replay. The bank is `tidal/question_keepalive_pos_data.py`. It does not replay the rule.

## What a listener marks

`pos_keep`. The person the chain is about. The anchor names them, or addresses them, and later turns keep referring with `he` / `she` / `you` or `他` / `她` / `你` / `您`, mixed with acknowledgements that do not introduce anyone else. The gap between the anchor and the question is 2, 3, 4, 5, or 6 turns. An acknowledgement does not switch the person. `you` / `你` / `您` is the person who was addressed. `he` / `she` / `他` / `她` is the other person who was named, not the passed hearer.

`pos_voc`. The person addressed, then a `you` / `你` / `您` question. The address is one of: the name at the start with a comma; a Chinese name at the start with a dunhao `、`; `啊` or `呀` after the name; `老` plus the surname; surname plus `老师`; `Dr.` plus the name; the name immediately followed by `you` / `你`. One shape addresses them and a later turn does not repeat the address; the following `you` question is still that person.

Titles map to the one stored person of that surname: `老祁` and `祁老师` are 祁松, `老康` and `康老师` are 康梧. `Dr. Ranulf` is Ranulf, and likewise Isolde and Petronilla.

`neg_near`. A near miss a listener does not accept. The label is `ABSENT`. The name is the subject, or the object, or a title used as the subject. Two people are named and the pronoun is `he` / `she` / `他` / `她`. Two names are both addressed. The name is only inside a quotation. The leading name is not on the roster.

## Counts

| kind | asks | en | zh | listener |
|---|---:|---:|---:|---|
| `pos_keep` | 48 | 24 | 24 | the person the chain keeps |
| `pos_voc` | 48 | 24 | 24 | the person addressed |
| `neg_near` | 20 | 10 | 10 | `ABSENT` |

116 scored asks, one per episode, so a second question does not lengthen the gap. People: Ranulf, Isolde, Petronilla, Wymond, Mahel, Goscelin; 祁松, 康梧, 钮芮, 费桐, 薛蔓, 穆宁. Goscelin and 穆宁 are stored first and are not named in the dialogue. Wymond and 费桐 speak every turn. Mahel and 薛蔓 are the passed hearer on third-person chains only. The stamp is `gallnut tally`. These strings are not in the 42-ask bank, the 94-ask expansion, or the #35 bank.

## Shapes

`pos_keep`. Twelve chains in each language. Telephone and birthday are two episodes of the same chain. The turns between the anchor and the question are:

| gap | intervening turns | English | Chinese |
|---|---|---|---|
| 2 | pronoun, acknowledgement | he Ranulf; you Ranulf | 他 祁松; 你 祁松 |
| 3 | acknowledgement, pronoun, acknowledgement | she Isolde; you Isolde | 她 康梧; 您 康梧 |
| 4 | pronoun, acknowledgement, pronoun, acknowledgement | he Ranulf; you Petronilla | 他 祁松; 你 钮芮 |
| 5 | acknowledgement, pronoun, acknowledgement, acknowledgement, pronoun | she Isolde; you Ranulf | 她 康梧; 您 祁松 |
| 6 | acknowledgement, pronoun, three times | he Petronilla; you Isolde | 他 钮芮; 你 康梧 |
| 6 | five acknowledgements, then the pronoun | he Ranulf; you Petronilla | 他 祁松; 您 钮芮 |

A third-person anchor names that one person. A second-person anchor addresses that person with a comma and passes no hearer. An acknowledgement names nobody. `he` / `他` is that named person even when the name is Petronilla or 钮芮. The label is not filtered by gender.

`pos_voc`. Twelve address shapes in each language, again split into a telephone episode and a birthday episode. The question is `you`, or `你` / `您`. `later` is one acknowledgement that does not address anyone.

| shape | who |
|---|---|
| comma, then the question | Ranulf; Isolde; Petronilla; 祁松 |
| dunhao `、`, then the question | 康梧 |
| `Dr.`, then the question | Dr. Ranulf; Dr. Isolde |
| `啊` / `呀`, then the question | 祁松啊; 康梧呀 |
| `老` plus the surname, then the question | 老祁 is 祁松 |
| surname plus `老师`, then the question | 祁老师 is 祁松 |
| name, comma, `you` / `你` | Ranulf, you left; Isolde, you left; 祁松，你把 |
| name then `你` with no comma | 康梧你把 |
| comma, then a later acknowledgement | Ranulf; 老康 is 康梧 |
| `Dr.`, then a later acknowledgement | Dr. Ranulf; Dr. Petronilla |
| name then `you`, then a later acknowledgement | Isolde you left; Petronilla, you left |
| `啊`, then a later acknowledgement | 钮芮啊 is 钮芮 |
| `老师`, then a later acknowledgement | 康老师 is 康梧 |
| dunhao, then a later acknowledgement | 钮芮 |

`neg_near` is `ABSENT`. English: Ranulf as subject; Isolde as subject; Petronilla as object; Ranulf as object; Ranulf and Isolde with `he`; Petronilla and Isolde with `she`; Ranulf only inside a quotation; Ranulf and Isolde both addressed; Swithin, who is not stored; Dr. Ranulf as subject. Chinese mirrors that with 祁松, 康梧, 钮芮, 鄂棠, and 祁老师 as subject. The question after a subject, object, quotation, unknown name, title-as-subject, or double address is `you` / `你`. The two-person lines use `he` / `she` / `他` / `她`.

## After the commit

Score once. State pass is the state-mode answer against the replayed rule. Empty pass is the fresh-store answer against the replayed rule. Listener is the state-mode answer against the label in this file. A false accept is a rule abstain that returned a value. Every miss is reported verbatim. None of them is patched, and no label is edited after the score.

Disjointness is required for the bank to be usable: literal overlap 0 and max token Jaccard under 0.50 against every earlier bank, including the 42-ask bank, the 94-ask expansion, and #35. Tokenizer is `tidal.question_heldout_data._tokens`.
