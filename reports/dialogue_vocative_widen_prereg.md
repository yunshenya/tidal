# Wider vocative hearers — pre-registration

Written before the code and before the fresh held-out is scored. The rule at `8a3b881` is the base. `TURN_WINDOW` stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input and its labels are not edited. `name_spans` is not edited: a `老` / `小` / `大` / `阿` prefix still blocks a name on the ask path. The #35 row `老闵昭的电话` stays a gap.

## What changes

Only `vocative_hearer`, when `note_turn` was not given a hearer. A caller-passed hearer still wins, including one that is not on the roster. An empty log, with no utterance and no stored hearer, stays the ranker.

The utterance, after a left strip, addresses exactly one roster person at the start. The head is one of:

- The exact roster name.
- An English courtesy title, then whitespace, then that name. The title is `Dr`, `Mr`, `Mrs`, `Ms`, or `Miss`, with an optional period. `Hey` is not a title.
- `老` / `小` / `大` / `阿` plus that person's surname, the first character of the name. The next character is the address mark below, so `老` plus the full name is a different head: the full name itself after the prefix, and only when that full name is on the roster.
- The surname, or the full name, plus `老师`.

The surname form counts only when exactly one stored name starts with that character. Two people who share it leave the hearer unset. A title or a surname that matches nobody on the roster leaves it unset.

The address mark, with no extra word between the head and the mark, is one of:

- The existing punctuation `,` `，` `、` `:` `：` `!` `！`.
- One particle `啊`, `呀`, or `呢`, and an optional punctuation mark from that same set after the particle.
- `you` as a whole word, after optional whitespace. Not `your`. Chinese `你` or `您` immediately after the head, and not `你们`.

A second head in that same position after the mark leaves the hearer unset. A name used as subject or object, a name later in the sentence, a name only inside a quotation, an edit, a nickname, or a pinyin form does not qualify. `祁老师说`, `Dr. Ranulf said you`, `啊，算了`, and a period after the name do not qualify: `说` / `said` / a leading particle / `.` is not an address mark on a unique roster head.

The five-acknowledgement `pos_keep` misses are the window. A mention is kept while its turn is greater than `turn - 4`. The fourth later turn drops it, so five quiet turns and then `he` / `他` do not restamp. `TURN_WINDOW` is not changed.

## Development gate

The development rows are the #36 positive slice, questions `322afe2b…`, bank `b32f1392…`. The 26 `pos_voc` listener misses are the rows this change is for. The 22 `pos_voc` hits stay hits. `neg_near` stays abstention.

| slice | listener | false accepts |
|---|---|---|
| `pos_voc` | 48/48 | 0 |
| `neg_near` | 20/20 | 0 |
| `pos_keep` | 44/48, the four quiet-gap misses stay | 0 |
| 42-ask bank + 94-ask expansion + this slice, 252 asks | not worse than 222/252; the expected score is 248/252 | 0 |

State and empty still match the replay. #35 gated 106/106 and ordinary talk 52/52 stay listener hits, false accepts 0. Probe answerable 365/370, false abstain 0/370, unanswerable 72/72. Frozen answerable hits 188/216, false abstain at most 19/216. Dialogue dev 102/102. Dialogue held-out gated 108/108, misses 0. Deixis dev 48/48. Deixis held-out 128/136 with the log and 136/136 empty.

## Fresh held-out

Built after this gate, and scored once. Listener labels are read off the dialogue text and the roster before that score. They are not taken from `vocative_hearer`. The bank does not replay the rule. Literal overlap 0 and max token Jaccard under 0.50 against every earlier bank, including the three keep-alive banks. Tokenizer is `tidal.question_heldout_data._tokens`.

A positive is the addressed person. A near miss is `ABSENT`.

| kind | asks | en | zh |
|---|---:|---:|---:|
| positive titles, particles, and name + `you` / `你` / `您` | 68 | 34 | 34 |
| near misses of those shapes | 44 | 22 | 22 |

English positives: `Dr` / `Mr` / `Mrs` / `Ms` / `Miss` plus comma, exclamation, or colon; the title then `you` with no comma; the bare name then `you`; and the same heads with one later acknowledgement that does not repeat the address. Chinese positives: `老` / `小` / `大` / `阿` plus the surname; `老` plus the full name; surname or full name plus `老师`; `啊` / `呀` / `呢`; the name then `你` or `您` with no comma; a title plus a particle; and one later acknowledgement after `老`, `呢`, or `你`. Telephone and birthday are two episodes of the same shape, so the second ask does not lengthen the gap.

Near misses, listener `ABSENT`: `Dr. X said you`; a title used as the subject or the object; a title of someone who is not stored; two heads addressed; the name only inside a quotation; `Hey Name`; a period; `your` rather than `you`; a name that is not at the start; `老师说`; a bare `啊，算了` / `呀，算了` / `呢，…`; a surname shared by two stored people. An acknowledgement names nobody. `you` / `你` / `您` after a real address is that person. A subject, a quotation, or an unknown title is not.

Per-shape state, empty, listener, and false accepts are reported after the one score, with every miss verbatim. A miss is not relabeled to match the rule.
