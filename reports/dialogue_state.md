# Dialogue state

**Shipped.** Flat-row `ask` can accept one same-script edit of a stored name when that person is on the recent dialogue log, and it abstains when the log does not name them. An empty log leaves name resolution where `main` left it.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The rule was fixed on the dev bank below. The frozen adversarial questions were scored after that, and they were not an input to the wording.

## Rule

`DialogueState` is one log per group (`tidal/dialogue_state.py`). The design, the update rules, and the deixis extension that is not implemented are in `docs/dialogue_state.md`.

`ask` reads `conversational()`: utterance and ask mentions inside the last `TURN_WINDOW` (4) turns, most recent first, one entry per person. The question being answered is not its own prior mention. A fact write is on `recent()` and is not on that list, so a store that merely contains the person does not license an edit. `new_session` drops that group's mentions and keeps the facts. `episodic.clear()` drops every log.

In `diagnose`, a span the head refuses is the stored person when the span is an edit of distance 1 (`_ambiguous_edit`), the span is not an extension, and exactly one active stored name is that edit. Zero active names, or two, and the span stays unmatched. Extensions stay a different person when the shorter name is active (`Swithuno`, `储晏然`, `老储晏`).

## Dev bank

`tidal/question_dialogue_dev_data.py`. The matcher does not import it. Question sha256 `11bc041256941969c1055bb5bfc3cd491407718131689aa2f253a2d38e669f2a`, n=102 (50 English, 52 Chinese). The bank is 340 strings once questions, turns, earlier asks, and statements are deduped. Bank sha256 `69bd3ed6327625afc07cd06a46cbe771d998e806f4a430b3da2aeba014f3ec98`. Sentence cache `tidal/question_dialogue_emb.npz` sha256 `959b03d0e963007359b49ae863a2bed7ddbddc057dc1f8e1b7b07a57aed53634`, n=340, same embedder as the other caches (`intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, mean pool, L2, maxlen 64, no query prefix).

| kind | n | with the log | with an empty log |
|---|---:|---|---|
| typo_hit_utterance | 16 | answer the person just named | abstain `absent_person` |
| typo_abstain_silent | 16 | abstain; the fact write is not a mention | abstain |
| typo_abstain_other | 16 | abstain; only a different person was named | abstain |
| typo_hit_prior_ask | 8 | answer; an earlier ask named them | abstain `absent_person` |
| typo_abstain_stale | 4 | abstain; four later turns pushed the mention out | abstain |
| typo_abstain_tie | 2 | abstain; two stored names are both that edit | abstain |
| spelling_hit | 8 | answer the exact name | answer |
| nickname_hit | 6 | answer the diminutive | answer |
| latin_extension | 2 | abstain; the longer name stays rejected | abstain |
| cjk_superstring | 4 | abstain | abstain |
| cjk_prefix | 2 | abstain | abstain |
| focus_hit | 4 | answer the restricted name | answer |
| dismiss_hit | 4 | answer the name outside the dismissed clause | answer |
| open_either | 4 | answer either value | answer either |
| indirect_hit | 2 | answer the unique first-person person | answer |
| indirect_noself | 2 | abstain `absent_person` | abstain |
| indirect_missing | 2 | abstain `missing_rel` | abstain |

Names: Swithun, Eadburg, Cuthbert, Frideswide, Beornwyn, Mildthryth, Osgyth, Aelfgifu, Wulfrun, Cyneburg, Eormengyth, Saethryth, Virginia, Rebecca, Anthony, Dorothy, Hrothgar; 储晏, 毕芩, 伍蘅, 屠芃, 钮茜, 柯菱, 管菁, 莫荃, 宣蘅, 杭萱, 包菁, 左荃, 缪荭. Anchors are `barmkin` / `oxter` / `thridNN` / `㐀㐁`. The same typo surface is the hit and the abstain (`Swithan` after Swithun was named, and `Swithan` when only the write exists or only Wulfrun was named).

Literal overlap with every reference bank is 0. Token Jaccard is the probe tokenizer. The cap is 0.50. Max across the bank is 0.353.

| against | sha | n | literal | max token Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 strings | 0 | 0.313 |
| surface cue corpus | — | 1124 strings | 0 | 0.250 |
| abstain dev | `26889048…` | 381 strings | 0 | 0.222 |
| frozen adversarial | `cce20b6b…` | 381 strings | 0 | 0.353 |
| CJK dev | `5e8b86f6…` | 201 | 0 | 0.115 |
| nickname dev | `a766f748…` | 43 | 0 | 0.091 |
| indirect questions | `daf4619f…` | 50 | 0 | 0.192 |
| indirect bank | `18b5e2d1…` | 72 | 0 | 0.192 |
| typo dev | `68260c01…` | 68 | 0 | 0.143 |
| misleading-cue dev | `ed3bd6aa…` | 44 | 0 | 0.214 |
| bare-locative dev | `a0d73095…` | 100 | 0 | 0.143 |
| residual misleading | `b35a4c42…` | 100 | 0 | 0.214 |
| two-name dev | `9c72a8b3…` | 120 strings | 0 | 0.235 |
| focus dev | `03dea71d…` | 132 strings | 0 | 0.250 |
| typo-edit dev | `3447aaaa…` | 96 | 0 | 0.133 |
| lastname questions | `064d7052…` | 101 | 0 | 0.167 |
| lastname bank | `626514af…` | 309 | 0 | 0.182 |

Closest frozen pair: "barmkin oxter kilnnote between Wulfrun and Swithun ask only about Swithun telephone thrid278 㐀㐁" ‖ "Between Ann and Catherine, ask only about Catherine: which telephone is filed?" Closest probe pair: "barmkin oxter 我留了生日 thrid338 㐀㐁" ‖ "我们为顾清的生日留了哪一天？"

Ask was scored with that embedder on the shipped bundle (`alpha` 0.3, abstain ridge, margin 0.1). "With the log" plays the episode's turns and earlier asks, then the question. "Empty log" writes the same facts and asks once. Both columns are episodes whose label for that pass was met. A typo hit meets the log label by returning the gold value, and meets the empty-log label by abstaining `absent_person`.

| kind | n | with the log | empty log |
|---|---:|---:|---:|
| typo_hit_utterance | 16 | 16/16 | 16/16 |
| typo_abstain_silent | 16 | 16/16 | 16/16 |
| typo_abstain_other | 16 | 16/16 | 16/16 |
| typo_hit_prior_ask | 8 | 8/8 | 8/8 |
| typo_abstain_stale | 4 | 4/4 | 4/4 |
| typo_abstain_tie | 2 | 2/2 | 2/2 |
| spelling_hit | 8 | 8/8 | 8/8 |
| nickname_hit | 6 | 6/6 | 6/6 |
| latin_extension | 2 | 2/2 | 2/2 |
| cjk_superstring | 4 | 4/4 | 4/4 |
| cjk_prefix | 2 | 2/2 | 2/2 |
| focus_hit | 4 | 4/4 | 4/4 |
| dismiss_hit | 4 | 4/4 | 4/4 |
| open_either | 4 | 4/4 | 4/4 |
| indirect_hit | 2 | 2/2 | 2/2 |
| indirect_noself | 2 | 2/2 | 2/2 |
| indirect_missing | 2 | 2/2 | 2/2 |
| total | 102 | 102/102 | 102/102 |

Failures 0. The 24 typo hits (16 after an utterance, 8 after an earlier ask) are the gain: the empty log abstains them, and the log returns the gold value. The 16 silent rows are the same surfaces with the fact written and nobody mentioned, and they abstain. A different person, a mention older than four turns, and a two-name edit tie abstain with the log filled. Exact spelling, nicknames, the focus restrictor, dismiss, an open two-name question, and the indirect hearer agree on both passes. Longer names stay abstentions while the short name is active.

## Frozen adversarial

Question sha `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`, n=327. Wording was not edited. Scores use the committed cache. The harness writes facts and asks once, so the conversational list is empty. The run matches the focus-restrictor column on `main`.

| | this run |
|---|---:|
| answerable hits | 188/216 |
| answerable false abstain | 19/216 |
| missing_rel abstained | 60/69 |
| missing_rel still answered | 9 |
| absent_person | 42/42 |
| near_dup_absent | 30/30 |
| English answerable hits | 92/103 |
| Chinese answerable hits | 96/113 |
| English false abstain | 11/103 |
| Chinese false abstain | 8/113 |
| English missing_rel abstained | 29/35 |
| Chinese missing_rel abstained | 31/34 |

`name_variant` typos stay false abstains: 8/8 English and 8/8 Chinese. Those rows do not record a turn, and a write does not license the edit. That is the same 16 the string-only attempts could not move. `near_dup_absent` stays 30/30, including the edit-1 substitutions, insertions, and the three CJK edits that share a surface with a typo label on the same file.

## Probe and surface margin

`eval_heldout` clears the store per case, writes facts, and asks. The conversational list stays empty across the re-read of the same question (`ignore_surface`, and a repeat of that string does not open another turn).

| category | n | cosine ret@1 | ask ret@1 | phone cos·ask | date cos·ask | tokens cos·ask | false cos·ask | gold lost @0.9 |
|---|---:|---|---|---|---|---|---|---|
| en_direct | 36 | 12/36 | 36/36 | 2/12 · 12/12 | 4/12 · 12/12 | 6/12 · 12/12 | — | 0/36 |
| zh_direct | 36 | 9/36 | 36/36 | 4/12 · 12/12 | 2/12 · 12/12 | 3/12 · 12/12 | — | 0/36 |
| en_indirect | 40 | 14/40 | 39/40 | 3/13 · 13/13 | 7/14 · 13/14 | 4/13 · 13/13 | — | 0/40 |
| zh_indirect | 40 | 10/40 | 38/40 | 3/13 · 12/13 | 4/14 · 14/14 | 3/13 · 12/13 | — | 0/40 |
| en_cross_turn | 37 | 8/37 | 37/37 | 3/13 · 13/13 | 4/12 · 12/12 | 1/12 · 12/12 | — | 0/37 |
| zh_cross_turn | 37 | 6/37 | 36/37 | 2/13 · 12/13 | 1/12 · 12/12 | 3/12 · 12/12 | — | 0/37 |
| en_after_distractors | 36 | 5/36 | 35/36 | 0/12 · 12/12 | 1/12 · 11/12 | 4/12 · 12/12 | — | 0/36 |
| zh_after_distractors | 36 | 6/36 | 36/36 | 0/12 · 12/12 | 2/12 · 12/12 | 4/12 · 12/12 | — | 0/36 |
| en_after_update | 36 | 12/36 | 36/36 | 6/12 · 12/12 | 2/12 · 12/12 | 4/12 · 12/12 | — | 0/36 |
| zh_after_update | 36 | 2/36 | 36/36 | 0/12 · 12/12 | 2/12 · 12/12 | 0/12 · 12/12 | — | 0/36 |
| en_unanswerable | 36 | — | — | — | — | — | 36/36 · 0/36 | — |
| zh_unanswerable | 36 | — | — | — | — | — | 36/36 · 0/36 | — |
| answerable | 370 | 84/370 | 365/370 | 23/124 · 122/124 | 29/124 · 122/124 | 32/122 · 121/122 | — | 0/370 |
| unanswerable | 72 | — | — | — | — | — | 72/72 · 0/72 | — |

Ask abstain on that run: answerable 365/370 with false abstain 0/370 (en 183/185, zh 182/185; phone 122/124, birthday 122/124, places 121/122); missing_rel 36/36; absent_person 36/36; unanswerable 72/72. Co-resident pairs 8390, cosine above 0.9 is 0, max 0.8927. Gold lost at overwrite 0.9 is 0/370.

Surface dev at margin 0.1: false abstain 0/192 English and 0/192 Chinese, missing 188/192 and 189/192, absent 36/36 both, deictic false 0/6 both.

## What stayed

Lexical hits, the name head, the alias list, CJK verb attachment, and the longer-name rejection are unchanged when the log is empty. The indirect hearer still resolves only to the unique first-person fact. Focus, dismiss, and a unique possessor still narrow two names. The 16 frozen typo false abstains stay false abstains, because that file does not record a turn. `near_dup_absent` stays 30/30 for the same reason.

## Tests

`python3 -m pytest -q`: 289 passed. The new file locks the dev sha, the Jaccard cap, literal disjointness, the cache sha, the edit rule (one active name, extensions rejected, a tie abstains), the four-turn window, and a tiny-encoder pass of all 102 episodes: a write leaves the typo abstaining, a turn or an earlier ask returns the gold, a repeat of the same question does not open another turn, and `new_session` drops the mention while the exact name still answers. Frozen numeric scores are not a pytest gate.
