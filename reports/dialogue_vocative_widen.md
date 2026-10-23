# Wider vocative hearers

Titles, particles, and a name followed by `you` / `你` / `您`.
The rule change is only inside `vocative_hearer`. `name_spans` is unchanged, so `老闵昭的电话` stays a gap on the ask path.
Listener labels on the fresh bank were committed in `b6e9549` before this score. A miss is not relabeled.

## Development (#36 positive slice)

The 26 `pos_voc` misses at `8a3b881` are the rows this change is for. After it:

| slice | state | empty | listener | false accepts |
|---|---:|---:|---:|---:|
| `pos_voc` | 48/48 | 48/48 | 48/48 | 0 |
| `neg_near` | 20/20 | 20/20 | 20/20 | 0 |
| `pos_keep` | 48/48 | 48/48 | 44/48 | 0 |
| 42-ask bank | 42/42 | 42/42 | 42/42 | 0 |
| 94-ask expansion | 94/94 | 94/94 | 94/94 | 0 |
| union of those three, 252 asks | 252/252 | 252/252 | 248/252 | 0 |

#35 gated rows stay 106/106 and ordinary talk 52/52, false accepts 0. The four listener misses that remain are the window, below. No encoder miss on this slice.

## Five acknowledgements, then he / 他

`TURN_WINDOW` is 4 and was not changed. A mention is kept while its turn is greater than `turn - 4`. The anchor is still visible on the next three turns and drops when the fourth later turn opens. Five acknowledgements and then `he` / `他` are already outside the window, so the pronoun does not restamp. These four rows stay misses:

- `pos_keep_en_he6a_phone`: Ranulf rested the walnut hull by the dye pot after the bench was wiped. Then five quiet turns. `He pointed at the walnut hull near the dye pot.` `Which telephone did he leave with the walnut hull near the dye pot?`
- `pos_keep_en_he6a_birthday`: Ranulf set the clover tow beside the heckle pin and stepped back. Then five quiet turns. `He folded the clover tow once the heckle pin cooled.` `Which birthday did he mark beside the clover tow and the heckle pin?`
- `pos_keep_zh_ta6a_phone`: 祁松把核桃壳挪到染锅边上，擦了擦手就停住。 Then five quiet turns. `他指了指染锅旁边的核桃壳。` `他的电话落在核桃壳挨着染锅的哪一栏？`
- `pos_keep_zh_ta6a_birthday`: 祁松把苜蓿缕搁到栉钉旁边，然后退开半步。 Then five quiet turns. `他把苜蓿缕折好，栉钉已经凉了。` `他的生日收在苜蓿缕同栉钉一块的哪一栏？`

`you` / `你` / `您` after a vocative still hit across those quiet turns, because the hearer field stays set and second person does not need the mention in the window.

## Fresh held-out

Titles, particles, and a name followed by `you` / `你` / `您`.
Listener labels were committed before this score. A miss is not relabeled.
`TURN_WINDOW` stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15.
No enable flag. `LLMMemory` is untouched. `name_spans` is unchanged.
The frozen file `cce20b6b…` is not an input.

Episodes 112. Scored asks 112.
Questions `6ff0bc0811dfa57a8a5d92c8c92d1b05cd02073803860df51d66993f3617ba86`.
Bank `3f518ce41813a00357788ba063f36244fb53855280b3786da31f20313374ea1c` (2432 strings).

| against | sha | n | literal | max Jaccard |
|---|---|---:|---:|---:|
| probe | — | 744 | 0 | 0.250 |
| surface | — | 1124 | 0 | 0.286 |
| abstain_dev | 26889048ad93… | 381 | 0 | 0.207 |
| frozen | cce20b6bf29f… | 381 | 0 | 0.263 |
| cjk | 5e8b86f694d8… | 201 | 0 | 0.200 |
| nickname | a766f748cfc2… | 43 | 0 | 0.133 |
| typo | 68260c01d5f1… | 68 | 0 | 0.150 |
| indirect_questions | daf4619f2d24… | 50 | 0 | 0.250 |
| indirect_bank | 18b5e2d128fb… | 72 | 0 | 0.250 |
| cue | ed3bd6aadfa9… | 44 | 0 | 0.212 |
| bare | a0d7309555ec… | 100 | 0 | 0.227 |
| residual | b35a4c42560a… | 100 | 0 | 0.259 |
| two_name | 9c72a8b3f65e… | 120 | 0 | 0.194 |
| focus | 03dea71d6a47… | 132 | 0 | 0.235 |
| typo_edit | 3447aaaa0b64… | 96 | 0 | 0.111 |
| lastname_questions | 064d7052e2ca… | 101 | 0 | 0.136 |
| lastname_bank | 626514af8f1f… | 309 | 0 | 0.136 |
| dialogue_dev_questions | 11bc04125694… | 102 | 0 | 0.150 |
| dialogue_dev_bank | 69bd3ed63276… | 340 | 0 | 0.188 |
| dialogue_heldout_questions | 99acea85195e… | 118 | 0 | 0.150 |
| dialogue_heldout_bank | e09fbcc043fd… | 538 | 0 | 0.150 |
| deixis_dev_questions | 4c582b14b181… | 48 | 0 | 0.214 |
| deixis_dev_bank | 1f9674cb2323… | 216 | 0 | 0.214 |
| deixis_heldout_questions | d414c44654db… | 136 | 0 | 0.107 |
| deixis_heldout_bank | 901390f59988… | 670 | 0 | 0.107 |
| dialogue_real_questions | 117ef589bb02… | 142 | 0 | 0.259 |
| dialogue_real_bank | 0be42845836e… | 1032 | 0 | 0.346 |
| keepalive_42_questions | 092252c79252… | 42 | 0 | 0.219 |
| keepalive_42_bank | 694fed5e12b4… | 509 | 0 | 0.227 |
| keepalive_expand_questions | 9af36973b625… | 94 | 0 | 0.333 |
| keepalive_expand_bank | 30af4bb72c1f… | 1176 | 0 | 0.333 |
| keepalive_pos_questions | 322afe2bba89… | 116 | 0 | 0.280 |
| keepalive_pos_bank | b32f13929119… | 2546 | 0 | 0.310 |

Closest pair is 0.346 against dialogue_real_bank.
`条子上写着：「容昭啊，把竹箍留在桶板边。」 桶号2291`
`闵昭，霍涟和霍琏都把条子留在粉笔板边上。`

## By kind

| | n | state rule | empty rule | listener | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| pos | 68 | 68/68 | 68/68 | 68/68 | 0 | 0 |
| neg | 44 | 44/44 | 44/44 | 44/44 | 0 | 0 |
| total | 112 | 112/112 | 112/112 | 112/112 | 0 | 0 |

## By language

| | n | state rule | empty rule | listener | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| en | 56 | 56/56 | 56/56 | 56/56 | 0 | 0 |
| zh | 56 | 56/56 | 56/56 | 56/56 | 0 | 0 |
| total | 112 | 112/112 | 112/112 | 112/112 | 0 | 0 |

## By shape

| | n | state rule | empty rule | listener | state FA | empty FA |
|---|---:|---:|---:|---:|---:|---:|
| neg_en_dr_said_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_hey | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_medial | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_period | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_quote | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_title_object | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_title_subject | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_two_heads | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_two_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_unknown_title | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_en_your | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_bare_a | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_bare_ne | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_bare_ya | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_lao_full_subject | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_medial | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_nimen | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_quote | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_shared_surname | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_teacher_said | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_two_heads | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| neg_zh_unknown_title | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_bare_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_bare_you_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_dr_bang | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_dr_comma | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_dr_comma_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_dr_you_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_miss_bang_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_miss_comma | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_miss_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_mr_comma | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_mr_comma_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_mr_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_mrs_comma | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_mrs_you | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_ms_colon | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_ms_comma | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_en_ms_you_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_a_sur | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_bare_ni | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_bare_nin | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_da_sur | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_full_teacher | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_lao_full | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_lao_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_lao_sur | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_lao_ya | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_ne_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_ni_later | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_particle_a | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_particle_ne | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_particle_ya | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_sur_teacher | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_teacher_a | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| pos_zh_xiao_sur | 2 | 2/2 | 2/2 | 2/2 | 0 | 0 |
| total | 112 | 112/112 | 112/112 | 112/112 | 0 | 0 |

## Listener misses

None.

## Encoder misses

None. State and empty matched the replayed rule on every ask.

## Floors

Re-run after the vocative change. `name_spans`, `TURN_WINDOW`, `ABSTAIN_MARGIN` (0.1), and `LEX_RIDGE_GAP` (0.15) were not edited. The adversarial report was written under `/tmp/floors`, not over `reports/ask_abstain_adversarial.md`.

| floor | result |
|---|---|
| probe answerable | 365/370, false abstain 0/370 |
| probe unanswerable | 72/72 abstained |
| probe by language | en 183/185, zh 182/185 |
| probe by relation | phone 122/124, birthday 122/124, places 121/122 |
| probe store | gold lost at cosine 0.9: 0; co-resident pairs 8390; pairs above 0.9: 0; max cosine 0.8927 |
| frozen `cce20b6b…` | answerable hits 188/216, false abstain 19 (en 11 = name_variant 8 + compact 2 + misleading 1; zh 8 name_variant) |
| frozen unanswerable | missing_rel abstained 60/69, absent_person 42/42, near_dup_absent 30/30 |
| dialogue held-out `--score` | gated 108/108 with the log and 108/108 empty, misses 0, false accepts 0 |
| dialogue held-out lag | 4/10 with the log, 10/10 empty (not gated) |
| #35 | gated 106/106, ordinary talk 52/52, false accepts 0 |
| deixis dev | 48/48 both passes |
| deixis held-out | 128/136 with the log (`addressee_extra` 0/8), empty 136/136, false accepts 0 |
| keep-alive union | 248/252 listener, false accepts 0; #36 `neg_near` 20/20 |
