# Adversarial ask abstain

Shipped flat-row `ask` on a held-out set written apart from the cue list,
the probe, and the surface fit/dev phrasing. Not a pass/fail gate.

## Disjointness

| split | against | literal | max token Jaccard | closest Jaccard pair | max sentence cosine | closest cosine pair |
|---|---|---:|---:|---|---:|---|
| test | probe | 0 | 0.455 | The birthday of mine, which date is kept? ‖ Past the decoys, which date is Nico's birthday? | 0.939 | 我真正会接的语音联络写在我的便签上。 ‖ 我的电话号码是我真正会接的那一部。 |
| test | surface | 0 | 0.417 | Roughly which birthday does the file keep for Margaret? ‖ Which locations does the roster keep for Arthur? | 0.941 | Ignore the birthday feast and say how to get through to Catherine. ‖ Ignore the feast and keep the birthday Beatrice claims. |

## Sizes

test: n=327 sha256=`cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`

| slice | n |
|---|---:|
| bare_cue | 16 |
| compact_form | 26 |
| deixis | 72 |
| misleading_cue | 24 |
| name_variant | 75 |
| near_dup_absent | 30 |
| relation_paraphrase | 48 |
| two_names | 36 |

by language {'en': 160, 'zh': 167}, by label {'answerable': 216, 'missing_rel': 69, 'absent_person': 42}

## Ask

Abstained: `ask` returned no fact (`missing_rel` or `absent_person`).
Still answered: a stored fact was returned.
Ask hits: answerable and the decode is the gold value (either gold, when a pronoun could be either person).
False abstain: answerable and abstained.
On an unanswerable row, still answered is the missed abstain. False abstain stays zero there by the label.

### test

| slice | lang | label | n | abstained | still answered | ask hits | false abstain |
|---|---|---|---:|---:|---:|---:|---:|
| bare_cue | en | answerable | 4 | 1 | 3 | 3 | 1 |
| bare_cue | en | missing_rel | 4 | 3 | 1 | 0 | 0 |
| bare_cue | zh | answerable | 4 | 1 | 3 | 3 | 1 |
| bare_cue | zh | missing_rel | 4 | 2 | 2 | 0 | 0 |
| compact_form | en | answerable | 7 | 2 | 5 | 5 | 2 |
| compact_form | en | missing_rel | 7 | 5 | 2 | 0 | 0 |
| compact_form | zh | answerable | 6 | 0 | 6 | 6 | 0 |
| compact_form | zh | missing_rel | 6 | 5 | 1 | 0 | 0 |
| deixis | en | answerable | 36 | 6 | 30 | 30 | 6 |
| deixis | zh | answerable | 36 | 6 | 30 | 21 | 6 |
| misleading_cue | en | answerable | 6 | 3 | 3 | 3 | 3 |
| misleading_cue | en | missing_rel | 6 | 4 | 2 | 0 | 0 |
| misleading_cue | zh | answerable | 6 | 4 | 2 | 2 | 4 |
| misleading_cue | zh | missing_rel | 6 | 3 | 3 | 0 | 0 |
| name_variant | en | answerable | 32 | 17 | 15 | 15 | 17 |
| name_variant | zh | answerable | 43 | 34 | 9 | 9 | 34 |
| near_dup_absent | en | absent_person | 16 | 16 | 0 | 0 | 0 |
| near_dup_absent | zh | absent_person | 14 | 10 | 4 | 0 | 0 |
| relation_paraphrase | en | answerable | 12 | 0 | 12 | 12 | 0 |
| relation_paraphrase | en | missing_rel | 12 | 10 | 2 | 0 | 0 |
| relation_paraphrase | zh | answerable | 12 | 0 | 12 | 12 | 0 |
| relation_paraphrase | zh | missing_rel | 12 | 12 | 0 | 0 | 0 |
| two_names | en | absent_person | 6 | 0 | 6 | 0 | 0 |
| two_names | en | answerable | 6 | 0 | 6 | 3 | 0 |
| two_names | en | missing_rel | 6 | 3 | 3 | 0 | 0 |
| two_names | zh | absent_person | 6 | 0 | 6 | 0 | 0 |
| two_names | zh | answerable | 6 | 0 | 6 | 3 | 0 |
| two_names | zh | missing_rel | 6 | 6 | 0 | 0 | 0 |

| slice | lang | subtype | label | n | abstained | still answered | ask hits | false abstain |
|---|---|---|---|---:|---:|---:|---:|---:|
| bare_cue | en | bare | answerable | 4 | 1 | 3 | 3 | 1 |
| bare_cue | en | bare | missing_rel | 4 | 3 | 1 | 0 | 0 |
| bare_cue | zh | bare | answerable | 4 | 1 | 3 | 3 | 1 |
| bare_cue | zh | bare | missing_rel | 4 | 2 | 2 | 0 | 0 |
| compact_form | en | lexical | answerable | 3 | 0 | 3 | 3 | 0 |
| compact_form | en | lexical | missing_rel | 3 | 3 | 0 | 0 | 0 |
| compact_form | en | nocue | answerable | 4 | 2 | 2 | 2 | 2 |
| compact_form | en | nocue | missing_rel | 4 | 2 | 2 | 0 | 0 |
| compact_form | zh | lexical | answerable | 3 | 0 | 3 | 3 | 0 |
| compact_form | zh | lexical | missing_rel | 3 | 3 | 0 | 0 | 0 |
| compact_form | zh | nocue | answerable | 3 | 0 | 3 | 3 | 0 |
| compact_form | zh | nocue | missing_rel | 3 | 2 | 1 | 0 | 0 |
| deixis | en | first_person | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | en | pronoun_ambiguous | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | en | pronoun_unique | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | en | second_indirect | answerable | 6 | 6 | 0 | 0 | 6 |
| deixis | en | second_person | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | en | second_reflexive | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | zh | first_person | answerable | 6 | 0 | 6 | 5 | 0 |
| deixis | zh | pronoun_ambiguous | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | zh | pronoun_unique | answerable | 6 | 0 | 6 | 6 | 0 |
| deixis | zh | second_indirect | answerable | 6 | 6 | 0 | 0 | 6 |
| deixis | zh | second_person | answerable | 6 | 0 | 6 | 2 | 0 |
| deixis | zh | second_polite | answerable | 6 | 0 | 6 | 2 | 0 |
| misleading_cue | en | cue_birthday | answerable | 2 | 2 | 0 | 0 | 2 |
| misleading_cue | en | cue_birthday | missing_rel | 2 | 0 | 2 | 0 | 0 |
| misleading_cue | en | cue_phone | answerable | 2 | 0 | 2 | 2 | 0 |
| misleading_cue | en | cue_phone | missing_rel | 2 | 2 | 0 | 0 | 0 |
| misleading_cue | en | cue_places | answerable | 2 | 1 | 1 | 1 | 1 |
| misleading_cue | en | cue_places | missing_rel | 2 | 2 | 0 | 0 | 0 |
| misleading_cue | zh | cue_birthday | answerable | 2 | 2 | 0 | 0 | 2 |
| misleading_cue | zh | cue_birthday | missing_rel | 2 | 1 | 1 | 0 | 0 |
| misleading_cue | zh | cue_phone | answerable | 2 | 1 | 1 | 1 | 1 |
| misleading_cue | zh | cue_phone | missing_rel | 2 | 1 | 1 | 0 | 0 |
| misleading_cue | zh | cue_places | answerable | 2 | 1 | 1 | 1 | 1 |
| misleading_cue | zh | cue_places | missing_rel | 2 | 1 | 1 | 0 | 0 |
| name_variant | en | extra_space | answerable | 3 | 0 | 3 | 3 | 0 |
| name_variant | en | honorific_full | answerable | 3 | 0 | 3 | 3 | 0 |
| name_variant | en | honorific_short | answerable | 3 | 3 | 0 | 0 | 3 |
| name_variant | en | lowercase | answerable | 4 | 0 | 4 | 4 | 0 |
| name_variant | en | missing_space | answerable | 1 | 0 | 1 | 1 | 0 |
| name_variant | en | nickname | answerable | 6 | 6 | 0 | 0 | 6 |
| name_variant | en | possessive | answerable | 4 | 0 | 4 | 4 | 0 |
| name_variant | en | typo | answerable | 8 | 8 | 0 | 0 | 8 |
| name_variant | zh | extra_space | answerable | 4 | 0 | 4 | 4 | 0 |
| name_variant | zh | honorific_full | answerable | 2 | 0 | 2 | 2 | 0 |
| name_variant | zh | honorific_short | answerable | 4 | 4 | 0 | 0 | 4 |
| name_variant | zh | nickname | answerable | 8 | 8 | 0 | 0 | 8 |
| name_variant | zh | no_surname | answerable | 6 | 6 | 0 | 0 | 6 |
| name_variant | zh | pinyin | answerable | 8 | 8 | 0 | 0 | 8 |
| name_variant | zh | possessive | answerable | 3 | 0 | 3 | 3 | 0 |
| name_variant | zh | typo | answerable | 8 | 8 | 0 | 0 | 8 |
| near_dup_absent | en | edit1_del | absent_person | 1 | 1 | 0 | 0 | 0 |
| near_dup_absent | en | edit1_ins | absent_person | 1 | 1 | 0 | 0 | 0 |
| near_dup_absent | en | edit1_sub | absent_person | 4 | 4 | 0 | 0 | 0 |
| near_dup_absent | en | latin_hyphen | absent_person | 3 | 3 | 0 | 0 | 0 |
| near_dup_absent | en | latin_prefix_letter | absent_person | 4 | 4 | 0 | 0 | 0 |
| near_dup_absent | en | latin_space | absent_person | 3 | 3 | 0 | 0 | 0 |
| near_dup_absent | zh | cjk_edit1 | absent_person | 3 | 3 | 0 | 0 | 0 |
| near_dup_absent | zh | cjk_middle | absent_person | 3 | 3 | 0 | 0 | 0 |
| near_dup_absent | zh | cjk_superstring | absent_person | 8 | 4 | 4 | 0 | 0 |
| relation_paraphrase | en | idiom | answerable | 3 | 0 | 3 | 3 | 0 |
| relation_paraphrase | en | idiom | missing_rel | 3 | 2 | 1 | 0 | 0 |
| relation_paraphrase | en | indirect | answerable | 9 | 0 | 9 | 9 | 0 |
| relation_paraphrase | en | indirect | missing_rel | 9 | 8 | 1 | 0 | 0 |
| relation_paraphrase | zh | idiom | answerable | 5 | 0 | 5 | 5 | 0 |
| relation_paraphrase | zh | idiom | missing_rel | 5 | 5 | 0 | 0 | 0 |
| relation_paraphrase | zh | indirect | answerable | 7 | 0 | 7 | 7 | 0 |
| relation_paraphrase | zh | indirect | missing_rel | 7 | 7 | 0 | 0 | 0 |
| two_names | en | absent_mentioned_with_stored | absent_person | 6 | 0 | 6 | 0 | 0 |
| two_names | en | both_have | answerable | 6 | 0 | 6 | 3 | 0 |
| two_names | en | other_has | missing_rel | 6 | 3 | 3 | 0 | 0 |
| two_names | zh | absent_mentioned_with_stored | absent_person | 6 | 0 | 6 | 0 | 0 |
| two_names | zh | both_have | answerable | 6 | 0 | 6 | 3 | 0 |
| two_names | zh | other_has | missing_rel | 6 | 6 | 0 | 0 | 0 |
