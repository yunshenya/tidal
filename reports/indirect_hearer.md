# Indirect hearer descriptions

`跟前这位的电话该回读哪一条？` and `For the addressee alone, which telephone is the one to read?` name nobody in the store. Since #18 they false-abstain, 12/12, on the frozen adversarial file. `you` / `你` / `您` were already fixed there. These twelve are open descriptions. The two splits do not share a word (`addressee` / `the person being addressed` versus `跟前这位`), so a list of those strings was not the fix.

Always on. No flag. `W` and `alpha` are the same surface bundle (`820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9`, alpha 0.3). The name-head bundle and the Bob/Robert alias tuple are unchanged (22 groups, `robert` → `bob` / `bobby`). `LLMMemory` is untouched (`tidal/llm_memory.py` sha256 `36210f02048c243a77df80259b647a39e56077566bf765bf051cb55bacaacc7d`, inject weights `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`).

Before is `1cd3f08` (#23). The frozen adversarial question sha256 is `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5`. It was not used to choose the rule.

## What the store can actually resolve

Continuum has no speaker and no addressee. A slot is name, relation, time, group, and a sentence vector. The surface string is kept beside that state only when `embed_event(..., surface=)` was used. Turn-taking speaker fields are a different pipeline and `ask` does not read them.

The ask path already treats a first-person stored sentence as the participant's own fact. That is how `you` / `你` are answerable on the probe: the gold statement is `The voice contact I actually answer...` / `我真正会接...`. The same convention is the referent policy here.

An indirect hearer description resolves to a stored person only when every visible slot has a surface and exactly one of those people is written in the first person (`I` / `me` / `my` / `mine` / `myself`, or `我` not `我们`). Ranking then uses only that person's facts, the same scope #23 uses for a matched name. If that person lacks the asked relation, the ask is `missing_rel`.

It does not resolve when nobody is first person, when two people are, or when any surface is missing. A single third-person participant is not enough. The last write is not the referent. The only person who holds the asked relation is not the referent when the self lacks it. A stored name in the question still wins, including a nickname or pinyin. An unmatched name blocks the description. No surface string, and the description selects nobody: the old `absent_person` path.

The hearer class is small and closed, and it does nothing without that store gate:

* Role nouns: `addressee`, `interlocutor`, `hearer`, `listener` (plural allowed).
* A participial hearer: `the person / one / party` plus `addressed`, `spoken to`, `spoken with`, `talked to`, `talked with`, with optional `being` or `who is`.
* A facing locative after `person / one / party`, `this/that person`, or `someone sitting/standing`: `across the table/desk`, `opposite`, `in front` (not `in front of`), `facing`.
* Chinese: `跟前` / `面前` / `眼前` / `对面` / `对过` / `身边` / `身旁`, optional `的`, optional `这` / `那`, then `一位` / `个人` / `位` / `人`.

`隔壁那位`, a bare `那位`, `另一位`, `the neighbor`, and a bare `this person` do not match. `in front of` does not match: the abstain-dev sentence `You are the one in front of the porter...` stays ordinary deixis and still ranks the whole store.

The same frozen question abstains when every stored surface is third person, and it abstains again when a second person is also first person. The twelve strings are not a special case.

## What was measured and not shipped

Deixis on the frozen file, before this change: `second_indirect` is the only subtype that still false-abstains (12/12). `second_person`, `second_reflexive`, `second_polite` (`您`), `first_person`, `pronoun_unique`, and `pronoun_ambiguous` already have 0 false abstain. Some of those are ranker misses. `W` was not retuned for them.

On the abstain-dev split the sibling is also `second_indirect`, and the words differ (`interlocutor`, `对面那位`, plus one inversion). 11 of those 12 match the class above. The one that does not, and still false-abstains, is `Across the desk sits someone whose birthday is in the green diary. Which date?` Locative inversion plus indefinite `someone` is outside the pattern. It was not special-cased.

Sentence cosine does not separate the class. Cached `multilingual-e5-small` vectors, no new embedder:

| comparison | indirect | other |
|---|---|---|
| max cosine to the 12 surface-dev `you` / `你` prototypes | en min 0.852, zh min 0.863 | en absent-person max 0.843 (gap 0.009); zh absent-person max 0.886 (overlap) |
| max cosine to frozen `you` / `I` questions | en 0.898–0.940, zh 0.907–0.971 | en absent max 0.887; zh absent max 0.903; nicknames and pinyin sit in the same band (`Under Bill...` 0.924, `名册写着小洋时...` 0.933) |

The gold first-person phone statement and the other person's same-relation statement cosine 0.867. A threshold on that space would either keep the false abstains or start answering nicknames and absent people. It was not shipped. `W` and `alpha` were not fit on these questions.

Once the self is selected, the abstain ridge is not the thing that was wrong. On the twelve, the lexical cue and the ridge agree. The ridge gap is 0.389–0.892, all above `LEX_RIDGE_GAP` 0.15, and the label is the asked relation. Scoping the ranker to the self is what changes the answer. The other person is written with the same relation. Ranking the whole store, which is what a bare "do not abstain" would do, hits 5/12: English 5/6, and Chinese 0/6 because 李娜's sentence outscores 王伟 on every row. The English miss is Elizabeth's places.

## Dev bank

`tidal/question_indirect_dev_data.py`. The matcher does not import it.

Questions sha256 `daf4619f2d24db78396f2516e7a4d315bb71eefb04dba825347c6fc755874b93`. Bank sha256, including labels and store slots, `18b5e2d128fb6b626f28644b851d405581169c4befc842c96536cb84b5f6efd6`. 50 questions (en 27, zh 23): answerable 24, `absent_person` 24, `missing_rel` 2.

The positive phrasings are not the frozen strings and not the abstain-dev strings. English uses `the hearer`, `the listener`, `spoken with`, `talked to`, `across the table`, `the party opposite`, `someone sitting opposite`, `this person in front`, `the person facing`, `talked with`. Chinese uses `眼前` / `面前` / `身边` / `身旁` / `对过`, not `跟前这位` or `对面那位`. `addressee` and `interlocutor` are the same noun class and are not question strings here.

Store setups the labels use:

| setup | what is stored | label |
|---|---|---|
| `unique_self` | third person, then the self, then another person last with the same relation | answerable, referent is the self |
| `single_self` | only the self, first person | answerable |
| `no_self` | two third-person people, both with the relation | `absent_person` |
| `two_selves` | two first-person people | `absent_person` |
| `single_third` | one third-person person | `absent_person` |
| `missing_rel` | the self lacks the relation; the other person, last, has it | `missing_rel`, referent is still the self |
| `multi_fact` | the self holds two relations; the other person, last, holds the asked one too | answerable, both self rows in scope |
| `name_wins` | the question names the third-person person; the other slot is the self | answerable, referent is the named person |
| `absent_name` | a unique self, and the question also names `Ysolde` / `郗茉` | `absent_person` |
| non-hearer | unique self, asked about a neighbor, a clerk, `this person`, `隔壁那位`, `另一位`, `那位` | `absent_person` |

Names (`Saffi`, `Torvald`, `Eulalie`, `Ysolde`, `倪棠`, `卞菽`, `糜涧`, `郗茉`) and the mark words (`nephoscope`, `yaffingale`, `龖`, `爩`) are not in the probe, the surface corpus, either adversarial split, the CJK dev bank, or the nickname dev bank.

Disjointness is literal equality and token Jaccard (CJK by character, Latin words lowercased), over questions and stored surfaces (72 strings):

| against | literal | max Jaccard | pairs above 0.5 |
|---|---:|---:|---:|
| probe | 0 | 0.364 | 0 |
| surface corpus | 0 | 0.312 | 0 |
| abstain dev | 0 | 0.333 | 0 |
| frozen adversarial test | 0 | 0.400 | 0 |
| CJK dev (`5e8b86f6…`) | 0 | 0.200 | 0 |
| nickname dev (`a766f748…`) | 0 | 0.053 | 0 |
| typo dev, closed PR #24 (`68260c01…`, n=68) | 0 | 0.200 | 0 |

Sentence cosine of this bank was not computed. The strings are not in `question_adversarial_emb.npz` or `question_heldout_emb.npz`, and this run did not download `multilingual-e5-small`. The Jaccard bar the other dev tests assert is 0.50. The closest pair is 0.400, function words in `Which telephone is filed for Ysolde...` against `Which telephone was filed for Desmond?`.

## Before / after

Scored with the shipped `ask` on the cached adversarial vectors (`score_split`). No refit.

| check | before (`1cd3f08`) | after |
|---|---:|---:|
| `second_indirect` false abstain | 12/12 | 0/12 |
| `second_indirect` ask hits | 0/12 | 12/12 |
| answerable false abstain | 39/216 | 27/216 |
| answerable ask hits | 162/216 | 174/216 |
| answerable still answered | 177/216 | 189/216 |
| `missing_rel` abstained | 53/69 | 53/69 |
| `absent_person` abstained | 42/42 | 42/42 |
| near_dup absent, en | 16/16 | 16/16 |
| near_dup absent, zh | 14/14 | 14/14 |
| two_names absent | 12/12 | 12/12 |
| two_names en / zh answerable | 3/6, 3/6 | 3/6, 3/6 |
| en two-name missing relation, still answered | 3/6 | 3/6 |
| zh two-name missing relation, abstained | 6/6 | 6/6 |
| cjk_superstring abstained | 8/8 | 8/8 |
| typos | 16/16 false abstain | 16/16 false abstain |
| en nickname ask hits | 6/6 | 6/6 |
| pinyin ask hits | 8/8 | 8/8 |
| `Under Maggie, which birthday is on file?` | Margaret `1939-04-04` | Margaret `1939-04-04` |
| `Under Ms. Jane, which ordered locations is on file?` | `north271-ridge271-ford271` | same |
| en `name_variant` answerable false abstain / hits | 8 / 24 | 8 / 24 |
| other deixis false abstain | 0/60 | 0/60 |

The only rows that moved are the twelve. Each now returns the first-person person's gold value (Robert / 王伟 on that store), not the other person who holds the same relation.

Abstain-dev `second_indirect`: 11/12 now hit. `Across the desk sits someone...` stays a false abstain. `Walk the interlocutor's ordered locations off the brown card. What are they?` was already not an abstain, because of `they`. It now ranks only the unique self, and the gold is that self, so it still hits.

## Probe

No probe string and no surface-corpus string matches the hearer class (0/744, 0/1124), so this path does not scope any probe ask. `eval_heldout` on this revision:

| check | #23 | this revision |
|---|---:|---:|
| answerable ask ret@1 | 365/370 | 365/370 |
| false abstain | 0/370 | 0/370 |
| unanswerable abstain | 72/72 | 72/72 |
| gold lost at overwrite 0.9 | 0/370 | 0/370 |
| co-resident cosine > 0.9 | 0 (max 0.8927) | 0 (8390 pairs, max 0.8927) |
| en / zh | 183/185, 182/185 | 183/185, 182/185 |
| phone / birthday / places | 122/124, 122/124, 121/122 | 122/124, 122/124, 121/122 |

`en_direct` 36/36, `zh_direct` 36/36, `en_after_distractors` 35/36, `zh_after_distractors` 36/36. The other answerable categories are the #23 counts (`en_indirect` 39/40, `zh_indirect` 38/40, `en_cross_turn` 37/37, `zh_cross_turn` 36/37, `en_after_update` 36/36, `zh_after_update` 36/36).

## What stays open

`Across the desk sits someone whose birthday is in the green diary. Which date?` One locative inversion. Not worth a second pattern fitted to that sentence.

One-character typos, 16/16 still false abstain. Misleading cues are unchanged. English two-name answerable asks are still 3/6. `Ms. Jane` still matches Mary Jane through the substring head. The alias list was not extended.

A hearer question with no recorded surfaces, or with one slot missing its surface, still abstains. That is the flat store refusing to guess, not a dialogue-state tracker. A real addressee that was never written in the first person, or two participants both written in the first person, will not resolve. Those need speaker state this ask path does not have.

## Checks

`python3 -m pytest`: 249 passed, 1 skipped. The skip is `test_tiny_llama_empty_store_matches_and_script_runs`, which imports `transformers`. That package is not installed here. The other `LLMMemory` tests passed. `tidal/llm_memory.py` and the inject weights are unchanged. Probe invariants in that suite are green: false abstain 0/370, unanswerable abstain 72/72, gold lost at overwrite 0.9 is 0, and no co-resident pair has cosine above 0.9.
