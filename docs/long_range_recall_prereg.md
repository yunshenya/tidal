# Long-range recall held-out evaluation — pre-registration

Protocol `long-range-recall-held-v0`. This document is committed before any
dialogue is scored. The criteria below are locked. They will not be edited
after numbers exist. A failed hypothesis is a valid result.

This is a measurement-only draft. It does not change packet fields, live
`CUE_INTERFACE_VERSION` (stays 7), or the reconstruct bank (stays
`interface_version` 6, frozen sha256
`bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589`). It does
not retune any held floor or shipped constant. It does not add a runtime
feature flag in `tidal/`. Ablation off-switches live only inside the eval
harness.

Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live
`CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not
merged.

## Why this bank exists

The stack on main now owns #77 McGaugh arousal consolidation gain
(`604236c`), #79 sleep two-phase SWS/REM (`b5cb75e`), #81 retroactive
interference (`8e7c37a`), and #83 weak-cue spontaneous recollection
(`e0b0fb3`). Each mechanism has its own numeric fit/held gates. None of
those gates ask whether a planted event, preference *content*, or
ordinary dialogue line can still be retrieved after many later turns,
after a sleep episode, under a strong cue versus a weak cue, after an
update conflict, or under an unrelated probe. A line such as
`别说话了` is that last class: a plot / social-dialogue fact, the same
kind of write as `那天吃了抹茶蛋糕`. It is not a mute policy, not a
system instruction, not a preference-channel write, and not a gate
command. This protocol does not score subsequent speak rate and does
not ask the stack to stop talking.

This protocol is that long-range recall measurement. It is not a new
memory operator. It is not a retune.

## Frozen interface (must not move)

| name | locked value |
|---|---|
| live `CUE_INTERFACE_VERSION` | 7 |
| reconstruct `canonical_surfaces()["interface_version"]` | 6 |
| reconstruct `bank_sha256()` | `bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589` |
| `CUE_THETA` | 0.32 |
| `WEAK_FLOOR` / `WEAK_OVERLAP` / `WEAK_CEILING` | 0.10 / 0.25 / 0.32 |
| sleep floor | 0.1 |
| arousal baseline / tau / window | 0.2 / 8 / 8 |
| cue half-life / lambda | 30 / 0.3 |
| dims (live growbrain) | 192 / 64 / 32 |
| packet keys | `interface_version`, `cue_dim`, `memory_id`, `vec`, `activation`, `strength_confidence`, `valence`, `modality`, `circadian_phase`, `triggered`, `gap`, `winner_cosine`, `sleep_pressure`, `arousal`, `source`, `fingerprint`, `fusion_confidence` |

No packet field is added. `fusion_weights` stays off the cue wire. The
reconstruct bank is not rehashed.

## Public API

Every dialogue uses only main's public memory API:

- `remember` — plant a fact or a distractor
- `cue_fields` — strong cue, weak cue presentation, unrelated probe
- `offline_ticks` — no-strong-cue ticks; the weak-cue path after a weak
  presentation
- `wake_for` / `sleep_for` — process S and the sleep episode

The harness may construct a tiny CPU stub model so the suite does not
download SmolLM2 or growbrain. That stub is eval-only. It is not a
runtime backend and not a feature flag.

## Encoder geometry (harness-only, locked)

The stub maps each locked utterance to one token and one unit vector so
the strong / weak / unrelated bands are guaranteed. Live `remember` and
`cue_fields` still run. The harness does not call `store.write` over the
public API.

| utterance class | cosine with the planted target key |
|---|---|
| strong cue | 1.0 (≥ `CUE_THETA` 0.32) |
| weak cue | 0.25 (in `[0.10, 0.32)`) |
| unrelated probe | 0.0 |
| similar distractor | 0.95 (≥ retroactive tau 0.90) |
| ordinary distractor | 0.0 |

Hidden width of the stub is 64. Query and key centers are zero. This
width is the eval stub, not a change to live `cue_dim`.

## Seeds, items, N

- Seeds: `{0, 1, 2, 3, 4}` (5 seeds; cheap).
- Items per seed: 11 locked facts below (ids 0–9 plus dialogue line
  id 10). Item `k` on seed `s` is the same surface on every seed; only
  distractor ids and similar-neighbor axes vary with `s`.
- Distractor depths: `N ∈ {10, 30, 100}`.
- Per cell: 5 seeds × 11 items = 55 dialogues. Report the per-seed rate
  over the 11 items, then mean ± sample std over the 5 seeds.

### Locked target facts

`answer` is a substring of `text`, as `remember` requires. Item 0 is the
event example from the request (`那天吃了抹茶蛋糕` / `我又吃到了`).
Item 10 is the social-dialogue example (`别说话了` / `刚才她说了啥`).
Both are ordinary `kind=fact` writes. Item 10's relation is `line`, not
`command`, `mute`, `preference`, or `instruction`.

| id | name | rel | plant text | answer | strong cue | weak cue |
|---|---|---|---|---|---|---|
| 0 | hari | food | That day I ate matcha cake. | matcha cake | That day I ate matcha cake. | I ate some again. |
| 1 | nora | seat | I prefer sitting by the window. | the window | I prefer sitting by the window. | I sat down again. |
| 2 | ada | pet | Ada's cat is named Mochi. | Mochi | Ada's cat is named Mochi. | The cat came back. |
| 3 | tess | meet | The meeting moved to Thursday. | Thursday | The meeting moved to Thursday. | When did it move? |
| 4 | cole | drink | I cannot drink coffee after noon. | coffee | I cannot drink coffee after noon. | I already had some. |
| 5 | bea | mug | The blue mug lives on the second shelf. | blue mug | The blue mug lives on the second shelf. | Where did the mug go? |
| 6 | remy | train | We booked the late train home. | late train | We booked the late train home. | The train again. |
| 7 | mio | soup | My favorite soup is miso. | miso | My favorite soup is miso. | I had soup again. |
| 8 | fern | plant | The plant on the desk is a fern. | fern | The plant on the desk is a fern. | The plant is still there. |
| 9 | kit | key | I left the key under the mat. | under the mat | I left the key under the mat. | Where is the key? |
| 10 | lin | line | 别说话了 | 别说话了 | 别说话了 | 刚才她说了啥 |

Chinese gloss, item 0 only, not passed to `remember`: plant
`那天吃了抹茶蛋糕`; weak cue `我又吃到了`. The English pair is the
scored surface so `ByteTokenizer`-range tests and the utterance stub
share one bank. Item 10 plants and probes the Chinese surfaces
directly. The weak cue asks what she said just then. That is a recall
probe, not a mute confirmation.

### Dialogue lines are not mute commands

Item 10 is in the held-out set as an ordinary conversational fact.

- Plant `别说话了` with `remember(..., kind="fact", rel="line")`.
- After `N` distractors, probe with the strong surface and with
  `刚才她说了啥` on the weak path.
- Score hit / miss in the same `N ×` cue-strength buckets as ids 0–9.
- Do not add a criterion that subsequent speak rate must drop.
- Do not mark the write high-priority, and do not put it on a command
  or preference channel.
- Immediate mute / stop-talking as a live dialogue reaction is out of
  scope for this pull request.

### Locked update-conflict restatements (B)

Same `(name, rel)` as the plant. `remember` of B must overwrite A.

| id | restatement B | answer B |
|---|---|---|
| 0 | That day I ate strawberry cake. | strawberry cake |
| 1 | I prefer sitting by the aisle. | the aisle |
| 2 | Ada's cat is named Sable. | Sable |
| 3 | The meeting moved to Friday. | Friday |
| 4 | I cannot drink tea after noon. | tea |
| 5 | The green mug lives on the second shelf. | green mug |
| 6 | We booked the early train home. | early train |
| 7 | My favorite soup is broth. | broth |
| 8 | The plant on the desk is a fig. | fig |
| 9 | I left the key under the pot. | under the pot |
| 10 | 先等一下 | 先等一下 |

### Locked unrelated probes

One probe per item. No planted match. Vector cosine with every planted
key in that dialogue is 0 by construction.

| id | unrelated text |
|---|---|
| 0 | The printer is jammed again. |
| 1 | Helsinki is cold in March. |
| 2 | The violin needs new rosin. |
| 3 | A red kite hangs in the hall. |
| 4 | The library closes at six. |
| 5 | Salt spilled on the counter. |
| 6 | The radio plays a waltz. |
| 7 | Fog covers the south dock. |
| 8 | Someone moved the chess clock. |
| 9 | The oven beeped twice already. |
| 10 | The kettle clicked off. |

### Locked distractors

Ordinary distractor `i` on seed `s`, item `k`, turn `t` (`t` in
`1..N`):

`Distractor s-k-t notes the weather was mild.`

Answer span: `mild`. Name `d{s:02d}{k:02d}{t:03d}`, rel `chatnote`,
kind `fact`. Vector is a unique axis, cosine 0 with the target.

Similar distractors (so #81 can fire): two per recall and unrelated
dialogue, at turns `1` and `max(2, N // 3)`. Text
`Near s-k-t shares the same day.` Answer `same day`. Name
`n{s:02d}{k:02d}{t:03d}`, rel `nearnote`. Vector cosine with the
target key is 0.95. Conflict dialogues skip similar distractors so the
overwrite of B is the only same-slot write.

## Realistic schedule (locked)

Process S starts at the floor 0.1. Circadian phase stays 0 (the stub
does not tick the store clock). Target writes use `experience: +1`.
Arousal is set to 0.8 only for that target `remember`, then restored to
0.2, so McGaugh gain is above 1 on the planted slot and later
distractors stay at baseline.

Shared prefix, every `N`:

1. Set arousal 0.8. `remember(target, experience=+1)`. Restore arousal
   0.2.
2. `wake_for(1.0)`.

Distractor loop, turn `i = 1 .. N`:

3. `remember(distractor_i)` (similar neighbor when the turn is a
   similar turn and the family is recall or unrelated).
4. If `i % 4 == 0`: `wake_for(0.5)`.
5. If `N >= 30` and `i == N // 2`: `wake_for(4.0)` then
   `sleep_for(2.0)`. That is the sleep episode. High process S at onset
   is the SWS-like pass; `sleep_for` also runs prediction-error replay
   and `floor(2)` offline ticks; the post-sleep pressure after
   `wake_for(4)` plus `sleep_for(2)` is the REM-or-none exit the live
   scheduler already owns.
6. If `N >= 100` and `i == 75`: `offline_ticks(16)` (one spontaneous
   period).

Post-loop:

7. If `N == 10`: `offline_ticks(8)` (short offline; period 16 may or
   may not fire depending on the tick index and process S).
8. If `N == 30`: `offline_ticks(4)` (the sleep episode already ran).
9. If `N == 100`: no extra ticks (the `i == 75` block already ran 16).

Why this schedule can matter:

- `N = 10` is short-range plus a little offline. No sleep episode.
- `N = 30` inserts one sleep episode at turn 15, so SWS age and
  spontaneous ticks inside `sleep_for` can move the planted slot.
- `N = 100` adds a later `offline_ticks(16)` at turn 75, so #83
  weak-overlap spontaneous recall can fire in the middle of a long
  distractor stream, not only at probe time.

Strong and weak probes each replay the dialogue to this post-schedule
state on a fresh memory object. They do not share a store, so the
strong cue cannot plant a context fragment that the weak probe then
reads.

### Probe actions

**Strong probe.** `cue_fields(strong_cue)`. Hit when the last token has
`memory_id` equal to the planted target slot, `triggered` is true,
`source == "cue"`, and `winner_cosine >= 0.32`.

**Weak probe.** `cue_fields(weak_cue)` first. That call must not take
the strong path: if `triggered` is true or `source == "cue"`, the
dialogue is a weak-band miss (not a hit). Then `offline_ticks(16)`.
Hit when at least one of those ticks has `source == "spontaneous"`,
`memory_id` equal to the planted target slot, and `winner_cosine` in
`[0.10, 0.32)`.

**Conflict probe.** After plant A, the first `N // 2` ordinary
distractors, `remember(B)` on the same `(name, rel)`, the remaining
distractors, and the same `N`-schedule, run the strong probe. Newest-
take when the winning slot's `answer` is B's answer and is not A's
answer.

**Unrelated probe.** After the recall-family plant and schedule,
`cue_fields(unrelated_text)`. False recall when `triggered` is true or
`memory_id >= 0`.

Recall@1 is the strong-probe hit. Hit rate is the same indicator,
bucketed by `N ×` cue strength (strong vs weak). Conflict newest-take
rate is the conflict-probe hit. False-recall rate is the unrelated-
probe false-positive rate.

## Conditions (ablations)

All off-switches are monkeypatches or test-side overrides inside the
eval harness. None of them exist on `LLMMemory` as a constructor flag,
environment variable, or `remember` argument.

| condition | what the harness turns off |
|---|---|
| `full` | nothing |
| `off_77` | `LLMMemory._apply_consolidation_gain` stores 1.0; McGaugh gain is not written |
| `off_79` | `LLMMemory._sws_pass` and `LLMMemory._rem_pass` are no-ops |
| `off_81` | `tidal.retroactive.apply_retroactive` returns 0 and does not age neighbors |
| `off_83` | `tidal.weak_cue.weak_winner` returns `None` (salience offline ranking stays) |
| `baseline` | all four of the above, plus `_reconsolidate_store` returns `[]`, `write_keep` is 1.0, and after every distractor `remember` every slot's `strength` is multiplied by `0.5 ** (1/30)` |

`baseline` is pure time-decay memory: cosine retrieve, strength decay
per turn with the shipped 30-step half-life, no McGaugh gain, no SWS /
REM, no retroactive age-hit, no prediction-error rewrite, no weak-cue
overlap, no encode/consolidate write keep.

## Metrics

For each condition, seed, and `N`:

- `recall_at_1_strong` — strong-probe hits / 11 items
- `hit_rate_weak` — weak-probe hits / 11 items
- `conflict_newest_take` — conflict newest-take / 11 items
- `false_recall` — unrelated false positives / 11 items

Then mean ± sample standard deviation across the 5 seeds. A cell with
one seed (the pytest smoke) is not a held cell.

Wilson 95% intervals on the pooled 50-dialogue counts are reported as
descriptive columns. They are not extra gates.

## Pass / fail criteria (locked before any run)

Validity gates must pass or the report is incomplete. Hypothesis gates
are scored as written. They are not moved if a number misses.

### Validity (must pass)

1. **Frozen interface.** Live `CUE_INTERFACE_VERSION` is 7. Reconstruct
   `canonical_surfaces()["interface_version"]` is 6. Reconstruct
   `bank_sha256()` equals
   `bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589`.
   Cue export keys are exactly the locked packet set. No new field.
2. **No runtime switch.** `tidal/llm_memory.py`, `tidal/consolidation.py`,
   `tidal/sleep_phases.py`, `tidal/retroactive.py`, and `tidal/weak_cue.py`
   do not gain an eval flag, environment read, or `remember` argument
   used to disable #77 / #79 / #81 / #83. Off-switches exist only in
   the harness.
3. **Band integrity.** Every counted strong hit has `source == "cue"`
   and `winner_cosine >= 0.32`. Every counted weak hit has
   `source == "spontaneous"` and `winner_cosine` in `[0.10, 0.32)`.
   A weak presentation that triggers the cue path is a miss.
4. **Schedule integrity.** `N=10` has no `sleep_for`. `N=30` calls
   `sleep_for(2.0)` once, at turn 15. `N=100` calls `sleep_for(2.0)`
   once at turn 50 and `offline_ticks(16)` once at turn 75. The
   documented `wake_for` durations are the ones the harness calls.
5. **Ablation integrity.** `off_77` / `off_79` / `off_81` / `off_83`
   each disable only that mechanism. `baseline` disables all four plus
   prediction-error rewrite and encode/consolidate keep, and applies
   the locked turn decay.
6. **Report completeness.** `reports/long_range_recall.json` and
   `reports/long_range_recall.md` contain every condition × `N` ×
   {strong, weak, conflict, unrelated} cell as mean ± std over seeds
   `{0,1,2,3,4}`. The docs page states each hypothesis pass/fail
   honestly.
7. **Suite.** Existing pytest stay green. Prior count on this checkout
   was 302 passed / 1 skipped. New tests may raise the passed count.
   They must not flip the skip or fail an old test.

### Hypotheses (may fail; do not retune to pass)

These compare seed-means. Ties fail the strict inequalities. No
confidence interval is required for the gate. The numbers are not
used to change a constant.

1. **H1 strong short-range.** `full` mean `recall_at_1_strong` at
   `N=10` is strictly greater than `baseline` mean
   `recall_at_1_strong` at `N=10`.
2. **H2 strong long-range.** `full` mean `recall_at_1_strong` at
   `N=100` is strictly greater than `baseline` mean
   `recall_at_1_strong` at `N=100`.
3. **H3 weak long-range needs #83.** `full` mean `hit_rate_weak` at
   `N=100` is strictly greater than `off_83` mean `hit_rate_weak` at
   `N=100`.
4. **H4 newest-take.** `full` mean `conflict_newest_take`, averaged
   over the three `N` seed-means, is at least 0.80.
5. **H5 false recall.** `full` mean `false_recall`, averaged over the
   three `N` seed-means, is at most 0.05.

H1 and H2 can fail if orthogonal cosine retrieve already finds the
target without tidal mechanisms. That is an allowed outcome. It is
not a reason to add overlap, change theta, or drop the baseline
decay.

## What this run does not do

- It does not score the reconstruct held bank.
- It does not rescore cue-triggered floors, sleep gates, or arousal
  gates.
- It does not change `CUE_THETA`, half-life, lambda, McGaugh span,
  SWS/REM keeps, retroactive tau, or the weak band.
- It does not merge.

## Held result

The held bank was not scored.

The hypothesis gates were not scored.
