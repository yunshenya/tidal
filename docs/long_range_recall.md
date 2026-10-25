# Long-range recall held-out evaluation

Measurement only. Protocol `long-range-recall-held-v0`. This page does
not change packet fields, live `CUE_INTERFACE_VERSION` (stays 7), or the
reconstruct bank (`interface_version` 6, frozen sha256
`bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589`).
It does not retune a held floor or a shipped constant. Ablation
off-switches live only in the eval harness.

Pre-registration, committed before any dialogue was scored:
[long_range_recall_prereg.md](long_range_recall_prereg.md).

Scored-once reports: [reports/long_range_recall.md](../reports/long_range_recall.md)
and [reports/long_range_recall.json](../reports/long_range_recall.json).

The pull request stays a draft and is not merged.

## Dialogue lines are ordinary facts

Item 10 plants `别说话了` with `remember(..., kind="fact", rel="line")`.
That is the same class of write as item 0 (`那天吃了抹茶蛋糕` /
`That day I ate matcha cake.`). It is not a mute policy, not a system
instruction, not a preference-channel write, and not a gate command.

After `N` distractors the probes are ordinary recall cues: the strong
surface `别说话了`, and the weak surface `刚才她说了啥`. Hit / miss
uses the same `N ×` cue-strength buckets as ids 0–9. The conflict
restatement is `先等一下` on the same `(name, rel)`.

This page does not score subsequent speak rate. Immediate mute /
stop-talking as a live dialogue reaction is out of scope.

## Frozen interface

| name | locked value |
|---|---|
| live `CUE_INTERFACE_VERSION` | 7 |
| reconstruct `canonical_surfaces()["interface_version"]` | 6 |
| reconstruct `bank_sha256()` | `bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589` |
| item-bank sha256 | `b41fd9409f37b4af1f2f4c18b309f054fb110558474e26ef69c8d9302a4102ba` |
| seeds | `{0, 1, 2, 3, 4}` |
| items | 11 (ids 0–10) |
| `N` | `{10, 30, 100}` |
| dialogues per cell | 55 |

Band integrity: true. Schedule integrity: true. Git of the scored
revision: `29ca30d3f640a43a70ff99255be90c131ba683bd`. Elapsed 387.754
seconds.

## Cells (mean ± std over 5 seeds)

| condition | N | recall@1 strong | hit rate weak | newest-take | false recall |
|---|---|---|---|---|---|
| full | 10 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| full | 30 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| full | 100 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_77 | 10 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_77 | 30 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_77 | 100 | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_79 | 10 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_79 | 30 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_79 | 100 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_81 | 10 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_81 | 30 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_81 | 100 | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_83 | 10 | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_83 | 30 | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| off_83 | 100 | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| baseline | 10 | 0.000 ± 0.000 (0/55) | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| baseline | 30 | 0.000 ± 0.000 (0/55) | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |
| baseline | 100 | 0.000 ± 0.000 (0/55) | 0.000 ± 0.000 (0/55) | 1.000 ± 0.000 (55/55) | 0.000 ± 0.000 (0/55) |

Wilson 95% intervals on the pooled 55-dialogue counts are in the JSON.
They are descriptive. They are not extra gates.

## Hypotheses (locked before the run)

A failed hypothesis is a valid result. Thresholds were not moved.

| id | claim | result | numbers |
|---|---|---|---|
| H1 | full strong `N=10` > baseline | **pass** | 1.0 > 0.0 |
| H2 | full strong `N=100` > baseline | **pass** | 1.0 > 0.0 |
| H3 | full weak `N=100` > `off_83` | **pass** | 1.0 > 0.0 |
| H4 | full newest-take ≥ 0.80 | **pass** | 1.0 ≥ 0.80 |
| H5 | full false-recall ≤ 0.05 | **pass** | 0.0 ≤ 0.05 |

Validity gates on this pass: frozen interface 7 / 6 / `bad8e653…`;
no runtime flag in `tidal/llm_memory.py` or the four mechanism
modules; band and schedule integrity true; every condition × `N` cell
present; pytest of the existing suite stayed green.

## What the ablations show

These are observations. They are not a reason to retune.

- `baseline` misses every strong and weak probe (0/55). Ranking is
  log-strength first. Pure turn decay plus a cosine-0.95 neighbor
  leaves a newer distractor above the planted slot, so orthogonal
  cosine retrieve does not save the target.
- `off_77` misses every strong probe at `N=100` (0/55) and hits at
  `N=10` and `N=30`. Without McGaugh gain, the long stream's replay
  refresh prefers a newer 0.95 neighbor.
- `off_83` misses every weak probe (0/55). The weak path needs
  `weak_winner`.
- `off_79` and `off_81` match `full` on this bank. That does not
  move a sleep or retroactive constant.
- Newest-take is 55/55 on every condition, including `baseline`.
  Same-`(name, rel)` overwrite is enough for H4.
- False recall is 0/55 on every condition.

## What this run does not do

- It does not score the reconstruct held bank.
- It does not rescore cue-triggered floors, sleep gates, or arousal
  gates.
- It does not change `CUE_THETA`, half-life, lambda, McGaugh span,
  SWS/REM keeps, retroactive tau, or the weak band.
- It does not write item 10 onto a command or preference channel.
- It does not ask the stack to stop talking.
- It does not merge.
