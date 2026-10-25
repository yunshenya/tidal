# Long-range recall held-out results

Protocol `long-range-recall-held-v0`.
Pre-registration: [`docs/long_range_recall_prereg.md`](../docs/long_range_recall_prereg.md).
Item 10 (`别说话了`) is an ordinary dialogue fact, not a mute command.
No speak-rate criterion was scored.

## Methodology

Locked seeds `{0, 1, 2, 3, 4}` (5 seeds). Each cell is the mean ± sample standard deviation of the per-seed rates (11 items / seed). Pooled counts `(hits/55)` are descriptive. Item-level raw hits are in `reports/long_range_recall.json` (`per_seed_raw`) and `reports/long_range_recall_seeds.json`.

## Interface

Live cue interface 7. Reconstruct interface 6. Reconstruct sha256 `bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589`.

Band integrity: `True`. Schedule integrity: `True`.

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

## Reading the ablations

#77 / #83 / `baseline` gaps versus `full` are informative. #79 / #81 matching `full` means this harness cannot detect those mechanisms or they do not contribute on this bank. It does not verify them. Gaps are not a reason to retune.

- #77 / `off_77` differs from `full`: recall_at_1_strong at N=100 is 0.000 vs full 1.000.
- #79 / `off_79` matches `full` on every scored cell. This harness cannot detect it here, or it does not contribute on this bank. That is not a verification.
- #81 / `off_81` matches `full` on every scored cell. This harness cannot detect it here, or it does not contribute on this bank. That is not a verification.
- #83 / `off_83` differs from `full`: hit_rate_weak at N=10 is 0.000 vs full 1.000; hit_rate_weak at N=30 is 0.000 vs full 1.000; hit_rate_weak at N=100 is 0.000 vs full 1.000.
- `baseline` differs from `full`: recall_at_1_strong at N=10 is 0.000 vs full 1.000; hit_rate_weak at N=10 is 0.000 vs full 1.000; recall_at_1_strong at N=30 is 0.000 vs full 1.000; hit_rate_weak at N=30 is 0.000 vs full 1.000; recall_at_1_strong at N=100 is 0.000 vs full 1.000; hit_rate_weak at N=100 is 0.000 vs full 1.000.

## Hypotheses

| id | claim | pass | numbers |
|---|---|---|---|
| H1 | full strong N=10 > baseline | **pass** | `{'full': 1.0, 'baseline': 0.0}` |
| H2 | full strong N=100 > baseline | **pass** | `{'full': 1.0, 'baseline': 0.0}` |
| H3 | full weak N=100 > off_83 | **pass** | `{'full': 1.0, 'off_83': 0.0}` |
| H4 | full newest-take ≥ 0.80 | **pass** | `{'full': 1.0, 'floor': 0.8}` |
| H5 | full false-recall ≤ 0.05 | **pass** | `{'full': 0.0, 'ceiling': 0.05}` |

A failed hypothesis is a valid result. Thresholds were not moved.

Elapsed 445.444 seconds.

Git `481f6a40c08c785dc938f23d679f5082e9ee73d7`.

