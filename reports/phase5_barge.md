# Forward barge-in (2026-10-08)

Pre-registration, written and pushed before any of these numbers existed: [phase5_barge_prereg.md](phase5_barge_prereg.md). The head was not retuned after the losses below.

`y_barge` is decided while someone else still holds the floor. Speech: sample every 0.5 s starting 0.5 s into the other speaker's segment; positive iff `self` starts within 1.0 s and before that segment ends. Text (`pub_tg` only): at each non-self message, positive iff the next message is `self` and within 8 s. A positive's decision time is before the onset. No word list. Backbone frozen: `m3_ablate_m2`, text emotion on, cache `p5_h_m2`, plus four past-only timing columns.

Public rows after masking: train 813848 (positive rate 0.0180), val 136648 (0.0255), test 101786 (0.0198). Private real chat was not a training row.

| | val BCE | test BCE (report only) |
|---|---|---|
| seed 0 | 0.0989 | 0.0785 |
| seed 1 | 0.0992 | 0.0776 |
| seed 2 | 0.0988 | 0.0779 |
| mean | 0.0989 | 0.0780 |
| linear probe, same 132-d features | 0.1012 | — |
| train-prevalence constant | 0.1199 | — |

Mean val BCE 0.0989 < probe 0.1012, so the head is adopted. It is logged as `p_barge` (mean of the three sigmoids) on `ControlOut` and in the `P4` shadow row, only while a non-self speaker holds the floor. It is not a tick feature and does not change the action. The retrospective `p_interrupt` field is unchanged. Weights stay in local `models/P5barge_s{0,1,2}.pt` and are not pushed.

Tick latency, interrupt head on, before → after this head (p50 / p95 ms): 0 events 0.16 → 0.15 / 0.30 → 0.30; 1 event 1.49 → 1.95 / 2.15 → 2.44; 5 events 7.03 → 7.12 / 9.80 → 9.42; 20 events (200 ticks) 30.1 → 26.2 / 34.2 → 32.9. The 20-event swing is smaller than the run's noise. All inside the 100 ms tick. See `reports/duplex_bench_barge.json`.
