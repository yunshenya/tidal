# Forward barge-in pre-registration (written 2026-10-08 23:40 UTC+8, before any barge-head training)

The phase-5 `y_interrupt` head scores a **finished** segment: the label is attached to an onset only once the overlap and the yield are already known. This file locks a different target, decided **while the floor is still held**, before that outcome.

Nothing below is changed after seeing a loss. No keyword list, no lexicon, no embedding threshold. Speech emotion stays off. The backbone stays frozen `m3_ablate_m2` with text emotion on (`P4emo_m3_ablate_m2_s0`, cache `data/proc/p5_h_m2.npy`). Continuum stays shadow-only. The retrospective `p_interrupt` log is left in place.

## 1. Decision time

One event stream. `self` is the relative role already used in phase 5 (for a speech conversation, the speaker with the earliest segment start; for text, the `role` column). A decision exists only when some **other** speaker holds the floor and `self` is not already in a segment that covers `t`.

- **Speech intervals** (`pub_candor`, `pub_aishell4`, `pub_magic`, `pub_oto`), same spans as `reports/phase5_prereg.md` §3, including the MagicData non-speech-span filter. The floor holder at `t` is the other speaker whose segment covers `t`; if several do, the latest start. Decision times are `floor_start + 0.5, +1.0, +1.5, …` seconds while `t` is still inside that segment. The 0.5 s minimum hold and the 0.5 s step are fixed.
- **Point events** (`pub_tg` only). One decision at each non-self message, `t` = that message's timestamp, floor holder = its speaker, `floor_start` = `t`. `pub_irc` is not used: phase 5 already recorded that its timestamps are minute resolution, so an 8 s horizon is not identifiable. Livestream rows, `pub_crosswoz`, `pub_superdialseg`, Krisp, and private real chat are not decision rows.

## 2. Label `y_barge` (future of `t` only)

- Speech, horizon **H = 1.0 s**. `y = 1` iff `self` has a segment start `u` with `t < u ≤ t + H` and `u < floor_end` (self starts while this floor is still going). `y = 0` iff no such `u` and the recording extends to `min(floor_end, t + H)`. Otherwise masked.
- Text, horizon **H = 8.0 s** (the phase-5 text gap, not a new constant). `y = 1` iff the next message in the conversation is from `self` and within H. `y = 0` iff a next message exists within H and is not `self`, or the conversation is observed through `t + H` with no next message. Otherwise masked.

The horizon follows the event clock (speech interval vs point event), not a dataset name. A positive's decision time is strictly before the onset it predicts.

Splits are the conversation splits already on the phase-5 rows (phase-3 splits, or md5(conv) % 100 for MagicData and oto). No row from a val or test conversation is used to train.

## 3. Features (frozen)

At `t`, in order:

1. `h`: the 128-d `p5_h_m2` state of the latest phase-5 event in the conversation with `ts ≤ t`. No event at or before `t` → the decision is dropped. The body is not updated.
2. Four past-only timing columns: `log1p(t − floor_start)`, `log1p(t − last_event_ts)`, `log1p(min(60, t − last_self_ts))` (60 if self has not yet spoken), and `1` when the floor event is a point event else `0`.

Input dimension is 132. No text is read for the label.

## 4. Comparison (locked before training)

- Head: the phase-5 `BinHead` (`Linear 132→64, GELU, Linear 64→1`), one head, not the joint three-head loss. AdamW lr 5e-4, weight decay 1e-2, batch 64, max 40 epochs, patience 8. Early stopping and model selection use **public validation BCE only**. Seeds 0, 1, 2. The test split is scored once at the selected epoch and is not used to choose anything.
- Baseline: `LogisticRegression(max_iter=400, C=1.0)` fit on the same 132-d training rows only, scored with natural-log log loss on the validation rows (the same probe as phase 5). The constant train-prevalence BCE is reported and is not the bar.
- **Adopt** iff the mean of the three seeds' validation BCE is **strictly lower** than the probe's validation BCE. Otherwise do not wire the head.
- Test numbers, if computed, are reported and cannot reverse the decision.

## 5. Shadow, only if adopted

Log the mean sigmoid of the three seed heads as `p_barge` on the event encoder, on `ControlOut`, and in the `P4` shadow-log row. It is not a tick feature (`TICK_FEATS` unchanged) and it must not change the controller's action or action probabilities. The retrospective `p_interrupt` field stays. If §4 fails, `p_barge` is not added.

Weights stay local (`models/*.pt` is gitignored). They are not pushed.
