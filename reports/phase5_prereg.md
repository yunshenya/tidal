# tidal phase 5 pre-registration (written 2026-10-08 19:13 CST, before any phase-5 training)

> **2026-10-08 事后决定（不改下面的预注册原文）。** 新鲜 P5bb_* bake-off 的平均真实 val 差距大于 0.005，测试集也偏向消融，影子 `WINNER` 因此改为 `m3_ablate_m2`（文本情绪开，语音情绪关）。`mamba3_siso` 仍是主干选项。连续体记忆仍只做影子回放。已采用的打断头还没有接进影子 tick。记录下来的差距见 `docs/architecture.md`。


Locked before training. Test numbers are report-only and are never used to choose a head, a seed's epoch, or a backbone. Speech emotion stays off. Continuum memory stays shadow-only (not retrained, not a live input). No DeepSeek calls. CPU only. No keyword or lexicon rules for any label. Krisp is not read. Twitch chat, Artemis, DanmakuTPP, and Krisp are not interrupt-training data.

## 1. Backbone bake-off (phase-4 task, not the new heads)

Candidates are exactly `{mamba3_siso, m3_ablate_m2}`. Fresh runs only. Phase-4 val losses, including the ablation's, are not inputs: the ablation was named a candidate only after those results.

Protocol, identical to `scripts/p4_backbones.sh` (dataset `p3`, no emotion columns, no new heads):

- One public pretrain per candidate. Data `tg,irc,twitch,candor,aishell4`, `--val pub`, `--pre-epochs 10`, `--epochs 20`, `--patience 4`. Tags `P5bb_pub_<kind>`.
- Then real + LLM fine-tune, seeds 0, 1, 2, `--data real,llm --init` that public checkpoint, default fine-tune hyperparameters (60 epochs, patience 8). Tags `P5bb_ft_<kind>_s<seed>`.

**Winner** = lowest mean real-validation head-sum loss over the three fine-tune seeds (the same `hl` used in phase 4). If the two means differ by ≤ 0.005, the winner is the one with lower single-thread streaming step p95, remeasured with `tidal.bench4.bench` (eager, batch 1, 1 thread, random weights of that architecture). Test-split numbers are reported once and do not break a tie.

**The shadow backbone is not switched**, even if `m3_ablate_m2` wins. Report the ranking and stop. `tidal/shadow/p4.py` `WINNER` stays `mamba3_siso` with text emotion on and speech emotion off.

## 2. New-head adoption

A head change is adopted only if its mean validation loss (3 seeds) beats **mamba3_siso + text-emotion without that head**. Speech-emotion features are not an input.

- Frozen body for every head run: `models/P4emo_mamba3_siso_s0.pt` (phase-4 mamba3_siso fine-tune with the text-emotion block). Body, VAP head, and the five heads other than `y_addr` stay frozen. Seeds re-initialise only the trained head parameters (and the optimiser), not the body.
- Three head changes are joint-trained: loss is the unweighted mean of the three heads' own mean binary cross-entropies (each head is normalised by its own labelled rows, so a larger corpus cannot dominate).
- Early stopping uses validation only: the unweighted mean of (private real `y_addr` BCE, public message-value `y_addr` BCE, interrupt BCE, topic BCE), patience 8, max 40 epochs. The selected epoch is the one with the best such mean. Each test split is scored once at that epoch.
- **`y_addr` (not a new head).** Adopt the generalised target iff the mean over seeds 0–2 of private real-val BCE is strictly lower than the mean of `P4emo_mamba3_siso_s0..s2` on the same real-val rows and the same original addressed-to-bot labels.
- **`y_interrupt` and `y_topic` (new heads, same MLP as the other binary heads).** Private chat has no gold for these, so their rows stay masked there and their validation loss is the public validation split. Baseline without the head = logistic regression on the frozen encoder's hidden state (the mamba3_siso + text-emotion representation, no dedicated head), fit on that head's training rows only, scored on its validation rows. Adopt the head iff the 3-seed mean val BCE is strictly lower than that probe. The constant train-prevalence BCE is reported and is not the adoption bar.

## 3. Interrupt labels (data already on disk; no download)

Positive = floor-taking barge-in. Negative = a wait, or a short overlap during which the other side continues (backchannel). Anything else is masked. No transcript lexicon, and CANDOR's empty `tt_label` / `llm_label` / `final_tag` / `overlap_ratio` columns are not read.

Speech intervals (`magicdata_ms`, `oto141`, `candor_tt`, `aishell4_seg`):

- `B` overlaps incumbent `A` (other speaker) when `A.start ≤ B.start < A.end` and `B.start − A.start ≥ 0.05` s. If several incumbents match, use the one with the latest start.
- Positive: that overlap, and `A.end ≤ B.end`, and `B.end − A.end ≥ 0.20` s, and `A` has no segment starting in `(A.end, A.end + 0.40]` s.
- Negative backchannel: that overlap, and `B` lasts ≤ 1.0 s, and `A.end > B.end + 0.05` s.
- Negative wait: `B` starts ≥ 0.20 s after every other speaker's speech (no overlap) and some other speaker has already spoken in the conversation.
- The labelled row is `B`'s onset, not the incumbent's.

Sources:

- `magicdata_ms` (zh, main): human spans already on disk. Drop a span whose speaker is `none` or whose whole text is a bracketed non-speech mark such as `[*]` or `[+]`. That is annotation filtering, not a backchannel word list.
- `oto141`: energy-VAD intervals already cached under `data/public/proc/audio_oto` (no audio re-download). The cache's placeholder tag string is ignored; only start and end are used.
- `candor_tt`: overlap is computed from `offset` and `duration`. Its label columns are empty and are not read.
- `aishell4_seg`: RTTM start, duration, speaker.
- `tg_ru`, text interjection only (no message text). Turn gap G = 8 s. Message `B` by speaker `b` is an interjection when the previous message is from `a ≠ b`, `ts(B) − ts(prev) ≤ G`, and `a` speaks again somewhere later in the conversation. Negative: `a`'s next message is within G of `B` and before `b`'s next message (`a` continues). Positive: `a` has no message within G after `B` (`a` yields) and `b` has another message within G (`b` keeps the floor). Every other `tg_ru` row is masked. `tg_ru` is not given speech-style wait negatives.

Conversation splits for `magicdata_ms` and `oto141`: md5(conv) mod 100, train `<80`, val `<90`, else test (seed-free, fixed). `candor_tt`, `aishell4_seg`, and `tg_ru` keep the phase-3 conversation splits.

## 4. Topic-shift labels (the only downloads)

Sizes checked on the Hugging Face API before download (2026-10-08): both are small text. Nothing else is downloaded.

| set | id | license on the card | language | size | label |
|---|---|---|---|---|---|
| CrossWOZ | `ConvLab/crosswoz` | Apache-2.0 | zh | `data.zip` 15.92 MB (repo total 16.7 MB) | dialogue-state domain change |
| SuperDialseg | `Coldog2333/super_dialseg` (author upload of the SuperDialseg release; not the test-only `dialseg711` mirror) | Apache-2.0 | en | train 35.96 + validation 6.86 + test 6.70 MB | `segmentation_label` |

- CrossWOZ: on each turn, domains = the set of dialogue-state domains that hold a non-empty value. `y_topic = 1` iff that set differs from the previous turn's set, else 0 (the first turn is 0). No keyword list and no embedding threshold.
- SuperDialseg: `y_topic = segmentation_label` (1 = end of a topic). Official train / validation / test splits. No relabeling.

## 5. Reply-target head (`y_addr`, same head, message-value target)

- Group, `irc_dis` and `tg_ru` only: `y = 1` iff a later message in the conversation has a gold reply link pointing at this message. `y = 0` iff no such link exists and the conversation continues for at least 60 s after this message (IRC's minute timestamps: at least two later messages). Otherwise masked. Nick strings are not a label.
- Dyad, `candor_tt` / `magicdata_ms` / `oto141`: on segment `S` by speaker `a`, `y = 1` iff the other speaker's next segment is a non-backchannel floor take (it is not itself a backchannel negative from §3, and it either starts within 5.0 s after `S.end` or is a positive barge-in against `S`). `y = 0` iff 5.0 s after `S.end` is observed and no such response occurs.
- Livestream rows (`twitchchat`, `artemis`, `danmaku_tpp`) stay masked. They are reported as the distribution of sigmoid scores only, never as accuracy.
- Private real rows keep the original addressed-to-bot `y_addr`. They are not relabelled. That is the adoption metric in §2.

## 6. What is not trained on

`twitchchat`, `artemis`, `danmaku_tpp` (masked livestream; scores only), `krisp_tt` (eval-only license; not read this phase), and any newly downloaded corpus other than the two in §4. Weights that see private real rows stay local and are not pushed. Real-data evaluation JSON stays local and gitignored.
