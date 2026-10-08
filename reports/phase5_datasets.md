# Phase 5 datasets and licenses (verified 2026-10-08 UTC+8)

No dataset is redistributed. Loaders and label code only. No weights are published.
Speech emotion stayed off. Continuum memory was not retrained. Krisp was not read.
Twitch chat, Artemis, and DanmakuTPP were not training rows; their `y_addr` scores are a distribution only.

## Already on disk (interrupt and message-value; no new download)

| key | license (phase 3/4 card check) | language | role |
|---|---|---|---|
| magicdata_ms | Apache-2.0 tag; card says academic research only | zh | main interrupt + dyad message value. Human time spans. Bracket-only non-speech marks and speaker `none` dropped |
| oto141 | CC BY 4.0 (gated; local energy-VAD cache) | en | interrupt + dyad message value. Start/end only; the cache's placeholder tag is ignored |
| candor_tt | CC BY 4.0, research-only | en | overlap from `offset` and `duration`. Label columns are empty and were not read. 9 floor-taking positives out of 64,849 labelled onsets |
| aishell4_seg | Apache-2.0 | zh | RTTM start/duration/speaker. Interrupt only (not a dyad) |
| tg_ru | Apache-2.0 | ru | text interjection only (timestamps and speakers, no message text) + group reply-link message value |
| irc_dis | CC BY 4.0 | en | group message value from gold `connections`. Not an interrupt source |

## Downloaded for topic shift (the only downloads)

Sizes checked on the Hugging Face API before download.

| key | id | license on the card | language | size | label |
|---|---|---|---|---|---|
| crosswoz | ConvLab/crosswoz | Apache-2.0 | zh | `data.zip` 15.92 MB (repo 16.7 MB) | dialogue-state domain change. Turns with no `state` are masked and do not reset the previous state. 50,804 labelled turns, 10,691 changes |
| superdialseg | Coldog2333/super_dialseg (author upload; not the test-only `dialseg711` mirror) | Apache-2.0 | en | train 35.96 + validation 6.86 + test 6.70 MB | `segmentation_label` as released. 125,746 turns, 30,623 positives |

No other corpus was downloaded. No keyword list and no embedding-threshold label was used.
