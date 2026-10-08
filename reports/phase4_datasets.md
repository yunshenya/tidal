# Phase 4 datasets and licenses (verified 2026-10-08 UTC+8 from the HF API / dataset cards)

No dataset is redistributed. Loaders and manifest entries only (`tidal/public_data/manifest.py`). No weights are published.

## Used
| key | HF id | license | language | labels | use |
|---|---|---|---|---|---|
| go_emotions | google-research-datasets/go_emotions (simplified) | Apache-2.0 | en | human, 27+neutral → 8 (Ekman grouping; amusement → playful) | text train / dev / test |
| go_emotions_ml | Horizon-Labs/go-emotions-multilingual | Apache-2.0 | zh (Chinese (Simplified) subset) | machine-translated text, original labels | zh text train (+10% dev), secondary eval only |
| brighter_cat | brighter-dataset/BRIGHTER-emotion-categories (chn, eng) | CC BY 4.0 | zh, en | human, multi-label, 6 emotions | **main human-labelled zh/en evaluation** (official test), train split also used |
| brighter_int | brighter-dataset/BRIGHTER-emotion-intensities (chn, eng) | CC BY 4.0 | zh, en | human intensities 0–3 | arousal/valence check only (never trained on) |
| zh_med_s | zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset | Apache-2.0 | zh | 8 tones (partly AI-generated sentences, manually annotated) | train 80% / dev / test (secondary; 疑问 → surprise, 关心 → joy) |
| crema_d | myleslinder/crema-d | ODbL | en (acted speech) | human (6 emotions + intended intensity) | speech head, speaker-held-out (actor id % 5) |
| (teacher) | SenseVoice-Small via sherpa-onnx int8 | FunASR Model Open Source License v1.1 | zh/en/… | model-generated emotion tags | offline teacher on CREMA-D and MagicData segments (public audio only); attribution: SenseVoice-Small, FunASR / Tongyi Lab |
| magicdata_ms | (phase 3 entry) | see phase3_datasets.md | zh | — | audio for teacher distillation |

The Traditional-Chinese original of zh_med_s (Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset, MIT) was downloaded for reference but not used.

## Skipped
- CASIA mirrors (unclear redistribution rights). MELD: research-only at most. CPED: no license.
- EmoBank: not on HF under JULIELab; the mirror has no license.
- dair-ai/emotion: license "other".
- Tongji Robot-EQ: CC BY-NC. BAAI Emotiontalk: CC BY-NC-SA and gated (listed for the user, not requested).
- No license-clear, human-labelled Chinese **speech** emotion dataset was found. Chinese speech affect therefore comes only from teacher distillation.
