"""Which public datasets tidal uses, with the license as verified on the dataset card (2026-10-08) and the files fetched.
`research_only=True` -> weights trained on it must not be published (tidal publishes no weights by default anyway)."""
from pathlib import Path
from tidal.config import ROOT

DATA = ROOT / "data" / "public"

DATASETS = {
    # (a) multi-party text chat with timestamps + speakers (+ reply links)
    "tg_ru": dict(repo="ru-dataset/tg-ru-group-chats", license="apache-2.0", research_only=False,
                  files=["group_data_clean.json"], scenario="group", lang="ru",
                  use="multi-party event stream; reply_to -> which-message / addressed; sender.is_bot -> 'self' role"),
    "irc_dis": dict(repo="jkkummerfeld/irc_disentangle", license="cc-by-4.0", research_only=False,
                    files=["ubuntu/train-00000-of-00001.parquet", "ubuntu/validation-00000-of-00001.parquet",
                           "ubuntu/test-00000-of-00001.parquet"], scenario="group", lang="en",
                    use="gold reply-to links (who responds to which message) in a busy multi-party channel"),
    "ubuntu_days": dict(repo="pttrn-io/ubuntu-irc-days", license="cc0-1.0", research_only=False, files=None,
                        scenario="group", lang="en/nl", use="multi-party timing stream (minute timestamps)"),
    # (b) livestream chat
    "twitchchat": dict(repo="Michielo/twitchchat", license="cc-by-4.0", research_only=False, files=None,
                       scenario="live", lang="en", use="real livestream chat timing (replaces scripted danmaku stream)"),
    "artemis": dict(repo="JosJenn/nasa-artemis-ii-live-chat-comments", license="cc-by-4.0", research_only=False,
                    files=None, scenario="live", lang="multi", use="real high-rate live chat (YouTube + Twitch, one broadcast)"),
    # (c) spoken dialogue with turn-taking
    "magicdata_ms": dict(repo="MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese",
                         license="apache-2.0 tag; card: academic research only, no commercial use", research_only=True,
                         files=None, scenario="1on1_voice", lang="zh",
                         use="per-speaker audio tracks + segment timestamps -> audio front end, EOT/backchannel from audio, VAP"),
    "candor_tt": dict(repo="hirotakahiraki/candor-turntaking-annotations", license="cc-by-4.0 (card; follows CANDOR)",
                      research_only=True, files=None, scenario="1on1_voice", lang="en",
                      use="per-speaker segment timing for 1,656 video-call conversations -> spoken 1:1 timing stream"),
    "aishell4_seg": dict(repo="AISHELL/AISHELL-4", license="apache-2.0", research_only=False,
                         files_glob=["*.TextGrid", "*.rttm"], scenario="meeting", lang="zh",
                         use="speaker-turn timing of Mandarin multi-party meetings (annotations only, no audio)"),
    # gated sets (terms accepted by the repo owner on HF; read token taken from the HF_TOKEN env var, never stored)
    "danmaku_tpp": dict(repo="FRENKIE-CHIANG/DanmakuTPP", license="unspecified (no license on card; gated)", research_only=True,
                        gated=True, files_glob=["DanmakuTPP-Events-part-*.zip"], scenario="live", lang="zh",
                        use="Bilibili danmaku with precise timestamps -> Chinese live-comment timing stream (LOSO livestream)"),
    "krisp_tt": dict(repo="Krisp-AI/turn-taking-test-v1", license="other: benchmarking only; training/fine-tuning forbidden",
                     research_only=True, eval_only=True, gated=True, files=["data/test.parquet", "LICENSE"], scenario="1on1_voice",
                     lang="en", use="EVAL ONLY: shift/hold labels on mono 16 kHz clips -> extra audio end-of-turn test"),
    "oto141": dict(repo="otoearth/otoSpeech-full-duplex-processed-141h", license="cc-by-4.0 (gated)", research_only=False,
                   gated=True, files=[f"data/train/shard-{i:06d}.tar" for i in range(8)], scenario="1on1_voice", lang="en",
                   use="channel-separated 2-speaker full-duplex speech (subset: 8 of 61 shards) -> audio VAP / EOT / backchannel"),
    # phase 4: emotion / affect (text + speech)
    "go_emotions": dict(repo="google-research-datasets/go_emotions", license="apache-2.0", research_only=False,
                        files=["simplified/train-00000-of-00001.parquet", "simplified/validation-00000-of-00001.parquet",
                               "simplified/test-00000-of-00001.parquet"], scenario="emotion_text", lang="en",
                        use="human-labelled Reddit comments, 27 emotions + neutral -> 8 shared classes (amusement -> playful)"),
    "go_emotions_ml": dict(repo="Horizon-Labs/go-emotions-multilingual", license="apache-2.0 (machine-translated GoEmotions)",
                           research_only=False, files=["train.parquet"], scenario="emotion_text", lang="zh (+34 others)",
                           use="TRAIN ONLY: zh machine translations of GoEmotions train (original human labels)"),
    "brighter_cat": dict(repo="brighter-dataset/BRIGHTER-emotion-categories", license="cc-by-4.0", research_only=False,
                         files=[f"{l}/{s}-00000-of-00001.parquet" for l in ("chn", "eng") for s in ("train", "dev", "test")],
                         scenario="emotion_text", lang="zh,en", use="human-annotated multi-label 6 emotions (SemEval-2025 T11) -> main zh/en eval"),
    "brighter_int": dict(repo="brighter-dataset/BRIGHTER-emotion-intensities", license="cc-by-4.0", research_only=False,
                         files=[f"{l}/{s}-00000-of-00001.parquet" for l in ("chn", "eng") for s in ("train", "dev", "test")],
                         scenario="emotion_text", lang="zh,en", use="human emotion intensities 0-3 -> arousal evaluation"),
    "zh_med": dict(repo="Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset", license="mit", research_only=False, files=["data.csv"],
                   scenario="emotion_text", lang="zh-Hant", use="4,159 utterances, 8 tones (partly AI-generated per card; manually annotated)"),
    "zh_med_s": dict(repo="zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset", license="apache-2.0", research_only=False,
                     files=["Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv"], scenario="emotion_text", lang="zh-Hans",
                     use="simplified-Chinese conversion of zh_med (same items; used instead of zh_med)"),
    "crema_d": dict(repo="myleslinder/crema-d", license="odbl", research_only=False, files=["data/crema_d.tar.gz"],
                    scenario="emotion_speech", lang="en", use="7,442 acted clips, 91 speakers, 6 emotions x intensity -> speech emotion head"),
}
