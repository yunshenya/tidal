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
}
