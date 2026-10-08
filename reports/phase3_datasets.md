# tidal 第三阶段：公开数据集选择（2026-10-08，UTC+8）

> 规则：只用许可证明确、允许研究使用的数据；数据集只下载到本地 `data/public/`（gitignore），**不再分发**，公开仓库只放加载 / 转换代码。
> 卡片上写"仅限学术研究 / 非商用"或源语料是非商用许可的，标为 research-only：在它上面训练的权重不发布（tidal 默认不发布任何权重）。
> 需要登录或同意条款（gated）的数据集一律不绕过。其中 3 个后来由仓库所有者本人在 HF 上同意了条款并提供了只读 token（只放在环境变量里，不写进任何文件），见下面"选用（gated）"。
> 许可证不明确、"other" 或非商用的，一律按 research-only 处理：不发布权重、不再分发，数据只放在 gitignore 的 `data/public/`。许可证以 2026-10-08 时数据集卡片为准。

## 选用

| 类别 | HF id | 大小（下载） | 许可证 | 时间戳 / 标签 | 对应 tidal 的头 |
|---|---|---|---|---|---|
| (a) 多人文字群聊 | `ru-dataset/tg-ru-group-chats` | 0.71 GB | Apache-2.0 | 94.8 万条 Telegram 技术群消息，Unix 时间戳（秒），匿名发言人 ID，`reply_to`（56% 有回复链），`is_bot` | 多人事件流预训练；回复链 → "回应哪条消息" / 被叫；机器人发言 → self 角色；说/不说、接话、续话 |
| (a) 多人 IRC（金标回复链） | `jkkummerfeld/irc_disentangle` | 32 MB | CC BY-4.0 | Ubuntu IRC，分钟级时间戳 + 昵称；人工标注的"这条回复哪条"链接 | 回应哪条消息（金标）；多人轮次 |
| (a) 多人 IRC | `pttrn-io/ubuntu-irc-days` | 24 MB | CC0-1.0 | 两个 Ubuntu 频道 5 年，分钟级时间戳 + 昵称 | 多人时间节奏（补充） |
| (b) 直播弹幕 | `Michielo/twitchchat` | 0.49 GB | CC BY-4.0（原作者 Ringer 等，2020） | Twitch 直播聊天，每场一份：消息、匿名用户、相对开播的秒级时间戳、观众数等元数据 | 用真实直播聊天替换脚本直播流；留一场景测试 |
| (b) 直播弹幕 | `JosJenn/nasa-artemis-ii-live-chat-comments` | 13 MB | CC BY-4.0 | 一场 4.5 小时直播在 YouTube + Twitch 的 24 万条聊天，毫秒级时间戳，匿名作者 | 高并发直播聊天（压力场景）；留一场景测试 |
| (c) 中文双人对话语音（分轨） | `MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese` | 1.21 GB | HF 标签 Apache-2.0，**卡片写明仅限学术研究、禁止商用** → research-only | 10 小时普通话自由对话，每人单独一轨 16 kHz WAV + 分段时间戳和转写 | 音频前端（log-mel → 小型流式编码器）；音频 EOT / 换人、附和（backchannel）、VAP 式未来语音活动 |
| (c) 英文视频通话对话（只有时间和转写） | `hirotakahiraki/candor-turntaking-annotations` | 36 MB | CC BY-4.0（卡片称沿用 CANDOR 许可；按 research-only 处理） | 1,656 场对话、17 万个分说话人片段：起止时间、静默、重叠、ASR 转写 | 口语 1:1 时间节奏流（替换脚本 1:1）；换人 / 续说 |
| (c) 中文多人会议（只有标注） | `AISHELL/AISHELL-4` 的 `*.TextGrid` / `*.rttm` | 约 30 MB（不下载 50 GB 音频） | Apache-2.0 | 211 场会议的说话人分段起止时间 | 多人口语轮次时间节奏（会议场景） |

### 选用（gated，所有者已同意条款）

| 类别 | HF id | 大小（下载） | 许可证 / 条款（卡片原文要点） | 时间戳 / 标签 | 用法 |
|---|---|---|---|---|---|
| (b) 中文弹幕 | `FRENKIE-CHIANG/DanmakuTPP` | 0.39 GB（只下 Events 10 个 zip；不下 QA 和视频帧） | **卡片和标签都没有写许可证**，gate 没有附加条款 → 按 research-only，只在本地做评测和训练，不发布权重 | B 站视频弹幕，带精确时间戳（相对视频的秒数）和文本 | 中文弹幕时间流，加入留一场景（直播）评测。注意：这是点播视频弹幕，不是直播间实时弹幕，也没有主播发言 |
| (c) 音频轮次测试 | `Krisp-AI/turn-taking-test-v1` | 0.42 GB | **other**：LICENSE 只允许做基准测试（benchmark），**明确禁止用于训练或微调模型**、产品开发、再分发和派生数据集 | 2,730 段单声道 16 kHz 英文片段，人工标注 shift / hold（976 / 1,754），最后静默时长 | **只做评测**：音频前端的额外 EOT 测试集，从不参与训练或调参 |
| (c) 双声道全双工语音 | `otoearth/otoSpeech-full-duplex-processed-141h` | 19.4 GB 全集；本阶段只下 8 / 61 个分片（约 2.6 GB，约 18 小时） | CC BY-4.0（gated: auto，需填用途表） | 44.1 kHz 分声道双人英文对话，会话元数据、脱敏区间 | 音频 VAP / EOT / 附和训练（CPU 时间有限，只用一个有代表性的子集） |

合计下载：开放数据约 2.6 GB + gated 约 3.4 GB ≈ 6 GB，低于 25 GB 预算。

## 看过但没用

| HF id | 原因 |
|---|---|
| `lparkourer10/twitch_chat`（CC BY-SA-4.0） | 只有消息文本，没有时间戳和用户，无法转成事件流（已下载后删除） |
| `consulted-graphs/twitch_raw_vod_chats`、`WUJUNCHAO/Danmaku`、`wybxc/danmaku`、`wybxc/mygo-danmaku` | 没有许可证 |
| `EaseZh/magicdata_ramc`（MagicData-RAMC 镜像） | HF 上没有许可证（原始 RAMC 为 CC BY-NC-ND），15 GB |
| `fvdfs41/Discord-Unveiled`（CC BY-4.0） | 2.1 TB；Telegram + IRC 已覆盖多人文字场景，留作以后按服务器部分下载 |
| `ryota-komatsu/fullduplex`（CC BY-NC-SA-4.0） | 128 GB，超预算 |
| `argmaxinc/ali-meetings`（CC BY-SA-4.0） | 9.8 GB 音频；AISHELL-4 的标注已覆盖中文会议轮次，本阶段的音频前端只用双人分轨数据 |
| `anyreach-ai/dualturn-switchboard-turn-taking` | 许可证 "other"，Switchboard 源自 LDC |
| `anonseoul/turn-taking-dataset` | 没有许可证，200 GB |
| `Humbleguava/Personal_Wechat_Msg` | 个人微信记录，涉及第三方隐私，不用 |
| 各类中文多轮对话（`thu-coai/kdconv`、`CJY/Chinese-Dialogue-180k` 等） | 没有时间戳和发言人时序，只是文本对话 |

**没找到**：许可证清楚、带时间戳和发言人的**中文**多人群聊或中文弹幕数据集。中文只在语音侧（MagicData、AISHELL-4）有覆盖。

## 需要用户操作 / 暂不使用（gated）

| HF id | 状态 | 许可证 | 用途 |
|---|---|---|---|
| `otoearth/otoSpeech-full-duplex-280h` | 有权限，**暂不下载**（48.8 GB；141h 处理版已覆盖） | CC BY-4.0 | 同 141h 的原始未筛选版 |
| `otoearth/otoSpeech-full-duplex-turn-104h` | **等待批准**（403，未重试） | other | 双人轮次语音 |
| `nvidia/video-full-duplex-benchmark` | **等待批准**（403，未重试） | other（gated: manual） | 视频全双工基准 |

## 下一阶段候选：情绪 / 情感头（只是浏览时记下的，未下载，许可证需逐个核对卡片和源语料）

| HF id | 模态 / 语言 | HF 许可证标签 | 备注 |
|---|---|---|---|
| `google-research-datasets/go_emotions` | 文本 / 英文 | Apache-2.0 | 5.8 万条 Reddit 评论，27 类情绪 + 中性，多标签 |
| `myleslinder/crema-d` | 语音 / 英文 | ODbL | CREMA-D（原始许可也是 ODbL），91 名演员，6 类情绪，表演式 |
| `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset`、`zzhdbw/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset` | 文本对话 / 中文 | MIT / Apache-2.0 | 需确认来源（可能是 LLM 生成）和标签质量 |
| `BillyLin/CASIA_speech_emotion_recognition` | 语音 / 中文 | 标 Apache-2.0 | **存疑**：CASIA 汉语情感语料原本是授权发售的，HF 镜像的标签不可信，用之前要核实 |
| `declare-lab/MELD` | 对话 文本 + 音视频 / 英文 | GPL-3.0 | 来自电视剧《老友记》，版权存疑 → 只能 research-only |
| `scutcyr/CPED` | 中文对话（情绪 + 人格 + 对话行为） | HF 未标 | 需要找到原始许可证，否则不用 |
