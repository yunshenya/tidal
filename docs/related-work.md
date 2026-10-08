# tidal 相关工作综述（related-work）

> 写于 2026-10-08（UTC+8）。范围：全双工与轮次（turn-taking）、小型多模态模型、CPU 上的语音与音乐前端、高效序列架构、小模型训练技术。
> **核验方式**：文中每篇论文都在 2026-10-08 通过 arXiv 官方 API（`export.arxiv.org`）逐条核对过标题、作者和发表日期；ACL Anthology 论文核对了官方页面的标题和作者；非论文资源（Gemma 3n、Smart Turn、LiveKit、Silero VAD 等）核对过官方页面，且链接返回 HTTP 200。没能核实的条目一律没有收录。文中凡是"参数量、延迟、准确率"这类数字，都来自对应论文的摘要、正文或官方模型卡；需要谨慎看待的地方另有标注。
> 本文只做调研和设计建议，不改动代码。

---

## 0. 读法与 tidal 组件约定

tidal 的**当前实现**以 [architecture.md](architecture.md) 为准，不以本节的第一阶段描述为准。事件主干已实现 GRU、RoPE 因果 Transformer（`tx_kv`）和 Mamba-3（`mamba3` / `mamba3_siso` / 消融 `m3_ablate_m2`，`tidal/backbones.py`）。按字面预注册规则（含消融）胜出的是 `m3_ablate_m2`；四个被点名的主干里，第四阶段胜出的是 Mamba-3 SISO，`mamba3_siso` 仍是代码里的主干选项。影子加载配置现在是 `m3_ablate_m2`（2026-10-08 预注册第五阶段 bake-off，head-sum 真实 val 差距 0.0215），文本情绪开（`emo_features` 的 11 列事件特征，不是 `HEAD_DIMS` 里的头），语音情绪头存在但不进入决策。连续体记忆（`tidal/continuum.py`）只做影子回放。6 个已实现的头仍是 EOT、续话、被叫、说/等/不说、重检、人类接话；打断、话题转移、泛化的回复对象还不是头（`HEAD_DIMS` 里没有）。

下面这一段是**第一阶段**的出发点（见 `reports/phase1.md`），后文的建议都是相对当时写的：输入只有文本和时间节奏，事件级因果 GRU 约 55 万参数。场景是 QQ 群聊和私聊。第一阶段的主要瓶颈在**数据**：文本只覆盖 29% 的消息，被叫正例在测试集上只有个位数，所有头都没有稳定、显著地超过强基线。

下文每篇论文都会标注它对应 tidal 的哪个组件：

| 代号 | 组件 | 含义 |
|---|---|---|
| **F** | 前端 | 原始信号到特征：log-mel、帧采样、分词、时间节奏特征 |
| **E** | 编码器 | 各模态的小编码器（音频、图像、视频、文本），通常冻结并缓存嵌入 |
| **B** | 主干 | 跨模态、跨事件的因果流式时序模型（已实现 GRU / `tx_kv` / Mamba-3；现状见 architecture.md，不是只剩 GRU） |
| **H** | 头 | EOT、续话、被叫、说/等/不说、重检、接话、附和（backchannel）、打断等决策头 |
| **T** | 训练 | 蒸馏、自监督、对比、合成数据、课程、QAT、剪枝、校准 |
| **D** | 部署 | ONNX、int8/int4、级联推理、按需触发 |
| **V** | 评测 | 外部基准与协议 |

**预算假设**（本文自拟，供讨论）：常驻主干 ≤ 5M 参数；各模态编码器按需调用，全部加起来 ≤ 100M 参数，int8 后约 ≤ 100 MB；纯 CPU 单线程下，一次"说不说"决策 p95 ≤ 50 ms（不含视频编码）。下文的"可行性"都按这个预算判断。

tidal 面对两种节奏不同的场景，下文会分开讨论：
- **异步 IM（现状）**：消息是离散事件，图片、语音条、视频、音乐分享都以"附件事件"的形式到来；
- **实时语音/视频流（扩展）**：以 10–100 ms 的帧为单位连续输入，需要全双工式的轮次控制。

---

## 1. 差距分析：市面上的模型覆盖了什么，tidal 补的是什么

### 1.1 对比表

图例：✅ 原生支持；◐ 部分支持或有限制；— 不支持或文献未声明。"何时说话"指模型**自身**是否学会了"说 / 听 / 沉默 / 被打断"之类的决策，不算外挂的 VAD。

| 模型（链接） | 视频 | 图像 | 音乐 | 语音 | 文本 | 何时说话 / 轮次 | 规模 | CPU 可行性 |
|---|---|---|---|---|---|---|---|---|
| [MiniCPM-o 4.5](https://arxiv.org/abs/2604.27393)（OpenBMB, 2026） | ✅ | ✅ | ◐ 通用音频（MMAU） | ✅ 入 + 出 | ✅ | ✅ 全双工：每 1 s 一个块，输出 `[listen]` 或内容，会主动提醒 | 9.34B | 边缘端 < 12 GB RAM（INT4，llama.cpp-omni，测于 RTX 4090 / DGX Spark）；没有报告纯 CPU 实时 |
| [Gander / Multimodal Duplex Interaction Agent](https://arxiv.org/abs/2609.08977)（2026） | ✅ | ✅ | ◐ | ✅ | ✅ | ✅ 全双工，可被打断，会主动追问（基于 MiniCPM-o 4.5） | 约 9B 级 | 否 |
| [DuplexOmni](https://arxiv.org/abs/2606.09186)（2026） | ✅ 流式 | ◐ | ◐ | ✅ | ✅ | ✅ 交互层与思考层异步并行 | 摘要未给出（LLM 级） | 否 |
| [Qwen3-Omni](https://arxiv.org/abs/2509.17765)（阿里, 2025） | ✅ | ✅ | ✅（有音乐理解评测，另有 Captioner） | ✅ 入 + 出 | ✅ | ◐ 流式低首包延迟，但以轮次式交互为主，论文没有把"沉默决策"当作能力 | 30B-A3B（MoE） | 否 |
| [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)（阿里, 2025） | ✅ | ✅ | ◐ | ✅ 入 + 出 | ✅ | ◐ 流式 Thinker-Talker，轮次式 | 3B / 7B（3B 版 HF 权重共约 5.5B，含编码器和 Talker） | 3B 量化后勉强能跑，视频不实时 |
| [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n/model_card)（Google, 2025） | ✅ | ✅ | ◐ 通用音频 | ✅ 仅入 | ✅ | — | E2B / E4B 有效规模（E2B 权重共约 5.4B） | 为手机和笔记本设计，CPU 可跑，但不是 100M 级 |
| [Phi-4-Multimodal](https://arxiv.org/abs/2503.01743)（Microsoft, 2025） | ◐ 多图 | ✅ | ◐ | ✅ 仅入 | ✅ | — | 3.8B 主干 + 模态 LoRA | 勉强 |
| [Ming-Omni](https://arxiv.org/abs/2506.09344)（Inclusion AI, 2025） | ✅ | ✅ | ◐ | ✅ 入 + 出 | ✅ | — | MoE（Ling） | 否 |
| [Megrez-3B-Omni](https://arxiv.org/abs/2502.15803)（2025） | — | ✅ | ◐ | ✅ 仅入 | ✅ | — | 3B | 端侧 |
| [Mini-Omni2](https://arxiv.org/abs/2410.11190)（2024） | — | ✅（CLIP） | — | ✅ 入 + 出（Whisper） | ✅ | ◐ 只能靠"stop omni"之类的口令打断 | Qwen2 小主干 | 勉强 |
| [VITA-1.5](https://arxiv.org/abs/2501.01957)（2025） | ✅ | ✅ | — | ✅ 入 + 出 | ✅ | ◐ 接近实时的语音交互 | LLM 级 | 否 |
| [Moshi](https://arxiv.org/abs/2410.00037)（Kyutai, 2024） | — | — | — | ✅ 全双工 | ✅ 内心独白 | ✅ 双流全双工 | 7.69B（HF 权重） | 否（GPU） |
| [MinMo](https://arxiv.org/abs/2501.06282) / [Freeze-Omni](https://arxiv.org/abs/2411.00774) / [SALMONN-omni](https://arxiv.org/abs/2505.17060) | — | — | — | ✅ | ✅ | ✅ 全双工状态预测 | 约 8B / LLM 级 | 否 |
| [DuplexMamba](https://arxiv.org/abs/2502.11123)（2025） | — | — | — | ✅ 仅入 | ✅ | ✅ 双工状态 | LLM 级（Mamba） | 否 |
| [LLM 增强对话管理（semantic VAD）](https://arxiv.org/abs/2502.14145)（2025） | — | — | — | ✅ | ✅ | ✅ 4 个控制 token | 0.5B | 勉强 |
| [Easy Turn](https://arxiv.org/abs/2509.23938)（ASLP, 2025） | — | — | — | ✅（中文数据） | ✅（ASR 文本） | ✅ 完整 / 不完整 / 附和 / 等待 | Whisper-Medium + Qwen2.5-0.5B（论文表 2 记为 850 MB） | 勉强 |
| [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3)（Pipecat, 开源） | — | — | — | ✅ 仅波形 | — | ◐ 二分类（说完 / 没说完） | 8M（int8 ONNX 8 MB） | ✅ |
| [LiveKit EOU](https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection)（2024 博客） | — | — | — | — | ✅（STT 文本） | ◐ 话语结束概率，用来调 VAD 静音阈值 | 135M（SmolLM v2） | ✅ CPU 约 50 ms |
| [VAP](https://arxiv.org/abs/2205.09812) / [VAP 实时版](https://arxiv.org/abs/2401.04868) | — | — | — | ✅ 双声道 | — | ✅ 未来语音活动投影（仅二人对话） | 小（CPC + 小 Transformer） | ✅ CPU 实时 |
| [SpeculativeETD](https://arxiv.org/abs/2503.23439)（ACL 2026） | — | — | — | ✅ | — | ◐ 说完 / 停顿 | 端侧 GRU 约 202K 参数 + 服务端 wav2vec2 | ✅（端侧部分） |
| [SmolVLM](https://arxiv.org/abs/2504.05299)（HF, 2025） | ✅ | ✅ | — | — | ✅ | — | 256M / 500M / 2.2B | ◐ 256M 版可在 CPU 上跑 |
| [Mellow](https://arxiv.org/abs/2503.08540) / [Mizar](https://arxiv.org/abs/2609.28344) / [Samsone](https://arxiv.org/abs/2609.21666) / [TinyMU](https://arxiv.org/abs/2604.15849) | — | — | ✅（TinyMU 专攻音乐；其余为通用音频） | ✅ 仅入 | ✅ | — | 167M / 159M / 99M–356M / 229M | ✅（Mizar 在单 CPU 上端到端 1.09 s；Samsone 有安卓 App） |
| [Omni-C](https://arxiv.org/abs/2603.05528)（2026） | — | ✅ | ◐（AudioSet 和 NSynth 评测） | ◐ 音频 | ✅ | — | 111.9M 共享编码器（只出嵌入，不生成） | ✅ |
| [GroupGPT](https://arxiv.org/abs/2603.01059)（2026） | ◐ 视频消息 | ◐ 表情包 / 图片 | — | ◐ 语音消息 | ✅ | ✅ 群聊介入时机（端云协同，把时机判断与生成解耦） | 框架，依赖 LLM（在 MUIR 上评测了从小型开源模型到闭源 LLM 的多种模型） | ◐ 端侧部分 |

### 1.2 实话实说：缺口究竟在哪里

1. **"原生多模态 + 自主决定何时说话"在大模型规模上已经有人做了，tidal 不能自称首个。** MiniCPM-o 4.5（9B）的 Omni-Flow 把视频、音频和输出对齐到同一条时间轴上，每个时间块自行决定 `[listen]` 还是输出，还能主动提醒；Gander 和 DuplexOmni（2026）也在同一方向。Qwen3-Omni 覆盖了视频、图像、音乐、语音、文本五种模态，但以轮次式交互为主。
2. **在 1B 以下，经核实，没有一个模型同时覆盖"视频 + 图像 + 语音 + 音乐 + 文本"并学会何时说话。** 规模最小的现有方案各自只占一角：
   - 轮次检测器很小，但都是单一或双模态：Smart Turn（8M，只看波形）、LiveKit EOU（135M，只看文本）、VAP（只看双声道音频）、SpeculativeETD（202K 的端侧 GRU）；
   - 小型理解模型不做轮次：SmolVLM-256M 只有视觉和文本；Mellow、Mizar、Samsone、TinyMU 只有音频和文本；Omni-C 是 112M 的图像、音频、文本嵌入编码器。
   - **100M 以下**：没有查到任何 omni（同时含视觉和音频）生成模型。最接近的是 Samsone-99M（只有音频和文本）和 Omni-C（112M，略超 100M，且没有视频和轮次）。
3. **多方群聊（异步 IM）里的"说不说、是不是在叫我、对方会不会接着发"是公认的开放问题，而且现有方案都靠 4B–7B 级 LLM。**
   - Speak or Stay Silent（2026）：8 个 LLM 在零样本下普遍失败，加推理痕迹做 SFT 后平衡准确率最多 +23 个百分点；
   - MP-Bench（2026，EMNLP Findings）：实时语音代理在多方轮次上接近随机；
   - IWSDS 2025 的基准：GPT-4o 的被叫识别只比随机略好；
   - GroupGPT 用端云协同把介入时机和回复生成解耦，但仍依赖 LLM。
4. **因此，tidal 能诚实主张的定位是**：一个**只做感知和发言权控制、不负责生成**的**微型（目标 ≤ 100M，常驻 ≤ 5M）、纯 CPU、事件驱动**的多模态轮次模型。它的输入是群聊事件流（文本、时间节奏、图片、语音条、视频、音乐分享），以后可扩展到实时语音；它的输出是 EOT、续话、被叫、说/等/不说、重检延迟、附和/打断等决策和校准过的置信度。生成交给外部 LLM。同一尺寸档上的对手是 Smart Turn、LiveKit EOU 和 VAP，不是 MiniCPM-o。
5. **需要警惕**：这个缺口只是定位上的，还不是已经证明的能力。第一阶段的结果显示，tidal 在纯文本和时间节奏上还没有显著超过强基线。多模态只有在数据和自监督目标（见第 2 节和第 7 节）跟上之后，才可能带来净收益。

---

## 2. 方向一：全双工对话与轮次

### 2.1 奠基工作

- **TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog**：Ekstedt & Skantze（KTH），Findings of EMNLP 2020。[arXiv:2010.10874](https://arxiv.org/abs/2010.10874)
  - 要点：在语言模型里加一个"轮次结束"特殊 token，逐词预测 EOT 概率，证明了光靠文本语用线索就能预测轮次。
  - tidal：**H / T**。把每条消息的 EOT 改成**逐字或逐子词的 EOT 概率**，可以从 bge 之外的一个微型字符级因果 LM 头得到；"消息以'然后'结尾"这类手工特征可以由它学出来。
  - 可行性：高。几 M 参数的字符级 GRU 或线性注意力就够用。

- **Response-conditioned Turn-taking Prediction (RC-TurnGPT)**：Jiang, Ekstedt, Skantze，Findings of ACL 2023。[arXiv:2305.02036](https://arxiv.org/abs/2305.02036)
  - 要点：EOT 预测同时以"我打算说什么"为条件。人类接话不只看对方说没说完，也看自己要说的话合不合时宜。
  - tidal：**H**。`y_act`（说/等/不说）可以以外部 LLM 的候选回复嵌入为条件：先由 LLM 生成草稿，tidal 再判断此刻发出去是否合适。
  - 可行性：中。只在要做"发不发"决策时多编码一次草稿。

- **Voice Activity Projection (VAP): Self-supervised Learning of Turn-taking Events**：Ekstedt & Skantze，Interspeech 2022。[arXiv:2205.09812](https://arxiv.org/abs/2205.09812)，代码 [VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection)
  - 要点：不标注轮次事件，而是**自监督地预测未来 2 s 内两位说话人在若干离散时间桶里的语音活动**（共 256 种联合状态）。轮次转换、保持、附和都可以从这个分布里零样本读出来。
  - tidal：**H / T，最重要的借鉴**。把它改造成群聊版的"未来事件投影"：对每个事件，预测未来若干时间桶（如 0–10 s、10–30 s、30–120 s、2–10 min）里"同一发言人 / 其他人 / 机器人"各自会不会发消息。这个目标**只需要时间戳和角色**，可以直接用上全部 10,973 条消息（包括 71% 没有文本的消息），绕开文本覆盖率的问题；现有的 `y_self`、`y_hreply`、`y_recheck` 都可以从投影分布里推出来，或者把它当作辅助损失。
  - 可行性：高。输出头只是一个小 softmax 或多个 sigmoid。

- **Turn-Taking Prediction for Natural Conversational Speech**：Chang 等（Google），Interspeech 2022。[arXiv:2208.13321](https://arxiv.org/abs/2208.13321)
  - 要点：在流式 ASR 里联合预测轮次，处理犹豫、停顿和重复等不流利现象。
  - tidal：**H（语音扩展）**。将来接入语音时，EOT 头和流式 ASR 编码器共享。
  - 可行性：中。

- **Generative Spoken Dialogue Language Modeling (dGSLM)**：Nguyen 等（Meta），2022（arXiv）。[arXiv:2203.16502](https://arxiv.org/abs/2203.16502)
  - 要点：首个双塔、双声道的无文本口语对话 LM，自然地建模了重叠和停顿。
  - tidal：**B**。"每个参与者一条流、跨流交叉注意力"的思路可以借鉴：群聊里给"机器人"和"其他人"各留一路状态。
  - 可行性：中（只借结构，不借规模）。

### 2.2 VAP 的后续工作

- **Real-time and Continuous Turn-taking Prediction Using VAP**：Inoue 等，IWSDS 2024。[arXiv:2401.04868](https://arxiv.org/abs/2401.04868)，代码 [VAP-Realtime](https://github.com/inokoj/VAP-Realtime)
  - 要点：CPC 编码器加自注意力和交叉注意力，缩短上下文后**能在 CPU 上实时运行**，性能下降很小。
  - tidal：**D**。证明 VAP 类模型可以部署在纯 CPU 上；上下文长度和处理频率可以作为部署旋钮。
  - 可行性：高。

- **Multilingual Turn-taking Prediction Using VAP**：Inoue 等，LREC-COLING 2024。[arXiv:2403.06487](https://arxiv.org/abs/2403.06487)
  - 要点：一个 VAP 模型同时服务英语、普通话和日语。
  - tidal：**T**。中文对话的轮次线索可以和别的语言共享，外部英文语料可用于预训练。

- **Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of VAP**：Inoue, Lala, Skantze, Kawahara，NAACL 2025。[arXiv:2410.15929](https://arxiv.org/abs/2410.15929)
  - 要点：先在通用对话上预训练 VAP，再在附和数据上微调，逐帧预测附和的**时机和类型**，并且是在不平衡的真实数据上做的。
  - tidal：**H / T**。给群聊加一个"附和"动作（表情回应、"哈哈"、点赞），而不是只有说和不说；"先通用预训练、再小数据微调"正好适合 tidal 的数据规模。
  - 可行性：高。

- **Prompt-Guided Turn-Taking Prediction**：Inoue 等，SIGdial 2025。[arXiv:2506.21191](https://arxiv.org/abs/2506.21191)
  - 要点：把"更快一点""更沉稳"之类的文本提示嵌入注入 VAP，动态调节轮次风格；提示数据由 LLM 合成。
  - tidal：**H**。用一个"群配置或人设"向量控制话多话少（不同群对机器人插话的容忍度不同），取代对每个群单独调阈值。
  - 可行性：高。只是一个条件向量。

- **Voice Activity Projection Model with Multimodal Encoders**：Saga & Pelachaud，2025。[arXiv:2506.03980](https://arxiv.org/abs/2506.03980)
  - 要点：给 VAP 加上预训练的音频和人脸编码器，捕捉细微表情，在轮次指标上达到或超过 SOTA。
  - tidal：**E**。印证了"冻结的预训练模态编码器加小融合层"这条路线。

- **Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction (MM-VAP)**：O'Connor Russell & Harte，Findings of ACL 2025。[ACL Anthology](https://aclanthology.org/2025.findings-acl.12/)
  - 要点：加入面部表情、头部姿态和注视后，视频会议里 hold/shift 的准确率从 79% 升到 84%；面部表情贡献最大；用 ASR 自动对齐来训练也可行。
  - tidal：**E / H（视频扩展）**。在实时视频场景里，人脸动作特征的性价比最高。

- **Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals (MM-F2F)**：Lin 等，ACL 2025。[ACL Anthology](https://aclanthology.org/2025.acl-long.743/)
  - 要点：自动采集 210 小时对话视频，**支持文本、音频、视频任意组合输入**，轮次 F1 提升 10%，附和 F1 提升 33%。
  - tidal：**B / T**。"任意模态子集都能推理"正是群聊的常态（多数消息只有文本，偶尔有图或语音），训练时应随机丢弃模态。

- **Gaze-Enhanced Multimodal Turn-Taking Prediction in Triadic Conversations**：Heo 等，Interspeech 2025。[arXiv:2505.13688](https://arxiv.org/abs/2505.13688)
  - 要点：轻量 CNN 加二值 VAD 和说话人定位热图，用于三人对话的轮次预测。
  - tidal：**H**。多方场景可以表示成"每个参与者一个通道"的热图或集合，而不是二人场景的双声道。

### 2.3 EOT 与语义 VAD（工程化，小模型）

- **Speculative End-Turn Detector (SpeculativeETD)**：Ok, Yoo, Lee，ACL 2026。[arXiv:2503.23439](https://arxiv.org/abs/2503.23439)
  - 要点：端侧用约 **202K 参数的 GRU**（Conv2D 前端加单层 GRU，40 维 log-mel）快速检测非说话段，难例交给服务端 wav2vec2；同时发布了首个公开的 ETD 数据集（TTS 合成加真实数据）。论文报告的计算量（每 100 个样本）：GRU 约 45 MFLOPs，VAP 约 10,355 MFLOPs。
  - tidal：**D / T**。①级联：tidal 本身就是端侧小模型，不确定时升级到 LLM；②用 TTS 合成停顿来扩充数据（tidal 已有合成管线）。
  - 可行性：很高，和 tidal 当前的 GRU 规模一致。

- **Easy Turn**：Li 等（西工大 ASLP），2025。[arXiv:2509.23938](https://arxiv.org/abs/2509.23938)，代码 [Easy-Turn](https://github.com/ASLP-lab/Easy-Turn)
  - 要点：Whisper-Medium 加 Qwen2.5-0.5B，先做 ASR 再判轮次，预测**完整、不完整、附和、等待**四种状态；发布 1,145 小时的**中文**轮次训练集。论文对比显示 TEN Turn Detection 依赖 7B LLM，Smart Turn V2 只有两类。
  - tidal：**H / T / V**。①四状态的标签体系可以直接借来扩展 `y_eot`；②中文语音轮次数据可用于将来语音分支的预训练和评测；③它可以当教师，蒸馏到 tidal 的小音频头。
  - 可行性：数据和标签高；模型本身对 tidal 太大，只适合当教师。

- **Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction**：Wu 等，2025。[arXiv:2509.20410](https://arxiv.org/abs/2509.20410)
  - 要点：基于 LLM 的流式语义端点检测，滑窗训练；语义不完整时用更长的超时。
  - tidal：**H**。"按语义完整度动态调等待时长"可以直接映射到 `y_recheck`：输出的不是固定阈值，而是等待时长的分布。

- **JAL-Turn**：Yang 等，2026。[arXiv:2603.26515](https://arxiv.org/abs/2603.26515)
  - 要点：和 ASR **共享冻结编码器**，用交叉注意力融合声学和语言特征来预测 hold/shift，不增加端到端延迟；还有一条从大规模真实对话自动导出轮次标签的流水线。
  - tidal：**E / T**。语音分支应和 ASR 共享编码器；"从真实日志自动导出标签"和 tidal 的 `labels.py` 思路一致。

- **LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems**：Zhang 等，2025。[arXiv:2502.14145](https://arxiv.org/abs/2502.14145)
  - 要点：0.5B 的语义 VAD 当对话管理器，预测 4 个控制 token（开始说、继续说、开始听、继续听），能区分有意打断和无意打断；核心对话引擎只在需要生成时才激活。
  - tidal：**H / D**。正好是 tidal 的系统定位：**轮次控制和内容生成解耦**。这 4 个控制 token 可以作为 `y_act` 在语音场景下的扩展。

- **Turn-taking and Backchannel Prediction with Acoustic and LLM Fusion**：Wang 等（Amazon），ICASSP 2024。[arXiv:2401.14717](https://arxiv.org/abs/2401.14717)
  - 要点：声学模型和 LLM 融合，在 Switchboard 上预测轮次和附和，多任务指令微调。
  - tidal：**T**。可以作为 LLM 教师的融合方式，蒸馏给小学生模型。

- **Smart Turn v3**（Pipecat，开源）：[HF 模型卡](https://huggingface.co/pipecat-ai/smart-turn-v3)，[GitHub](https://github.com/pipecat-ai/smart-turn)
  - 要点：Whisper Tiny 编码器加线性头，**8M 参数**，int8 ONNX 8 MB，只看波形判断说完没说完。
  - tidal：**E / D**。将来语音条和实时语音的 EOT 基线兼对照组；它的 Whisper-Tiny 编码器也可以当 tidal 的语音编码器候选。

- **LiveKit End-of-Utterance 模型**（[博客，2024-12](https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection)，[HF](https://huggingface.co/livekit/turn-detector)）
  - 要点：135M 的 SmolLM v2，以最近 4 轮的 STT 文本为输入预测话语结束，用来**动态拉长或缩短 VAD 静音阈值**；CPU 推理约 50 ms，意外打断减少 85%（与只用 VAD 相比）。
  - tidal：**H / D**。"模型不直接替代阈值，而是调制阈值"是稳妥的上线方式，和 tidal 的影子模式很契合。

- **Silero VAD**（[GitHub](https://github.com/snakers4/silero-vad)）：业界常用的 CPU 级 VAD，MiniCPM-o 4.5 的数据管线也在用。
  - tidal：**F**。语音分支的第一道门，只在有语音时才跑后面的编码器。

### 2.4 端到端全双工模型（了解原理，不照搬规模）

- **Moshi**：Défossez 等（Kyutai），2024。[arXiv:2410.00037](https://arxiv.org/abs/2410.00037)：双音频流加"内心独白"文本流，全双工，没有显式轮次分割。
  - tidal：**B**。借"机器人流和用户流并行建模"的思想；"文本内心独白"对应 tidal 里"机器人自己的状态"那一路输入。可行性：模型本身不可行（7.7B）。
- **Beyond Turn-Based Interfaces: Synchronous LLMs (SyncLLM)**：Veluri 等（UW / Meta），EMNLP 2024。[arXiv:2409.15594](https://arxiv.org/abs/2409.15594)：把真实时钟注入 LLM，用 212k 小时**由文本对话合成的语音对话**训练，只需要 2k 小时真实数据。
  - tidal：**T**。"先由文本对话合成，再用少量真实数据校正"的配方，正是 tidal 合成管线可以学的。
- **Language Model Can Listen While Speaking (LSLM)**：Ma 等，2024。[arXiv:2408.02622](https://arxiv.org/abs/2408.02622)：边说边听，能检测打断。
  - tidal：**H**。增加"机器人正在输出时，用户是否在打断"的头，对应群聊里"机器人刚发了一半，有人插话"的场景。
- **OmniFlatten**：Zhang 等（阿里），2024。[arXiv:2410.17799](https://arxiv.org/abs/2410.17799)：把多流扁平化成单一序列，训练分三个阶段：模态对齐、半双工对话、全双工对话。
  - tidal：**T**。课程式训练（先学轮次式，再学重叠和打断）。
- **Freeze-Omni**（[arXiv:2411.00774](https://arxiv.org/abs/2411.00774)）、**MinMo**（[arXiv:2501.06282](https://arxiv.org/abs/2501.06282)）、**SALMONN-omni**（[arXiv:2505.17060](https://arxiv.org/abs/2505.17060)，NeurIPS 2025）：分别是冻结 LLM 加多任务双工、8B 全双工、无编解码器的"思考"状态 token（每个时间块决定听还是说，并用 RL 改进）。
  - tidal：**H**。共同结论是"状态转移"应该作为显式的分类目标来学。
- **DuplexMamba**：Lu 等，2025。[arXiv:2502.11123](https://arxiv.org/abs/2502.11123)：用 Mamba 做语音编码器和语言模型，以状态 token 实现双工。
  - tidal：**B**。证明了 SSM 适合双工流式。
- **MiniCPM-o 4.5 / Omni-Flow**：OpenBMB，2026。[arXiv:2604.27393](https://arxiv.org/abs/2604.27393)
  - 要点：视频、音频、输出三条流按时间块交织；**消融显示 1.0 s 的块最好，0.1–0.2 s 明显变差；显式边界 token 更好；"先判断听/说、再生成内容"（LS）优于把 `[listen]` 混在词表里（LT）**。
  - tidal：**B / H，很重要**。三条结论都可以直接用到 tidal：①事件或时间块之间显式插入边界 token；②"说不说"由独立的头决定，和生成彻底分开；③决策频率不必太高，群聊里按事件触发加周期性重检就够了。
- **Gander**（[arXiv:2609.08977](https://arxiv.org/abs/2609.08977)）：小脑-大脑协同，小脑负责实时交互，大脑负责复杂推理。
  - tidal：**D**。tidal 就是"小脑"，LLM 是"大脑"。
- **DuplexOmni**（[arXiv:2606.09186](https://arxiv.org/abs/2606.09186)）：交互层和思考层异步并行，思路同上。

### 2.5 多方与群聊：被叫识别与"说不说"

- **MPC-BERT**：Gu 等，ACL 2021。[arXiv:2106.01541](https://arxiv.org/abs/2106.01541)
  - 要点：多方对话的预训练任务，包括"谁回复谁"、说话人识别、指代和共享节点检测。
  - tidal：**T**。把"回复对象预测""同一说话人检测"作为自监督辅助任务。QQ 的回复和 @ 标记本身就是免费标签（注意**特征里不能用标签来源列**，这条约束在 `features.py` 里已经执行）。
- **Multi-Party Chat (MultiLIGHT)**：Wei 等（Meta），2023。[arXiv:2304.13835](https://arxiv.org/abs/2304.13835)：指出两人对话模型在群聊里缺两种能力，一是**决定何时说话**，二是基于多角色的连贯生成。
- **MUCA**：Mao 等，2024。[arXiv:2401.04883](https://arxiv.org/abs/2401.04883)：群聊机器人的 3W 维度（What / When / Who），外加 LLM 多用户模拟器。
  - tidal：**T**。用 LLM 多用户模拟器来生成合成群聊（和现有的 `synth.py` 对齐）。
- **Proactive Conversational Agents with Inner Thoughts**：Liu 等，2024-12（arXiv）。[arXiv:2501.00383](https://arxiv.org/abs/2501.00383)：AI 先形成"内心想法"，再评估说出来的动机，而不是只看下一个说话人是谁。
  - tidal：**H**。`y_act` 可以接收外部 LLM 的"想法强度"作为输入特征（RC-TurnGPT 的群聊版）。
- **DiscussLLM: Teaching LLMs When to Speak**：Patel 等（NEC Labs），2025。[arXiv:2508.18167](https://arxiv.org/abs/2508.18167)
  - 要点：两阶段**合成**多方讨论数据，标注 5 类介入；训练模型在不需要介入时输出 silent token；比较了端到端和**解耦的分类器加生成器**两种方案（后者延迟更低）。
  - tidal：**T / H**。合成数据配方，加上"解耦分类器"的设计。tidal 就是那个分类器。
- **Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games**：Eckhaus, Berger, Stanovsky，Findings of EMNLP 2025。[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.608/)
  - 要点：**异步**群聊里拆成两步，先由调度器决定说不说，再决定说什么；调度器会根据自己最近说话的频率自我调节。
  - tidal：**H / 特征**。和 QQ 群聊最相近。"机器人近期发言占比"应作为显式特征（tidal 已有 `log_since_bot`），也可以作为调节话量的条件。
- **An LLM Benchmark for Addressee Recognition in Multi-modal Multi-party Dialogue**：Inoue 等，IWSDS 2025。[ACL Anthology](https://aclanthology.org/2025.iwsds-1.36/)：三人对话里约 20% 的轮次有明确的受话人；**GPT-4o 只比随机略好**。
  - tidal：**V**。说明 `y_addr` 难，仅靠大模型零样本不够，必须专门训练。
- **Friends-MMC**：Wang 等，AAAI 2025。[arXiv:2412.17295](https://arxiv.org/abs/2412.17295)：多模态多方对话数据集（24k 句加视频，标注了说话人和人脸）。
  - tidal：**T / V**。视频分支的"谁在说、对谁说"预训练数据。
- **Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue**：Bhagtani 等，2026（投稿 Interspeech 2026）。[arXiv:2603.11409](https://arxiv.org/abs/2603.11409)，代码 [context_aware_modeling](https://github.com/ishikilabsinc/context_aware_modeling)
  - 要点：在 AMI、Friends、SPGI 三个语料上构建了 **12 万+ 个"停顿处说还是不说"的决策点**；8 个 LLM 零样本普遍失败；用**蒸馏的推理痕迹**做 SFT 后，平衡准确率最多 +23 个百分点；结论是"这不是涌现能力，必须专门训练"。
  - tidal：**T / V，高优先级**。①直接作为 `y_act` 和 `y_addr` 的外部预训练和评测数据（都是公开语料的转写）；②"LLM 推理痕迹蒸馏"可以改造成：教师给出推理和软标签，tidal 只学软标签和辅助的理由类别。
- **GroupGPT / MUIR**：Shen 等，2026。[arXiv:2603.01059](https://arxiv.org/abs/2603.01059)
  - 要点：**端云协同，把介入时机和回复生成解耦**，敏感信息在端侧处理，token 用量最多降低 3 倍；支持表情包、图片、视频、语音消息；MUIR 有 2,500 段带介入标签和理由的群聊。
  - tidal：**D / V**。和 tidal 的定位几乎一样（端侧小模型判时机，加隐私脱敏），可以作为直接对标；MUIR 可以用作外部测试集（需确认许可证）。
- **ProACT**：Yang 等，2026。[arXiv:2607.03730](https://arxiv.org/abs/2607.03730)：多用户协作中的"崩溃感知"主动介入（分歧、目标含糊、讨论绕圈、参与失衡）。
  - tidal：**H**。可以把"介入理由"作为 `y_act` 的细分类别。
- **MP-Bench**：Shih 等，Findings of EMNLP 2026。[arXiv:2609.13076](https://arxiv.org/abs/2609.13076)：四人语音对话中评测显式、隐式和负向轮次（即**不该说的时候不说**）；12 个语音代理在多方轮次上接近随机。
  - tidal：**V**。语音扩展后的外部评测。

### 2.6 IM 文本节奏：连发、等待、重叠

- **"Wait, I'm Still Talking!" Imagine-Then-Arbitrate**：Lin 等，2020。[arXiv:2002.09616](https://arxiv.org/abs/2002.09616)
  - 要点：IM 用户常把一句话拆成几条短消息发出，代理要先判断"等对方继续发"还是"现在回"；方法是想象对方和自己的下一句，再做仲裁。
  - tidal：**H**。这几乎就是 `y_self`（自我续话）加 `y_eot` 的原始定义，可以作为基线和动机引用。
- **Stephanie2**：Yang 等，2026。[arXiv:2601.05657](https://arxiv.org/abs/2601.05657)：逐步决定发送还是等待，把延迟建模成"思考时间加打字时间"，消息节奏自适应。
  - tidal：**H / D**。`y_recheck` 的输出可以直接驱动"多久后再看一次"；回复的分条和节奏也可以借鉴。
- **Beyond Turn-taking: Text-based Overlap (OverlapBot)**：Kim 等，2025。[arXiv:2501.18103](https://arxiv.org/abs/2501.18103)：文本聊天里也存在重叠，比如对方没打完就回一句"嗯"，用户觉得更自然。
  - tidal：**H**。支持在文本场景里加一个"附和"动作（短回应或表情）。

### 2.7 视频流里的"何时说"

- **VideoLLM-online (LIVE)**：Chen 等，CVPR 2024。[arXiv:2406.11816](https://arxiv.org/abs/2406.11816)：在视频流里逐帧决定是否输出（流式 EOS）。
- **MMDuet**：Wang 等，Findings of EMNLP 2025。[arXiv:2411.17991](https://arxiv.org/abs/2411.17991)：每帧有"信息量"和"相关性"两个分数，加权超过阈值就回复。
- **MMDuet2**：Wang 等，ICLR 2026。[arXiv:2512.06810](https://arxiv.org/abs/2512.06810)：改为文本式的"NO REPLY / 回复"决策，用多轮 RL 训练。
- **LiveCC**：Chen 等，CVPR 2025。[arXiv:2504.16030](https://arxiv.org/abs/2504.16030)：把 ASR 字幕和视频帧按时间戳密集交织来训练流式视频 LLM。
  - tidal：**H / T**。视频消息或直播流的"值不值得评论"头可以照搬 MMDuet 的"两个分数加阈值"结构（便于校准）；LiveCC 说明 **ASR 时间戳可以作为免费的弱监督**。

### 2.8 评测基准与综述

- **Talking Turns**（Arora 等，ICLR 2025，[arXiv:2503.01174](https://arxiv.org/abs/2503.01174)）：用一个训练好的轮次模型当裁判；发现 Moshi 打断过于激进，系统很少附和。
- **Full-Duplex-Bench**（Lin 等，ASRU 2025，[arXiv:2503.04721](https://arxiv.org/abs/2503.04721)）和 **v2**（ACL 2026，[arXiv:2510.07838](https://arxiv.org/abs/2510.07838)），代码 [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)：覆盖停顿处理、附和、轮次和打断，v2 增加了多轮评测和自动考官。
- **FLEXI**（[arXiv:2509.22243](https://arxiv.org/abs/2509.22243)）：包含紧急情况下的模型打断。
- **ICASSP 2026 HumDial**（[arXiv:2604.21406](https://arxiv.org/abs/2604.21406)）：真人录制的打断和重叠数据。
- **全双工综述**（Lu 等，2026，[arXiv:2606.19453](https://arxiv.org/abs/2606.19453)）：提出 L0–L3 架构层级、"时间关系 × 意图 × 响应"的交互本体，以及 IDLE / LISTEN / SPEAK / WAIT / DUAL 状态机。
  - tidal：**V / H**。①可以把 tidal 的动作空间对齐到这个状态机（群聊下 DUAL 基本不出现）；②按这个本体给评测切片，比如"附和单元格上失败"，让负面结果更好解读。

---

## 3. 方向二：小而高效的多模态模型

### 3.1 小型视觉-语言与视觉编码器

- **SmolVLM**：Marafioti 等（Hugging Face），2025。[arXiv:2504.05299](https://arxiv.org/abs/2504.05299)
  - 要点：256M 的版本推理时 GPU 内存 < 1 GB，支持图像和视频；关键在于激进而高效的图像 token 化（大幅压缩视觉 token）和数据配比。
  - tidal：**E**。不把整个 VLM 当编码器，只借它的视觉塔和 token 压缩做法；或者离线把它当"图片描述教师"，生成 caption 后再走 tidal 的文本路径（先做这个最便宜）。
  - 可行性：256M 只适合离线或按需调用，不适合常驻。
- **TinyLLaVA**（[arXiv:2402.14289](https://arxiv.org/abs/2402.14289)）：小规模 LMM 的系统消融（视觉编码器、连接器、LLM、数据、训练配方），结论是更好的数据质量加更好的训练配方，能让小 LMM 追平更大的模型。
- **FastVLM**（Apple，CVPR 2025，[arXiv:2412.13303](https://arxiv.org/abs/2412.13303)）：混合视觉编码器 FastViTHD，高分辨率下视觉 token 更少、编码更快。
- **OmniVLM**（[arXiv:2412.11475](https://arxiv.org/abs/2412.11475)）：968M，把视觉 token 从 729 个压到 81 个。
- **MobileCLIP**（Apple，CVPR 2024，[arXiv:2311.17049](https://arxiv.org/abs/2311.17049)）和 **MobileCLIP2**（TMLR 2025，[arXiv:2508.20691](https://arxiv.org/abs/2508.20691)）：面向移动端的图文对比模型，"多模态强化训练"即把多个教师的嵌入和合成 caption 离线存下来再蒸馏。
  - tidal：**E / T，图片分支首选**。用最小档 MobileCLIP2 的图像塔做冻结编码器，嵌入离线缓存；"离线存教师输出再蒸馏"的做法也适用于 tidal 的其他模态。
- **TinyCLIP**（Microsoft，ICCV 2023，[arXiv:2309.12314](https://arxiv.org/abs/2309.12314)）：通过亲和力模仿和权重继承来蒸馏 CLIP。
  - tidal：**T**。需要更小的图像塔时，可以按这个方法自己蒸馏。
- **SigLIP**（ICCV 2023，[arXiv:2303.15343](https://arxiv.org/abs/2303.15343)）和 **SigLIP 2**（[arXiv:2502.14786](https://arxiv.org/abs/2502.14786)）：sigmoid 对比损失不需要全局 softmax，小 batch 也能稳定训练；SigLIP 2 是多语种的。
  - tidal：**T**。跨模态对齐用 sigmoid 损失，适合 CPU 上的小 batch 训练。
- **MiniCPM-V**（[arXiv:2408.01800](https://arxiv.org/abs/2408.01800)）和 **MiniCPM-V 4.5**（[arXiv:2509.18154](https://arxiv.org/abs/2509.18154)）：端侧 MLLM；4.5 版用 3D-Resampler 统一压缩图像和视频 token。
  - tidal：**E**。"用固定数量的查询把可变长视觉输入压成 K 个 token"，是让视频进入 tidal 事件流的标准接口。

### 3.2 小型音频-语言模型

- **Mellow**：Deshmukh 等（CMU / Microsoft），2025。[arXiv:2503.08540](https://arxiv.org/abs/2503.08540)：167M，MMAU 52.11，与 Qwen2-Audio 相当，参数少 50 倍；ReasonAQA 中 70% 是 LLM 合成的问答。
- **Mizar**：Li 等，2026（投稿 ICASSP 2027）。[arXiv:2609.28344](https://arxiv.org/abs/2609.28344)：159.3M，由 **CED-Small 音频编码器、频率合并映射器和 SmolLM2-135M** 组成，三阶段训练，**单 CPU 上平均 1.09 s 完成端到端问答**。
- **Samsone**：Masztalski 等，Interspeech 2026。[arXiv:2609.21666](https://arxiv.org/abs/2609.21666)：**99M / 134M / 356M** 一个系列，只用公开数据训练，有移动端权重和安卓 App。
- **TinyMU**：Li, Quelennec, Essid，ICASSP 2026。[arXiv:2604.15849](https://arxiv.org/abs/2604.15849)：229M 的音乐-语言模型，由 MATPAC++ 编码器、线性投影和小 LM 组成；MusicSkills-3.5M 数据集；在 MuChoMusic 上达到 SOTA 的 82%，规模小 35 倍。
  - tidal：**E / T**。说明"CED 级的小音频编码器加 100M 级的小 LM"是 2026 年小型音频理解的主流配方。tidal 不需要 LM 部分，**只取它们的音频编码器加投影层**（Mizar 的频率合并映射器值得借鉴）；它们的问答数据（ReasonAQA、MusicSkills）可以用来做音频嵌入和文本的对齐预训练。
  - 可行性：编码器部分高，整个 ALM 只能离线当教师。

### 3.3 Omni 模型（大），以及向下压缩的线索

- **Qwen2.5-Omni**（[arXiv:2503.20215](https://arxiv.org/abs/2503.20215)）：TMRoPE 用时间对齐的位置编码统一音频和视频；分块流式编码。
  - tidal：**B**。事件的位置编码应该用"真实时间"，而不是序号（tidal 已有 `log_gap_*` 特征，可以升级为连续时间编码）。
- **Qwen3-Omni**（[arXiv:2509.17765](https://arxiv.org/abs/2509.17765)）：多模态之间不互相拖累；覆盖音乐理解；30B-A3B。
  - tidal：**T**。可以当离线教师，给图片、视频、音乐附件生成描述和标签（**只用于合成或公开数据，真实聊天数据不出本机**）。
- **Gemma 3n**（[模型卡](https://ai.google.dev/gemma/docs/gemma-3n/model_card)）：端侧 omni 输入（文本、图像、视频、音频），音频为每秒 6.25 个 token，图像 256 个 token；用选择性参数激活实现 E2B / E4B 有效规模。
  - tidal：**E / D**。"音频 6.25 token/s"是很好的参考：**语音进入主干前要大幅降低帧率**。
- **MatFormer**（NeurIPS 2024，[arXiv:2310.07707](https://arxiv.org/abs/2310.07707)）：嵌套 FFN，一次训练就能抽取多个尺寸的子模型。
  - tidal：**D**。一次训练同时得到"常驻小档"和"高置信大档"。
- **Phi-4-Multimodal**（[arXiv:2503.01743](https://arxiv.org/abs/2503.01743)）：mixture-of-LoRAs 加模态路由，新增模态时不干扰原有模态。
  - tidal：**T**。新增模态时冻结主干，只训练该模态的适配器，避免灾难性遗忘。
- **Omni-C**（[arXiv:2603.05528](https://arxiv.org/abs/2603.05528)）：**单个稠密 ViT 共享编码图像、音频、文本**，每种模态有自己的 patch 嵌入和投影头，在无配对数据上做单模态对比预训练，共 111.9M。消融显示**共享投影头会让音频和文本混在一起，必须用模态专属头**。
  - tidal：**E**。如果想用一个共享小编码器省内存，这是现成证据，但要用模态专属的嵌入层和头；音频和文本的零样本性能下降较多，需要线性探针或 PEFT 来恢复。
- **Mini-Omni**（[arXiv:2408.16725](https://arxiv.org/abs/2408.16725)）和 **Mini-Omni2**（[arXiv:2410.11190](https://arxiv.org/abs/2410.11190)）：小主干的 omni，打断机制只靠口令。
  - tidal：反例。**口令式打断不够**，需要学出来的打断头。
- **ImageBind**（CVPR 2023，[arXiv:2305.05665](https://arxiv.org/abs/2305.05665)）和 **LanguageBind**（ICLR 2024，[arXiv:2310.01852](https://arxiv.org/abs/2310.01852)）：分别以图像或语言为锚点，把多种模态绑定到同一个嵌入空间。
  - tidal：**T**。以**文本嵌入（tidal 已有 bge）为锚**，把图片、音频、视频的小编码器各自对齐到 bge 空间。这样主干只需要认识一种空间，任意模态缺失时都退化为"只有文本"的情形。可行性高：只训练投影层。

### 3.4 高效视频理解：帧采样与 token 合并

- **Token Merging (ToMe)**：Bolya 等（Meta），ICLR 2023。[arXiv:2210.09461](https://arxiv.org/abs/2210.09461)：在 ViT 层间合并相似 token，不用训练即可加速。
- **FastV**（ECCV 2024，[arXiv:2403.06764](https://arxiv.org/abs/2403.06764)）：第 2 层之后剪掉一半视觉 token。
- **VisionZip**（[arXiv:2412.04467](https://arxiv.org/abs/2412.04467)）：只保留少量显著视觉 token。
- **LongVU**（[arXiv:2410.17434](https://arxiv.org/abs/2410.17434)）：先用 DINOv2 相似度去掉冗余帧，再做时空压缩。
- **DyCoke**（[arXiv:2411.15024](https://arxiv.org/abs/2411.15024)）、**FrameFusion**（ICCV 2025，[arXiv:2501.01986](https://arxiv.org/abs/2501.01986)）、**HoliTom**（[arXiv:2505.21334](https://arxiv.org/abs/2505.21334)）、**STTM**（ICCV 2025，[arXiv:2507.07990](https://arxiv.org/abs/2507.07990)）：2025 年免训练的视频 token 合并。其中 STTM 用四叉树做多粒度空间合并，再做有向时间合并，与查询无关，所以 KV 可以复用；FrameFusion 先按相邻帧相似度合并，再按重要性剪枝。
- **Adaptive Keyframe Sampling (AKS)**（CVPR 2025，[arXiv:2502.21271](https://arxiv.org/abs/2502.21271)）：在相关性和覆盖度之间折中选关键帧。
- **VideoMamba**（[arXiv:2403.06977](https://arxiv.org/abs/2403.06977)）：用 SSM 做视频编码，复杂度线性。
- **VideoMAE**（NeurIPS 2022，[arXiv:2203.12602](https://arxiv.org/abs/2203.12602)）：高掩码率的视频自监督预训练，数据效率高。
  - tidal：**F / E**。视频消息的推荐流水线是：①先用廉价的帧差或相似度去重（LongVU、FrameFusion 的思路）；②再均匀加关键帧采样 4–8 帧（AKS）；③用冻结图像塔编码每帧，与查询无关地合并 token（STTM、ToMe 的思路），最后得到 ≤ 16 个 token；④用 Resampler 或均值池化得到 1 个"视频事件嵌入"进入主干。全程与查询无关，所以嵌入可以缓存。
  - 可行性：高，只要不在 CPU 上做逐帧密集编码。实时视频流应把帧率降到 ≤ 1 fps（MiniCPM-o 4.5 的全双工块长也是 1 s）。

---

## 4. 方向三：CPU 上的语音与音乐前端

### 4.1 前端和流式语音编码器

- **Conformer**（Google，Interspeech 2020，[arXiv:2005.08100](https://arxiv.org/abs/2005.08100)）：卷积加自注意力，是语音编码器的标准块。
- **Squeezeformer**（NeurIPS 2022，[arXiv:2206.00888](https://arxiv.org/abs/2206.00888)）：U 形时间下采样，同等精度下 FLOPs 更低。
- **Zipformer**（ICLR 2024，[arXiv:2310.11230](https://arxiv.org/abs/2310.11230)）：多帧率 U-Net 式结构加 BiasNorm，更快更准。
- **Emformer**（ICASSP 2021，[arXiv:2010.10759](https://arxiv.org/abs/2010.10759)）：带记忆的分块流式 Transformer，延迟低。
  - tidal：**F / E**。语音分支的配置是 16 kHz、64–80 维 log-mel、10 ms 帧移，经过 4–8 倍时间下采样（Squeezeformer、Zipformer 的思路），再接 2–4 层小型流式 Conformer 或 Emformer（d≈144），参数 2–5M；最终给主干的帧率 ≤ 12.5 Hz（参考 Gemma 3n 的 6.25 token/s）。
  - 可行性：高。这一档在 CPU 上跑实时流没有问题。

### 4.2 现成小型 ASR 和语音编码器（可当教师或冻结编码器）

- **Whisper**（[arXiv:2212.04356](https://arxiv.org/abs/2212.04356)）和 **Distil-Whisper**（[arXiv:2311.00430](https://arxiv.org/abs/2311.00430)）：后者用大规模伪标签蒸馏，并按 WER 过滤。
  - tidal：**T**。"伪标签加质量过滤"可以直接用来给语音条生成转写。
- **Moonshine**（Useful Sensors，[arXiv:2410.15608](https://arxiv.org/abs/2410.15608)）：用 RoPE、不做零填充，10 s 片段的计算量比 Whisper tiny 低 5 倍，WER 不升。
  - tidal：**E / F**。语音条转写的 CPU 首选候选：转写后的文本走现有的 TS 路径。
- **FunAudioLLM / SenseVoice**（阿里，[arXiv:2407.04051](https://arxiv.org/abs/2407.04051)）：多语种 ASR，同时输出情感和音频事件，中文强。
  - tidal：**E / T**。中文语音条的转写加副语言标签（笑声、情绪），可以当教师。
- **HuBERT**（[arXiv:2106.07447](https://arxiv.org/abs/2106.07447)）和 **DistilHuBERT**（ICASSP 2022，[arXiv:2110.01900](https://arxiv.org/abs/2110.01900)）：逐层蒸馏，模型缩小 75%，SUPERB 性能大体保持。
  - tidal：**T**。自己蒸馏小语音编码器时，用"多层预测头加逐层蒸馏"的方案。

### 4.3 通用音频（含音乐）编码器

- **EfficientAT**：Schmid, Koutini, Widmer（JKU），ICASSP 2023。[arXiv:2211.04772](https://arxiv.org/abs/2211.04772)，代码 [EfficientAT](https://github.com/fschmid56/EfficientAT)：把 Transformer 集成离线蒸馏到 MobileNetV3，从极低复杂度一直覆盖到 AudioSet 0.483 mAP 的 SOTA。
- **Dynamic CNNs as Efficient Pre-trained Audio Models (DyMN)**（[arXiv:2310.15648](https://arxiv.org/abs/2310.15648)）：动态卷积，效率更高。
- **CED: Consistent Ensemble Distillation**：Dinkel 等（小米），ICASSP 2024。[arXiv:2308.11957](https://arxiv.org/abs/2308.11957)，代码 [CED](https://github.com/RicherMans/CED)：**把教师 logits 和增强参数一起存盘**（只多占 0.3% 磁盘），学生不需要标签；10M 的模型达到 AudioSet 49.0 mAP。Mizar 用的正是 CED-Small。
  - tidal：**E，音频和音乐分支首选**。用 CED 小档或 EfficientAT 的 mn 小档做冻结编码器，输出 AudioSet 527 类的 logits 加嵌入。音乐、笑声、掌声、环境声都能覆盖。也可以按 CED 的做法自己蒸馏一个更小的版本。
  - 可行性：很高。卷积或小 ViT 在 CPU 上处理一段 10 s 音频只需几十毫秒量级。
- **BEATs**（Microsoft，[arXiv:2212.09058](https://arxiv.org/abs/2212.09058)）、**EAT**（[arXiv:2401.03497](https://arxiv.org/abs/2401.03497)）、**Dasheng**（Interspeech 2024，[arXiv:2406.06992](https://arxiv.org/abs/2406.06992)，1.2B，272k 小时）：大型通用音频 SSL 编码器。
  - tidal：**T**。只当教师。
- **USAD: Universal Speech and Audio Distillation**：Chang 等（MIT），ASRU 2025。[arXiv:2506.18843](https://arxiv.org/abs/2506.18843)：把语音 SSL 教师和通用音频 SSL 教师**逐层蒸馏到同一个学生**，在 SUPERB 和 HEAR 上都接近 SOTA。
  - tidal：**T，中期目标**。用一个小学生同时服务语音和音乐，省掉两个编码器；短期先用现成的 CED 或 EfficientAT。
- **CLAP**（Microsoft，[arXiv:2206.04769](https://arxiv.org/abs/2206.04769)）和 **LAION-CLAP**（[arXiv:2211.06687](https://arxiv.org/abs/2211.06687)）：音频-文本对比学习。
  - tidal：**T**。音频嵌入与 bge 文本锚对齐时，可以用 CLAP 的 caption 数据或教师。
- **MERT**（ICLR 2024，[arXiv:2306.00107](https://arxiv.org/abs/2306.00107)）和 **MuQ**（[arXiv:2501.01108](https://arxiv.org/abs/2501.01108)）：音乐自监督表示，后者用 Mel-RVQ 作为目标。
  - tidal：**T**。音乐教师（风格、情绪、乐器），只离线使用。
- **Audio Mamba**（[arXiv:2406.03344](https://arxiv.org/abs/2406.03344)）：用双向 SSM 做音频表示。
  - tidal：**E**。可选项；CPU 上 CNN 一般更省事。

---

## 5. 方向四：适合微型 CPU 推理的高效架构

- **Mamba**（Gu & Dao，[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)）和 **Mamba-2 / SSD**（ICML 2024，[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)）：选择性状态空间，推理时状态大小固定。
- **RWKV-7 "Goose"**（[arXiv:2503.14456](https://arxiv.org/abs/2503.14456)）：带动态状态演化的 RNN，训练可并行，推理是 O(1) 状态。
- **Griffin**（DeepMind，[arXiv:2402.19427](https://arxiv.org/abs/2402.19427)）：门控线性递归加局部注意力。
- **xLSTM**（[arXiv:2405.04517](https://arxiv.org/abs/2405.04517)）：带指数门控的 LSTM，以及矩阵记忆的 mLSTM。
- **Gated DeltaNet**（ICLR 2025，[arXiv:2412.06464](https://arxiv.org/abs/2412.06464)）和 **Kimi Linear / KDA**（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）：delta 规则加门控的线性注意力；Kimi Linear 的混合架构首次在公平对比下超过全注意力。
- **Hymba**（NVIDIA，[arXiv:2411.13676](https://arxiv.org/abs/2411.13676)）：同一层里并行放注意力头和 SSM 头，面向小模型。
- **LFM2 Technical Report**（Liquid AI，[arXiv:2511.23404](https://arxiv.org/abs/2511.23404)）
  - 要点：**硬件在环的架构搜索**得到"多数是门控短卷积加少量 GQA"的混合结构，CPU 上 prefill 和 decode 最多快 2 倍；训练用解耦的 Top-K 蒸馏、**难度排序的课程**和模型合并。
  - tidal：**B / T，架构首选参考**。在 CPU 上，"门控短卷积（因果 depthwise conv）加 1–2 层注意力"比纯 SSM 更省事，ONNX 兼容性也好。tidal 的主干可以做成 4 层门控短卷积加 1 层局部注意力（d=128–192）。
- **Were RNNs All We Needed? (minGRU / minLSTM)**：Feng 等（Mila / Borealis），[arXiv:2410.01201](https://arxiv.org/abs/2410.01201)
  - 要点：去掉门对隐状态的依赖后，GRU 和 LSTM 可以用并行扫描训练，性能接近 Mamba。
  - tidal：**B，迁移成本最低**。现有 GRU 可以直接换成 minGRU：训练可并行（CPU 上训练更快），推理仍是 O(1) 状态，ONNX 导出简单。
- **Efficient Streaming LMs with Attention Sinks (StreamingLLM)**（ICLR 2024，[arXiv:2309.17453](https://arxiv.org/abs/2309.17453)）：保留开头几个"汇聚 token"，滑窗注意力就能稳定无限流。
  - tidal：**B**。如果保留一层注意力，就用"会话摘要 token（汇聚）加最近 64 个事件"的滑窗。
- **Vision Transformers Need Registers**（[arXiv:2309.16588](https://arxiv.org/abs/2309.16588)）：加 register token 吸收高范数伪影。
  - tidal：**B**。给主干留 1–2 个可学习的"全局状态 token"，作用同上。
- **MobileLLM**（ICML 2024，[arXiv:2402.14905](https://arxiv.org/abs/2402.14905)）：亚 1B 规模下，"深而窄、嵌入共享、GQA"更好。
  - tidal：**B**。小预算下优先加深度，而不是加宽度。
- **Emformer**（见 4.1）和 **DuplexMamba**（见 2.4）：流式和双工场景的具体例子。

**结论（主干）**：第一阶段真正卡住的是数据，不是主干容量（GRU 和 Transformer 的差距在噪声范围内）。所以主干升级排在中后段：优先 **minGRU**（改动最小），其次是 **LFM2 式门控短卷积加 1 层局部注意力**。Mamba、RWKV、DeltaNet 在 CPU 和 ONNX 上的工具链成本更高，对 ≤ 5M 的主干收益不明显，暂缓。

---

## 6. 方向五：小模型的训练技术

### 6.1 蒸馏

- **Distilling the Knowledge in a Neural Network**（Hinton 等，2015，[arXiv:1503.02531](https://arxiv.org/abs/1503.02531)）：用软标签和温度做蒸馏。
- **Knowledge distillation: A good teacher is patient and consistent**（Beyer 等，CVPR 2022，[arXiv:2106.05237](https://arxiv.org/abs/2106.05237)）：教师和学生看**同一个增强视图**，训练要足够长。
- **On-Policy Distillation (GKD)**（Agarwal 等，ICLR 2024，[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)）：在学生自己产生的序列上蒸馏，减少训练和推理的分布不一致。
- **Distillation Scaling Laws**（Apple，ICML 2025，[arXiv:2502.08606](https://arxiv.org/abs/2502.08606)）：已有教师、或者要蒸馏多个学生时，蒸馏优于监督训练，直到某个随学生规模可预测的计算量为止。
- **Minitron: Compact LMs via Pruning and KD**（NVIDIA，[arXiv:2407.14679](https://arxiv.org/abs/2407.14679)）：先结构化剪枝，再蒸馏。
  - tidal：**T**。
    - ①**轮次决策的教师**：用 LLM 对"说/不说/被叫/会不会续发"给出**软概率加理由类别**（Speak or Stay Silent 的推理痕迹 SFT、DiscussLLM 的合成管线），tidal 用 KL 学软标签。**隐私约束**：第一阶段的原则是真实聊天不出本机、不发给外部 API，所以教师标注只能用于合成数据、公开语料（AMI、Friends、SPGI、MUIR），或者在本机运行的开源教师。
    - ②"patient and consistent"：音频和图像分支的蒸馏要用一致的增强，并离线存下教师输出（和 CED、MobileCLIP 的做法一致）。
    - ③GKD 的思想：tidal 上线前先在影子模式下跑，把它自己的决策序列（例如"它决定等待之后的后续事件"）交给教师复标，减少暴露偏差。

### 6.2 自监督与跨模态对比预训练

- **CLIP**（[arXiv:2103.00020](https://arxiv.org/abs/2103.00020)）、**SigLIP**（见 3.1）、**ImageBind / LanguageBind**（见 3.3）、**VATT**（NeurIPS 2021，[arXiv:2104.11178](https://arxiv.org/abs/2104.11178)，用原始视频、音频、文本做多模态自监督）。
- **MAE**（[arXiv:2111.06377](https://arxiv.org/abs/2111.06377)）、**VideoMAE**（见 3.4）、**HuBERT**（见 4.2）：掩码预测式自监督。
- **Matryoshka Representation Learning**（NeurIPS 2022，[arXiv:2205.13147](https://arxiv.org/abs/2205.13147)）：一个嵌入同时在多个前缀维度上都有效。
  - tidal：**T / D**。
    - ①**VAP 式未来事件投影**（见 2.1），作为主干在全部时间戳数据上的自监督目标，优先级最高；
    - ②**掩码事件建模**：随机遮住某条消息的文本或模态嵌入，让主干从上下文去重建它（MPC-BERT 的回复对象预测也属于这一类）；
    - ③图片、音频、视频的嵌入用 **sigmoid 对比损失对齐到 bge 文本锚**（只训练投影层）；
    - ④主干输入嵌入采用 Matryoshka，部署时可以截断维度换速度。

### 6.3 合成数据与课程

- **TinyStories**（[arXiv:2305.07759](https://arxiv.org/abs/2305.07759)）、**Textbooks Are All You Need**（[arXiv:2306.11644](https://arxiv.org/abs/2306.11644)）：用高质量、受控的合成数据，可以让很小的模型学会特定能力。
- **SmolLM2**（[arXiv:2502.02737](https://arxiv.org/abs/2502.02737)）：以数据为中心，多阶段调整配比。
- 合成轮次数据的具体配方：**SyncLLM**（由文本对话合成语音对话，见 2.4）、**DiscussLLM**（两阶段合成介入，见 2.5）、**SpeculativeETD**（TTS 合成停顿，见 2.3）、**Mellow**（70% 由 LLM 合成问答，见 3.2）、**Prompt-Guided VAP**（LLM 合成提示，见 2.2）。
- 课程：**LFM2**（难度排序数据，见第 5 节）、**OmniFlatten**（先半双工再全双工，见 2.4）。
  - tidal：**T**。第一阶段发现合成数据对"被叫"和"说/等/不说"**有害**，对 EOT 和续话有小幅帮助。据此建议：
    - ①合成数据按头分别加权，或者只给部分头用（"被叫"和"说/等/不说"的合成数据先关掉，或换成 DiscussLLM / MUCA 式更真实的生成）；
    - ②课程顺序：先在合成数据和公开多方语料上预训练（只用 VAP 投影和 EOT 损失），再在真实数据上微调全部头；
    - ③合成数据里显式包含"不该说"的负样本（MP-Bench 的负向轮次）。

### 6.4 多任务

- **Multi-Task Learning Using Uncertainty to Weigh Losses**（Kendall 等，CVPR 2018，[arXiv:1705.07115](https://arxiv.org/abs/1705.07115)）：用可学习的同方差不确定性自动给各任务损失加权。
  - tidal：**T**。6 个以上的头加辅助损失，手工调权成本高；用它自动加权，成本只是几个标量参数。

### 6.5 量化、QAT、剪枝

- **Quantization and Training of NNs for Integer-Arithmetic-Only Inference**（Jacob 等，Google，CVPR 2018，[arXiv:1712.05877](https://arxiv.org/abs/1712.05877)）：int8 推理和伪量化训练的标准做法。
- **Learned Step Size Quantization (LSQ)**（ICLR 2020，[arXiv:1902.08153](https://arxiv.org/abs/1902.08153)）：可学习的量化步长，低比特 QAT 的强基线。
- **GPTQ**（ICLR 2023，[arXiv:2210.17323](https://arxiv.org/abs/2210.17323)）和 **AWQ**（MLSys 2024，[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)）：大模型的训练后权重量化。
- **BitNet b1.58**（[arXiv:2402.17764](https://arxiv.org/abs/2402.17764)）、**ParetoQ**（NeurIPS 2025，[arXiv:2502.02631](https://arxiv.org/abs/2502.02631)，研究极低比特的缩放规律）。
- **Squat: Quant Small Language Models on the Edge**（ICCAD 2025，[arXiv:2402.10787](https://arxiv.org/abs/2402.10787)）：面向移动端 SIMD 的小模型 QAT。
- **Wanda**（ICLR 2024，[arXiv:2306.11695](https://arxiv.org/abs/2306.11695)）、**Sheared LLaMA**（[arXiv:2310.06694](https://arxiv.org/abs/2310.06694)）：剪枝。
  - tidal：**D**。
    - ①主干和头先用 ONNX Runtime 的**动态 int8**（不需要再训练），如果 parity 或指标下降，再用 LSQ 式 QAT 微调几个 epoch；
    - ②编码器（CED、MobileCLIP、Whisper-tiny 级）也做 int8，Smart Turn v3 的 8 MB int8 ONNX 就是先例；
    - ③int4 和 1.58-bit 对 ≤ 5M 的主干收益很小（内存本来就只有几 MB），而且 CPU 上缺少高效 kernel，**暂不建议**；
    - ④剪枝只在编码器上考虑（Minitron 式结构化剪枝加蒸馏）。

### 6.6 校准与弃权

- **On Calibration of Modern Neural Networks**（Guo 等，ICML 2017，[arXiv:1706.04599](https://arxiv.org/abs/1706.04599)）：温度缩放。
- **A Gentle Introduction to Conformal Prediction**（Angelopoulos & Bates，[arXiv:2107.07511](https://arxiv.org/abs/2107.07511)）：与分布无关的覆盖保证。
- **SelectiveNet**（ICML 2019，[arXiv:1901.09192](https://arxiv.org/abs/1901.09192)）：网络内置拒绝选项，在给定覆盖率下优化风险。
  - tidal：**H / D，高优先级且便宜**。
    - ①每个头在 val 上做温度缩放，报告 ECE；
    - ②`y_act` 用保形或选择性预测给出"有把握说 / 有把握不说 / 没把握"三态，**没把握时映射为"等待加 `y_recheck` 给出的重检时间"或升级给 LLM**，这正是 SpeculativeETD 和 GroupGPT 的级联逻辑；
    - ③对数据量小、方差大的 `y_addr` 头，要求在给定覆盖率下精度 ≥ 目标值，达不到就退回规则（@ 或回复即视为被叫）。

---

## 7. 推荐的 tidal 目标结构（草图，按依据组织）

> 这是第一阶段写的目标草图（当时建议换成 minGRU）。已经落地的结构见 [architecture.md](architecture.md)：四个被点名主干里，第四阶段胜出的是 Mamba-3 SISO；影子配置现在加载 `m3_ablate_m2`（`mamba3_siso` 仍是主干选项）。本节不描述现状。

```
事件流（消息 / 语音帧 / 视频帧，带真实时间戳）
 ├─ F: 时间节奏特征（已有） + 连续时间编码（TMRoPE 思路）
 ├─ E(文本): bge（已有，冻结、缓存）── 锚空间
 ├─ E(图片): MobileCLIP2 小档图像塔（冻结）→ sigmoid 对比投影到 bge 空间
 ├─ E(音频/音乐): CED / EfficientAT 小档（冻结）→ logits + 嵌入 → 投影
 ├─ E(语音): Silero VAD → Moonshine / SenseVoice 转写（走文本路径）
 │            + 小型流式 Conformer（≤ 5M，≤ 12.5 Hz）供实时 EOT / 打断使用
 ├─ E(视频): 帧去重 + AKS 采样 4–8 帧 → 图像塔 → STTM / ToMe 合并 → Resampler → 1 个事件嵌入
 └─ B: minGRU（或 LFM2 式门控短卷积 + 1 层局部注意力），模态缺失随机丢弃，显式边界 token
     └─ H: VAP 式未来事件投影（自监督主目标）
           + EOT 四态（完整 / 不完整 / 附和 / 等待，按 Easy Turn）+ 续话 + 被叫 + 说/等/不说
           + 重检时长分布 + 打断（语音）
           → 温度缩放 + 选择性弃权 → 不确定时升级给 LLM（端云级联）
```

---

## 8. 优先采纳清单（按顺序）

> 排序依据：①能否直接缓解第一阶段的头号瓶颈，也就是真实数据少、文本覆盖率低、被叫正例极少；②能否在纯 CPU 和现有代码规模内低成本落地；③有没有 2025–2026 年的直接证据支持。

1. **VAP 式"未来事件投影"作为自监督主目标**（[VAP](https://arxiv.org/abs/2205.09812)，[实时 VAP](https://arxiv.org/abs/2401.04868)，[附和 VAP](https://arxiv.org/abs/2410.15929)）。**组件**：H / T。
   - 为什么：只需要时间戳和角色，就能用上**全部** 10,973 条消息（包括 71% 没有文本的消息），不受文本选择性泄漏的影响；续话、接话、重检都能从同一个投影分布里推出来；"先通用预训练、再小数据微调"在附和预测上已有证据。
   - 成本：一个输出头加标签构造。

2. **用外部多方语料和 LLM 教师来补"被叫"和"说/不说"**（[Speak or Stay Silent](https://arxiv.org/abs/2603.11409)，[DiscussLLM](https://arxiv.org/abs/2508.18167)，[GroupGPT / MUIR](https://arxiv.org/abs/2603.01059)，[MPC-BERT](https://arxiv.org/abs/2106.01541)）。**组件**：T / V。
   - 为什么：`y_addr` 测试正例只有个位数，统计上没法评估。12 万+ 个公开决策点，加上"推理痕迹蒸馏能带来 +23 个百分点"的证据，加上 MUIR 的群聊介入标签，能同时解决训练量和外部可比评测的问题。
   - 约束：教师标注**只用于合成数据、公开语料或本机开源模型**，真实聊天不外发。

3. **校准加选择性弃权，把"没把握"变成"等待或重检"、或者升级给 LLM**（[温度缩放](https://arxiv.org/abs/1706.04599)，[保形预测](https://arxiv.org/abs/2107.07511)，[SelectiveNet](https://arxiv.org/abs/1901.09192)；级联依据 [SpeculativeETD](https://arxiv.org/abs/2503.23439)）。**组件**：H / D。
   - 为什么：小数据下点估计不可靠，但"知道自己不知道"可以立刻提升线上安全性；成本几乎为零；也和"影子模式到逐步放量"的上线路线吻合。

4. **轮次控制与内容生成彻底解耦，"说不说"作为独立的头，并用显式事件边界**（[MiniCPM-o 4.5 的 LS > LT 消融](https://arxiv.org/abs/2604.27393)，[LLM 增强 DM](https://arxiv.org/abs/2502.14145)，[Time to Talk](https://aclanthology.org/2025.findings-emnlp.608/)，[Gander](https://arxiv.org/abs/2609.08977)）。**组件**：H / D。
   - 为什么：9B 模型的消融也证明"先判听/说、再生成"更稳，这恰好是 tidal 这种微型控制器存在的理由。同时把"机器人近期发言占比"作为调节条件（Mafia 论文的自我调节）。

5. **多模态通过"冻结小编码器、离线缓存嵌入、投影到 bge 文本锚、训练时随机丢弃模态"接入**（[ImageBind](https://arxiv.org/abs/2305.05665) / [LanguageBind](https://arxiv.org/abs/2310.01852)，[SigLIP](https://arxiv.org/abs/2303.15343)，[MobileCLIP2](https://arxiv.org/abs/2508.20691)，[CED](https://arxiv.org/abs/2308.11957)，[MM-F2F 任意模态组合](https://aclanthology.org/2025.acl-long.743/)，[Omni-C 模态专属头](https://arxiv.org/abs/2603.05528)）。**组件**：E / T。
   - 为什么：这是在 ≤ 100M、纯 CPU 下实现"原生多模态"最省的方式：主干只认一种空间，缺任何模态都能退化为"只有文本"的情形；只训练投影层，数据需求小。
   - **第一步最便宜**：先让离线教师给图片、语音条、视频、音乐生成描述或转写，走现有的文本路径，再逐步换成嵌入。

6. **视频和音乐的按需前端：关键帧采样加免训练 token 合并；音频用 CED 或 EfficientAT 小档**（[AKS](https://arxiv.org/abs/2502.21271)，[STTM](https://arxiv.org/abs/2507.07990)，[FrameFusion](https://arxiv.org/abs/2501.01986)，[ToMe](https://arxiv.org/abs/2210.09461)，[EfficientAT](https://arxiv.org/abs/2211.04772)，[Mizar 的 CED-Small 配方](https://arxiv.org/abs/2609.28344)）。**组件**：F / E / D。
   - 为什么：视频是 CPU 预算的最大风险，必须在进入编码器之前就把帧数和 token 数压到个位或十位；CED 和 EfficientAT 一个模型就覆盖语音以外的声音和音乐，有 2026 年 159M 级 ALM 用它在 CPU 上跑通的证据。

7. **主干换成 minGRU，必要时演进到 LFM2 式"门控短卷积加 1 层局部注意力"；加多任务不确定性加权**（[minGRU](https://arxiv.org/abs/2410.01201)，[LFM2](https://arxiv.org/abs/2511.23404)，[Kendall 2018](https://arxiv.org/abs/1705.07115)）。**组件**：B / T。
   - 为什么：minGRU 和现有 GRU 几乎无缝衔接，训练可以并行，推理仍是 O(1) 状态；LFM2 是 2025 年唯一用硬件在环在 CPU 上搜出来的小模型结构；头变多以后需要自动加权。
   - 排在第 7：第一阶段的证据表明当前瓶颈不在主干。

8. **部署：ONNX 动态 int8 起步，需要时加 LSQ 式 QAT；用 MatFormer 或 Matryoshka 做弹性档位**（[Jacob 2018](https://arxiv.org/abs/1712.05877)，[LSQ](https://arxiv.org/abs/1902.08153)，[MatFormer](https://arxiv.org/abs/2310.07707)，[Matryoshka](https://arxiv.org/abs/2205.13147)；先例 [Smart Turn v3 int8](https://huggingface.co/pipecat-ai/smart-turn-v3)）。**组件**：D。
   - 为什么：编码器一旦加入，内存和延迟主要花在编码器上，int8 是最稳的收益；弹性档位让"常驻小档"和"难例大档"共用一套权重。

**暂缓或不建议**：
- 直接训练或部署端到端 omni 生成模型（Moshi、MiniCPM-o 一类），在 CPU 和 100M 预算下不可行，也偏离"控制器"的定位；
- 对 ≤ 5M 的主干做 int4 或 1.58-bit，收益很小，工具链成本高；
- 第一阶段证明有害的那类合成数据，不要再直接用于"被叫"和"说/等/不说"这两个头；
- 视频逐帧密集编码。

---

## 附录 A：外部评测与数据资源速查

| 资源 | 用途（tidal 头） | 链接 |
|---|---|---|
| Speak or Stay Silent（AMI / Friends / SPGI，12 万+ 决策点） | `y_act`、`y_addr` 的预训练和评测 | [arXiv:2603.11409](https://arxiv.org/abs/2603.11409) |
| MUIR（2,500 段群聊介入标注） | `y_act` 的外部测试 | [arXiv:2603.01059](https://arxiv.org/abs/2603.01059) |
| MP-Bench | 多方语音轮次，含负向轮次 | [arXiv:2609.13076](https://arxiv.org/abs/2609.13076) |
| Easy Turn trainset（1,145 h，中文四态） | 语音 EOT | [arXiv:2509.23938](https://arxiv.org/abs/2509.23938) |
| ETD Dataset（SpeculativeETD） | 语音停顿和说完的区分 | [arXiv:2503.23439](https://arxiv.org/abs/2503.23439) |
| Full-Duplex-Bench v1 / v2、Talking Turns、FLEXI、HumDial | 语音全双工行为评测 | 见 2.8 |
| Friends-MMC | 视频多方对话：说话人与受话人 | [arXiv:2412.17295](https://arxiv.org/abs/2412.17295) |
| 全双工综述的状态机和交互本体 | 动作空间与评测切片 | [arXiv:2606.19453](https://arxiv.org/abs/2606.19453) |

（使用任何外部数据前，需逐一确认许可证。）

## 附录 B：核验说明

- arXiv 条目：2026-10-08 通过 `https://export.arxiv.org/api/query?id_list=…` 批量获取标题、作者、首次提交日期和注释（会议信息取自作者注释或论文页）。
- ACL Anthology 条目：抓取官方页面的 `<title>` 和 `citation_author` 进行核对。
- 非论文资源：Gemma 3n 模型卡、Smart Turn v3 模型卡、LiveKit 博客已阅读原文；文中出现的 GitHub / HF 链接均返回 HTTP 200。HF 上的参数量（Qwen2.5-Omni-3B 约 5.54B，Moshi 约 7.69B，Gemma 3n E2B 约 5.44B，SmolVLM2-256M 约 256M，LFM2-Audio 约 1.47B）取自 HF API 的 `safetensors.total` 字段。
- 规模写作"LLM 级"的条目，是因为原文摘要没有给出确切参数量，为避免误报，没有填具体数字。

---

## 第四阶段补充（2026-10-08，UTC+8）：主干对比、情绪头与持续记忆

> 核验方式同上：下列 arXiv 条目于 2026-10-08 通过 `export.arxiv.org` API 核对标题、作者和日期。

### 4.1 小型 SSM / 线性 RNN 主干（组件 B）

- **Mamba-3**：Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu，*Mamba-3: Improved Sequence Modeling using State Space Principles*，arXiv:2603.15569（2026-03，ICLR 2026）。官方代码：`state-spaces/mamba` 的 `mamba_ssm/modules/mamba3.py`，参考实现为 `tests/ops/triton/test_mamba3_siso.py`（`mamba3_siso_step_ref` / `_fwd_ref`）和 `tests/ops/tilelang/test_mamba3_mimo.py`（`mamba3_MIMO_step_ref`）。论文的三项核心改动：
  1. 指数-梯形离散化（Prop. 1），数据相关的 λ；
  2. 通过数据相关 RoPE 实现复值状态（Prop. 4）；
  3. MIMO（秩 R）。
  
  此外还有 BC/QK RMSNorm、头级 B/C 偏置（初始化为 1），用于取代 short conv；块布局为 Llama 式交替的 Mamba-3 / SwiGLU，采用 pre-norm。
  - **tidal 实现**：`tidal/backbones.py`，纯 PyTorch、CPU 上运行。按官方参考逐项复现上述机制，以及 heavy-tail A、softplus DT 和官方初始化。并行形式有两种：单块二次形式（与官方 fwd_ref 相同），以及 SSD 分块形式（块内二次、块间传递状态）。`step()` 为精确递推，每事件 O(1)。`tests/test_phase4.py` 验证了以下等价性：并行形式 = 逐步递推（SISO 与 MIMO，float64，误差 1e-9）；分块形式 = 二次形式；手写官方 SISO step 公式 = 本实现；左填充不变性。
  - **为 CPU 所做的简化**（不改变数学）：
    - 全程 fp32；
    - 窗口 ≤ 64，因此不用 Triton/TileLang 核；
    - 角度不做 mod 2π，这只影响低精度核；
    - MIMO 也用相邻成对旋转，官方 MIMO 核为 rotate-half，两者只差一个通道置换；
    - ngroups = 1，与论文 MVA 布局一致。
  - **消融 `m3_ablate_m2`**：λ = 1（指数-Euler）、不用 RoPE、R = 1。这样得到的是 Mamba-2（SSD）递推，但保留了 Mamba-3 的 BCNorm 和偏置，所以**不是**原版 Mamba-2 模块，仅作为廉价对照。
- **Mamba-2 / SSD**：Dao & Gu，*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*，arXiv:2405.21060（2024）。
- **RWKV-7 "Goose"**：Peng 等，arXiv:2503.14456（2025）；**RetNet**：Sun 等，arXiv:2307.08621（2023）。线性注意力 RNN 一类，本阶段未实现。
- **设备端小 SSM**：
  - MambaLite-Micro（Xu 等，arXiv:2509.05488，2025）：在 MCU 上运行 Mamba 做 KWS/HAR，峰值内存降低 83%；
  - Keyword Mamba（Ding, Dong, Mao，arXiv:2508.07363，2025）：在 KWS 上以更少参数优于 KWT。
  
  两者都说明小 SSM 在边缘端可行，但同时指出，官方实现依赖 GPU 核，导出困难。这与本仓库的经验一致：MIMO R=4 在 CPU 上的训练成本约为 GRU 的 4 倍，见 `reports/phase4.md`。

### 4.2 轮次 / VAP（组件 H、V）

- **VAP**：Ekstedt & Skantze，arXiv:2205.09812（2022）。
- **实时与多语 VAP**：Inoue 等，arXiv:2401.04868（2024）和 arXiv:2403.06487（2024）。
- **噪声鲁棒的实地系统**：Inoue 等，arXiv:2503.06241（2025）。
- **多模态 VAP**：Saga & Pelachaud，arXiv:2506.03980（2025）。

目前没有找到把 Mamba 系主干直接用于 VAP 并做系统比较的已发表工作。本阶段的主干对比只是一个小规模、事件级的经验点，不能推广。

### 4.3 情绪（组件 E、H）

- **GoEmotions**：Demszky 等，arXiv:2005.00547（2020），Apache-2.0。映射到 8 类时采用论文中的 Ekman 分组。
- **BRIGHTER**：Muhammad 等，arXiv:2502.11926（2025），CC BY 4.0，人工标注。用作中文和英文的主评测。
- **SenseVoice / FunAudioLLM**：An 等，arXiv:2407.04051（2024）。SenseVoice-Small 采用 FunASR Model Open Source License v1.1，只作为**离线教师**，在公开音频上打情绪标签。本仓库不分发它的权重。署名：SenseVoice-Small，FunASR / 通义实验室。
- **CREMA-D**：Cao 等，IEEE Trans. Affective Computing 2014，ODbL。

### 4.4 持续记忆 / 测试时学习（第四阶段 C）

- **Nested Learning / Hope / 连续记忆系统（CMS）**：Behrouz, Razaviyayn, Zhong, Mirrokni，*Nested Learning: The Illusion of Deep Learning Architectures*，NeurIPS 2025，arXiv:2512.24695。CMS 是一串 MLP 块，第 ℓ 块每 C^(ℓ) 步更新一次（式 70–71）。初始状态在更低频的层级中学习，或者直接用预训练权重初始化（§7.3 "Ad-hoc level stacking"）。
- **Titans**：Behrouz, Zhong, Mirrokni，*Titans: Learning to Memorize at Test Time*，arXiv:2501.00663（2024-12）。用"惊讶度"（损失对输入的梯度）、动量和遗忘门来更新神经记忆。
- **注意**：两篇论文的结果都是在 LLM 规模（数亿到十亿级参数、语言建模）上得到的。tidal 的实验在约 50 万参数的事件级模型上进行，信号是延迟约 2 s 才能观测到的 VAP 目标，并不验证原论文的结论。
