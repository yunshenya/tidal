# tidal 潮汐

**English.** tidal is a tiny (≤ 2M resident parameters), CPU-only model for conversational turn-taking in group and
private chats. It predicts when an utterance is finished, whether the speaker will add more, whether the bot is being
addressed, and whether the bot should speak, wait or stay silent. The design goal is a *natively multimodal* model:
video, images, music, speech and text, with one shared tiny causal encoder and per-modality front ends. **Status today:**
only timing and text inputs are implemented, plus content-free message-type metadata (image, voice, video, sticker…).
The image and video front ends are designed and scaffolded but not yet implemented; phase 3 added a real streaming
audio front end (~73k parameters, ~0.7 ms per 100 ms on one CPU thread). Phase 2 added a VAP-style
self-supervised future-event projection objective, temperature calibration + selective abstention, identity-free roles
with cold-start / online-adaptation evaluation, a scenario-general event stream (no scenario enum), a 100 ms tick-based
full-duplex control loop (control separate from content; runs at well over 10 Hz on one CPU thread), ONNX export with
parity checks, and side-by-side shadow scoring. Phase 3 added loaders for license-checked public datasets (multi-party
chat, Twitch / live chat, Bilibili danmaku, spoken turn-taking, full-duplex speech; data is never redistributed),
public-data pretraining, leave-one-scenario-out on real livestream chat, the audio front end and int8 quantization.
No phase showed a stable, significant win over strong baselines on
real held-out data; results are reported with block-bootstrap CIs, including the negative ones. A shadow-mode harness
(log-only, never acts) collects fresh evaluation data. MIT licensed. No real chat data and no weights trained on real
chat are published.

---

## 为什么做 tidal

能同时原生处理**视频、图像、音乐、语音和文本**，又会自己判断**什么时候该说话、什么时候该沉默**的模型，目前市面上很少，几乎找不到。尺寸小到能在 CPU 上常驻的就更少了。现有的方案大致分两类：
- 大型全双工或多模态模型：能力强，但体积大，往往依赖 GPU；
- 很小的轮次检测器：多数只看语音或只看文本，而且只回答"这句话说完没有"。

tidal 想填的就是这个空档。详细的对比和文献见 [docs/related-work.md](docs/related-work.md)。

## 现在能做什么、不能做什么

| 能力 | 状态 |
|---|---|
| 时间节奏 + 角色特征，因果事件流主干（约 0.5M 参数），6 个任务头 | ✅ 已实现 |
| 文本：冻结的 bge-small-zh（int8 ONNX）句向量 | ✅ 已实现 |
| 消息类型元数据：图片、语音、视频、文件、贴纸、表情、分享（只从占位符解析，不读内容） | ✅ 已实现采集，尚未作为模型输入 |
| 音频前端：流式 log-mel → 因果卷积 + GRU（约 7.3 万参数），双方语音活动、换人 / 保持、附和；已接入 tick 循环 | ✅ 第三阶段 |
| 图像、视频帧、音乐前端 | 📋 计划中（接口已定义，见 `tidal/modalities/`） |
| 公开数据加载 / 转换（多人群聊、直播聊天、B 站弹幕、口语轮次、全双工语音）；公开数据预训练 | ✅ 第三阶段（数据不再分发） |
| 情绪：文本情绪头（冻结 bge + MLP）、语音情绪头（音频编码器 + GRU）；文本情绪已作为事件特征接入 | ✅ 第四阶段（语音情绪只在影子接口输出） |
| 主干可选：GRU、RoPE + KV 缓存 Transformer、Mamba-3 SISO / MIMO（纯 PyTorch CPU 实现）、Mamba-2 式消融；全部可导出 ONNX 流式单步 | ✅ 第四阶段 |
| 连续体记忆（Nested Learning CMS 式多时间尺度在线适配器） | 🧪 第四阶段，只做影子回放 |
| 校准（温度缩放）、ONNX 导出和一致性测试、泄漏审计、带 CI 的基线对比 | ✅ |
| VAP 式未来事件投影（只用时间和相对角色的自监督目标），预训练 → 多任务微调 | ✅ 第二阶段 |
| 选择性弃权（风险-覆盖曲线、"等一下再看"策略）、ECE 报告 | ✅ 第二阶段 |
| 身份无关的相对角色；冷启动评测（加入后前 20 / 50 / 100 条）和按会话在线重新校准 | ✅ 第二阶段 |
| 场景通用的统一事件流（没有场景枚举，参与人数是连续特征）、留一场景评测 | ✅ 第二阶段（直播和 1:1 流是脚本合成的） |
| 全双工 tick 控制循环（100 ms，多路输入，她自己的输出也是输入，控制与内容生成分离） | 🚧 控制骨架 + 合成 sanity 测试 + CPU 基准；真实音频流已接入，tick 策略仍只在模拟器上训练；视觉仍是接口 |
| 影子模式（只记录不执行，延迟打标签，定时任务，报告；新旧模型在同一批标签上并行打分） | ✅ |
| int8 动态量化（ONNX Runtime） | ✅ 第三阶段（这个尺寸上收益很小） |
| 蒸馏、剪枝、LoRA、保形预测 | 📋 计划中 |

完整的设计、参数预算和"技术 → 组件 → 状态"对照表见 **[docs/architecture.md](docs/architecture.md)**。

**第一阶段的结论**（[reports/phase1.md](reports/phase1.md)，已脱敏）：模型在任何一个头上都没有稳定、显著地超过最强基线。多数头的 Δ 为正，但 95% CI 跨过 0。真实数据太少，覆盖也不完整。所以下一步是先跑影子模式、补数据，**在 CI 显著优于基线之前，不接管任何真实决策**。

**第二阶段的结论**（[reports/phase2.md](reports/phase2.md)，已脱敏）：能力补齐了，但效果没有实质提升。VAP 投影能学，却没有超过同特征上的逻辑回归；零样本读出在"自我续话"上显著超过 val 选出的基线，但不超过 test 上最好的 GBDT；温度缩放只让 EOT 的 ECE 显著下降；冷启动对 EOT / 接话几乎没有代价；跨场景（合成直播、1:1）基本不迁移；全双工控制器在单线程 CPU 上远超 10 Hz，但在合成测试里"让出"和"冷场主动开口"两项不如调过参的规则。结论不变：继续影子模式，不接管真实决策。

**第三阶段的结论**（[reports/phase3.md](reports/phase3.md)，已脱敏；数据集清单见 [reports/phase3_datasets.md](reports/phase3_datasets.md)）：公开数据补上了真实直播和口语数据，但文字侧在真实留出数据上仍然没有一致、显著的提升。音频前端是这一阶段最实在的进展：英文全双工语音预训练后，中文附和预测显著变好；但在 Krisp 测试集上没超过"静默时长"这个零参数基线。结论不变：继续影子模式，不接管真实决策。

**第四阶段的结论**（[reports/phase4.md](reports/phase4.md)，已脱敏；数据集清单见 [reports/phase4_datasets.md](reports/phase4_datasets.md)）：文本情绪特征按预注册规则被采用（真实 val loss 下降，测试集上"说 / 等 / 不说"头在两个切分上都变好）；语音情绪在音频轮次上没有帮助，不采用。主干对比按预注册规则胜出的是 Mamba-2 式消融（四个指定主干里最好的是 Mamba-3 SISO），但测试集上没有一个主干一致超过 GRU。连续体记忆只带来很小的自监督 loss 改进，决策头不变，保持影子模式。总体结论不变：不接管真实决策。

## 任务头

| 头 | 问题 |
|---|---|
| `y_eot` | 这条消息说完这一轮了吗？ |
| `y_self` | 同一个人一会儿还会补充吗？ |
| `y_addr` | 这条群消息是在叫机器人吗？ |
| `y_act` | 机器人现在应该：说 / 等 / 不说？ |
| `y_recheck` | 多久之后值得再看一次？（7 个时间桶） |
| `y_hreply` | 群里其他人会接话吗？ |

## 目录

```
tidal/                 核心代码
  features.py          因果的逐事件特征（时间、角色、文本形态）
  labels.py            从可观测结构推导弱标签
  model.py             小型因果主干 + 多任务头
  train.py / evaluate.py / baselines.py / metrics.py / audit.py / export.py / bench.py
  synth.py             用 LLM 生成合成群聊（只用抽象的场景描述；需要你自己的 key）
  modalities/          各模态前端（meta、text 已实现；image、video、audio 是计划中的接口）
  features_g.py        第二阶段：场景通用的事件输入（时间 / 节奏 / 相对角色 + 可丢弃的人数、模态块）
  vap_targets.py       第二阶段：VAP 式未来事件投影目标（4 个相对通道 × 5 个时间桶）
  vap.py               第二阶段：投影预训练 → 多任务微调；流式 step()
  scenarios.py         第二阶段：脚本合成的直播（弹幕 + 礼物 + 主播）和 1:1 事件流（不用 LLM）
  dataset2.py          第二阶段：统一事件流数据集（真实数据经私有适配器接入）
  evaluate2.py / coldstart.py / loso.py / report2.py   校准与弃权、冷启动与在线自适应、留一场景、投影质量
  duplex.py            第二阶段：100 ms tick 全双工控制循环（控制与内容分离）
  duplex_sim.py        第二阶段：合成全双工 sanity 测试 + CPU 实时基准
  export2.py           第二阶段：ONNX 导出 + 一致性检查
  public_data/         第三阶段：公开数据集清单（许可证）、下载器、转换器、快速打标签
  dataset3.py / eval3.py / loso3.py / quant3.py / turnorder_eval.py   第三阶段：数据集、评测、留一场景、int8 量化、轮次顺序评测
  audio_fe.py / audio_train.py / audio_oto.py   第三阶段：流式音频前端、训练和评测（MagicData、otoSpeech、Krisp）
  backbones.py         第四阶段：Mamba-3 SISO / MIMO（含分块 SSD）、RoPE + KV 缓存 Transformer、Mamba-2 式消融
  emotion.py / emotion_speech.py / sv_soft.py / emo_features.py / emo_speech_integ.py   第四阶段：文本 / 语音情绪头、SenseVoice 离线教师软标签、情绪特征、语音情绪接入测试
  eval4.py / bench4.py / report4.py / continuum.py   第四阶段：评测、流式延迟与 ONNX、汇总、连续体记忆（影子回放）
  corpus_fmt.py        "!语料" 对话片段格式的通用解析器（示例见 examples/corpus_example.txt，纯虚构）
  shadow/              影子模式：数据源插件、SQLite 存储、推理、打标签、报告
scripts/               流水线、cron 包装、示例数据生成
docs/                  架构与路线图、相关工作
reports/phase1.md      第一阶段报告（脱敏版）
reports/phase2.md      第二阶段报告（脱敏版）
reports/phase3.md      第三阶段报告（脱敏版，只含公开数据上的数字）
reports/phase3_datasets.md   第三阶段公开数据集清单、许可证、跳过原因
reports/phase4.md      第四阶段报告（脱敏版）
reports/phase4_datasets.md   第四阶段数据集清单、许可证、跳过原因
reports/*_phase4*.json  第四阶段只用公开数据的评测结果（情绪、主干公开评测、延迟、连续体记忆公开回放）
reports/audio_*.json   音频前端评测（只用公开数据）
reports/duplex_*.json  全双工合成 sanity 测试和 CPU 基准（纯合成输入）
shadow/README.md       影子模式说明
examples/              纯合成的演示事件流（脚本生成，没有用 LLM）
tests/                 单元测试和端到端测试（只用合成数据）
```

## 快速开始（纯 CPU）

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q                                   # 测试只用合成数据，不需要模型权重

mkdir -p .secrets && head -c 32 /dev/urandom | base64 > .secrets/pseudo_salt   # 脱敏用的盐，只留在本地
cp config/local.example.json config/local.json                                   # 机器人名字、留出的群等部署配置
# 影子模式演示（没有模型时会跳过预测，只入库和打标签）
TIDAL_SHADOW_JSONL=examples/synthetic_stream.jsonl python -m tidal.shadow.run --source jsonl
```

训练需要你自己的数据：一张统一的事件表（字段说明见 `tidal/labels.py`）。另外，bge-small-zh-v1.5 的 ONNX 文件和 tokenizer 要放在 `models/bge/`。流程见 `scripts/run_all.sh`，以及 `reports/phase1.md` 的第 9 节。

### 中文音频接话实验

独立于私有聊天和文本编码器，可复现实验协议见 [audio_turn_protocol.md](reports/audio_turn_protocol.md)：

```bash
pip install huggingface-hub
python -m tidal.public_data.download magicdata_ms
OMP_NUM_THREADS=2 python -m tidal.audio_train prep
python -m tidal.audio_turn                 # 三组说话人轮换留出、三个随机种子
```

数据及权重仅限本地学术研究；输出 `reports/audio_turn_optimization.json`。模型直接预测接话，输入严格限制为决策前已完整到达的音频；使用人工片段边界，尚未接入真实控制。只有相对 LR 的排序和概率误差均通过预定门槛，报告才会标记候选模型可采用；不会自动替换影子模型。

### 无人值守优化与轻量影子推理

完整训练、冻结评估、ONNX导出及流式核验：

```bash
.venv/bin/python -m tidal.optimize          # 本工作区环境已配置；重复执行可恢复完成结果
# 新的 macOS arm64 / Python3.12 环境：
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python requirements-optimization-macos.lock
```

第二轮使用CANDOR对话留出及新的英文说话人测试；中文旧留出只作为开发资料。结果与局限见 [无人值守优化报告](reports/unattended_optimization.md)，固定协议见 [第二轮协议](reports/audio_turn_v2_protocol.md)。模型和校准由验证集选择并冻结，测试结果不会触发重调。新模型未接管中文真实决策。

独立的 [turn_shadow.py](tidal/turn_shadow.py) 仅依赖 [NumPy及ONNX Runtime](requirements-turn-runtime.txt)，无需PyTorch、sklearn或pandas。它返回接话行为的影子概率，要求调用方提供已观察的语音片段边界；尚未验证真实VAD/ASR。权重在本地 `models/turn_v2/`，不再分发。

### 策略与目标任务头（无人值守落地）

新增独立的人工策略监督、草稿条件回复指针、在线重叠结果辅助头。主模型仍为六头；`y_act_behavior` / `y_next_gap` 明确原行为与时间标签含义。人工 `action_policy`、语义结束、策略重检与重叠意图需要真实标注，缺失时明确报告，不创建随机权重冒充训练结果。所有新输出均接入可选 `ControlOut.task_shadow`，只记录。

```bash
.venv/bin/python -m tidal.head_optimize
.venv/bin/python -m tidal.head_v2_train  # 第二轮词权重/在线重叠优化
```

使用与人工数据格式见 [任务头说明](docs/task_heads.md)，固定评估协议见 [训练前协议](reports/task_heads_protocol.md) 与 [第二轮协议](reports/head_v2_protocol.md)，实际结果见 [第一轮报告](reports/task_heads.md) 与 [第二轮报告](reports/head_v2.md)。IRC回复链接训练是已有草稿的目标匹配代理；语音弱标签预测自然话轮结果，均尚不能证明中文机器人策略效果。

## 隐私与数据

- 仓库里**没有任何真实聊天数据**：没有原始数据，也没有脱敏后的数据。在真实聊天上训练的权重也不公开，因为小模型同样可能记住训练数据。
- 也不包含 LLM 生成的合成对话，只包含生成脚本。原因是生成时的提示词里带有部署方机器人的名字。
- 所有 ID 都用带盐 HMAC 处理，文本先去掉手机号、QQ 号这类长数字串、URL、邮箱、身份证号和 CQ 码。盐、密钥和部署配置都放在 `.secrets/` 和 `config/local.json` 里，两者都已加入 `.gitignore`。
- 影子模式不调用任何外部 API，只读取上游数据，从不写入。
- 公开数据集只下载到本地 `data/public/`（已 gitignore），仓库里只有加载和转换代码。gated 数据集需要你自己在 HF 上同意条款，token 只放环境变量 `HF_TOKEN`。许可证不明确、"other" 或非商用的数据按 research-only 处理；Krisp 测试集只能用于评测。

## 许可

[MIT](LICENSE)
