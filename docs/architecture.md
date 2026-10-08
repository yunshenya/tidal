# tidal 架构与路线图

> 目标：做一个**原生多模态**（视频、图像、音乐、语音、文本）、**会判断什么时候该说话、什么时候该沉默**的极小模型。它只用 CPU，常驻参数 ≤ 2M。
> 本文给出具体设计，并把每一项现代建模技术对应到 tidal 的某个组件上，逐项标注状态：✅ 已实现 · 🚧 进行中 · 📋 计划中。
> 状态以本仓库代码为准，不夸大。**目前真正跑通的是"时间节奏 + 文本"两路输入、内容无关的消息类型元数据、语音的流式音频前端，以及第四阶段的主干对比和情绪特征。图像、视频、音乐的前端还是计划。**
> 阶段：第一阶段是 GRU 与 6 个头（[../reports/phase1.md](../reports/phase1.md)）。第二阶段（[../reports/phase2.md](../reports/phase2.md)）补上了自监督投影、校准与弃权、冷启动与在线自适应、场景通用输入、全双工 tick 控制骨架，但都没有带来显著的效果提升。第三阶段（[../reports/phase3.md](../reports/phase3.md)）加了公开数据加载与预训练、真实直播留一场景、音频前端（`tidal/audio_fe.py`）和动态 int8（`tidal/quant3.py`）。第四阶段（[../reports/phase4.md](../reports/phase4.md)）见下一节。第五阶段（[../reports/phase5.md](../reports/phase5.md)）训了打断、话题转移，并把 `y_addr` 试成消息价值；只有打断头按规则采用，而且都还没进影子 tick。
> 调研和差距分析见 [related-work.md](related-work.md)。

## 第四阶段（已完成的判定）与第五阶段（已判定，未进影子 tick）

按 [../reports/phase4.md](../reports/phase4.md) 的预注册规则，对照代码：

- **主干。** 四个被点名的主干是 GRU、`tx_kv`（RoPE + KV 缓存）、`mamba3`（Mamba-3 MIMO）、`mamba3_siso`（Mamba-3 SISO），外加消融 `m3_ablate_m2`（`tidal/backbones.py` 的 `make_body`）。字面规则把消融也算进候选，真实验证集上胜出的是 `m3_ablate_m2`。**只在这四个被点名的主干里选，胜出的是 Mamba-3 SISO。** 影子模式加载的就是这一份：`tidal/shadow/p4.py` 的 `WINNER` 是 `kind=mamba3_siso`，文本情绪开、语音情绪关。不把消融改写成"没赢"。
- **文本情绪：采用，但是事件特征，不是头。** `tidal/emo_features.py` 的 11 列（8 类概率 + 效价 + 唤醒度 + 有无标记）拼在 `features_g` 后面。`Event.emotion` / `EventFeaturizer`（`tidal/duplex.py`）在 `n_extra > 0` 时写入同样的 10 个数加一个有无标记。`model.py` 的 `HEAD_DIMS` 里没有情绪头。
- **语音情绪：头存在，不进入决策。** `tidal/emotion_speech.py` 的 `SpeechEmoHead` 在。预注册判定是不采用。全双工只把它放进 `ControlOut.affect`（`tidal/duplex.py`），事件编码器不读语音情绪。SenseVoice 只做离线教师（`emotion_speech.teacher`，可选依赖 `requirements-teacher.txt`）；软后验在 `tidal/sv_soft.py`，用的是 onnxruntime，不是 sherpa。
- **连续体记忆：只做影子回放。** `tidal/continuum.py` 写明 "No live decisions"。
- **第五阶段的判定（影子 tick 不改）。** 打断头、话题转移头，以及把 `y_addr` 训成消息价值，都在冻结的 `mamba3_siso` + 文本情绪上联合训练过（3 个种子，规则见 [../reports/phase5.md](../reports/phase5.md)）。按预注册：打断头采用；话题转移头不采用；`y_addr` 的新目标不采用。`HEAD_DIMS` 仍是原来的六个，影子 tick 不加载这几个新头。`duplex.ACTIONS` 里的 `yield` 仍是 tick 动作，不是这个打断头。`EventEncoder.address` 仍是 `p_speak × 时间衰减`。
- **主干重跑。** 候选只有 `mamba3_siso` 和 `m3_ablate_m2`。真实 val 上消融更低，而且差距大于 0.005，所以排名第一的是消融。**影子配置不因此改主干**，`WINNER` 仍是 `mamba3_siso`，文本情绪开，语音情绪关。

参数量沿用第四阶段报告，不在这里另算：`mamba3_siso` 不加情绪约 469,890（报告里四个新主干大约 0.47–0.50M）。文本情绪的 11 列只加宽输入投影，不另报一个新的总数。

## 1. 总体结构

```
   每个事件（一条消息 / 一段音频帧窗口 / 一个视频片段）
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 时间 / 角色特征     → Linear ─┐                                       │
   │ 消息类型元数据 meta  → Linear ─┤                                       │
   │ 文本   (bge 冻结 / 字节 BPE) → proj ─┤                                │
   │ 文本情绪 (11 列，已接入事件特征) ─┤                                    │
   │ 图像   (patch/conv stem)     → proj ─┼─ Σ + 模态掩码 → LayerNorm → token│
   │ 视频   (帧采样 → 图像 stem → 时间池化) ─┤                              │
   │ 音频   (log-mel → 因果 conv → 小 GRU；语音。音乐仍是计划) ─┘           │
   └────────────────────────────────────────────────────────────────────────┘
                     │  每个事件一个 token（d = 128）
                     ▼
   共享因果时序编码器（都已实现，影子配置用 mamba3_siso）：
     GRU | 可学习位置的因果 Transformer | tx_kv（RoPE + KV）| Mamba-3 SISO / MIMO / 消融
   流式推理：新事件到来时，只基于已经发生的事件更新状态。填充位不改 GRU / Mamba 隐状态，也不写入 tx_kv 的 KV。
                     │
                     ▼
   多任务头（已实现）：EOT · 续话 · 被叫 · 说/等/不说 · 重检延迟 · 人类接话
             ＋ VAP 投影头（vap.py）
             ＋ 第五阶段侧头：打断（规则采用，未进 tick）· 话题转移（不采用）· `y_addr` 消息价值（不采用）
             ＋ 计划：模态理解（这是什么媒体、声学事件）
                     │
                     ▼
   校准（温度缩放）→ 不确定性 / 弃权 → 决策层硬规则（被 @ 必须回应、发言预算、深夜静默）
```

代码映射：

- 前端接口：`tidal/modalities/`（`FrontEnd`：`dim`、`available()`、`encode()`）。`modalities/audio.py` 的 `available()` 仍是 False（计划中的统一前端）。已经能跑的语音前端是 `tidal/audio_fe.py`，不是这个 stub。
- 时间特征：`tidal/features.py`；场景通用事件特征：`tidal/features_g.py`。
- 主干和头：`tidal/model.py`、`tidal/backbones.py`；VAP：`tidal/vap.py`、`tidal/vap_targets.py`。
- 情绪：`tidal/emotion.py`、`tidal/emo_features.py`、`tidal/emotion_speech.py`。
- 全双工 tick：`tidal/duplex.py`。连续体：`tidal/continuum.py`。
- 训练：`tidal/train.py`、`tidal/vap.py`。
- 校准和评测：`tidal/evaluate.py`、`tidal/metrics.py`。
- 导出与 int8：`tidal/export.py`、`tidal/export2.py`、`tidal/quant3.py`（动态 int8；QAT 仍是计划）。
- 影子模式：`tidal/shadow/`。第四阶段配置：`tidal/shadow/p4.py`（`TIDAL_SHADOW_P4=1` 或 `shadow/state/frozen_p4.json` 才加载，默认不改变第一阶段 ONNX tick）。

缺失的模态在输入端就是"零向量 + 掩码位"，所以主干不需要知道某个事件带了哪些模态。

## 2. 参数预算（常驻 ≤ 2M，CPU 单线程）

| 组件 | 预算 | 现状 |
|---|---|---|
| 时间 / 元数据投影 | < 10k | ✅ 已有时间投影；元数据在影子模式里记录。`features_g` 有模态块（已知位 + 若干模态列），不是独立的大投影 |
| 文本前端 | 冻结 bge-small-zh int8（约 24M，**不计入常驻预算**，按需调用）；计划换成字节级或小 BPE 嵌入 + 2 层小编码器，约 0.2–0.3M | ✅ bge；📋 自有小文本前端 |
| 文本情绪 | 冻结 bge 上的小 MLP，报告里 13.4 万参数，**不计入事件主干**；接入时只加 11 列输入 | ✅ 已采用为事件特征 |
| 图像前端 | patch/conv stem（96–128 px 缩略图），约 0.3M | 📋 `modalities/image.py`，`available()` 为 False |
| 视频前端 | 复用图像 stem，额外约 0.05M | 📋 |
| 音频前端（语音） | 16 kHz、40 维 log-mel、10 ms 帧移、两帧叠成 20 ms 一步 → 因果 conv → GRU(96) | ✅ `audio_fe.py`。`modalities/audio.py` 仍是 stub（64 mel 常量），不要把它当成这条前端。音乐 📋 |
| 语音情绪头 | 报告里约 4.2 万参数 | ✅ 头在；❌ 不进入轮次决策 |
| 共享时序主干 | 0.5–0.8M。报告：不加情绪的 `mamba3_siso` 469,890；GRU 499,042 | ✅ GRU、可学习位置 Transformer、`tx_kv`、Mamba-3 SISO / MIMO、消融 |
| 多任务头 | 约 0.1M | ✅ 6 个头 + VAP 投影头 |

大教师模型只在**离线**环境下使用，从不常驻。SenseVoice 依赖是可选的（`requirements-teacher.txt`）；缺了它时 `emotion_speech.teacher` 会说明要装 `teacher` extra，而不是在 sherpa 内部抛一层无关错误。

## 3. 技术 → 组件 → 状态

| 技术 | 在 tidal 里的落点 | 状态 | 说明 |
|---|---|---|---|
| 按模态划分的前端：图像 / 视频帧用 patch 或 conv | `modalities/image.py`、`video.py` | 📋 | 接口已定义，`available()` 返回 False |
| 按模态划分的前端：语音 / 音乐用 log-mel | `modalities/audio.py` 与 `audio_fe.py` | 📋 / ✅ | stub 的 `available()` 仍是 False。能跑的是 `audio_fe.py`（40 mel、16 kHz、10 ms、stack 2） |
| 按模态划分的前端：文本 | `modalities/text.py`、`embed.py` | ✅ / 📋 | 现在用冻结的 bge int8；计划换成字节级或小 BPE 的自有前端 |
| 内容无关的消息类型元数据 | `modalities/meta.py`、影子库 `events.media_*` | ✅（采集）/ 🚧（作为输入） | 从平台占位符和 CQ 码解析，不读取媒体内容。`features_g` 有模态列，但是可选块 |
| 共享小主干：GRU | `model.py` kind=gru | ✅ | 2 层，隐藏层 192。填充位走 `run_gru`（pack / 逐步掩码），不再在 pad 上更新状态 |
| 共享小主干：因果 Transformer | `model.py` 里不在 `BODY_KINDS` 的 kind | ✅ | 可学习位置，因果 mask，pad key 不参与注意力 |
| RoPE + KV 缓存 | `backbones.py` `CausalTransformer`，kind=`tx_kv` | ✅ | 不是计划项。`step` 在 `valid` 为假时不写 KV、不推进位置 |
| SSM / Mamba-3 | `backbones.py` `mamba3` / `mamba3_siso` / `m3_ablate_m2` | ✅ | 纯 PyTorch CPU。pad 步不更新残差后的隐状态。影子配置是 SISO |
| 文本情绪 | `emotion.py`、`emo_features.py`、`duplex.Event.emotion` | ✅ | 8 类 + 效价 / 唤醒度，作为 11 列事件特征接入。不是 `HEAD_DIMS` 的头 |
| 语音情绪 | `emotion_speech.py` | ✅ 头 / ❌ 决策 | 不采用。只出现在 `ControlOut.affect`。教师是可选 extra |
| 跨模态时间融合 | 事件 token 求和 + 模态掩码 | 🚧 | 时间、文本、文本情绪已接入事件模型；图像 / 视频 / 音乐未接 |
| 流式 / 因果推理 | `features.py`、`model.py`、`vap.py` `step`、`shadow/infer.py` | ✅ | 影子模式逐条消息预测。第四阶段模型默认不加载 |
| 多任务头 | `model.py` `HEAD_DIMS` | ✅ | 六个：EOT、续话、被叫、说/等/不说、重检、人类接话 |
| 打断 / 话题转移 / 泛化回复对象 | `phase5_labels.py`、`phase5_train.py`（侧头，不在 `HEAD_DIMS`） | ✅ 打断采用 / ❌ 话题与新 `y_addr` 不采用 | 影子 tick 仍是第四阶段的六个头。`yield` 仍是 tick 动作 |
| 自监督：VAP 式的未来事件投影 | `vap_targets.py`、`vap.py` | ✅ | 4 个相对通道 × 5 个时间桶。桶的右端超过已观测时间则为 NaN，不记成负例 |
| 弱标签的未来窗 | `labels.py`、`public_data/fastlabels.py` | ✅ | 续话 / 人类接话 / "沉默"只有整段窗口被观测到才标 0 或 2；否则 NaN。正例不受影响 |
| 自监督：掩码事件建模 | `vap.py` | 📋 | |
| 自监督：跨模态对比对齐 | 前端 + 投影层 | 📋 | |
| 知识蒸馏 | `emotion_speech.teacher`、`sv_soft.py` | ✅（语音情绪教师）/ 📋（其余） | 教师离线、只碰公开音频。真实聊天不发给外部 API |
| LLM 合成数据增强 | `synth.py`、`synth_clean.py` | ✅ | 第一阶段：对节奏类的头有帮助，对"被叫"和"动作"有害 |
| 课程学习 | `vap.py` | 🚧 | 先投影、再微调全部头 |
| 数据增强 | `features_g.py` | 🚧 | 人数块和模态块可整块丢弃。时间抖动、文本扰动未做 |
| 场景通用的统一事件流 | `features_g.py`、`scenarios.py`、`loso.py` | ✅ / 🚧 | 没有场景枚举。留一场景基本不迁移（第二阶段） |
| 身份无关角色 + 在线自适应 | `coldstart.py` | ✅ | 收益很小 |
| 全双工 tick 控制（100 ms） | `duplex.py`、`duplex_sim.py` | 🚧 | 动作空间含 `yield`。只在合成数据上做过 sanity 测试 |
| 连续体记忆 | `continuum.py` | ✅（影子回放） | 不做线上决策 |
| int8 量化 | `embed.py`；`quant3.py` | ✅ 动态 int8 / 📋 QAT | 主干的 QAT 仍是计划 |
| 剪枝 / LoRA | `model.py` | 📋 | |
| 校准与弃权 | `evaluate.py`、`evaluate2.py`、`shadow/infer.py` | ✅ / 🚧 | 温度缩放在；线上弃权率不能直接用离线阈值 |
| 离策略修正（IPS） | `train.py` | 📋 | |
| ONNX 导出 | `export.py`、`export2.py` | ✅ | 与 PyTorch 的最大差约 2e-7（报告）。填充位不再让 GRU 更新状态；本文不另报一个新的左填充误差数字 |
| 与基线的对比评测 | `baselines.py`、`evaluate.py`、`metrics.py` | ✅ | |
| 泄漏审计 | `audit.py` | ✅ | |
| 影子模式 | `tidal/shadow/` | ✅（基础设施）/ 🚧（积累数据中） | 只记录不执行。跨平台锁在 `shadow/lock.py`（不在 import 时依赖 `fcntl` / `resource`）。**在 CI 显著优于基线之前，不接管任何真实决策** |

## 4. 落地顺序

1. 🚧 影子模式继续积累数据，然后重新评测。第四阶段配置要显式打开（`TIDAL_SHADOW_P4=1` 或 `frozen_p4.json`），默认 tick 仍是第一阶段 ONNX。
2. 📋 把 meta 元数据稳定地接为模型输入，加上"丢弃模态"增强，再加 IPS 动作头。
3. ✅ 音频前端（第三阶段）。✅ 文本情绪已作为事件特征采用；语音情绪头已训练但**不采用**为决策输入。📋 其余离线教师（转写、声学 EOT）仍是计划。
4. 📋 图像 / 视频前端。
5. ✅ 主干里的 RoPE（`tx_kv`）和 SSM（Mamba-3）已经实现并对比过。📋 掩码事件预训练、主干 QAT、按群挂 adapter 仍是计划。
6. ✅/❌ 第五阶段训练已按预注册判定：打断头采用，话题转移和新的 `y_addr` 目标不采用。三项都还没进影子 tick 的 `HEAD_DIMS`。
7. 每一步都要先过影子模式门控，才考虑上线。
