# tidal 架构与路线图

> 目标：做一个**原生多模态**（视频、图像、音乐、语音、文本）、**会判断什么时候该说话、什么时候该沉默**的极小模型。它只用 CPU，常驻参数 ≤ 2M。
> 本文给出具体设计，并把每一项现代建模技术对应到 tidal 的某个组件上，逐项标注状态：✅ 已实现 · 🚧 进行中 · 📋 计划中。
> 状态以本仓库代码为准，不夸大。**目前真正跑通的是"时间节奏 + 文本"两路输入、内容无关的消息类型元数据、语音的流式音频前端，以及第四阶段的主干对比和情绪特征。图像、视频、音乐的前端还是计划。**
> 阶段：第一阶段是 GRU 与 6 个头（[../reports/phase1.md](../reports/phase1.md)）。第二阶段（[../reports/phase2.md](../reports/phase2.md)）补上了自监督投影、校准与弃权、冷启动与在线自适应、场景通用输入、全双工 tick 控制骨架，但都没有带来显著的效果提升。第三阶段（[../reports/phase3.md](../reports/phase3.md)）加了公开数据加载与预训练、真实直播留一场景、音频前端（`tidal/audio_fe.py`）和动态 int8（`tidal/quant3.py`）。第四阶段（[../reports/phase4.md](../reports/phase4.md)）见下一节。第五阶段（[../reports/phase5.md](../reports/phase5.md)）训了打断、话题转移，并把 `y_addr` 试成消息价值；只有打断头按规则采用；它在新主干上重训后，作为影子字段 `p_interrupt` 接进了影子 tick（只记录，不改动作）。前向插话 `p_barge` 另按预注册采用，同样只记录。
> 调研和差距分析见 [related-work.md](related-work.md)。

## 第四阶段（已完成的判定）与第五阶段（已判定；打断头只作影子字段）

按 [../reports/phase4.md](../reports/phase4.md) 的预注册规则，对照代码：

- **主干。** 四个被点名的主干是 GRU、`tx_kv`（RoPE + KV 缓存）、`mamba3`（Mamba-3 MIMO）、`mamba3_siso`（Mamba-3 SISO），外加消融 `m3_ablate_m2`（`tidal/backbones.py` 的 `make_body`）。字面规则把消融也算进候选，第四阶段真实验证集上胜出的是 `m3_ablate_m2`；只在那四个被点名的主干里选，第四阶段胜出的是 Mamba-3 SISO，而 `mamba3_siso` 仍留在代码里作为主干选项。**影子模式现在加载的官方胜者是 `m3_ablate_m2`**：2026-10-08 的预注册第五阶段 bake-off（新鲜 P5bb_*）里，head-sum 真实 val 均值是 3.9061，`mamba3_siso` 是 3.9276，差距 0.0215，大于 0.005。`tidal/shadow/p4.py` 的 `WINNER` 是 `kind=m3_ablate_m2`，文本情绪开、语音情绪关。
- **文本情绪：采用，但是事件特征，不是头。** `tidal/emo_features.py` 的 11 列（8 类概率 + 效价 + 唤醒度 + 有无标记）拼在 `features_g` 后面。`Event.emotion` / `EventFeaturizer`（`tidal/duplex.py`）在 `n_extra > 0` 时写入同样的 10 个数加一个有无标记。`model.py` 的 `HEAD_DIMS` 里没有情绪头。
- **语音情绪：头存在，不进入决策。** `tidal/emotion_speech.py` 的 `SpeechEmoHead` 在。预注册判定是不采用。全双工只把它放进 `ControlOut.affect`（`tidal/duplex.py`），事件编码器不读语音情绪。SenseVoice 只做离线教师（`emotion_speech.teacher`，可选依赖 `requirements-teacher.txt`）；软后验在 `tidal/sv_soft.py`，用的是 onnxruntime，不是 sherpa。
- **连续体记忆：只做影子回放。** `tidal/continuum.py` 写明 "No live decisions"。
- **第五阶段的头判定。** 打断头、话题转移头，以及把 `y_addr` 训成消息价值，都在冻结的 `mamba3_siso` + 文本情绪上联合训练过（3 个种子，规则见 [../reports/phase5.md](../reports/phase5.md)）。按预注册：打断头采用；话题转移头不采用；`y_addr` 的新目标不采用。`HEAD_DIMS` 仍是原来的六个；话题头和新 `y_addr` 目标不加载。`duplex.ACTIONS` 里的 `yield` 仍是 tick 动作，不是这个打断头。`EventEncoder.address` 仍是 `p_speak × 时间衰减`。
- **主干重跑。** 候选只有 `mamba3_siso` 和 `m3_ablate_m2`。2026-10-08 的预注册第五阶段 bake-off（新鲜 P5bb_*）里，head-sum 真实 val 均值差距是 0.0215，测试集也偏向消融，所以 `WINNER` 已切换为 `m3_ablate_m2`，文本情绪开，语音情绪关。连续体记忆仍只做影子回放。
- **打断头接入影子（只记录）。** 原来的打断头是在冻结 `mamba3_siso` 的状态上训的，不能直接放到 `m3_ablate_m2` 上。所以按同样的行、同样的联合训练和早停协议、同样的 3 个种子，在冻结的 `P4emo_m3_ablate_m2_s0` 状态上只重训侧头（`phase5_encode` / `phase5_train` 的 `p5_h_m2` 缓存）。公开 val BCE：0.2494 / 0.2519 / 0.2503（平均 0.2505），同一缓存上的线性探针 0.2600，训练集先验 0.4043。平均严格低于探针，按预注册规则仍然采用。公开 test（只报告）平均 0.2078。`tidal/shadow/p4.py` 把 3 个种子头的平均概率作为 `p_interrupt`：`EventEncoder` 每个事件记一份，`DuplexController` 的 `ControlOut.p_interrupt` 带上最后一个事件的值，cron 影子 tick 另写一个系统 `model:p4_m3_ablate_m2`（口径 `P4`，每条事件的六个头概率 + `p_interrupt`）。它**不是** tick 特征（`TICK_FEATS` 没变），不改任何动作，也没有重训 tick 控制器。用别的主干状态训出的头会被拒绝加载。标签含义：事件是一段说完的话，1 = 它在别人占着话轮时开口并拿到话轮，0 = 等到空档或只是附和。聊天流里没有这个真值，而且是在语音片段上训的，所以影子报告只给分布。每 tick 延迟（单线程、eager、随机权重、合成输入，`python -m tidal.duplex_sim bench_shadow` → `reports/duplex_bench_shadow.json`，影子编码器 + tick GRU，不加 → 加 3 个种子头，p50 / p95 ms）：每 tick 0 个事件 0.14 → 0.14 / 0.26 → 0.23；1 个 1.36 → 1.51 / 2.04 → 2.21；5 个 6.80 → 7.85 / 9.32 → 11.86；20 个 41.4 → 43.2 / 57.9 → 64.5。大约每个事件多 0.15 ms，都在 100 ms 的 tick 之内。

- **前向插话 `p_barge`（只记录）。** 上面的 `p_interrupt` 是给一段已经说完的话打分。`p_barge` 是另一件事：别人还占着话轮的时候，预测自己会不会在时限内开口，决策时刻早于开口。规则写在 [../reports/phase5_barge_prereg.md](../reports/phase5_barge_prereg.md)，先于训练提交。语音每 0.5 秒一个决策点、时限 1 秒；文本（`pub_tg`）在对方每条消息上决策、时限 8 秒。标签只来自时间和话轮，没有词表。特征是冻结的 `p5_h_m2` 状态加上 4 个过去的时间量，主干不重训。3 个种子的公开 val BCE 是 0.0989 / 0.0992 / 0.0988（平均 0.0989），同一特征上的线性探针 0.1012，训练集先验 0.1199。平均严格低于探针，按预注册采用。公开 test（只报告）0.0785 / 0.0776 / 0.0779。`ControlOut.p_barge` 和影子日志在对方占着话轮时记下 3 个种子的平均概率；自己占着话轮时不记。它不是 tick 特征，不改动作，也没有重训 tick 控制器。别的特征上训出的头会被拒绝。每 tick 延迟（打断头开着，加前向头之前 → 之后，p50 / p95 ms，`reports/duplex_bench_barge.json`）：0 个事件 0.16 → 0.15 / 0.30 → 0.30；1 个 1.49 → 1.95 / 2.15 → 2.44；5 个 7.03 → 7.12 / 9.80 → 9.42；20 个（只跑了 200 tick）30.1 → 26.2 / 34.2 → 32.9，高负载那一档噪声大于头本身。都在 100 ms 的 tick 之内。

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
   共享因果时序编码器（都已实现，影子配置用 m3_ablate_m2）：
     GRU | 可学习位置的因果 Transformer | tx_kv（RoPE + KV）| Mamba-3 SISO / MIMO / 消融
   流式推理：新事件到来时，只基于已经发生的事件更新状态。填充位不改 GRU / Mamba 隐状态，也不写入 tx_kv 的 KV。
                     │
                     ▼
   多任务头（已实现）：EOT · 续话 · 被叫 · 说/等/不说 · 重检延迟 · 人类接话
             ＋ VAP 投影头（vap.py）
             ＋ 第五阶段侧头：打断（规则采用，影子字段 `p_interrupt`，不改动作）· 前向插话 `p_barge`（规则采用，只记录）· 话题转移（不采用）· `y_addr` 消息价值（不采用）
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
- 影子模式：`tidal/shadow/`。第四阶段配置：`tidal/shadow/p4.py`（`TIDAL_SHADOW_P4=1` 或 `shadow/state/frozen_p4.json` 才加载，默认不改变第一阶段 ONNX tick）。打开后 `tidal/shadow/run.py` 的 `predict_p4` 单独写 `P4` 口径（含 `p_interrupt`），出错只记进 stats，不影响第一阶段。

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
| SSM / Mamba-3 | `backbones.py` `mamba3` / `mamba3_siso` / `m3_ablate_m2` | ✅ | 纯 PyTorch CPU。pad 步不更新残差后的隐状态。影子配置是 `m3_ablate_m2`；`mamba3_siso` 仍是可选主干 |
| 文本情绪 | `emotion.py`、`emo_features.py`、`duplex.Event.emotion` | ✅ | 8 类 + 效价 / 唤醒度，作为 11 列事件特征接入。不是 `HEAD_DIMS` 的头 |
| 语音情绪 | `emotion_speech.py` | ✅ 头 / ❌ 决策 | 不采用。只出现在 `ControlOut.affect`。教师是可选 extra |
| 跨模态时间融合 | 事件 token 求和 + 模态掩码 | 🚧 | 时间、文本、文本情绪已接入事件模型；图像 / 视频 / 音乐未接 |
| 流式 / 因果推理 | `features.py`、`model.py`、`vap.py` `step`、`shadow/infer.py` | ✅ | 影子模式逐条消息预测。第四阶段模型默认不加载 |
| 多任务头 | `model.py` `HEAD_DIMS` | ✅ | 六个：EOT、续话、被叫、说/等/不说、重检、人类接话 |
| 打断 / 话题转移 / 泛化回复对象 | `phase5_labels.py`、`phase5_train.py`（侧头，不在 `HEAD_DIMS`） | ✅ 打断采用（影子字段 `p_interrupt`）/ ❌ 话题与新 `y_addr` 不采用 | 决策仍只用六个头和 tick 控制器；`p_interrupt` 只记录。`yield` 仍是 tick 动作 |
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
| 全双工 tick 控制（100 ms） | `duplex.py`、`duplex_sim.py` | 🚧 | 动作空间含 `yield`；控制器校验 tick 单调性、未来事件和输入状态。只在合成数据上做过 sanity 测试 |
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
6. ✅/❌ 第五阶段训练已按预注册判定：打断头采用，话题转移和新的 `y_addr` 目标不采用。打断头在 `m3_ablate_m2` 上重训后只作影子字段 `p_interrupt`；三项都不在 `HEAD_DIMS`，也不是 tick 输入。
7. 每一步都要先过影子模式门控，才考虑上线。


## 策略、回复指针与在线重叠（2026-10-09）

在六主头之外增加可选 `ControllerTaskShadow`，见 [task_heads.md](task_heads.md)。原 `y_act` 明确保留行为监督，人工策略标签独立，旧检查点兼容。`reply_to` 是已生成草稿条件下的32候选+null指针；`overlap_outcome` 是自己讲话、对方开口200ms后的因果声学弱行为预测；人工 `overlap_intent` 与 `action_policy` 等通过本地JSONL训练入口，不靠时序代替意图真值。缺少人工标签的任务不提供模型。`ControlOut.task_shadow` 与所有实际动作分开；完整协议和实测结果见 [报告](../reports/task_heads.md)。


### 任务头第二轮

`tidal/head_v2_train.py` 训练 `reply_to_v2` 和 `overlap_timing`。前者用Ubuntu train词权重和作者 `channel_two/test` 冻结外部诊断，后者用MagicData/MagicHub分组并修复100ms tick与200ms观察点的交付错位。`V2Shadow` 只提供影子概率，不能改变 `DuplexController` 的动作或TTS信号。
