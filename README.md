# tidal 潮汐

**English.** tidal is a tiny (≤ 2M resident parameters), CPU-only model for conversational turn-taking in group and
private chats. It predicts when an utterance is finished, whether the speaker will add more, whether the bot is being
addressed, and whether the bot should speak, wait or stay silent. The design goal is a *natively multimodal* model:
video, images, music, speech and text, with one shared tiny causal encoder and per-modality front ends. **Status today:**
only timing and text inputs are implemented, plus content-free message-type metadata (image, voice, video, sticker…).
The image, video and audio front ends are designed and scaffolded but not yet implemented. Phase 2 added a VAP-style
self-supervised future-event projection objective, temperature calibration + selective abstention, identity-free roles
with cold-start / online-adaptation evaluation, a scenario-general event stream (no scenario enum), a 100 ms tick-based
full-duplex control loop (control separate from content; runs at well over 10 Hz on one CPU thread), ONNX export with
parity checks, and side-by-side shadow scoring. Neither phase showed a stable, significant win over strong baselines on
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
| 时间节奏 + 角色特征，因果 GRU / Transformer 主干（约 0.5M 参数），6 个任务头 | ✅ 已实现 |
| 文本：冻结的 bge-small-zh（int8 ONNX）句向量 | ✅ 已实现 |
| 消息类型元数据：图片、语音、视频、文件、贴纸、表情、分享（只从占位符解析，不读内容） | ✅ 已实现采集，尚未作为模型输入 |
| 图像、视频帧、音频（语音 + 音乐）前端 | 📋 计划中（接口已定义，见 `tidal/modalities/`） |
| 校准（温度缩放）、ONNX 导出和一致性测试、泄漏审计、带 CI 的基线对比 | ✅ |
| VAP 式未来事件投影（只用时间和相对角色的自监督目标），预训练 → 多任务微调 | ✅ 第二阶段 |
| 选择性弃权（风险-覆盖曲线、"等一下再看"策略）、ECE 报告 | ✅ 第二阶段 |
| 身份无关的相对角色；冷启动评测（加入后前 20 / 50 / 100 条）和按会话在线重新校准 | ✅ 第二阶段 |
| 场景通用的统一事件流（没有场景枚举，参与人数是连续特征）、留一场景评测 | ✅ 第二阶段（直播和 1:1 流是脚本合成的） |
| 全双工 tick 控制循环（100 ms，多路输入，她自己的输出也是输入，控制与内容生成分离） | 🚧 控制骨架 + 合成 sanity 测试 + CPU 基准；音频 / 视觉前端仍是接口 |
| 影子模式（只记录不执行，延迟打标签，定时任务，报告；新旧模型在同一批标签上并行打分） | ✅ |
| 蒸馏、主干量化、剪枝、LoRA、保形预测 | 📋 计划中 |

完整的设计、参数预算和"技术 → 组件 → 状态"对照表见 **[docs/architecture.md](docs/architecture.md)**。

**第一阶段的结论**（[reports/phase1.md](reports/phase1.md)，已脱敏）：模型在任何一个头上都没有稳定、显著地超过最强基线。多数头的 Δ 为正，但 95% CI 跨过 0。真实数据太少，覆盖也不完整。所以下一步是先跑影子模式、补数据，**在 CI 显著优于基线之前，不接管任何真实决策**。

**第二阶段的结论**（[reports/phase2.md](reports/phase2.md)，已脱敏）：能力补齐了，但效果没有实质提升。VAP 投影能学，却没有超过同特征上的逻辑回归；零样本读出在"自我续话"上显著超过 val 选出的基线，但不超过 test 上最好的 GBDT；温度缩放只让 EOT 的 ECE 显著下降；冷启动对 EOT / 接话几乎没有代价；跨场景（合成直播、1:1）基本不迁移；全双工控制器在单线程 CPU 上远超 10 Hz，但在合成测试里"让出"和"冷场主动开口"两项不如调过参的规则。结论不变：继续影子模式，不接管真实决策。

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
  shadow/              影子模式：数据源插件、SQLite 存储、推理、打标签、报告
scripts/               流水线、cron 包装、示例数据生成
docs/                  架构与路线图、相关工作
reports/phase1.md      第一阶段报告（脱敏版）
reports/phase2.md      第二阶段报告（脱敏版）
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

## 隐私与数据

- 仓库里**没有任何真实聊天数据**：没有原始数据，也没有脱敏后的数据。在真实聊天上训练的权重也不公开，因为小模型同样可能记住训练数据。
- 也不包含 LLM 生成的合成对话，只包含生成脚本。原因是生成时的提示词里带有部署方机器人的名字。
- 所有 ID 都用带盐 HMAC 处理，文本先去掉手机号、QQ 号这类长数字串、URL、邮箱、身份证号和 CQ 码。盐、密钥和部署配置都放在 `.secrets/` 和 `config/local.json` 里，两者都已加入 `.gitignore`。
- 影子模式不调用任何外部 API，只读取上游数据，从不写入。

## 许可

[MIT](LICENSE)
