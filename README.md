# tidal 潮汐

**English.** tidal is a tiny (≤ 2M resident parameters), CPU-only model for conversational turn-taking in group and
private chats. It predicts when an utterance is finished, whether the speaker will add more, whether the bot is being
addressed, and whether the bot should speak, wait or stay silent. The design goal is a *natively multimodal* model:
video, images, music, speech and text, with one shared tiny causal encoder and per-modality front ends. **Status today:**
only timing and text inputs are implemented, plus content-free message-type metadata (image, voice, video, sticker…).
The image, video and audio front ends are designed and scaffolded but not yet implemented. Phase 1 did **not** show a
stable win over strong baselines. A shadow-mode harness (log-only, never acts) is included for collecting fresh
evaluation data. MIT licensed. No real chat data and no weights trained on real chat are published.

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
| 影子模式（只记录不执行，延迟打标签，定时任务，报告） | ✅ |
| 蒸馏、自监督预训练、主干量化、剪枝、LoRA、不确定性与弃权 | 📋 计划中 |

完整的设计、参数预算和"技术 → 组件 → 状态"对照表见 **[docs/architecture.md](docs/architecture.md)**。

**第一阶段的结论**（[reports/phase1.md](reports/phase1.md)，已脱敏）：模型在任何一个头上都没有稳定、显著地超过最强基线。多数头的 Δ 为正，但 95% CI 跨过 0。真实数据太少，覆盖也不完整。所以下一步是先跑影子模式、补数据，**在 CI 显著优于基线之前，不接管任何真实决策**。

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
  shadow/              影子模式：数据源插件、SQLite 存储、推理、打标签、报告
scripts/               流水线、cron 包装、示例数据生成
docs/                  架构与路线图、相关工作
reports/phase1.md      第一阶段报告（脱敏版）
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
