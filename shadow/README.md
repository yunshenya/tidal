# tidal 影子模式（shadow mode）

影子模式下，每条新消息到来时，tidal 模型和各个基线**都会做一次预测，但只记录、不执行**。等结果窗口过去后，再根据"实际发生了什么"给这些预测打标签：发言人有没有接着说、有没有人接话、机器人有没有被叫、旧策略当时说没说话。积累 1–2 周的数据后，就可以比较准确率，并用新数据重新训练。

- **不调用任何 LLM 或外部 API**，也不发送任何消息。
- 数据源通过插件接入（`tidal/shadow/sources/`），上游必须是**只读**的。
- 数据落盘前先脱敏：所有 ID 做带盐 HMAC（盐放在 `.secrets/pseudo_salt`），文本先经过 `tidal.privacy.scrub` 处理。
- 所有数据都写在 `shadow/state/`、`shadow/logs/`、`shadow/reports/` 里，这三个目录都已加入 `.gitignore`。

## 组成

| 文件 | 作用 |
|---|---|
| `tidal/shadow/run.py` | 执行一次 tick：拉取 → 脱敏 → upsert 事件 → 预测新事件 → 给到期的事件打标签。幂等，带文件锁 |
| `tidal/shadow/sources/base.py` | 数据源接口和统一事件 schema |
| `tidal/shadow/sources/jsonl.py` | 通用 JSONL 数据源，按字节偏移增量读取，用于演示和测试 |
| `tidal/shadow/infer.py` | ONNX 模型和基线推理，单线程 CPU；bge 嵌入缓存在本地库里 |
| `tidal/shadow/setup.py` | 一次性准备：重新拟合基线，冻结标准化参数和 val 上选出的阈值 |
| `tidal/shadow/report.py` | 生成报告：按任务对比模型和基线 |
| `scripts/shadow_cron.sh` | 给 cron 调用的包装脚本：flock 单实例、`nice`、`ionice`、超时、日志轮转 |
| `scripts/shadow_cron_install.sh` | 安装、移除、查看 cron 条目 |

自己接数据源时，实现 `fetch(store, now) -> DataFrame`（列见 `EVENT_COLUMNS`）和 `default_predict_after(now)` 两个方法，然后在 `config/local.json` 里设置 `"shadow_source": "your.module"`。原始 ID 和原始文本不能离开数据源模块。

## 运行

```bash
.venv/bin/python -m tidal.shadow.setup                 # 需要你自己训练出来的模型和基线（仓库里不带权重）
TIDAL_SHADOW_JSONL=examples/synthetic_stream.jsonl .venv/bin/python -m tidal.shadow.run --source jsonl
scripts/shadow_cron_install.sh install                 # 每 10 分钟跑一次 tick，每天 09:17 生成一份报告
scripts/shadow_cron_install.sh status
```

缺少模型时，tick 仍然会入库并打标签，只是跳过预测（运行统计里会记 `predict_skipped`）。

## 停止

```bash
scripts/shadow_cron_install.sh remove
```

## 读报告

```bash
.venv/bin/python -m tidal.shadow.report --max-lag-min 30      # 只统计"准实时"的预测
```

报告按任务分节，包括：
- 二分类任务：准确率（Wilson CI）、F1、ROC-AUC、PR-AUC（分块 bootstrap CI）、Brier、ECE 和校准分箱；
- 多分类任务：准确率、macro-F1、ECE、NLL；
- 每个任务最后都有"模型 − 各基线"的配对 Δ 及其 CI。**CI 不含 0 才算显著**。阈值冻结自 val，不在影子数据上调。

## 第二阶段模型并行打分

第二阶段的 VAP 模型作为**第二个系统**（`model:<tag>`，T 口径）接入同一个定时任务，和第一阶段模型在同一批事件、同一批延迟标签上打分；它加载或推理出错时只会在日志里记一行并跳过（下次运行重试），不会影响第一阶段系统。

```bash
.venv/bin/python -m tidal.export2 P2a_s1        # 导出 models/P2a_s1.onnx（带一致性检查）
.venv/bin/python -m tidal.shadow.setup2 P2a_s1  # 冻结温度、阈值、弃权阈值 -> shadow/state/frozen_p2.json
```

之后的每次定时运行会自动给新事件打第二阶段的分；首次接入时会回填观测窗口内还没有第二阶段预测的事件（只用消息到达时已有的信息，但不是实时做出的）。报告里会多出新系统的一行、"新 − 旧"的配对 Δ，以及弃权统计（弃权率、作答 / 弃权部分的准确率、弃权时建议的复查间隔）。删掉 `frozen_p2.json` 即可停用第二阶段系统。

## 多模态元数据

`tidal/modalities/meta.py` 会从平台占位符和 CQ 码里解析消息类型：图片、贴纸、表情、语音、视频、文件、音乐、分享、转发；如果码里带时长，也一并记下。结果写进 `events.media_kinds`、`media_counts`、`media_refs`（引用经过哈希）和 `media_duration_s`，为以后的多模态训练积累数据。tidal **不下载媒体，也不会把媒体发给任何外部服务**。

## 局限

- EOT、续话、接话这几个标签都需要知道发言人是谁；发言人未知时只能屏蔽。
- "该不该说话"没有真值：说 / 等 / 不说这个头只是在模仿旧策略，属于离策略学习，带有旧策略的偏差。
- "被叫"的正例通常很少，置信区间会很宽。
