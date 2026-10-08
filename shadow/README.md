# tidal 影子模式（shadow mode）

影子模式下，每条新消息到来时，tidal 模型和第一阶段的各个基线**都做一次预测，但只记录，不执行**。等结果窗口过去后，再用"实际发生了什么"给这些预测打标签：发言人有没有继续说、有没有人接话、机器人有没有被叫、旧策略说没说话等等。积累 1–2 周后对比准确率，并用新数据重训。

- **不调用任何 LLM 或外部 API**，不发送任何消息，不碰 kovi-bot 的代码、配置和服务。
- 上游数据**只读**：走第一阶段用过的同一条路径（SSH 隧道 → 生产 Postgres，会话强制 `default_transaction_read_only=on` 并做核验），只执行固定列的 SELECT。
- 落盘前先脱敏：ID 一律做带盐 HMAC（盐在 `.secrets/pseudo_salt`，和第一阶段相同），文本先过 `tidal.privacy.scrub`。原始 ID 和原始文本不会离开 `tidal/shadow/sources/kovi_pg.py`。
- 所有数据都在 `shadow/state/`（chmod 700，库文件 600）、`shadow/logs/` 和 `shadow/reports/` 里，这三个目录都已加入 .gitignore。

## 组成

| 文件 | 作用 |
|---|---|
| `tidal/shadow/run.py` | 一次 tick：拉取 → 脱敏 → upsert 事件 → 预测新事件 → 给到期事件打标签 |
| `tidal/shadow/sources/kovi_pg.py` | 私有数据源：短时 SSH 隧道 + 只读增量拉取（每次多读最近 1 小时作为重叠，靠 upsert 保证幂等），对最近 48 小时重新做对齐 |
| `tidal/shadow/sources/jsonl.py` | 通用 JSONL 数据源（演示和测试用） |
| `tidal/shadow/infer.py` | ONNX 模型（`models/T_rs_gru_s1.onnx`、`models/TS_rs_gru_s1.onnx`）和基线的推理，单线程 CPU |
| `tidal/shadow/setup.py` | 一次性准备：重拟合第一阶段的基线（已核验和第一阶段的预测**完全一致**，最大差 0.0），冻结标准化参数、val 上选的阈值和最强基线 |
| `tidal/shadow/report.py` | 报告：按任务对比模型和基线 |
| `scripts/shadow_cron.sh` | cron 调用的包装脚本：单实例（flock）、`nice 10`、`ionice idle`、10 分钟超时、日志超过 5 MB 时轮转 |
| `scripts/shadow_report_cron.sh` | 每天 09:17 生成一份报告，写到 `shadow/reports/report-YYYY-MM-DD.md`，同时更新 `latest.md` |
| `scripts/shadow_cron_install.sh` | 安装、移除或查看 cron 条目 |

SQLite 表（`shadow/state/shadow.db`）：
- `events`：脱敏后的事件，带非文本元数据；
- `predictions`：每个（消息, 口径, 系统）只写一次，时间取第一次看到这条消息的时候，并记录 `lag_s` = 预测时刻 − 消息时刻；
- `labels`：消息过去 `TIDAL_SHADOW_SETTLE_S`（默认 3600 s）后打标，只写一次；
- `runs`：每次运行的统计；
- `st_*`：数据源的脱敏暂存；`emb_cache`：bge 嵌入缓存。

## 运行

```bash
cd /workspace/tidal
.venv/bin/python -m tidal.shadow.setup         # 只需一次（依赖第一阶段的 data/proc、models、reports）
.venv/bin/python -m tidal.shadow.run           # 手动跑一次 tick（可以重复跑，是幂等的）
scripts/shadow_cron_install.sh install         # 装 cron：每 10 分钟（第 3、13、23… 分钟）跑一次，每天 09:17 出一份报告
scripts/shadow_cron_install.sh status          # 查看 cron 条目和 cron 守护进程
tail -f shadow/logs/shadow.log                 # 每次 tick 一行 JSON 统计
```

本机没有 systemd，cron 守护进程是手动起的（`sudo cron`）。**box 重启后需要重新执行 `sudo cron`**，可以用 `scripts/shadow_cron_install.sh status` 检查。

## 停止

```bash
scripts/shadow_cron_install.sh remove          # 删掉 cron 条目（其他 crontab 条目不受影响）
pkill -f "tidal.shadow.run"                    # 如果某次 tick 正在跑，可以顺手结束它（中断是安全的，下次会补上）
```
数据保留在 `shadow/state/` 里。要彻底清除，删掉 `shadow/state shadow/logs shadow/reports` 这几个目录即可。

## 看报告

```bash
.venv/bin/python -m tidal.shadow.report                          # 全部已打标的预测（包括首轮回填）
.venv/bin/python -m tidal.shadow.report --max-lag-min 30         # 只看"准实时"预测（预测时延 ≤ 30 分钟），推荐
.venv/bin/python -m tidal.shadow.report --since 2026-10-08 --out shadow/reports/x.md --json shadow/reports/x.json
```

报告按口径（T / TS）和任务分节：

| 任务 | 含义 | 指标 |
|---|---|---|
| `y_eot` | 这条消息是不是这个人这一轮的最后一条（20 s 内本人没有再发） | 二分类：准确率（Wilson CI）、F1、ROC-AUC、PR-AUC（分块 bootstrap CI）、Brier、ECE 和校准分箱 |
| `y_self` | 话轮结束后，同一个人 180 s 内会不会再补充 | 同上 |
| `y_addr` | 群聊入站消息是不是在叫机器人 | 同上（正例很少，CI 会很宽） |
| `y_hreply` | 60 s 内有没有其他人接话 | 同上 |
| `y_act` | 旧策略：说 / 等 / 不说 | 多分类：准确率、macro-F1（CI）、"说"一类的 ROC-AUC、ECE、NLL |
| `y_recheck` | 到下一条消息的间隔，分 7 个桶 | 同上 |

每个任务最后都有"模型 − 各基线"的配对 bootstrap Δ（按"会话 × 小时"分块）。**只有 CI 不含 0 才算显著。** 准确率和 F1 用的阈值冻结自第一阶段 val，不会在影子数据上重新调。

- **T 口径**只用时间和角色特征，严格因果：预测时用到的信息，消息到达时都已经有了。这是主要的对比口径。
- **TS 口径**还用文本和发言人特征。文本来自机器人的 LLM 上下文快照，到达有延迟，而且有选择性（只在机器人走 LLM 回合时才有），所以 TS 的绝对数值偏乐观，和第一阶段一样。

## 第二阶段模型并行打分

第二阶段的 VAP 模型作为**第二个系统**（`model:<tag>`，T 口径）接入同一个定时任务，和第一阶段模型在同一批事件、同一批延迟标签上打分；它加载或推理出错时只会在日志里记一行并跳过（下次运行重试），不会影响第一阶段系统。

```bash
.venv/bin/python -m tidal.export2 P2a_s1        # 导出 models/P2a_s1.onnx（带一致性检查）
.venv/bin/python -m tidal.shadow.setup2 P2a_s1  # 冻结温度、阈值、弃权阈值 -> shadow/state/frozen_p2.json
```

之后的每次定时运行会自动给新事件打第二阶段的分；首次接入时会回填观测窗口内还没有第二阶段预测的事件（只用消息到达时已有的信息，但不是实时做出的）。报告里会多出新系统的一行、"新 − 旧"的配对 Δ，以及弃权统计（弃权率、作答 / 弃权部分的准确率、弃权时建议的复查间隔）。删掉 `frozen_p2.json` 即可停用第二阶段系统。

## 第四阶段胜者 + 打断头（P4 口径，只记录）

`shadow/state/frozen_p4.json`（或 `TIDAL_SHADOW_P4=1`）打开后，每次 tick 另写一个系统 `model:p4_m3_ablate_m2`，口径 `P4`：`m3_ablate_m2` + 文本情绪的六个头概率，加上第五阶段打断头的 `p_interrupt`（3 个种子头的平均）。文本情绪只用 TS 路径已经缓存的 bge 嵌入，没有缓存的事件按"未知"处理。聊天流没有打断真值，报告里只给 `p_interrupt` 的分布。对方还占着话轮时另记前向概率 `p_barge`（预注册采用，同样只记录、不改动作）；自己占着话轮的事件不记这一项。它不改变任何动作，出错只记进 `runs.stats.p4_error`，不影响第一阶段。删掉 `frozen_p4.json` 即可停用。

```json
{"ckpt": "models/P4emo_m3_ablate_m2_s0.pt", "interrupt_heads": ["models/P5hd_m2_s0.pt", "models/P5hd_m2_s1.pt", "models/P5hd_m2_s2.pt"]}
```

## 能采集到的非文本 / 多模态信号

上游数据库里没有媒体文件，也没有 URL。tidal 不下载任何媒体，也不把媒体发给任何外部服务。目前能拿到的有：
1. **全部事件**：时间戳、方向（入站 / 出站）、会话类型、连发和节奏（T 口径用的特征）。
2. **2026-10-07 13:56 之后的全部入站消息**：replay 里的预决策字段，包括 `msg_chars`（字符数，**为 0 的多半是图片 / 语音 / 表情等非文本消息**）、`has_question`、`mentioned_bot`、`reply_to_bot`、`burst_len`、`group_activity_1m`。
3. **有文本的事件**（约 30–50%）：占位符里的媒体类型（`[图片]`、`[语音]`、`[视频]`、`[表情包]`、`[STICKER 名称]`、QQ 表情名、CQ 码），由 `tidal/modalities/meta.py` 解析，写入 `media_kinds` 和 `media_counts`。贴纸名称做哈希后存进 `media_refs`。CQ 码里如果带时长，会存进 `media_duration_s`（目前上游的文本里没有出现过）。
4. `kovi_bot_sticker_observations` 里的贴纸观测：只取 sticker_key 的哈希，用平台消息 ID 的哈希和事件关联，`user_text` 列不读。

拿不到的：图片和视频的像素、语音和音乐的音频、实际时长。要采集这些，需要使用方主动记录，tidal 这边不会去改 kovi-bot。

## 局限（哪些标签从被动数据推不出来）

- **EOT、续话和人类接话这几个标签都需要知道发言人**，而发言人只能从 LLM 快照里恢复（首轮约 49% 的事件可以打 EOT 标签）。快照之外的消息，这几个标签只能屏蔽。
- **"该不该说话"没有真值。** `y_act` 只是在模仿旧策略（离策略、有偏差）。被动数据无法回答"如果机器人当时说了 / 没说，效果会怎样"，这需要线上实验或人工标注。
- **被叫标签**依赖 replay 的 attention_reason（匹配率约 94%）和名字匹配，私聊不参与评测，而且正例非常少。
- **首轮是回填**：第一阶段导出之后的 361 条消息是在 1–10 小时之后才预测的。T 口径不受影响，因为它严格因果；但 TS 口径看到的文本会更全。所以看报告时建议加上 `--max-lag-min 30`。
- 上游的 replay 和快照表有保留期（都有 retention 索引），影子任务停下来太久的话，这些数据会丢。
- bge 嵌入在影子模式下是逐条编码的（int8 动态量化会让批量编码的结果依赖批的组成），和第一阶段的批量编码相比，TS 模型的输出最多差约 0.03。T 模型逐位一致（差 5e-7）。
- 一次 tick 大约需要 10–20 s（大部分时间花在 SSH 建连上），内存峰值约 0.6 GB（主要是首次加载 bge 和 ORT）；没有新消息时约 0.14 GB。
