# 任务头落地协议（训练前固定）

日期：2026-10-09。目标：1 个监督重定义 + 回复指针 + 在线让话信号；全程影子输出。现有六主头及真实动作不自动替换。

## 监督定义

- `y_act` 保留历史行为含义；另显式输出 `y_act_behavior`。新增 `action_policy` 只接受 human / user_feedback 标注的 speak / wait / silent，不把未发生的动作或自然人接话当作助手适当性。
- `eot_semantic`（incomplete / complete）和 `recheck_policy`（七时间桶）使用同一人工标注入口。现有 `y_recheck` 是下一事件时间，不等于最优重检时间；新增 `y_next_gap` 作为真实含义别名。无可靠人工标注时，不训练这些策略头。
- `reply_to` 是 **已有候选回复草稿条件下** 的指针；含“无可见目标”。IRC 自然人回复文本是查询，人工 reply-link 是监督。这是草稿目标匹配代理任务，不能声称预测尚未生成的回复或已证明机器人迁移。在线只有草稿存在才给分。
- 在线 `overlap_intent` 类别：backchannel / floor_claim / stop_request / correction / unrelated。使用决策时已可见的语音转写前缀和机器人播放状态；需要人工意图标签。
- 另训练 `overlap_outcome` 声学辅助头：在机器人正讲话、对方开口200ms之后，预测对方后续是短附和还是拿到话轮。双通道自然对话的时序弱标签不能替代 stop_request 等意图，更不能证明“应该停TTS”。歧义样本屏蔽。

## 输入与标签边界

人工 JSONL 的 context 只含 decision_time 之前到达的事件；prefix 必须带 prefix_available_at <= decision_time。当前语音完整文本、结束时间、未来时长及下一片段均不可输入在线重叠头。时间/PCM特征共用纯NumPy实现，音频帧可用时刻 k*20ms+35ms。

真实群聊上下文、人工样本、原始文本、权重和逐条预测只存本地忽略路径。公开报告仅聚合。

## 固定划分与搜索

- IRC：HF作者官方 train / validation / test；按日志日期检查不重叠。revision f6cfa2cefb604d2fbabcec58be01ee34939fdd54，CC-BY-4.0。每查询最多32个过去候选 + null；连接是双向邻接，只把更早连接与self-root当目标；future-only / 无连接行不当负样本。超出缓冲的目标映射“无可见目标”，多目标用集合概率损失。
- 音频：沿用已下载语料固定分组：train zh A1012/A1091 + en Group0006/0030；val zh A1102 + en Group0046；test en Group0078。两种语言缓存来自锁定revision。测试说话人不重叠，但该测试集合曾在前轮不同任务评测中查看，不称为全新数据。
- 人工样本：按conversation SHA256前8位 mod100，train<70，val<85，test其余。每一任务要求三集合均含全部类别；不满足就明确 data_insufficient，而不是用合成样本宣称实测效果。
- 候选仅 linear 与32维MLP，固定种子0/1/2，AdamW lr=.001，最多20epoch，patience4。归一化仅train。神经ensemble由val loss与linear ensemble比较；校准温度在val选 .5/.75/1/1.5/2/3。冻结模型、来源/源码签名、选型与校准后才读取test标签。
- 回复基线：null、最近消息、最近被点名作者、linear；报告候选覆盖率、top1集合准确率、null precision/recall、NLL、每日志日期bootstrap差值区间。
- 重叠基线：训练先验、linear；报告ROC-AUC、Brier、balanced accuracy和按对话bootstrap区间。
- 人工任务：macro-F1、accuracy、NLL；策略质量与任务代理指标分开。性能门槛要求test的准确率改善区间下界>0且NLL改善区间下界>0；样本或块不足不采用。策略部署仍需目标场景评测。

## 执行与完成标准

一键入口下载/验证数据 → 构建训练/验证 → 有界训练 → 冻结 → 测试 → ONNX导出/概率一致性 → 完整pytest → 聚合报告。缓存只在源码、数据内容及协议签名一致且模型校验一致时恢复。

MUIR作者当前不公开下载，需申请：[作者说明](https://github.com/Eliot-Shen/GroupGPT#MUIR-Dataset)。不自动发送申请，不使用合成/教师标签冒充人工。没有人工标注时，交付完整schema、采集模板、训练/推理接口，并在报告标明策略头未训练。
