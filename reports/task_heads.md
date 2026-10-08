# 任务头落地报告

**代码、训练和影子接口已落地；策略适当性和语义意图仍受人工标注数量约束。所有新输出只供影子评估。**

| 任务 | 状态 | 训练 / 验证 / 测试 | 选型 | 测试准确率 |
|---|---|---:|---|---:|
| reply_to | trained_evaluated | 59217/2321/4605 | mlp | 0.6619 |
| overlap_outcome | trained_evaluated | 226/21/166 | linear | 0.5301 |
| action_policy | human_annotations_missing | 0/0/0 | 未训练 | 未测 |
| eot_semantic | human_annotations_missing | 0/0/0 | 未训练 | 未测 |
| recheck_policy | human_annotations_missing | 0/0/0 | 未训练 | 未测 |
| overlap_intent | human_annotations_missing | 0/0/0 | 未训练 | 未测 |

## 监督与能力边界

- 主模型仍为六个头。`y_act_behavior` 保留历史行为，`y_act_policy` 只接受有来源的人工标签；`y_next_gap` 明确现有重检头实际预测的是下一事件时间。
- `action_policy`、`eot_semantic`、`recheck_policy`、`overlap_intent` 均有人工JSONL校验和训练入口；缺标注时不创建随机权重模型，不用自然接话或合成数据冒充助手适当性。
- `reply_to` 以已经可用的草稿为条件，输出最近32条消息或null。用IRC人工图训练的是自然人回复匹配代理；不证明机器人草稿或中文群聊迁移。无链接行屏蔽；超出缓冲目标明确映射为“无可见目标”。
- `overlap_outcome` 在自己仍讲话、对方开口200ms时，使用已经完整收到的1s双通道音频；预测后续短附和或话轮接管的弱行为标签。它没有 stop_request / correction 真值，也不代表应该停TTS。
- 所有选型、归一化、温度校准在训练/验证完成；冻结权重SHA后才构建测试。结果无论正负均报告。
- 音频说话人隔离，但沿用了前轮其他任务已查看的外部英文集合；不是全新未暴露的数据。按对话的区间仅有五个测试块、两名说话人，证据有限。
- 新策略与目标输出接入 `ControlOut.task_shadow`；与原动作、回复目标、内容请求和TTS停止信号分开。异常写入 `task_shadow_error`，不改变原控制结果。
- 特征/推理只依赖NumPy和ONNX Runtime；哈希文本是轻量词面特征，不应视作成熟语义理解。

## 实际训练结果

### reply_to

验证选择：mlp；神经模型通过预定门槛：True。

选中模型：`{"n": 4605, "accuracy": 0.6618892508143323, "nll": 1.1522778557087994, "null_precision": 0.6282485875706215, "null_recall": 0.8593508500772797, "visible_target_rate": 0.8597176981541802, "null_target_rate": 0.14049945711183495}`

基线：`{"null": {"n": 4605, "accuracy": 0.14049945711183495, "nll": 7.891914135542321, "null_precision": 0.14049945711183495, "null_recall": 1.0, "visible_target_rate": 0.8597176981541802, "null_target_rate": 0.14049945711183495}, "latest": {"n": 4605, "accuracy": 0.304885993485342, "nll": 6.396114577436722, "null_precision": 0.0, "null_recall": 0.0, "visible_target_rate": 0.8597176981541802, "null_target_rate": 0.14049945711183495}, "mention": {"n": 4605, "accuracy": 0.46406080347448425, "nll": 4.9314869989132335, "null_precision": 0.0, "null_recall": 0.0, "visible_target_rate": 0.8597176981541802, "null_target_rate": 0.14049945711183495}, "linear": {"n": 4605, "accuracy": 0.6128121606948969, "nll": 1.3766984125211672, "null_precision": 0.5678642714570858, "null_recall": 0.8794435857805255, "visible_target_rate": 0.8597176981541802, "null_target_rate": 0.14049945711183495}}`

按块改善区间：`{"null": {"blocks": 10, "accuracy_improvement_ci": [0.4881442272877981, 0.5471955939659289], "nll_improvement_ci": [6.586709330868204, 6.8888997672399155]}, "latest": {"blocks": 10, "accuracy_improvement_ci": [0.2612087122620149, 0.44432933655722], "nll_improvement_ci": [4.545470936767014, 5.931489073466985]}, "mention": {"blocks": 10, "accuracy_improvement_ci": [0.13751047524868049, 0.25231465357726435], "nll_improvement_ci": [3.3913224442601377, 4.202654754414279]}, "linear": {"blocks": 10, "accuracy_improvement_ci": [0.036590329960162066, 0.06167232789647134], "nll_improvement_ci": [0.1892183507188947, 0.2628919663588253]}}`

ONNX最大概率差：8.94e-08；参数：1347。

### overlap_outcome

验证选择：linear；神经模型通过预定门槛：False。

选中模型：`{"n": 166, "accuracy": 0.5301204819277109, "nll": 0.6961445720581222, "macro_f1": 0.5284090909090908, "balanced_accuracy": 0.5477245862884161, "roc_auc": 0.5695921985815603, "brier": 0.2513745717105231}`

基线：`{"prior": {"n": 166, "accuracy": 0.5662650602409639, "nll": 0.6858017538330294, "macro_f1": 0.36153846153846153, "balanced_accuracy": 0.5, "roc_auc": 0.5, "brier": 0.24631944647569842}, "linear": {"n": 166, "accuracy": 0.5301204819277109, "nll": 0.6961445720581222, "macro_f1": 0.5284090909090908, "balanced_accuracy": 0.5477245862884161, "roc_auc": 0.5695921985815603, "brier": 0.2513745717105231}}`

按块改善区间：`{"prior": {"blocks": 5, "accuracy_improvement_ci": [-0.18543046357615894, 0.08839779005524862], "nll_improvement_ci": [-0.06716927479152146, 0.03402084163714818]}, "linear": {"blocks": 5, "accuracy_improvement_ci": [0.0, 0.0], "nll_improvement_ci": [0.0, 0.0]}}`

ONNX最大概率差：0.00e+00；参数：90。

## 独立影子运行时实测

回复特征构建+ONNX的p95 0.157ms，新进程峰值RSS 56.6MiB。每次迭代更换一个消息和草稿；32个合成候选，不包含VAD/ASR、生成模型或完整控制器。

音频整块/100ms/不规则块最大概率差 0.0e+00；未来文本及未来PCM不改变过去评分；音频缓存有界；新进程不导入Torch/pandas/sklearn。

## 使用

```bash
.venv/bin/python -m tidal.head_optimize
# 有真实人工标注后：
.venv/bin/python -m tidal.head_optimize --annotations data/head_annotations/judgements.jsonl
```

人工格式、采集命令和影子接入示例见 [任务头使用说明](../docs/task_heads.md)。模型、原文及逐条预测保存在本地忽略目录。

数据：[IRC作者卡](https://huggingface.co/datasets/jkkummerfeld/irc_disentangle)、[MUIR访问说明](https://github.com/Eliot-Shen/GroupGPT#MUIR-Dataset)。IRC为CC-BY-4.0；MagicData/MagicHub仅学术研究不可商用；本轮未使用认证token。

验证：完整pytest退出码 0；模型恢复依赖源码、协议、数据内容和产物校验。
