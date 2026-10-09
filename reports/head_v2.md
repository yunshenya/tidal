# 任务头第二轮优化报告

**本轮完成词权重回复指针、MagicData/MagicHub时序重叠辅助训练及100ms调度修复；仍为影子输出。**

回复指针使用验证集选择的细粒度温度校准；重叠头保留较粗校准网格，因为验证仅含一个对话块。

## 回复匹配

验证选择：mlp，训练/验证为 59217/2321 个查询；词IDF仅使用Ubuntu train。

此前未查看的channel_two测试：796 个查询，准确率 0.6244；冻结v1 0.5980；同特征linear 0.5854。

**该频道只有一个连续日志，因此只报告点指标，不声称显著升级；不是中文或助手草稿测试。**

已暴露的Ubuntu test回归诊断：v2 0.7001、v1 0.6619。此结果不参与选型、再校准或调参。

输入仍是已经可用的草稿；增加4096桶train-only词IDF余弦、加权重叠、字符二元组、作者名边界匹配和可见作者历史。哈希词面特征不等于深层语义理解。

机器指标：`{"n": 796, "accuracy": 0.6243718592964824, "nll": 1.1729133191829293, "null_precision": 0.5056179775280899, "null_recall": 0.6617647058823529, "visible_target_rate": 0.914572864321608, "null_target_rate": 0.08542713567839195}`

## 时序重叠辅助

MagicData/MagicHub训练/验证/测试候选：226/21/166，对话块：{'train': 21, 'val': 1, 'test': 5}；验证选中 mlp。

选中模型：`{"n": 166, "accuracy": 0.5662650602409639, "nll": 0.7186937842994602, "macro_f1": 0.5553571428571429, "balanced_accuracy": 0.597517730496454, "roc_auc": 0.6292848699763594, "brier": 0.2606346571643976}`

基线：`{"linear": {"n": 166, "accuracy": 0.46987951807228917, "nll": 0.7544373524334193, "macro_f1": 0.43720141778394206, "balanced_accuracy": 0.5107860520094563, "roc_auc": 0.492612293144208, "brier": 0.2788916514957073}, "prior": {"n": 166, "accuracy": 0.5662650602409639, "nll": 0.6858017538330294, "macro_f1": 0.36153846153846153, "balanced_accuracy": 0.5, "roc_auc": 0.5, "brier": 0.24631944647569842}}`

按对话改善区间：`{"linear": {"blocks": 5, "accuracy_improvement_ci": [0.02976190476190476, 0.208955223880597], "nll_improvement_ci": [-0.02966706458524314, 0.09621737652329435]}, "prior": {"blocks": 5, "accuracy_improvement_ci": [-0.16477272727272727, 0.14210526315789473], "nll_improvement_ci": [-0.14876026612687668, 0.07050271988946545]}}`；神经门槛通过：False。

只使用已完成历史、自己已讲话时长、200ms内已观察到的对方活动；不读取当前完整文本/未来总时长。候选和VAD状态源自参考ASR分段，尚未验证真实在线VAD边界。测试分组来自MagicData/MagicHub，且已在前轮任务查看，不包装为全新语料。

标签仍是自然短附和或后续话轮接管的时序弱行为，不是“要求停止”的意图，也不证明该停TTS。缺失的人工策略标签没有被填成弱标签。

## 调度与产物

修复真实开口时刻与100ms tick错位导致重叠头几乎不触发的问题：首次覆盖开口+200ms的tick交付，每次开口只交付一次。特征仍严格截在200ms，交付延迟期间的PCM不进入该次评分。新分数不改变真实控制动作。

ONNX最大概率差：回复 1.19e-07、时序 5.96e-08。

独立v2模块回复完整调用p95 0.456ms、时序p95 0.034ms；峰值RSS 57.7MiB。32个合成候选、每次更换消息和草稿；不含VAD/ASR/内容生成/控制器。运行时不导入Torch/pandas/sklearn。

测试：125 passed, 14 warnings in 7.51s。

一键入口：` .venv/bin/python -m tidal.head_v2_train `。恢复会校验源码、协议、源文件及每个产物SHA256。本地模型目录 `models/head_v2/`；可选 `V2Shadow` 只返回概率。使用见 [任务头说明](../docs/task_heads.md)，协议见 [训练前固定协议](head_v2_protocol.md)。

数据：[IRC作者卡](https://huggingface.co/datasets/jkkummerfeld/irc_disentangle)、[MagicData中文](https://huggingface.co/datasets/MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese)、[MagicHub英文](https://huggingface.co/datasets/MagicHub/multi-stream-spontaneous-conversation-training-datasets_english)，固定来源沿用本地清单，原始资料/文本/逐条预测/权重不提交Git。
