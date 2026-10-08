# 无人值守优化：完成报告

**数据、训练、独立评估、ONNX导出、流式验证及可恢复执行均已完成。没有证据支持升级中文主模型，因此保持现有真实动作策略。**

## 实际训练与评估

本轮固定方案见 [audio_turn_v2_protocol.md](audio_turn_v2_protocol.md)，不是在原中文测试集上反复调参。

| 场景 | 训练 / 验证 / 独立测试候选数 | 神经ROC-AUC | LR ROC-AUC | 采用概率指标门槛 |
|---|---:|---:|---:|---|
| CANDOR时序 | 29666 / 7279 / 6101 | 0.6599 | 0.6575 | 小幅通过 |
| 英文分轨外部音频 | 1349 / 679 / 1409 | 0.6151 | 0.6261 | 未通过 |

时序任务按UUID划分为 1144 / 275 / 236 段对话。ROC增量95%区间 [0.000111, 0.004716]，Brier改善区间 [0.000034, 0.001224]。只保证对话隔离：公开资料没有参与者身份，无法证明跨集合无重复参与者，对话级区间也未覆盖这种相关性。

音频候选比较原始谱输入和窗口CMVN，均为48维GRU、三个种子；验证集选中CMVN。外部测试为提前锁定的英文Group0078、两名新说话人、五段对话；实际speaker ID与训练/验证不重叠。神经相对LR的ROC增量区间 [-0.0306, 0.0108]，未胜出。测试Brier 0.265，也比类别先验 0.178 更差，显示跨域概率失准；没有按测试分布重新校准。

CANDOR时序模型再作中文零样本诊断：ROC 0.464，LR 0.474，不能直接迁移。中文资料已在前轮实验中查看，因此明确标为**开发数据诊断**，不包装成新的独立测试；模型和校准均未因此改变。

## 接话策略结论

固定阈值.75（正确接话收益1、误接话代价3）下，时序模型接话 107 次、错误 29 次，召回 2.6%；英文音频模型接话 17 次、错误 13 次，召回 1.5%。效用均为负。小幅排序收益不等于有效接话策略。

公开资料只有自然发言行为和片段时间，CANDOR的适当性/共识/accepted等列为空；没有把空占位值当标签，也不读取`next_*`等未来列。机会依赖参考分段，CANDOR本身为ASR片段，尚未验证实时VAD/ASR边界和助手发言适当性。**所有导出的模型仅返回影子分数，不选择动作、不请求内容、不停止TTS，也不替换现有模型。**

## 落地的工程优化

- 声学帧在35ms完整到达后才可用；精确筛选+200ms停顿机会，避免决策之后的音频参与筛选。
- VA/VAP/附和训练目标统一到帧可用时间；未来未观察完整的目标屏蔽损失，旧标签检查点明确拒绝作新评测。
- BOM、纯噪声、无效片段处理及缓存版本；下载版本锁定、并发重试及不完整下载明确失败。
- LR正则、神经候选、早停、单调校准均只在训练/验证选择；冻结文件及权重SHA256后才评测新测试集合。
- `.venv`已安装，环境版本记录在 [requirements-optimization-macos.lock](../requirements-optimization-macos.lock)。重复运行会恢复已完成模型，不重新训练。
- 独立影子运行时只需NumPy和ONNX Runtime；不导入PyTorch、sklearn、pandas。训练和推理共用纯NumPy时序/声学定义。

## 实测推理

| 独立影子模块 | 参数（包含三个种子的ensemble） | ONNX体积 | 统计特征+ONNX评分p95 | 新进程峰值RSS |
|---|---:|---:|---:|---:|
| 时序 | 1,491 | 约17KB | 0.206ms | 57.4MiB |
| 音频 | 72,756 | 约307KB | 0.852ms | 63.4MiB |

本机macOS arm64、CPython3.12.14，ONNX Runtime使用两个CPU线程；基准使用100段合成历史。音频100ms PCM处理+评分的完整调用p95 **1.223ms**。这些是独立影子模块数据，不包括上游VAD/ASR、文本编码器、内容生成器或整个控制系统。

ONNX与PyTorch最大概率差约1.2e-7；PCM整块/100ms/不规则小块一致，实测最大差 0.0e+00；决策后的PCM及事件不会改变已存在的过去预测，帧缓存有界。拆分训练依赖后，九个模型全部权重、归一化及测试预测逐项与拆分前一致，证明没有利用测试结果重调参数；原冻结清单保存在本地，校验记录见 [turn_v2_refactor_parity.json](turn_v2_refactor_parity.json)。

验证：全部111项测试通过。一键流程在测试失败时明确失败，不会报告完成。

## 数据与产物

- [MagicData中文](https://huggingface.co/datasets/MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese)：27段对话、六名说话人、54轨，10.489小时轨道 / 5.245小时双人时长。
- [MagicHub英文](https://huggingface.co/datasets/MagicHub/multi-stream-spontaneous-conversation-training-datasets_english)：8段对话、八名说话人、16轨，5.532小时轨道 / 2.766小时双人时长。
- [CANDOR标注](https://huggingface.co/datasets/hiraki/candor-turntaking-annotations)：172591个ASR片段、1656段对话，CC-BY-4.0；其中1655段存在本协议下的有效接话候选。

MagicData/MagicHub仅学术研究、不可商用；原始资料、缓存、训练权重及ONNX均留在Git忽略的本地目录。公开数据本轮不需要token，未写入任何认证信息。

可恢复的完整入口：

```bash
.venv/bin/python -m tidal.optimize
```

本地影子产物：`models/turn_v2/timing_champion.json/.onnx`、`audio_champion.json/.onnx`；接口见 [turn_shadow.py](../tidal/turn_shadow.py)，轻量运行依赖见 [requirements-turn-runtime.txt](../requirements-turn-runtime.txt)。

机器结果：[时序](turn_v2_timing.json)、[音频](turn_v2_audio.json)、[中文迁移诊断](turn_v2_zh_transfer.json)、[运行时](turn_v2_runtime.json)。源码、报告和权重不改变现有线上行为。
