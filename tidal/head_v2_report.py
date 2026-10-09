"""Render bounded optimization outcomes without overstating exposed or proxy tests."""
import json

def markdown(r):
    reply=r['reply_to'];timing=r['overlap_timing'];stats=r['runtime_checks']['runtime']
    known=reply['known_ubuntu_diagnostic']
    lines=['# 任务头第二轮优化报告','','**本轮完成词权重回复指针、MagicData/MagicHub时序重叠辅助训练及100ms调度修复；仍为影子输出。**','',
           '回复指针使用验证集选择的细粒度温度校准；重叠头保留较粗校准网格，因为验证仅含一个对话块。','',
           '## 回复匹配','',
           f"验证选择：{reply['champion']}，训练/验证为 {reply['train_n']}/{reply['val_n']} 个查询；词IDF仅使用Ubuntu train。",'',
           f"固定channel_two测试（首轮后已暴露，本次作为回归诊断）：{reply['test_n']} 个查询，准确率 {reply['selected']['accuracy']:.4f}；冻结v1 {reply['baselines']['frozen_v1']['accuracy']:.4f}；同特征linear {reply['baselines']['linear']['accuracy']:.4f}。",'',
           '**该频道只有一个连续日志，因此只报告点指标，不声称显著升级；不是中文或助手草稿测试。**', '',
           f"已暴露的Ubuntu test回归诊断：v2 {known['selected']['accuracy']:.4f}、v1 {known['frozen_v1']['accuracy']:.4f}。此结果不参与选型、再校准或调参。",'',
           '输入仍是已经可用的草稿；增加4096桶train-only词IDF余弦、加权重叠、字符二元组、作者名边界匹配和可见作者历史。哈希词面特征不等于深层语义理解。', '',
           f"机器指标：`{json.dumps(reply['selected'])}`",'',
           '## 时序重叠辅助','',
           f"MagicData/MagicHub训练/验证/测试候选：{timing['train_n']}/{timing['val_n']}/{timing['test_n']}，对话块：{timing['blocks']}；验证选中 {timing['champion']}。",'',
           f"选中模型：`{json.dumps(timing['selected'])}`",'',
           f"基线：`{json.dumps(timing['baselines'])}`",'',
           f"按对话改善区间：`{json.dumps(timing.get('deltas',{}))}`；神经门槛通过：{timing['neural_passed_gate']}。",'',
           '只使用已完成历史、自己已讲话时长、200ms内已观察到的对方活动；不读取当前完整文本/未来总时长。候选和VAD状态源自参考ASR分段，尚未验证真实在线VAD边界。测试分组来自MagicData/MagicHub，且已在前轮任务查看，不包装为全新语料。', '',
           '标签仍是自然短附和或后续话轮接管的时序弱行为，不是“要求停止”的意图，也不证明该停TTS。缺失的人工策略标签没有被填成弱标签。','',
           '## 调度与产物','',
           '修复真实开口时刻与100ms tick错位导致重叠头几乎不触发的问题：首次覆盖开口+200ms的tick交付，每次开口只交付一次。去重使用单个开口时间水位，长期运行不累积开口集合；音频不足时不推进水位，可在交付窗口内重试。特征仍严格截在200ms，交付延迟期间的PCM不进入该次评分。新分数不改变真实控制动作。','',
           f"ONNX最大概率差：回复 {reply['export']['onnx_parity_max_error']:.2e}、时序 {timing['export']['onnx_parity_max_error']:.2e}。",'',
           f"独立v2模块回复完整调用p95 {stats['reply_p95_ms']:.3f}ms、时序p95 {stats['timing_p95_ms']:.3f}ms；峰值RSS {stats['peak_rss_mb']:.1f}MiB。32个合成候选、每次更换消息和草稿；不含VAD/ASR/内容生成/控制器。运行时不导入Torch/pandas/sklearn。",'',
           f"测试：{r['checks']['summary']}。",'',
           '一键入口：` .venv/bin/python -m tidal.head_v2_train `。恢复会校验源码、协议、源文件及每个产物SHA256。本地模型目录 `models/head_v2/`；可选 `V2Shadow` 只返回概率。使用见 [任务头说明](../docs/task_heads.md)，协议见 [训练前固定协议](head_v2_protocol.md)。','',
           '数据：[IRC作者卡](https://huggingface.co/datasets/jkkummerfeld/irc_disentangle)、[MagicData中文](https://huggingface.co/datasets/MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese)、[MagicHub英文](https://huggingface.co/datasets/MagicHub/multi-stream-spontaneous-conversation-training-datasets_english)，固定来源沿用本地清单，原始资料/文本/逐条预测/权重不提交Git。','']
    return '\n'.join(lines)
