# 策略、回复目标与在线让话任务头

新增任务使用独立的小型头和显式特征契约，六个主头/旧检查点兼容。全套可恢复执行：

```bash
.venv/bin/python -m tidal.head_optimize
```

协议：[task_heads_protocol.md](../reports/task_heads_protocol.md)。聚合结果：[task_heads.md](../reports/task_heads.md)。训练使用本地PyTorch，影子运行只需 `requirements-turn-runtime.txt` 的NumPy和ONNX Runtime。数据、权重、原文、逐条预测不会提交Git。

## 人工标注

`action_policy`：speak / wait / silent；`eot_semantic`：incomplete / complete；`recheck_policy`：0_2s / 2_5s / 5_10s / 10_20s / 20_60s / 60_300s / 300s_plus；`overlap_intent`：backchannel / floor_claim / stop_request / correction / unrelated。

同一条上下文在同一个决策时刻，每个task一行。下面是**格式示例**，不是训练真值；保存到 `data/head_annotations/judgements.jsonl` 时必须由实际标注者填写label和annotator。

```json
{"task":"action_policy","conversation":"local-group-1","decision_time":100.0,"context":[{"id":"m1","ts":99.0,"text":"问题的已到达文本","speaker":"u1","role":"other"}],"prefix":"","label":null,"label_source":"human","annotator":"pending","self_state":{"speaking":false,"elapsed_s":0.0,"remaining_s":0.0,"incoming_elapsed_s":0.0}}
```

- `ts` 是事件实际可见/到达时刻，不能把发言起点配上未来才识别完成的整句文本。
- `context` 只能含决策时已可见的事件，不超过32条会用于模型；未来上下文拒绝导入。
- 如有实时ASR `prefix`，必须同时填写 `prefix_available_at`，且不晚于 `decision_time`。重叠意图要求 `self_state.speaking=true`；要区分“停下”和附和，必须保存当时可见的转写前缀。
- `remaining_s` 是机器人自己已知的播放计划，不是人类语音未来总时长；`incoming_elapsed_s` 从已检测到的开口算起。
- `label_source` 只接受 human / user_feedback。未知标签null被屏蔽，冲突标注整组屏蔽，不靠多数票自动伪造真值；相同标签但上下文/状态不一致也屏蔽。
- 各人工任务的训练/验证/测试都需至少20例且覆盖所有类别，按conversation固定划分；不足会报告data_insufficient。重新标注/增加数据后可重新训练；旧测试数据不得按结果反复调参。

从本地事件流生成待填写的采集模板：输入每行 `{"conversation":"...","event":{...HeadEvent字段...}}`，按时间排序。

```bash
.venv/bin/python -m tidal.head_annotations prepare \
  --events data/head_annotations/events.jsonl \
  --out data/head_annotations/pending.jsonl
.venv/bin/python -m tidal.head_annotations validate data/head_annotations/judgements.jsonl
.venv/bin/python -m tidal.head_optimize --annotations data/head_annotations/judgements.jsonl
```

这个模板不产生标签。重叠意图样本请另外保存真实在线的prefix、可见时刻和播放状态，不能直接拿已结束的片段文本标注实时问题。MUIR目前需作者审核申请，因此默认流程会明确报告human_annotations_missing。

## 回复指针

```python
from tidal.task_heads import HeadEvent, TaskHeadShadow
shadow = TaskHeadShadow("models/task_heads")
history = [HeadEvent("m1", 99.0, "音频设置在哪里？", "u1", "other")]
# 草稿已经生成且已经可用，才能用于匹配；本例不代表中文效果。
score = shadow.reply(history, "音频设置在设置页面。", now=100.0)
```

候选0是null，其余是最后32条已到达事件，softmax只在有效候选上归一化。输出target和概率；null表示没有可见目标。IRC训练以自然人回复文本作为草稿代理，查询不加入候选；多个人工链接用集合概率损失。无连接不代表不值得回应，也不自动标成null。超过缓冲窗口的目标属于无可见目标。

## 控制器影子接入

```python
from tidal.head_shadow import ControllerTaskShadow
from tidal.duplex import DuplexController, TickInput
controller = DuplexController(tick_model, event_encoder,
    task_shadow=ControllerTaskShadow("models/task_heads"))
# 原有TickInput字段不变；有草稿/ASR前缀时额外传对应的实际到达时刻。
result = controller.step(TickInput(t=100.0,
    draft_text="已经可用的草稿", draft_available_at=99.9))
print(result.task_shadow)  # 没有训练过的任务输出None
```

`Event.text` 是可选的新字段。`TickInput` 新增 draft_text/draft_available_at、incoming_prefix/prefix_available_at、incoming_onset。需要实时重叠结果时，PCM按(incoming, self)传入、16kHz、同步连续块，并填写当前机器人是否讲话及已讲话时长。ASR前缀不与完成文本互换。

`overlap_outcome` 只在 incoming_onset+200ms、机器人仍讲话且之前已持有话轮至少500ms时输出。输入为决策前完整可用的50帧；每帧可用时刻35ms+20ms*k。机器人已停、音频缺失/不连续或上下文不足时不评分。弱标签预测后续自然话轮行为，不表示用户停止指令。

`ControlOut.task_shadow` 与原action/address_event/request_content_for/stop_tts分开；异常只有task_shadow_error。不会用未通过目标场景验证的概率停TTS或改变回复对象。

## 旧头的含义

- `y_act`/`y_act_behavior`：历史机器人行为；`y_act_policy`：单独的人工策略监督。原六头训练仍用历史行为目标，不能宣称已经升级为策略适当性模型。
- `y_eot`：事件间隔与换人弱标签；`eot_semantic`：人工语义完整度独立头，无标签时不训练。
- `y_recheck`/`y_next_gap`：到下一事件的七时间桶；`recheck_policy`：人工认为值得重新判断的时间桶。
- `p_interrupt`：事后片段分类；`p_barge`：自己未来插话；`overlap_outcome`/`overlap_intent`：自己正说话时，对方开口后的在线辅助结果/语义意图。

最终校验/来源约束补强后，12个模型的权重、归一化和逐条预测均与初次冻结结果一致；记录见 [校验报告](../reports/task_heads_parity.json)。再次执行一键命令已实测恢复产物并完成全部测试，没有重新训练。


## 第二轮 v2

第二轮入口：

```bash
.venv/bin/python -m tidal.head_v2_train
```

`reply_to_v2` 仍要求已有草稿，增加只从Ubuntu train拟合的词权重、字符和作者历史特征；训练后固定评估作者 `channel_two/test`。`overlap_timing` 使用MagicData/MagicHub已锁定分组，并在第一次覆盖“对方开口+200ms”的100ms tick交付；交付延迟音频不进入该次特征。两个v2头都只返回影子概率，实际动作保持由旧控制器产生。

v2模型需要用 `tidal.head_v2_runtime.V2Shadow` 加载；加载器核对特征名、类别、模型与IDF校验和。旧版 `TaskHeadShadow` 和 v2 分开，避免不同特征合同混用。
