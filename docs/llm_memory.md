# 冻结语言模型的测试时记忆 / Frozen-LM test-time memory

`tidal/llm_memory.py` 把一条事实写进 tidal 自己的 `EpisodicStore`，生成时把检索到的值加进冻结因果语言模型某一层的残差。事实文本不进 prompt。默认模型是 `HuggingFaceTB/SmolLM2-135M`。`torch` 和 `transformers` 只在这个模块里 import。安装：

```bash
pip install tidal[llm]
# or: pip install -r requirements-llm.txt
```

`tidal/llm_memory.py` writes each fact into tidal's own `EpisodicStore` and, at generation, adds the retrieved value to one layer of a frozen causal LM. The fact text is not in the prompt. The default checkpoint is `HuggingFaceTB/SmolLM2-135M`. `torch` and `transformers` are imported only there.

```python
from tidal.llm_memory import LLMMemory

mem = LLMMemory()  # downloads SmolLM2-135M on first use
mem.remember("Ada's phone is +1-415-555-0134", name="ada", rel="phone")
mem.generate("Ada's phone is")  # the number is not in this prompt
```

空库时 hook 不改那一层的输出，贪心生成和去掉 hook 的同一只模型一致。同 `(name, rel)` 覆盖整槽，读出来的是新值。没有 `enable_*` 开关。

An empty store does not change that layer's output, so greedy tokens match the same model with the hook removed. The same `(name, rel)` replaces the slot with the new value. There is no `enable_*` switch.

## 设计 / Design

这是当年 SmolLM 研究里留下来的 A2s 注入，槽用的是现在的 `EpisodicStore`，不再单独做一套记忆。

1. **答案跨度池化。** 在注入层的隐状态上，只对答案 span 的 token 做均值。不做 L2 归一化，幅度留给恒等拷贝。
2. **槽。** 一条事实一个槽 `(k, v, name, rel)`。同身份直接覆盖。容量默认 256，满了按写入时间 FIFO。近重复键（余弦 > 0.9，同一 group）也覆盖。
3. **soft top-4。** 温度 0.05，对 value 做凸组合。排序是原始余弦，和 `heads` / `recall` 一样。问句对比头仍然只在 `ask` / `query`（`tidal/question_head.py`），不接在这条 LM 路径上。
4. **注入。** 默认层是 `n_layers // 2`。加在该层残差上，每个位置同一向量：`add = v + MLP(v)`（`scale` 固定为 1）。MLP 是 Linear → GELU → Linear，末层零初始化，所以没训过的时候就是把 `v` 原样加进去。
5. **写门控。** `kind="chat"` / `"noise"` 不写槽。空库读出是零，生成时不加。

连续体 `heads` 上的电话、日期仍然是位置字节码，不走这个 MLP。在那些字节码上训过的 inject 会把 `+1-415-555-0134` 推离网格（最大绝对偏差 0.00111271），号码解不出来。那边的说明在 [episodic_memory.md](episodic_memory.md)。

This is the A2s inject from the SmolLM study, sitting on the current `EpisodicStore` rather than a second memory. The value is the mean of the answer-span hidden states at the inject layer, not L2-normalized. Slots overwrite on the same identity, evict FIFO at 256, and also overwrite a near-duplicate key (cosine > 0.9 in the same group). Readout is soft top-4 at temperature 0.05, ranked by raw cosine. The contrastive question head stays on `ask` / `query` only. The default layer is `n_layers // 2`. The add is `v + MLP(v)` with the MLP's last layer at zero, so an untrained module copies `v` through. Chat and noise writes are dropped. An empty store adds nothing.

Continuum `heads` still store phones and dates as positional byte codes. An inject MLP trained on those codes moved `+1-415-555-0134` by max abs 0.00111271 and the code no longer decoded, so that path does not use this MLP. See [episodic_memory.md](episodic_memory.md).

训练是教师强制的 next-token 交叉熵：prompt 里没有事实，注入的是该事实的答案跨度值。可选 20% 的错误值 hinge（`relu(2 - CE_wrong) * 0.3`），和当时的 v21 一样。

```bash
python -m tidal.llm_memory_train --tiny --steps 1 --n-cache 4 --neg-p 0   # CPU, no download
python -m tidal.llm_memory_train --steps 1500 --n-cache 2000              # SmolLM2-135M
```

Training is teacher-forced next-token cross-entropy: the prompt does not contain the fact, and the injected vector is that fact's answer-span pool. An optional 20% wrong-value hinge matches v21.

## 没留在这条路径上的 / Left off this path

| 做法 | 结果 |
|---|---|
| 矩阵读出，以及多块矩阵再求和（v16） | 值是和别的槽混在一起的外积，信号被稀释。v16 的 S4/S8 路由求和把 N32 自由生成打到大约 0.06–0.09。 |
| 朴素 key shield（v17） | 写完新事实又按旧键把旧读出加回去。同名更新倒置，new/old 约 **0.021 / 0.437**。 |
| 联合 CE 同时训检索和 inject（v23） | 没抬高 N0 天花板（停在 0.656），N16 还掉了点。不作为这里的训练目标。 |

A matrix readout, and a sum of several matrices (v16), mixes other slots into the value. The v16 routed sum fell to about 0.06–0.09 N32 free-gen. Naïve key shield (v17) wrote the new fact and then added the old key's readout back, so same-name updates inverted (new/old about **0.021 / 0.437**). Joint CE of retrieval and inject (v23) left the N0 ceiling at 0.656 and regressed N16, so this trainer does not do that.

长流上的对比 q/k（v20 / v22）留在 tidal 特征的 `ask` / `query`，不在这个 LM 模块里再训一个检索头。

The long-stream contrastive q/k adapters (v20 / v22) remain the `ask` / `query` head on tidal features. This module does not train a second retriever.

## 旧 SmolLM 研究的数字 / Old SmolLM study

下面全是**已删除的 neuromem 仓库**里、冻结 **SmolLM2-135M**、held-out test、**n=96** 自由生成字符串匹配的数字。不是 `python -m tidal.episodic_eval` 的精确匹配，也不是 tidal 特征上的 13/48、47/48。

All figures below are from the deleted neuromem study: frozen **SmolLM2-135M**, held-out test, **n=96** free-gen string match. They are not `python -m tidal.episodic_eval`, and they are not the 13/48 and 47/48 counts on tidal features.

| config | N0 | N4 | N16 | N32 | update new/old | K100 ret@1 |
|---|---|---|---|---|---|---|
| kind_protect (v15) | 0.583 | 0.417 | 0.333 | 0.208 | 0.542/0.000 | — |
| slot softk4 C256 (v19) | 0.594 | 0.479 | 0.458 | 0.469 | 0.667/0.000 | 0.451 |
| retr softk4 (v20) | 0.594 | 0.469 | 0.458 | 0.469 | 0.667/0.000 | 0.541 |
| retr softk4 + inj (v21) | **0.656** | 0.573 | 0.594 | 0.573 | 0.646/0.000 | 0.541 |
| retr v22 + inj softk4 | 0.656 | 0.573 | **0.604** | **0.594** | **0.667**/0.000 | **0.590** |
| bge RAG | 0.604 | 0.604 | 0.552 | 0.635 | — | — |
| fact in context | 0.604 | 0.615 | 0.562 | 0.688 | — | — |

N32 从 kind_protect 的 0.208 到 v22 的 0.594。更新大约 0.67 / 0，没有旧答案残留。K100 ret@1 从 v19 的 0.451 到 v22 的 0.590。同一套评测里 bge RAG 的 N32 是 0.635，事实直接放进上下文是 0.688。N0 在 ret@1=1 时，恒等注入是 0.594，残差 MLP 把它抬到 0.656。

N32 moved from 0.208 (kind_protect) to 0.594 (v22). Updates were about 0.67 / 0 with no stale answer. K100 ret@1 moved from 0.451 (v19) to 0.590 (v22). On that same eval, bge RAG at N32 was 0.635 and the fact-in-context baseline was 0.688. At N0 with ret@1=1, identity inject was 0.594 and the residual MLP raised it to 0.656.
