# 冻结语言模型的测试时记忆 / Frozen-LM test-time memory

`tidal/llm_memory.py` 把一条事实写进 tidal 自己的 `EpisodicStore`，生成时把检索到的值加进冻结因果语言模型某一层的残差。事实文本不进 prompt。默认模型是 `HuggingFaceTB/SmolLM2-135M`。`torch` 和 `transformers` 只在这个模块里 import。安装：

```bash
pip install tidal[llm]
# or: pip install -r requirements-llm.txt
```

`tidal/llm_memory.py` writes each fact into tidal's own `EpisodicStore` and, at generation, adds the retrieved value to one layer of a frozen causal LM. The fact text is not in the prompt. The default checkpoint is `HuggingFaceTB/SmolLM2-135M`. `torch` and `transformers` are imported only there.

```python
from tidal.llm_memory import LLMMemory

mem = LLMMemory()  # downloads SmolLM2-135M; default layer loads the shipped trained inject
mem.remember("Ada's phone is +1-415-555-0134", name="ada", rel="phone")
mem.generate("Ada's phone is")  # the number is not in this prompt
```

空库时 hook 不改那一层的输出，贪心生成和去掉 hook 的同一只模型一致。同 `(name, rel)` 覆盖整槽，读出来的是新值。没有 `enable_*` 开关。

`LLMMemory()` 自己加载默认的 `HuggingFaceTB/SmolLM2-135M`、并且注入层是默认的 `n_layers // 2`（第 15 层，隐藏维 576）时，会装上随包的 `tidal/llm_inject_smollm2_135m.pt`。这是键修复之后训的 1500 step。调用方传入自己的模型、换一个 checkpoint、或换一层，末层仍是零，加进去的是 `v`，直到调用 `load_inject`。没有把这条路径关掉的开关。

An empty store does not change that layer's output, so greedy tokens match the same model with the hook removed. The same `(name, rel)` replaces the slot with the new value. There is no `enable_*` switch.

`LLMMemory()` loads `tidal/llm_inject_smollm2_135m.pt` when it itself loads `HuggingFaceTB/SmolLM2-135M` at the default inject layer (`n_layers // 2`, layer 15, hidden 576). That file is the 1500-step MLP trained after the key fix. A caller-supplied model, another checkpoint, or another layer keeps the zero last layer, so the add is `v`, until `load_inject`. There is no switch that turns this off.

## 设计 / Design

这是当年 SmolLM 研究里留下来的 A2s 注入，槽用的是现在的 `EpisodicStore`，不再单独做一套记忆。

1. **答案跨度池化。** 在注入层的隐状态上，只对答案 span 的 token 做均值。不做 L2 归一化，幅度留给恒等拷贝。值不减中心，也不丢弃 sink token。
2. **键。** 键不是这个池。SmolLM2-135M 的第一个 token 是注意力 sink：整向量范数大约是其他 token 的 100 倍，第 507 维大约两万。对 token 做均值时，键几乎被这个 sink 决定，不同事实的余弦大约 0.99，默认 `overwrite_cos=0.9` 会把一条流收成一个槽。主路径上，键和问句先丢掉范数超过该字符串 token 范数中位数 8 倍的 token（只看当前字符串，不是会话状态），再减去一次标定得到的固定中心。事实键减 `key_mean`，问句减 `query_mean`。中心来自 12 个标定名字（Sam、Robin、Alex、Jordan、Casey、Riley、Avery、Quinn、Parker、Drew、Shawn、Logan）乘 6 种关系，和评测名字不相交；答案 span 不参与，所以中心不依赖答案文本。默认 SmolLM2-135M 第 15 层的中心存在 `tidal/llm_key_center.npz`。传入别的模型或层时，初始化时用同一批文本估一次，之后不随事实更新。没有开关。
3. **槽。** 一条事实一个槽 `(k, v, name, rel)`。同身份直接覆盖。容量默认 256，满了按写入时间 FIFO。近重复键（余弦 > 0.9，同一 group）也覆盖。
4. **读出是 hard top-1。** 排序是原始余弦，取余弦最大的那一槽的值，不做凸组合。`EpisodicStore` 里的 soft top-k 还在，连续体 `heads` 仍用 soft top-4、温度 0.05；这条 LM 路径不走那个混合。测量见下面的消融：温度 0.05 的 soft top-4 和 top-1 在前缀上打平，问句上 top-1 更高，温度 0.20 会把别的槽混进来并掉点。问句对比头仍然只在 `ask` / `query`（`tidal/question_head.py`），不接在这条 LM 路径上。这条路径不另训低维投影：在 sink 丢掉并减去固定中心之后，键已经能分开。
5. **注入。** 默认层是 `n_layers // 2`（SmolLM2-135M 是第 15 层）。加在该层残差上，每个位置同一向量：`add = v + MLP(v)`（`scale` 固定为 1）。MLP 是 Linear → GELU → Linear，末层零初始化，所以没装上训练权重时就是把 `v` 原样加进去。`LLMMemory` 自己加载默认模型、并且就是这一层时，会换成 `tidal/llm_inject_smollm2_135m.pt`。调用方传入的模型、别的 checkpoint、别的层，末层保持为零。文件内容和 `save_inject` 一样：inject 的 state dict，加上 `layer`、`hidden`。
6. **写门控。** `kind="chat"` / `"noise"` 不写槽。空库读出是零，生成时不加。

连续体 `heads` 上的电话、日期仍然是位置字节码，不走这个 MLP。在那些字节码上训过的 inject 会把 `+1-415-555-0134` 推离网格（最大绝对偏差 0.00111271），号码解不出来。那边的说明在 [episodic_memory.md](episodic_memory.md)。

This is the A2s inject from the SmolLM study, sitting on the current `EpisodicStore` rather than a second memory. The value is the mean of the answer-span hidden states at the inject layer, not L2-normalized, not centered, and not sink-filtered. The key is a different pool. On SmolLM2-135M the first token is an attention sink (norm about 100× the other tokens, hidden unit 507 near 2e4). A mean over tokens is that sink, distinct facts land near cosine 0.99, and the default cosine overwrite of 0.9 keeps one slot. Keys and queries drop tokens whose norm exceeds 8× the median token norm of that same string, then subtract a fixed center: `key_mean` for fact keys, `query_mean` for questions. The centers are estimated once on 12 calibration names (Sam, Robin, Alex, Jordan, Casey, Riley, Avery, Quinn, Parker, Drew, Shawn, Logan) times 6 relations, disjoint from the held-out eval names. The answer span is excluded, so the center does not depend on the answer text. The default SmolLM2-135M layer-15 centers ship in `tidal/llm_key_center.npz`. A passed-in model, or another layer, fits those same strings at init and does not update as facts arrive. There is no switch and no learned low-dim projection: after the sink drop and the fixed center, the keys separate. Slots overwrite on the same identity, evict FIFO at 256, and also overwrite a near-duplicate key (cosine > 0.9 in the same group). Readout on this path is hard top-1, ranked by raw cosine. `EpisodicStore` still implements soft top-k, and Continuum `heads` still use soft top-4 at temperature 0.05; this LM path does not mix. The ablation below is why: soft top-4 at temperature 0.05 tied top-1 on the answer-stripped prefix and trailed it on natural questions, and temperature 0.20 mixed other slots in and lost hits. The contrastive question head stays on `ask` / `query` only. The default layer is `n_layers // 2`. The add is `v + MLP(v)` (`scale` fixed at 1). The MLP is Linear → GELU → Linear with a zero last layer, so an untrained module copies `v` through. When `LLMMemory` itself loads the default SmolLM2-135M checkpoint at that layer, it replaces the MLP with `tidal/llm_inject_smollm2_135m.pt`. A caller-supplied model, another checkpoint, or another layer keeps the zero last layer. The file is the `save_inject` dict: the inject state dict plus `layer` and `hidden`. Chat and noise writes are dropped. An empty store adds nothing.

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
| 中心之后再去主成分、白化或投到低维 PCA | 间隙变窄，或者余弦重新贴到 1、槽又合并。不在主路径上。 |
| 再训一个对比用的低维 key/query 投影 | sink 丢掉并减去固定中心之后，键已经分开。见下面的 K100。 |

A matrix readout, and a sum of several matrices (v16), mixes other slots into the value. The v16 routed sum fell to about 0.06–0.09 N32 free-gen. Naïve key shield (v17) wrote the new fact and then added the old key's readout back, so same-name updates inverted (new/old about **0.021 / 0.437**). Joint CE of retrieval and inject (v23) left the N0 ceiling at 0.656 and regressed N16, so this trainer does not do that. Removing principal components, whitening, or a low-dim PCA after the center narrowed the gap or pushed cosine back to 1 and slots merged again. A second contrastive key/query projection is not on this path: after the sink drop and the fixed center, the keys already separate.

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

## 这次键的测量 / This key fix, measured

下面是本仓库 `scripts/eval_llm_memory.py` 在冻结 **SmolLM2-135M**、**float32**、CPU 上的数字。事实生成器是 `tidal/llm_memory_data.py`（评测名字和标定名字不相交；关系含电话、生日和多词答案；一个 N 的会话是 1 条目标事实加 N 个干扰轮，干扰轮里事实和闲聊各一半）。检索问句是自然语言（`What is Nell's phone number?`）。自由生成用的是去掉答案后的前缀（`Nell's phone number is `），最多 16 个新 token，停规则和旧 v22 一样（换行或句末标点），匹配是字母数字包含。前缀是因为这条模型对自然问句的贪心续写以换行开头，v22 停规则会把每一种系统都打成空串。事实进上下文和 bge RAG 都是「被选中的那一行 + 换行 + 同一个前缀」。bge 用自然问句检索。这一节的自由生成是没训练的恒等注入，也就是键修复之后、装上 `llm_inject_smollm2_135m.pt` 之前的测量。`n=32` 个会话；K20/K50 各 4 条流，K100 两条流。Transformers 5 默认把这只模型载成 bfloat16，这台机器上 bfloat16 一次前向大约 1.7 秒，float32 大约 0.03 秒。中心是在 bfloat16 隐状态上估的，float32 直接用同一份。另有一条 bfloat16 的 24 事实流：24/24 槽，问句 ret@1 23/24，键的非对角余弦均值 0.092、最大 0.802。

Numbers below are from `scripts/eval_llm_memory.py` on frozen **SmolLM2-135M** in **float32** on CPU. Facts come from `tidal/llm_memory_data.py` (eval names disjoint from the calibration names; relations include phone, birthday, and multi-token answers; an N session is one target fact plus N distractor turns, half fact and half chat). Retrieval queries are natural questions. Free generation uses the answer-stripped prefix (`Nell's phone number is `), at most 16 new tokens, the old v22 stop (newline or sentence-ending punctuation), and alphanumeric containment. The prefix is there because a natural question makes this model start with a newline, and the v22 stop then scores every system as an empty string. Fact-in-context and bge RAG are "the chosen line + newline + that same prefix". bge retrieves with the natural question. The free-gen row is the untrained identity inject: the pre-train measurement after the key fix, before `llm_inject_smollm2_135m.pt` was installed. **n=32** sessions; K20 and K50 are 4 streams, K100 is 2 streams. Transformers 5 loads this checkpoint as bfloat16. On this machine a bfloat16 forward is about 1.7s and a float32 forward is about 0.03s. The centers were estimated on bfloat16 hidden states and reused for float32. A separate bfloat16 stream of 24 facts kept 24/24 slots, question ret@1 23/24, off-diagonal key cosine mean 0.092 and max 0.802.

默认 `overwrite_cos=0.9`。修复前每条会话收成 1 个槽（K 流的槽均值是 1，ret@1 正好是 1/K）。修复后槽数等于写入的事实数。

Default `overwrite_cos=0.9`. Before the fix every session collapsed to 1 slot (K-stream slot mean 1, ret@1 exactly 1/K). After the fix the slot count equals the number of facts written.

| | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| ret@1 before | 32/32 = 1.000 | 13/32 = 0.406 | 6/32 = 0.188 | 6/32 = 0.188 |
| ret@1 after | 32/32 = 1.000 | 32/32 = 1.000 | 32/32 = 1.000 | 32/32 = 1.000 |
| slots / facts before | 32/32 | 32/89 | 32/305 | 32/552 |
| slots / facts after | 32/32 | 89/89 | 305/305 | 552/552 |
| target still stored, before | 1.000 | 0.406 | 0.188 | 0.188 |
| target still stored, after | 1.000 | 1.000 | 1.000 | 1.000 |
| free-gen before | 0/32 | 0/32 | 0/32 | 0/32 |
| free-gen after (untrained inject, pre-train) | 0/32 | 0/32 | 0/32 | 0/32 |
| no memory | 0/32 | 0/32 | 0/32 | 0/32 |
| fact in context | 17/32 = 0.531 | 17/32 = 0.531 | 11/32 = 0.344 | 13/32 = 0.406 |
| bge RAG | 17/32 = 0.531 | 17/32 = 0.531 | 10/32 = 0.312 | 13/32 = 0.406 |
| bge hit | 32/32 | 32/32 | 31/32 | 32/32 |

事实进上下文的提示只有目标事实加前缀，不包含那 N 个干扰轮。N0/N4/N16/N32 之间的起伏是不同种子（`5000+N`）抽到的事实不一样，n=32，不是上下文变长。

The fact-in-context prompt is only the target fact plus the prefix. It does not include the N distractor turns. The swing across N0/N4/N16/N32 is different facts from seeds `5000+N` at n=32, not a longer context.

| | K20 (4×20 = 80) | K50 (4×50 = 200) | K100 (2×100 = 200) |
|---|---|---|---|
| ret@1 before | 4/80 = 0.050 | 4/200 = 0.020 | 2/200 = 0.010 |
| ret@1 after | 80/80 = 1.000 | 197/200 = 0.985 | 196/200 = 0.980 |
| mean slots before | 1 | 1 | 1 |
| mean slots after | 20 | 50 | 100 |

K50 和 K100 的槽是满的，那几次 ret@1 失手是排序，不是被覆盖掉。

K50 and K100 kept every slot. The ret@1 misses are ranking, not overwrites.

20 条不同事实的键，修复前非对角余弦均值 0.9999（最小 0.9997，最大 0.99997）。修复后均值 0.084（最小 −0.267，最大 0.829）。0.829 低于 0.9。两条 K=100 流都是 100/100 槽，所以这些流里没有一对事实跨过 0.9。

On 20 different facts, off-diagonal key cosine before the fix had mean 0.9999 (min 0.9997, max 0.99997). After the fix the mean is 0.084 (min −0.267, max 0.829). 0.829 is under 0.9. Both K=100 streams stored 100/100 slots, so no pair in those streams crossed 0.9.

改写同一条事实（另一套模板，答案同样剥掉）和自己的键：均值 0.546，最小 0.357。和别的事实最像的一对：最大 0.618，均值 0.385。最小的改写余弦低于最大的异事实余弦，所以一个余弦阈值分不开「同一事实的另一种说法」和「另一条事实」。同名同关系、只换答案时，两条键的余弦均值是 1.000（最小 1.000，n=32），因为答案不进键。覆盖靠的是 `(name, rel)` 身份，不是阈值。阈值碰巧也会换掉这对，因为余弦是 1。身份覆盖之后槽里是新值 32/32，旧值 0/32。训练前的恒等注入说不出新答案，也说不出旧答案（0/32 和 0/32）。把整段对话放进上下文时，新答案 7/32、旧答案 1/32。

A paraphrase (a second template, answer stripped) vs its own key has mean cosine 0.546 and min 0.357. The closest different fact has max 0.618 and mean 0.385. The worst paraphrase sits below the nearest different fact, so no cosine threshold separates "same fact, other wording" from "a different fact". A same-name, same-relation update that only changes the answer has key cosine mean 1.000 (min 1.000, n=32), because the answer is not in the key. The overwrite that is supposed to fire is the `(name, rel)` identity match, not the threshold. The threshold would also replace this pair, since the cosine is 1. After the identity overwrite the slot holds the new value in 32/32 sessions and the old value in 0/32. Free generation with the untrained inject, before the shipped MLP, emits neither the new answer nor the old one (0/32 and 0/32). With the whole transcript in the prompt, the new answer shows up in 7/32 and the old answer in 1/32.

**和 RAG 比（训练前）。** 这是装上训练权重之前的测量。N16 上没训练的注入是 0/32，bge RAG 是 10/32，事实进上下文是 11/32。N32 上注入仍是 0/32，RAG 和事实进上下文都是 13/32。检索在这几格是 32/32，所以不是键没取回来。恒等注入把答案跨度的隐状态加到每一位置（测过的一条答案池 L2 范数是 190），续写是标点（`,,,,,.'`、`..`），不是答案。它没有打平 RAG，也没有超过 RAG。旧研究里大约 0.59 的 N32 自由生成用的是训过的检索头和注入，不是这一次的未训练拷贝。现在的默认路径是下一节的训过注入，每一档 N 都高于 RAG。

**Against RAG (pre-train).** This is the measurement before the trained weights were installed. At N16 the untrained inject is 0/32, bge RAG is 10/32, and fact-in-context is 11/32. At N32 the inject is still 0/32, and RAG and fact-in-context are both 13/32. Retrieval on those cells is 32/32, so the key did come back. The identity inject adds the answer-span hidden state at every position (one measured answer pool had L2 norm 190), and the continuation is punctuation (`,,,,,.'`, `..`), not the answer. It does not match RAG and it does not beat RAG. The old study's roughly 0.59 N32 free-gen used a trained retrieval head and a trained inject, not this untrained copy. The default path now is the trained inject in the next section, and that run beats RAG at every N.

## 训过的注入 / Trained inject

上面自由生成全是 0 的那一行，是键修好之后、MLP 还没训的测量。默认 `LLMMemory()` 现在装的是训过的权重。

The free-gen zeros above are the measurement after the key fix and before this MLP was trained. The default `LLMMemory()` now loads the trained weights.

训练是冻结的 SmolLM2-135M、float32、CPU。事实是 `tidal.llm_memory_data` 的 train 切分，1500 条。prompt 是去掉答案的补全前缀，不是自然问句。1500 step，batch 8，lr 1e-3，neg_p 0.2。只有 inject MLP 在更新。墙钟 634.5 秒。checkpoint 写成 `save_inject` 的三个字段，放在 `tidal/llm_inject_smollm2_135m.pt`（layer 15，hidden 576）。没有在这份文件之外再训一次。

Training was a frozen SmolLM2-135M in float32 on CPU. The 1500 facts are the train split of `tidal.llm_memory_data`. The prompt is the answer-stripped completion prefix, not the natural question. 1500 steps, batch 8, lr 1e-3, neg_p 0.2. Only the inject MLP was updated. Wall time was 634.5 seconds. The checkpoint is the three `save_inject` fields in `tidal/llm_inject_smollm2_135m.pt` (layer 15, hidden 576). It was not trained again for this package.

协议和上一节的自由生成相同，样本改成 **n=64**。种子 `5000+N`。会话仍是 1 条目标事实加 N 个干扰轮，干扰轮里事实和闲聊各一半。检索问句是自然语言。自由生成用去掉答案的前缀（`Nell's phone number is `），最多 16 个新 token，v22 的停止规则（换行或句末标点），匹配是字母数字包含。事实进上下文和 bge RAG 都是「被选中的那一行 + 换行 + 同一个前缀」。bge 用自然问句检索。

The protocol is the same answer-stripped prefix protocol as the key-fix free-gen, with **n=64** sessions. Seeds are `5000+N`. A session is still one target fact plus N distractor turns, half fact and half chat. Retrieval queries are natural questions. Free generation uses the answer-stripped prefix (`Nell's phone number is `), at most 16 new tokens, the v22 stop (newline or sentence-ending punctuation), and alphanumeric containment. Fact-in-context and bge RAG are the chosen fact line plus a newline plus that same prefix. bge retrieves with the natural question.

| config | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| no memory | 0/64 = 0.000 | 0/64 = 0.000 | 0/64 = 0.000 | 0/64 = 0.000 |
| **trained inject** | 48/64 = **0.750** | 46/64 = **0.719** | 34/64 = **0.531** | 42/64 = **0.656** |
| fact in context | 29/64 = 0.453 | 32/64 = 0.500 | 25/64 = 0.391 | 30/64 = 0.469 |
| bge RAG | 29/64 = 0.453 | 32/64 = 0.500 | 23/64 = 0.359 | 30/64 = 0.469 |

每一档 N 上，训过的注入都高于事实进上下文，也高于 bge RAG。事实进上下文的提示只有目标事实加前缀，不含那 N 个干扰轮，所以 N 之间的起伏主要是种子不同抽到的事实不同。这一节的读出是当时的 soft top-4、温度 0.05。下面的消融在同一套种子的前 48 条上重测了前缀：top-1 和 soft top-4 打平，所以这张表里 N16 的 0.531 不是那个混合造成的。

Trained inject beats fact-in-context and bge RAG at every N in this table. The fact-in-context prompt is only the target fact plus the prefix, so the swing across N is mostly different facts from seeds `5000+N`, not a longer context. The readout in this table is the soft top-4 at temperature 0.05 that was the path at the time. The ablation below reruns that prefix on the first 48 sessions of the same seeds: top-1 ties soft top-4, so the 0.531 at N16 in this table is not that mix.

同一批会话的 ret@1 是 64/64 = **1.000**（N0、N4、N16、N32 都是）。同名更新 n=48、种子 42：训过的注入给出新答案 **0.604**、旧答案 **0.000**。整段对话放进上下文时，新答案 0.188，旧答案 0.062。

On those same sessions ret@1 is 64/64 = **1.000** at N0, N4, N16, and N32. A same-name update (n=48, seed 42) recalls the new answer in **0.604** of sessions and the old answer in **0.000**. With the whole transcript in the prompt, the new answer shows up in 0.188 of sessions and the old answer in 0.062.

## 自然问句的自由生成 / Natural-question free generation

前缀协议是绕开空串的办法，不是这条路径现在的生成方式。`LLMMemory.generate` 对贪心续写先丢掉开头的空白（包括换行），再按原来的 v22 规则停：下一个换行，或者在已经有一个以上非空字符时遇到句末标点（`.` `?` `!`）。SmolLM2-135M 对自然问句的无记忆续写以换行开头，旧停规则会把事实进上下文和 bge RAG 都打成空串。训过的注入本身通常不以换行开头，它以空格加答案开头，所以两种停规则在注入上分数一样。

The prefix protocol was a way around the empty string. It is not how this path generates now. `LLMMemory.generate` drops leading whitespace, including newlines, and then applies the old v22 stop: the next newline, or sentence-ending punctuation (`.` `?` `!`) once more than one non-space character has been kept. With no memory, SmolLM2-135M starts a natural question with a newline, and the old stop scores fact-in-context and bge RAG as empty. The trained inject usually does not start with a newline. It starts with a space and the answer, so the two stops score the inject the same.

下面是 `scripts/eval_llm_question.py`，冻结 **SmolLM2-135M**、**float32**、CPU，**n=48**，种子 `5000+N`，和上一节是同一条会话序列的前 48 条。一条问句和一条「事实进上下文 + 前缀」上，KV cache 的 12 个贪心 token 和不用 cache 的主循环一致，之后这次数用了 cache。墙钟 999 秒。问句是自然语言（`What is Nell's phone number?`）。自由生成最多 16 个新 token。匹配是字母数字包含。注入读出是 hard top-1，也就是现在的主路径。事实进上下文和 bge RAG 都是「被选中的那一行 + 换行 + 同一个问句」，空库。bge 用这个问句检索。`span` 是 `generate` 的停规则。`v22` 是不跳过开头换行的旧停规则，列在这里是为了说明空串从哪来。同一批 token 上，只切第一行、或留下全部 16 个 token，问句的命中数和 `span` 相同，所以没有改成更松的停规则。

Numbers below are from `scripts/eval_llm_question.py` on frozen **SmolLM2-135M** in **float32** on CPU, **n=48**, seeds `5000+N`, the first 48 sessions of the same sequence as the prefix table. A KV cache matched the uncached greedy ids on one question and one fact-in-context prefix (12 tokens) and was then used for the run. Wall time was 999 seconds. The prompt is the natural question (`What is Nell's phone number?`). Free generation is at most 16 new tokens. Match is alphanumeric containment. The inject readout is hard top-1, which is the main path now. Fact-in-context and bge RAG are the chosen line, a newline, and that same question, with an empty store. bge retrieves with the question. `span` is the stop `generate` uses. `v22` is the old stop that does not skip a leading newline; it is here to show where the empty string came from. On these tokens, cutting at the first line only, or keeping all 16 tokens, matched `span` on every question cell, so the stop was not loosened.

| config | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| no memory | 0/48 = 0.000 | 0/48 = 0.000 | 0/48 = 0.000 | 0/48 = 0.000 |
| **trained inject, top-1, span** | 37/48 = **0.771** | 35/48 = **0.729** | 31/48 = **0.646** | 36/48 = **0.750** |
| fact in context, span | 16/48 = 0.333 | 15/48 = 0.312 | 13/48 = 0.271 | 14/48 = 0.292 |
| bge RAG, span | 16/48 = 0.333 | 15/48 = 0.312 | 13/48 = 0.271 | 14/48 = 0.292 |
| fact in context and bge RAG, v22 | 0/48 = 0.000 | 0/48 = 0.000 | 0/48 = 0.000 | 0/48 = 0.000 |

问句 ret@1 四档都是 48/48。bge 命中是 48/48、48/48、46/48、48/48。注入的问句自由生成每一档都高于事实进上下文，也高于 bge RAG。v22 把两个基线打成 0，是开头换行，不是基线没说出答案。`span` 把那一行留下来。无记忆在两种停规则下都是 0。

注入自己的空串不是这个换行。四档里电话一共 28 条，全部空：第一个贪心 token 就是 eos（id 0），`ids` 是空的，放到 64 个 token 仍是空。生日 25 条里中了 1 条，其余是差了几天或一年的日期，整段字母数字对不上。城市、颜色、队伍全中。职业 34 条里中了 33 条。这四类是 138/139。四种读出的空串条数相同，所以这不是 soft top-k 的事。

Question ret@1 is 48/48 at every N. bge hits are 48/48, 48/48, 46/48, and 48/48. Trained-inject question free-gen beats fact-in-context and bge RAG at every N. v22 scores both baselines at 0 because of the leading newline, not because those baselines never said the answer. `span` keeps that line. No-memory is 0 under both stops.

The inject's own empty strings are not that newline. All 28 phone questions are empty: the first greedy token is eos (id 0), so `ids` is empty, and 64 tokens is still empty. Birthdays hit 1 of 25; the rest are off by a few days or a year, and the full alphanumeric date does not match. City, color, and team all hit. Job hits 33 of 34. Those four relations are 138/139. Every readout had the same empty counts, so this is not the soft top-k mix.

## soft top-k 消融 / Soft top-k ablation

同一批 n=48 会话。注入自由生成用 `span`。ret@1 是 argmax，不随混合方式变化，四档问句和前缀都是 48/48。有两个以上槽时，top-1 和 top-2 的余弦差全部大于 0.05（N4 / N16 / N32 问句均值 0.442 / 0.261 / 0.186）。所以 margin 0.05（差距大于 0.05 用 top-1，否则 soft top-4、温度 0.05）在这张表里和 hard top-1 是同一条路径。

Same n=48 sessions. Inject free-gen uses `span`. ret@1 is argmax, so it does not depend on the mix: 48/48 on questions and on prefixes at every N. Whenever a session had two or more slots, the top-1 cosine beat top-2 by more than 0.05 (question means 0.442 / 0.261 / 0.186 at N4 / N16 / N32). A margin gate at 0.05 (top-1 when the gap is larger, otherwise soft top-4 at temperature 0.05) is therefore the same path as hard top-1 in this table.

问句自由生成：

Question free-gen:

| readout | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| soft top-4, T=0.05 | 37/48 = 0.771 | 35/48 = 0.729 | 30/48 = 0.625 | 34/48 = 0.708 |
| **hard top-1** | 37/48 = 0.771 | 35/48 = 0.729 | 31/48 = 0.646 | 36/48 = **0.750** |
| soft top-4, T=0.20 | 37/48 = 0.771 | 32/48 = 0.667 | 23/48 = 0.479 | 19/48 = 0.396 |
| soft top-2, T=0.05 | 37/48 = 0.771 | 35/48 = 0.729 | 30/48 = 0.625 | 34/48 = 0.708 |
| margin 0.05, else soft top-4 T=0.05 | 37/48 = 0.771 | 35/48 = 0.729 | 31/48 = 0.646 | 36/48 = 0.750 |

前缀自由生成（`Nell's phone number is `）。soft top-4、温度 0.05 和 hard top-1 四档都相同。温度 0.20 在 N32 掉到 27/48。

Prefix free-gen (`Nell's phone number is `). Soft top-4 at temperature 0.05 and hard top-1 agree at every N. Temperature 0.20 falls to 27/48 at N32.

| readout | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| soft top-4 T=0.05 and hard top-1 | 36/48 = 0.750 | 35/48 = 0.729 | 29/48 = 0.604 | 34/48 = 0.708 |
| soft top-4, T=0.20 | 36/48 = 0.750 | 35/48 = 0.729 | 30/48 = 0.625 | 27/48 = 0.562 |
| fact in context | 21/48 = 0.438 | 25/48 = 0.521 | 19/48 = 0.396 | 20/48 = 0.417 |
| bge RAG | 21/48 = 0.438 | 25/48 = 0.521 | 17/48 = 0.354 | 20/48 = 0.417 |

主路径是 **hard top-1**。问句上它每一档都不低于 soft top-4、温度 0.05，N32 是 36/48 对 34/48。前缀上两者打平。温度 0.20 会把邻近槽混进注入：N32 问句从 36/48 掉到 19/48。soft top-2、温度 0.05 和 soft top-4 相同，第三名和第四名几乎没有权重。margin 0.05 没有一次落到混合上。

N16 前缀从 N0 的 36/48 掉到 29/48，top-1 和 soft top-4 是同一个数字。top-1 命中时注入的就是目标槽的值，提示里也没有干扰轮，所以这 7 条之差是种子 `5000+N` 抽到的事实不同，不是混合。soft top-4、温度 0.05 的问句纯度（读出和目标值的余弦）是 1、0.9999、0.9996、0.9992，没有任何一条低于 0.90。温度 0.20 的 N32 问句纯度是 0.979，自由生成已经明显变差。

The main path is **hard top-1**. On questions it is at least soft top-4 at temperature 0.05 in every cell, and at N32 it is 36/48 against 34/48. On the prefix the two agree. Temperature 0.20 mixes neighboring slots into the inject and drops N32 questions from 36/48 to 19/48. Soft top-2 at temperature 0.05 matches soft top-4; the third and fourth neighbors get almost no weight. The margin gate at 0.05 never took the mixture.

The prefix drop from 36/48 at N0 to 29/48 at N16 is the same number for top-1 and for soft top-4. When top-1 hits, the injected vector is that slot's value and the prompt does not contain the distractors, so those 7 sessions are different facts from seeds `5000+N`, not a mix. Question purity of soft top-4 at temperature 0.05 (cosine of the readout with the target value) is 1, 0.9999, 0.9996, and 0.9992, and no session is under 0.90. At temperature 0.20 the N32 question purity is 0.979 and free-gen is already clearly worse.

连续体 `heads` 没有改。它的读出仍是 soft top-4、温度 0.05。

Continuum `heads` is unchanged. Its readout is still soft top-4 at temperature 0.05.
