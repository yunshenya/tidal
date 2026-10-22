# 情景事实记忆 / Episodic fact memory

连续体（`tidal/continuum.py`）在影子回放里始终带着一块情景事实槽（`tidal/episodic.py`）。它是 `heads` 的一部分，空库读出是零，不加载 135M 权重。冻结因果语言模型的测试时注入在 `tidal/llm_memory.py`（`pip install tidal[llm]`），设计和当年的 SmolLM 数字在 [llm_memory.md](llm_memory.md)。那条路径不进 `heads`。

Continuum's shadow replay always owns an episodic slot store. Readout sits on the `heads` path and does not load a 135M checkpoint. Test-time residual inject for a frozen causal LM is `tidal/llm_memory.py` (`pip install tidal[llm]`). The design and the old SmolLM numbers are in [llm_memory.md](llm_memory.md). That path is not added into `heads`.

## 行为 / Behavior

`Continuum` 一构造就有一个 `EpisodicStore`。每个头之前：

```
h_adapt = h + Σ_levels adapter(h)
h' = h_adapt + soft_topk(h_adapt)
```

空库时 `soft_topk` 是全零。连续体那层适配器也还是零的时候（刚构造，或 `levels=()`），回放就和静态模型一样。没有 `enable_episodic`，也没有把读出乘成 0 的比例开关。调用方如果仍传入这类键，读出照样加上。`heads` / `replay` / `recall` 的排序是原始余弦，分母是 `kn*qn + 1e-12`。

问句走另一条路，也没有开关。`ask` / `query` 用对比 q/k 头给槽打分，再走同一套 soft top-k 和字节码解码。这个头不进 `heads`，replay 也不调用它。编码器对不上随包权重时，头的末层仍是零，残差是 0，直到 `fit_question_head(..., install=True)` 把对该编码器的拟合装上。那不是一条另写的余弦分支。

An empty store adds zeros. With zero continuum adapters (a fresh model, or `levels=()`), replay matches the static model. `heads`, `replay`, and `recall` rank by raw cosine (`kn*qn + 1e-12`). There is no `enable_episodic` and no scale that multiplies the readout by zero.

Questions use the other path, also with no switch. `ask` / `query` score slots with the contrastive query/key head, then the same soft top-k and byte-code decode. That head is not added into `heads`, and replay does not call it. When the encoder hash does not match the shipped weights, the head's last layer is still zero, so the residual is 0, until `fit_question_head(..., install=True)` copies a fit for that encoder. That is not a second cosine implementation.

一条事实是一个槽 `(k, v, name, rel, group)`。

- 同一个 `(group, name, rel)`（`name` 非空）直接覆盖。新的键和值换上，旧键不会被补回去。
- 身份对不上、但 `cos(k, k_i) > 0.9` 且在同一 `group` 里，当成同一事实的改写，也覆盖。身份匹配优先于这条近重复规则。
- 容量默认 256。满了淘汰最旧的写入时间（FIFO）。覆盖会刷新时间，刚更新的事实不会被当成旧槽丢掉。
- 读出是 soft top-k：余弦最高的 4 个槽，温度 0.05 做 softmax，对 **value** 做凸组合（权重和为 1）。查询某个 `group` 时，能看见该群的槽和 `group is None` 的全局槽，看不见别的群。
- 精确回忆读**赢的那个槽**里的码（`recall` / `recall_tokens`），不读混合残差。
- 写门控丢掉 `kind="chat"` 和 `kind="noise"`，也丢掉没有方向的键。`surprise_thr` 只有在配置了阈值、并且这次写入带了 `surprise` 时才起作用；不传 surprise 不会挡事实。`replay` 不把原始事件流写成事实。
- 电话、生日、多 token 字符串写成**位置字节码**（见下节），不是词向量的平均，也不做 L2 归一化。同身份覆盖换成整段新码。

键和值用 tidal 已有的通路，宽度就是编码器状态 `d`：

- `embed_event(feats)`：冻结模型的 `fproj` + 主干一步。`feats` 是 `features.py` / `features_g.py` 的一行（模型若带情绪列，把那几列算进去）。
- `embed_event(feats, text_emb)`：`text_emb` 是 `tidal.embed` / `TextFrontEnd` 的 512 维 bge。只有模型自己带 `tproj` 时才走这条文本投影。特征行若大半能量都在常数方向上（各维差不多大），`fproj` 加偏置就是一个共享向量，比单位句向量的 `tproj` 大好几倍，原始余弦会把不同句子收成一条事实。这时丢掉该常数和特征偏置，并加上文本向量的固定符号码，幅度取剩余特征纹波和 `tproj` 里较大的那个。有对比度的稀疏行（问句头用的名字/关系码）不在这个区间，编码器状态仍是原来的 `fproj + tproj`。没有开关。
- `remember(k, v, ...)` 的数值向量必须已经在 `d` 里。512 维 bge 不能直接当槽。字符串或 token 列表会编成宽度 `d` 的位置字节码再写入。

An empty store adds zeros. With zero adapters (a fresh model, or `levels=()`), replay matches the static model. Same-identity overwrite replaces the whole slot. Soft top-4 (temperature 0.05) is a convex combination of values. Chat and noise writes are dropped. Numeric keys and values are encoder states from the frozen event model; a 512-d bge vector goes through that model's `tproj` when it exists. A phone, date, or token list is a positional byte code of the same width.

`embed_event(feats, text)` leaves a feature row alone when the constant direction holds at most half its energy. That is the question head's sparse name/relation code, so those encoder states stay `fproj + tproj`. A flat row (every coordinate the same size) is the other regime: `fproj` plus its bias is one shared vector, several times `tproj` of a unit sentence, and raw cosine then overwrites unrelated sentences at 0.9. The flat row drops that offset and the bias, and adds a fixed sign code of the 512-d text vector, scaled to the larger of the leftover feature ripple and `tproj`. The code is not a learned map and it is not in the question-head fingerprint. There is no switch.

## 难值 / Hard values

均值池化留不住数字顺序：`555-0134` 和它的逆序字节袋是同一个向量。把新码加到旧码上会留下旧数字。L2 归一化会把字节码和 copy-through 要用的幅度一起毁掉。这三条都不做。

`remember(k, "+1-415-555-0134", ...)` 或 `remember(k, ["cedar", "lantern"], ...)` 把值写成位置字节码（每 lane 两个字节，`k/2**16`，float32 里也精确，不归一化）。宽度 `d` 能放 `2*(d-4)` 字节（含 1 字节类型标记）；平时 `d=128` 是 248 字节。同身份覆盖替换整段码，槽里读不出旧号码或旧日期。

读出仍是两条常开的路，没有开关：

- 加到 `h` 上的是 soft top-k 凸组合。空库仍是精确的零。间隙够大时，这个残差四舍五入后还是赢的那个码。
- 精确回忆读**赢的那个槽**里的码（`recall` / `recall_tokens`），不读混合残差。近邻把残差拉离字节网格时，混合结果对不上校验，不会变成另一个号码；槽里仍是上一次写入。

评测是精确匹配（`python -m tidal.episodic_eval`，pytest 会跑）。用例数大于原先 n≈96 的自由生成抽样：覆盖、soft top-k、群隔离、写门控、因果顺序、电话 / 生日 / 多 token 都必须逐条对上。问句一节另报 ret@1 的整数（余弦 13/48，头 47/48），并逐条核对 `ask` 跟头的 argmax、`recall` 跟余弦，以及 replay 在问句权重被改乱时逐位不变。0.02 那种幅度在 96 条里分不出来，这里不估一个率。

Mean-pooling drops digit order, adding the new code onto the old one leaves stale digits, and L2-normalizing destroys the code. A string or token list is stored as a positional byte code in the value vector (`2*(d-4)` bytes, 248 at `d=128`). Same-identity overwrite replaces that code. Heads still add the soft top-k convex combination; exact recall decodes the winning slot. A blended residual fails its checksum instead of becoming a different phone. The exact-match harness is larger than the old n≈96 free-gen sample and is not a rate estimate. Its question section reports integer ret@1 (cosine 13/48, head 47/48) and checks that `ask` follows the head, `recall` follows cosine, and replay is bitwise unchanged when the question weights are scrambled.

## 从 neuromem 留下什么 / What was kept

neuromem 在冻结的 SmolLM2-135M 上试过好几轮。tidal 只留下不依赖那套权重的部分。

| 做法 | 在 tidal 里 |
|---|---|
| 离散槽 `(k, v, identity)`，同身份覆盖 | 保留。更新换掉整槽，不把新旧值加在一起。 |
| soft top-4，温度 0.05 | 保留。近邻混合，一次 top-1 检错不至于整段读错。 |
| 写门控跳过闲聊 | 保留。未标注的事件流不是事实，`replay` 不写槽。 |
| 多块矩阵读出再求和 | 不采用。求和会把检索到的值稀释掉。这里只有一块槽表。 |
| 朴素 key shield（写完按旧键把旧读出补回去） | 不采用。同名更新会写不进去。覆盖后槽里的键就是新键。 |
| 对比学习 q/k MLP | 只在 `ask` / `query`。`heads` / `recall` / replay 仍是原始余弦。问句 ret@1 从 13/48 到 47/48。同一头若用来查「存进去的键、干扰是其他事实」会从 48/48 掉到 18/48，所以 `recall` 不接它。 |
| 残差 inject MLP | 不采用。会把电话字节码推离网格。 |
| SmolLM2-135M、`transformers` | 核心 `import tidal` 不 import。可选模块 `tidal/llm_memory.py`，`pip install tidal[llm]`。 |
| 答案跨度做均值池化，或写入时 L2 归一化 | 不采用。池化分不出顺序；归一化毁掉字节码。字符串走位置字节码。 |

## 问句头和余弦路径 / Question head and the cosine path

对比 q/k 曾经直接接在 `read` 上，末层零初始化。问句检索会涨，但「查询 = 存进去的编码器键、干扰是其他事实」从 48/48 掉到 18/48，所以不能替换 `heads` / `recall`。残差 inject 另训过 300 步，把 `+1-415-555-0134` 推离字节网格（最大绝对偏差 0.00111271），解不出来。所以这条字节码路径上没有 inject MLP。冻结 LM 的 inject 在 `tidal/llm_memory.py`，注入的是隐状态，不读这些字节码。

现在 q/k 只服务 `ask` / `query`。同一套协议又量了一次：seed 0，`d=32`，冻结的 `fproj` + `tproj` + 一步 GRU，400 步，InfoNCE τ=0.07，难负例。问句向量用的是反转后的关系 token，不是存进去的键。这不是 SmolLM 的 n≈96 自由生成。脚本是 `python -m tidal.question_head`，权重在 `tidal/question_qk.npz`（合成编码器和合成事实，不是在真实聊天上训的 phase-1 权重）。

| 集合 | 原始余弦 ret@1 | 问句头 |
|---|---|---|
| 问句 ≠ 键，8 个事实干扰 | 13/48 = 0.2708 | 47/48 = 0.9792（`ask` 用这个） |
| episodic_eval 那种：查询 = 存进去的键，8 个随机单位向量 | 48/48 = 1（`recall` 用这个） | 若误用头仍是 48/48 = 1 |
| 随机单位键，查询 = 键，8 个单位干扰 | 256/256 = 1 | 若误用头仍是 256/256 = 1 |
| 查询 = 存进去的编码器键，干扰是另外 8 条事实的键 | 48/48 = 1（`recall`） | 若把 `ask` 用在这条上是 18/48 = 0.3750 |

`recall` 的分数和一份独立的余弦实现逐位相同。把问句头的权重改乱，replay 的输出逐位不变，空库时 `heads` 仍和静态模型逐位相同。未训练的末层是零，问句分数和原始余弦逐位相同；随包的拟合装上之后，问句集合上两者会分开（评测里至少有一条余弦选错、头选对，也至少有一条存进去的键上余弦选对、头选错）。

旧 SmolLM 研究（[llm_memory.md](llm_memory.md)）里的 K100 ret@1 0.451→0.590、N0 生成 0.594→0.656 是另一个模型，不能当成这里的结果。

The contrastive maps used to sit on `read`. Question retrieval rose, and recall of the stored encoder key among other facts fell from 48/48 to 18/48, so they do not replace `heads` / `recall`. A residual inject fit (300 steps) moved `+1-415-555-0134` by max abs 0.00111271 and the code no longer decoded. This byte-code path has no inject MLP. Frozen-LM inject lives in `tidal/llm_memory.py` and adds hidden states, not these codes.

The query/key maps now serve only `ask` / `query`. The same protocol was measured again: seed 0, `d=32`, frozen `fproj` + `tproj` + one GRU step, 400 steps, InfoNCE τ=0.07, hard negatives. The question state uses the reversed relation token, not the stored key. This is not SmolLM's n≈96 free-gen sample. `python -m tidal.question_head` writes `tidal/question_qk.npz` (a synthetic encoder and synthetic facts, not phase-1 weights trained on real chat).

| set | raw cosine ret@1 | question head |
|---|---|---|
| question ≠ key, 8 fact distractors | 13/48 = 0.2708 | 47/48 = 0.9792 (`ask` uses this) |
| episodic_eval style: query = stored key, 8 random unit vectors | 48/48 = 1 (`recall` uses this) | 48/48 = 1 if the head were applied |
| random unit keys, query = key, 8 unit distractors | 256/256 = 1 | 256/256 = 1 if the head were applied |
| query = stored encoder key, distractors are 8 other fact keys | 48/48 = 1 (`recall`) | 18/48 = 0.3750 if `ask` were used on that key |

`recall` scores match an independent cosine bitwise. Scrambling the question weights leaves replay outputs bitwise unchanged, and an empty store still matches the static model bitwise on `heads`. A zero last layer makes question scores bitwise-equal to raw cosine; after the shipped fit is installed, the two ranks diverge on the question set (the harness has at least one cosine miss that the head hits, and at least one stored key that cosine hits and the head misses). SmolLM's K100 and free-gen numbers are in [llm_memory.md](llm_memory.md) and are not these figures.

## 影子策略 / Shadow policy

连续体仍然只回放已记录的流、报告指标，不驱动线上动作。情景槽不进 `DuplexController`，不改 tick。`ask` 是读事实的调用，不是 tick 输入。按群写入的事实只影响该群的读出（加上全局槽），所以一条群的事实不会改另一条群的影子分数。

Continuum stays shadow-only. Episodic readout is inside that replay path. `ask` reads facts; it is not a live tick input.

## 用法 / Usage

```python
from tidal.continuum import Continuum, _cfg
from tidal.vap import VAPModel

model = VAPModel(n_feat).eval()
cont = Continuum(model, _cfg(dict(levels=("med",))))
k = cont.embed_event(key_feats)                 # feature row -> encoder state
v = cont.embed_event(value_feats, bge)          # bge only if model.tproj exists
cont.remember(k, v, name="ada", rel="city", group=group_id)
k_phone = cont.embed_event(phone_key_feats)     # a different key; cos > 0.9 would overwrite
cont.remember(k_phone, "+1-415-555-0134", name="ada", rel="phone", group=group_id)
cont.recall(k_phone)                            # raw cosine, winning slot's string
q = cont.embed_event(question_feats, question_bge)
cont.ask(q)                                     # question head, then the winning slot's byte code
cont.query(q)                                   # same rank, soft top-k value; not added to heads
cont.ask_tokens(q)
```

随包的拟合对应 `tidal.question_head.load_reference()` 那只编码器（seed 0，`d=32`，`fproj` / `tproj` / 一步 GRU）。别的冻结编码器调用 `fit_question_head(cont, install=True)`，只把映射装进 `ask` / `query`。`episodic_cfg` 只调容量、`soft_k`、温度、覆盖余弦和可选的 surprise 阈值。它不是开关。

The shipped fit is the encoder from `tidal.question_head.load_reference()` (seed 0, `d=32`, `fproj` / `tproj` / one GRU step). `fit_question_head(cont, install=True)` fits any other frozen encoder and installs the maps on `ask` / `query` only. `episodic_cfg` tunes capacity, `soft_k`, temperature, overwrite cosine, and the optional surprise threshold. It is not a switch.
