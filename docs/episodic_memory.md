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
- `embed_event(feats, text_emb)`：`text_emb` 是 `tidal.embed` / `TextFrontEnd` 的 512 维 bge。只有模型自己带 `tproj` 时才走这条文本投影。
- `remember(k, v, ...)` 的数值向量必须已经在 `d` 里。512 维 bge 不能直接当槽。字符串或 token 列表会编成宽度 `d` 的位置字节码再写入。

An empty store adds zeros. With zero adapters (a fresh model, or `levels=()`), replay matches the static model. Same-identity overwrite replaces the whole slot. Soft top-4 (temperature 0.05) is a convex combination of values. Chat and noise writes are dropped. Numeric keys and values are encoder states from the frozen event model; a 512-d bge vector goes through that model's `tproj` when it exists. A phone, date, or token list is a positional byte code of the same width.

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

旧 SmolLM 研究（[llm_memory.md](llm_memory.md)）里的 K100 ret@1 0.451→0.590、N0 生成 0.594→0.656 是另一个模型，不能当成这里的结果。上表这 48 条是反转关系 token 的合成几何。中英表面问句在下一节。

The contrastive maps used to sit on `read`. Question retrieval rose, and recall of the stored encoder key among other facts fell from 48/48 to 18/48, so they do not replace `heads` / `recall`. A residual inject fit (300 steps) moved `+1-415-555-0134` by max abs 0.00111271 and the code no longer decoded. This byte-code path has no inject MLP. Frozen-LM inject lives in `tidal/llm_memory.py` and adds hidden states, not these codes.

The query/key maps now serve only `ask` / `query`. The same protocol was measured again: seed 0, `d=32`, frozen `fproj` + `tproj` + one GRU step, 400 steps, InfoNCE τ=0.07, hard negatives. The question state uses the reversed relation token, not the stored key. This is not SmolLM's n≈96 free-gen sample. `python -m tidal.question_head` writes `tidal/question_qk.npz` (a synthetic encoder and synthetic facts, not phase-1 weights trained on real chat).

| set | raw cosine ret@1 | question head |
|---|---|---|
| question ≠ key, 8 fact distractors | 13/48 = 0.2708 | 47/48 = 0.9792 (`ask` uses this) |
| episodic_eval style: query = stored key, 8 random unit vectors | 48/48 = 1 (`recall` uses this) | 48/48 = 1 if the head were applied |
| random unit keys, query = key, 8 unit distractors | 256/256 = 1 | 256/256 = 1 if the head were applied |
| query = stored encoder key, distractors are 8 other fact keys | 48/48 = 1 (`recall`) | 18/48 = 0.3750 if `ask` were used on that key |

`recall` scores match an independent cosine bitwise. Scrambling the question weights leaves replay outputs bitwise unchanged, and an empty store still matches the static model bitwise on `heads`. A zero last layer makes question scores bitwise-equal to raw cosine; after the shipped fit is installed, the two ranks diverge on the question set (the harness has at least one cosine miss that the head hits, and at least one stored key that cosine hits and the head misses). SmolLM's K100 and free-gen numbers are in [llm_memory.md](llm_memory.md) and are a different model. The 48 rows above are the reversed-relation synthetic geometry. Chinese and English surface questions are the next section.

## 表面问句留出评测 / Held-out surface questions

上一节的 13/48 → 47/48 来自 `build_facts`：问句状态是 `name + reversed(rel_k)`，键是 `name + rel_k + 0.3*value`，特征行是 `_feats`，没有中文或英文句子。这一节量 `ask` 在留出的表面问句上的精确解码。语料在看到 ret@1 之前写定，没有按命中率改措辞，也没有重训 `tidal/question_qk.npz`。

`python -m tidal.episodic_eval` 会打印这一节，`python -m tidal.question_heldout` 只跑它。ret@1 是测量。空库 `heads` 等于静态模型、空库 `ask` / `recall` 为 `None`、事实写入成功、闲聊写入被拒、`overwrite_cos=1` 时身份集合等于事实写入的 `(name, rel)`、金标在槽里、同身份更新后旧值不在槽里（0.9 的影子库同样检查）、`ask` / `recall` 等于 `question_scores` / `cosine_scores` 的 top-1 解码、打乱问句权重后 replay 与 `heads` 逐位不变：这些失败即失败。pytest 不因 ret@1 偏低而失败。

The 13/48 → 47/48 figure is `build_facts`: the question state is `name + reversed(rel_k)`, the key is `name + rel_k + 0.3*value`, the feature row is `_feats`, and there is no Chinese or English sentence. This section measures exact decode of `ask` on held-out surface questions. The corpus was frozen before ret@1 was read. Phrasings were not rewritten to chase hits, and `tidal/question_qk.npz` was not refit.

`python -m tidal.episodic_eval` prints the section. `python -m tidal.question_heldout` runs only this probe. ret@1 is a measurement. These still fail the process: empty `heads` equal the static model, empty `ask` / `recall` are `None`, fact writes succeed, chat writes are rejected, at `overwrite_cos=1` the identity set equals the fact writes' `(name, rel)`, the gold value is stored, a same-identity update leaves the old value out of the store (the 0.9 shadow store is checked the same way), `ask` / `recall` match top-1 decode of `question_scores` / `cosine_scores`, and scrambling the question weights leaves replay and `heads` bitwise unchanged. pytest stays green when ret@1 is mediocre.

### 分类 / Categories

十二类，442 条：可答 370，无答案 72。值按 `(语言, 人, 关系, 版本)` 唯一。电话是 `+44-…`（训练电话是 `+1-…-555-…`）。日期的年份从 1920 加序号。地点是混合进制抽出的三个 token（英文稀有词或中文地名），校验器拒绝它们出现在该条问句或陈述里。干扰环固定，不按余弦挑选。写入的 `group` 都是 `None`。问句模板按人下标轮换，同一类里 12 个人不共用一句。

Twelve categories, 442 cases: 370 answerable, 72 unanswerable. Values are unique per `(language, person, relation, version)`. Phones are `+44-…` (training phones are `+1-…-555-…`). Date years start at 1920 plus the serial. Places are an injective mixed-radix triple of rare English words or Chinese place names; the validator rejects a case whose gold token appears in that question or statement. The distractor ring is fixed and is not chosen by cosine. Every write uses `group=None`. Question templates rotate with the person index, so the twelve people in a category do not share one sentence.

| 类别 / category | n | 写入顺序 / write order | 问的是什么 / what is asked |
|---|---:|---|---|
| `en_direct` / `zh_direct` | 36 / 36 | 4 条其他 `(人, 关系)` 的 primary 事实，然后金标（version 0） | 直接问电话、生日或地点。12 人 × 3 关系。 |
| `en_indirect` / `zh_indirect` | 40 / 40 | 同上：4 条干扰，然后金标 | 改写和间接问法。36 条点名，外加每种语言 4 条第二人称。 |
| `en_cross_turn` / `zh_cross_turn` | 37 / 37 | 金标先写（`cross` 家族，version 0），然后 5 条其他事实 | 跨轮指代。36 条点名，外加每种语言 1 条不点名。 |
| `en_after_distractors` / `zh_after_distractors` | 36 / 36 | 8 条其他事实，3 条 `kind="chat"`（不入库），金标，再 2 条其他事实 | 中间夹了别的事实和闲聊之后，仍问当前金标。 |
| `en_after_update` / `zh_after_update` | 36 / 36 | 4 条其他事实，同身份旧值（version 1，`update_old`），再同身份新值（version 2，`update_new`） | 更新之后该用哪个值。回答等于旧值记为 stale。 |
| `en_unanswerable` / `zh_unanswerable` | 36 / 36 | 问到的 `(人, 关系)` 没有写入 | 事实从未存储。`ask` 在非空库上总是返回 top-1 槽。 |

直接问法是 “What is Ada's phone number?” / “林晓的电话号码是多少？” 这一类。间接问法不说 “phone number” 或 “电话号码” 也能指到电话，例如 “How do I reach Ada?” / “怎么联系林晓？”。每种语言另有 4 条第二人称，身份仍是 `people[0]`（Ada / 林晓），陈述用第一人称，干扰跳过这个 `(人, 关系)`：

- “How do I reach you?”
- “When's your birthday?” 与 “What day were you born?”（同一条生日陈述）
- “Which spots should I use when I come to you?”
- “我怎么联系你？”
- “你的生日是哪一天？” 与 “你是哪一天出生的？”
- “我去找你的话该在哪些地点碰面？”

跨轮问法指向更早的一轮，例如 “That number Ada mentioned earlier, what was it?” / “林晓前面提到的那个号码是多少？”。不点名的那两条是 “that number you mentioned earlier” 和 “你刚才说的那个号码是多少？”，也是金标先写、再写 5 条干扰。subtype `second_person` 8 条、`anaphora_unnamed` 2 条。

无答案拆成两个 subtype，每种语言 18 + 18：

- `missing_rel`：`people[0..5]`。这个人的另外两种关系已经写入，问的那种没有。再补其他人的事实，写到 6 条为止。
- `absent_person`：缺席名字（Clara、Victor、Edith、Hassan、Naomi、Felix / 马岚、宋北、冯棠、韩澈、邓星、曹野）。库里只有 6 条在场的人的事实。

Direct asks look like “What is Ada's phone number?” / “林晓的电话号码是多少？”. Indirect asks can name the phone without the words “phone number” / “电话号码”, as in “How do I reach Ada?” / “怎么联系林晓？”. Each language adds four second-person asks. The identity is still `people[0]` (Ada / 林晓), the stored utterance is first person, and the distractors skip that `(person, relation)`:

- “How do I reach you?”
- “When's your birthday?” and “What day were you born?” (one birthday statement)
- “Which spots should I use when I come to you?”
- “我怎么联系你？”
- “你的生日是哪一天？” and “你是哪一天出生的？”
- “我去找你的话该在哪些地点碰面？”

Cross-turn asks point at an earlier turn, as in “That number Ada mentioned earlier, what was it?” / “林晓前面提到的那个号码是多少？”. The two unnamed lines are “that number you mentioned earlier” and “你刚才说的那个号码是多少？”, also written gold-first and then five distractors. Subtype `second_person` is 8 cases. Subtype `anaphora_unnamed` is 2.

Unanswerable cases split into two subtypes, 18 + 18 per language:

- `missing_rel`: `people[0..5]`. The person's other two relations are stored. The asked relation is not. Other people's facts fill the case out to 6 writes.
- `absent_person`: the absent names (Clara, Victor, Edith, Hassan, Naomi, Felix / 马岚、宋北、冯棠、韩澈、邓星、曹野). The store holds 6 facts about people who are present.

### 和训练不相交 / Disjoint from training

`fit_question_head` / `build_facts` 不 import 这套语料。训练输入（seed 0）是整数姓名和关系 id，不是句子。不相交在 `eval_heldout` 里是硬检查，和 ret@1 分开：

1. 特征行。训练 `_feats` 是 `f[0] ∈ {+1, -1}`，`f[1:4]` 里一个 1，`f[4:8]` 里一个 0.5，其余为 0。探针 `surface_feats` 是 `sha256(role + "\\n" + text)` 的 8 维，每一维在 `{0.2, 0.4, 0.6, 0.8}`，没有 0。角色是 `question`、`statement` 或 `chat`。和全部训练 `_feats` 逐行相等的条数是 0。
2. 编码器状态。每条探针句子经 `embed_event(surface_feats, bge 行)`。这些状态和 `build_facts` 的 q、k 逐位相等的条数是 0。同一段评测把重建的训练特征行和训练高斯文本再送进 `embed_event`，结果与 `build_facts` 逐位相等（整段关掉 mkldnn，结束时恢复），所以比较的是同一条编码通路。
3. 文本余弦。L2 之后的 bge 行对 L2 之后的训练高斯文本，最大余弦是 0.1983。硬门槛是 < 0.5。
4. 字面量。问句不等于 `question_head.py` 的字符串常量，也不出现在随包 meta JSON 里。长度 ≥ 24 的常量既不包含问句，也不被问句包含。命中数是 0。

语料校验另要求：问句互异、金标和旧值不出现在问句或陈述里、问句不等于陈述、每类至少 24 条且至少 12 个问句框架、更新类的旧值与新值不同且旧值确实写过、干扰类恰好 3 条闲聊、无答案类没有写入被问的身份。固定问句必须在集内：上面列出的第二人称和不点名指代。

`fit_question_head` / `build_facts` do not import this corpus. Training inputs (seed 0) are integer name and relation ids, not sentences. Disjointness is a hard check inside `eval_heldout`, separate from ret@1:

1. Feature rows. Training `_feats` uses `f[0] ∈ {+1, -1}`, a single 1 in `f[1:4]`, a single 0.5 in `f[4:8]`, and zeros elsewhere. Probe `surface_feats` is an 8-d SHA-256 of `role + "\\n" + text` with each lane in `{0.2, 0.4, 0.6, 0.8}` and no zeros. The role is `question`, `statement`, or `chat`. Exact row matches against every training `_feats` row: 0.
2. Encoder states. Each probe string goes through `embed_event(surface_feats, bge row)`. Bitwise matches against `build_facts` q and k: 0. The same run feeds the rebuilt training feature rows and training Gaussian text back through `embed_event` and requires bitwise equality with `build_facts` (mkldnn off for the run, restored afterwards), so the comparison uses that encoder path.
3. Text cosine. L2-normalized bge rows against L2-normalized training Gaussian text, max cosine 0.1983. The hard ceiling is < 0.5.
4. Literals. No question equals a string constant in `question_head.py` or appears in the shipped meta JSON. No constant of length ≥ 24 contains a question or is contained in one. Hits: 0.

The corpus validator also requires unique questions, no gold or old value inside the question or the statement, a question that differs from its statement, at least 24 cases and 12 question frames per category, a distinct old value that was actually written on updates, exactly 3 chat turns on the distractor category, and no write of the asked identity on unanswerable cases. The second-person and unnamed lines listed above are required members of the set.

### 嵌入和覆盖阈值 / Embedder and overwrite

文本前端与 `tidal.embed` 一致：`BAAI/bge-small-zh-v1.5`，revision `7999e1d3359715c523056ef9478215996d62a620`，CLS，L2，maxlen 64，没有 query instruction 前缀。缓存是 `tidal/question_heldout_emb.npz`，744 条字符串，sha256 `8413bb1a4a964d26042ca61338745ccef1a2494643b54eafa074a5d5f7c20bb3`。测试读缓存，不下载模型。

排序用 `overwrite_cos=1.0`，近重复键不能在提问前删掉另一个身份，和合成问句池的选择相同。`gold lost @0.9` 把同一批写入在 `overwrite_cos=0.9` 的影子库里重放，数金标值已经不在槽里的条数。同身份更新在 0.9 上仍是硬检查。

The text front end matches `tidal.embed`: `BAAI/bge-small-zh-v1.5`, revision `7999e1d3359715c523056ef9478215996d62a620`, CLS, L2, maxlen 64, no query-instruction prefix. The cache is `tidal/question_heldout_emb.npz`, 744 strings, sha256 `8413bb1a4a964d26042ca61338745ccef1a2494643b54eafa074a5d5f7c20bb3`. Tests read the cache and do not download the model.

Ranking uses `overwrite_cos=1.0`, so a near-duplicate key cannot delete another identity before the question, the same choice as the synthetic question pools. `gold lost @0.9` replays the same writes on a shadow store at `overwrite_cos=0.9` and counts cases whose gold value is no longer stored. Same-identity updates stay a hard check at 0.9.

### 结果 / Results

随包问句头，没有重训，没有开关，`ask` 仍走对比残差。ret@1 是当前值的精确解码（电话 / 日期字符串，或地点 token 元组）。配对单元格是余弦 · 头。stale 只在更新类有定义，等于覆盖前的旧值。false 是无答案问句仍返回了某条已存事实。`ask` 没有弃权：非空库总是解码 top-1 槽，只有空库返回 `None`。

Shipped question head, no refit, no switch. `ask` still uses the contrastive residual. ret@1 is an exact decode of the current value (phone or date string, or place-token tuple). Paired cells are cosine · head. Stale is defined only on updates and means the answer equals the value from before the overwrite. False means an unanswerable question still returned some stored fact. `ask` has no abstain: a nonempty store always decodes the top-1 slot, and only an empty store returns `None`.

| category | n | jaccard | cosine ret@1 | head ret@1 | head only | cosine only | phone cos·head | date cos·head | tokens cos·head | stale cos·head | false cos·head | gold lost @0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en_direct | 36 | 0.12 | 9/36 | 8/36 | 2/36 | 3/36 | 1/12 · 1/12 | 4/12 · 4/12 | 4/12 · 3/12 | — | — | 0/36 |
| zh_direct | 36 | 0.28 | 8/36 | 8/36 | 4/36 | 4/36 | 3/12 · 3/12 | 2/12 · 3/12 | 3/12 · 2/12 | — | — | 0/36 |
| en_indirect | 40 | 0.10 | 12/40 | 10/40 | 4/40 | 6/40 | 3/13 · 4/13 | 5/14 · 2/14 | 4/13 · 4/13 | — | — | 0/40 |
| zh_indirect | 40 | 0.19 | 7/40 | 9/40 | 4/40 | 2/40 | 3/13 · 3/13 | 4/14 · 5/14 | 0/13 · 1/13 | — | — | 0/40 |
| en_cross_turn | 37 | 0.17 | 6/37 | 6/37 | 3/37 | 3/37 | 0/13 · 3/13 | 4/12 · 2/12 | 2/12 · 1/12 | — | — | 34/37 |
| zh_cross_turn | 37 | 0.22 | 4/37 | 7/37 | 5/37 | 2/37 | 0/13 · 0/13 | 3/12 · 4/12 | 1/12 · 3/12 | — | — | 35/37 |
| en_after_distractors | 36 | 0.12 | 3/36 | 2/36 | 1/36 | 2/36 | 0/12 · 0/12 | 1/12 · 1/12 | 2/12 · 1/12 | — | — | 25/36 |
| zh_after_distractors | 36 | 0.24 | 1/36 | 4/36 | 3/36 | 0/36 | 0/12 · 2/12 | 1/12 · 2/12 | 0/12 · 0/12 | — | — | 23/36 |
| en_after_update | 36 | 0.17 | 10/36 | 14/36 | 9/36 | 5/36 | 4/12 · 5/12 | 4/12 · 3/12 | 2/12 · 6/12 | 0/36 · 0/36 | — | 0/36 |
| zh_after_update | 36 | 0.29 | 7/36 | 3/36 | 1/36 | 5/36 | 2/12 · 1/12 | 1/12 · 0/12 | 4/12 · 2/12 | 0/36 · 0/36 | — | 0/36 |
| en_unanswerable | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 36/36 | — |
| zh_unanswerable | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 36/36 | — |
| answerable | 370 | 0.19 | 67/370 | 71/370 | 36/370 | 32/370 | 16/124 · 22/124 | 29/124 · 26/124 | 22/122 · 23/122 | 0/72 · 0/72 | — | 117/370 |
| unanswerable | 72 | — | — | — | — | — | — | — | — | — | 72/72 · 72/72 | — |
| all | 442 | 0.19 | 67/370 | 71/370 | 36/370 | 32/370 | 16/124 · 22/124 | 29/124 · 26/124 | 22/122 · 23/122 | 0/72 · 0/72 | 72/72 · 72/72 | 117/370 |

可答合计：余弦 67/370，头 71/370。头独中 36/370，余弦独中 32/370。按值的种类：电话 16/124 · 22/124，日期 29/124 · 26/124，多 token 22/122 · 23/122。没有一类接近合成集上的 47/48。问句对金标陈述的 token Jaccard 均值是 0.19，只说明两边用词重叠低，不进分数。

残差并不接近 0。问句映射 ‖Δq‖/‖q‖：训练均值 0.9331（中位 0.9422），留出均值 0.7645（中位 0.7614）。键映射：训练均值 0.8794，留出陈述均值 0.6749。头在表面状态上加了一个大残差，排序仍和原始余弦接近。这不是「头只记住了一种问句模板、换个说法就崩」的图案：直接、间接、跨轮、干扰、更新来回互有胜负。因此没有改主路径，没有把 `ask` 换回余弦，也没有在这 442 条上重训。

同身份更新的 stale 是 0/72 · 0/72。旧字节码在覆盖后不在槽里；答错的是别人的事实。无答案的 false 是 72/72 · 72/72。

分数（排序器自己的 top-1 余弦，以及 top-1 减 top-2）在可答和无答案上重叠，设不出一条能分开两者的弃权线：

- 头 margin，可答：min 0.0004，p10 0.0061，p25 0.0164，p50 0.0345，p75 0.0593，p90 0.0950，max 0.2155，mean 0.0441
- 头 margin，无答案：min 0.0023，p10 0.0106，p25 0.0219，p50 0.0459，p75 0.0773，p90 0.1152，max 0.1904，mean 0.0570
- 余弦 margin，可答：min 0.0000，p10 0.0025，p25 0.0065，p50 0.0146，p75 0.0263，p90 0.0458，max 0.2203，mean 0.0198
- 余弦 margin，无答案：min 0.0007，p10 0.0034，p25 0.0083，p50 0.0178，p75 0.0357，p90 0.0496，max 0.0877，mean 0.0240
- 头 top-1，可答：min 0.1391，p10 0.3704，p25 0.4785，p50 0.5578，p75 0.6289，p90 0.6747，max 0.7630，mean 0.5424
- 头 top-1，无答案：min 0.3446，p10 0.4034，p25 0.4827，p50 0.5595，p75 0.6187，p90 0.6654，max 0.7397，mean 0.5479
- 余弦 top-1，可答：min 0.8551，p10 0.9302，p25 0.9478，p50 0.9650，p75 0.9762，p90 0.9841，max 0.9948，mean 0.9597
- 余弦 top-1，无答案：min 0.9013，p10 0.9320，p25 0.9527，p50 0.9656，p75 0.9753，p90 0.9828，max 0.9957，mean 0.9612

若弃权定义为 margin < t（没有实现，`ask` 仍总是返回 top 槽）：

| ranker | t | answerable kept | exact among kept | unanswerable still answered |
|---|---:|---:|---:|---:|
| head | 0.01 | 314/370 | 64/314 | 65/72 |
| head | 0.02 | 256/370 | 55/256 | 58/72 |
| head | 0.05 | 123/370 | 26/123 | 30/72 |
| head | 0.10 | 32/370 | 7/32 | 11/72 |
| head | 0.20 | 1/370 | 0/1 | 0/72 |
| head | 0.50 | 0/370 | 0/0 | 0/72 |
| cosine | 0.01 | 244/370 | 51/244 | 51/72 |
| cosine | 0.02 | 135/370 | 35/135 | 33/72 |
| cosine | 0.05 | 28/370 | 6/28 | 7/72 |
| cosine | 0.10 | 2/370 | 1/2 | 0/72 |
| cosine | 0.20 | 1/370 | 1/1 | 0/72 |
| cosine | 0.50 | 0/370 | 0/0 | 0/72 |

`gold lost @0.9` 合计 117/370。直接、间接、更新都是 0/n，因为金标最后写入。跨轮金标先写：英文 34/37，中文 35/37。干扰项金标在中间：英文 25/36，中文 23/36。排序探针里共存的键对有 8390 对，余弦 > 0.9 的有 5267 对，最大 0.9950。`d=32` 的随机 `tproj` 把不同身份的表面陈述压过线上近重复阈值，后写的另一条事实会盖掉先写的槽。默认阈值留在 0.9：它管所有事实，改它并不会把表面问句的 67/370 变成合成集的 47/48。

subtype（定义与上表相同）：

| category | n | jaccard | cosine ret@1 | head ret@1 | head only | cosine only | phone cos·head | date cos·head | tokens cos·head | stale cos·head | false cos·head | gold lost @0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absent_person | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 36/36 | — |
| anaphora_unnamed | 2 | 0.13 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 · 0/2 | — | — | — | — | 2/2 |
| missing_rel | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 36/36 | — |
| second_person | 8 | 0.20 | 3/8 | 3/8 | 2/8 | 2/8 | 1/2 · 1/2 | 2/4 · 0/4 | 0/2 · 2/2 | — | — | 0/8 |

Answerable total: cosine 67/370, head 71/370. Head-only 36/370, cosine-only 32/370. By value kind: phone 16/124 · 22/124, date 29/124 · 26/124, multi-token 22/122 · 23/122. No category is near the synthetic 47/48. Mean token Jaccard of a question against its gold statement is 0.19. That only records low lexical overlap. It is not a score.

The residual is far from zero. Question-map ‖Δq‖/‖q‖: train mean 0.9331 (median 0.9422), held-out mean 0.7645 (median 0.7614). Key-map: train mean 0.8794, held-out statements mean 0.6749. The head adds a large residual on these surface states, and the ranking stays close to raw cosine. The pattern is not “the head memorized one question template and misses every paraphrase”: direct, indirect, cross-turn, distractor, and update rows trade wins. The main path is unchanged. `ask` stays on the contrastive residual. The head was not refit on these 442 cases.

Stale after same-identity updates is 0/72 · 0/72. The old byte code is gone after the overwrite; a miss is someone else's fact. False on unanswerable questions is 72/72 · 72/72.

The ranker's own top-1 cosine, and top-1 minus top-2, overlap for answerable and unanswerable questions, so there is no abstain cut that separates them:

- head margin, answerable: min 0.0004, p10 0.0061, p25 0.0164, p50 0.0345, p75 0.0593, p90 0.0950, max 0.2155, mean 0.0441
- head margin, unanswerable: min 0.0023, p10 0.0106, p25 0.0219, p50 0.0459, p75 0.0773, p90 0.1152, max 0.1904, mean 0.0570
- cosine margin, answerable: min 0.0000, p10 0.0025, p25 0.0065, p50 0.0146, p75 0.0263, p90 0.0458, max 0.2203, mean 0.0198
- cosine margin, unanswerable: min 0.0007, p10 0.0034, p25 0.0083, p50 0.0178, p75 0.0357, p90 0.0496, max 0.0877, mean 0.0240
- head top-1, answerable: min 0.1391, p10 0.3704, p25 0.4785, p50 0.5578, p75 0.6289, p90 0.6747, max 0.7630, mean 0.5424
- head top-1, unanswerable: min 0.3446, p10 0.4034, p25 0.4827, p50 0.5595, p75 0.6187, p90 0.6654, max 0.7397, mean 0.5479
- cosine top-1, answerable: min 0.8551, p10 0.9302, p25 0.9478, p50 0.9650, p75 0.9762, p90 0.9841, max 0.9948, mean 0.9597
- cosine top-1, unanswerable: min 0.9013, p10 0.9320, p25 0.9527, p50 0.9656, p75 0.9753, p90 0.9828, max 0.9957, mean 0.9612

If abstain were `margin < t` (not implemented; `ask` still returns the top slot):

| ranker | t | answerable kept | exact among kept | unanswerable still answered |
|---|---:|---:|---:|---:|
| head | 0.01 | 314/370 | 64/314 | 65/72 |
| head | 0.02 | 256/370 | 55/256 | 58/72 |
| head | 0.05 | 123/370 | 26/123 | 30/72 |
| head | 0.10 | 32/370 | 7/32 | 11/72 |
| head | 0.20 | 1/370 | 0/1 | 0/72 |
| head | 0.50 | 0/370 | 0/0 | 0/72 |
| cosine | 0.01 | 244/370 | 51/244 | 51/72 |
| cosine | 0.02 | 135/370 | 35/135 | 33/72 |
| cosine | 0.05 | 28/370 | 6/28 | 7/72 |
| cosine | 0.10 | 2/370 | 1/2 | 0/72 |
| cosine | 0.20 | 1/370 | 1/1 | 0/72 |
| cosine | 0.50 | 0/370 | 0/0 | 0/72 |

`gold lost @0.9` totals 117/370. Direct, indirect, and update are 0/n because gold is written last. Cross-turn writes gold first: English 34/37, Chinese 35/37. Distractor cases put gold in the middle: English 25/36, Chinese 23/36. Co-resident key pairs in the ranking probe: 8390, of which 5267 have cosine > 0.9, max 0.9950. The random `tproj` at `d=32` pushes surface statements of different identities over the live near-duplicate threshold, so a later fact overwrites an earlier slot. The default threshold stays 0.9. It governs every fact, and moving it would not turn the surface 67/370 into the synthetic 47/48.

Subtypes use the same definitions:

| category | n | jaccard | cosine ret@1 | head ret@1 | head only | cosine only | phone cos·head | date cos·head | tokens cos·head | stale cos·head | false cos·head | gold lost @0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absent_person | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 36/36 | — |
| anaphora_unnamed | 2 | 0.13 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 · 0/2 | — | — | — | — | 2/2 |
| missing_rel | 36 | — | — | — | — | — | — | — | — | — | 36/36 · 36/36 | — |
| second_person | 8 | 0.20 | 3/8 | 3/8 | 2/8 | 2/8 | 1/2 · 1/2 | 2/4 · 0/4 | 0/2 · 2/2 | — | — | 0/8 |

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
