# 情景事实记忆 / Episodic fact memory

连续体（`tidal/continuum.py`）在影子回放里始终带着一块情景事实槽（`tidal/episodic.py`）。它是 `heads` 的一部分，不是 SmolLM2 旁路，也不加载 135M 权重。neuromem 那个仓库已经删除；当时的笔记在 `research/neuromem/`，活的设计在 `tidal/episodic.py`。

Continuum's shadow replay always owns an episodic slot store. Readout sits on the `heads` path. Nothing here imports the neuromem package, SmolLM2, or a 135M checkpoint. The study notes live in `research/neuromem/`.

## 行为 / Behavior

`Continuum` 一构造就有一个 `EpisodicStore`。每个头之前：

```
h_adapt = h + Σ_levels adapter(h)
h' = h_adapt + soft_topk(h_adapt)
```

空库时 `soft_topk` 是全零。连续体那层适配器也还是零的时候（刚构造，或 `levels=()`），回放就和静态模型一样。没有 `enable_episodic`，也没有把读出乘成 0 的比例开关。调用方如果仍传入这类键，读出照样加上。排序是原始余弦，分母是 `kn*qn + 1e-12`。

An empty store adds zeros. With zero continuum adapters (a fresh model, or `levels=()`), replay matches the static model. Ranking is raw cosine (`kn*qn + 1e-12`). There is no `enable_episodic` and no scale that multiplies the readout by zero.

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

评测是精确匹配（`python -m tidal.episodic_eval`，pytest 会跑）。用例数大于原先 n≈96 的自由生成抽样：覆盖、soft top-k、群隔离、写门控、因果顺序、电话 / 生日 / 多 token 都必须逐条对上。0.02 那种幅度在 96 条里分不出来，这里不估一个率。

Mean-pooling drops digit order, adding the new code onto the old one leaves stale digits, and L2-normalizing destroys the code. A string or token list is stored as a positional byte code in the value vector (`2*(d-4)` bytes, 248 at `d=128`). Same-identity overwrite replaces that code. Heads still add the soft top-k convex combination; exact recall decodes the winning slot. A blended residual fails its checksum instead of becoming a different phone. The exact-match harness is larger than the old n≈96 free-gen sample and is not a rate estimate.

## 从 neuromem 留下什么 / What was kept

neuromem 在冻结的 SmolLM2-135M 上试过好几轮。tidal 只留下不依赖那套权重的部分。

| 做法 | 在 tidal 里 |
|---|---|
| 离散槽 `(k, v, identity)`，同身份覆盖 | 保留。更新换掉整槽，不把新旧值加在一起。 |
| soft top-4，温度 0.05 | 保留。近邻混合，一次 top-1 检错不至于整段读错。 |
| 写门控跳过闲聊 | 保留。未标注的事件流不是事实，`replay` 不写槽。 |
| 多块矩阵读出再求和 | 不采用。求和会把检索到的值稀释掉。这里只有一块槽表。 |
| 朴素 key shield（写完按旧键把旧读出补回去） | 不采用。同名更新会写不进去。覆盖后槽里的键就是新键。 |
| 对比学习 q/k MLP、残差 inject MLP | 不在主路径上。在 tidal 的 `fproj`/`tproj` 上量过，问句检索会涨，但用存进去的键在其他事实里查询会从 48/48 掉到 18/48；inject 把电话字节码推离网格。两套头已从代码里删掉。数字见下一节。 |
| SmolLM2-135M、`transformers` | 不引入。存档脚本在 `research/neuromem/`，核心代码不 import。 |
| 答案跨度做均值池化，或写入时 L2 归一化 | 不采用。池化分不出顺序；归一化毁掉字节码。字符串走位置字节码。 |

## 量过、没有留下的头 / Heads that were measured and removed

对比 q/k MLP 和残差 inject MLP 曾经接在这条读出上，末层零初始化。在 tidal 自己的 `fproj` + `tproj` 上训过一次（seed 0，`d=32`，检索 400 步，注入 300 步，InfoNCE τ=0.07，难负例）。问句向量不是存进去的键。这不是 SmolLM 的 n≈96 自由生成，训练脚本也没有留在仓库里。

| 集合 | raw cosine ret@1 | 训过的 q/k 头 |
|---|---|---|
| 问句 ≠ 键，8 个事实干扰 | 13/48 = 0.2708 | 47/48 = 0.9792 |
| episodic_eval：查询 = 存进去的键，8 个随机单位向量 | 48/48 = 1 | 48/48 = 1 |
| 随机单位键，查询 = 键，8 个单位干扰 | 256/256 = 1 | 256/256 = 1 |
| 查询 = 存进去的编码器键，干扰是另外 8 条事实的键 | 48/48 = 1 | 18/48 = 0.3750 |

问句检索涨了。episodic_eval 那组随机单位干扰上余弦已经是 1，头没有更好。`recall(存进去的键)` 在其他事实中间从 48 掉到 18。注入没有语言模型可以抄，目标是把电话字节码的混合拉回干净码；300 步后 `+1-415-555-0134` 的最大绝对偏差是 0.00111271，解不出来。所以两套头不在 `tidal/episodic.py` 里。排序仍是原始余弦，空库仍是精确的零。

SmolLM 存档（`research/neuromem/`）里的 K100 ret@1 0.451→0.590、N0 生成 0.594→0.656 是另一个模型，不能当成这里的结果。

A contrastive query/key MLP and a residual inject MLP were fit once on tidal `fproj`/`tproj` states (seed 0, `d=32`, 400 retrieval steps, 300 inject steps) and then deleted. Question vs key rose from 13/48 to 47/48. episodic_eval's random-unit distractors stayed 48/48, and random-unit keys stayed 256/256: cosine was already perfect. Recall of the stored encoder key among other facts fell from 48/48 to 18/48. The inject fit moved `+1-415-555-0134` by max abs 0.00111271 and the code no longer decoded. Slot match is raw cosine. SmolLM's K100 and free-gen numbers stay in `research/neuromem/` and are not these figures.

## 影子策略 / Shadow policy

连续体仍然只回放已记录的流、报告指标，不驱动线上动作。情景槽不进 `DuplexController`，不改 tick。按群写入的事实只影响该群的读出（加上全局槽），所以一条群的事实不会改另一条群的影子分数。

Continuum stays shadow-only. Episodic readout is inside that replay path. It is not a live tick input.

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
cont.recall(k_phone)                            # winning slot's string, this group's scope
```

`episodic_cfg` 只调容量、`soft_k`、温度、覆盖余弦和可选的 surprise 阈值。它不是开关。
