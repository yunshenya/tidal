# 情景事实记忆 / Episodic fact memory

连续体（`tidal/continuum.py`）在影子回放里始终带着一块情景事实槽（`tidal/episodic.py`）。它是 `heads` 的一部分，不是 SmolLM2 旁路，也不加载 [neuromem](https://github.com/yunshenya/neuromem) 的包或 135M 权重。那个仓库只说明哪几种写法值得留下。

Continuum's shadow replay always owns an episodic slot store. Readout sits on the `heads` path. Nothing here imports the neuromem package, SmolLM2, or a 135M checkpoint. The sister repo is the design reference only.

## 行为 / Behavior

`Continuum` 一构造就有一个 `EpisodicStore`。每个头之前：

```
h_adapt = h + Σ_levels adapter(h)
h' = h_adapt + soft_topk(h_adapt)
```

空库时 `soft_topk` 是全零。适配器也还是零的时候（刚构造，或 `levels=()`），回放就和静态模型一样。没有 `enable_episodic`，也没有把读出乘成 0 的比例开关。调用方如果仍传入这类键，读出照样加上。

一条事实是一个槽 `(k, v, name, rel, group)`。

- 同一个 `(group, name, rel)`（`name` 非空）直接覆盖。新的键和值换上，旧键不会被补回去。
- 身份对不上、但 `cos(k, k_i) > 0.9` 且在同一 `group` 里，当成同一事实的改写，也覆盖。身份匹配优先于这条近重复规则。
- 容量默认 256。满了淘汰最旧的写入时间（FIFO）。覆盖会刷新时间，刚更新的事实不会被当成旧槽丢掉。
- 读出是 soft top-k：余弦最高的 4 个槽，温度 0.05 做 softmax，对 **value** 做凸组合（权重和为 1）。查询某个 `group` 时，能看见该群的槽和 `group is None` 的全局槽，看不见别的群。
- 写门控丢掉 `kind="chat"` 和 `kind="noise"`，也丢掉没有方向的键。`surprise_thr` 只有在配置了阈值、并且这次写入带了 `surprise` 时才起作用；不传 surprise 不会挡事实。`replay` 不把原始事件流写成事实。

键和值用 tidal 已有的通路，宽度就是编码器状态 `d`：

- `embed_event(feats)`：冻结模型的 `fproj` + 主干一步。`feats` 是 `features.py` / `features_g.py` 的一行（模型若带情绪列，把那几列算进去）。
- `embed_event(feats, text_emb)`：`text_emb` 是 `tidal.embed` / `TextFrontEnd` 的 512 维 bge。只有模型自己带 `tproj` 时才走这条文本投影。
- `remember(k, v, ...)` 只接受已经在 `d` 里的向量。512 维 bge 不能直接当槽。

An empty store adds zeros. With zero adapters (a fresh model, or `levels=()`), replay matches the static model. Same-identity overwrite replaces the whole slot. Soft top-4 (temperature 0.05) is a convex combination of values. Chat and noise writes are dropped. Keys and values are encoder states from the frozen event model; a 512-d bge vector goes through that model's `tproj` when it exists.

## 从 neuromem 留下什么 / What was kept

neuromem 在冻结的 SmolLM2-135M 上试过好几轮。tidal 只留下不依赖那套权重的部分。

| 做法 | 在 tidal 里 |
|---|---|
| 离散槽 `(k, v, identity)`，同身份覆盖 | 保留。更新换掉整槽，不把新旧值加在一起。 |
| soft top-4，温度 0.05 | 保留。近邻混合，一次 top-1 检错不至于整段读错。 |
| 写门控跳过闲聊 | 保留。未标注的事件流不是事实，`replay` 不写槽。 |
| 多块矩阵读出再求和 | 不采用。求和会把检索到的值稀释掉。这里只有一块槽表。 |
| 朴素 key shield（写完按旧键把旧读出补回去） | 不采用。同名更新会写不进去。覆盖后槽里的键就是新键。 |
| 对比学习 q/k 适配器、残差 inject MLP、SmolLM2-135M | 不引入。注入就是把读出加到 `h` 上。 |

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
```

`episodic_cfg` 只调容量、`soft_k`、温度、覆盖余弦和可选的 surprise 阈值。它不是开关。
