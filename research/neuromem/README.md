# neuromem（仓库已删除；活路径不包含那两套头）

这是一次**冻结 SmolLM2-135M** 上的测试时记忆研究，不是 tidal 的运行时依赖。GitHub 上的 neuromem 仓库已经删除。留下的设计在 `tidal/episodic.py`：离散槽、同身份覆盖、soft top-k、写门控、位置字节码。对比 q/k MLP 和残差 inject MLP 在 tidal 自己的 `fproj` / `tproj` 上量过，然后从主路径删掉。这里不加载 135M，也不提交 `.pt`。

This directory archives a frozen-SmolLM2-135M study. The neuromem GitHub repo is gone. What tidal kept is the slot store in `tidal/episodic.py` (raw cosine, soft top-k, same-identity overwrite, the write gate, positional byte codes). The contrastive query/key MLP and the residual inject MLP were measured on tidal's encoder and removed. No 135M checkpoint is loaded. Do not commit `.pt` files here.

## 留在 tidal 里的 / What tidal kept

- 一条事实一个槽 `(k, v, identity)`，同身份覆盖，容量 FIFO。
- soft top-4，温度 0.05。排序是原始余弦。
- 写门控丢掉闲聊。空库读出是精确的零。
- 电话、日期、多 token 用位置字节码，不均值池化，不 L2 归一化。

## 量过并删掉的 / Measured, then removed

在 tidal 特征上（seed 0，`d=32`，不是 SmolLM）：问句检索 ret@1 从 13/48 升到 47/48；episodic_eval 那种随机单位干扰上余弦已经是 48/48 和 256/256，头没有更好；用存进去的编码器键在其他事实里查询从 48/48 掉到 18/48。inject 把 `+1-415-555-0134` 推离字节网格（最大绝对偏差 0.00111271），号码解不出来。两套头和当时的训练脚本不在 `tidal/` 里。说明见 `docs/episodic_memory.md`。

SmolLM 上的自由生成（n=96，约 0.02 的起伏分不出来）不能当成 tidal 的结果。

On tidal features the question task rose from 13/48 to 47/48, the episodic_eval unit-distractor sets stayed saturated at 1, and recall of the stored key among other facts fell from 48/48 to 18/48. The inject fit broke the phone byte code. Those modules are not in `tidal/`.

## 更早明确不要的 / Discarded on SmolLM

- 多块矩阵读出再求和（v16）：值被稀释。
- 朴素 key shield（v17）：同名更新写不进去。
- v23 把 retr 和 inj 用 next-token CE 绑在一起训：不作为 tidal 的路径。tidal 没有 token 语言模型。

## 文件 / Files

| 文件 | 内容 |
|---|---|
| `SUMMARY.md` | v13 A2s → v22 的总结 |
| `REPORT_v20.md` | 对比 q/k。K100 ret@1 0.451→0.541。N32 生成没动（inject 天花板） |
| `REPORT_v21.md` | 残差 inject。N0 生成 0.594→0.656（ret@1=1.0） |
| `REPORT_v22.md` | 长流难负例。K100 ret@1 → 0.590 |
| `REPORT_v23.md` | 联合 CE。不采用 |
| `train_v21.py`, `train_v22.py` | 当时的训练脚本，依赖已删除的 `neuromem_v13` / SmolLM。只作存档 |
| `requirements.txt` | 仅供阅读那两个脚本。tidal 核心不要装 `transformers` |

权重（`mem_ckpt_v13A2s.pt`、`retr_v20.pt`、`retr_v22.pt`、`inj_v21.pt`）没有放进这个目录。
