# neuromem（已并入 tidal，仓库已删除）

这是一次**冻结 SmolLM2-135M** 上的测试时记忆研究，不是 tidal 的运行时依赖。GitHub 上的 neuromem 仓库已经删除。这里只留设计记录：哪些写法留下来了，哪些明确失败。对应的实现现在在 `tidal/episodic.py`，用的是 tidal 自己的编码器状态（`fproj` / `tproj`），不加载 135M，也不读任何 `.pt`。

This directory archives a frozen-SmolLM2-135M study. The neuromem GitHub repo is gone. The design that survived lives in `tidal/episodic.py` and is trained, when at all, by `python -m tidal.episodic_train` on tidal's own encoder. No 135M checkpoint is loaded. Do not commit `.pt` files here.

## 留下的设计 / What moved into tidal

- 一条事实一个槽 `(k, v, identity)`，同身份覆盖，容量 FIFO。
- soft top-4，温度 0.05。写门控丢掉闲聊。
- 检索：两层 q/k MLP（Linear → GELU → Linear），InfoNCE，难负例（同名异关系、同关系异名）。在 tidal 里这是**残差**（末层零初始化）。没训练时排序就是原始余弦。SmolLM 实验里的适配器是投影到 128 维再 L2 归一化，不是残差；投影不能零初始化成恒等映射，所以 tidal 用残差。
- 读出：残差 inject MLP，末层零初始化。没训练、且库为空时，加到 `h` 上的是 0。空库在进 inject 之前就返回 0。
- 电话、日期、多 token 用位置字节码，不均值池化，不 L2 归一化。

SmolLM 上的自由生成数字（n=96，约 0.02 的起伏分不出来）不能当成 tidal 的结果。tidal 上重新量过的 ret@1 写在 `docs/episodic_memory.md`。问句检索会涨，episodic_eval 那种随机单位干扰上余弦已经是 1，训过的头不会更好；用存进去的编码器键在其他事实里做查询时，训过的头会退步，所以默认不加载权重。

## 明确不要的 / Discarded

- 多块矩阵读出再求和（v16）：值被稀释。
- 朴素 key shield（v17）：同名更新写不进去。
- v23 把 retr 和 inj 用 next-token CE 绑在一起训：没有作为 tidal 的路径。tidal 没有 token 语言模型，inject 不能抄那份损失。

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
