# neuromem 总结 / SUMMARY（v13 A2s → v22）

## 目标 / Goal

在**测试时**给冻结的 `SmolLM2-135M` 挂上神经记忆，使模型能回忆多关系对话事实，**而不把事实文本放进 prompt**。对照：事实在上下文（in-context）与 `bge-small-en-v1.5` RAG。

Test-time neural memory on a frozen SmolLM2-135M so it recalls multi-relation dialogue facts without putting them in the prompt; vs in-context and bge RAG.

评测：held-out **test**，n=96，free-gen 字符串匹配；会话干扰轮次数 N∈{0,4,16,32}；另测事实流遗忘（K20/50/100）与同名同关系更新。

---

## 有效路径 / Path that worked

1. **v13 A2s** — answer-span 值池化（`vpool=ans`，`dv=576`）+ **identity inject**（`freeze_inj`，`inj=I`）。值不做 L2 归一化（幅度对 copy-through 注入必要）。骨干：`mem_ckpt_v13A2s.pt`。
2. **kind_protect（v15）** — 评测时跳过 chat 写入/衰减；事实写仍用小 λ。混合会话 N4/N16 提升，不修事实流早期遗忘。
3. **episodic slot store（v19）** — 每条事实一个 slot `(k,v,name,rel)`；同 `(name,rel)` 或 `cos(k,·)>0.9` 覆盖；容量 C、FIFO 淘汰；读出 `softk4`（top-4 softmax，τ=0.05）再走 `inj(scale(v))`。N32 **0.208→0.469**，更新 **~0.67/0**。
4. **contrastive retrieval（v20 / v22）** — 冻结骨干上训 2 层 q/k 适配器（256→256 GELU→128，InfoNCE）。v20：K100 ret@1 **0.45→0.54**；v22：长流难负例再训 → K100 ret@1 **0.59**，N32 gen **0.594**。
5. **residual inject MLP（v21）** — `add = inj(scale(v)) + MLP(scale(v))`（末层零初始化），教师强制 next-token CE。N0 天花板 **0.594→0.656**，并带到各 N。

最终系统 ≈ **A2s backbone + slot store + retr_v22 + inj_v21（softk4）**。

---

## 失败 / Failed

| 尝试 | 结果 |
|------|------|
| **multi-matrix sum readout（v16）** | S4/S8 旋转路由：读出 Σ M_s@q 稀释信号；N32 差到 0.06–0.09。明确失败。 |
| **naïve key_shield（v17）** | 写后外积恢复旧 key 读出 → 早期事实保护（K20 first 0.275），但**同名更新倒置**（new/old **0.021/0.437**）。不可用。 |
| UA / cos-skip shield（v18/18b） | 部分修好更新，但仍是 protection↔plasticity 权衡，未同时达标；最终被 slot store 取代。 |

---

## 最终数字 / Final numbers

held-out test，n=96 free-gen。括号内为 ret@1（slot 配置）。

| config | N0 | N4 | N16 | N32 | update new/old | K100 ret@1 |
|--------|----|----|-----|-----|----------------|------------|
| kind_protect (v15) | 0.583 | 0.417 | 0.333 | **0.208** | 0.542/0.000 | — |
| slot_softk4_C256 (**v19**) | 0.594 | 0.479 | 0.458 | 0.469 | **0.667**/0.000 | 0.451 |
| retr_softk4 (**v20**) | 0.594 | 0.469 | 0.458 | 0.469 | 0.667/0.000 | **0.541** |
| retr_softk4 + inj (**v21**) | **0.656** | 0.573 | 0.594 | 0.573 | 0.646/0.000 | 0.541 |
| **retr_v22 + inj softk4 (v22)** | 0.656 | 0.573 | **0.604** | **0.594** | **0.667**/0.000 | **0.590** |
| bge RAG | 0.604 | 0.604 | 0.552 | **0.635** | — | — |
| fact in context | 0.604 | 0.615 | 0.562 | 0.688 | — | — |

**高亮：**

- N32：**0.208 → 0.594**（kind_protect → v22）
- 更新：约 **0.67 / 0**（无 stale）
- K100 ret@1：**0.45 → 0.59**（v19 → v22）
- 相对 RAG N32：**差约 0.041**（0.635 − 0.594）

---

## 剩余差距与下一步 / Remaining gap & next levers

相对 RAG 约 **0.04**（N32）。分解大致为：长流/会话上仍有检索 miss（N32 ret@1≈0.875；K100≈0.59）× inject 解码天花板（N0 在 ret@1=1.0 时 gen≈0.66）。

可能杠杆：

1. 更强的长流对比学习 / hard negatives / 跨会话负例，继续抬 ret@1。
2. 更好的 value / inject（多 token 答案、电话号码等难类型）。
3. 联合微调 retr+inj（v22 未做 joint CE）。
4. 更大容量或分层记忆；或与轻量 RAG 混合。

---

## 关键文件 / Key files

| 文件 | 作用 |
|------|------|
| `neuromem_v13.py`（+ `neuromem.py`…`neuromem_v9.py` 继承链） | A2s 骨干定义 |
| `data_v12.py` | 多关系对话数据（需本地 `data12/`） |
| `mem_ckpt_v13A2s.pt` | 冻结 A2s 记忆权重 |
| `gate_head_v13A2s.pt` | 可选写门控头 |
| `retr_v20.pt` / `retr_v22.pt` | 对比检索适配器 |
| `inj_v21.pt` | 残差 inject MLP |
| `run_v19.py`…`run_v22.py`, `train_v20.py`…`train_v22.py` | 评测 / 训练 |
| `REPORT_v*.md`, `results_v*.json` | 各轮报告与原始数字 |
| `SUMMARY.md`（本文件）, `README.md` | 发布说明 |

---

## 复现评测 / Reproduce eval（v22）

```bash
cd neuromem
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers

# 首次需下载 SmolLM2-135M；本机若 xet 卡住可用普通 HTTP，之后离线：
export HF_HUB_OFFLINE=1

# 准备 data12/（DailyDialog 对话文本 + Synthetic-Persona-Chat CSV；见 README）
# 目录结构需含 data12/dd_{train,validation,test}/…/dialogues_*.txt 与 spc_{train,valid,test}.csv

HF_HUB_OFFLINE=1 .venv/bin/python run_v22.py \
  --ckpt mem_ckpt_v13A2s.pt \
  --retr retr_v20.pt --retr_v22 retr_v22.pt --inj inj_v21.pt \
  --configs slot_retr_v22_softk4_inj_C256 \
  --n_eval 96 --Ns 0,4,16,32 --tag v22_repro
```

训练适配器（可选）：`train_v20.py` → `train_v21.py` → `train_v22.py`（均冻结 A2s）。

灵感来自 Titans / HOPE 类测试时记忆，**非官方实现、无隶属关系**。
