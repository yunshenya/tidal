"""Shadow-mode report: per task (head) metrics of the tidal model vs the phase-1 baselines on shadow predictions
whose outcome labels are available. Metrics: n / positives, accuracy (Wilson 95% CI), F1 + macro-F1, ROC-AUC, PR-AUC,
Brier, ECE (+ reliability bins), and a paired block-bootstrap Δ (model − baseline) -- blocks = conversation × hour.
Thresholds are the ones frozen from phase-1 validation (never tuned on shadow data).
usage: python -m tidal.shadow.report [--since YYYY-MM-DD] [--max-lag-min 30] [--boot 500] [--out FILE.md] [--json FILE]"""
import argparse, json, sys, time
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tidal import metrics as M
from tidal.shadow import store as S

TZ = timezone(timedelta(hours=8))

# fast numpy metrics for the bootstrap loops (sklearn per-call overhead dominates at these sizes)
def _auc(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def _ap(y, p):
    n1 = y.sum()
    if n1 == 0 or n1 == len(y): return np.nan
    o = np.argsort(-p, kind="mergesort"); yy = y[o]; ps = p[o]
    tp = np.cumsum(yy); k = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]       # last index of each tied group
    prec = tp[k] / (k + 1); rec = tp[k] / n1
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))
def _f1(y, yh):
    tp = np.sum(yh & (y == 1)); fp = np.sum(yh & (y == 0)); fn = np.sum(~yh & (y == 1))
    return float(2 * tp / max(1, 2 * tp + fp + fn))
def _mf1(y, yh, K):
    return float(np.mean([_f1((y == k).astype(int), yh == k) for k in range(K)]))
BIN = ["y_eot", "y_self", "y_addr", "y_hreply"]; MULTI = {"y_act": 3, "y_recheck": 7}
DESC = {"y_eot": "话轮结束（EOT）", "y_self": "发言人会继续补充（180s 内）", "y_addr": "是否在叫机器人（群聊入站）",
        "y_hreply": "60s 内有其他人接话", "y_act": "机器人 说/等/不说（模仿旧策略）", "y_recheck": "到下一条消息的间隔（7 桶）"}

def load(c, since, until, max_lag):
    p = pd.read_sql_query("select p.msg_id, p.regime, p.system, p.probs, p.event_ts, p.lag_s, l.* from predictions p "
                          "join labels l on l.msg_id = p.msg_id where p.event_ts >= ? and p.event_ts < ?", c, params=(since, until))
    p = p.loc[:, ~p.columns.duplicated()]
    if max_lag is not None: p = p[p.lag_s <= max_lag]
    p["block"] = p.conv + ":" + (p.event_ts // 3600).astype(int).astype(str)
    return p

def reliability(y, p, bins=10):
    out = []
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & ((p < lo + 1 / bins) if lo + 1 / bins < 1 else (p <= 1))
        if m.sum(): out.append((round(lo, 1), int(m.sum()), round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)))
    return out

def boot_ci(fn, blocks, n_boot, seed=0):
    bs = M.boot(fn, blocks, n_boot, seed); return [M.ci(bs[:, j]) for j in range(bs.shape[1])]

def bin_metrics(y, p, thr, blocks, n_boot):
    y = y.astype(int); yh = p >= thr; n = len(y); k = int((yh == y).sum()); ok = 0 < y.sum() < n
    r = dict(n=n, pos=int(y.sum()), threshold=round(thr, 4), accuracy=k / n, accuracy_ci=M.wilson(k, n),
             f1=float(f1_score(y, yh, zero_division=0)), f1_macro=float(f1_score(y, yh, average="macro", labels=[0, 1], zero_division=0)),
             roc_auc=float(roc_auc_score(y, p)) if ok else np.nan, pr_auc=float(average_precision_score(y, p)) if ok else np.nan,
             brier=float(np.mean((p - y) ** 2)), ece=M.ece(y, p), reliability=reliability(y, p))
    if ok and n_boot:
        def fn(i):
            yy = y[i]; return (_f1(yy, yh[i]), _auc(yy, p[i]), _ap(yy, p[i]), M.ece(yy, p[i]))
        r["f1_ci"], r["roc_auc_ci"], r["pr_auc_ci"], r["ece_ci"] = boot_ci(fn, blocks, n_boot)
    return r

def multi_metrics(y, P, blocks, n_boot, K):
    y = y.astype(int); yh = P.argmax(1); n = len(y); k = int((yh == y).sum())
    r = dict(n=n, dist=np.bincount(y, minlength=K).tolist(), accuracy=k / n, accuracy_ci=M.wilson(k, n),
             f1_macro=float(f1_score(y, yh, average="macro", labels=list(range(K)), zero_division=0)),
             ece=M.ece((yh == y).astype(int), P.max(1)), nll=float(-np.mean(np.log(np.clip(P[np.arange(n), y], 1e-9, 1)))))
    ys = (y == 0).astype(int)                  # class 0: y_act 'speak' / y_recheck '<2 s'
    if 0 < ys.sum() < n: r["c0_roc_auc"] = float(roc_auc_score(ys, P[:, 0])); r["c0_pr_auc"] = float(average_precision_score(ys, P[:, 0]))
    if n_boot:
        r["f1_macro_ci"], = boot_ci(lambda i: (_mf1(y[i], yh[i], K),), blocks, n_boot)
    return r

def paired(y, pa, pb, blocks, n_boot, binary):
    y = y.astype(int)
    if binary:
        def fn(i):
            yy = y[i]
            return (_auc(yy, pa[i]) - _auc(yy, pb[i]), _ap(yy, pa[i]) - _ap(yy, pb[i]), np.mean((pb[i] - yy) ** 2) - np.mean((pa[i] - yy) ** 2))
        names = ["d_roc_auc", "d_pr_auc", "d_brier_gain"]
    else:
        K = pa.shape[1]
        fn = lambda i: (_mf1(y[i], pa[i].argmax(1), K) - _mf1(y[i], pb[i].argmax(1), K),)
        names = ["d_macro_f1"]
    bs = M.boot(fn, blocks, n_boot, 1)
    return {nm: dict(mean=float(np.nanmean(bs[:, j])), ci=M.ci(bs[:, j])) for j, nm in enumerate(names)}

def aurc(y, P, binary):
    """area under the risk-coverage curve (decision at 0.5 / argmax, confidence = max prob); lower is better."""
    dec = (P >= .5).astype(int) if binary else P.argmax(1); conf = np.maximum(P, 1 - P) if binary else P.max(1)
    o = np.argsort(-conf, kind="stable"); c_ = (dec == y.astype(int))[o]; risk = 1 - np.cumsum(c_) / np.arange(1, len(c_) + 1)
    return float(risk.mean())

def build(c, since, until, max_lag, n_boot):
    frozen = json.loads((S.STATE / "frozen.json").read_text())
    p2f = S.STATE / "frozen_p2.json"; frozen2 = json.loads(p2f.read_text()) if p2f.exists() else None
    d = load(c, since, until, max_lag); R = dict(generated_at=time.time(), since=since, until=until, max_lag_s=max_lag, regimes={})
    for regime, g in d.groupby("regime"):
        probs = {s: dict(zip(gg.msg_id, gg.probs.map(json.loads))) for s, gg in g.groupby("system")}
        lab = g.drop_duplicates("msg_id").set_index("msg_id")
        models = sorted(s for s in probs if s.startswith("model:")); model = models[0]; RR = {}
        for h in BIN + list(MULTI):
            y = lab[h].dropna(); ids = [m for m in y.index if all(m in probs[s] and h in probs[s][m] for s in probs)]
            if len(ids) < 5: RR[h] = dict(n=len(ids), note="too few labeled events"); continue
            yv = y.loc[ids].to_numpy(); blocks = lab.loc[ids, "block"].to_numpy(); hr = dict(systems={}, strongest_baseline=frozen["strongest_baseline"][regime][h])
            for s in probs:
                v = np.array([probs[s][m][h] for m in ids], float)
                if h in BIN:
                    thr = frozen2["thresholds"].get(h, 0.5) if (frozen2 and s == "model:" + frozen2["tag"]) else frozen["thresholds"][regime][h].get(s, 0.5)
                    hr["systems"][s] = bin_metrics(yv, v, thr, blocks, n_boot)
                else:
                    hr["systems"][s] = multi_metrics(yv, v, blocks, n_boot, MULTI[h])
            arr = {s: np.array([probs[s][m][h] for m in ids], float) for s in probs}
            hr["paired"] = {md: {s: paired(yv, arr[md], arr[s], blocks, n_boot, h in BIN) for s in probs if s != md} for md in models}
            hr["aurc"] = {s: aurc(yv, arr[s], h in BIN) for s in probs}
            if h == "y_act":   # selective abstention ('wait and recheck') logged by the phase-2 model
                for md in models:
                    ab = [probs[md][m].get("abstain") for m in ids]
                    if all(a is not None for a in ab):
                        ab = np.array(ab, bool); corr = arr[md].argmax(1) == yv.astype(int)
                        hr.setdefault("abstention", {})[md] = dict(coverage=float((~ab).mean()), acc_all=float(corr.mean()),
                            acc_answered=float(corr[~ab].mean()) if (~ab).any() else None, acc_abstained=float(corr[ab].mean()) if ab.any() else None,
                            median_recheck_s=float(np.median([probs[md][m]["recheck_after_s"] for m, a in zip(ids, ab) if a])) if ab.any() else None)
            RR[h] = hr
        R["regimes"][regime] = dict(model=model, models=models, n_events=int(lab.shape[0]), heads=RR)
    tot = lambda q: c.execute(q).fetchone()
    R["coverage"] = dict(events=tot("select count(*) from events")[0], predicted=tot("select count(distinct msg_id) from predictions")[0],
                         labeled=tot("select count(*) from labels")[0],
                         runs_ok=tot("select count(*) from runs where status='ok'")[0], runs_err=tot("select count(*) from runs where status!='ok'")[0])
    pa = S.get_meta(c, "predict_after")
    mm = pd.read_sql_query("select role, media_kinds, msg_chars, text from events where ts > ?", c, params=(pa,))
    kinds = {}
    for k in mm.media_kinds.dropna(): 
        for x in json.loads(k): kinds[x] = kinds.get(x, 0) + 1
    R["multimodal_capture"] = dict(events=len(mm), with_text=int(mm.text.notna().sum()), media_kinds_from_placeholders=kinds,
                                   inbound_with_msg_chars=int(mm.msg_chars.notna().sum()), inbound_msg_chars_zero=int((mm.msg_chars == 0).sum()))
    return R

def fmt_ci(ci): return "" if ci is None or any(x != x for x in ci) else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
def f3(x): return "—" if x is None or x != x else f"{x:.3f}"

def markdown(R):
    t = lambda x: datetime.fromtimestamp(x, TZ).strftime("%Y-%m-%d %H:%M")
    L = [f"# tidal 影子模式报告", "", f"生成于 {t(R['generated_at'])}（UTC+8）；事件时间范围 {t(R['since'])} – {t(min(R['until'], R['generated_at']))}；"
         + (f"只计入预测时延 ≤ {R['max_lag_s']/60:.0f} 分钟的预测" if R["max_lag_s"] else "包含回填预测"), "",
         f"覆盖：事件 {R['coverage']['events']}，已预测 {R['coverage']['predicted']}，已打标 {R['coverage']['labeled']}；成功运行 {R['coverage']['runs_ok']} 次，失败 {R['coverage']['runs_err']} 次。", "",
         "阈值（准确率 / F1 用）冻结自第一阶段 val，不在影子数据上调。CI：准确率用 Wilson，其余用按 会话×小时 分块的 bootstrap。", ""]
    for regime, rr in R["regimes"].items():
        L += [f"## {regime} 口径（模型 {', '.join(rr.get('models', [rr['model']]))}，{rr['n_events']} 个已打标事件）", ""]
        for h, hr in rr["heads"].items():
            if "systems" not in hr: L += [f"### {h} {DESC[h]}：样本不足（n={hr['n']}）", ""]; continue
            L += [f"### {h} {DESC[h]}（val 选出的最强基线：{hr['strongest_baseline']}）", ""]
            if h in BIN:
                L += ["| 系统 | n | 正例 | 准确率 [CI] | F1 [CI] | ROC-AUC [CI] | PR-AUC [CI] | Brier | ECE |", "|---|---|---|---|---|---|---|---|---|"]
                for s, m in hr["systems"].items():
                    L.append(f"| {s} | {m['n']} | {m['pos']} | {m['accuracy']:.3f}{fmt_ci(m['accuracy_ci'])} | {m['f1']:.3f}{fmt_ci(m.get('f1_ci'))} | "
                             f"{f3(m['roc_auc'])}{fmt_ci(m.get('roc_auc_ci'))} | {f3(m['pr_auc'])}{fmt_ci(m.get('pr_auc_ci'))} | {m['brier']:.3f} | {m['ece']:.3f} |")
                for md, pp in hr["paired"].items():
                    L += ["", f"{md} − 其他系统（配对 bootstrap，均值 [95% CI]；CI 不含 0 才算显著）：", ""]
                    for s, dd in pp.items():
                        L.append(f"- vs {s}: ΔROC-AUC {dd['d_roc_auc']['mean']:+.3f}{fmt_ci(dd['d_roc_auc']['ci'])}；ΔPR-AUC {dd['d_pr_auc']['mean']:+.3f}{fmt_ci(dd['d_pr_auc']['ci'])}；Brier 改善 {dd['d_brier_gain']['mean']:+.4f}{fmt_ci(dd['d_brier_gain']['ci'])}")
                    mr = hr["systems"][md]["reliability"]
                    L += ["", f"{md} 校准分箱（下界, n, 平均预测, 实际正例率）：" + "; ".join(f"{a}:{b}/{c_:.2f}/{d_:.2f}" for a, b, c_, d_ in mr)]
            else:
                L += ["| 系统 | n | 类别分布 | 准确率 [CI] | macro-F1 [CI] | 类0 ROC-AUC | ECE(top) | NLL |", "|---|---|---|---|---|---|---|---|"]
                for s, m in hr["systems"].items():
                    L.append(f"| {s} | {m['n']} | {m['dist']} | {m['accuracy']:.3f}{fmt_ci(m['accuracy_ci'])} | {m['f1_macro']:.3f}{fmt_ci(m.get('f1_macro_ci'))} | {f3(m.get('c0_roc_auc'))} | {m['ece']:.3f} | {m['nll']:.3f} |")
                for md, pp in hr["paired"].items():
                    L += ["", f"{md} − 其他系统 macro-F1：" + "；".join(f"vs {s} {dd['d_macro_f1']['mean']:+.3f}{fmt_ci(dd['d_macro_f1']['ci'])}" for s, dd in pp.items())]
                for md, ab in hr.get("abstention", {}).items():
                    L += ["", f"{md} 选择性弃权（置信度低于 val 阈值时“等一下再看”）：覆盖率 {ab['coverage']:.2f}，全部准确率 {ab['acc_all']:.3f}，"
                          f"作答部分准确率 {f3(ab['acc_answered'])}，弃权部分准确率 {f3(ab['acc_abstained'])}，弃权时建议复查间隔中位数 {f3(ab['median_recheck_s'])} s"]
            L += ["", "风险-覆盖曲线下面积 AURC（越低越好）：" + "；".join(f"{s} {v:.3f}" for s, v in hr.get("aurc", {}).items()), ""]
    mc = R["multimodal_capture"]
    L += ["## 非文本 / 多模态信号采集情况", "", f"预测窗口内事件 {mc['events']}，其中有文本 {mc['with_text']}；占位符识别出的媒体类型：{mc['media_kinds_from_placeholders'] or '无'}；"
          f"带 msg_chars 的入站 {mc['inbound_with_msg_chars']}，其中 msg_chars=0（多半是图片/语音/表情等非文本）{mc['inbound_msg_chars_zero']}。", ""]
    return "\n".join(L)

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--since"); ap.add_argument("--until"); ap.add_argument("--max-lag-min", type=float)
    ap.add_argument("--boot", type=int, default=500); ap.add_argument("--out"); ap.add_argument("--json"); a = ap.parse_args(argv)
    c = S.connect()
    day = lambda s: datetime.fromisoformat(s).replace(tzinfo=TZ).timestamp()
    since = day(a.since) if a.since else S.get_meta(c, "predict_after"); until = day(a.until) if a.until else time.time() + 1
    R = build(c, since, until, a.max_lag_min * 60 if a.max_lag_min else None, a.boot)
    md = markdown(R)
    if a.out: open(a.out, "w").write(md)
    if a.json: json.dump(R, open(a.json, "w"), indent=1, default=float, ensure_ascii=False)
    print(md)

if __name__ == "__main__":
    main()
