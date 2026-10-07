"""Aggregate reports/eval_{T,TS}.json into markdown tables (seed mean±sd for model groups)."""
import json, sys, numpy as np, re, collections
def fmt_ci(c): return f"[{c[0]:.3f},{c[1]:.3f}]" if c and not any(np.isnan(c)) else ""
def main(regime, pick):
    R = json.load(open(f"reports/eval_{regime}.json")); out = []
    for h, r in R["heads"].items():
        sb = r["strongest_baseline"]
        for sp, ss in r["systems"].items():
            key = "pr_auc" if "pr_auc" in next(iter(ss.values())) else "macro_f1"
            out.append(f"\n**{regime} · {h} · {sp}**（主指标 {key}；val 选出的最强基线 = {sb}；test 上事后最好的基线 = {r['best_baseline_on_test'][sp]}）\n")
            out.append("| 系统 | n | 正例/分布 | 主指标 [95% CI] | macro-F1 [CI] | ECE | Brier/NLL |")
            out.append("|---|---|---|---|---|---|---|")
            for name, m in ss.items():
                if name.startswith("model:") and name != "model:" + pick: continue
                if key == "pr_auc":
                    out.append(f"| {name} | {m['n']} | {m['pos']} ({m['prevalence']:.2f}) | {m['pr_auc']:.3f} {fmt_ci(m.get('pr_auc_ci'))} | {m['f1_macro']:.3f} {fmt_ci(m.get('f1_macro_ci'))} | {m['ece']:.3f} | {m['brier']:.3f} |")
                else:
                    out.append(f"| {name} | {m['n']} | {m['dist']} | {m['macro_f1']:.3f} {fmt_ci(m.get('macro_f1_ci'))} | — | {m['ece_top']:.3f} | {m['nll']:.3f} |")
            m = ss["model:" + pick]; d = m["vs_strongest"]; d2 = m["vs_best_on_test"]
            out.append(f"\nΔ(模型−最强基线 {sb}) = {d['d_primary_mean']:+.3f} CI{fmt_ci(d['d_primary_ci'])}；Δ(模型−事后最好基线) = {d2['d_primary_mean']:+.3f} CI{fmt_ci(d2['d_primary_ci'])}")
            # seed groups
            groups = collections.defaultdict(list)
            for name, mm in ss.items():
                if name.startswith("model:"):
                    g = re.sub(r"_s\d+$", "", name[6:]); groups[g].append(mm[key])
            out.append("；".join(f"{g}: {np.mean(v):.3f}±{np.std(v):.3f}(n={len(v)})" for g, v in groups.items()))
    open(f"reports/summary_{regime}.md", "w").write("\n".join(out)); print("\n".join(out))
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
