"""Phase 4B summary: backbone table from checkpoints + bench + eval jsons; applies the pre-registered winner rule.
python -m tidal.report4 [--private] -> markdown on stdout (private adds real-data val/test numbers)."""
import json, os, sys, numpy as np, torch
KINDS = [("gru", "GRU", "P3pub_s0", [f"P3ft_s{s}" for s in range(3)]), ("tx_kv", "RoPE+KV transformer", "P4pub_tx_kv", None),
         ("mamba3_siso", "Mamba-3 SISO", "P4pub_mamba3_siso", None), ("mamba3", "Mamba-3 MIMO (r=4)", "P4pub_mamba3", None),
         ("m3_ablate_m2", "Mamba-3 ablation (no trapezoid/RoPE ≈ Mamba-2 recurrence)", "P4pub_m3_ablate_m2", None)]

def ck(tag):
    p = f"models/{tag}.pt"
    return torch.load(p, weights_only=False, map_location="cpu") if os.path.exists(p) else None

def table(private=False):
    B = json.load(open("reports/bench_phase4.json")) if os.path.exists("reports/bench_phase4.json") else {}
    P = json.load(open("reports/eval_phase4_pub.json")) if os.path.exists("reports/eval_phase4_pub.json") else {}
    rows = []
    for k, name, pub, ft in KINDS:
        ft = ft or [f"P4ft_{k}_s{s}" for s in range(3)]; cs = [ck(t) for t in ft]; cp = ck(pub)
        r = dict(kind=k, name=name, done=all(c is not None for c in cs), pub_done=cp is not None)
        if r["done"]: r["val"] = [float(c["best_val"]) for c in cs]; r["val_mean"] = float(np.mean(r["val"])); r["params"] = int(cs[0]["params"]); r["ft_sec"] = [c["train_seconds"] for c in cs]
        if cp is not None: r["pub_train_sec"] = cp["train_seconds"]; r["pub_best_val_heads"] = float(cp["best_val"])
        b = B.get(k, {}); r.update(step_p50=b.get("step_p50_ms"), step_p95=b.get("step_p95_ms"), state_bytes=b.get("state_bytes"),
                                    onnx=(b.get("onnx") or {}).get("kind"), ort_p95=(b.get("onnx") or {}).get("ort_p95_ms"), onnx_err=(b.get("onnx") or {}).get("max_abs_err"))
        if k in P: r["pub_test_vap"] = P[k]["vap_loss"]; r["pub_d_vs_gru"] = P[k].get("vap_loss_delta_vs_gru")
        rows.append(r)
    done = [r for r in rows if r["done"]]       # literal rule: every trained backbone is a candidate (incl. the Mamba-2-style ablation)
    best = min(r["val_mean"] for r in done); tied = [r for r in done if r["val_mean"] - best <= 0.005]
    win = min(tied, key=lambda r: r["step_p95"] if r["step_p95"] is not None else 1e9)
    return rows, win, tied

if __name__ == "__main__":
    private = "--private" in sys.argv; rows, win, tied = table(private)
    print("| backbone | params | " + ("real-VAL loss (3 seeds) | mean | " if private else "") + "public test VAP loss | step p50/p95 ms (1 thread) | state | ONNX streaming step |")
    print("|---|---|" + ("---|---|" if private else "") + "---|---|---|---|")
    for r in rows:
        v = (f"{', '.join(f'{x:.4f}' for x in r['val'])} | {r['val_mean']:.4f} | " if r["done"] else "running | – | ") if private else ""
        pv = f"{r['pub_test_vap']:.4f}" if r.get("pub_test_vap") is not None else "–"
        on = f"{r['onnx']} (ORT p95 {r['ort_p95']} ms, err {r['onnx_err']:.1e})" if r.get("onnx_err") is not None else (r["onnx"] or "–")
        print(f"| {r['name']} | {r.get('params', '–')} | {v}{pv} | {r['step_p50']}/{r['step_p95']} | {r['state_bytes']} B | {on} |")
    print(f"\nwinner by rule: {win['name']} (tied within 0.005: {[t['name'] for t in tied]})"); json.dump(rows, open("/tmp/report4_rows.json", "w"), default=str)

def cmp_table(key, path="reports/eval_phase4.json"):
    """markdown table of one eval4 comparison (private: real-data numbers)."""
    C = json.load(open(path))["comparisons"][key]; out = [f"**{key}** (a = first, b = second; Δ = a − b, 95% CI)", "", "| head | split | n | a | b | Δ |", "|---|---|---|---|---|---|"]
    for h, H in C.items():
        if h.startswith("vap_loss"): continue
        for spn, R in H.items():
            d = R["delta"]; star = " *" if d["d_primary_ci"][0] > 0 or d["d_primary_ci"][1] < 0 else ""
            out.append(f"| {h} | {spn} | {R['n']} | {R['a']:.3f} | {R['b']:.3f} | {d['d_primary_mean']:+.3f} [{d['d_primary_ci'][0]:+.3f}, {d['d_primary_ci'][1]:+.3f}]{star} |")
    for spn in ("test_time", "test_group"):
        R = C[f"vap_loss_{spn}"]; out.append(f"| VAP loss (lower better) | {spn} | {R['n'] if 'n' in R else ''} | {R['a']:.4f} | {R['b']:.4f} | {R['d_mean']:+.4f} [{R['ci'][0]:+.4f}, {R['ci'][1]:+.4f}] |")
    return "\n".join(out)
