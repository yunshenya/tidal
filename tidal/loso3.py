"""Phase 3 leave-one-scenario-out with REAL public streams (replaces the script-synthetic livestream of phase 2).
  LIVE-A: trained on real + LLM-synth + public group chats/IRC + spoken 1:1 + meetings, NO livestream
          -> held-out Twitch streams, and a different platform/broadcast (YouTube+Twitch live chat, no host)
  LIVE-ref: same + Twitch training streams (in-domain reference)
  PUB->REAL: trained on public data only (no tidal data) -> tidal's real held-out chats
  which-event: Telegram / IRC gold reply links: when "self" replies, which of the preceding messages is it answering?
Baselines fitted on exactly the same training rows (prior; LR on the same event features). Blocks: conv x 5 min
(public) / conv x hour (real). usage: TIDAL_DATASET=p3 python -m tidal.loso3 -> reports/loso_phase3.json"""
import json, os, numpy as np, warnings
os.environ.setdefault("TIDAL_DATASET", "p3")
from tidal import vap as VP
from tidal.loso import train_mask, fit_baselines, evaluate_set
warnings.filterwarnings("ignore")
PUB = ["tg", "irc", "twitch", "candor", "aishell4"]

def blocks_for(meta, rows):
    ts = meta.ts.to_numpy()[rows]; conv = meta.conv.to_numpy()[rows]; real = (meta.source.to_numpy()[rows] == "real")
    return np.array([f"{c}:{int(t // (3600 if r else 300))}" for c, t, r in zip(conv, ts, real)])

def which_event(D, rows, tags, max_back=20):
    meta = D["meta"]; role = meta.role.to_numpy(); mid = meta.msg_id.to_numpy(); conv = meta.conv.to_numpy(); ts = meta.ts.to_numpy()
    rset = set(rows.tolist()); host = [r for r in rows if role[r] == "self" and isinstance(meta.responds_to.iat[r], str)]
    cands = {}
    for r in host:
        c = [q for q in range(max(0, r - 200), r) if conv[q] == conv[r] and role[q] != "self" and q in rset][-max_back:]
        if c and meta.responds_to.iat[r] in set(mid[c]): cands[r] = c
    allc = sorted({q for c in cands.values() for q in c}); pos = {q: k for k, q in enumerate(allc)}; out = {}
    H = {}
    preds = {t: VP.predict(t, D, np.array(allc), conv_mask=np.isin(meta.source.to_numpy(), meta.source.to_numpy()[allc[:1]]))[0] for t in tags} if allc else {}
    for name in tags + ["most_recent", "random"]:
        hits = []
        for r, c in cands.items():
            tgt = meta.responds_to.iat[r]
            if name == "most_recent": hits.append(float(mid[c[-1]] == tgt)); continue
            if name == "random": hits.append(1.0 / len(c)); continue
            lg = preds[name]["y_act"][[pos[q] for q in c]]; ps = np.exp(lg[:, 0]) / np.exp(lg).sum(1)
            hits.append(float(mid[c[int(np.argmax(ps * np.exp(-(ts[r] - ts[c]) / 60.0)))]] == tgt))
        h = np.array(hits); out[name] = dict(hit1=float(h.mean()) if len(h) else None, n=len(h)); H[name] = h
    # paired bootstrap over conversations: model - most_recent
    convs = np.array([conv[r] for r in cands])
    if len(convs):
        from tidal.metrics import boot, ci
        for t in tags:
            d = boot(lambda idx: H[t][idx].mean() - H["most_recent"][idx].mean(), convs, 1000, 0)
            out[t]["minus_most_recent"] = dict(mean=float(np.nanmean(d)), ci=ci(d))
    out["_n_candidates_mean"] = float(np.mean([len(c) for c in cands.values()])) if cands else None
    return out

def main(boot=300):
    D = VP.load("p3"); meta = D["meta"]; src = meta.source.to_numpy(); sp = meta.split.to_numpy()
    have = lambda t: os.path.exists(f"models/{t}.pt")
    sets = {"twitch_test": np.flatnonzero((src == "pub_twitch") & (sp == "pub_test")),
            "artemis_live": np.flatnonzero(src == "pub_artemis"),
            "danmaku_test": np.flatnonzero((src == "pub_danmaku") & (sp == "pub_test")),
            "real_test": np.flatnonzero((src == "real") & np.isin(sp, ["test_time", "test_group"]))}
    for k in sets:                                            # keep evaluation affordable: subsample long livestream sets by conversation-time blocks
        r = sets[k]
        if len(r) > 60000:
            b = blocks_for(meta, r); ub = np.unique(b); keep = set(np.random.default_rng(0).choice(ub, int(len(ub) * 60000 / len(r)), replace=False))
            sets[k] = r[np.array([x in keep for x in b])]
    plan = [("LIVE-A: no livestream in training -> held-out Twitch streams", "twitch_test", ["real", "llm", "tg", "irc", "candor", "aishell4"], ["P3L_nolive_s0"]),
            ("LIVE-A: no livestream in training -> other platform/broadcast live chat (no host)", "artemis_live", ["real", "llm", "tg", "irc", "candor", "aishell4"], ["P3L_nolive_s0"]),
            ("phase-2 model (real + LLM-synth only) -> held-out Twitch streams", "twitch_test", ["real", "llm"], ["P2a_s1"]),
            ("LIVE-ref: with Twitch training streams -> held-out Twitch streams", "twitch_test", ["real", "llm"] + PUB, ["P3mix_s0", "P3mix_s1", "P3mix_s2"]),
            ("LIVE-ref: with Twitch training streams -> other platform/broadcast live chat", "artemis_live", ["real", "llm"] + PUB, ["P3mix_s0", "P3mix_s1", "P3mix_s2"]),
            ("LIVE-A: no livestream in training -> Chinese danmaku (Bilibili, VOD-aligned; timing/VAP only)", "danmaku_test", ["real", "llm", "tg", "irc", "candor", "aishell4"], ["P3L_nolive_s0"]),
            ("LIVE-ref(Twitch): English livestream chat but no danmaku -> Chinese danmaku", "danmaku_test", ["real", "llm"] + PUB, ["P3mix_s0", "P3mix_s1", "P3mix_s2"]),
            ("DM-ref: with danmaku training videos -> held-out danmaku videos", "danmaku_test", ["real", "llm"] + PUB + ["danmaku"], ["P3dm_s0"]),
            ("DM-ref model -> held-out Twitch streams", "twitch_test", ["real", "llm"] + PUB + ["danmaku"], ["P3dm_s0"]),
            ("PUB->REAL: public data only (no tidal data) -> tidal real held-out chats", "real_test", PUB, ["P3pub_s0"])]
    out = dict(note="public streams are real human data under their licenses (not redistributed); real_test = tidal's private held-out chats (metrics only)",
               sets={k: int(len(v)) for k, v in sets.items()}, experiments=[])
    for name, sname, data, tags in plan:
        tags = [t for t in tags if have(t)]
        if not tags: print("skip", name); continue
        rows = sets[sname]; tm = train_mask(meta, data); bl, vbl = fit_baselines(D, tm)
        cm = np.isin(src, np.unique(src[rows]))
        res, _ = evaluate_set(D, rows, blocks_for(meta, rows), tags, bl, vbl, cm, boot=boot, vap_boot=200)
        res.update(name=name, test=sname, train=data, models=tags); out["experiments"].append(res)
        print(name); [print(f"  {h:10s} n={r['n']:6d} prior={list(r['prior'].values())[0]:.3f} lr_G={list(r['lr_G'].values())[0]:.3f} " +
                           " ".join(f"{t}={list(r[t].values())[0]:.3f}(Δ{r[t]['vs_lr_G']['d_primary_mean']:+.3f}[{r[t]['vs_lr_G']['d_primary_ci'][0]:+.3f},{r[t]['vs_lr_G']['d_primary_ci'][1]:+.3f}])" for t in tags))
                      for h, r in res["heads"].items()]
        print("  vap", {k: (round(v["mean_auc"], 3) if v["mean_auc"] else None, v.get("minus_lr_G")) for k, v in res["vap"].items()}, flush=True)
    we = {}
    for s in ("pub_tg", "pub_irc"):
        rows = np.flatnonzero((src == s) & (sp == "pub_test"))
        we[s] = which_event(D, rows, [t for t in ["P3pub_s0", "P3mix_s0", "P2a_s1"] if have(t)])
        print("which-event", s, we[s], flush=True)
    out["which_event"] = we
    json.dump(out, open("reports/loso_phase3.json", "w"), indent=1, default=float)

if __name__ == "__main__": main()
