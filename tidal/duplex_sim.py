"""SYNTHETIC full-duplex sanity test + CPU realtime benchmark for tidal.duplex (全双工 / 非单线程).

Everything here is script-generated (no real audio/chat). A rule-based reference host acts on TRUE voice activity;
the tick model only sees NOISY VAD + events + her own speaking state + stub vision noise, and must reproduce the
reference controls: start / keep / yield (barge-in) / continue after a short interruption / backchannel / initiate on
dead air. Open-loop (teacher-forced self state). This tests the plumbing + learnability, NOT real-world quality.
usage: python -m tidal.duplex_sim train ENC_TAG | eval | bench ENC_TAG"""
import sys, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import os
from tidal.duplex import (ACTIONS, TICK_S, Event, AudioFrame, VisionFrame, SelfState, TickInput, TickFeaturizer, TickModel,
                          EventEncoder, DuplexController, N_TICK)
from tidal import vap as VP
from tidal.features_g import NB as FG_NB
A = {a: i for i, a in enumerate(ACTIONS)}
torch.set_num_threads(1)

def session(seed):
    r = np.random.default_rng(seed); dur = r.uniform(90, 240); n = int(dur / TICK_S)
    voice = r.random() < 0.75; rate = 0.0 if (voice and r.random() < 0.6) else r.uniform(0.05, 1.5)
    t0 = 1.7e9 + r.uniform(0, 86400 * 30); npart = 2 if rate == 0 else int(r.uniform(20, 3000))
    has_vis = r.random() < 0.5; music = (not voice) and r.random() < 0.5; DEAD = 8.0
    vad = np.zeros(n, bool); user_start = 1 + r.uniform(1, 4) if voice else 1e9
    seg = None; segs = []          # active user utterance: dict(end, pauses, kind)
    bot = dict(speaking=False, end=0, start=0, rem_int=0.0, int_seg=None)
    pending = False; ready_at = None; ready = False; last_user_end = -1e9; answered = True; next_bc_ok = 0
    events, labels, ticks = [], np.zeros(n, int), []
    danmaku = np.sort(r.uniform(0, dur, r.poisson(rate * dur))) if rate > 0 else np.array([])
    di = 0; quiet = 0.0; last_dm = -1e9; uid = 0; barge = None; bcs = []; carry_prev = []
    for k in range(n):
        t = k * TICK_S; T = t0 + t; evs = []; lab = "keep" if bot["speaking"] else "silent"
        # --- danmaku arrivals
        while di < len(danmaku) and danmaku[di] <= t:
            evs.append(Event(f"d{di}", t0 + danmaku[di], "other", f"v{r.integers(0, max(2, npart))}", "text")); last_dm = danmaku[di]; di += 1
            if not bot["speaking"] and not pending and not ready: pending = True; ready_at = t + r.uniform(0.3, 1.5)
        # --- user voice
        if voice and seg is None and t >= user_start and not bot["speaking"]:
            d = r.uniform(1, 6); pauses = []
            pt = t + r.uniform(1.5, 3)
            while d > 3 and pt < t + d - 1: pauses.append((pt, pt + r.uniform(0.25, 0.45))); pt += r.uniform(1.5, 3)
            seg = dict(start=t, end=t + d, pauses=pauses, kind="turn")
        if voice and barge is not None and seg is None and t >= barge[0]:
            seg = dict(start=t, end=t + barge[1], pauses=[], kind="barge"); barge = None
        u_on = False
        if seg is not None:
            if t >= seg["end"]:
                if seg["kind"] == "turn" or seg["end"] - seg["start"] >= 1.2:
                    evs.append(Event(f"u{uid}", t0 + seg["end"], "current", "u", "voice")); uid += 1
                    pending = True; ready_at = t + r.uniform(0.3, 1.5); ready = False; answered = False; bot["rem_int"] = 0.0
                last_user_end = t; was = seg; seg = None
                user_start = t + (r.uniform(10, 25) if r.random() < 0.25 else r.uniform(0.3, 3)) + 30 * (not voice)
            else:
                u_on = not any(a <= t < b for a, b in seg["pauses"])
        for bt in bcs:
            if bt <= t < bt + r.uniform(0.15, 0.25): u_on = True
        vad[k] = u_on
        if pending and ready_at is not None and t >= ready_at: pending = False; ready = True; ready_at = None
        # --- inputs at this tick = state BEFORE this tick's decision (the decision only affects the next tick)
        noisy = float(u_on) if r.random() > 0.04 else 1.0 - float(u_on)
        audio = AudioFrame(vad_other=noisy if voice or music else 0.0, vad_self=float(bot["speaking"]), music=float(music) * r.uniform(.5, 1), energy=float(u_on or bot["speaking"]) * r.uniform(.5, 1))
        vis = VisionFrame(salience=r.random(), scene_change=float(r.random() < .01)) if has_vis else None
        ss = SelfState(speaking=bot["speaking"], elapsed_s=max(0.0, t - bot["start"]) if bot["speaking"] else 0.0,
                       remaining_s=max(0.0, bot["end"] - t) if bot["speaking"] else 0.0,
                       content_ready=ready, pending_request=pending, interrupted_remaining_s=bot["rem_int"])
        pre_evs = list(evs)
        # --- reference host policy (acts on TRUE vad)
        run = 0
        j = k
        while j >= 0 and vad[j]: run += 1; j -= 1
        quiet = 0.0 if (u_on or bot["speaking"] or evs) else quiet + TICK_S
        if bot["speaking"]:
            if run * TICK_S >= 0.3 - 1e-9:
                lab = "yield"; bot["speaking"] = False; bot["rem_int"] = bot["end"] - t
            elif t >= bot["end"]:
                bot["speaking"] = False; lab = "silent"; user_start = max(user_start, t + r.uniform(0.3, 3)) if voice and r.random() > 0.25 else t + r.uniform(10, 25)
                bcs = []
        else:
            if bot["rem_int"] > 0 and seg is None and not u_on and t - last_user_end >= 0.5 and last_user_end > bot["start"]:
                lab = "continue"; bot.update(speaking=True, start=t, end=t + bot["rem_int"]); bot["rem_int"] = 0.0
            elif ready and seg is None and not u_on and t - last_user_end >= 0.5:
                lab = "start"; d = r.uniform(2, 8); bot.update(speaking=True, start=t, end=t + d); ready = False; answered = True
                evs.append(Event(f"b{k}", T, "self", "self", "voice"))
                if voice and r.random() < 0.35: barge = (t + r.uniform(0.5, max(0.6, d - 0.3)), r.uniform(0.6, 3.0)); user_start = 1e9
                bcs = [t + x for x in r.uniform(0.3, d, r.poisson(0.8))] if voice else []
            elif quiet >= DEAD and not pending and seg is None:
                lab = "initiate"; d = r.uniform(2, 6); bot.update(speaking=True, start=t, end=t + d); quiet = 0.0
                evs.append(Event(f"b{k}", T, "self", "self", "voice"))
            elif seg is not None and seg["kind"] == "turn" and t - seg["start"] >= 2.5 and t >= next_bc_ok and \
                    any(abs(t - (a + 0.2)) < TICK_S / 2 for a, b in seg["pauses"]):
                lab = "backchannel"; next_bc_ok = t + 3
        labels[k] = A[lab]
        carry = [e for e in evs if e not in pre_evs]
        ticks.append(TickInput(T, pre_evs + carry_prev, audio, vis, ss, float(npart) if r.random() < .5 else None))
        carry_prev = carry
    meta = dict(vad=vad, voice=voice, rate=rate)
    return ticks, labels, meta

def featurize(ticks, enc):
    enc.reset(); tf = TickFeaturizer(); X = np.zeros((len(ticks), N_TICK), np.float32); H = np.zeros((len(ticks), enc.m.body.hidden_size), np.float32)
    for k, ti in enumerate(ticks):
        for e in ti.events: enc.push(e, ti.n_participants)
        X[k] = tf(ti, enc.last); H[k] = enc.h[0].numpy()
    return X, H

def build(seeds, enc):
    out = []
    for s in seeds:
        ticks, y, meta = session(s); X, H = featurize(ticks, enc); out.append((X, H, y, meta))
    return out

W = torch.tensor([1, 20, 1, 20, 20, 20, 20], dtype=torch.float32)
def train(enc_tag, n_tr=240, n_va=40):
    m, ck = VP.load_model(enc_tag); enc = EventEncoder(m, ck["mu"], ck["sd"]); t0 = time.time()
    tr = build(range(1000, 1000 + n_tr), enc); va = build(range(5000, 5000 + n_va), enc); t_build = time.time() - t0
    torch.manual_seed(0); rng = np.random.default_rng(0); tm = TickModel(); opt = torch.optim.AdamW(tm.parameters(), 1e-3, weight_decay=1e-2)
    L = 400; best = (1e9, None); t1 = time.time()
    def chunks(data):
        for X, H, y, _ in data:
            for a in range(0, len(y) - 50, L): yield X[a:a + L], H[a:a + L], y[a:a + L]
    trc = list(chunks(tr)); vac = list(chunks(va))
    def stack(c):
        T = max(len(x[2]) for x in c); B = len(c)
        X = np.zeros((B, T, N_TICK), np.float32); H = np.zeros((B, T, 192), np.float32); Y = np.full((B, T), -100, np.int64)
        for i, (x, h, y) in enumerate(c): X[i, :len(y)] = x; H[i, :len(y)] = h; Y[i, :len(y)] = y
        return torch.from_numpy(X), torch.from_numpy(H), torch.from_numpy(Y)
    for ep in range(25):
        tm.train(); perm = rng.permutation(len(trc))
        for b in range(0, len(perm), 16):
            X, H, Y = stack([trc[i] for i in perm[b:b + 16]]); lg, _ = tm(X, H)
            loss = Fn.cross_entropy(lg.reshape(-1, len(ACTIONS)), Y.reshape(-1), weight=W, ignore_index=-100)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(tm.parameters(), 1.0); opt.step()
        tm.eval()
        with torch.no_grad():
            vl = np.mean([float(Fn.cross_entropy(tm(*stack(vac[i:i + 32])[:2])[0].reshape(-1, len(ACTIONS)), stack(vac[i:i + 32])[2].reshape(-1), weight=W, ignore_index=-100)) for i in range(0, len(vac), 32)])
        print(json.dumps(dict(ep=ep, val=round(vl, 4))), flush=True)
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in tm.state_dict().items()})
    torch.save(dict(state=best[1], enc=enc_tag, params=sum(p.numel() for p in tm.parameters()), train_s=round(time.time() - t1, 1), build_s=round(t_build, 1)), "models/duplex_tick.pt")
    print("saved", sum(p.numel() for p in tm.parameters()), "params", round(time.time() - t1), "s")

def match(pred, true, tol=3):
    used = set(); hits = 0; lat = []
    for t in true:
        c = [p for p in pred if abs(p - t) <= tol and p not in used]
        if c: p = min(c, key=lambda p: abs(p - t)); used.add(p); hits += 1; lat.append(p - t)
    return hits, lat

def events_of(seq, a):
    idx = np.flatnonzero(seq == a); return [i for j, i in enumerate(idx) if j == 0 or idx[j - 1] != i - 1]

RULE_GRID = [dict(yr=yr, q=q, dead=dead) for yr in (2, 3, 4) for q in (0.3, 0.5, 0.8) for dead in (4.0, 6.0, 8.0, 10.0)]
def rule_controller(X, yr=3, q=0.5, dead=8.0):
    """hand-written controller on the SAME noisy observed tick features (the meaningful baseline: the reference is a rule)."""
    from tidal.duplex import TICK_FEATS as TF
    c = lambda n: X[:, TF.index(n)]; ex = np.expm1
    spk = c("speaking") > .5; nv = c("vad_other") >= .5; oq = ex(c("log_other_quiet")); aq = ex(c("log_all_quiet"))
    se = ex(c("log_since_event")); ready = c("content_ready") > .5; pend = c("pending") > .5; irem = ex(c("log_interrupted_rem"))
    out = np.full(len(X), A["silent"]); run = 0; dq = 0.0; doq = 0.0      # quiet timers on DEBOUNCED vad (run >= yr)
    for k in range(len(X)):
        run = run + 1 if nv[k] else 0; on = run >= yr
        doq = 0.0 if on else doq + TICK_S; dq = 0.0 if (on or spk[k] or se[k] < TICK_S) else dq + TICK_S
        oq_k = min(oq[k], 60.0) if c("audio_avail")[k] < .5 else doq; aq_k = dq
        if spk[k]: out[k] = A["yield"] if run >= yr else A["keep"]
        elif irem[k] > 0 and oq_k >= q and not on: out[k] = A["continue"]
        elif ready[k] and oq_k >= q and not on: out[k] = A["start"]
        elif aq_k >= dead and se[k] >= dead and not pend[k]: out[k] = A["initiate"]
    return out

def evaluate(n_te=60):
    ck = torch.load("models/duplex_tick.pt", weights_only=False); tm = TickModel(); tm.load_state_dict(ck["state"]); tm.eval()
    m, eck = VP.load_model(ck["enc"]); enc = EventEncoder(m, eck["mu"], eck["sd"])
    va = build(range(5000, 5040), enc); best = (-1, None)          # tune the rule on the validation sessions
    for prm in RULE_GRID:
        f = []
        for a in ("start", "yield", "continue", "initiate"):
            tp = nt = npd = 0
            for X, H, y, _ in va:
                pe = events_of(rule_controller(X, **prm), A[a]); tr = events_of(y, A[a]); h, _ = match(pe, tr); tp += h; nt += len(tr); npd += len(pe)
            P_, R_ = tp / max(1, npd), tp / max(1, nt); f.append(2 * P_ * R_ / max(1e-9, P_ + R_))
        if np.mean(f) > best[0]: best = (float(np.mean(f)), prm)
    rule_prm = best[1]
    te = build(range(9000, 9000 + n_te), enc); res = {a: dict(tp=0, n_true=0, n_pred=0) for a in ACTIONS[1:] if a != "keep"}
    rres = {a: dict(tp=0, n_true=0, n_pred=0) for a in res}
    ylat = []; fy_bc = [0, 0]; rule = {r: dict(tp=0, n_pred=0, n_true=0, fy=0) for r in ("vad_1tick", "vad_2tick", "vad_3tick")}
    for X, H, y, meta in te:
        with torch.no_grad(): p = tm(torch.from_numpy(X)[None], torch.from_numpy(H)[None])[0][0].argmax(-1).numpy()
        for a in res:
            ia = A[a]; pe = events_of(p, ia); tr = events_of(y, ia); h, lat = match(pe, tr)
            res[a]["tp"] += h; res[a]["n_true"] += len(tr); res[a]["n_pred"] += len(pe)
            if a == "yield": ylat += lat
            pr = rule_controller(X, **rule_prm); pe = events_of(pr, ia); h, _ = match(pe, tr)
            rres[a]["tp"] += h; rres[a]["n_true"] += len(tr); rres[a]["n_pred"] += len(pe)
        # rule baselines for yield on NOISY vad while (reference) speaking
        from tidal.duplex import TICK_FEATS
        speaking = np.isin(y, [A["keep"], A["yield"]]); nv = X[:, TICK_FEATS.index("vad_other")] >= 0.5
        for r, need in (("vad_1tick", 1), ("vad_2tick", 2), ("vad_3tick", 3)):
            run = np.zeros(len(nv), int)
            for k in range(len(nv)): run[k] = run[k - 1] + 1 if (nv[k] and k) else int(nv[k])
            pe = events_of((speaking & (run >= need)).astype(int), 1); tr = events_of(y, A["yield"]); h, _ = match(pe, tr)
            rule[r]["tp"] += h; rule[r]["n_pred"] += len(pe); rule[r]["n_true"] += len(tr)
    for d in list(res.values()) + list(rule.values()) + list(rres.values()):
        d["precision"] = d["tp"] / max(1, d["n_pred"]); d["recall"] = d["tp"] / max(1, d["n_true"]); d["f1"] = 2 * d["precision"] * d["recall"] / max(1e-9, d["precision"] + d["recall"])
    out = dict(synthetic=True, sessions=n_te, tolerance_ms=300, per_action=res, yield_latency_ticks_vs_reference=dict(mean=float(np.mean(ylat)) if ylat else None),
               yield_rule_baselines_noisy_vad=rule, rule_controller_per_action=rres, rule_params_tuned_on_val=rule_prm, params=ck["params"], train_s=ck["train_s"], encoder=ck["enc"])
    json.dump(out, open("reports/duplex_sanity.json", "w"), indent=1); print(json.dumps(out, indent=1))

def bench(enc_tag, n=2000):
    """single-thread CPU latency per control tick (all streams: events, audio, vision, her own state).
    real-time factor (RTF) = compute time / tick period; max sustainable rate = ticks/s when the loop runs flat out
    (wall clock incl. Python overhead), and the conservative 1000 / p99 bound."""
    import os
    m, eck = VP.load_model(enc_tag); enc = EventEncoder(m, eck["mu"], eck["sd"])
    try:
        ck = torch.load("models/duplex_tick.pt", weights_only=False); tm = TickModel(); tm.load_state_dict(ck["state"])
    except FileNotFoundError: tm = TickModel()
    ctl = DuplexController(tm, enc); r = np.random.default_rng(0); out = {}
    for load in (0, 1, 5, 20):
        ctl.reset(); lat = []
        tis = []
        for k in range(n):
            t = 1.7e9 + k * TICK_S
            evs = [Event(f"e{k}_{j}", t - r.uniform(0, TICK_S), "other", f"v{r.integers(50)}", "text") for j in range(load)]
            tis.append(TickInput(t, evs, AudioFrame(r.random(), 0, 0, r.random()), VisionFrame(r.random(), 0), SelfState(), 300.0))
        w0 = time.perf_counter()
        for ti in tis:
            s = time.perf_counter(); ctl.step(ti); lat.append((time.perf_counter() - s) * 1000)
        wall = time.perf_counter() - w0
        lat = np.array(lat[50:]); p50, p95, p99 = (float(np.percentile(lat, q)) for q in (50, 95, 99))
        out[f"{load}_events_per_tick"] = dict(p50_ms=p50, p95_ms=p95, p99_ms=p99, max_ms=float(lat.max()),
                                             rtf_at_10hz_p50=p50 / 100, rtf_at_10hz_p95=p95 / 100, rtf_at_5hz_p95=p95 / 200,
                                             max_sustained_hz_flat_out=float(n / wall), max_hz_by_p99=float(1000 / p99),
                                             events_per_s=float(load * 10))
    params = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in tm.parameters())
    rep = dict(encoder=enc_tag, threads=torch.get_num_threads(), cpu_count=_cpu_count(), cpu=_cpu_name(), params_total=params,
               tick_s=TICK_S, n_ticks=n, note="single thread, PyTorch eager, synthetic inputs; 'events_per_tick' = chat/danmaku events arriving per 100 ms tick", results=out)
    json.dump(rep, open("reports/duplex_bench.json", "w"), indent=1); print(json.dumps(rep, indent=1))

def bench_audio(enc_tag, audio_path, n=1500):
    """same as bench() but the audio stream is raw 16 kHz PCM for 2 channels pushed through AudioFrontEnd every tick
    (log-mel + causal conv + GRU), i.e. the full per-tick cost incl. the audio encoder."""
    import os
    from tidal.duplex import AudioFrontEnd
    m, eck = VP.load_model(enc_tag); enc = EventEncoder(m, eck["mu"], eck["sd"])
    try:
        ck = torch.load("models/duplex_tick.pt", weights_only=False); tm = TickModel(); tm.load_state_dict(ck["state"])
    except FileNotFoundError: tm = TickModel()
    afe = AudioFrontEnd(audio_path); ctl = DuplexController(tm, enc, afe); r = np.random.default_rng(0); out = {}
    nsamp = int(TICK_S * 16000)
    for load in (0, 1, 5, 20):
        ctl.reset(); lat = []; alat = []; tis = []
        for k in range(n):
            t = 1.7e9 + k * TICK_S
            evs = [Event(f"e{k}_{j}", t - r.uniform(0, TICK_S), "other", f"v{r.integers(50)}", "text") for j in range(load)]
            amp = 0.1 if (k // 20) % 2 else 0.005
            pcm = ((r.standard_normal(nsamp) * amp).astype(np.float32), (r.standard_normal(nsamp) * 0.003).astype(np.float32))
            tis.append(TickInput(t, evs, None, VisionFrame(r.random(), 0), SelfState(), 300.0, pcm))
        w0 = time.perf_counter()
        for ti in tis:
            s = time.perf_counter(); ctl.step(ti); lat.append((time.perf_counter() - s) * 1000)
        wall = time.perf_counter() - w0
        afe.reset()
        for ti in tis[:300]:
            s = time.perf_counter(); afe(*ti.pcm); alat.append((time.perf_counter() - s) * 1000)
        lat = np.array(lat[50:]); p50, p95, p99 = (float(np.percentile(lat, q)) for q in (50, 95, 99))
        out[f"{load}_events_per_tick"] = dict(p50_ms=p50, p95_ms=p95, p99_ms=p99, max_ms=float(lat.max()), rtf_at_10hz_p95=p95 / 100,
                                             max_sustained_hz_flat_out=float(n / wall), audio_fe_only_p50_ms=float(np.percentile(alat[20:], 50)),
                                             audio_fe_only_p95_ms=float(np.percentile(alat[20:], 95)))
    params = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in tm.parameters()); pa = sum(p.numel() for p in afe.s.m.parameters())
    rep = dict(encoder=enc_tag, audio_fe=os.path.basename(audio_path), threads=torch.get_num_threads(), cpu_count=_cpu_count(), cpu=_cpu_name(),
               params_event_and_tick=params, params_audio_fe=pa, tick_s=TICK_S, n_ticks=n,
               note="single thread, PyTorch eager; per tick: 100 ms x 2 channels of 16 kHz PCM -> log-mel -> audio encoder (5 steps) + event encoder + tick GRU", results=out)
    json.dump(rep, open("reports/duplex_bench_audio.json", "w"), indent=1); print(json.dumps(rep, indent=1))

def bench_shadow(n=2000, reps=3):
    """per-tick latency of the SHADOW config: m3_ablate_m2 + text-emotion event encoder (tidal/shadow/p4.py) + tick GRU,
    without vs with the phase-5 interrupt side head (3 seed heads, mean prob -> ControlOut.p_interrupt).
    Random weights (latency does not depend on them), single thread, eager. Runs are interleaved (off, on, off, on, ...)
    and the per-load median over reps is reported."""
    import os
    from tidal.model import BinHead
    from tidal.shadow import p4
    m = p4.build(dict(interrupt=False, barge=False)); with_head = p4.InterruptHead([BinHead() for _ in range(3)]).eval()
    tm = TickModel(enc_dim=m.vap[0].in_features).eval(); r = np.random.default_rng(0); res = {"off": {}, "on": {}}
    emo = [0.0] * 8 + [0.1, -0.2]
    for load in (0, 1, 5, 20):
        tis = []
        for k in range(n):
            tt = 1.7e9 + k * TICK_S
            evs = [Event(f"e{k}_{j}", tt - r.uniform(0, TICK_S), "other", f"v{r.integers(50)}", "text", emo if j % 2 else None) for j in range(load)]
            tis.append(TickInput(tt, evs, AudioFrame(r.random(), 0, 0, r.random()), VisionFrame(r.random(), 0), SelfState(), 300.0))
        runs = {"off": [], "on": []}
        for _ in range(reps):
            for mode in ("off", "on"):
                m.side_heads = {"p_interrupt": with_head} if mode == "on" else {}
                ctl = DuplexController(tm, EventEncoder(m, np.zeros(FG_NB), np.ones(FG_NB))); lat = []
                for ti in tis:
                    s = time.perf_counter(); out = ctl.step(ti); lat.append((time.perf_counter() - s) * 1000)
                assert (out.p_interrupt is not None) == (mode == "on" and load > 0)
                lat = np.array(lat[50:]); runs[mode].append([float(np.percentile(lat, q)) for q in (50, 95, 99)])
        for mode in runs:
            a = np.median(np.array(runs[mode]), 0)
            res[mode][f"{load}_events_per_tick"] = dict(p50_ms=float(a[0]), p95_ms=float(a[1]), p99_ms=float(a[2]))
    m.side_heads = {}
    rep = dict(config="shadow: m3_ablate_m2 + text emotion (28 inputs), tick GRU enc_dim=128; interrupt head = 3 x BinHead(128-64-1)",
               threads=torch.get_num_threads(), cpu_count=_cpu_count(), cpu=_cpu_name(), tick_s=TICK_S, n_ticks=n, reps=reps,
               params_encoder=sum(p.numel() for p in m.parameters()), params_tick=sum(p.numel() for p in tm.parameters()),
               params_interrupt_head=sum(p.numel() for p in with_head.parameters()),
               note="random weights, single thread, PyTorch eager, synthetic inputs; median of interleaved reps", without_head=res["off"], with_head=res["on"])
    json.dump(rep, open("reports/duplex_bench_shadow.json", "w"), indent=1); print(json.dumps(rep, indent=1))


def bench_barge(n=600, reps=2):
    """Tick latency of the shadow config with the interrupt head, before vs after the forward barge head (3 x BinHead 132).
    Random weights, single thread, eager. Load 20 uses fewer ticks (the per-tick cost dominates)."""
    import os
    from tidal.model import BinHead
    from tidal.shadow import p4
    m = p4.build(dict(barge=False))
    m.side_heads = {"p_interrupt": p4.InterruptHead([BinHead() for _ in range(3)]).eval()}
    barge = p4.BargeHead([BinHead(d=132) for _ in range(3)]).eval()
    tm = TickModel(enc_dim=m.vap[0].in_features).eval(); r = np.random.default_rng(1); res = {"before": {}, "after": {}}
    emo = [0.0] * 8 + [0.1, -0.2]
    for load, nn in ((0, n), (1, n), (5, n), (20, 200)):
        tis = []
        for k in range(nn):
            tt = 1.7e9 + k * TICK_S
            evs = [Event(f"e{k}_{j}", tt - r.uniform(0, TICK_S), "other", f"v{r.integers(50)}", "text", emo if j % 2 else None) for j in range(load)]
            tis.append(TickInput(tt, evs, AudioFrame(r.random(), 0, 0, r.random()), VisionFrame(r.random(), 0), SelfState(), 300.0))
        runs = {"before": [], "after": []}
        for _ in range(reps):
            for mode in ("before", "after"):
                object.__setattr__(m, "barge_head", barge if mode == "after" else None)
                ctl = DuplexController(tm, EventEncoder(m, np.zeros(FG_NB), np.ones(FG_NB))); lat = []
                for ti in tis:
                    s = time.perf_counter(); out = ctl.step(ti); lat.append((time.perf_counter() - s) * 1000)
                assert (out.p_barge is not None) == (mode == "after" and load > 0)
                lat = np.array(lat[20:]); runs[mode].append([float(np.percentile(lat, q)) for q in (50, 95)])
        for mode in runs:
            a = np.median(np.array(runs[mode]), 0)
            res[mode][f"{load}_events_per_tick"] = dict(p50_ms=float(a[0]), p95_ms=float(a[1]), n_ticks=nn)
    object.__setattr__(m, "barge_head", None)
    rep = dict(config="shadow m3_ablate_m2 + text emotion + 3-seed interrupt head; before = no barge head, after = 3 x BinHead(132)",
               threads=torch.get_num_threads(), cpu=_cpu_name(), tick_s=TICK_S, reps=reps,
               note="random weights, single thread, eager, synthetic inputs; median of interleaved reps", **res)
    json.dump(rep, open("reports/duplex_bench_barge.json", "w"), indent=1); print(json.dumps(rep, indent=1))

def _cpu_name():
    try: return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception: return None

def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train(sys.argv[2])
    elif cmd == "eval": evaluate()
    elif cmd == "bench": bench(sys.argv[2])
    elif cmd == "bench_audio": bench_audio(sys.argv[2], sys.argv[3])
    elif cmd == "bench_shadow": bench_shadow()
    elif cmd == "bench_barge": bench_barge()
    elif cmd == "show":
        ticks, y, meta = session(int(sys.argv[2])); print(len(y), {a: int((y == i).sum()) for a, i in A.items()}, meta["voice"], meta["rate"])
