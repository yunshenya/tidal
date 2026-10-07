"""Per-decision latency on this box's CPU (1 thread): bge int8 encode of the new message + ONNX temporal model over 64 events.
Run in a fresh process so RSS reflects only the deployed pieces. usage: python -m tidal.bench TAG"""
import sys, time, json, resource, os, numpy as np, onnxruntime as ort
def main(tag):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    s = ort.InferenceSession(f"models/{tag}.onnx", so, providers=["CPUExecutionProvider"])
    ins = {i.name: i.shape for i in s.get_inputs()}; use_text = "text" in ins; nf = ins["feats"][2]
    from tidal.embed import Encoder
    from tidal.config import BOT_NAMES
    enc = Encoder(threads=1) if use_text else None
    rng = np.random.default_rng(0)
    texts = ["哈哈哈哈哈", "今晚谁上号", "我跟你说，然后那个", BOT_NAMES[0] + "你觉得呢？", "等下我先去吃个饭", "[图片]", "这个版本的新角色强度怎么样啊有没有人抽了"]
    F = rng.standard_normal((1, 64, nf)).astype(np.float32); V = np.ones((1, 64), bool); E = np.zeros((1, 64, 512), np.float32)
    lat_m, lat_e, lat_tot = [], [], []
    for k in range(1100):
        t0 = time.perf_counter()
        if use_text:
            e = enc.encode([texts[k % len(texts)]], bs=1); E = np.roll(E, -1, 1); E[0, -1] = e[0]
        t1 = time.perf_counter()
        feed = {"feats": F, "valid": V}
        if use_text: feed["text"] = E
        s.run(None, feed); t2 = time.perf_counter()
        if k >= 100: lat_e.append((t1 - t0) * 1e3); lat_m.append((t2 - t1) * 1e3); lat_tot.append((t2 - t0) * 1e3)
    pc = lambda v: dict(p50=round(float(np.percentile(v, 50)), 3), p95=round(float(np.percentile(v, 95)), 3), p99=round(float(np.percentile(v, 99)), 3))
    res = dict(tag=tag, threads=1, model_ms=pc(lat_m), encoder_ms=pc(lat_e) if use_text else None, decision_ms=pc(lat_tot),
               onnx_bytes=os.path.getsize(f"models/{tag}.onnx"), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               rss_before_sessions_mb=round(rss0, 1), cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":", 1)[1].strip(), n_cpu=os.cpu_count())
    json.dump(res, open(f"reports/bench_{tag}.json", "w"), indent=1); print(json.dumps(res, indent=1))
if __name__ == "__main__":
    main(sys.argv[1])
