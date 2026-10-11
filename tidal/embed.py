"""Frozen bge-small-zh-v1.5 sentence embeddings (int8 ONNX, CPU), cached by sha1(text).
Cache: data/emb/cache.npz  {keys: [sha1...], emb: float16 [N,512]}  (L2-normalized CLS pooling)"""
import hashlib, time, os, json, numpy as np, onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer
MODEL_DIR = "models/bge"; CACHE = "data/emb/cache.npz"; MAXLEN = 64
def key(t): return hashlib.sha1(t.encode()).hexdigest()
class Encoder:
    def __init__(self, threads=4, model="bge_q.onnx"):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(f"{MODEL_DIR}/{model}", so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json"); self.tok.enable_truncation(MAXLEN)
        self.ins = [i.name for i in self.s.get_inputs()]
    def encode(self, texts, bs=64):
        out = np.zeros((len(texts), 512), np.float32)
        order = np.argsort([len(t) for t in texts])          # length bucketing
        for b in range(0, len(texts), bs):
            idx = order[b:b + bs]; encs = self.tok.encode_batch([texts[i] for i in idx])
            L = max(len(e.ids) for e in encs)
            ids = np.zeros((len(idx), L), np.int64); am = np.zeros_like(ids)
            for r, e in enumerate(encs): ids[r, :len(e.ids)] = e.ids; am[r, :len(e.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": am}
            if "token_type_ids" in self.ins: feed["token_type_ids"] = np.zeros_like(ids)
            h = self.s.run(None, feed)[0][:, 0]                 # CLS pooling (bge)
            out[idx] = h / np.linalg.norm(h, axis=1, keepdims=True)
        return out
def load_cache(cache=CACHE):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False); return dict(zip(z["keys"].tolist(), range(len(z["keys"])))), z["emb"]
    return {}, np.zeros((0, 512), np.float16)
def ensure(texts, threads=4, bs=64, workers=1, cache=CACHE):
    """Embed any texts not yet cached; returns (key->row dict, emb matrix fp16) and throughput stats."""
    k2i, emb = load_cache(cache)
    todo = sorted({t for t in texts if t and key(t) not in k2i})
    stats = {}
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        from threading import local
        state = local(); t0 = time.perf_counter(); keys = list(k2i.keys())
        def encode_chunk(chunk):
            if not hasattr(state, "encoder"): state.encoder = Encoder(threads)
            return state.encoder.encode(chunk, bs=bs).astype(np.float16)
        chunks = [todo[i:i + 4096] for i in range(0, len(todo), 4096)]
        target = Path(cache); target.parent.mkdir(parents=True, exist_ok=True)
        done = 0; pending = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk, encoded in zip(chunks, pool.map(encode_chunk, chunks)):
                keys.extend(key(t) for t in chunk); pending.append(encoded); done += len(chunk)
                if len(pending) >= 4 or done == len(todo):
                    emb = np.concatenate([emb, *pending]); pending = []
                    temp = target.with_suffix(".tmp.npz")
                    np.savez(temp, keys=np.array(keys), emb=emb); temp.replace(target)
                    print(f"[embed] {done}/{len(todo)} new texts in {time.perf_counter() - t0:.1f}s", flush=True)
        dt = time.perf_counter() - t0
        stats = dict(new_texts=len(todo), seconds=round(dt, 2), msgs_per_s=round(len(todo) / dt, 1), threads=threads,
                     batch_size=bs, workers=workers)
        k2i = dict(zip(keys, range(len(keys))))
    return k2i, emb, stats
def bench_single(n=300, threads=1):
    from tidal.config import BOT_NAMES
    enc = Encoder(threads); texts = ["今天晚上一起打游戏吗", "哈哈哈哈", "我跟你说，然后那个", BOT_NAMES[0] + "你觉得呢？"] * (n // 4)
    enc.encode(texts[:4], bs=1); lat = []
    for t in texts:
        t0 = time.perf_counter(); enc.encode([t], bs=1); lat.append((time.perf_counter() - t0) * 1000)
    return dict(threads=threads, p50_ms=round(float(np.percentile(lat, 50)), 2), p95_ms=round(float(np.percentile(lat, 95)), 2))
