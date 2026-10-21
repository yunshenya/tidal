"""v22: retrain contrastive q/k adapters with hard long-stream negatives.
Keeps same Retr architecture as v20 (256→256 GELU→128). Caching frozen encodings.
Negatives: K=50–100 fact pools; same-name/diff-rel; same-rel/diff-name; update pairs prefer NEW.
"""
import argparse, random, time, torch, torch.nn as nn, torch.nn.functional as F
import data_v12 as D
from neuromem_v13 import build

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="mem_ckpt_v13A2s.pt")
p.add_argument("--init", default="retr_v20.pt")  # warm-start
p.add_argument("--out", default="retr_v22.pt")
p.add_argument("--dr", type=int, default=128); p.add_argument("--hidden", type=int, default=256)
p.add_argument("--steps", type=int, default=2500); p.add_argument("--bs", type=int, default=24)
p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--temp", type=float, default=0.07)
p.add_argument("--n_long", type=int, default=400); p.add_argument("--n_sess", type=int, default=600)
p.add_argument("--seed", type=int, default=1)
args = p.parse_args(); torch.set_num_threads(8); torch.manual_seed(args.seed); random.seed(args.seed)

m, tok, _ = build("HuggingFaceTB/SmolLM2-135M", dv=576, nslots=1, vpool="ans", freeze_inj=True, norm_v=False)
m.load_state_dict(torch.load(args.ckpt), strict=False); m.eval()
for p_ in m.parameters(): p_.requires_grad_(False)
dk = m.dk

@torch.no_grad()
def enc_k(t):
    h, am, sp = m.hidden([t["text"]], [t.get("n")])
    return m.key(h, am, sp)[0].float()
@torch.no_grad()
def enc_q(t):
    h, am, sp = m.hidden([t["q"]], [t["n"]]); return m.qvec(h, am, sp)[0].float()

print("caching long pools + sessions…", flush=True)
t0 = time.time(); cache = []; rng = random.Random(args.seed)

def pack_encoded(facts, apply_overwrite=True):
    """Encode facts; optionally collapse same (n,rel) keeping latest. Returns list of dicts with k,q,n,rel."""
    encoded = []
    for t in facts:
        k = enc_k(t); q = enc_q(t)
        e = dict(k=k, q=q, n=t["n"], rel=t["rel"], a=t["a"])
        if apply_overwrite:
            hit = next((j for j, x in enumerate(encoded) if (x["n"], x["rel"]) == (t["n"], t["rel"])), None)
            if hit is not None: encoded[hit] = e
            else: encoded.append(e)
        else:
            encoded.append(e)
    return encoded

# A) long fact-only streams K in {40,60,80,100}
for i in range(args.n_long):
    K = rng.choice([40, 50, 60, 80, 100])
    facts = [D.fact(rng, "train") for _ in range(K)]
    # inject 1–3 updates
    for _ in range(rng.randint(1, 3)):
        A = rng.choice(facts)
        B = D.fact(rng, "train", n=A["n"], rel=A["rel"], c=A["c"])
        while D.norm(B["a"]) == D.norm(A["a"]): B = D.fact(rng, "train", n=A["n"], rel=A["rel"], c=A["c"])
        facts.append(B)
    enc = pack_encoded(facts, apply_overwrite=True)
    if len(enc) < 8: continue
    Ks = torch.stack([e["k"] for e in enc]); Qs = torch.stack([e["q"] for e in enc])
    meta = [(e["n"], e["rel"]) for e in enc]
    cache.append(dict(kind="long", K=Ks, Q=Qs, meta=meta, n=len(enc)))
    if (i + 1) % 50 == 0: print(f"  long {i+1}/{args.n_long} kept={len(cache)} {time.time()-t0:.0f}s", flush=True)

# B) mixed sessions with hard negs + updates
for i in range(args.n_sess):
    N = rng.choice([4, 8, 16, 24, 32])
    s = D.session(rng, "train", N)
    facts = [t for t in s if t["kind"] == "fact"]
    # ensure hard negs: add same-name diff-rel and same-rel diff-name
    if facts:
        A = rng.choice(facts)
        facts.append(D.fact(rng, "train", n=A["n"]))  # same name, likely different rel
        facts.append(D.fact(rng, "train", rel=A["rel"]))  # same rel, different name
        if rng.random() < 0.5:
            B = D.fact(rng, "train", n=A["n"], rel=A["rel"], c=A["c"])
            while D.norm(B["a"]) == D.norm(A["a"]): B = D.fact(rng, "train", n=A["n"], rel=A["rel"], c=A["c"])
            facts.append(B)
    enc = pack_encoded(facts, apply_overwrite=True)
    if len(enc) < 3: continue
    Ks = torch.stack([e["k"] for e in enc]); Qs = torch.stack([e["q"] for e in enc])
    meta = [(e["n"], e["rel"]) for e in enc]
    cache.append(dict(kind="sess", K=Ks, Q=Qs, meta=meta, n=len(enc)))
    if (i + 1) % 100 == 0: print(f"  sess {i+1}/{args.n_sess} kept={len(cache)} {time.time()-t0:.0f}s", flush=True)

print(f"cache size {len(cache)} in {time.time()-t0:.0f}s", flush=True)

class Retr(nn.Module):
    def __init__(self, din, dout, hid):
        super().__init__()
        self.wk = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, dout))
        self.wq = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, dout))
    def fk(self, x): return F.normalize(self.wk(x), dim=-1)
    def fq(self, x): return F.normalize(self.wq(x), dim=-1)

retr = Retr(dk, args.dr, args.hidden)
if args.init:
    ck = torch.load(args.init, map_location="cpu")
    retr.load_state_dict(ck["state"]); print("warm-start", args.init, flush=True)
opt = torch.optim.AdamW(retr.parameters(), lr=args.lr, weight_decay=1e-4)

def sample_loss():
    # bias toward long streams
    long = [c for c in cache if c["kind"] == "long"]
    ex = random.choice(long if random.random() < 0.7 and long else cache)
    K, Q, n = ex["K"], ex["Q"], ex["n"]
    i = random.randrange(n)
    qe = retr.fq(Q[i:i+1]); ke = retr.fk(K)
    logits = (qe @ ke.T) / args.temp
    loss = F.cross_entropy(logits, torch.tensor([i]))
    # hard-neg emphasis: if there are same-name or same-rel distractors, add focused CE over that subset+pos
    ni, ri = ex["meta"][i]
    hard_idx = [j for j, (n2, r2) in enumerate(ex["meta"]) if j != i and (n2 == ni or r2 == ri)]
    if hard_idx and random.random() < 0.5:
        idxs = [i] + hard_idx[:16]
        sub = ke[idxs]; logits_h = (qe @ sub.T) / args.temp
        loss = loss + 0.5 * F.cross_entropy(logits_h, torch.tensor([0]))
    acc = int(logits.argmax() == i)
    return loss, acc

log = []
for step in range(1, args.steps + 1):
    retr.train(); loss_sum = 0.0; nacc = 0
    for _ in range(args.bs):
        loss, acc = sample_loss(); loss_sum = loss_sum + loss; nacc += acc
    loss = loss_sum / args.bs; opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 1:
        retr.eval(); va = vn = 0; vaL = vnL = 0
        with torch.no_grad():
            for ex in cache[-80:]:
                K, Q, n = ex["K"], ex["Q"], ex["n"]
                qe = retr.fq(Q); ke = retr.fk(K)
                pred = (qe @ ke.T).argmax(1)
                hit = int((pred == torch.arange(n)).sum()); va += hit; vn += n
                if ex["kind"] == "long": vaL += hit; vnL += n
            # frozen baseline on same long tail
            fb = fl = 0; fn = fnL = 0
            for ex in cache[-80:]:
                K, Q, n = ex["K"], ex["Q"], ex["n"]
                Kn = F.normalize(K, dim=-1); Qn = F.normalize(Q, dim=-1)
                pred = (Qn @ Kn.T).argmax(1)
                hit = int((pred == torch.arange(n)).sum()); fb += hit; fn += n
                if ex["kind"] == "long": fl += hit; fnL += n
        msg = dict(step=step, loss=float(loss), train_acc=nacc/args.bs,
                   val_ret1=va/max(1,vn), val_long=vaL/max(1,vnL),
                   frozen=fb/max(1,fn), frozen_long=fl/max(1,fnL))
        log.append(msg); print(msg, flush=True)

torch.save(dict(state=retr.state_dict(), args=vars(args), log=log), args.out)
print("saved", args.out, flush=True)
