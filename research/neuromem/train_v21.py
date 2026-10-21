"""v21: train residual inject MLP on frozen A2s values.
Frozen backbone + scale + identity inj. add = scale(v) + MLP(scale(v)).
Teacher-force CE on answer tokens given query, with correct slot value injected.
"""
import argparse, json, random, time, torch, torch.nn as nn, torch.nn.functional as F
import data_v12 as D
from neuromem_v13 import build

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="mem_ckpt_v13A2s.pt")
p.add_argument("--out", default="inj_v21.pt")
p.add_argument("--hid", type=int, default=576)
p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--n_cache", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
p.add_argument("--neg_p", type=float, default=0.2)  # fraction of wrong-v contrast steps
args = p.parse_args(); torch.set_num_threads(8); torch.manual_seed(args.seed); random.seed(args.seed)

m, tok, _ = build("HuggingFaceTB/SmolLM2-135M", dv=576, nslots=1, vpool="ans", freeze_inj=True, norm_v=False)
m.load_state_dict(torch.load(args.ckpt), strict=False); m.eval()
for p_ in m.parameters(): p_.requires_grad_(False)
EOS = tok.eos_token_id

@torch.no_grad()
def enc_v(t):
    h, am, sp = m.hidden([t["text"]], [t.get("n")])
    vp = m.vpool_h(h, am, [t["text"]], [t.get("n")], [t.get("a")])[0]
    v = m.Wv(vp); v = v[0] if v.dim() == 2 else v
    return v.float()

print("caching (v, q, a)…", flush=True)
t0 = time.time(); cache = []; rng = random.Random(args.seed)
for i in range(args.n_cache):
    t = D.fact(rng, "train")
    v = enc_v(t)
    # also store answer-only pooled hidden as optional aux target
    ans_txt = f"Answer: {t['a']}"
    h2, am2, _ = m.hidden([ans_txt], [None])
    b = tok([ans_txt], return_tensors="pt", return_offsets_mapping=True)
    off = b["offset_mapping"][0].tolist(); j = ans_txt.find(t["a"])
    mask = torch.tensor([float(bb > st and st < j + len(t["a"]) and bb > j) for st, bb in off]) * b["attention_mask"][0].float()
    den = mask.sum().clamp(min=1.0)
    vans = ((h2[0] * mask.unsqueeze(-1)).sum(0) / den).float()
    cache.append(dict(v=v, vans=vans, q=t["q"], a=t["a"], rel=t["rel"], n=t["n"]))
    if (i + 1) % 400 == 0: print(f"  {i+1}/{args.n_cache} {time.time()-t0:.0f}s", flush=True)
print(f"cache {len(cache)} in {time.time()-t0:.0f}s", flush=True)

class InjMLP(nn.Module):
    def __init__(self, d=576, h=576):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, v):
        # residual only; caller adds to identity inj(scale(v))
        return self.net(v)

inj = InjMLP(m.dv, args.hid)
opt = torch.optim.AdamW(inj.parameters(), lr=args.lr, weight_decay=1e-4)

def make_add(v):
    # match eval path: scale then (identity) inj, plus residual MLP
    vs = m.scale(v.view(1, -1))
    return m.inj(vs) + inj(vs)  # inj is eye; equiv vs + mlp(vs)

def ce_for(v, q, a):
    # teacher-force answer tokens after query
    q_ids = tok([q], return_tensors="pt")["input_ids"][0]
    a_ids = tok([" " + a], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    # strip leading space quirk: prefer encoding answer alone if query ends with Answer:
    if q.rstrip().endswith("Answer:"):
        a_ids = tok([a], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # keep a leading space token if tokenizer would insert one in free gen after Answer:
        sp = tok([" " + a], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        a_ids = sp
    ids = torch.cat([q_ids, a_ids]).unsqueeze(0)
    qlen = q_ids.numel()
    add = make_add(v)
    m._inject = lambda x: add.unsqueeze(1)
    try:
        logits = m.base(input_ids=ids).logits
    finally:
        m._inject = None
    # predict answer tokens: positions qlen-1 .. -2 -> targets a_ids
    pred = logits[0, qlen - 1: qlen - 1 + a_ids.numel()]
    return F.cross_entropy(pred, a_ids)

@torch.no_grad()
def gen_ok(v, q, a, maxn=10):
    add = make_add(v)
    ids = tok([q], return_tensors="pt")["input_ids"]; out = []
    m._inject = lambda x: add.unsqueeze(1)
    try:
        o = m.base(input_ids=ids, use_cache=True); pkv = o.past_key_values
        for _ in range(maxn):
            nx = int(o.logits[0, -1].argmax()); s = tok.decode(out + [nx])
            if nx == EOS or "\n" in s: break
            out.append(nx)
            if s.rstrip().endswith((".", "?", "!")) and len(s.strip()) > 1: break
            o = m.base(input_ids=torch.tensor([[nx]]), past_key_values=pkv, use_cache=True); pkv = o.past_key_values
    finally:
        m._inject = None
    return D.match(tok.decode(out), a)

# baseline gen on first 64 cache
base_ok = sum(gen_ok(cache[i]["v"], cache[i]["q"], cache[i]["a"]) for i in range(64)) / 64
print("baseline_N0_proxy", base_ok, flush=True)

log = []
for step in range(1, args.steps + 1):
    inj.train(); losses = []
    for _ in range(args.bs):
        ex = cache[random.randrange(len(cache))]
        loss = ce_for(ex["v"], ex["q"], ex["a"])
        if random.random() < args.neg_p:
            # wrong value: push CE up via hinge against a correct-ish margin
            other = cache[random.randrange(len(cache))]
            while other["a"] == ex["a"]: other = cache[random.randrange(len(cache))]
            loss_neg = ce_for(other["v"], ex["q"], ex["a"])
            loss = loss + F.relu(2.0 - loss_neg) * 0.3
        losses.append(loss)
    loss = torch.stack(losses).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 1:
        inj.eval()
        with torch.no_grad():
            ok = sum(gen_ok(cache[i]["v"], cache[i]["q"], cache[i]["a"]) for i in range(64)) / 64
            # held-outish: last 64 of cache (still train split)
            ok2 = sum(gen_ok(cache[-64 + i]["v"], cache[-64 + i]["q"], cache[-64 + i]["a"]) for i in range(64)) / 64
        msg = dict(step=step, loss=float(loss), gen64=ok, gen_tail64=ok2)
        log.append(msg); print(msg, flush=True)

inj.eval()
torch.save(dict(state=inj.state_dict(), args=vars(args), log=log, baseline_N0_proxy=base_ok), args.out)
print("saved", args.out, flush=True)
