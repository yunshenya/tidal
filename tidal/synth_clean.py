"""Validate + dedupe synthetic generations -> data/synth/clean.jsonl, stats -> data/synth/clean_stats.json"""
import json, hashlib, re, collections, numpy as np
ALLOWED = {"addr_bot", "unfinished", "self_cont", "interrupt", "side", "eot"}
def norm(t): return re.sub(r"\s+|[，。！？!?,.~～…]+", "", t)
def validate(c):
    msgs = c.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 20: return None, "too_short"
    out = []
    for m in msgs:
        if not isinstance(m, dict): return None, "bad_msg"
        s, t, dt = m.get("s"), m.get("text"), m.get("dt")
        if not isinstance(s, str) or not isinstance(t, str) or not t.strip(): return None, "bad_field"
        try: dt = float(dt)
        except Exception: return None, "bad_dt"
        if not (0 <= dt <= 7200): return None, "dt_range"
        tags = [g for g in (m.get("tags") or []) if isinstance(g, str) and (g in ALLOWED or re.fullmatch(r"reply_to:\d+", g))]
        out.append(dict(s=s.strip(), dt=dt, text=t.strip()[:500], tags=tags))
    out[0]["dt"] = 0.0
    humans = {m["s"] for m in out if m["s"] != "BOT"}
    if len(humans) < 2: return None, "few_humans"
    if not any(m["s"] == "BOT" for m in out): return None, "no_bot"
    if sum(m["s"] == "BOT" for m in out) > 0.6 * len(out): return None, "bot_dominates"
    return out, "ok"
def main():
    st = collections.Counter(); kept = []; seen_hash = set(); shingles = []
    for line in open("data/synth/raw_generations.jsonl"):
        r = json.loads(line)
        if "error" in r: st["api_error"] += 1; continue
        try: c = json.loads(r["content"])
        except Exception: st["json_error"] += 1; continue
        msgs, why = validate(c); st[why] += 1
        if msgs is None: continue
        h = hashlib.sha1("\n".join(m["text"] for m in msgs).encode()).hexdigest()
        if h in seen_hash: st["dup_exact"] += 1; st["ok"] -= 1; continue
        sh = {norm(m["text"]) for m in msgs if len(norm(m["text"])) >= 4}
        if any(len(sh & o) / max(1, len(sh | o)) >= 0.3 for o in shingles): st["dup_near"] += 1; st["ok"] -= 1; continue
        seen_hash.add(h); shingles.append(sh)
        kept.append(dict(id=h[:12], spec=r["spec"], messages=msgs))
    with open("data/synth/clean.jsonl", "w") as f:
        for c in kept: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_msgs = sum(len(c["messages"]) for c in kept)
    tags = collections.Counter(g.split(":")[0] for c in kept for m in c["messages"] for g in m["tags"])
    stats = dict(validation=dict(st), conversations=len(kept), messages=n_msgs,
                 bot_messages=sum(m["s"] == "BOT" for c in kept for m in c["messages"]), tag_counts=dict(tags),
                 dt_median=float(np.median([m["dt"] for c in kept for m in c["messages"][1:]])))
    json.dump(stats, open("data/synth/clean_stats.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False, indent=1))
if __name__ == "__main__": main()
