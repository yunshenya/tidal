"""One shadow-mode tick: pull -> pseudonymize -> upsert events -> predict new events -> label settled events.
Idempotent (upserts keyed on pseudonymous msg ids; a prediction is written once, at first sight) and guarded by a
file lock so overlapping cron ticks are no-ops. Makes no calls to any LLM / external API.
usage: python -m tidal.shadow.run [--source NAME|module.path] [--no-pull]   (default: config `shadow_source`, else jsonl)"""
import argparse, json, os, sys, time, uuid
import numpy as np, pandas as pd
from tidal.privacy import pseudo
from tidal.modalities import meta as media
from tidal.shadow import store as S
from tidal.config import get
from tidal.shadow.lock import try_lock

SETTLE_S = float(os.environ.get("TIDAL_SHADOW_SETTLE_S", 3600))     # label an event once this much time has passed
CONTEXT_S = 7 * 24 * 3600.0                                          # history loaded as model context
MAX_BACKFILL_S = 7 * 24 * 3600.0                                     # never predict events older than this
LABEL_VERSION = "phase1-heuristics+horizon"
EVCOLS = ["msg_id", "conv", "conv_type", "ts", "role", "speaker", "speaker_known", "text", "attention_reason", "chosen_action",
          "cur_addressed", "cur_reply"]

def _nn(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (bool, np.bool_)): return int(v)
    return v

def upsert_events(c, ev, now):
    n_new = n_upd = 0
    for r in ev.to_dict("records"):
        info = media.extract(r.get("text"))
        refs = [pseudo("media", x) for x in info["refs"]] + list(r.get("media_refs_extra") or [])
        kinds = set(info["kinds"]); counts = dict(info["counts"])
        if r.get("media_refs_extra"): kinds.add("sticker"); counts["sticker"] = max(1, counts.get("sticker", 0))
        txt = _nn(r.get("text"))
        vals = dict({k: _nn(r.get(k)) for k in EVCOLS}, media_kinds=json.dumps(sorted(kinds)), media_counts=json.dumps(counts),
                    media_refs=json.dumps(sorted(set(refs))), media_duration_s=info["duration_s"],
                    text_chars=info["text_chars"] if txt is not None else None,
                    **{k: _nn(r.get(k)) for k in ["msg_chars", "has_question", "mentioned_bot", "reply_to_bot", "burst_len", "group_activity_1m"]})
        old = c.execute("select text, first_seen_at, text_seen_at from events where msg_id=?", (vals["msg_id"],)).fetchone()
        if old is None:
            vals.update(first_seen_at=now, text_seen_at=now if txt is not None else None, updated_at=now); n_new += 1
        else:
            vals.update(first_seen_at=old["first_seen_at"],
                        text_seen_at=old["text_seen_at"] or (now if txt is not None else None), updated_at=now)
            # a later rebuild may lose text that an earlier one had (window edge) -> never downgrade
            if txt is None and old["text"] is not None:
                for k in ("text", "speaker", "speaker_known", "media_kinds", "media_counts", "media_refs", "text_chars"): vals.pop(k, None)
            n_upd += 1
        cols = list(vals)
        c.execute(f"insert into events({','.join(cols)}) values({','.join('?' * len(cols))}) "
                  f"on conflict(msg_id) do update set {','.join(f'{k}=excluded.{k}' for k in cols if k not in ('msg_id', 'first_seen_at'))}",
                  [vals[k] for k in cols])
    return n_new, n_upd

def load_frame(c, convs, lo):
    q = f"select * from events where conv in ({','.join('?' * len(convs))}) and ts >= ? order by conv, ts"
    d = pd.read_sql_query(q, c, params=list(convs) + [lo])
    d["speaker_known"] = d.speaker_known.fillna(0).astype(bool)
    d["text"] = d.text.astype(object).where(d.text.notna(), None)
    d["speaker"] = d.speaker.astype(object).where(d.speaker.notna(), None)
    return d

def predict(c, now, run_id, stats):
    pa = S.get_meta(c, "predict_after"); lo = max(pa, now - MAX_BACKFILL_S)
    tT = [r[0] for r in c.execute("select e.msg_id from events e left join predictions p on p.msg_id=e.msg_id and p.regime='T' "
                                  "where e.ts > ? and e.ts <= ? and p.msg_id is null", (lo, now))]
    tTS = [r[0] for r in c.execute("select e.msg_id from events e left join predictions p on p.msg_id=e.msg_id and p.regime='TS' "
                                   "where e.ts > ? and e.ts <= ? and e.text is not null and e.text != '' and p.msg_id is null", (lo, now))]
    from tidal.shadow.infer import p2_cfg
    p2 = p2_cfg(); tP2 = []
    if p2:   # phase-2 model = separate system; its own targets (it may backfill events phase 1 already predicted)
        tP2 = [r[0] for r in c.execute("select e.msg_id from events e left join predictions p on p.msg_id=e.msg_id and p.system=? "
                                       "where e.ts > ? and e.ts <= ? and p.msg_id is null", ("model:" + p2["tag"], lo, now))]
    stats.update(targets_T=len(tT), targets_TS=len(tTS), targets_P2=len(tP2))
    if not tT and not tTS and not tP2: return
    ids = list(set(tT) | set(tTS) | set(tP2))
    info = pd.read_sql_query(f"select msg_id, conv, ts from events where msg_id in ({','.join('?' * len(ids))})", c, params=ids)
    d = load_frame(c, sorted(info.conv.unique()), info.ts.min() - CONTEXT_S)
    from tidal.shadow.infer import Predictor, ready
    why = ready()
    if why: stats["predict_skipped"] = why; return
    P = Predictor(c); preds = P.predict(d, tT, tTS, tP2); stats['texts_embedded'] = P.n_encoded
    ets = dict(zip(info.msg_id, info.ts)); ver = json.dumps(dict(P.cfg["models"], P2=P.p2["tag"]) if P.p2 else P.cfg["models"])
    c.executemany("insert or ignore into predictions values(?,?,?,?,?,?,?,?,?)",
                  [(m, r, s, json.dumps(p), ets[m], now, now - ets[m], run_id, ver) for m, r, s, p in preds])
    stats["predictions_written"] = len(preds)

def label(c, now, stats):
    from tidal.adapters import real_frame
    from tidal.labels import compute_labels
    pa = S.get_meta(c, "predict_after")
    todo = pd.read_sql_query("select e.msg_id, e.conv, e.ts from events e left join labels l on l.msg_id=e.msg_id "
                             "where e.ts > ? and e.ts <= ? and l.msg_id is null", c, params=(pa, now - SETTLE_S))
    stats["labels_due"] = len(todo)
    if not len(todo): return
    d = load_frame(c, sorted(todo.conv.unique()), todo.ts.min() - 3600)
    f = real_frame(d=d.assign(attention=None, delivered=None, salience=None, replay_t=None))
    f["conv_type"] = d.conv_type.values
    L = compute_labels(f, horizon=now - 60)
    L = L[L.msg_id.isin(set(todo.msg_id))]
    rows = [(r.msg_id, r.conv, r.conv_type, r.ts, *[_nn(getattr(r, h)) for h in ["y_eot", "y_self", "y_addr", "y_act", "y_recheck", "y_hreply"]],
             int(r.speaker_known), now, LABEL_VERSION) for r in L.itertuples()]
    c.executemany("insert or ignore into labels values(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    stats["labels_written"] = len(rows)


def _maxrss_mb():
    """Peak RSS in MiB, or None where the Unix rlimit API does not exist (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":   # bytes on macOS, KiB on Linux
        rss /= 1024
    return round(rss / 1024, 1)

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default=os.environ.get("TIDAL_SHADOW_SOURCE") or get("shadow_source", "jsonl"))
    ap.add_argument("--no-pull", action="store_true"); a = ap.parse_args(argv)
    S.STATE.mkdir(parents=True, exist_ok=True)
    lockf = try_lock(S.STATE / "run.lock")
    if lockf is None:
        print(time.strftime("%F %T"), "another run holds the lock; skipping"); return 0
    now = time.time(); run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    c = S.connect(); stats = {}; status = "ok"
    c.execute("insert into runs values(?,?,?,?,?)", (run_id, now, None, "running", None)); c.commit()
    try:
        from tidal.shadow.sources import load
        src = load(a.source)
        if S.get_meta(c, "predict_after") is None:
            S.set_meta(c, "predict_after", src.default_predict_after(now)); S.set_meta(c, "source", a.source); c.commit()
        if not a.no_pull:
            t0 = time.time(); ev = src.fetch(c, now); stats["pull_s"] = round(time.time() - t0, 2)
            stats["fetched_events"] = len(ev); stats["pulled_rows"] = S.get_meta(c, "source_last_pull")
            stats["events_new"], stats["events_updated"] = upsert_events(c, ev, now); c.commit()
        t0 = time.time(); predict(c, now, run_id, stats); c.commit(); stats["predict_s"] = round(time.time() - t0, 2)
        t0 = time.time(); label(c, now, stats); c.commit(); stats["label_s"] = round(time.time() - t0, 2)
        stats["totals"] = dict(events=c.execute("select count(*) from events").fetchone()[0],
                               predicted_events=c.execute("select count(distinct msg_id) from predictions").fetchone()[0],
                               predictions=c.execute("select count(*) from predictions").fetchone()[0],
                               labeled=c.execute("select count(*) from labels").fetchone()[0])
    except Exception as e:
        status = "error"; stats["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    stats["maxrss_mb"] = _maxrss_mb()
    c.execute("update runs set finished_at=?, status=?, stats=? where run_id=?", (time.time(), status, json.dumps(stats), run_id)); c.commit()
    print(time.strftime("%F %T"), run_id, status, json.dumps(stats, ensure_ascii=False), flush=True)
    return 0 if status == "ok" else 1

if __name__ == "__main__":
    sys.exit(main())
