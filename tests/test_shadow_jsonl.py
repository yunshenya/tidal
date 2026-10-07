"""End-to-end shadow tick on a synthetic JSONL stream (no network, no real data)."""
import json, os, sqlite3, time
from tidal.shadow import run, store as S

def test_tick_is_idempotent(tmp_path):
    t0 = time.time() - 4 * 3600
    ev = []
    for i in range(30):
        bot = i % 7 == 6
        ev.append(dict(msg_id=f"m{i}", conv="g1", conv_type="group", ts=t0 + i * 25, role="self" if bot else "other",
                       speaker=None if bot else f"p{i % 3}", text="[图片]" if i == 3 else f"消息{i}", addressed=(i == 5)))
    p = tmp_path / "ev.jsonl"; p.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ev))
    os.environ["TIDAL_SHADOW_JSONL"] = str(p)
    assert run.main(["--source", "jsonl"]) == 0
    c = sqlite3.connect(S.DB)
    n_ev = c.execute("select count(*) from events").fetchone()[0]
    n_lab = c.execute("select count(*) from labels").fetchone()[0]
    kinds = c.execute("select media_kinds from events where text like '%图片%'").fetchone()[0]
    assert n_ev == 30 and n_lab == 30 and json.loads(kinds) == ["image"]
    assert run.main(["--source", "jsonl"]) == 0          # second tick: nothing new, nothing duplicated
    assert c.execute("select count(*) from events").fetchone()[0] == 30
    assert c.execute("select count(*) from labels").fetchone()[0] == 30
