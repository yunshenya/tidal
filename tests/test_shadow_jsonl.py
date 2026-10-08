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

def _tick_imports_torch(tmp_path, p4_on):
    """Run one cron tick in a fresh interpreter (pytest itself has torch loaded) and report whether torch got imported."""
    import subprocess, sys, pathlib
    t0 = time.time() - 4 * 3600
    ev = [dict(msg_id=f"z{i}", conv="g2", conv_type="group", ts=t0 + i * 30, role="other", speaker=f"p{i % 2}", text=f"x{i}") for i in range(6)]
    p = tmp_path / "ev.jsonl"; p.write_text("".join(json.dumps(e) + "\n" for e in ev))
    salt = tmp_path / "salt"; salt.write_text("test-salt-not-secret")
    env = dict(os.environ, TIDAL_SHADOW_JSONL=str(p), TIDAL_SHADOW_STATE=str(tmp_path / "st"), TIDAL_PSEUDO_SALT_FILE=str(salt))
    env.pop("TIDAL_SHADOW_P4", None)
    if p4_on: env["TIDAL_SHADOW_P4"] = "1"
    code = ("import sys, json; from tidal.shadow import run; rc = run.main(['--source', 'jsonl']); "
            "print(json.dumps(dict(rc=rc, torch='torch' in sys.modules)))")
    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=root, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])

def test_tick_with_p4_off_does_not_import_torch(tmp_path):
    r = _tick_imports_torch(tmp_path, p4_on=False)
    assert r == dict(rc=0, torch=False)

def test_tick_with_p4_on_imports_torch(tmp_path):
    assert _tick_imports_torch(tmp_path, p4_on=True) == dict(rc=0, torch=True)
