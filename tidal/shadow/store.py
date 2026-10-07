"""Local SQLite store for shadow mode (shadow/state/shadow.db, chmod 600, gitignored).
Holds ONLY pseudonymized ids and scrubbed text. Safe to re-run: every write is an upsert keyed on stable ids."""
import json, os, sqlite3, pathlib, time
from tidal.config import ROOT
STATE = pathlib.Path(os.environ.get("TIDAL_SHADOW_STATE", ROOT / "shadow" / "state"))
DB = STATE / "shadow.db"
SCHEMA = """
create table if not exists meta(key text primary key, value text);
create table if not exists events(
  msg_id text primary key, conv text, conv_type text, ts real, role text, speaker text, speaker_known int,
  text text, attention_reason text, chosen_action text, cur_addressed int, cur_reply int,
  -- non-text / multimodal metadata (content-free)
  media_kinds text, media_counts text, media_refs text, media_duration_s real, text_chars int,
  msg_chars int, has_question int, mentioned_bot int, reply_to_bot int, burst_len int, group_activity_1m int,
  first_seen_at real, text_seen_at real, updated_at real);
create index if not exists events_conv_ts on events(conv, ts);
create index if not exists events_ts on events(ts);
create table if not exists predictions(
  msg_id text, regime text, system text, probs text, event_ts real, predicted_at real, lag_s real, run_id text,
  version text, primary key(msg_id, regime, system));
create index if not exists predictions_ts on predictions(event_ts);
create table if not exists labels(
  msg_id text primary key, conv text, conv_type text, event_ts real, y_eot real, y_self real, y_addr real, y_act real,
  y_recheck real, y_hreply real, speaker_known int, labeled_at real, label_version text);
create table if not exists runs(
  run_id text primary key, started_at real, finished_at real, status text, stats text);
"""
def connect():
    STATE.mkdir(parents=True, exist_ok=True); os.chmod(STATE, 0o700)
    new = not DB.exists()
    c = sqlite3.connect(DB, timeout=60); c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=wal"); c.execute("pragma synchronous=normal")
    c.executescript(SCHEMA)
    if new: os.chmod(DB, 0o600)
    return c
def get_meta(c, key, default=None):
    r = c.execute("select value from meta where key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default
def set_meta(c, key, value):
    c.execute("insert into meta(key,value) values(?,?) on conflict(key) do update set value=excluded.value", (key, json.dumps(value)))
