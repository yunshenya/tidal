"""Deployment-specific settings. Public defaults are generic; a local, gitignored `config/local.json`
(or env vars) supplies the real values for a given deployment (bot names, held-out groups, ...)."""
import json, os, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL = ROOT / "config" / "local.json"
_cache = None
def local() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(LOCAL.read_text()) if LOCAL.exists() else {}
    return _cache
def get(key, default=None):
    env = os.environ.get("TIDAL_" + key.upper())
    if env is not None:
        return env.split("|") if isinstance(default, list) else env
    return local().get(key, default)
# names the bot is called by (used for the has_name feature and the weak "addressed" label)
BOT_NAMES = get("bot_names", ["小潮"])
# names used when prompting the synthetic-dialogue generator
SYNTH_BOT_NAMES = get("synth_bot_names", ["小潮"] * 8 + ["潮潮", "阿潮", "小助手"])
# pseudonymous conversation ids held out entirely for the unseen-group test split
HOLDOUT_GROUPS = get("holdout_groups", [])
def bot_name_pattern() -> str:
    import re
    return "|".join(re.escape(n) for n in BOT_NAMES)
