import os, sys, pathlib, tempfile
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# tests never touch real secrets / state: throwaway salt + state dir
_tmp = pathlib.Path(tempfile.mkdtemp(prefix="tidal-test-"))
(_tmp / "salt").write_text("test-salt-not-secret")
os.environ["TIDAL_PSEUDO_SALT_FILE"] = str(_tmp / "salt")
os.environ["TIDAL_SHADOW_STATE"] = str(_tmp / "state")
