"""Shadow tick lock must import on Windows, where fcntl and resource do not exist."""
import ast, builtins, importlib, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _top_imports(path):
    tree = ast.parse(path.read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names

def test_run_module_does_not_import_fcntl_or_resource_at_top():
    names = _top_imports(ROOT / "tidal/shadow/run.py")
    assert "fcntl" not in names and "resource" not in names

def test_lock_imports_on_simulated_win32_without_fcntl_or_resource(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    real = builtins.__import__
    def guarded(name, *a, **k):
        if name in ("fcntl", "resource") or name.startswith("fcntl.") or name.startswith("resource."):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)
    sys.modules.pop("tidal.shadow.lock", None)
    importlib.import_module("tidal.shadow.lock")

def test_win32_lock_uses_msvcrt(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    fake = types.ModuleType("msvcrt")
    held = {"n": 0}
    def locking(fd, mode, n):
        if held["n"]:
            raise OSError("locked")
        held["n"] += 1
    fake.LK_NBLCK = 1
    fake.locking = locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    from tidal.shadow.lock import try_lock
    first = try_lock(tmp_path / "a.lock")
    assert first is not None
    assert try_lock(tmp_path / "a.lock") is None

def test_posix_lock_is_exclusive(tmp_path):
    from tidal.shadow.lock import try_lock
    a = try_lock(tmp_path / "b.lock")
    assert a is not None and try_lock(tmp_path / "b.lock") is None
