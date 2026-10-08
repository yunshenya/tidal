"""Missing optional extras raise a message that names the extra, not a deep import error."""
import builtins
import importlib
import pytest

def test_require_names_the_extra(monkeypatch):
    def boom(name):
        raise ImportError("no module named " + name)
    monkeypatch.setattr(importlib, "import_module", boom)
    # tidal.optional bound importlib at import time
    import tidal.optional as opt
    monkeypatch.setattr(opt.importlib, "import_module", boom)
    with pytest.raises(ImportError, match=r"extra 'audio'") as e:
        opt.require("soundfile", "audio")
    assert "requirements-audio.txt" in str(e.value)
    with pytest.raises(ImportError, match=r"extra 'teacher'") as e:
        opt.require("sherpa_onnx", "teacher", pip_name="sherpa-onnx")
    assert "sherpa-onnx" in str(e.value) and "requirements-teacher.txt" in str(e.value)

def test_sensevoice_teacher_missing_names_the_extra(monkeypatch):
    real = builtins.__import__
    def guarded(name, *a, **k):
        if name == "sherpa_onnx" or name.startswith("sherpa_onnx."):
            raise ImportError("simulated missing sherpa_onnx")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)
    from tidal.emotion_speech import teacher
    with pytest.raises(ImportError, match=r"extra 'teacher'"):
        teacher()
