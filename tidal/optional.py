"""Clear errors for optional extras (see requirements-audio.txt / requirements-teacher.txt)."""
import importlib

def require(module, extra, pip_name=None):
    """Import `module` or raise ImportError naming the extra and how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        pkg = pip_name or module.split(".")[0]
        raise ImportError(
            f"optional dependency '{pkg}' is not installed (extra '{extra}'). "
            f"Install with: pip install -r requirements-{extra}.txt"
        ) from e
