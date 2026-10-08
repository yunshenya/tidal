"""Cross-platform non-blocking file lock for shadow ticks.

Unix uses fcntl.flock; Windows uses msvcrt.locking. Neither module is imported
until acquire time, so importing this file does not require fcntl or resource.
"""
import os, sys

def try_lock(path):
    """Exclusive non-blocking lock on `path`. Returns the open file (keep it alive) or None if another holder has it."""
    path = os.fspath(path)
    fh = open(path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            if fh.seek(0, os.SEEK_END) == 0:
                fh.write(b"\0"); fh.flush()
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                fh.close(); return None
        else:
            import fcntl
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close(); return None
    except Exception:
        fh.close(); raise
    return fh
