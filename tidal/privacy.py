"""Pseudonymization + PII scrubbing. Salt lives in .secrets/pseudo_salt (never committed)."""
import hmac, hashlib, re, pathlib
_SALT = None
def salt():
    global _SALT
    if _SALT is None:
        import os
        p = os.environ.get("TIDAL_PSEUDO_SALT_FILE") or pathlib.Path(__file__).resolve().parent.parent.joinpath(".secrets/pseudo_salt")
        _SALT = pathlib.Path(p).read_bytes().strip()
    return _SALT
def pseudo(kind: str, raw) -> str:
    if raw is None: return None
    return kind + "_" + hmac.new(salt(), f"{kind}:{raw}".encode(), hashlib.sha256).hexdigest()[:12]

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_QQ = re.compile(r"(?<!\d)[1-9]\d{4,11}(?!\d)")       # QQ numbers / long ids
_CQ = re.compile(r"\[CQ:[^\]]*\]")
_DOUYIN_CODE = re.compile(r"[A-Za-z0-9]{1,4}@[A-Za-z0-9.]{2,6}\s*:?\d{0,2}[ap]m|\b[A-Za-z]{3}:/")   # share tokens
def scrub(text: str) -> str:
    if text is None: return None
    t = _CQ.sub("<CQ>", text)
    t = _URL.sub("<URL>", t)
    t = _EMAIL.sub("<EMAIL>", t)
    t = _IDCARD.sub("<ID>", t)
    t = _PHONE.sub("<PHONE>", t)
    t = _QQ.sub("<NUM>", t)
    t = _DOUYIN_CODE.sub("<TOK>", t)
    return t
