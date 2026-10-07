"""Message-type / media metadata front end (implemented). Content-free: works from platform placeholders only.

Chat platforms (and bot frameworks) render non-text messages as placeholders such as "[图片]", "[语音]",
"[视频]", "[表情包]", "[STICKER name]" or OneBot CQ codes like "[CQ:image,file=...]". This module turns them into
a small multi-hot vector so timing models can already condition on "this was a voice note / sticker / image"
while the real image/audio/video encoders are still planned. Media file references (if any) are returned so a
caller can store them pseudonymized for future multimodal training -- this module never fetches media."""
import re
import numpy as np

KINDS = ["image", "sticker", "face", "voice", "video", "file", "music", "share", "forward", "reply", "at", "other_media"]
_PLACEHOLDER = [
    (re.compile(r"\[(?:图片|图像|照片|image)\]", re.I), "image"),
    (re.compile(r"\[(?:表情包|动画表情|STICKER[^\]]{0,40}|sticker[^\]]{0,40})\]"), "sticker"),
    (re.compile(r"\[(?:表情|face)\]"), "face"),
    (re.compile(r"\[(?:语音|voice|record)\]", re.I), "voice"),
    (re.compile(r"\[(?:视频|video|短视频)\]", re.I), "video"),
    (re.compile(r"\[(?:文件|file)[^\]]{0,40}\]", re.I), "file"),
    (re.compile(r"\[(?:音乐|music)[^\]]{0,40}\]", re.I), "music"),
    (re.compile(r"\[(?:分享|链接|小程序|卡片|抖音|位置|share)[^\]]{0,40}\]", re.I), "share"),
    (re.compile(r"\[(?:聊天记录|合并转发|forward)\]", re.I), "forward"),
]
_CQ = re.compile(r"\[CQ:([a-z_]+)((?:,[^\]]*)?)\]")
_CQ_KIND = {"image": "image", "mface": "sticker", "face": "face", "record": "voice", "video": "video", "file": "file",
            "music": "music", "json": "share", "xml": "share", "share": "share", "location": "share", "forward": "forward",
            "node": "forward", "reply": "reply", "at": "at"}
_CQ_REF = re.compile(r"(?:^|,)(?:file|file_id|url|id)=([^,\]]+)")
_CQ_DUR = re.compile(r"(?:^|,)(?:duration|time|length)=(\d+(?:\.\d+)?)")
_STICKER_NAME = re.compile(r"\[STICKER\s+([^\]]{1,40})\]")
_FACE_NAME = re.compile(r"\[[\u4e00-\u9fff]{1,5}\]")      # QQ built-in face names, e.g. [害羞] [握手]
_PLACEHOLDER_ANY = re.compile(r"\[[^\]]{1,40}\]|<CQ>|<URL>")

def extract(text):
    """-> dict(kinds=[...], counts={kind: n}, refs=[raw refs], duration_s=float|None, text_chars=int).
    `refs` are raw identifiers (CQ file ids, sticker names): callers must pseudonymize before storing."""
    if not isinstance(text, str) or not text:
        return dict(kinds=[], counts={}, refs=[], duration_s=None, text_chars=0)
    counts, refs, dur = {}, [], None
    for m in _CQ.finditer(text):
        k = _CQ_KIND.get(m.group(1), "other_media"); counts[k] = counts.get(k, 0) + 1
        refs += _CQ_REF.findall(m.group(2) or "")
        d = _CQ_DUR.search(m.group(2) or "")
        if d: dur = (dur or 0.0) + float(d.group(1))
    for rx, k in _PLACEHOLDER:
        n = len(rx.findall(text))
        if n: counts[k] = counts.get(k, 0) + n
    faces = [f for f in _FACE_NAME.findall(text) if not any(rx.fullmatch(f) for rx, _ in _PLACEHOLDER)]
    if faces: counts["face"] = counts.get("face", 0) + len(faces)
    refs += ["sticker:" + s.strip() for s in _STICKER_NAME.findall(text)]
    if "<CQ>" in text: counts["other_media"] = counts.get("other_media", 0) + text.count("<CQ>")
    plain = _PLACEHOLDER_ANY.sub("", _CQ.sub("", text)).strip()
    return dict(kinds=sorted(counts), counts=counts, refs=refs, duration_s=dur, text_chars=len(plain))

def vector(info) -> np.ndarray:
    """multi-hot media kinds + log1p(plain text chars) -> float32 [len(KINDS)+1]"""
    v = np.zeros(len(KINDS) + 1, np.float32)
    for k, n in (info.get("counts") or {}).items():
        v[KINDS.index(k)] = 1.0
    v[-1] = np.log1p(info.get("text_chars") or 0)
    return v
