from tidal.modalities import meta, REGISTRY
import tidal.modalities.image, tidal.modalities.audio, tidal.modalities.video, tidal.modalities.text

def test_placeholders():
    assert meta.extract("[语音]")["kinds"] == ["voice"]
    assert meta.extract("看这个[图片][图片]")["counts"] == {"image": 2}
    assert meta.extract("[STICKER 晴天]")["refs"] == ["sticker:晴天"]
    assert meta.extract("[害羞]好的")["kinds"] == ["face"]
    assert meta.extract("普通文本")["kinds"] == []

def test_cq_codes():
    i = meta.extract("[CQ:record,file=abc.silk,duration=7]")
    assert i["kinds"] == ["voice"] and i["refs"] == ["abc.silk"] and i["duration_s"] == 7.0
    assert meta.extract("[CQ:image,file=x.jpg]hi")["text_chars"] == 2

def test_vector_shape():
    assert meta.vector(meta.extract("[视频]")).shape == (len(meta.KINDS) + 1,)

def test_planned_frontends_declared():
    assert {"text", "image", "audio", "video"} <= set(REGISTRY)
    for n in ("image", "audio", "video"):
        assert REGISTRY[n]().available() is False
