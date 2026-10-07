from tidal.privacy import scrub, pseudo

def test_scrub_removes_ids():
    t = scrub("加我 12345678 或者 13812345678，邮箱 a.b@example.com 看 https://x.y/z")
    assert "12345678" not in t and "13812345678" not in t and "example.com" not in t and "https" not in t
    assert "<NUM>" in t and "<PHONE>" in t and "<EMAIL>" in t and "<URL>" in t

def test_pseudo_is_stable_and_salted():
    a, b = pseudo("u", "alice"), pseudo("u", "alice")
    assert a == b and a.startswith("u_") and "alice" not in a and pseudo("u", "bob") != a
    assert pseudo("u", None) is None
