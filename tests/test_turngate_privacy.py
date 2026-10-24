import ast, inspect, re
import pytest
from tidal import turngate
from tidal.public_data import convert, manifest


def row(key, group, t, human=True, resp="answer", source=None):
    prov = {"source": source or (turngate.HUMAN if human else turngate.AUTO)}
    return {"source_key": key, "scope_id": group, "occurred_at": t, "label_provenance": prov,
            "labels": {"response": resp, "completion": "flush_now"}, "current_text": "x",
            "context": {"recent_turns": [{"role": "member", "speaker": "Alice", "text": "y"}]}}


def test_split_groups_do_not_intersect_and_weak_rows_are_filtered():
    rows = [row(f"k{g}_{i}", f"g{g}", 1_700_000_000 + 86400 * 3 + i * 60) for g in range(60) for i in range(5)]
    rows += [row(f"w{g}", f"g{g}", 1_700_000_000 + 86400 * 3, human=False) for g in range(60)]
    rows += [row("k0_0", "g0", 0, human=False)]  # duplicate key: human wins
    out = turngate.assign(rows, include_weak=True)
    units = {}
    for r, sp in out:
        if turngate.is_human(r): units.setdefault(sp, set()).add(r["scope_id"])
    assert set(units) == {"train", "val", "test"}
    assert not (units["train"] & units["val"]) and not (units["train"] & units["test"]) and not (units["val"] & units["test"])
    evalg = units["val"] | units["test"]
    weak = [r for r, sp in out if not turngate.is_human(r)]
    assert weak and all(sp == "train" for r, sp in out if not turngate.is_human(r))
    assert not {r["scope_id"] for r in weak} & evalg
    assert sum(r["source_key"] == "k0_0" for r, _ in out) == 1
    assert not turngate.assign(rows, include_weak=False) or all(turngate.is_human(r) for r, _ in turngate.assign(rows))


def test_weak_rows_off_by_default_and_legacy_rule_labels_refused():
    rows = [row("a", "g1", 1e9), row("b", "g2", 1e9, human=False), row("c", "g3", 1e9, human=False, source="assistant_auto_v1")]
    assert [r["source_key"] for r, _ in turngate.assign(rows)] == ["a"]
    keys = {r["source_key"] for r, _ in turngate.assign(rows, include_weak=True)}
    assert "c" not in keys


def test_missing_group_id_is_an_error():
    r = row("a", None, 1e9); r.pop("scope_id")
    with pytest.raises(ValueError, match="group/scope"):
        turngate.assign([r])


def test_frame_hashes_speakers_and_masks_timing():
    d = turngate.to_frame([(row("a", "g", 1e9), "train")], b"salt")
    assert "Alice" not in set(d.speaker) and not d.timing_known.any()
    assert not d.conv.str.contains("a$").any()
    import numpy as np, pandas as pd
    from tidal.features import FEATURES, TIMING_FEATURES, compute
    df = pd.DataFrame(dict(conv=["c"] * 3, ts=[0., 50., 4000.], role=["other"] * 3, speaker=["s"] * 3, speaker_known=[True] * 3,
                           text=["a", "b", "c"], conv_type=["group"] * 3, timing_known=[False] * 3))
    X = compute(df); idx = [FEATURES.index(c) for c in TIMING_FEATURES]
    assert np.all(X[:, idx] == 0)


def test_scrub_removes_pii():
    from scripts.prepare_private_data import scrub, scrub_tree
    s = scrub("call 13812345678 or mail a.b@example.com, id 11010519491231002X see https://x.y/z")
    assert "13812345678" not in s and "example.com" not in s and "11010519491231002X" not in s and "https" not in s
    assert scrub_tree({"a": ["13812345678"]}) == {"a": ["<PHONE>"]}


def test_auto_labeler_has_no_lexical_lists():
    src = open("scripts/auto_label_turngate.py", encoding="utf-8").read()
    assert "re.compile" not in src and "import re" not in src
    assert not re.search(r"[\u4e00-\u9fff]", src)


def test_every_converter_has_a_licensed_manifest_entry():
    tree = ast.parse(inspect.getsource(convert))
    converters = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)
                  and any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "finish" for n in ast.walk(f))}
    assert converters == set(manifest.CONVERTERS)
    for name, key in manifest.CONVERTERS.items():
        entry = manifest.DATASETS[key]
        assert entry.get("license") and isinstance(entry.get("research_only"), bool), name
    assert manifest.DATASETS["interactspeech"]["research_only"] and manifest.DATASETS["magicdata_ms"]["research_only"]
