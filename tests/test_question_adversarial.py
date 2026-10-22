"""Adversarial abstain strings are frozen and disjoint. Scores are not a gate.

The test split is not an input to a later fit. The dev split is the only
generated bank a change may consult. Sentence cosine and ask scores are
measured by ``python -m tidal.question_adversarial`` and are not asserted here.
"""
from pathlib import Path

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_stress import build_cases, jaccard_report, summary, validate_cases

# sha256 of the question list only. A wording change moves this. Do not retune
# after the abstain scores exist.
TEST_SHA256 = "cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5"
DEV_SHA256 = "26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a"

TEST_SLICES = {
    "relation_paraphrase": 48,
    "compact_form": 26,
    "bare_cue": 16,
    "misleading_cue": 24,
    "name_variant": 75,
    "near_dup_absent": 30,
    "two_names": 36,
    "deixis": 72,
}
DEV_SLICES = {
    "relation_paraphrase": 48,
    "compact_form": 24,
    "bare_cue": 16,
    "misleading_cue": 24,
    "name_variant": 77,
    "near_dup_absent": 30,
    "two_names": 36,
    "deixis": 72,
}


def test_splits_validate_and_stay_frozen():
    test = build_cases(TEST_BANK)
    dev = build_cases(DEV_BANK)
    assert validate_cases(test, other=dev) == ()
    assert validate_cases(dev, other=test) == ()
    test_info = summary(test)
    dev_info = summary(dev)
    assert test_info["sha256"] == TEST_SHA256
    assert dev_info["sha256"] == DEV_SHA256
    assert test_info["n"] == 327
    assert dev_info["n"] == 327
    assert test_info["slices"] == TEST_SLICES
    assert dev_info["slices"] == DEV_SLICES
    assert test_info["by_lang"] == {"en": 160, "zh": 167}
    assert dev_info["by_lang"] == {"en": 160, "zh": 167}
    assert set(test_info["by_label"]) == {"answerable", "missing_rel", "absent_person"}
    assert set(dev_info["by_label"]) == {"answerable", "missing_rel", "absent_person"}
    for lang in ("en", "zh"):
        for label in ("answerable", "missing_rel"):
            assert test_info["cells"][f"relation_paraphrase|{lang}|{label}"] >= 8
            assert test_info["cells"][f"misleading_cue|{lang}|{label}"] >= 4
            assert test_info["cells"][f"bare_cue|{lang}|{label}"] >= 4
            assert test_info["cells"][f"two_names|{lang}|{label}"] >= 4
        assert test_info["cells"][f"name_variant|{lang}|answerable"] >= 8
        assert test_info["cells"][f"near_dup_absent|{lang}|absent_person"] >= 8
        assert test_info["cells"][f"deixis|{lang}|answerable"] >= 24
        assert test_info["cells"][f"two_names|{lang}|absent_person"] >= 4
    assert all(case.pool == "" for case in test)
    pools = {}
    for case in dev:
        if case.pool:
            pools[case.pool] = pools.get(case.pool, 0) + 1
    assert pools == {"fit": 38, "select": 38}


def test_token_jaccard_stays_under_the_probe_bar():
    test = build_cases(TEST_BANK)
    dev = build_cases(DEV_BANK)
    for report in (jaccard_report(test, other=dev), jaccard_report(dev, other=test)):
        assert report["literal_hits"] == []
        for key in ("probe", "surface", "other_split"):
            assert report[key]["max"] < 0.65, (key, report[key]["max"], report[key]["pair"])
            assert report[key]["pairs_above_0.5"] == 0


def test_surface_fit_does_not_read_the_stress_sets():
    root = Path("tidal")
    surface = (root / "question_surface.py").read_text(encoding="utf-8")
    assert "question_adversarial" not in surface
    assert "question_stress" not in surface
    assert "question_abstain_dev" not in surface
    blob = surface
    for name in (
        "question_adversarial.py",
        "question_adversarial_data.py",
        "question_abstain_dev_data.py",
        "question_stress.py",
    ):
        blob += (root / name).read_text(encoding="utf-8")
    assert "enable_abstain" not in blob
    assert "use_abstain" not in blob
