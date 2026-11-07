"""E2 summary rows: true stage probes stay comparable; pseudo-stages do not."""

from __future__ import annotations

import json
import os

from tidal.growbrain_e2_probes import (
    handwritten_accs,
    is_pseudo_stage_probe_row,
    pseudo_stage_probe_accs,
    stage_probe_accs,
    unscored_stage_probes,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str):
    with open(os.path.join(ROOT, "reports", name), encoding="utf-8") as handle:
        return json.load(handle)


def test_archive_separates_random_order_from_stage_probes():
    rows = _load("growbrain_e2_multiseed_12.json")
    by_arm: dict[str, list] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    for row in by_arm["random_order"]:
        assert is_pseudo_stage_probe_row(row)
        assert stage_probe_accs(row) is None
        pseudo = pseudo_stage_probe_accs(row)
        assert pseudo is not None
        assert pseudo["probe0_acc"] == row["probe0_acc"]
        assert pseudo["probe3_acc"] == row["probe3_acc"]
        assert unscored_stage_probes(row) == ()

    for arm in ("coupled", "decoupled", "graduation_only"):
        for row in by_arm[arm]:
            assert not is_pseudo_stage_probe_row(row)
            probes = stage_probe_accs(row)
            assert probes is not None
            assert probes["probe0_acc"] == row["probe0_acc"]
            assert probes["probe3_acc"] == row["probe3_acc"]
            assert unscored_stage_probes(row) == ()
            assert pseudo_stage_probe_accs(row) is None

    for row in by_arm["growth_only"]:
        probes = stage_probe_accs(row)
        assert probes is not None
        assert probes["probe0_acc"] == row["probe0_acc"]
        assert probes["probe1_acc"] == row["probe1_acc"]
        assert probes["probe2_acc"] is None
        assert probes["probe3_acc"] is None
        assert unscored_stage_probes(row) == (2, 3)
        assert handwritten_accs(row)["hw0_acc"] == row["hw0_acc"]
        assert "extra_handwritten0_acc" not in handwritten_accs(row)
        assert "hw1_acc" not in handwritten_accs(row)


def test_scratch_final_probe_columns_are_true_stage_probes():
    results = _load("growbrain_e2_scratch_final_012.json")["results"]
    assert len(results) == 3
    for row in results:
        assert row["probe_pseudo_stages"] is False
        assert not is_pseudo_stage_probe_row(row)
        probes = stage_probe_accs(row)
        assert probes is not None
        assert probes["probe2_acc"] == row["probe2_acc"]
        assert unscored_stage_probes(row) == ()
        hw = handwritten_accs(row)
        assert hw["extra_handwritten0_acc"] == row["extra_handwritten0_acc"]
        assert hw["extra_handwritten3_acc"] == row["extra_handwritten3_acc"]
        assert "hw0_acc" not in hw
        assert pseudo_stage_probe_accs(row) is None


def test_explicit_flag_overrides_the_archive_random_order_default():
    pseudo_only = {
        "arm": "coupled",
        "probe_pseudo_stages": True,
        "probe0_acc": 0.5,
        "probe1_acc": 0.5,
        "probe2_acc": 0.5,
        "probe3_acc": 0.5,
    }
    assert is_pseudo_stage_probe_row(pseudo_only)
    assert stage_probe_accs(pseudo_only) is None
    assert pseudo_stage_probe_accs(pseudo_only)["probe0_acc"] == 0.5

    fixed_random = {
        "arm": "random_order",
        "probe_pseudo_stages": False,
        "probe0_acc": 0.61,
        "probe1_acc": 0.42,
        "probe2_acc": None,
        "probe3_acc": 0.43,
    }
    assert not is_pseudo_stage_probe_row(fixed_random)
    probes = stage_probe_accs(fixed_random)
    assert probes is not None
    assert probes["probe0_acc"] == 0.61
    assert probes["probe2_acc"] is None
    assert unscored_stage_probes(fixed_random) == (2,)
    assert pseudo_stage_probe_accs(fixed_random) is None


def test_report_keeps_pseudo_stages_out_of_the_stage_ladder():
    report = open(os.path.join(ROOT, "reports", "growbrain_e2.md"), encoding="utf-8").read()
    for needle in (
        "shuffled pseudo-stages",
        "probe_pseudo_stages",
        "not a stage ladder",
        "Pseudo-stage probes",
        "unscored",
    ):
        assert needle in report
