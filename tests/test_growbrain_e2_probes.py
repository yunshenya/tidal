"""E2 summary rows: true stage probes stay comparable; pseudo-stages do not."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tidal.growbrain_e2_probes import (
    arm_stage_probe_means,
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
        "48ad257",
        "e2_eval_heldout.py",
        "no evidence either way",
        "not a capability measure",
        "not evidence that growth helps",
        "equal steps, not equal compute",
        "Evaluation limits",
        "87.8%",
        "41.7%",
        "in progress",
        "growbrain_e2_true_stage_reeval.json",
    ):
        assert needle in report
    for banned in (
        "does not beat",
        "Graduation shows no benefit",
        "ordered by that mean",
        "highest mean",
        "ties direct",
    ):
        assert banned not in report
    assert "0.5460860315567231" in report
    assert "0.6183554063860668" in report
    assert "0.5497171153218631" in report


def test_true_stage_reeval_scores_every_checkpoint():
    payload = _load("growbrain_e2_true_stage_reeval.json")
    assert payload["growbrain_commit"] == "48ad257"
    assert payload["script"] == "scripts/e2_eval_heldout.py"
    assert payload["eval_only"] is True
    assert payload["training"] is False
    assert payload["optimizer_steps"] == 0
    assert payload["probe_pseudo_stages"] is False
    rows = payload["rows"]
    assert len(rows) == 13
    arms: dict[str, set[int]] = {}
    for row in rows:
        assert row["probe_pseudo_stages"] is False
        assert row["device"] == "cpu"
        assert row["data_seed"] == row["seed"]
        assert not is_pseudo_stage_probe_row(row)
        probes = stage_probe_accs(row)
        assert probes is not None
        assert unscored_stage_probes(row) == ()
        assert pseudo_stage_probe_accs(row) is None
        for stage in (0, 1, 2, 3):
            assert isinstance(probes[f"probe{stage}_acc"], float)
        arms.setdefault(row["arm"], set()).add(row["seed"])
    assert arms["scratch_final"] == {0, 1, 2}
    for arm in ("coupled", "decoupled", "graduation_only", "growth_only", "random_order"):
        assert arms[arm] == {1, 2}
    means = arm_stage_probe_means(rows)
    assert set(means) == set(arms)
    for arm, stats in means.items():
        stored = payload["arm_means"][arm]
        assert stats["n_seeds"] == stored["n_seeds"]
        for key in ("probe0", "probe1", "probe2", "probe3", "mean"):
            assert stats[key] == stored[key]


def test_compare_script_prints_the_reeval_after_the_training_split():
    completed = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "growbrain_e2_compare.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    out = completed.stdout
    assert "multi-seed pseudo-stage probes" in out
    assert "pseudo_rows\t2" in out
    assert "true-stage re-evaluation" in out
    assert "comparable_rows\t13" in out
    assert "growth_only\t2\t" in out
    assert "scratch_final_matches_training_probes\ttrue" in out
    assert "shuffled pseudo-stages and are not the true-stage re-evaluation columns" in out
    assert out.index("multi-seed pseudo-stage probes") < out.index("true-stage re-evaluation")
    label = "equal_weight_mean_of_probe0_3\tnot a capability measure"
    assert label in out
    per_probe = out.split(label, 1)[1]
    assert per_probe.index("\ncoupled\t") < per_probe.index("\ngrowth_only\t")
    assert per_probe.index("\ngrowth_only\t") < per_probe.index("\nscratch_final\t")


def test_scratch_final_reeval_matches_training_time_scores():
    reeval = _load("growbrain_e2_true_stage_reeval.json")["rows"]
    training = _load("growbrain_e2_scratch_final_012.json")["results"]
    by_seed = {row["seed"]: row for row in reeval if row["arm"] == "scratch_final"}
    assert set(by_seed) == {row["seed"] for row in training}
    for row in training:
        left = stage_probe_accs(row)
        right = stage_probe_accs(by_seed[row["seed"]])
        assert left == right
        assert left["probe0_acc"] == row["probe0_acc"]
        assert left["probe3_acc"] == row["probe3_acc"]
