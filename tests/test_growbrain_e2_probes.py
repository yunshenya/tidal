"""E2 summary rows: true stage probes stay comparable; pseudo-stages do not."""

from __future__ import annotations

import csv
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
        "not a capability measure",
        "not evidence that growth helps",
        "equal steps, not equal compute",
        "Evaluation limits",
        "87.8%",
        "41.7%",
        "in progress",
        "growbrain_e2_true_stage_reeval.json",
        "growbrain_e2_cjk_l128.json",
        "0.34717033415639487",
        "0.350",
        "not a tie",
        "flips sign",
        "too tight or too small",
        "129–512",
    ):
        assert needle in report
    for banned in (
        "does not beat",
        "Graduation shows no benefit",
        "ordered by that mean",
        "highest mean",
        "ties direct",
        "no evidence either way",
    ):
        assert banned not in report
    assert "0.5460860315567231" in report
    assert "0.6183554063860668" in report
    assert "0.5497171153218631" in report


def test_task_probe_section_matches_the_arm_csv():
    report = open(os.path.join(ROOT, "reports", "growbrain_e2.md"), encoding="utf-8").read()
    section = report.split("## 任务题正式对照（task probes）", 1)[1]
    assert "/workspace" not in section
    assert "e112519" in section
    assert "不进入臂间比较" in section
    assert "0.2218458933" in section
    assert "train_filter.checked=false" in section
    arms_path = os.path.join(ROOT, "reports", "growbrain_e2_taskprobe_arms.csv")
    runs_path = os.path.join(ROOT, "reports", "growbrain_e2_taskprobe_runs.csv")
    with open(arms_path, encoding="utf-8") as handle:
        arms = list(csv.DictReader(handle))
    with open(runs_path, encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    assert len(arms) == 21
    assert len(runs) == 39
    assert all(row["arm"] != "staged_final" for row in runs)
    assert all(row["n_truncated"] == "0" for row in runs)
    assert "/workspace" not in open(arms_path, encoding="utf-8").read()
    assert "/workspace" not in open(runs_path, encoding="utf-8").read()
    for row in arms:
        if row["arm"] == "scratch_final" and row["stage"] == "0":
            assert row["exact_match_mean"] in section
        if row["arm"] == "coupled" and row["stage"] == "0":
            assert row["exact_match_mean"] in section
        if row["arm"] == "graduation_only" and row["stage"] == "2":
            assert row["cjk_token_acc_mean"] in section
        if row["arm"] == "staged_final" and row["stage"] == "0":
            assert row["n_seeds"] == "2"
            assert row["n_scored"] == "360"
            assert row["exact_match_mean"] in section
            assert row["cjk_token_acc_mean"] in section


def test_staged_final_trajectory_matches_the_arm_means():
    report = open(os.path.join(ROOT, "reports", "growbrain_e2.md"), encoding="utf-8").read()
    section = report.split("## staged_final 对照臂", 1)[1]
    for needle in (
        "0d8d8f1",
        "18406322",
        "8000",
        "15000",
        "39000",
        "9000",
        "16000",
        "37000",
        "0.12777777777777777",
        "0.6430990685859441",
        "0.6257408975444538",
        "0.5855484837139648",
        "0.5015599650567827",
        "0.44847165704551906",
        "0.4244345604239614",
        "0.5566037734",
        "0.7248677248",
        "0.0596951736140559",
        "0.0440304826555462",
        "210000",
        "200000",
    ):
        assert needle in section
    path = os.path.join(ROOT, "reports", "growbrain_e2_staged_final_trajectory.csv")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert "/workspace" not in open(path, encoding="utf-8").read()
    by_seed: dict[str, list] = {}
    for row in rows:
        assert row["when"] == "eval"
        assert row["stage"] == "0"
        assert row["n_scored"] == "360"
        assert row["n_truncated"] == "0"
        by_seed.setdefault(row["seed"], []).append(row)
    assert set(by_seed) == {"1", "2"}
    for seed_rows in by_seed.values():
        steps = [int(row["step"]) for row in seed_rows]
        assert steps == list(range(1000, 100001, 1000))
    end = {seed: rows_for_seed[-1] for seed, rows_for_seed in by_seed.items()}
    assert end["1"]["exact_match"] == "0.12777777777777777"
    assert end["1"]["cjk_token_acc"] == "0.6430990685859441"
    assert end["2"]["exact_match"] == "0.125"
    assert end["2"]["cjk_token_acc"] == "0.6257408975444538"
    exact = [float(end[seed]["exact_match"]) for seed in ("1", "2")]
    cjk = [float(end[seed]["cjk_token_acc"]) for seed in ("1", "2")]
    arms_path = os.path.join(ROOT, "reports", "growbrain_e2_taskprobe_arms.csv")
    with open(arms_path, encoding="utf-8") as handle:
        arms = list(csv.DictReader(handle))
    stage0 = next(row for row in arms if row["arm"] == "staged_final" and row["stage"] == "0")
    assert float(stage0["exact_match_mean"]) == sum(exact) / 2
    assert float(stage0["cjk_token_acc_mean"]) == sum(cjk) / 2


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
    assert "shared-span CJK acc at L=128" in out
    assert "tokens_129_to_512\tnot scored separately" in out
    cjk_section = out.split("shared-span CJK acc at L=128", 1)[1]
    assert "graduation_only\t2\t" in cjk_section
    assert cjk_section.index("\ncoupled\t") < cjk_section.index("\ngrowth_only\t")


def test_cjk_l128_probe2_means_match_the_report():
    payload = _load("growbrain_e2_cjk_l128.json")
    assert payload["shared_max_seq_len"] == 128
    assert payload["n_complete"] == 13
    assert payload["probe2_subset"] is False
    by_arm = {row["arm"]: row for row in payload["aggregates"]}
    report = open(os.path.join(ROOT, "reports", "growbrain_e2.md"), encoding="utf-8").read()
    for arm in ("coupled", "decoupled", "graduation_only", "growth_only", "random_order", "scratch_final"):
        mean = by_arm[arm]["probe2_cjk_acc_mean"]
        se = by_arm[arm]["probe2_cjk_acc_typical_item_se"]
        assert json.dumps(mean) in report
        assert json.dumps(se) in report
    scratch = by_arm["scratch_final"]["probe2_cjk_acc_mean"]
    coupled = by_arm["coupled"]["probe2_cjk_acc_mean"]
    graduation = by_arm["graduation_only"]["probe2_cjk_acc_mean"]
    assert scratch - coupled > 0.06
    assert abs(graduation - 0.34717033415639487) == 0.0
    assert graduation < 0.350


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
