"""Separate E2 true stage probes from shuffled pseudo-stage probes.

The columns the runs already write are ``probe0_acc`` .. ``probe3_acc``,
``probe_pseudo_stages``, ``hwN_acc``, and ``extra_handwrittenN_acc``.
This module does not score a model and does not add a metric.

``probe0_acc`` .. ``probe3_acc`` are the comparable held-out stage probes
when ``probe_pseudo_stages`` is false. They are shuffled pseudo-stages
when ``probe_pseudo_stages`` is true. The multi-seed archive omitted the
flag on ``random_order``; those rows are pseudo-stages. A later
``random_order`` row that sets ``probe_pseudo_stages`` false is the true
stage ladder for that arm. ``reports/growbrain_e2_true_stage_reeval.json``
is that pass: every row sets the flag false, including ``random_order``.

A pseudo-stage row has no comparable stage-probe columns. Logging only
those four numbers does not stand in for the held-out stage probes.
When a run logs both, the comparable columns are the row (or object)
with ``probe_pseudo_stages`` false. The pseudo-stage row keeps
``probe_pseudo_stages`` true and is not that comparison.

Null probe cells are unscored. They are not zeros.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STAGES = (0, 1, 2, 3)
# The summaries log handwritten banks 0, 2, and 3. Stage 1 has no hw column.
HANDWRITTEN_STAGES = (0, 2, 3)


def is_pseudo_stage_probe_row(row: Mapping[str, Any]) -> bool:
    """True when this row's probe0–3 are shuffled pseudo-stages."""
    flag = row.get("probe_pseudo_stages")
    if isinstance(flag, bool):
        return flag
    return row.get("arm") == "random_order"


def stage_probe_accs(row: Mapping[str, Any]) -> dict[str, float | None] | None:
    """Held-out stage probe accuracies, or None when probe0–3 are pseudo-only."""
    if is_pseudo_stage_probe_row(row):
        return None
    return {f"probe{stage}_acc": _acc(row, f"probe{stage}_acc") for stage in STAGES}


def pseudo_stage_probe_accs(row: Mapping[str, Any]) -> dict[str, float | None] | None:
    """Shuffled pseudo-stage accuracies, or None when probe0–3 are stage probes."""
    if not is_pseudo_stage_probe_row(row):
        return None
    return {f"probe{stage}_acc": _acc(row, f"probe{stage}_acc") for stage in STAGES}


def unscored_stage_probes(row: Mapping[str, Any]) -> tuple[int, ...]:
    """Stage indexes whose comparable probe accuracy is null."""
    probes = stage_probe_accs(row)
    if probes is None:
        return ()
    return tuple(stage for stage in STAGES if probes[f"probe{stage}_acc"] is None)


def arm_stage_probe_means(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, dict[str, float | int]]:
    """Mean of fully scored true stage probes, grouped by arm.

    A pseudo-stage row is omitted. A stage-probe row with any null probe
    is omitted. The mean is the unweighted mean of probe0–3.
    """
    grouped: dict[str, list[dict[str, float | None]]] = {}
    for row in rows:
        probes = stage_probe_accs(row)
        if probes is None:
            continue
        if any(probes[f"probe{stage}_acc"] is None for stage in STAGES):
            continue
        grouped.setdefault(str(row.get("arm")), []).append(probes)
    means: dict[str, dict[str, float | int]] = {}
    for arm, probes_list in grouped.items():
        n_seeds = len(probes_list)
        stats: dict[str, float | int] = {"n_seeds": n_seeds}
        for stage in STAGES:
            key = f"probe{stage}_acc"
            stats[f"probe{stage}"] = sum(item[key] for item in probes_list) / n_seeds
        stats["mean"] = sum(stats[f"probe{stage}"] for stage in STAGES) / len(STAGES)
        means[arm] = stats
    return means


def handwritten_accs(row: Mapping[str, Any]) -> dict[str, float | None]:
    """Handwritten banks the row already has. Missing banks are omitted."""
    found: dict[str, float | None] = {}
    for stage in HANDWRITTEN_STAGES:
        for key in (f"hw{stage}_acc", f"extra_handwritten{stage}_acc"):
            if key in row:
                found[key] = _acc(row, key)
    return found


def _acc(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)
