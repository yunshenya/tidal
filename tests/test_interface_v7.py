"""Import smoke for the frozen v7 multimodal stubs.

Does not call cue_fields. Does not change live interface_version.
Reads ``CUE_INTERFACE_VERSION`` from source so this file does not import
``llm_memory`` (and therefore torch).
"""
from __future__ import annotations

import ast
from pathlib import Path

from tidal.interface_v7 import (
    FINGERPRINT_DIM,
    FUSION_WEIGHT_DIM,
    FUSION_WEIGHT_ORDER,
    INTERFACE_VERSION_V7,
    MODALITY_VALUES,
    FusionWeightsStub,
    check_fingerprint,
    check_modality,
    empty_fusion_weights,
    zero_fingerprint,
)


def _live_interface_version() -> int:
    src = Path(__file__).resolve().parents[1] / "tidal" / "llm_memory.py"
    tree = ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CUE_INTERFACE_VERSION":
                    return ast.literal_eval(node.value)
    raise AssertionError("CUE_INTERFACE_VERSION not found")


def test_live_export_stays_v6():
    live = _live_interface_version()
    assert live == 6
    assert INTERFACE_VERSION_V7 == 7
    assert INTERFACE_VERSION_V7 != live


def test_stub_shapes():
    assert FINGERPRINT_DIM == 32
    assert FUSION_WEIGHT_DIM == 3
    assert FUSION_WEIGHT_ORDER == ("text", "audio", "image")
    assert MODALITY_VALUES == ("text", "audio", "image", "fused", "none")
    fp = zero_fingerprint()
    w = empty_fusion_weights()
    assert fp.shape == (32,) and fp.dtype.name == "float32"
    assert w.shape == (3,) and w.dtype.name == "float32"
    assert check_fingerprint(fp).shape == (32,)
    assert check_modality("fused") == "fused"
    stub = FusionWeightsStub(weights=w)
    assert stub.weights.shape == (3,)
