"""Error-pattern measurement. Production modules stay byte-identical."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1037/0278-7393.20.5.1063",
    "10.1037/0096-3445.128.2.186",
    "10.1177/1754073908100432",
    "10.1037/0278-7393.23.5.1083",
    "10.1038/nrn1764",
    "10.1146/annurev-psych-010213-115117",
    "10.1098/rstb.2019.0693",
)
PRODUCTION = {
    "tidal/llm_memory.py": "3551a6d64f0ff852d870b0f7499e6b75de6fad1a75b3c7f01419761f20e89ce3",
    "tidal/episodic.py": "e4d7d74e928a6794bbd50795000c7f3f839c75e3bdc02004b48b22c310b5bec2",
    "tidal/reconstruct.py": "55188f6179661bb918bb5dd487d9444ea67368edf52bd5a8205eef101ac0d475",
    "tidal/cue_inject.py": "f81697d2648244a74a5dd8182a20fa3dfb6be40576c31aba86385cfb7f3be680",
    "tidal/sleep.py": "dba0ecce7beca194840b1e33c8b16dbd3c85e80738b664ed38c2d25098f87d9a",
    "tidal/arousal.py": "91ee12a53868e4dce1dc799222705400add94e4d10320a38d98192fe3c366ea4",
    "tidal/clock_drift.py": "154c77f409c15ae4ce1776055203625840afd597d06f495db62da4df44a08b36",
    "tidal/backend.py": "b6fcc1487236e153d63d1a909b36b0c46ebc35b9bdb4e4f0a47b4300d730ce0f",
    "tidal/subconscious_eval.py": "1e4784b23fbd622ce30bd6d8eeeb00f518c8b041082c5450766859c1cd5a0c6d",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imports(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    return found


def test_bank_is_disjoint_and_unscored():
    data = _load(ROOT / "scripts" / "error_pattern_data.py", "error_pattern_data")
    assert data.PROTOCOL == "error-pattern-v1"
    assert len(data.FIT_ITEMS) == 8
    assert len(data.HELD_ITEMS) == 50
    assert set(data.FIT_IDS).isdisjoint(data.HELD_IDS)
    assert data.FIT_IDS[0] == "error-fit-000"
    assert data.HELD_IDS[-1] == "error-held-049"
    fit_words = set(data.FIT_ANSWERS) | set(data.FIT_FOILS) | set(data.FIT_DOMAINS)
    held_words = set(data.HELD_ANSWERS) | set(data.HELD_FOILS) | set(data.HELD_DOMAINS)
    assert fit_words.isdisjoint(held_words)
    assert set(data.FIT_ANSWERS).isdisjoint(data.FIT_FOILS)
    assert set(data.HELD_ANSWERS).isdisjoint(data.HELD_FOILS)
    from tidal.subconscious_cue_fit50_data import content_words as fit50_words
    from tidal.subconscious_cue_word_data import content_words as word_words

    banned = set(word_words()) | set(fit50_words())
    banned |= {"hand", "some", "state", "circles", "toward", "about"}
    assert data.content_words().isdisjoint(banned)
    assert data.PHONE == "+1-617-555-4826"
    assert data.BIRTHDAY == "April 21, 1984"
    assert len(data.bank_sha256()) == 64
    assert "torch" not in _imports(ROOT / "scripts" / "error_pattern_data.py")
    assert "os" not in _imports(ROOT / "scripts" / "error_pattern_data.py")
    assert "growbrain" not in _imports(ROOT / "scripts" / "error_pattern_data.py")
    prereg = (ROOT / "reports" / "error_pattern_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (prereg, readme, architecture):
        for doi in DOIS:
            assert doi in text
        assert "10.1007/BF00487332" not in text
        assert "The held bank was not scored." in text
        assert "error-pattern-v1" in text
        assert data.bank_sha256() in text
        assert "must not be merged" in text
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
    assert "The cue guard was not scored." in readme
    assert "cannot be equated with mind-wandering" in readme
    source = (ROOT / "scripts" / "eval_error_pattern.py").read_text(encoding="utf-8")
    summary = source.split("def _implicit_summary")[1].split("def _shape_summary")[0]
    assert "PHONE" not in summary
    assert "birthday" not in summary.lower()
    assert "subconscious_score" not in source


def test_production_modules_are_unchanged():
    for rel, digest in PRODUCTION.items():
        raw = (ROOT / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest


def test_continuations_are_one_token():
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    data = _load(ROOT / "scripts" / "error_pattern_data.py", "error_pattern_data_tokens")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    for item in data.FIT_ITEMS + data.HELD_ITEMS:
        assert item["good"] == " " + item["answer"]
        assert item["bad"] == " " + item["foil"]
        for prompt in (item["prompt"], item["text"] + "\n" + item["prompt"]):
            prompt_ids = tok.encode(prompt, add_special_tokens=True)
            for continuation in (item["good"], item["bad"]):
                full = tok.encode(prompt + continuation, add_special_tokens=True)
                assert full[: len(prompt_ids)] == prompt_ids
                assert len(full) - len(prompt_ids) == 1


def test_one_eye_item_matches_the_preregistered_shape():
    module = _load(ROOT / "scripts" / "eval_error_pattern.py", "eval_error_pattern")
    row = module.score_eye("error-fit-000")
    assert row["shape_pass"] is True
    assert row["explicit_pass"] is True
    assert row["intrusion_event"] is False
    assert row["canonical"]["rif_strengths"] == [1.0, 0.95, 1.0]
    assert row["canonical"]["false_events"] == 0
    assert row["canonical"]["spontaneous_rate"] == 0.0625
    assert row["canonical"]["surfaces"]["phone_emotion"] == "+1-617-555-4862"
    assert row["canonical"]["surfaces"]["birthday_low"] == "April 1, 1984"
    assert row["canonical"]["off"] is None
