import json
from types import SimpleNamespace
import pytest
pytest.importorskip("huggingface_hub")
from tidal.public_data import download


def setup_download(monkeypatch, tmp_path):
    calls = []
    info = SimpleNamespace(sha='pinned-sha', siblings=[])
    class Api:
        def dataset_info(self, repo, **kwargs):
            calls.append(('info', repo, kwargs))
            return info
    monkeypatch.setattr(download, 'HfApi', Api)
    monkeypatch.setattr(download, 'DATA', tmp_path)
    monkeypatch.setattr(download, 'DATASETS', {'sample': dict(repo='owner/data', revision='requested-sha', license='test', files=['a.txt'])})
    return calls


def test_download_pins_all_files_to_resolved_revision(monkeypatch, tmp_path):
    calls = setup_download(monkeypatch, tmp_path)
    monkeypatch.setattr(download, 'hf_hub_download', lambda *args, **kw: calls.append(('file', args, kw)))
    download.main(['sample'])
    assert calls[0][2]['revision'] == 'requested-sha'
    assert calls[1][2]['revision'] == 'pinned-sha'
    result = json.loads((tmp_path / 'sample' / '_done.json').read_text())
    assert result['revision'] == 'pinned-sha'
    assert result['ok'] == result['files'] == 1


def test_failed_download_does_not_report_success(monkeypatch, tmp_path):
    setup_download(monkeypatch, tmp_path)
    def fail(*args, **kw):
        raise OSError('unavailable')
    monkeypatch.setattr(download, 'hf_hub_download', fail)
    monkeypatch.setattr(download.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='Incomplete dataset'):
        download.main(['sample'])
    result = json.loads((tmp_path / 'sample' / '_done.json').read_text())
    assert result['ok'] == 0 and result['files'] == 1
