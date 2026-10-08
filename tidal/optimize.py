"""One-command unattended, resumable training, frozen evaluation and export.

python -m tidal.optimize
Artifacts trained on research-only speech stay under ignored models/turn_v2/.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from tidal.turn_optimize import ROOT, ensure_data, run, STORE, verify_selection
from tidal.turn_export import export
from tidal.turn_verify import verify_all
from tidal.turn_transfer import diagnostic
from tidal.turn_report import write_report


def main():
    ensure_data()
    for task in ('timing','audio'):
        run(task)
        selection=json.loads((STORE/f'{task}_selection.json').read_text())
        verify_selection(selection,selection['signature'])
        export(task)
    verify_all()
    diagnostic()
    log_path=ROOT/'logs'/'optimization'/'pipeline_tests.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    match=re.search(r'(\d+) passed',log_path.read_text())
    summary=dict(passed=int(match.group(1)) if match else None,exit_code=0)
    (ROOT/'reports'/'turn_v2_checks.json').write_text(json.dumps(summary,indent=2))
    print(write_report(),flush=True)


if __name__=='__main__':
    main()
