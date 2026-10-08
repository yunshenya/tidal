"""Prepare local judgement sheets from arrived events; never manufacture policy labels."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tidal.task_heads import HeadEvent, context_at
from tidal.head_data import read_annotations


def prepare(events_path, output):
    histories = {}; records = []; previous = {}
    for line in Path(events_path).read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line); conv = row['conversation']
        if not isinstance(conv, str) or not conv: raise ValueError('Conversation required')
        event = HeadEvent(**row['event'])
        if conv in previous and event.ts < previous[conv]: raise ValueError('Events must arrive in order')
        history = histories.setdefault(conv, []); history.append(event)
        history[:] = context_at(history, event.ts); previous[conv] = event.ts
        if event.role == 'self': continue
        for task in ('action_policy', 'eot_semantic', 'recheck_policy'):
            records.append(dict(task=task, conversation=conv, decision_time=event.ts,
                                context=[asdict(e) for e in history], label=None,
                                label_source='human', annotator='pending', self_state={'speaking': False}))
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    return dict(decisions=len(records), conversations=len(histories), labels_created=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--events', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('validate'); p.add_argument('path')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.events, args.out)
    else: _, result = read_annotations(args.path)
    print(json.dumps(result))
