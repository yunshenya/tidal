#!/usr/bin/env python3
"""Convert local turngate review JSONL into the unified causal event format (see tidal/turngate.py).

Raw JSONL stays under data/private/ (gitignored). Weak/auto rows are excluded unless
--include-auto is given, and then only for groups that have no val/test rows.
"""
import argparse, json
from pathlib import Path
import numpy as np
from tidal.turngate import HEADS, assign, read_jsonl, to_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="review JSONL files (human + optional auto-labeled)")
    ap.add_argument("--output", type=Path, default=Path("data/private/processed/turngate_group.parquet"))
    ap.add_argument("--salt-file", type=Path, default=Path(".secrets/pseudo_salt"))
    ap.add_argument("--include-auto", action="store_true", help="add model-labeled rows to train (weight 0.25)")
    ap.add_argument("--block-days", type=float, default=7.0)
    a = ap.parse_args()
    rows = [r for p in a.inputs for r in read_jsonl(p)]
    d = to_frame(assign(rows, a.block_days, include_weak=a.include_auto), a.salt_file.read_bytes().strip())
    a.output.parent.mkdir(parents=True, exist_ok=True); d.to_parquet(a.output, index=False)
    print(json.dumps({"rows": len(d), "samples": int(d.conv.nunique()),
                      "sources": d.groupby("source").conv.nunique().to_dict(),
                      "splits": d.groupby("split").conv.nunique().to_dict(),
                      "labels": {h: int(np.isfinite(d[h]).sum()) for h in HEADS}}, indent=2))


if __name__ == "__main__": main()
