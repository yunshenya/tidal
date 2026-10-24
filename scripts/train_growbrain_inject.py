"""Retrain the growbrain key center and inject MLP at d_model.

The stage0 checkpoint is required. This script does not write a zero MLP
and does not copy ``llm_inject_smollm2_135m.pt``.
"""
from __future__ import annotations

from tidal.backend import GROWBRAIN_D_MODEL, GROWBRAIN_MEM_DIM, load_growbrain_stage0


def main() -> None:
    if int(GROWBRAIN_D_MODEL) == int(GROWBRAIN_MEM_DIM):
        raise RuntimeError("d_model and mem_dim collapsed; the store width must stay d_model")
    load_growbrain_stage0(device="cpu")
    raise RuntimeError("loader returned without a checkpoint; refusing to write weights")


if __name__ == "__main__":
    main()
