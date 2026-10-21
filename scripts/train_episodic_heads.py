#!/usr/bin/env python3
"""Score the episodic retrieval adapter and inject MLP. CPU, no checkpoint.

  python3 scripts/train_episodic_heads.py
  python3 -m tidal.episodic_train

Prints ret@1 against raw cosine and whether those weights qualify to load.
The default store stays at a zero last layer either way.
"""
from tidal.episodic_train import main

if __name__ == "__main__":
    main()
