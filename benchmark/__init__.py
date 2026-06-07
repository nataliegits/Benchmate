"""Benchmate benchmarking toolkit.

Four pillars, mapped to modules:

  1. Trustworthy leaderboard   -> simulate.py + metrics.py   (free, no API)
  2. Validate vs ground truth  -> gold_set.py + run_benchmark.py validate
  3. Compare configurations    -> run_benchmark.py compare
  4. Better / fairer judge     -> fair_judge.py + judge_eval.py

Start with `python -m benchmark.run_benchmark simulate` — it's free and
tells you whether your current match budget and K-factor produce a
ranking you can trust.
"""
