"""Noisy referee (skill 0.50): pure coin-flip on every pair. The ranking
collapses — a weak hypothesis can win and spearman drops toward zero (~+0.03).

Run it any of these ways:

    python3 -m benchmark.run_skill_050      # from the project root
    python3 benchmark/run_skill_050.py      # from the project root
    python3 run_skill_050.py                # from inside benchmark/
"""
import os
import sys

# Make `import benchmark...` work no matter which directory we're run from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmark.demo_walkthrough import main

if __name__ == "__main__":
    # Same 16 rounds as the good-referee run, so the only thing that differs is
    # the referee's skill — the noise alone drives the spearman to the floor.
    sys.argv = [sys.argv[0], "--skill", "0.50", "--rounds", "16"]
    main()
