"""Good referee (skill 0.85): picks the better hypothesis on 85% of cross-tier
pairs. The A-tier hypothesis tops the board and spearman stays high (~+0.37).

Run it any of these ways:

    python3 -m benchmark.run_skill_085      # from the project root
    python3 benchmark/run_skill_085.py      # from the project root
    python3 run_skill_085.py                # from inside benchmark/
"""
import os
import sys

# Make `import benchmark...` work no matter which directory we're run from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmark.demo_walkthrough import main

if __name__ == "__main__":
    # 16 rounds so the Elo ratings actually converge — at the default 4 rounds
    # each hypothesis plays too few matches and even a good referee scores low.
    sys.argv = [sys.argv[0], "--skill", "0.85", "--rounds", "16"]
    main()
