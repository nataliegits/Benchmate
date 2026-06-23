#!/usr/bin/env bash
# Good referee (skill 0.85): picks the better hypothesis on 85% of cross-tier
# pairs. The A-tier hypothesis should top the board and spearman stays high.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m benchmark.demo_walkthrough --skill 0.85
