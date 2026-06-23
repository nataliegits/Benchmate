#!/usr/bin/env bash
# Noisy referee (skill 0.50): pure coin-flip on every pair. The ranking
# collapses — a weak hypothesis can win and spearman drops toward zero.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m benchmark.demo_walkthrough --skill 0.50
