"""Elo math + tournament scheduling for the Ranking agent.

Standard Elo with K-factor scheduling: high K early (ratings move fast),
low K once a hypothesis has played a lot of matches (ratings stabilise).
"""
from __future__ import annotations

import random
from .state import Hypothesis


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability A beats B given their current Elo ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played: int) -> float:
    """Bigger swings while a hypothesis is new; calmer once it's seasoned."""
    if matches_played < 5:
        return 40.0
    if matches_played < 15:
        return 20.0
    return 10.0


def update_elo(winner: Hypothesis, loser: Hypothesis, draw: bool = False) -> None:
    """Update both hypotheses' ratings in place after a match."""
    exp_w = expected_score(winner.elo, loser.elo)
    score_w = 0.5 if draw else 1.0
    score_l = 0.5 if draw else 0.0
    k_w = k_factor(winner.matches_played)
    k_l = k_factor(loser.matches_played)
    winner.elo += k_w * (score_w - exp_w)
    loser.elo += k_l * (score_l - (1.0 - exp_w))
    winner.matches_played += 1
    loser.matches_played += 1


def schedule_matches(hypotheses: list[Hypothesis], n_matches: int = 6
                     ) -> list[tuple[Hypothesis, Hypothesis]]:
    """Pick informative pairings.

    Strategy (paper-faithful):
      1. Newer hypotheses (low matches_played) need matches more urgently.
      2. Compare similar-Elo pairs — they're the informative ones.
      3. Occasionally cross-tier to keep the ranking globally calibrated.
    """
    if len(hypotheses) < 2:
        return []

    sorted_h = sorted(hypotheses, key=lambda h: h.elo, reverse=True)
    pairs: list[tuple[Hypothesis, Hypothesis]] = []

    # Priority pass: every under-matched hypothesis gets a fight
    under_matched = [h for h in hypotheses if h.matches_played < 3]
    for h in under_matched:
        opponent = next((o for o in sorted_h
                         if o.id != h.id and abs(o.elo - h.elo) < 150), None)
        if opponent is None:
            opponent = random.choice([o for o in sorted_h if o.id != h.id])
        pairs.append((h, opponent))
        if len(pairs) >= n_matches:
            return pairs

    # Fill remaining slots with neighbor-pair matches
    for i in range(len(sorted_h) - 1):
        if len(pairs) >= n_matches:
            break
        pairs.append((sorted_h[i], sorted_h[i + 1]))

    return pairs[:n_matches]
