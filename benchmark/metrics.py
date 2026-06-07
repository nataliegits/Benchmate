"""Ranking-quality metrics. Pure Python (no scipy) so this runs anywhere.

Everything here works on *orderings* (a list of item ids, best first) or on
{id: score} dicts. These are the yardsticks the simulator and the live
validators use to say whether one leaderboard is better or more stable
than another.
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence


# --------------------------------------------------------------------------
# Rank correlation — "do these two rankings agree?"
# --------------------------------------------------------------------------

def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional (average) ranks, ties shared. Rank 1 = smallest value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation between two aligned score lists (-1..1).

    1.0 = identical ordering, 0 = unrelated, -1 = reversed.
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-b: fraction of concordant minus discordant pairs."""
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = (a[i] > a[j]) - (a[i] < a[j])
            sb = (b[i] > b[j]) - (b[i] < b[j])
            prod = sa * sb
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


# --------------------------------------------------------------------------
# Top-k agreement — "do the rankings agree on the *winners*?"
# --------------------------------------------------------------------------

def topk_jaccard(order_a: Sequence[Hashable], order_b: Sequence[Hashable],
                 k: int) -> float:
    """Jaccard overlap of the top-k sets of two orderings (0..1).

    This is usually what you actually care about — you act on the top few
    hypotheses, not the long tail.
    """
    sa, sb = set(order_a[:k]), set(order_b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def top1_agreement(orders: Sequence[Sequence[Hashable]]) -> float:
    """Across many orderings, the fraction that share the single most common
    #1. A blunt but honest 'would the demo show the same winner?' number."""
    if not orders:
        return float("nan")
    from collections import Counter
    winners = Counter(o[0] for o in orders if o)
    top = winners.most_common(1)[0][1]
    return top / len(orders)


# --------------------------------------------------------------------------
# Convergence — "has the ranking stopped moving?"
# --------------------------------------------------------------------------

def rank_churn(order_prev: Sequence[Hashable],
               order_now: Sequence[Hashable]) -> float:
    """Mean absolute change in rank position between two snapshots, in
    'positions per item'. 0 = nothing moved. Track this across matches; once
    it flatlines near 0, the leaderboard has converged."""
    pos_prev = {h: i for i, h in enumerate(order_prev)}
    pos_now = {h: i for i, h in enumerate(order_now)}
    common = set(pos_prev) & set(pos_now)
    if not common:
        return float("nan")
    return sum(abs(pos_prev[h] - pos_now[h]) for h in common) / len(common)


# --------------------------------------------------------------------------
# Judge sanity — "is the referee self-consistent?"
# --------------------------------------------------------------------------

def transitivity_violations(beats: dict[tuple[Hashable, Hashable], Hashable]
                            ) -> tuple[int, int]:
    """Count intransitive triples (A>B, B>C, but C>A) among recorded results.

    `beats` maps a pair to its winner id (orientation doesn't matter). Returns
    (violations, fully_decided_triples). Each unordered triple is counted once:
    a triple is a violation iff all three members have exactly one win within it
    (a 3-cycle) rather than the {2,1,0} pattern of a transitive triple. A high
    ratio means the judge contradicts itself — the ranking can't be trusted no
    matter how many matches you run.
    """
    import itertools

    wins: dict[Hashable, set] = {}
    items: set = set()
    for (x, y), w in beats.items():
        items.update((x, y))
        loser = y if w == x else x
        wins.setdefault(w, set()).add(loser)

    def winner(x, y):
        if y in wins.get(x, ()):
            return x
        if x in wins.get(y, ()):
            return y
        return None

    violations = triples = 0
    for a, b, c in itertools.combinations(sorted(items, key=str), 3):
        results = [winner(a, b), winner(b, c), winner(a, c)]
        if None in results:
            continue                      # triple not fully decided
        triples += 1
        wc = {a: 0, b: 0, c: 0}
        for w in results:
            wc[w] += 1
        if set(wc.values()) == {1}:       # every member won exactly once -> cycle
            violations += 1
    return violations, triples
