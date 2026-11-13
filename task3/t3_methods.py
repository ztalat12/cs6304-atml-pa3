"""Task 3 selection rules over a visible candidate prefix (Steps 2, 3, 5).

All rules see only the candidates revealed so far (`idx`: rollout indices in the current random order).
Ties: answer votes and highest-score selection go to the EARLIEST supporting rollout in the current order;
tied RWC scores get average ranks. At N = 1 every rule returns the only candidate.
A rollout with no tokens has undefined scores (NaN); it ranks below every scored rollout.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EPS = 1e-8


class QuestionPool:
    """The 32 rollouts of one question: per-rollout scalars and (VLM) per-token kl / visual-share arrays."""

    def __init__(self, T, L, K, groups, correct, no_answer, V=None, kl=None, v=None, offsets=None):
        self.T = np.asarray(T, np.int64)
        self.L, self.K = np.asarray(L, float), np.asarray(K, float)
        codes = {}
        self.g = np.array([codes.setdefault(x, len(codes)) for x in groups], np.int64)   # group code per rollout
        self.group_names = list(codes)
        self.correct = np.asarray(correct, bool)
        self.no_answer = np.asarray(no_answer, bool)
        self.V = None if V is None else np.asarray(V, float)
        self.kl, self.v, self.off = kl, v, offsets                                        # rollout r: [off[r], off[r+1])

    # ---------------------------------------------------------------- scores that depend on the visible set
    def visconf(self, idx):
        """VisConf on the visible candidates: pool their token KLs, calibrate with the 5th/95th percentiles
        (linear interpolation), clip to [0, 1], average per rollout (C), Z = V * C; tokenwise Z = mean(c * v).
        Returns (Z, Z_token, C, (q05, q95)) as arrays over `idx`."""
        segs = [(self.off[r], self.off[r + 1]) for r in idx]
        lens = np.array([b - a for a, b in segs])
        if lens.sum() == 0:
            nan = np.full(len(idx), np.nan)
            return nan, nan, nan, (np.nan, np.nan)
        k = np.concatenate([self.kl[a:b] for a, b in segs]).astype(np.float64)
        v = np.nan_to_num(np.concatenate([self.v[a:b] for a, b in segs]).astype(np.float64), nan=0.0)
        q05, q95 = np.percentile(k, [5, 95])
        c = np.clip((k - q05) / max(q95 - q05, EPS), 0.0, 1.0)
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        safe = np.minimum(starts, len(k) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(lens > 0, np.add.reduceat(c, safe) / np.maximum(lens, 1), np.nan)
            Zt = np.where(lens > 0, np.add.reduceat(c * v, safe) / np.maximum(lens, 1), np.nan)
        Z = self.V[idx] * C
        return Z, Zt, C, (float(q05), float(q95))


def _clean(s):
    s = np.asarray(s, float)
    return np.where(np.isnan(s), -np.inf, s)


def first(idx):
    return int(idx[0])


def self_consistency(idx, g):
    """One vote per rollout (NO_ANSWER is a group too). Returns (selected rollout, votes of winner, unique leader?)."""
    gi = g[idx]
    uniq, first_pos, counts = np.unique(gi, return_index=True, return_counts=True)
    top = counts.max()
    tied = np.flatnonzero(counts == top)
    win = tied[np.argmin(first_pos[tied])]
    return int(idx[first_pos[win]]), int(top), len(tied) == 1


def highest(idx, score):
    s = _clean(score)
    return int(idx[int(np.flatnonzero(s == s.max())[0])])


def rwc(idx, score, g):
    """Rank-weighted consensus. rank 1 = best (average ranks for ties); w = (N - r) / (N - 1);
    W(a) = sum of w over rollouts with answer a. Returns (selected rollout, margin), where
    margin = (largest W - second largest W) / total W (runner-up 0 when there is one answer group)."""
    n = len(idx)
    if n == 1:
        return int(idx[0]), 1.0
    s = _clean(score)
    r = rankdata(-s, method="average")
    w = (n - r) / (n - 1)
    gi = g[idx]
    uniq, first_pos, inv = np.unique(gi, return_index=True, return_inverse=True)
    W = np.bincount(inv, weights=w, minlength=len(uniq))
    top = W.max()
    tied = np.flatnonzero(np.isclose(W, top, rtol=0, atol=1e-12))
    win = tied[np.argmin(first_pos[tied])]
    srt = np.sort(W)[::-1]
    second = srt[1] if len(srt) > 1 else 0.0
    total = W.sum()
    margin = float((srt[0] - second) / total) if total > 0 else 0.0
    return int(idx[first_pos[win]]), margin


def methods_for(vlm):
    base = ["FirstAns", "SC", "MaxLogProb", "MaxSelfCert", "SelfCert+RWC", "Shortest"]
    if vlm:
        base += ["MaxVision", "Vision+RWC", "MaxVisConf", "VisConf+RWC", "MaxVisConfTok", "VisConfTok+RWC"]
    return base


def select_all(pool: QuestionPool, idx, vlm):
    """{method: selected rollout} for one visible prefix."""
    idx = np.asarray(idx)
    out = {"FirstAns": first(idx), "SC": self_consistency(idx, pool.g)[0],
           "MaxLogProb": highest(idx, pool.L[idx]), "MaxSelfCert": highest(idx, pool.K[idx]),
           "SelfCert+RWC": rwc(idx, pool.K[idx], pool.g)[0], "Shortest": highest(idx, -pool.T[idx].astype(float))}
    if vlm:
        Z, Zt, _, _ = pool.visconf(idx)
        out.update({"MaxVision": highest(idx, pool.V[idx]), "Vision+RWC": rwc(idx, pool.V[idx], pool.g)[0],
                    "MaxVisConf": highest(idx, Z), "VisConf+RWC": rwc(idx, Z, pool.g)[0],
                    "MaxVisConfTok": highest(idx, Zt), "VisConfTok+RWC": rwc(idx, Zt, pool.g)[0]})
    return out


def stop_stats(pool: QuestionPool, idx, vlm):
    """What the stopping controllers need at a checkpoint: SC leader (unique? NO_ANSWER? share), SC selection,
    and the RWC winner + margin with the quality score (self-certainty for MATH, VisConf for MathVista)."""
    idx = np.asarray(idx)
    sel, votes, unique = self_consistency(idx, pool.g)
    q = pool.visconf(idx)[0] if vlm else pool.K[idx]
    rsel, margin = rwc(idx, q, pool.g)
    return {"sc_sel": sel, "share": votes / len(idx), "unique": unique, "leader_no_answer": bool(pool.no_answer[sel]),
            "agree": bool(pool.g[sel] == pool.g[rsel]), "margin": margin, "cost": int(pool.T[idx].sum())}
