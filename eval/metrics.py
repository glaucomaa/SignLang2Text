import numpy as np
import numba as nb


@nb.njit(cache=True)
def _levenshtein(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m]


def char_error_rate(ref: str, hyp: str) -> float:
    return _levenshtein(list(ref), list(hyp)) / max(1, len(ref))


def word_error_rate(ref: str, hyp: str) -> float:
    return _levenshtein(ref.split(), hyp.split()) / max(1, len(ref.split()))
