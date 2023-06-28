import numpy as np
from scipy.stats import binom

# Taken from Learn-then-Test code base:
# https://github.com/aangelopoulos/ltt


def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)

    def h1(y, mu):
        with np.errstate(divide='ignore'):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))

    hoeffding_p_value = np.exp(-n * h1(min(r_hat, alpha), alpha))
    return min(bentkus_p_value, hoeffding_p_value)


def bonferroni_search(p_values, delta, downsample_factor):
    N = p_values.shape[0]
    N_coarse = max(int(p_values.shape[0] / downsample_factor), 1)
    # Downsample, making sure to include the endpoints.
    coarse_indexes = set(range(0, N, downsample_factor))
    coarse_indexes.update({0, N - 1})
    R = set()
    for idx in coarse_indexes:
        _idx = idx
        while _idx >= 0 and _idx < N and p_values[_idx] < delta / N_coarse:
            R.update({_idx})
            _idx = _idx + 1
    return np.array(list(R))
