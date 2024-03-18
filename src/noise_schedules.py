import numpy as np
from utils import stable_cumprod

# NOTE: T must be the first argument for it to be initialised properly
# in the train.py script.
# Implementations for beta_t and alpha_t are all of length T+1 to make the indexing
# more intuitive. This way, beta_t[t] is the value of beta at time t.
# All functions need to return np arrays of dtype np.float32, by default
# np creates arrays of dtype np.float64 which can not be converted to tensors
# on the mps device.


def linear_noise_schedule(T, beta_1, beta_2):
    # NOTE: T must be the first argument for it to be initialised properly
    # in the train.py script.
    """Returns pre-computed schedules for DDPM sampling
    with a linear noise schedule."""
    assert beta_1 < beta_2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Calculate beta_t from t.
    beta_t = (beta_2 - beta_1) * np.arange(0, T, dtype=np.float32)
    beta_t /= T - 1
    beta_t += beta_1

    # Calculate alpha_t from beta_t.
    alpha_t = stable_cumprod(1 - beta_t)

    beta_t = np.insert(beta_t, 0, 0)
    alpha_t = np.insert(alpha_t, 0, 1)

    return beta_t, alpha_t


def cosine_noise_schedule(T, beta_max=0.1, s=0.008):
    # https://arxiv.org/abs/2102.09672

    def f(t, s):
        freq = ((t / T) + s) / (4 * (1 + s))
        return np.cos(2 * np.pi * freq) ** 2

    alpha_t = [f(t, s) / f(0, s) for t in range(T + 1)]
    beta_t = [1 - (alpha_t[t] / alpha_t[t - 1]) for t in range(1, T + 1)]

    beta_t = np.insert(beta_t, 0, 0).astype(np.float32)
    alpha_t = np.asarray(alpha_t).astype(np.float32)

    # Clip beta_t vals greater than beta_max since beta explodes at the end.
    beta_t = np.clip(beta_t, 0.0, beta_max)

    return beta_t, alpha_t


def const_noise_schedule(T, beta):
    beta_t = np.full(T + 1, beta, dtype=np.float32)
    alpha_t = stable_cumprod(1 - beta_t)
    return beta_t, alpha_t
