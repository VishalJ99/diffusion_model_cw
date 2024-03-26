import numpy as np

"""
Can define any custom noise schedule here and add it to the dict in
fetch_noise_schedule function in utils.py to use it in the training script.
Just pass it as a string in the config file and it will be fetched.

To define a custom noise schedule, create a function that takes T as the first
first argument for it to be initialised properly, any kwargs can be passed
after in the config.

All Implementations for beta_t and alpha_t are all of length T+1
to make the indexing more intuitive. This way, beta_t[t] is the value of beta at time t.

All functions need to return np arrays of dtype np.float32, by default
np creates arrays of dtype np.float64 which can not be converted to tensors
on the mps device.
"""


def stable_cumprod(x: np.ndarray) -> np.ndarray:
    """
    Performs a cumulative product of x, but in logspace with numerical stability.

    Parameters
    ----------
    x : np.ndarray
        The input array to perform the cumulative product on.

    Returns
    -------
    np.ndarray
        The cumulative product of x.
    """
    return np.exp(np.cumsum(np.log(x)))


def linear_noise_schedule(
    T: int, beta_1: int, beta_2: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear noise schedule for beta_t and alpha_t.

    Parameters
    ----------
    T : int
        The total number of timesteps.
    beta_1 : int
        The initial value of beta.
    beta_2 : int
        The final value of beta.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The beta_t and alpha_t values for the noise schedule.
    """
    assert beta_1 < beta_2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta_2 - beta_1) * np.arange(0, T, dtype=np.float32)
    beta_t /= T - 1
    beta_t += beta_1

    alpha_t = stable_cumprod(1 - beta_t)

    beta_t = np.insert(beta_t, 0, 0)
    alpha_t = np.insert(alpha_t, 0, 1)

    return beta_t, alpha_t


def cosine_noise_schedule(
    T: int, beta_max: int = 0.9, s: int = 0.008
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a cosine noise schedule for the given parameters. Used in
    Paper - https://arxiv.org/abs/2102.09672 to give better results for training
    diffusion models.
    TODO: Calculate optimal value for s for MNIST data.

    Parameters
    ----------
    T : int
        The total number of time steps.
    beta_max : int, optional
        The maximum value for beta_t. Defaults to 0.9.
    s : int, optional
        A small constant to prevent beta_t from becoming too large for small t.
        Should be set so sqrt(beta_1) = 1 / pixel bin width. Defaults to 0.008.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The beta_t and alpha_t values for the noise schedule.
    """

    def f(t, s):
        freq = ((t / T) + s) / (4 * (1 + s))
        return np.cos(2 * np.pi * freq) ** 2

    alpha_t = [f(t, s) / f(0, s) for t in range(T + 1)]
    beta_t = [1 - (alpha_t[t] / alpha_t[t - 1]) for t in range(1, T + 1)]

    beta_t = np.insert(beta_t, 0, 0).astype(np.float32)
    alpha_t = np.asarray(alpha_t).astype(np.float32)

    # Clip beta_t values greater than beta_max since beta explodes at the end.
    beta_t = np.clip(beta_t, 0.0, beta_max)

    return beta_t, alpha_t


def const_noise_schedule(T: int, beta: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Constant noise schedule for beta_t and alpha_t.

    Parameters
    ----------
    T : int
        The total number of timesteps.

    beta : int
        The constant value of beta.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The beta_t and alpha_t values for the noise schedule.
    """
    beta_t = np.full(T + 1, beta, dtype=np.float32)
    alpha_t = stable_cumprod(1 - beta_t)
    return beta_t, alpha_t
