import torch
import random
import numpy as np
import os
from models import CNN, DDPM
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def seed_everything(seed):
    # Set seed for all packages with built-in pseudo-raiAndom generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using CUDA, set also the below for determinism.
    if torch.cuda.is_available():
        # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

        # Ensures that the CUDA convolution uses deterministic algorithms.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        print(f"[INFO] {dir_path} does not exit, making dir(s)")
        os.makedirs(dir_path)
    else:
        if os.listdir(dir_path):
            print(f"[ERROR] Files exist in {dir_path}... exiting")
            exit(1)


def fetch_model(model_key):
    # TODO: move to a general diffusion model class init.
    model_dict = {
        "cnn": CNN,
        "ddpm": DDPM,
    }

    return model_dict[model_key]


def fetch_noise_schedule(schedule_key):
    # TODO: move to a more intuitive place.
    noise_schedule_dict = {
        "const": const_noise_schedule,
        "linear": linear_noise_schedule,
        "cosine": cosine_noise_schedule,
    }
    return noise_schedule_dict[schedule_key]


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
    alpha_t = np.exp(
        np.cumsum(np.log(1 - beta_t))
    )  # Cumprod in log-space (better precision).

    # Insert value for t=0 at the beginning of beta_t and alpha_t to make the indexing
    # more intuitive. This way, beta_t[t] is the value of beta at time t.
    beta_t = np.insert(beta_t, 0, 0)
    alpha_t = np.insert(alpha_t, 0, 1)

    return beta_t, alpha_t


def cosine_noise_schedule(T, beta_max=0.1, s=0.008):
    # NOTE: T must be the first argument for it to be initialised properly
    # in the train.py script.
    # https://arxiv.org/abs/2102.09672

    def f(t, s):
        freq = ((t / T) + s) / (4 * (1 + s))
        return np.cos(2 * np.pi * freq) ** 2

    alpha_t = [f(t, s) / f(0, s) for t in range(T + 1)]
    beta_t = [1 - (alpha_t[t] / alpha_t[t - 1]) for t in range(1, T + 1)]

    # Insert value for t=0 at the beginning of beta_t to make the indexing
    # more intuitive. This way, beta_t[t] is the value of beta at time t.
    beta_t = np.insert(beta_t, 0, 0).astype(np.float32)
    alpha_t = np.asarray(alpha_t).astype(np.float32)

    # Clip beta_t vals greater than beta_max since beta explodes at the end.
    beta_t = np.clip(beta_t, 0.0, beta_max)

    return beta_t, alpha_t


def const_noise_schedule(T, beta):
    # NOTE: T must be the first argument for it to be initialised properly
    # in the train.py script.
    """Returns pre-computed schedules for DDPM sampling
    with a constant noise schedule."""
    beta_t = np.full(T + 1, beta, dtype=np.float32)
    alpha_t = np.exp(
        np.cumsum(np.log(1 - beta_t))
    )  # Cumprod in log-space (better precision)

    return beta_t, alpha_t


def make_cond_samples_plot(z_t, visualise_ts, nrows, fs=12):
    # Normalise every image in the batch.
    for idx, img in enumerate(z_t):
        z_t[idx] = (img - img.min()) / (img.max() - img.min())

    # Make a grid of the images.
    grid = make_grid(z_t, nrow=nrows)

    # Plot the grid.
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed
    ax.imshow(grid.cpu().numpy().transpose(1, 2, 0), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    # Draw the variable names on the left hand border of the image.
    image_height = z_t.shape[2]
    pad = 2
    vpad = image_height // 2

    # Adding text annotations
    for i, t in enumerate([0] + visualise_ts + visualise_ts[::-1] + [0], start=0):
        y_pos = i * (image_height + pad) + vpad
        if i == 0:
            label = "$x$"
        elif i <= len(visualise_ts):
            label = f"$z_{{{t}}}$"
        else:
            label = f"$\hat{{x}}_{{{t}}}$"
        ax.text(
            -fs,
            y_pos,
            label,
            ha="right",
            va="center",
            fontsize=fs,
            transform=ax.transData,
        )

    # Draw curly braces and add labels for "Degradation" and "Restoration"
    # Note: This requires manual adjustment for positioning
    degradation_label_pos_y = (len(visualise_ts) / 2) * (image_height + pad) + vpad
    restoration_label_pos_y = ((len(visualise_ts) * 2) / 2) * (image_height + pad) * 2
    ax.text(
        -fs * 2,
        degradation_label_pos_y,
        "Degradation",
        ha="right",
        va="center",
        fontsize=fs + 2,
        rotation=90,
        transform=ax.transData,
    )
    ax.text(
        -fs * 2,
        restoration_label_pos_y,
        "Restoration",
        ha="right",
        va="center",
        fontsize=fs + 2,
        rotation=90,
        transform=ax.transData,
    )
    plt.tight_layout()
    return plt
