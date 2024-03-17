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
    # TODO: move to a general diffusion model class init
    model_dict = {
        "cnn": CNN,
        "ddpm": DDPM,
    }

    return model_dict[model_key]


def fetch_beta_schedule(schedule_key):
    fetch_beta_schedule_dict = {"linear": linear_beta_schedule}
    return fetch_beta_schedule_dict[schedule_key]


def linear_beta_schedule(T, beta_1, beta_2):
    # NOTE: T must be the first argument for it to be initialised properly
    # in the train.py script.
    """Returns pre-computed schedules for DDPM sampling
    with a linear noise schedule."""
    assert beta_1 < beta_2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta_2 - beta_1) * np.arange(0, T, dtype=np.float32)
    beta_t /= T - 1
    beta_t += beta_1

    return beta_t


def make_cond_samples_plot(z_t, visualise_ts, nrows, fs=8):
    # Normalise every image in the batch.
    for idx, img in enumerate(z_t):
        z_t[idx] = (img - img.min()) / (img.max() - img.min())

    # Make a grid of the images.
    grid = make_grid(z_t, nrow=nrows)

    # Plot the grid.
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    # Draw the variable names on the left hand border of the image.
    image_height = z_t.shape[2]
    pad = 2
    vpad = image_height // 2

    plt.text(
        -fs,
        0 * (image_height + pad) + vpad,
        "$x$",
        ha="right",
        va="center",
        fontsize=fs,
    )

    for i, t in enumerate(visualise_ts, start=1):
        plt.text(
            -fs,
            i * (image_height + pad) + vpad,
            f"$z_{{{t}}}$",
            ha="right",
            va="center",
            fontsize=fs,
        )

    visualise_ts = visualise_ts[::-1]

    for i, t in enumerate(visualise_ts, start=len(visualise_ts) + 1):
        plt.text(
            -fs,
            i * (image_height + pad) + vpad,
            f"$z_{{{t}}}$",
            ha="right",
            va="center",
            fontsize=fs,
        )

    i = 2 * len(visualise_ts) + 1
    plt.text(
        -fs,
        i * (image_height + pad) + vpad,
        "$z_0$",
        ha="right",
        va="center",
        fontsize=fs,
    )

    return plt
