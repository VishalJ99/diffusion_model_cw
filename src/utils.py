import torch
import random
import numpy as np
import os
from models import CNN, DDPM


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
    """Returns pre-computed schedules for DDPM sampling with a linear noise schedule."""
    assert beta_1 < beta_2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta_2 - beta_1) * torch.arange(0, T, dtype=torch.float32) \
        / (T - 1) + beta_1

    return beta_t
