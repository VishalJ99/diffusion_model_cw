import torch
import random
import numpy as np
import os
from models.decoder_models import CNN, Unet
from models.diffusion_models import DDPM, GaussianBlurDM, FashionMNISTDM
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from noise_schedules import (
    const_noise_schedule,
    linear_noise_schedule,
    cosine_noise_schedule,
)
from piq import ssim, psnr
from typing import Callable, List


def seed_everything(seed: int) -> None:
    """
    Seed everything for reproducibility.

    Parameters
    ----------
    seed : int
        The seed to use for reproducibility.

    Returns
    -------
    None
    """
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


def safe_create_dir(dir_path: str) -> None:
    """
    Create a directory if it does not exist, if it exists and is not empty, print error
    and exit.

    TODO:
    Consider raising an exception instead of exiting.
    Consider adding a force flag to delete the directory if it exists and is not empty.

    Parameters
    ----------
    dir_path : str
        The path to the directory to create.

    Returns
    -------
    None
    """
    if not os.path.isdir(dir_path):
        print(f"[INFO] {dir_path} does not exit, making dir(s)")
        os.makedirs(dir_path)
    else:
        if os.listdir(dir_path):
            print(f"[ERROR] Files exist in {dir_path}... exiting")
            exit(1)


def fetch_model(model_key: str) -> torch.nn.Module:
    """
    Define a dictionary of models and return the model class based on the key.
    Used to fetch the model class based on the key in the config file by the
    train.py, test.py and calc_fid.py scripts.

    TODO: move to a general diffusion model class init.

    Parameters
    ----------
    model_key : str
        The key to use to fetch the model class.

    Returns
    -------
    torch.nn.Module
        The model class to use for training.
    """
    model_dict = {
        "cnn": CNN,
        "ddpm": DDPM,
        "unet": Unet,
        "gaussian_blur": GaussianBlurDM,
        "fashion_mnist": FashionMNISTDM,
    }

    return model_dict[model_key]


def fetch_noise_schedule(schedule_key: str) -> Callable:
    """
    Define a dictionary of noise schedules and return the noise schedule function based
    on the key. Allows for custom noise schedules to be passed in the config file and
    used by train.py, test.py and calc_fid.py scripts.

    TODO: move to a more intuitive place.

    Parameters
    ----------
    schedule_key : str
        The key to use to fetch the noise schedule function.

    Returns
    -------
    function
        The noise schedule function to use for training.
    """
    noise_schedule_dict = {
        "const": const_noise_schedule,
        "linear": linear_noise_schedule,
        "cosine": cosine_noise_schedule,
    }
    return noise_schedule_dict[schedule_key]


def make_cond_samples_plot(
    z_t: torch.tensor, visualise_ts: List[int], nrows: int, fs: int = 12
) -> plt:
    """
    Makes a nicely formatted conditional sampling plot for the diffusion model.
    Winsorises and normalises the images in the batch before plotting as plt.imshow
    clips the values to [0, 1] which can make the images look washed out.

    Parameters
    ----------
    z_t : torch.tensor
        The tensor of images to plot. First dimension should be contiguous in the time
        step parameter t. Shape should be [B, C, H, W].

    visualise_ts : List[int]
        The time steps to visualise.

    nrows : int
        The number of rows to use in the grid.

    fs : int
        The font size to use for the labels.

    Returns
    -------
    plt
        The matplotlib plot object.
    """

    # Normalise every image in the batch.
    for idx, img in enumerate(z_t):
        img = winsorize_tensor(img, 1, 99)
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

    # Add labels for "Degradation" and "Restoration"
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


def make_uncond_samples_plot(xh: torch.tensor, nrows: int) -> plt:
    """
    Make an unconditional sampling plot for the diffusion model.
    Winsorises and normalises the images in the batch before plotting as plt.imshow
    clips the values to [0, 1] which can make the images look washed out.

    Parameters
    ----------
    xh : torch.tensor
        The tensor of images to plot. Shape should be [B, C, H, W].

    nrows : int
        The number of rows to use in the grid.

    Returns
    -------
    plt
        The matplotlib plot object.
    """
    # Normalise every image in the batch.
    for idx, img in enumerate(xh):
        img = winsorize_tensor(img, 1, 99)
        xh[idx] = (img - img.min()) / (img.max() - img.min())

    # Make a grid of the images.
    grid = make_grid(xh, nrow=nrows)

    # Plot the grid.
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed
    ax.imshow(grid.cpu().numpy().transpose(1, 2, 0), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    return plt


def calc_image_quality_metrics(
    x: torch.tensor, x_hat: torch.tensor
) -> tuple[float, float, float]:
    """
    Calculate the image quality metrics for a batch of images and their reconstructions.

    Parameters
    ----------
    x : torch.tensor
        The original images. Shape should be [B, C, H, W].

    x_hat : torch.tensor
        The reconstructed images. Shape should be [B, C, H, W].

    Returns
    -------
    tuple[float, float, float]
        The average RMSE, SSIM and PSNR scores for the batch.
    """
    # TODO: vectorise this. Saving mean so dont need individual scores.
    # Can vectorise the normalisation step, metric functions are already vectorised.
    for idx in range(len(x)):
        src_img = x[idx]
        rec_img = x_hat[idx]

        # Normalise the images.
        src_img = (src_img - src_img.min()) / (src_img.max() - src_img.min())
        rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min())

        x[idx] = src_img
        x_hat[idx] = rec_img

    # Calculate the metrics.
    rmse_score = torch.sqrt(torch.mean((x - x_hat) ** 2)).item()
    ssim_score = ssim(x_hat, x, data_range=1.0).item()
    psnr_score = psnr(x_hat, x, data_range=1.0).item()

    metrics = (rmse_score, ssim_score, psnr_score)
    return metrics


# Test this! Chat GPT code!
# def normalise_batch(batch):
#     # Assumes batch is of shape [N, C, H, W]
#     min_vals = torch.min(batch.view(batch.size(0), -1), dim=1)[0].view(-1, 1, 1, 1)
#     max_vals = torch.max(batch.view(batch.size(0), -1), dim=1)[0].view(-1, 1, 1, 1)
#     return (batch - min_vals) / (max_vals - min_vals)


def winsorize_tensor(
    tensor: torch.tensor, lower_percentile: int, upper_percentile: int
) -> torch.tensor:
    """
    ChatGPT Generated Code.
    Winsorize a tensor by clipping the values to the lower and upper percentiles.
    Percentiles are expressed as values between 0 and 100.

    Parameters
    ----------
    tensor : torch.tensor
        The tensor to winsorize.

    lower_percentile : int
        The lower percentile to clip the values to.

    upper_percentile : int
        The upper percentile to clip the values to.

    Returns
    -------
    torch.tensor
        The winsorized tensor.
    """

    # Calculate the percentile values
    lower_bound = torch.quantile(tensor, lower_percentile / 100.0)
    upper_bound = torch.quantile(tensor, upper_percentile / 100.0)

    # Winsorize the tensor
    winsorized_tensor = torch.where(tensor < lower_bound, lower_bound, tensor)
    winsorized_tensor = torch.where(
        tensor > upper_bound, upper_bound, winsorized_tensor
    )

    return winsorized_tensor
