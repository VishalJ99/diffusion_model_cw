import argparse
import os
from utils import (
    seed_everything,
    fetch_model,
    fetch_noise_schedule,
    calc_image_quality_metrics,
)
import torch
import sys
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import csv
import yaml
import numpy as np

# TODO: copy testing config / keep track of metric sample size since its not clear
# just from the metrics csv file how many samples were used.
# Consider moving metrics calc logic outside of test.py and into a separate script which
# also calculates the fid scores. Merge calc_fid and calc_metrics into one script.


def main(config):
    # Create the output directory for the run if it does not exist.
    output_dir = config["output_dir"]
    test_losses_csv_path = os.path.join(output_dir, "test_losses.csv")

    if os.path.isfile(test_losses_csv_path):
        print(f"Test loss file already exists in {output_dir}... Exiting.")
        sys.exit(1)

    if config["calc_metrics"]:
        metric_csv_path = os.path.join(output_dir, "test_metrics.csv")
        if os.path.isfile(metric_csv_path):
            print(f"Metrics file already exists in {output_dir}... Exiting.")
            sys.exit(1)
        else:
            with open(metric_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                # Write the header row
                writer.writerow(["avg_test_loss", "avg_rmse", "avg_ssim", "avg_psnr"])

    if os.path.exists(".git"):
        # Add the git hash to the config if the .git file exists.
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    # Save the config to the output directory for reproducibility.
    config_file = os.path.join(output_dir, "test_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    print("-" * 50)
    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Set the random seed for reproducibility.
    seed_everything(config["seed"])

    # Set the device to use for training. GPU -> MPS -> CPU.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

    pre_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    # Load the dataset.
    test_set = MNIST("./data", train=False, download=True, transform=pre_transforms)

    if config["quick_test"]:
        # Use a subset of the test set for quick testing.
        test_set = Subset(test_set, np.arange(0, 100))

    # Initialise the dataloader.
    test_loader = DataLoader(test_set, config["batch_size"], shuffle=False)

    # Fetch the models and noise schedule.
    decoder_model_class = fetch_model(config["decoder_model"])
    diffusion_model_class = fetch_model(config["diffusion_model"])
    noise_schedule = fetch_noise_schedule(config["noise_schedule"])

    # Create the noise schedule.
    T = config["T"]
    beta_t, alpha_t = noise_schedule(T, **config["custom_noise_schedule_params"])

    assert len(beta_t) - 1 == T, "Beta schedule must have T elements."

    # Initialise the models.
    decoder = decoder_model_class(**config["decoder_model_params"])
    model = diffusion_model_class(decoder, beta_t, alpha_t, device)

    # Load the model weights.
    model_weights_fpath = config["model_weights"]
    model.load_state_dict(torch.load(model_weights_fpath))
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    model, optim, test_loader = accelerator.prepare(model, optim, test_loader)

    total_test_loss = 0
    with torch.no_grad():
        # Training loop.
        model.eval()
        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        for idx, (x, _) in test_pbar:
            loss = model(x)
            test_pbar.set_description(f"loss: {loss.item():.3g}")
            total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"[INFO] Average test loss: {avg_test_loss:.3g}")
        with open(test_losses_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([avg_test_loss])
    if config["calc_metrics"]:
        print("[INFO] Calculating image quality metrics...")
        total_rmse = 0
        total_ssim = 0
        total_psnr = 0

        if not config["quick_test"] and config["metric_sample_size"]:
            # Calculate metrics on a subset of the test set.
            assert (
                config["metric_sample_size"] <= len(test_set)
            ), "Metric sample size must be less than or equal to the test set size."
            # Not needed if quick test already set.
            test_set = Subset(test_set, np.arange(0, config["metric_sample_size"]))

        # Batch size fixed as too large a batch size can cause memory issues.
        test_loader = DataLoader(test_set, 512, shuffle=False)

        test_loader = accelerator.prepare(test_loader)
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for idx, (x, _) in pbar:
                x_hat = model.cond_sample(x, [], accelerator.device)
                # cond_sample returns x since its typically used to generate plots.
                # Want the last batch size number of elements along the first axis.
                # to get the reconstructed x_hat.
                x_hat = x_hat[-x.shape[0] :]

                # These are averages across the batch.
                rmse, ssim, psnr = calc_image_quality_metrics(x, x_hat)
                total_rmse += rmse
                total_ssim += ssim
                total_psnr += psnr

            avg_rmse = total_rmse / len(test_loader)
            avg_ssim = total_ssim / len(test_loader)
            avg_psnr = total_psnr / len(test_loader)
        print(f"[INFO] Average RMSE: {avg_rmse:.3g}")
        print(f"[INFO] Average SSIM: {avg_ssim:.3g}")
        print(f"[INFO] Average PSNR: {avg_psnr:.3g}")
        row = [avg_test_loss, avg_rmse, avg_ssim, avg_psnr]
        with open(metric_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDPM model.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the yaml train config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(config["train_config_path"], "r") as f:
        training_config = yaml.safe_load(f)

    # Add the relevant keys from the training config to the test config.
    # Prevents duplicate entries in the test config.
    config["decoder_model"] = training_config["decoder_model"]
    config["decoder_model_params"] = training_config["decoder_model_params"]
    config["diffusion_model"] = training_config["diffusion_model"]
    config["noise_schedule"] = training_config["noise_schedule"]
    config["T"] = training_config["T"]
    config["custom_noise_schedule_params"] = training_config[
        "custom_noise_schedule_params"
    ]
    main(config)
