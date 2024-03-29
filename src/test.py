import argparse
import os
from utils import (
    seed_everything,
    fetch_model,
    fetch_noise_schedule,
    calc_image_quality_metrics
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
    else:
        with open(test_losses_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["id_", "loss"])

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
    # Want to return individual losses for each image so batch size is 1.
    test_loader = DataLoader(
        test_set, 1, shuffle=False
    )

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
    model_weights_fpath = config["diffusion_model_weights"]
    model.load_state_dict(torch.load(model_weights_fpath))
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    model, optim, test_loader = accelerator.prepare(
        model, optim, test_loader
    )

    losses_to_write = []
    total_test_loss = 0
    with torch.no_grad():
        # Training loop.
        model.eval()
        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        for idx, (x, _) in test_pbar:
            loss = model(x)
            if idx == 3966:
                print(f"Loss for image 3966: {loss.item()}")
                import matplotlib.pyplot as plt
                plt.imshow(x.squeeze().detach().cpu().numpy(), cmap="gray")
                plt.show()
                
            test_pbar.set_description(f"loss: {loss.item():.3g}")
            total_test_loss += loss.item()
            losses_to_write.append([idx, loss.item()])
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"[INFO] Average test loss: {avg_test_loss:.3g}")
        # Write test losses to the csv file.
        with open(test_losses_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerows(losses_to_write)

    if config["calc_metrics"]:
        total_rmse = 0
        total_ssim = 0
        total_psnr = 0

        if not config["quick_test"] and config["metric_sample_size"]:
            # Use a subset of the test set for quick testing.
            # Not needed if quick test already set.
            test_set = Subset(test_set, np.arange(0, config["metric_sample_size"]))

        # Batch size fixed as too large a batch size can cause memory issues.
        test_loader = DataLoader(
            test_set, 512, shuffle=False
        )

        test_loader = accelerator.prepare(test_loader)
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for idx, (x, _) in pbar:
                x_hat = model.cond_sample(x, [], accelerator.device)
                # cond_sample returns x since its typically used to generate plots.
                # Want the last batch size number of elements along the first axis.
                # to get the reconstructed x_hat.
                x_hat = x_hat[-x.shape[0]:]
                
                # These are averages across the batch.
                rmse, ssim, psnr = calc_image_quality_metrics(x, x_hat)
                total_rmse += rmse
                total_ssim += ssim
                total_psnr += psnr

            avg_rmse = total_rmse / len(test_loader)
            avg_ssim = total_ssim / len(test_loader)
            avg_psnr = total_psnr / len(test_loader)

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

    main(config)
