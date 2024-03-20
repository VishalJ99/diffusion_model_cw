import argparse
import os
from utils import (
    seed_everything,
    fetch_model,
    fetch_noise_schedule,
    make_cond_samples_plot,
    make_uncond_samples_plot,
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
import wandb


def main(config):
    # Create the output directory for the run if it does not exist.
    output_dir = config["output_dir"]

    try:
        weights_dir_path = os.path.join(output_dir, "model_weights")
        uncond_samples_dir_path = os.path.join(output_dir, "samples", "unconditional")
        cond_samples_dir_path = os.path.join(output_dir, "samples", "conditional")

        os.makedirs(weights_dir_path)
        os.makedirs(uncond_samples_dir_path)
        os.makedirs(cond_samples_dir_path)

    except OSError:
        print(f"Directory already exists in {config['output_dir']}..." " Exiting.")
        sys.exit(1)

    if os.path.exists(".git"):
        # Add the git hash to the config if the .git file exists.
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    if config["wandb"]:
        # Use wandb to log the run.
        run = wandb.init(project=config["wandb_project"], config=config)
        run_url = run.get_url()
        config["run_url"] = run_url

    # Save the config to the output directory for reproducibility.
    config_file = os.path.join(output_dir, "train_config.yaml")
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
    dataset = MNIST("./data", train=True, download=True, transform=pre_transforms)

    # Split the dataset into train and validation sets.
    val_fraction = config["val_fraction"]
    train_set, val_set = torch.utils.data.random_split(
        dataset, [1 - val_fraction, val_fraction]
    )

    if config["quick_test"]:
        # Create smaller subsets of the datasets for quick testing.
        train_indices = list(range(config["train_batch_size"]))
        val_indices = list(range(config["val_batch_size"]))

        train_set = Subset(train_set, train_indices)
        val_set = Subset(val_set, val_indices)

    # Initialise the dataloaders.
    train_loader = DataLoader(
        train_set, batch_size=config["train_batch_size"], shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["val_batch_size"],
        shuffle=False,
    )

    # Fetch the models and noise schedule.
    decoder_model_class = fetch_model(config["decoder_model"])
    diffusion_model_class = fetch_model(config["diffusion_model"])
    noise_schedule = fetch_noise_schedule(config["noise_schedule"])

    # Create the noise schedule.
    T = config["T"]
    beta_t, alpha_t = noise_schedule(T, **config["custom_noise_schedule_params"])

    assert len(beta_t) - 1 == T, f"Beta schedule must have {T+1} elements."

    # Initialise the models.
    decoder = decoder_model_class(**config["decoder_model_params"])
    model = diffusion_model_class(decoder, beta_t, alpha_t, device)

    # Load weights if specified.
    if config["diffusion_model_weights"]:
        model.load_state_dict(
            torch.load(config["diffusion_model_weights"], map_location=device)
        )
        print("[INFO] Loaded model weights from:", config["diffusion_model_weights"])

    # Define optimiser.
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    n_epoch = config["n_epoch"]

    # Load visualisation params.
    visualise_ts = config["visualise_ts"]
    num_cond_samples = config["num_cond_samples"]
    assert (
        num_cond_samples <= config["val_batch_size"]
    ), "num_cond_samples must be less than or equal to val_batch_size"

    best_model_weights_fname = "best_model.pth"
    best_model_weights_fpath = os.path.join(
        output_dir, "model_weights", best_model_weights_fname
    )
    all_train_losses = []
    lowest_val_loss = np.inf

    # Open the CSV file and prepare to write losses.
    loss_csv_path = os.path.join(output_dir, "losses.csv")
    with open(loss_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["epoch", "avg_training_loss", "avg_validation_loss"])

    for i in range(1, n_epoch + 1):
        # Training loop.
        model.train()
        total_train_loss = 0

        train_pbar = tqdm(train_loader)
        for x, _ in train_pbar:
            optim.zero_grad()
            loss = model(x)
            accelerator.backward(loss)
            optim.step()
            wandb.log({"train_loss": loss.item()})
            total_train_loss += loss.item()

            # Show running average of last 100 loss values in progress bar.
            all_train_losses.append(loss.item())
            running_avg_loss = np.average(
                all_train_losses[min(len(all_train_losses) - 100, 0) :]
            )
            train_pbar.set_description(f"loss: {running_avg_loss:.3g}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {i} - average loss: {avg_train_loss:.5g}")

        with torch.no_grad():
            # Validation loop.
            model.eval()
            total_val_loss = 0
            val_pbar = tqdm(val_loader)
            for x, _ in val_pbar:
                loss = model(x)
                wandb.log({"val_loss": loss.item()})
                total_val_loss += loss.item()

            average_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {i} - average val loss: {average_val_loss}")

            # Generate conditional samples.
            z_t = model.cond_sample(
                x[:num_cond_samples], visualise_ts, device=accelerator.device
            )

            plt = make_cond_samples_plot(z_t, visualise_ts, num_cond_samples)
            cond_image_path = os.path.join(
                cond_samples_dir_path, f"cond_sample_{i:04d}.png"
            )
            plt.savefig(cond_image_path, dpi=300)
            plt.close()

            # Generate and save samples unconditionally.
            xh = model.uncond_sample(16, (1, 28, 28), accelerator.device)
            plt = make_uncond_samples_plot(xh, 4)
            uncond_image_path = os.path.join(
                uncond_samples_dir_path, f"cond_sample_{i:04d}.png"
            )
            plt.savefig(uncond_image_path, dpi=300)
            plt.close()

        if average_val_loss < lowest_val_loss:
            # Save model weights.
            print(
                f"[INFO] Validation loss improved from {lowest_val_loss:.4f}"
                f" to {average_val_loss:.4f}"
            )
            print("[INFO] Saving model weights")
            torch.save(model.state_dict(), best_model_weights_fpath)
            lowest_val_loss = average_val_loss

        # Save train and validation loss to output_dir
        with open(loss_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i, avg_train_loss, average_val_loss])

        # Save model weights.
        model_weights_fname = f"epoch_{i}_model.pth"
        model_weights_fpath = os.path.join(
            output_dir, "model_weights", model_weights_fname
        )
        torch.save(model.state_dict(), model_weights_fpath)


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
