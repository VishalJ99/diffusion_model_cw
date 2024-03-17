import argparse
import os
from utils import seed_everything, fetch_model, fetch_beta_schedule
import torch
import sys
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from models import CNN, DDPM
from accelerate import Accelerator
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import csv
import yaml
import torch.nn as nn
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
        print(
            f"Directory already exists in {config['output_dir']}..."
            " Exiting."
        )
        sys.exit(1)

    if os.path.exists(".git"):
        # Add the git hash to the config if the .git file exists.
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    # Save the config to the output directory for reproducibility.
    config_file = os.path.join(output_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    if config["wandb"]:
        # Use wandb to log the run.
        wandb.init(project=config["m2_cw"], config=config)

    print("-" * 50)
    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Set the random seed for reproducibility.
    seed_everything(config["seed"])

    pre_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    # Load the dataset.
    dataset = MNIST("./data", train=True, download=True,
                    transform=pre_transforms)

    # Split the dataset into train and validation sets.
    val_fraction = config["val_fraction"]
    train_set, val_set = torch.utils.data.random_split(
        dataset, [1 - val_fraction, val_fraction]
    )

    if config["quick_test"]:
        # Create smaller subsets of the datasets for quick testing.
        train_indices = list(range(config["train_batch_size"]))
        val_indices = list(range(config["val_batch_size"]))

        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = torch.utils.data.Subset(val_set, val_indices)

    # Initialise the dataloaders.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config["train_batch_size"], shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config["val_batch_size"],
        shuffle=False,
    )

    decoder_model_class = fetch_model(config["decoder_model"])
    diffusion_model_class = fetch_model(config["diffusion_model"])

    # Load the diffusion model hyper parameters.
    beta_schedule = fetch_beta_schedule(config["beta_schedule"])
    T = config["T"]
    beta_t = beta_schedule(T, **config["custom_beta_schedule_params"])

    assert len(beta_t) == T, "Beta schedule must have T elements."

    # Initialise the models.
    decoder = decoder_model_class(**config["decoder_model_params"])
    model = diffusion_model_class(decoder, beta_t)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Set the device to use for training. GPU -> MPS -> CPU.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

   # Lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    model, optim, train_loader, val_loader = accelerator.prepare(model,
                                                                optim,
                                                                train_loader,
                                                                val_loader)
    n_epoch = config["n_epoch"]

    # Visualise samples at these epochs.
    # visualise_samples = [i in range(n_epoch)] if config["visualise_samples"] == 'all' else config["visualise_samples"]
    best_model_weights_fname = "best_model.pth"
    best_model_weights_fpath = os.path.join(
        output_dir, "model_weights", best_model_weights_fname
    )
    all_train_losses = []
    lowest_val_loss = np.inf
    # Open the CSV file and prepare to write losses.
    loss_csv_path = os.path.join(output_dir, 'losses.csv')
    with open(loss_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['epoch', 'avg_training_loss', 'avg_validation_loss'])

    for i in range(1, n_epoch+1):
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
            for x, _ in val_loader:
                loss = model(x)
                wandb.log({"val_loss": loss.item()})
                total_val_loss += loss.item()

            average_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {i} - average val loss: {average_val_loss}")

            # Generate and save samples unconditionally.
            xh = model.sample(16, (1, 28, 28), accelerator.device) 
            grid = make_grid(xh, nrow=4)
            image_path = os.path.join(uncond_samples_dir_path, f"ddpm_sample_{i:04d}.png")
            save_image(grid, image_path)

        if average_val_loss < lowest_val_loss:
            # Save model weights.
            print(
                f"[INFO] Validation loss improved from {lowest_val_loss:.4f} to {average_val_loss:.4f}"
            )
            print("[INFO] Saving model weights")
            torch.save(
                model.state_dict(), best_model_weights_fpath
            )
            lowest_val_loss = average_val_loss

        # Save train and validation loss to output_dir
        with open(loss_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, avg_train_loss, average_val_loss])

        # Save model weights.
        model_weights_fname = f"epoch_{i}_model.pth"
        model_weights_fpath = os.path.join(
            output_dir, "model_weights", model_weights_fname
        )
        torch.save(
            model.state_dict(), model_weights_fpath
        )


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