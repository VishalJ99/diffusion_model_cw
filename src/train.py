import argparse
import os
import sys
from utils import (
    seed_everything,
    fetch_model,
    fetch_beta_schedule,
)
import wandb
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from accelerate import Accelerator
import tqdm
import numpy as np
from torchvision.utils import save_image, make_grid
import json
import yaml

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
            f"Directory already exists in {config['output_dir']} already exists.\
              Exiting."
        )
        sys.exit(1)

    if os.path.exists(".git"):
        # Add the git hash to the config if the .git file exists.
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    # Save the config to the output directory for reproducibility.
    config_file = os.path.join(output_dir, "config.yaml")
    with open(config_file, "w") as f:
        json.dump(config, f)

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

    # Set the device to use for training. GPU -> MPS -> CPU.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

    # Define the pre transforms.
    # TODO: Make specifiable from config. Seperate into train and val if random
    # transformations are used. May require defining a custom dataset class.
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

    # Initialise the dataloaders.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config["train_batch_size"], shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config["val_batch_size"],
        shuffle=False,
    )

    # Fetch the decoder model. TODO: avoid hard coding the activation in
    # the decoder model class def. Make loadable from config.
    decoder_model_class = fetch_model(config["decoder_model"])
    diffusion_model_class = fetch_model(config["diffusion_model"])

    # Load the diffusion model hyper parameters.
    beta_schedule = fetch_beta_schedule(config["beta_schedule"])
    T = config["T"]
    beta_t = beta_schedule(T, **config["custom_beta_schedule_params"])
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    assert len(beta_t) == T, "Beta schedule must have T elements."
    assert len(alpha_t) == T, "Alpha schedule must have T elements."

    # Initialise the models.
    decoder = decoder_model_class(**config["decoder_model_params"])
    model = diffusion_model_class(decoder, beta_t, alpha_t, T)

    # Initialise optimiser.
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    ddpm, optim, dataloader = accelerator.prepare(model, optim, train_loader)
    
    n_epoch = config["n_epoch"]

    # Visualise samples at these epochs.
    # visualise_samples = [i in range(n_epoch)] if config["visualise_samples"] == 'all' else config["visualise_samples"]

    all_train_losses = []
    lowest_val_loss = np.inf
    for i in range(n_epoch):
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
        print(f"Epoch {i} - average loss: {avg_train_loss}")

        model.eval()
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

        if average_val_loss < lowest_val_loss:
            # Save model weights.
            model_weights_fname = "best_model.pth"
            model_weights_fpath = os.path.join(
                output_dir, "model_weights", model_weights_fname
            )
            print(
                f"[INFO] Validation loss improved from {lowest_val_loss:.4f} to {average_val_loss:.4f}"
            )
            print("[INFO] Saving model weights")
            lowest_val_loss = average_val_loss

        # Save train and validation loss to output_dir
    
        # TODO: Figure out sampling and metrics.
        # Unconditional generation of samples.
        # if i in visualise_samples:
            # x_uncond = model.unconditional_sample(16, (1, 28, 28), device)
            # x_cond = model.conditional_sample(4, (1, 28, 28), device)

            # grid = make_grid(x_uncond, nrow=4)
            # sample_images_fname = f"epoch_{i}.png"
            # unconditional_sample_images_fpath = os.path.join(
                # output_dir, "samples", "unconditional", unconditional_sample_images_fname
            # )
            # unconditional_sample_images_fpath = os.path.join(
            #     output_dir, "samples", "unconditional", unconditional_sample_images_fname
            # )
            # save_image(grid, f"./contents/ddpm_sample_{i:04d}.png")

        # save model.
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