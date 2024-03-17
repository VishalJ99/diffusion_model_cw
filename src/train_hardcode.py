import argparse
import os
from utils import seed_everything
import torch
import sys
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from models_hardcode import CNN, DDPM
from accelerate import Accelerator
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import csv
import yaml
import torch.nn as nn
import numpy as np


def main():
    # Create the output directory for the run if it does not exist.
    output_dir = 'runs/test_2'
    dir_ = output_dir

    try:
        weights_dir_path = os.path.join(output_dir, "models")
        contents_dir_path = os.path.join(output_dir, "contents")

        os.makedirs(weights_dir_path)
        os.makedirs(contents_dir_path)

    except OSError:
        print(
            f"Directory already exists..."
            " Exiting."
        )
        sys.exit(1)

    # Set the random seed for reproducibility.
    seed = 42
    seed_everything(seed)
    print(f"[INFO] Random seed set to {seed}")

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
    val_fraction = 0.2
    train_set, val_set = torch.utils.data.random_split(
        dataset, [1 - val_fraction, val_fraction]
    )

    # Initialise the dataloaders.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=512,
        shuffle=False,
    )

    gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
    ddpm = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    # Set the device to use for training. GPU -> MPS -> CPU.
    accelerator = Accelerator()
    device = accelerator.device
    print(f"[INFO] Device set to: {device}")

    # Lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
    ddpm, optim, train_loader, val_loader = accelerator.prepare(ddpm,
                                                                optim,
                                                                train_loader,
                                                                val_loader)
    n_epoch = 100
    loss_csv_path = f"{dir_}/loss.csv"
    with open(loss_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['epoch', 'avg_training_loss', 'avg_validation_loss'])

    for i in range(1, n_epoch+1):
        ddpm.train()
        train_losses = []
        val_losses = []
        pbar = tqdm(train_loader)  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()
            loss = ddpm(x)
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.3g}")  # Show running average of loss in progress bar

        ddpm.eval()
        with torch.no_grad():
            pbar = tqdm(val_loader)
            for x, _ in pbar:
                loss = ddpm(x)
                val_losses.append(loss.item())
                pbar.set_description(f"val_loss: {loss.item():.3g}")

            avg_train_loss = sum(train_losses) / len(train_losses)
            average_val_loss = sum(val_losses) / len(val_losses)
            
            xh = ddpm.sample(16, (1, 28, 28), accelerator.device)  # Can get device explicitly with `accelerator.device`
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory
            save_image(grid, f"{dir_}/contents/ddpm_sample_{i:04d}.png")

            # save model
            torch.save(ddpm.state_dict(), f"{dir_}/models/{i}_ddpm_mnist.pth")

            # Save train and validation loss to output_dir
            with open(loss_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, avg_train_loss, average_val_loss])


if __name__ == "__main__":
    main()

