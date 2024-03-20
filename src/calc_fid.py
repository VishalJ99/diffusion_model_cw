import argparse
import yaml
from utils import fetch_model, fetch_noise_schedule, seed_everything
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


def main(output_file, weights_file, config, num_samples, bs):
    # Set seed.
    seed_everything(config["seed"])
    print(f"[INFO] Seed set to: {config['seed']}")

    # Cant use accelerator here as mps is not supported by fid class...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device set to: {device}")

    # Load dataset.
    pre_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset = MNIST("./data", train=False, download=True, transform=pre_transforms)
    assert num_samples <= len(
        dataset
    ), "Number of samples must be less than or equal to the dataset size"
    dataset = Subset(dataset, list(range(num_samples)))

    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    # Load model.
    decoder_model_class = fetch_model(config["decoder_model"])
    diffusion_model_class = fetch_model(config["diffusion_model"])
    noise_schedule = fetch_noise_schedule(config["noise_schedule"])

    # Create the noise schedule.
    T = config["T"]
    beta_t, alpha_t = noise_schedule(T, **config["custom_noise_schedule_params"])

    # Initialise the models.
    decoder = decoder_model_class(**config["decoder_model_params"])
    model = diffusion_model_class(decoder, beta_t, alpha_t, device).to(device)

    # Load the weights.
    model.load_state_dict(torch.load(weights_file, map_location=device))

    with torch.no_grad():
        fid = FrechetInceptionDistance(normalize=True)
        # Need to pass data in batches since data is upsampled to 299x299,
        # which can cause memory issues if all data is passed at once.
        pbar = tqdm(dataloader, desc="Calculating FID score")
        for x, _ in pbar:
            x = x.to(device)

            # Generate fake samples.
            fake_samples = model.uncond_sample(bs, (1, 28, 28), device)

            # TODO: vectorise this / make it more efficient.
            for image in fake_samples:
                # normalise image. Cant use min and max over the entire set of images
                # min and max for the noisier latents will skew the entire set.
                image = (image - image.min()) / (image.max() - image.min())

            # Convert input to RGB for inception model.
            fake_samples_rgb = torch.repeat_interleave(fake_samples, 3, 1)
            x_rgb = torch.repeat_interleave(x, 3, 1)

            # Update fid statistics.
            fid.update(fake_samples_rgb, real=False)
            fid.update(x_rgb, real=True)

        fid_score = fid.compute()
        print(f"FID score: {fid_score}")

        # Save the score to a file.
        with open(output_file, "w") as file:
            file.write(f"FID score: {fid_score}")

        return fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID score")
    parser.add_argument(
        "output_file", type=str, help="Path to output file where score is saved."
    )
    parser.add_argument(
        "weights_file",
        type=str,
        help="Path to diffusion model\
                        weight .pt file",
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to train or test config file,\
                        used to fetch the diffusion model, initialise decoder model\
                        and noise schedule",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use to calculate fid score,\
                            default = 1000",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=100,
        help="Batch size used to define\
        batch size of inputs to the inception model, and the number of samples\
        generated from diffusion model. WARNING, too large a batch size can crash the\
        kernel. Default = 100",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    main(args.output_file, args.weights_file, config, args.num_samples, args.bs)
