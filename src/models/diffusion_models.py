import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from abc import ABC, abstractmethod
from typing import Optional
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# TODO: Consider making z_t a parameter of uncond_sample since for cold diffusion models
# Need to play around with z_t to find a good representation for the true latent space.


class DiffusionModel(nn.Module, ABC):
    def __init__(
        self,
        decoder: nn.Module,
        beta_t: list[float],
        alpha_t: list[float],
        device: torch.device,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()
        self.T: int = len(beta_t) - 1
        self.device: torch.device = device
        self.decoder: nn.Module = decoder
        self.criterion: nn.Module = criterion

        # Convert to tensors and register as buffers.
        beta_t_tensor: torch.Tensor = torch.tensor(beta_t, device=device)
        alpha_t_tensor: torch.Tensor = torch.tensor(alpha_t, device=device)
        self.register_buffer("beta_t", beta_t_tensor)
        self.register_buffer("alpha_t", alpha_t_tensor)

    @abstractmethod
    def degrade(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Method to be implemented by subclass: degrades the input x at time t."""
        pass

    @abstractmethod
    def restore(
        self, z_t: torch.Tensor, end_t: int, start_t: Optional[int] = None
    ) -> torch.Tensor:
        """Method to be implemented by subclass: restores z_t from end_t to start_t."""
        pass

    @abstractmethod
    def uncond_sample(self, n_sample: int, size, device) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince"""
        # Draw t uniformly from 1 to T.
        t = torch.randint(1, self.T + 1, (x.shape[0],), device=x.device)

        # Calculate the latent variable z_t from x and eps.
        z_t, eps = self.degrade(x, t)

        # Estimate the noise using the decoder.
        eps_hat = self.decoder(z_t, t / self.T)

        # Return the loss.
        return self.criterion(eps, eps_hat)

    def cond_sample(self, x, visualise_ts, device):
        # Check visualise t is in ascending order.
        assert all(
            visualise_ts[i] < visualise_ts[i + 1] for i in range(len(visualise_ts) - 1)
        )
        visualise_ts = torch.tensor(visualise_ts, device=device)

        # Reshape for broadcasting purposes.
        visualise_ts = visualise_ts.repeat(x.shape[0], 1).mT

        # Create a tensor to store the samples.
        samples = torch.zeros(
            2 * (len(visualise_ts) + 1) * x.shape[0], *x.shape[1:], device=x.device
        )

        # Add the original image.
        samples[: x.shape[0]] = x

        # Forward degradation.
        for idx, t in enumerate(visualise_ts, start=1):
            z_t, _ = self.degrade(x, t)
            samples[idx * x.shape[0] : (idx + 1) * x.shape[0]] = z_t

        # Backward reconstruction.
        visualise_ts = torch.flip(
            visualise_ts,
            [
                0,
            ],
        )

        # Start with fully degraded image.
        z_t, _ = self.degrade(x, self.T)
        prev_t = self.T

        # Restore iteratively and store the samples at the relevant indices.
        for idx, t in enumerate(visualise_ts, start=len(visualise_ts) + 1):
            t_scalar = t[0]
            z_t = self.restore(z_t, t_scalar, start_t=prev_t)
            samples[(idx) * x.shape[0] : (idx + 1) * x.shape[0]] = z_t
            prev_t = t_scalar

        # Last sample is the fully reconstructed image.
        z_t = self.restore(z_t, 0, start_t=prev_t)
        samples[-x.shape[0] :] = z_t

        return samples


class ColdDiffusionModel(DiffusionModel):
    """
    Diffusion model that predicts the image instead of the noise
    Also use algorithm 2 from the cold diffusion paper for sampling.
    https://arxiv.org/abs/2208.09392.
    """

    def __init__(
        self,
        decoder,
        beta_t,
        alpha_t,
        device,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__(decoder, beta_t, alpha_t, device, criterion)

    def restore(self, z_t, end_t, start_t=None):
        # Algo 2 in cold diffusion paper for reconstruction.
        _one = torch.ones(z_t.shape[0], device=self.device)
        if start_t:
            assert start_t <= self.T
        else:
            start_t = self.T

        for t in range(start_t, end_t, -1):
            x_hat_0 = self.decoder(z_t, (t / self.T) * _one)
            # TODO: Using -= breaks things, why?
            z_t = z_t - self.degrade(x_hat_0, t)[0] + self.degrade(x_hat_0, t - 1)[0]

        return z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Draw t uniformly from 1 to T.
        _one = torch.ones(x.shape[0], device=self.device)
        t = torch.randint(1, self.T + 1, (), device=x.device)

        # Degradation step.
        z_t, eps = self.degrade(x, t)

        # Reconstruction step.
        x_hat = self.decoder(z_t, (t / self.T) * _one)

        # Return loss between image and reconstructed image.
        return self.criterion(x, x_hat)


class DDPM(DiffusionModel):
    def __init__(
        self,
        decoder,
        beta_t,
        alpha_t,
        device,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__(decoder, beta_t, alpha_t, device, criterion)

    def degrade(self, x, t):
        # Degradation step.
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting.
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        return z_t, eps

    def restore(self, z_t, end_t, start_t=None):
        """Algorithm 18.2 in Prince"""
        # For broadcasting of scalar time. TODO: either make t a vector or
        # add same broadcasting for degrade.
        _one = torch.ones(z_t.shape[0], device=self.device)
        if start_t:
            assert start_t <= self.T
        else:
            start_t = self.T

        for t in range(start_t, end_t, -1):
            alpha_t = self.alpha_t[t]
            beta_t = self.beta_t[t]

            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.decoder(
                z_t, (t / self.T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            if t > 1:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)

        return z_t

    def uncond_sample(self, n_sample: int, size, device) -> torch.Tensor:
        z_t = torch.randn(n_sample, *size, device=device)
        samples = self.restore(z_t, 0)
        return samples


class GaussianBlurDM(ColdDiffusionModel):
    def __init__(
        self,
        decoder,
        beta_t,
        alpha_t,
        device,
        kernel_size=11,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__(decoder, beta_t, alpha_t, device, criterion)
        self.kernel_size = kernel_size

    def degrade(self, x, t):
        # Degradation step.
        # TODO: Convolution is a linear operator so find the
        # transform equivalent to applying the t gaussian blurs.
        # https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication

        # Forward method passes a vector of ts, currently only support one t.
        if not isinstance(t, int):
            t = int(t[0]) if t.numel() > 1 else int(t)

        z_t = x.clone()
        for i in range(1, t + 1):
            blur = GaussianBlur(self.kernel_size, float(self.beta_t[i]))
            z_t = blur(z_t)

        eps = z_t - x
        return z_t, eps

    def uncond_sample(self, n_sample: int, size, device) -> torch.Tensor:
        # Generate n_samples random floats between -0.4 and -0.3 (typical vals
        # of a heavily blurred mnist sample)
        c = 0.1 * torch.rand(n_sample, device=device) - 0.4

        # Create uniform tensors from each float and stack them along the batch axis.
        uniform_ts = [torch.full(size, value.item(), device=device) for value in c]
        z_t = torch.stack(uniform_ts, dim=0)

        samples = self.restore(z_t, 0)

        return samples


class FashionMNISTDM(ColdDiffusionModel):
    def __init__(
        self,
        decoder,
        beta_t,
        alpha_t,
        device,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__(decoder, beta_t, alpha_t, device, criterion)
        # Load FashionMNIST dataset.
        # (Refactor later to avoid loading dataset in model...)
        pre_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
        )
        self.train_dataset = FashionMNIST(
            root="data", train=True, download=True, transform=pre_transforms
        )
        self.test_dataset = FashionMNIST(
            root="data", train=False, download=True, transform=pre_transforms
        )

    def degrade(self, x, t, eps_original=None, test=False):
        if eps_original is None:
            # Fetch random mnist data.
            indices = torch.randint(0, len(self.test_dataset), (x.shape[0],))
            if test:
                eps = torch.stack([self.test_dataset[i][0] for i in indices]).to(
                    x.device
                )
            else:
                eps = torch.stack([self.train_dataset[i][0] for i in indices]).to(
                    x.device
                )
        else:
            # see if this helps restoration.
            eps = eps_original

        # Mix with input images according to noise schedule.
        alpha_t = self.alpha_t[t, None, None, None]

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps

        return z_t, eps

    def uncond_sample(self, n_sample: int, size, device):
        # Randonly sample fashion mnist test set to generate samples.
        indices = torch.randint(0, len(self.test_dataset), (n_sample,))
        z_t = torch.stack([self.test_dataset[i][0] for i in indices]).to(device)
        eps_original = z_t.clone()
        samples = self.restore(z_t, 0, eps_original=eps_original)
        return samples

    def restore(self, z_t, end_t, start_t=None, eps_original=None):
        # Algo 2 in cold diffusion paper for reconstruction.
        # Fixing eps for the restoration.
        if start_t:
            assert start_t <= self.T
        else:
            start_t = self.T
        _one = torch.ones(z_t.shape[0], device=self.device)
        for t in range(start_t, end_t, -1):
            x_hat_0 = self.decoder(z_t, (t / self.T) * _one)
            # TODO: Using -= breaks things, why?
            z_t = (
                z_t
                - self.degrade(x_hat_0, t, eps_original)[0]
                + self.degrade(x_hat_0, t - 1, eps_original)[0]
            )
        return z_t

    def cond_sample(self, x, visualise_ts, device):
        """Loop modified to pass eps_original to the restore method."""
        # Check visualise t is in ascending order.
        assert all(
            visualise_ts[i] < visualise_ts[i + 1] for i in range(len(visualise_ts) - 1)
        )
        visualise_ts = torch.tensor(visualise_ts, device=device)

        # Reshape for broadcasting purposes.
        visualise_ts = visualise_ts.repeat(x.shape[0], 1).mT

        # Create a tensor to store the samples.
        samples = torch.zeros(
            2 * (len(visualise_ts) + 1) * x.shape[0], *x.shape[1:], device=x.device
        )

        # Add the original image.
        samples[: x.shape[0]] = x

        # Forward degradation.
        for idx, t in enumerate(visualise_ts, start=1):
            z_t, _ = self.degrade(x, t)
            samples[idx * x.shape[0] : (idx + 1) * x.shape[0]] = z_t

        # Backward reconstruction.
        visualise_ts = torch.flip(
            visualise_ts,
            [
                0,
            ],
        )

        # Start with fully degraded image using a random image from the test set.
        z_t, _ = self.degrade(x, self.T, test=True)
        prev_t = self.T

        # Need to restore from the same fashion mnist case at each time step.
        eps_original = z_t.clone()

        # Restore iteratively and store the samples at the relevant indices.
        for idx, t in enumerate(visualise_ts, start=len(visualise_ts) + 1):
            t_scalar = t[0]
            z_t = self.restore(z_t, t_scalar, start_t=prev_t, eps_original=eps_original)
            samples[(idx) * x.shape[0] : (idx + 1) * x.shape[0]] = z_t
            prev_t = t_scalar

        # Last sample is the fully reconstructed image.
        z_t = self.restore(z_t, 0, start_t=prev_t, eps_original=eps_original)
        samples[-x.shape[0] :] = z_t

        return samples
