import torch.nn as nn
import torch
import numpy as np


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act(),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed


class DDPM(nn.Module):
    def __init__(
        self,
        decoder,
        beta_t,
        alpha_t,
        device,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()
        # beta_t has T + 1 elements, where T is the number of steps.
        # Allows for convenient indexing, e.g. beta_t[t] is the value of beta at time t.
        self.T = len(beta_t) - 1
        self.device = device
        self.decoder = decoder
        self.criterion = criterion
        """
        TODO: Define abstract class, standardise type of t for degradation
        and generation, currently t for degrade is a vector, t for generate
        is a scalar.
        """

        # Convert to tensors and register as buffers.
        beta_t = torch.tensor(beta_t, device=device)
        alpha_t = torch.tensor(alpha_t, device=device)

        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_t", alpha_t)

    def degrade(self, x, t):
        # Degradation step.
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting.
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        return z_t, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince"""
        # Draw t uniformly from 1 to T.
        t = torch.randint(1, self.T, (x.shape[0],), device=x.device)

        # Calculate the latent variable z_t from x and eps.
        z_t, eps = self.degrade(x, t)

        # Estimate the noise using the decoder.
        eps_hat = self.decoder(z_t, t / self.T)

        # Return the loss.
        return self.criterion(eps, eps_hat)

    def generate(self, z_t, end_t, start_t=None):
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
        """Algorithm 18.2 in Prince"""
        z_t = torch.randn(n_sample, *size, device=device)
        x_t = self.generate(z_t, 0)
        return x_t

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

        # Generate iteratively and store the samples at the relevant indices.
        for idx, t in enumerate(visualise_ts, start=len(visualise_ts) + 1):
            t_scalar = t[0]
            z_t = self.generate(z_t, t_scalar, start_t=prev_t)
            samples[(idx) * x.shape[0] : (idx + 1) * x.shape[0]] = z_t
            prev_t = t_scalar

        # Last sample is the fully reconstructed image.
        z_t = self.generate(z_t, 0, start_t=prev_t)
        samples[-x.shape[0] :] = z_t

        return samples
