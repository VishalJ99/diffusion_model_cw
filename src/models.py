import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Callable

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
        # Padding ensures that the spatial dimensions of the input
        # are preserved after the convolution.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.LayerNorm(expected_shape),
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
        # correct scale and shape (and avoid applying the activation).
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        # This part is literally just to put the single scalar "t" into the CNN
        # in a nice, high-dimensional way.
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
        T,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()
        # if beta_t, alpha_t and T are not tensors, convert them.
        if not torch.is_tensor(beta_t):
            beta_t = torch.tensor(beta_t)
        if not torch.is_tensor(alpha_t):
            alpha_t = torch.tensor(alpha_t)
        if not torch.is_tensor(T):
            T = torch.tensor(T)

        self.decoder = decoder
        self.register_buffer("beta_t", beta_t)
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("T", torch.tensor(T))
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince"""
        # Uniformly draw batch size number of t's from 1 to n_T.
        t = torch.randint(1, self.T, (x.shape[0],), device=x.device)
        
        # Draw noise from standard normal distribution.
        eps = torch.randn_like(x)  
        
        # Calculate z_t from x and eps.
        alpha_t = self.alpha_t[t, None, None, None]  # for broadcasting reasons
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        
        # Predict the noise term from this z_t.
        eps_hat = self.decoder(z_t, t / self.T)
        return self.criterion(eps, eps_hat)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)
        for i in range(self.T-1, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.decoder(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            # Add noise for all steps except the final step.
            if i > 1:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)

        return z_t
