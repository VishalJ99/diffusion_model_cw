import torch
import torch.nn as nn


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
        and generation, currently t for degrade is a vector, t for restore
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

    def restore(self, z_t, end_t, start_t=None):
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
        x_t = self.restore(z_t, 0)
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

        # restore iteratively and store the samples at the relevant indices.
        for idx, t in enumerate(visualise_ts, start=len(visualise_ts) + 1):
            t_scalar = t[0]
            z_t = self.restore(z_t, t_scalar, start_t=prev_t)
            samples[(idx) * x.shape[0] : (idx + 1) * x.shape[0]] = z_t
            prev_t = t_scalar

        # Last sample is the fully reconstructed image.
        z_t = self.restore(z_t, 0, start_t=prev_t)
        samples[-x.shape[0] :] = z_t

        return samples
