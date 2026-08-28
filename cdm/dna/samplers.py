"""
Sampler for the DNA MDLM.

duo noise convention:
  noise(t) -> (dalpha_t, alpha_t) with alpha_t = 1 - (1-eps)*t, sigma = -log(alpha_t)
Transition q(x_s | x_t, x0): unmask with prob (alpha_s - alpha_t), stay masked with 1 - alpha_s.
"""

import torch


def _sample_categorical(categorical_probs):
    """Sample from categorical distribution using Gumbel-max trick."""
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)



class MDLMSampler:
    """Ancestral sampler for the DNA MDLM Diffusion model."""

    def __init__(self, diffusion_model, num_steps=128, eps=1e-5):
        self.model = diffusion_model
        self.num_steps = num_steps
        self.eps = eps
        self.device = next(diffusion_model.parameters()).device
        self.mask_index = diffusion_model.mask_index
        self.noise = diffusion_model.noise

    def _get_timesteps(self):
        timesteps = torch.linspace(1, self.eps, self.num_steps + 1, device=self.device)
        dt = (1 - self.eps) / self.num_steps
        return timesteps, dt

    def _get_alpha_sigma(self, t):
        """Compute alpha_t and sigma_t from timestep t.

        Args:
            t: [B] or [B, 1] timestep tensor
        Returns:
            alpha_t: [B] keep probability
            sigma_t: [B] noise level = -log(alpha_t)
        """
        if t.ndim > 1:
            t = t.squeeze(-1)
        _, alpha_t = self.noise(t)
        sigma_t = -torch.log(alpha_t)
        return alpha_t, sigma_t

    def _ddpm_cache_step(self, model, x, t, dt, p_x0_cache=None):
        """One DDPM reverse step using alpha_t parameterization.

        Transition: q(x_s | x_t, x0)
          - unmask with prob proportional to p_x0 * (alpha_s - alpha_t)
          - stay masked with prob 1 - alpha_s
        """
        if t.ndim > 1:
            t_1d = t.squeeze(-1)
        else:
            t_1d = t

        alpha_t, sigma_t = self._get_alpha_sigma(t_1d)
        alpha_s, _ = self._get_alpha_sigma(t_1d - dt)

        if p_x0_cache is not None:
            p_x0 = p_x0_cache
        else:
            # model.forward expects sigma as [B, 1] or [B]
            p_x0 = model.forward(x, sigma_t.unsqueeze(-1)).exp()

        q_xs = p_x0 * (alpha_s - alpha_t)[:, None, None]
        q_xs[:, :, self.mask_index] = (1 - alpha_s)[:, None]

        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        x_next = copy_flag * x + (1 - copy_flag) * _x

        new_cache = p_x0 if torch.allclose(x_next, x) else None
        return x_next, new_cache

    def _noise_removal(self, model, x):
        """Final denoising: argmax at t=eps."""
        t_final = self.eps * torch.ones(x.shape[0], device=self.device)
        _, sigma_final = self._get_alpha_sigma(t_final)
        return model.forward(x, sigma_final.unsqueeze(-1)).argmax(dim=-1)
