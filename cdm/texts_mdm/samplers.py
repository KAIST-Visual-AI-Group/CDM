"""
MDLM sampler: continuous-time masked diffusion with a categorical transition kernel.

  - sigma_t = noise.total_noise(t), move_chance = 1 - exp(-sigma_t)
  - model.forward(x, sigma) returns log p(x0 | x_t, sigma), not raw logits
  - time runs from t=1 (fully masked) to t=eps (clean)
"""

import torch


def _sample_categorical(categorical_probs):
    """Sample from categorical distribution using Gumbel-max trick."""
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class MDLMSampler:
    """Ancestral sampler for MDLM Diffusion models."""

    def __init__(self, diffusion_model, num_steps=128, eps=1e-5):
        self.model = diffusion_model
        self.num_steps = num_steps
        self.eps = eps
        self.device = next(diffusion_model.parameters()).device
        self.mask_index = diffusion_model.mask_index
        self.noise = diffusion_model.noise

    def _get_timesteps(self):
        """Returns (timesteps, dt) for the reverse process: t=1 -> t=eps."""
        timesteps = torch.linspace(1, self.eps, self.num_steps + 1, device=self.device)
        dt = (1 - self.eps) / self.num_steps
        return timesteps, dt

    def _ddpm_cache_step(self, model, x, t, dt, p_x0_cache=None, gumbel_chunk_size=None):
        """
        DDPM reverse step x_t -> x_s, s = t - dt.

        Uses move_chance_t = t directly (loglinear shortcut, valid because
        for loglinear noise 1-exp(-sigma(t)) = (1-eps)*t ~ t), and reuses p_x0
        across steps in which nothing was unmasked.

        Returns: (x_next, new_p_x0_cache)
        """
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        if p_x0_cache is not None:
            p_x0 = p_x0_cache
        else:
            p_x0 = model.forward(x, sigma_t).exp()

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

        if gumbel_chunk_size is not None:
            B = q_xs.shape[0]
            _x_list = list()
            for chunk_idx in range(0, B, gumbel_chunk_size):
                chunk_size = min(gumbel_chunk_size, B - chunk_idx)
                _x_list.append(_sample_categorical(q_xs[chunk_idx:chunk_idx + chunk_size]))
            _x = torch.cat(_x_list, dim=0)
        else:
            _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        x_next = copy_flag * x + (1 - copy_flag) * _x

        new_cache = p_x0 if torch.allclose(x_next, x) else None
        return x_next, new_cache

    def _noise_removal(self, model, x):
        """Final denoising: argmax of model prediction at t=eps."""
        t_final = self.eps * torch.ones(x.shape[0], 1, device=self.device)
        sigma_final = self.noise(t_final)[0]
        if sigma_final.ndim > 1:
            sigma_final = sigma_final.squeeze(-1)
        return model.forward(x, sigma_final).argmax(dim=-1)

    @torch.no_grad()
    def sample(self, model, batch_size, seq_len, return_traj=False, stop_t=None):
        """
        Unguided ancestral sampling.

        When return_traj=True, returns (x0, xs, sigmas):
            x0:     [B, L] final clean samples after noise removal
            xs:     list of [B, L] int64 tensors, state at each reverse step
            sigmas: list of [B] float tensors, noise level at each step
        """
        x = self.mask_index * torch.ones(
            batch_size, seq_len, dtype=torch.int64, device=self.device)
        timesteps, dt = self._get_timesteps()

        if return_traj:
            xs, sigmas = list(), list()

        p_x0_cache = None
        for i in range(self.num_steps + 1):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

            if return_traj:
                sigma_t, _ = self.noise(t)
                if sigma_t.ndim > 1:
                    sigma_t = sigma_t.squeeze(-1)
                xs.append(x.clone())
                sigmas.append(sigma_t)

            if stop_t == i:
                if return_traj:
                    return None, xs, sigmas
                sigma_t, _ = self.noise(t)
                if sigma_t.ndim > 1:
                    sigma_t = sigma_t.squeeze(-1)
                return None, x.clone(), sigma_t
            if i == self.num_steps:
                break

            x, p_x0_cache = self._ddpm_cache_step(model, x, t, dt, p_x0_cache)

        x = self._noise_removal(model, x)
        if return_traj:
            return x, xs, sigmas
        return x
