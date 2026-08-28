import torch

from byprot.models.utils import (
    top_k_top_p_filtering,
    topk_masking,
    sample_from_categorical,
)


def add_gumbel_noise(logits, temperature=1.0):
    """Add Gumbel noise to logits for stochastic sampling."""
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    return logits + temperature * noise


class DPLMSampler():

    def __init__(self, denoiser, steps=10, temperature=1.0):
        self.denoiser = denoiser
        self.device = denoiser.device
        self.steps = steps
        self.temperature = temperature

        self.aa_mask_id = denoiser.aa_mask_id
        self.struct_mask_id = denoiser.struct_mask_id

    def _propose_step(self, x, step_idx, total_steps, sampling_strategy, logits=None):
        """One DPLM2 reverse step: forward, sample, unmask top-confidence
        positions per modality on a linear schedule.

        If `logits` is supplied the internal `self.denoiser(x)` forward pass
        is skipped and the caller-provided logits are used directly. They
        must be the mask-corrected output of `self.denoiser(x)` (i.e. the
        same thing this method would have computed). This lets callers fuse
        the proposal forward pass with the twist evaluation, so a single backbone
        pass yields both the LM logits AND the twist value psi^theta(x).
        """
        aa_mask_id = self.aa_mask_id
        struct_mask_id = self.struct_mask_id
        mask_index = (x == aa_mask_id) | (x == struct_mask_id)

        if logits is None:
            logits = self.denoiser(x)                              # [B, L, V]
        type_ids = self.denoiser.dplm.get_modality_type(x)
        logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "argmax":
            cur_scores, sampled = logits.max(-1)
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            temperature = min_temp + (max_temp - min_temp) * (1 - step_idx / total_steps)
            sampled, cur_scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == "random":
            sampled, cur_scores = sample_from_categorical(logits, temperature=self.temperature)
        else:
            raise NotImplementedError(f"Unknown sampling strategy: {sampling_strategy}")

        # Linear schedule: rate goes from ~0 (few unmasked) to 1 (fully unmasked).
        rate = (step_idx + 1) / total_steps

        aa_position = type_ids.eq(1) & mask_index
        cutoff_len_aa = (
            aa_position.sum(1, keepdim=True).type_as(cur_scores) * rate
        ).long()
        _scores_for_topk_aa = (-cur_scores).masked_fill(~aa_position, 1000.0)

        struct_position = type_ids.eq(0) & mask_index
        cutoff_len_struct = (
            struct_position.sum(1, keepdim=True).type_as(cur_scores) * rate
        ).long()
        _scores_for_topk_struct = (-cur_scores).masked_fill(~struct_position, 1000.0)

        noise_scale = 1.0
        unmask_mask_aa = topk_masking(
            _scores_for_topk_aa, cutoff_len_aa,
            stochastic=True, temp=noise_scale * rate,
        )
        unmask_mask_struct = topk_masking(
            _scores_for_topk_struct, cutoff_len_struct,
            stochastic=True, temp=noise_scale * rate,
        )

        unmask_this_step = (unmask_mask_aa & aa_position) | (unmask_mask_struct & struct_position)

        x_next = x.clone()
        x_next[unmask_this_step] = sampled[unmask_this_step]
        return x_next

    @torch.no_grad()
    def sample(self, x_t, start_step, sampling_strategy="random", stop_t=None):
        """Iterate the DPLM2 reverse process from `x_t` (a partially-masked
        state at diffusion step `start_step`) down to a clean sample.

        Args:
            x_t: [B, L] int64 token ids at step `start_step`.
            start_step: index of the first reverse step to apply (0 means
                `x_t` is fully masked, `self.steps` means `x_t` is clean).
            sampling_strategy: "random" | "argmax" | "annealing@max:min".
            stop_t: if given, stop early after `stop_t` total reverse steps
                have been applied (default: run to `self.steps`).

        Returns:
            x: [B, L] (fully) denoised tokens.
        """
        if stop_t is None:
            stop_t = self.steps
        if stop_t < 0 or stop_t > self.steps:
            raise ValueError(f"stop_t must be in [0, {self.steps}], got {stop_t}")

        x = x_t.clone().to(self.device)
        for i in range(start_step, stop_t):
            x = self._propose_step(x, i, self.steps, sampling_strategy)
        return x
