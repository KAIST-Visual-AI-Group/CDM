"""
Diffusion model for DNA sequences using CNN backbone.

Architecture matches the duo codebase (trainer_base.py + algo.py) so that
checkpoints trained with duo can be loaded directly.

Key design choices matching duo:
  - LogLinear noise: forward(t) -> (dalpha_t, alpha_t) where alpha_t = 1 - (1-eps)*t
  - sigma = -log(alpha_t)  (conversion done via _sigma_from_alphat)
  - _process_sigma: takes [B, L] or [B, 1] sigma, averages to [B]
  - q_xt(x, alpha_t): masks with probability 1 - alpha_t
  - MDLM subs parameterization for _process_model_output
"""

import itertools
import math

import lightning as L
import torch
import torchmetrics
import transformers
from torch import Tensor

import cdm.dna.models as models

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        return torch.exp(self.mean_value / self.weight)


# ══════════════════════════════════════════════════════════════
#  Noise schedule — matches duo/trainer_base.py LogLinear
# ══════════════════════════════════════════════════════════════

class LogLinear(torch.nn.Module):
    """LogLinear noise schedule from duo codebase.

    forward(t) returns (dalpha_t, alpha_t) where:
        alpha_t = 1 - (1 - eps) * t
        dalpha_t = -(1 - eps)
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-3

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = -(1 - self.eps)
        return dalpha_t, alpha_t


# ══════════════════════════════════════════════════════════════
#  Diffusion model — matches duo/trainer_base.py + algo.py
# ══════════════════════════════════════════════════════════════

class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer

        # DNA vocab: A=0, C=1, G=2, T=3 → vocab_size=4
        # Mask token appended as index 4 (matching duo AbsorbingState)
        self.vocab_size = self.tokenizer.vocab_size  # 4
        self.mask_index = self.vocab_size  # 4
        self.vocab_size += 1  # now 5

        self.parameterization = self.config.parameterization
        self.antithetic_sampling = self.config.training.antithetic_sampling

        # Build CNN backbone — matches duo: models.dnaconv.CNNModel(config.model, ...)
        if self.config.backbone == 'cnn':
            self.backbone = models.dnaconv.CNNModel(
                self.config.model,
                alphabet_size=self.vocab_size,
                num_cls=2,
            )
        else:
            raise ValueError(f'Unknown backbone: {self.config.backbone}')

        self.T = self.config.T
        self.subs_masking = self.config.subs_masking

        self.softplus = torch.nn.Softplus()
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(),
            'bpd': BPD(),
            'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')

        # Noise schedule — matches duo's LogLinear (no parameters)
        self.noise = LogLinear()

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0

    def on_load_checkpoint(self, checkpoint):
        if self.ema and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def _sigma_from_alphat(self, alpha_t):
        """Convert alpha_t to sigma: sigma = -log(alpha_t). Matches duo."""
        return -torch.log(alpha_t)

    def _process_sigma(self, sigma):
        """Process sigma for backbone input. Matches duo/trainer_base.py Diffusion._process_sigma.

        Input sigma can be [B, L] or [B, 1] or [B]; output is always [B].
        """
        if sigma is None:
            return sigma
        if sigma.ndim == 2:
            sigma = sigma.mean(-1).squeeze()
            if sigma.ndim == 0:
                sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma):
        """MDLM subs parameterization. Matches duo/algo.py MDLM._process_model_output."""
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity
        model_output = model_output - torch.logsumexp(model_output, dim=-1, keepdim=True)

        unmasked_indices = (xt != self.mask_index)
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def forward(self, x, sigma):
        """Returns log score: log p(x0 | xt, sigma).

        Matches duo/trainer_base.py TrainerBase.forward:
            sigma = self._process_sigma(sigma)
            model_output = self.backbone(xt, sigma)
            return self._process_model_output(model_output, xt, sigma)
        """
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            model_output = self.backbone(x, sigma)
        return self._process_model_output(model_output=model_output, xt=x, sigma=sigma)

    def forward_fused(self, x, sigma, twist_net):
        """Single backbone pass returning both log p(x0|xt) and twist value.

        For FrozenBackboneTwist (twist shares this model's backbone), one
        forward gives both. For FineTuneBackboneTwist (twist has its own
        fine-tuned backbone), the twist's backbone differs in weights, so
        we must recompute hidden states using the twist's backbone — only
        the proposal logits come from this model.

        Args:
            x: [B, L] token indices
            sigma: [B, 1] or [B] noise level
            twist_net: a twist module exposing _apply_head(hidden) -> [B]
                and a `needs_all_hidden` bool attribute indicating whether
                the head consumes per-layer intermediate hidden states.

        Returns:
            log_p_x0: [B, L, V] log probabilities (subs parameterized)
            value: [B] scalar twist values
        """
        sigma = self._process_sigma(sigma)
        return_all_hidden = twist_net.needs_all_hidden
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            logits, hidden = self.backbone.forward_fused(
                x, sigma, return_all_hidden=return_all_hidden,
            )
        log_p_x0 = self._process_model_output(model_output=logits, xt=x, sigma=sigma)

        # If twist has its own (fine-tuned) backbone, recompute hidden states
        # using twist's weights rather than the base model's.
        twist_bb = getattr(twist_net, 'backbone', None)
        if twist_bb is not None and twist_bb is not self.backbone:
            twist_hidden = twist_bb.get_hidden_states(x, sigma)
        else:
            twist_hidden = hidden

        value = twist_net._apply_head(twist_hidden)   # [B] — handles per-head pooling
        return log_p_x0, value

    def q_xt(self, x, alpha_t):
        """Forward noise process: mask positions with probability 1 - alpha_t.

        Matches duo/trainer_base.py AbsorbingState.q_xt.

        Args:
            x: [B, L] int64 token indices
            alpha_t: [B, 1] or scalar, probability of keeping each token
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _loss(self, x0, attention_mask=None):
        t = torch.rand(x0.shape[0], dtype=self.dtype, device=self.device)
        if self.antithetic_sampling:
            t = torch.cat([t[: x0.shape[0] // 2], 1 - t[: x0.shape[0] // 2]])
        if self.T > 0:
            t = (t * self.T).to(torch.int) / self.T
            t += (1 / self.T)

        # noise(t) -> (dalpha_t, alpha_t)
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)  # [B, 1]
        sigma = self._sigma_from_alphat(alpha_t)  # [B, 1]

        xt = self.q_xt(x0, alpha_t)
        log_x_theta = self.forward(xt, sigma=sigma)

        # MDLM continuous-time ELBO
        log_p_theta = torch.gather(log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        nll_per_token = log_p_theta * dalpha_t / (1 - alpha_t.squeeze(-1))
        diffusion_loss = -nll_per_token

        if attention_mask is not None:
            diffusion_loss = diffusion_loss * attention_mask

        from dataclasses import dataclass

        @dataclass
        class Loss:
            loss: torch.FloatTensor
            nlls: torch.FloatTensor
            token_mask: torch.FloatTensor

        return Loss(
            loss=diffusion_loss.mean(),
            nlls=diffusion_loss.sum(dim=-1),
            token_mask=torch.ones_like(x0, dtype=torch.float32),
        )

    def _compute_loss(self, batch, prefix):
        attention_mask = batch.get('attention_mask', None)
        losses = self._loss(batch['input_ids'], attention_mask)
        loss = losses.loss

        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix='train')
        self.log(name='trainer/loss', value=loss.item(), on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, prefix='val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(), self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
        return optimizer

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        """Ancestral sampling step matching duo AbsorbingState._ancestral_update.

        Uses alpha_t parameterization:
            q(x_s | x_t, x0): unmask with prob (alpha_s - alpha_t), stay masked with prob (1 - alpha_s)
        """
        _, alpha_t = self.noise(t)
        _, alpha_s = self.noise(t - dt)
        alpha_t = alpha_t.unsqueeze(-1)  # [B, 1]
        alpha_s = alpha_s.unsqueeze(-1)  # [B, 1]
        sigma = self._sigma_from_alphat(alpha_t)

        if p_x0 is None:
            p_x0 = self.forward(x, sigma).exp()

        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x

        if torch.allclose(xs, x) and not self.time_conditioning:
            p_x0_cache = p_x0
        else:
            p_x0_cache = None

        return p_x0_cache, xs

    @torch.no_grad()
    def _sample(self, num_steps=None, batch_size=None, seq_length=None):
        """Basic DDPM sampling for evaluation."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if batch_size is None:
            batch_size = self.config.loader.eval_batch_size
        if seq_length is None:
            seq_length = self.config.model.length

        x = self._sample_prior(batch_size, seq_length).to(self.device)
        timesteps = torch.linspace(1, self.sampling_eps, num_steps + 1, device=self.device)
        dt = (1 - self.sampling_eps) / num_steps

        p_x0_cache = None
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(batch_size, device=self.device)
            p_x0_cache, x = self._ddpm_caching_update(x, t, dt, p_x0_cache)

        # Final noise removal (greedy decode at t=eps)
        t_final = self.sampling_eps * torch.ones(batch_size, device=self.device)
        _, alpha_final = self.noise(t_final)
        sigma_final = self._sigma_from_alphat(alpha_final.unsqueeze(-1))
        x = self.forward(x, sigma_final).argmax(dim=-1)
        return x
