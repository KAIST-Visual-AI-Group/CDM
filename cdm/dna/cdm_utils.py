"""CDM (Contrastive Twist Learning) training utilities for DNA discrete diffusion.

Holds the sample collectors, SMC loop (propose-then-reweight), per-epoch
positive-sample buffer, and loss functions used by ``train_cdm`` in
``cdm/dna/main.py``.

Public entry points:
  - ``_run_smc``              : single-SMC loop, propose-then-reweight
  - ``_sample_with_traj``     : base DDPM chain that records per-step states
  - ``_collect_pos_samples``  : fills a :class:`PosSamples` for the pos buffer
  - ``_collect_neg_samples``  : IS / SMC neg sampler (reward never enters weights)
  - ``compute_pos_loss``      : positive-phase CDM loss term
  - ``compute_neg_loss``      : negative-phase CDM loss term
  - ``CDMPosBuffer``          : per-epoch positive-sample buffer
  - ``ess_from_weights``      : normalized ESS of a weight vector
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from cdm.dna.rewards import evaluate_generation
from cdm.dna.samplers import _sample_categorical
from cdm.dna.utils import ess_normalized, ess_summary


# ══════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════


@dataclass
class PosSamples:
    """One positive-phase sample batch for CDM.

    traj             : list of [B, L] states, length ``num_steps + 1``.
                       Entry 0 is the initial fully-masked state; entry -1
                       is the final state before noise removal.
    x0               : [B, L] denoised terminal sample (argmax at t=eps).
    rewards          : [B] terminal reward r(x_0) per particle.
    weights_per_step : list[Tensor], length ``num_steps + 1``, aligned with
                       ``traj``.  Entry 0 is uniform (1/N).  The IS branch
                       sets all subsequent entries to ``softmax(r/alpha)``
                       (shared terminal weights).  The SMC branch sets
                       each entry to ``softmax(log_w_accum)`` after that
                       step's resample decision — uniform at steps that
                       actually resampled, non-uniform otherwise.
    extra            : dict.  Reserved keys:
                        - ``"ess_pair"`` : (mean, min) per-step SMC ESS
                          (``(nan, nan)`` for IS).
    """

    traj: List[torch.Tensor]
    x0: torch.Tensor
    rewards: torch.Tensor
    weights_per_step: List[torch.Tensor]
    extra: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  ESS helper
# ══════════════════════════════════════════════════════════════


def ess_from_weights(w):
    """Normalized ESS in [1/N, 1] from (possibly un-normalised) positive weights."""
    w = w.detach().to(torch.float64)
    w = w / w.sum().clamp(min=1e-12)
    n = w.shape[0]
    return (1.0 / w.pow(2).sum().clamp(min=1e-12).item()) / n


# ══════════════════════════════════════════════════════════════
#  SMC loop for CDM sample collection (propose-then-reweight)
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _run_smc(args, base_model, sampler, num_seqs, score_type, twist_net, tokenizer):
    """Single-SMC loop — propose then reweight at every step.

        for i in [0, num_steps):
            x_next ~ p_ref(· | x_t)                  [propose]
            (v_next, p_x0_next) = score(x_next, ...) [score + cache logits]
            log_w_accum += v_next - v_curr           [accumulate]
            if ESS < threshold: resample + reset accumulator
            else: carry accumulator + caches forward

    ``score_type``:
      - ``twist``    : ``base_model.forward_fused(x, sigma, twist_net)``
                       returns ``(log_p_x0, value)`` in one backbone pass.
      - ``x0_pred``  : ``base_model.forward(x, sigma).exp()`` gives p_x0;
                       sample M x0_hats, evaluate reward, logsumexp/M.

    p_x0 from the score step is cached and fed straight to the next
    propose call's ``_ddpm_cache_step`` — exact match in (x, sigma).

    Returns
    -------
    traj_states      : list[Tensor], length ``num_steps + 1``.
    x0_final         : [B, L] denoised terminal sample.
    weights_per_step : list[Tensor], length ``num_steps + 1``, aligned
                       with ``traj_states``.
    ess_steps        : list[float], length ``num_steps``.
    """
    assert score_type in ("twist", "x0_pred"), (
        f"_run_smc: unsupported score_type '{score_type}' (expected 'twist' or 'x0_pred')"
    )

    device = args.device
    total = num_seqs
    seq_len = args.seq_len
    mask_index = sampler.mask_index

    x = mask_index * torch.ones(total, seq_len, dtype=torch.int64, device=device)
    timesteps, dt = sampler._get_timesteps()

    traj_states = [x.clone()]
    weights_per_step = [torch.full((total,), 1.0 / total, device=device)]
    ess_steps: List[float] = []

    v_cache = None
    p_x0_cache = None
    log_w_accum = torch.zeros(total, device=device)

    for i in range(sampler.num_steps):
        t = timesteps[i] * torch.ones(total, 1, device=device)

        # v_curr: cached from the previous step, or a constant on step 0
        # (all step-0 particles are identical, so any constant cancels
        # inside the softmax — skip a redundant forward pass).
        v_curr = (
            v_cache
            if v_cache is not None
            else torch.ones(total, device=device, dtype=torch.float32)
        )

        # ── Propose x_next ~ p_ref(·|x_t), reusing cached p_x0 if any ──
        x_next, _ = sampler._ddpm_cache_step(
            base_model, x, t, dt, p_x0_cache,
        )

        # ── Score x_next at sigma_next; also recover p_x0 for the next step ──
        t_next = timesteps[i + 1] * torch.ones(total, 1, device=device)
        _, sigma_next = sampler._get_alpha_sigma(t_next)

        if score_type == "twist":
            log_p_x0_next, v_next = base_model.forward_fused(
                x_next, sigma_next, twist_net=twist_net,
            )
            p_x0_next = log_p_x0_next.exp()
        else:  # x0_pred
            p_x0_next = base_model.forward(x_next, sigma_next).exp()
            M = args.twist_M
            p_x0_all = p_x0_next.repeat_interleave(M, dim=0)
            x_all = x_next.repeat_interleave(M, dim=0)
            x0_hat = _sample_categorical(p_x0_all)
            x0_hat = torch.where(x_all == mask_index, x0_hat, x_all)
            rewards_list = evaluate_generation(
                x0_hat[:, :args.gen_length], tokenizer, args,
            )
            rewards_m = torch.tensor(
                rewards_list, device=device, dtype=torch.float32,
            ).reshape(total, M)
            v_next = torch.logsumexp(rewards_m / args.kl_weight, dim=-1) - math.log(M)
        v_next = v_next.to(torch.float32)

        # ── Accumulate incremental log-weight ──
        log_w_accum = log_w_accum + (v_next - v_curr)

        ess = ess_normalized(log_w_accum, dim=0)
        ess_steps.append(ess)

        should_resample = ess < args.ess_threshold

        if should_resample:
            w = torch.softmax(log_w_accum, dim=0)
            indices = torch.multinomial(w, total, replacement=True)
            x = x_next[indices]
            v_cache = v_next[indices]
            p_x0_cache = p_x0_next[indices]
            log_w_accum = torch.zeros(total, device=device)
        else:
            x = x_next
            v_cache = v_next
            p_x0_cache = p_x0_next

        # softmax(zeros) = uniform — correct right after a resample too.
        weights_per_step.append(torch.softmax(log_w_accum, dim=0))
        traj_states.append(x.clone())

    x0_final = sampler._noise_removal(base_model, x)

    return traj_states, x0_final, weights_per_step, ess_steps


@torch.no_grad()
def _sample_with_traj(args, base_model, sampler, num_seqs):
    """Run standard DDPM sampling for ``num_steps`` and record every state.

    No scoring, no resampling — pure p_ref chain used by the IS branches
    for pos and neg sample collection.  Returns
    ``(traj_states, x0_final)`` where ``traj_states`` has length
    ``num_steps + 1`` (entry 0 is the fully-masked init).
    """
    device = args.device
    total = num_seqs
    seq_len = args.seq_len
    mask_index = sampler.mask_index

    x = mask_index * torch.ones(total, seq_len, dtype=torch.int64, device=device)
    timesteps, dt = sampler._get_timesteps()
    traj_states = [x.clone()]

    p_x0_cache = None
    for i in range(sampler.num_steps):
        t = timesteps[i] * torch.ones(total, 1, device=device)
        x, p_x0_cache = sampler._ddpm_cache_step(base_model, x, t, dt, p_x0_cache)
        traj_states.append(x.clone())

    x0_final = sampler._noise_removal(base_model, x)
    return traj_states, x0_final


# ══════════════════════════════════════════════════════════════
#  Pos / Neg sample collectors
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _collect_pos_samples(
    args, sampler, base_model, tokenizer, num_seqs, alpha, twist_net=None,
) -> PosSamples:
    """Collect positive-phase samples for CDM.

    Branches on ``args.cdm_pos_sample_method``:
      - ``is``          : base DDPM chain from fully-masked.  terminal IS
                          weights ``softmax(r/alpha)`` are replicated across
                          every step.  Loss forward-noises ``x0`` to ``x_t``.
      - ``smc`` / ``asmc`` : bootstrapped-value SMC using
                          ``args.twist_score`` (typically ``x0_pred``).
                          Per-step weights come from the ESS-based
                          accumulator.  At the loss site:
                            * ``smc``  — forward-noise ``x0`` to ``x_t``,
                              weight by terminal ``weights_per_step[-1]``.
                            * ``asmc`` — read ``traj[t_idx]`` directly,
                              weight by per-step ``weights_per_step[t_idx]``.
    """
    device = args.device
    method = args.cdm_pos_sample_method

    if method == "is":
        traj_states, x0 = _sample_with_traj(args, base_model, sampler, num_seqs)
        rewards_list = evaluate_generation(x0[:, :args.gen_length], tokenizer, args)
        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)

        terminal_weights = F.softmax((rewards / alpha).float(), dim=0)
        weights_per_step = [terminal_weights] * (sampler.num_steps + 1)

        return PosSamples(
            traj=traj_states,
            x0=x0,
            rewards=rewards,
            weights_per_step=weights_per_step,
            extra={"ess_pair": (float("nan"), float("nan"))},
        )

    elif method in ("smc", "asmc"):
        assert args.twist_score == "x0_pred", (
            f"pos-SMC requires twist_score='x0_pred' (tsmc not implemented), "
            f"got '{args.twist_score}'"
        )
        traj_states, x0, weights_per_step, ess_steps = _run_smc(
            args, base_model, sampler, num_seqs,
            score_type=args.twist_score,
            twist_net=twist_net,
            tokenizer=tokenizer,
        )
        rewards_list = evaluate_generation(x0[:, :args.gen_length], tokenizer, args)
        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)

        return PosSamples(
            traj=traj_states,
            x0=x0,
            rewards=rewards,
            weights_per_step=weights_per_step,
            extra={"ess_pair": ess_summary(ess_steps)},
        )

    raise ValueError(f"Unknown cdm_pos_sample_method '{method}'")


@torch.no_grad()
def _collect_neg_samples(args, sampler, base_model, twist_net, tokenizer, num_seqs):
    """Collect negative-phase samples for CDM.

    The neg phase targets π^θ = p_ref · ψ^θ, so **the true reward never
    enters the weights** — it is used only for logging on the final state.

    Branches on ``args.cdm_neg_sample_method``:
      - ``is``  : base DDPM chain from fully-masked.  No weights during
                  sampling — the IS correction is applied at the loss
                  site as ``softmax(log ψ^θ(x_t))``.  ``weights_per_step``
                  is ``None``.
      - ``smc`` : twist-guided SMC using ``twist_net`` (typically the EMA
                  shadow) as the score function.  Per-step weights come
                  from the accumulator.

    Returns ``(traj_states, rewards, weights_per_step, ess_pair)``.
    """
    device = args.device
    method = args.cdm_neg_sample_method

    if method == "is":
        traj_states, x0 = _sample_with_traj(args, base_model, sampler, num_seqs)
        rewards_list = evaluate_generation(x0[:, :args.gen_length], tokenizer, args)
        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
        return traj_states, rewards, None, (float("nan"), float("nan"))

    elif method == "smc":
        assert twist_net is not None, (
            "cdm_neg_sample_method='smc' requires a twist_net (e.g. EMA shadow)"
        )
        traj_states, x0, weights_per_step, ess_steps = _run_smc(
            args, base_model, sampler, num_seqs,
            score_type="twist",
            twist_net=twist_net,
            tokenizer=tokenizer,
        )
        rewards_list = evaluate_generation(x0[:, :args.gen_length], tokenizer, args)
        rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
        return traj_states, rewards, weights_per_step, ess_summary(ess_steps)

    raise ValueError(f"Unknown cdm_neg_sample_method '{method}'")


# ══════════════════════════════════════════════════════════════
#  Per-epoch positive-sample buffer
# ══════════════════════════════════════════════════════════════


class CDMPosBuffer:
    """Per-epoch buffer of positive CDM samples."""

    def __init__(self, args):
        self.args = args
        self.cdm_buffer_size = args.cdm_pos_batch_size
        self.data: Optional[PosSamples] = None

    def __len__(self):
        return 0 if self.data is None else self.data.rewards.shape[0]

    def clear(self):
        self.data = None

    @property
    def smc_ess_mean(self):
        if self.data is None:
            return float("nan")
        mean, _ = self.data.extra.get("ess_pair", (float("nan"), float("nan")))
        return float(mean)

    @property
    def smc_ess_min(self):
        if self.data is None:
            return float("nan")
        _, mn = self.data.extra.get("ess_pair", (float("nan"), float("nan")))
        return float(mn)

    @torch.no_grad()
    def fill(self, sampler, base_model, tokenizer, alpha, twist_net=None):
        self.data = _collect_pos_samples(
            self.args, sampler, base_model, tokenizer,
            num_seqs=self.cdm_buffer_size,
            alpha=alpha,
            twist_net=twist_net,
        )

    def sample(self, batch_size, device, idx=None) -> PosSamples:
        assert self.data is not None, "buffer is empty; call fill() first"
        n = self.data.rewards.shape[0]
        if idx is None:
            idx = torch.randperm(n, device=device)[:batch_size]

        traj = [s[idx] for s in self.data.traj]
        rewards = self.data.rewards[idx]
        # Rescale so the sub-sampled weighted sum remains an unbiased
        # estimate of the full-pool weighted sum.
        scale = self.cdm_buffer_size / max(batch_size, 1)
        weights_per_step = [w[idx] * scale for w in self.data.weights_per_step]

        return PosSamples(
            traj=traj,
            x0=self.data.x0[idx],
            rewards=rewards,
            weights_per_step=weights_per_step,
            extra=self.data.extra,
        )

    def is_ess(self, alpha):
        """IS ESS of the buffered pool.  Returns ``(ess, ess)``."""
        if self.data is None:
            return float("nan"), float("nan")
        method = self.args.cdm_pos_sample_method
        if method == "is":
            log_w = (self.data.rewards / alpha).to(torch.float64)
            ess = ess_normalized(log_w, dim=0)
            return ess, ess
        # SMC: terminal IS weights live at weights_per_step[-1].
        w = self.data.weights_per_step[-1].to(torch.float64)
        n = w.shape[0]
        ess = (1.0 / w.pow(2).sum().clamp(min=1e-12).item()) / n
        return ess, ess


# ══════════════════════════════════════════════════════════════
#  Pos / Neg loss
# ══════════════════════════════════════════════════════════════


def compute_pos_loss(args, base_model, pos, twist_net, sigma_t, t_idx):
    """Positive-phase CDM loss term at diffusion step ``t_idx``.

    Branches on ``args.cdm_pos_sample_method``:

      - ``is`` / ``smc`` : forward-noise ``pos.x0`` to ``x_t`` via
                 ``base_model.q_xt`` and weight with the terminal IS
                 weights ``pos.weights_per_step[-1]`` (``softmax(r/alpha)``
                 for ``is``; ESS-accumulator terminal weight for ``smc``).

      - ``asmc`` : consume the SMC intermediate ``pos.traj[t_idx]``
                   directly, weighted by ``pos.weights_per_step[t_idx]``.
                   Under ESS-based resampling these are uniform only at
                   steps that actually resampled — plain mean is NOT
                   correct in general.
    """
    method = args.cdm_pos_sample_method

    if method in ("is", "smc"):
        x0 = pos.x0.detach()
        # sigma_t is [B, 1]; duo convention: alpha_t = exp(-sigma_t).
        alpha_t = torch.exp(-sigma_t.squeeze(-1)).unsqueeze(-1)
        x_t_pos = base_model.q_xt(x0, alpha_t)
        log_psi_pos = twist_net(x_t_pos, sigma_t)
        W_bar = pos.weights_per_step[-1].detach()
        ess = ess_from_weights(W_bar)
        return (W_bar * log_psi_pos).sum(), ess

    if method == "asmc":
        x_t = pos.traj[t_idx].detach()
        log_psi_pos = twist_net(x_t, sigma_t)
        W_bar = pos.weights_per_step[t_idx].detach()
        ess = ess_from_weights(W_bar)
        return (W_bar * log_psi_pos).sum(), ess

    raise ValueError(f"Unknown cdm_pos_sample_method '{method}'")


def compute_neg_loss(
    traj, twist_net, sigma_t, t_idx,
    weights_per_step=None, head_weights=None,
):
    """Negative-phase CDM loss term at diffusion step ``t_idx``.

    Two heads may be in play:
      - ``twist_net``     : the live (trainable) head, always used for
                            the gradient-carrying ``log_psi_neg`` term.
      - ``head_weights``  : optional frozen head (e.g. EMA shadow) used
                            to compute the IS weights in the IS branch.
                            Decoupling weights from the live head makes
                            the neg gradient less self-referential.
                            ``None`` reuses the live head.

    IS branch (``weights_per_step is None``): q = p_ref was the proposal,
    so the p_ref → π^θ importance weight is ψ^θ itself; self-normalise
    with ``softmax(log ψ^θ(x_t))``.  The true reward never appears here.

    SMC branch (``weights_per_step`` given): particles already carry
    their own per-step IS weights — pick ``weights_per_step[t_idx]``.
    """
    x_t = traj[t_idx].detach()
    log_psi_neg = twist_net(x_t, sigma_t)

    if weights_per_step is None:
        if head_weights is not None and head_weights is not twist_net:
            with torch.no_grad():
                log_w = head_weights(x_t, sigma_t).detach().float()
        else:
            log_w = log_psi_neg.detach().float()
        weights = F.softmax(log_w, dim=0)
    else:
        weights = weights_per_step[t_idx].detach()

    ess = ess_from_weights(weights)
    return (weights * log_psi_neg).sum(), ess
