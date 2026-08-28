"""CDM (Contrastive Twist Learning) training utilities for DPLM2 protein generation.

Holds the sample collectors, SMC loop, forward noising, loss functions,
and positive-sample buffer used by ``train_cdm`` in ``cdm/proteins/main.py``.

Shared DPLM2 ops (``_score``, ``create_masked_input``) still live in main.py
and are imported lazily inside functions to avoid a circular import.

Public entry points:
  - ``_forward_noise_dplm``         : DPLM2 dual-modality forward noising
  - ``_collect_pos_samples_dplm``   : fills a PosSamples for the pos buffer
  - ``_collect_neg_samples_dplm``   : fresh-per-step neg-phase sample collector
  - ``compute_pos_loss_dplm``       : positive-phase CDM loss term
  - ``compute_neg_loss_dplm``       : negative-phase CDM loss term
  - ``CDMPosBuffer``                : per-epoch positive sample buffer
  - LR scheduler / alpha schedule / ESS helpers
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model, decay=None):
        decay = self.decay if decay is None else decay
        for p_ema, p in zip(self.shadow.parameters(), model.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ══════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════


@dataclass
class PosSamples:
    """One positive-phase sample batch for CDM.

    Attributes
    ----------
    traj : list of Tensors
        Per-step states ``[B, L]``, length ``steps + 1``. Entry 0 is the
        initial (fully-masked) state, entry ``-1`` is the clean x0.
    rewards : Tensor
        Terminal reward ``r(x_0)`` per particle, shape ``[B]``.
    weights_per_step : List[Tensor]
        Per-step IS weights aligned with ``traj`` (length ``steps + 1``).
        The loss site picks the appropriate entry:
          - ``is`` / ``smc`` / ``tsmc`` (forward-noised x0): ``[-1]``.
          - ``asmc`` (intermediate states): ``[t_idx]``.
        With ESS-based resampling the per-step weights are only uniform
        at steps that actually resampled.  For ``is``, all entries share
        the same terminal IS weights tensor.
    extra : dict
        Free-form metadata.  Reserved keys:
          - ``"ess_pair"`` : tuple[float, float] — (mean, min) per-step ESS.
    """

    traj: List[torch.Tensor]
    rewards: torch.Tensor
    weights_per_step: List[torch.Tensor]
    extra: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  Positive-sample buffer
# ══════════════════════════════════════════════════════════════


class CDMPosBuffer:
    """Per-epoch buffer of positive samples for CDM."""

    def __init__(self, args):
        self.args = args
        self.cdm_buffer_size = args.cdm_buffer_size
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
    def fill(self, sampler, denoiser, reward_fn, num_seqs, collect_fn, alpha):
        """Fill the buffer with one ``collect_fn`` call.

        ``collect_fn`` signature:
            ``(args, sampler, denoiser, reward_fn, num_seqs, alpha) -> PosSamples``
        """
        assert num_seqs == self.cdm_buffer_size, (
            f"CDMPosBuffer.fill expects num_seqs={num_seqs} == "
            f"cdm_buffer_size={self.cdm_buffer_size}"
        )
        self.data = collect_fn(
            self.args, sampler, denoiser, reward_fn, num_seqs, alpha,
        )

    def sample(self, batch_size, device, idx=None) -> PosSamples:
        """Draw a mini-batch from the buffer."""
        assert self.data is not None, "buffer is empty; call fill() first"
        n = self.data.rewards.shape[0]
        if idx is None:
            idx = torch.randperm(n, device=device)[:batch_size]

        traj = [s[idx] for s in self.data.traj]
        rewards = self.data.rewards[idx]

        # Rescale each step's IS weights so the sub-sampled weighted sum
        # remains an unbiased estimate of the full-pool weighted sum.
        scale = self.cdm_buffer_size / batch_size
        weights_per_step = [w[idx] * scale for w in self.data.weights_per_step]

        return PosSamples(
            traj=traj,
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
            ess = _ess_normalized(log_w, dim=0)
            return ess, ess
        # Terminal IS weights live at weights_per_step[-1].
        w = self.data.weights_per_step[-1].to(torch.float64)
        n = w.shape[0]
        ess = (1.0 / w.pow(2).sum().clamp(min=1e-12).item()) / n
        return ess, ess


# ══════════════════════════════════════════════════════════════
#  ESS helpers
# ══════════════════════════════════════════════════════════════


def _ess_normalized(log_w, dim):
    """Normalized ESS in [1/N, 1] from unnormalized log weights along ``dim``."""
    n = log_w.shape[dim]
    w = torch.softmax(log_w, dim=dim)
    ess = 1.0 / w.pow(2).sum(dim=dim).clamp(min=1e-12)
    return (ess / n).mean().item()


def _ess_from_weights(w):
    """Normalized ESS in [1/N, 1] from (possibly un-normalised) positive weights."""
    w = w.detach().to(torch.float64)
    w = w / w.sum().clamp(min=1e-12)
    n = w.shape[0]
    return (1.0 / w.pow(2).sum().clamp(min=1e-12).item()) / n


def ess_summary(ess_steps):
    """Reduce list of per-step ESS to (mean, min)."""
    if not ess_steps:
        return float("nan"), float("nan")
    return float(np.mean(ess_steps)), float(np.min(ess_steps))


# ══════════════════════════════════════════════════════════════
#  LR scheduler helpers
# ══════════════════════════════════════════════════════════════


def make_linear_decay_scheduler(optimizer, total_steps, decay_start_frac):
    """Constant LR, then linear decay to 0 starting at ``decay_start_frac``."""
    decay_start = int(total_steps * decay_start_frac)

    def lr_lambda(step):
        if step <= decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), decay_start


# ══════════════════════════════════════════════════════════════
#  Alpha schedule
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _forward_noise_dplm(x0, step_idx, total_steps, denoiser):
    """DPLM2 forward noising: mask each non-special token i.i.d. with prob
    ``step_idx / total_steps``.

    AA positions are replaced with ``aa_mask_id``, struct positions with
    ``struct_mask_id``.  Special tokens (BOS, EOS, PAD) are never masked.
    ``step_idx=0`` -> clean, ``step_idx=total_steps`` -> fully masked.
    """
    if step_idx == 0:
        return x0.clone()

    mask_prob = step_idx / total_steps
    rand = torch.rand_like(x0, dtype=torch.float32)
    should_mask = rand < mask_prob

    type_ids = denoiser.dplm.get_modality_type(x0)
    non_special = denoiser.dplm.get_non_special_symbol_mask(x0)

    # Only mask non-special token positions.
    should_mask = should_mask & non_special

    x_t = x0.clone()
    aa_mask = should_mask & (type_ids == denoiser.dplm.aa_type)
    struct_mask = should_mask & (type_ids == denoiser.dplm.struct_type)

    x_t[aa_mask] = denoiser.aa_mask_id
    x_t[struct_mask] = denoiser.struct_mask_id

    return x_t


# ══════════════════════════════════════════════════════════════
#  Trajectory-recording sampler wrapper
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _sample_with_traj(sampler, x_t, total_steps, sampling_strategy, chunk_size=None):
    """Run DPLM2 reverse process for ``total_steps`` while recording
    per-step states.

    Returns ``(x_final, traj)`` where ``traj`` is a list of length
    ``total_steps + 1``: entry 0 is ``x_t`` (initial masked state),
    entry -1 is the final denoised state.

    ``chunk_size`` optionally caps the B-axis per ``_propose_step`` call.
    """
    x = x_t.clone().to(sampler.device)
    traj = [x.clone()]
    total = x.shape[0]

    for i in range(total_steps):
        if chunk_size is not None and chunk_size < total:
            x_chunks = []
            for c0 in range(0, total, chunk_size):
                c1 = min(c0 + chunk_size, total)
                x_chunks.append(
                    sampler._propose_step(x[c0:c1], i, total_steps, sampling_strategy)
                )
            x = torch.cat(x_chunks, dim=0)
        else:
            x = sampler._propose_step(x, i, total_steps, sampling_strategy)
        traj.append(x.clone())

    return x, traj


# ══════════════════════════════════════════════════════════════
#  SMC loop for CDM sample collection
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _run_smc_dplm(
    args, sampler, denoiser, reward_fn, num_seqs, score_type, smc_mode, alpha,
    stop_t=None, terminal_reward=False,
):
    """SMC loop for DPLM2 CDM sample collection.

    Unconditional protein co-generation — starts from fully-masked
    dual-modality sequences.  Uses DPLM2 reverse step as q = p_ref.

    Every step uniformly applies the same score/accumulate/ESS-resample
    logic with ``score_type``.  When ``terminal_reward=True`` (tsmc
    pattern) the last step skips the twist-based ESS resample, and a
    post-loop correction overwrites ``weights_steps[-1]`` to swap the
    last-step twist contribution for the true reward:
    ``log_w_last = log_w_accum + r(x_T)/alpha - psi(x_T)``.
    Skipping the terminal resample preserves the telescoped
    ``log_w_accum = log psi(x_0) - log psi(x_last_resample)``, so the
    final weight reduces to ``r(x_0)/alpha - log psi(x_last_resample)``
    (and to ``r(x_0)/alpha - 1`` when there were no resamples).

    ``smc_mode``:
      - ``single_smc``: K=1, resample across the whole batch (dim 0).
      - ``multi_smc`` : K=cdm_K per sequence, resample within K-groups
        [not yet implemented for ESS-based resampling].

    ``stop_t`` (optional, default = ``args.steps``):
        Number of SMC steps to run.  Must satisfy 1 <= stop_t <= steps.
        ``terminal_reward`` requires ``stop_t == steps``.

    Returns
    -------
    traj_states   : list[Tensor], length ``stop_t + 1``.
    rewards       : ``[total]`` true reward on final state when
                    ``terminal_reward=True``; ``None`` otherwise.
    weights_steps : list[Tensor] of length ``stop_t + 1``, aligned with
                    ``traj_states``.  Entry 0 is uniform (initial
                    particles are identical).  Each subsequent entry is
                    softmax(log_w_accum) after that step's resample
                    decision — uniform whenever resampling occurred,
                    non-uniform whenever it was skipped.  With
                    ``terminal_reward=True`` the last entry folds in the
                    true reward correction.
    extras        : dict with ``"x0_all"``, ``"ess_steps"``.
    """
    # Deferred imports to avoid circular import with main.py.
    from main import _score, create_masked_input

    device = args.device
    total_steps = args.steps
    if stop_t is None:
        stop_t = total_steps
    assert 1 <= stop_t <= total_steps, (
        f"stop_t must be in [1, {total_steps}], got {stop_t}"
    )
    B = num_seqs
    K = 1 if smc_mode == "single_smc" else args.cdm_K
    total = B * K

    # Initialise fully-masked sequences.
    x = create_masked_input(
        tokenizer=denoiser.dplm.tokenizer,
        seq_length=args.seq_length,
        num_seqs=total,
        device=device,
    )

    # ── State / caches / accumulator ──
    traj_states = [x.clone()]
    # Per-step normalised weights aligned with ``traj_states``.  Index 0 is
    # uniform (initial particles are identical).  Each subsequent entry is
    # softmax(log_w_accum) after that step's resample decision — uniform
    # whenever resampling occurred, non-uniform whenever it was skipped.
    weights_steps = [torch.full((total,), 1.0 / total, device=device)]
    ess_steps = []
    v_cache = None          # None on step 0 → v_curr = 1 (constant)
    logits_cache = None
    log_w_accum = torch.zeros(total, device=device)

    fuse_twist = score_type == "twist"
    cache_logits = score_type in ("twist", "x0_pred")
    chunk_s = args.cdm_train_chunk_size

    rewards_all = None

    for i in range(stop_t):
        # ── Propose (B-chunked; reuse cached logits from prior eval) ──
        if chunk_s is not None and chunk_s < total:
            x_next_chunks = []
            for c0 in range(0, total, chunk_s):
                c1 = min(c0 + chunk_s, total)
                lc = logits_cache[c0:c1] if logits_cache is not None else None
                x_next_chunks.append(
                    sampler._propose_step(
                        x[c0:c1], i, total_steps, args.sampling_strategy, logits=lc,
                    )
                )
            x_next = torch.cat(x_next_chunks, dim=0)
        else:
            x_next = sampler._propose_step(
                x, i, total_steps, args.sampling_strategy, logits=logits_cache,
            )

        # ── v_curr: cached from prior step, or constant 1 on step 0 ──
        # All step-0 particles are identical (fully masked), so any constant
        # cancels inside the softmax — skip the forward pass.
        if v_cache is None:
            v_curr = torch.ones(total, device=device)
        else:
            v_curr = v_cache

        # ── Score x_next (fused with logits when possible) ──
        if fuse_twist:
            if chunk_s is not None and chunk_s < total:
                logits_chunks, v_chunks = [], []
                for c0 in range(0, total, chunk_s):
                    c1 = min(c0 + chunk_s, total)
                    l_c, v_c = denoiser(
                        x_next[c0:c1], return_logits=True, return_value=True,
                    )
                    logits_chunks.append(l_c)
                    v_chunks.append(v_c)
                logits_next = torch.cat(logits_chunks, dim=0)
                v_next = torch.cat(v_chunks, dim=0).to(torch.float32)
            else:
                logits_next, v_next = denoiser(
                    x_next, return_logits=True, return_value=True,
                )
                v_next = v_next.to(torch.float32)
        elif cache_logits:
            v_next, logits_next = _score(
                denoiser, sampler, x_next, i + 1, args, reward_fn,
                score=score_type,
                chunk_size=args.chunk_m_size,
                return_logits=True,
            )
            v_next = v_next.to(torch.float32)
        else:
            logits_next = None
            v_next = _score(
                denoiser, sampler, x_next, i + 1, args, reward_fn,
                score=score_type,
                chunk_size=args.chunk_m_size,
            ).to(torch.float32)

        # Accumulate incremental log-weight across (possibly skipped) steps.
        log_w_accum = log_w_accum + (v_next - v_curr)

        # Skip the twist-based resample at the terminal step when the
        # caller will apply a true-reward correction (tsmc)
        skip_resample = terminal_reward and (i == stop_t - 1)

        if smc_mode == "single_smc":
            ess_norm = _ess_normalized(log_w_accum, dim=0)
            ess_steps.append(ess_norm)

            should_resample = (not skip_resample) and (ess_norm < args.ess_threshold)

            if should_resample:
                w = torch.softmax(log_w_accum, dim=0)
                indices = torch.multinomial(w, total, replacement=True)

                x = x_next[indices]
                v_cache = v_next[indices]
                logits_cache = logits_next[indices] if cache_logits else None
                log_w_accum = torch.zeros(total, device=device)

            else:
                x = x_next
                v_cache = v_next
                logits_cache = logits_next if cache_logits else None

            # softmax(zeros) = uniform — correct after resample as well.
            step_weights = torch.softmax(log_w_accum, dim=0)

        else:
            # multi_smc: the `total = B * K` particles are arranged as
            # B independent SMC groups of K particles each. Each group
            # gets its own normalised ESS and decides independently
            # whether to resample. log_w / weights are normalised within
            # each group (softmax over dim=1).
            log_w_mat = log_w_accum.view(B, K)
            w_mat = torch.softmax(log_w_mat, dim=1)                    # [B, K]

            ess_per_group = 1.0 / w_mat.pow(2).sum(dim=1).clamp(min=1e-12)  # [B]
            ess_per_group_norm = ess_per_group / K                          # [B], in [1/K, 1]
            ess_steps.append(ess_per_group_norm.mean().item())

            need_resample = (
                torch.zeros(B, dtype=torch.bool, device=device)
                if skip_resample
                else ess_per_group_norm < args.ess_threshold
            )                                                           # [B]

            if need_resample.any():
                # `row_idx` is the identity [0..K-1]; `resampled_idx` is
                # drawn from each group's weights. Groups that need to
                # resample take `resampled_idx`, the rest keep identity.
                row_idx = torch.arange(K, device=device).unsqueeze(0).expand(
                    B, K,
                )
                resampled_idx = torch.multinomial(w_mat, K, replacement=True)  # [B, K]
                idx_mat = torch.where(
                    need_resample.unsqueeze(1), resampled_idx, row_idx,
                )                                                           # [B, K]
                base = torch.arange(B, device=device).unsqueeze(1) * K       # [B, 1]
                flat_idx = (idx_mat + base).view(-1)                         # [total]

                x = x_next[flat_idx]
                v_cache = v_next[flat_idx]
                logits_cache = logits_next[flat_idx] if cache_logits else None

                # Reset only the groups that resampled; keep the running
                # log_w for the groups that didn't.
                new_log_w_mat = log_w_mat.clone()
                new_log_w_mat[need_resample] = 0.0
                log_w_accum = new_log_w_mat.view(-1)
            else:
                x = x_next
                v_cache = v_next
                logits_cache = logits_next if cache_logits else None

            # Per-group-normalised weights aligned with the (possibly
            # resampled) log_w_accum — uniform within groups that
            # resampled, non-uniform within the rest.
            step_weights = torch.softmax(log_w_accum.view(B, K), dim=1).view(-1)

        traj_states.append(x.clone())
        weights_steps.append(step_weights)

    rewards_all = None
    if terminal_reward:
        assert stop_t == total_steps, (
            "terminal_reward correction requires the full chain "
            f"(stop_t={stop_t}, total_steps={total_steps})"
        )
        # True-reward correction on the final state: replaces the last-step
        # twist contribution with r(x_T)/alpha.  Overwrites weights_steps[-1].
        reward_chunk = args.chunk_m_size
        if reward_chunk is not None and reward_chunk < total:
            reward_parts = []
            for c0 in range(0, total, reward_chunk):
                c1 = min(c0 + reward_chunk, total)
                reward_parts.append(reward_fn(x[c0:c1]).to(torch.float32))
            rewards_all = torch.cat(reward_parts, dim=0)
        else:
            rewards_all = reward_fn(x).to(torch.float32)

        # NOTE: v_cache = log psi(x_0) (added in the last step of the for loop) 
        # subtracting cancels log psi(x_0), yielding r(x_0)/alpha - log psi(x_1).
        log_w_last = log_w_accum + rewards_all / alpha - v_cache
        weights_steps[-1] = torch.softmax(log_w_last, dim=0)

    extras = {
        "x0_all": x,
        "ess_steps": ess_steps,
    }

    return traj_states, rewards_all, weights_steps, extras


# ══════════════════════════════════════════════════════════════
#  Pos / Neg sample collectors
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _collect_pos_samples_dplm(
    args, sampler, denoiser, reward_fn, num_seqs, alpha,
) -> PosSamples:
    """Collect positive samples for the CDM pos buffer.

    Returns a :class:`PosSamples` with ``(traj, rewards, weights_per_step,
    extra)``.

    Branches on ``args.cdm_pos_sample_method``:
      - ``is``   : base DPLM2 chain.  weights_per_step = terminal
                   ``softmax(r/alpha)`` repeated across all steps.
      - ``smc``  : plain SMC.  weights_per_step from SMC trajectory.
      - ``asmc`` : same SMC; loss uses weights_per_step[t_idx].
      - ``tsmc`` : twist-guided SMC.  weights_per_step from SMC trajectory.
    """
    from main import create_masked_input

    device = args.device
    method = args.cdm_pos_sample_method
    total_steps = args.steps

    assert method in ("is", "smc", "asmc", "tsmc"), (
        f"Unknown cdm_pos_sample_method '{method}'. "
        "Expected one of: is, smc, asmc, tsmc"
    )

    if method == "is":
        init_seq = create_masked_input(
            tokenizer=denoiser.dplm.tokenizer,
            seq_length=args.seq_length,
            num_seqs=num_seqs,
            device=device,
        )

        _, traj_states = _sample_with_traj(
            sampler, init_seq, total_steps, args.sampling_strategy,
            chunk_size=args.cdm_train_chunk_size,
        )
        x0 = traj_states[-1]

        reward_chunk = args.chunk_m_size
        if reward_chunk is not None and reward_chunk < x0.shape[0]:
            reward_parts = []
            for c0 in range(0, x0.shape[0], reward_chunk):
                c1 = min(c0 + reward_chunk, x0.shape[0])
                reward_parts.append(reward_fn(x0[c0:c1]).to(torch.float32))
            rewards = torch.cat(reward_parts, dim=0)
        else:
            rewards = reward_fn(x0).to(torch.float32)

        # Full-pool IS weights (sum=1 over N).  The buffer rescales by N/B
        # on each minibatch draw for unbiasedness.  IS loss only consumes
        # weights_per_step[-1], so share the terminal tensor across all
        # steps — cheap and keeps the list API uniform.
        terminal_weights = F.softmax((rewards / alpha).float(), dim=0)
        weights_per_step = [terminal_weights] * (total_steps + 1)

        return PosSamples(
            traj=traj_states,
            rewards=rewards,
            weights_per_step=weights_per_step,
            extra={"ess_pair": (float("nan"), float("nan"))},
        )

    # ── SMC-based branches ──
    smc_mode = args.cdm_pos_smc_mode
    assert smc_mode in ("single_smc", "multi_smc"), (
        f"cdm_pos_smc_mode='{smc_mode}' not supported. "
        "Expected one of: single_smc, multi_smc"
    )

    if method == "tsmc":
        assert denoiser.head is not None, (
            "cdm_pos_sample_method='tsmc' requires an attached twist head"
        )
        assert args.twist_score == "twist", (
            f"cdm_pos_sample_method='tsmc' requires twist_score='twist', "
            f"got '{args.twist_score}'"
        )

    traj_states, rewards_all, weights_per_step, extras = _run_smc_dplm(
        args, sampler, denoiser, reward_fn, num_seqs,
        score_type=args.twist_score,
        smc_mode=smc_mode,
        alpha=alpha,
        terminal_reward=(method == "tsmc"),
    )

    # smc/asmc don't apply the terminal-reward correction inside
    # _run_smc_dplm, so compute the true reward on the final state here
    # for logging / buffer bookkeeping.
    if rewards_all is None:
        x0 = traj_states[-1]
        reward_chunk = args.chunk_m_size
        if reward_chunk is not None and reward_chunk < x0.shape[0]:
            parts = []
            for c0 in range(0, x0.shape[0], reward_chunk):
                c1 = min(c0 + reward_chunk, x0.shape[0])
                parts.append(reward_fn(x0[c0:c1]).to(torch.float32))
            rewards_all = torch.cat(parts, dim=0)
        else:
            rewards_all = reward_fn(x0).to(torch.float32)

    return PosSamples(
        traj=traj_states,
        rewards=rewards_all,
        weights_per_step=weights_per_step,
        extra={"ess_pair": ess_summary(extras["ess_steps"])},
    )


@torch.no_grad()
def _collect_neg_samples_dplm(
    args, sampler, denoiser, reward_fn, num_seqs, alpha,
):
    """Collect negative samples for the CDM neg phase.

    The neg phase targets the twist-approximated distribution, so all
    IS weighting is done with ``psi^theta`` alone.  ``reward_fn`` is
    only used for logging on the final sample — it never enters the
    weights.

    Branches on ``args.cdm_neg_sample_method``:
      - ``is``  : base DPLM2 chain from fully-masked.  IS correction
                  happens at the loss site (``softmax(log psi^theta)``).
      - ``smc`` : twist-guided SMC over the full trajectory.
                  ``score_type='twist'`` throughout, no terminal-reward
                  correction (psi already absorbs alpha).  ``smc_mode``
                  follows ``args.cdm_pos_smc_mode``.

    Returns ``(traj, rewards, weights_per_step, ess_pair)``:
      - ``traj``            : per-step list of ``[B, L]`` states
                              (length ``steps + 1``).
      - ``rewards``         : true reward on x0 per particle — for
                              logging only, never enters the weights.
      - ``weights_per_step``: ``None`` for IS; list of per-step IS
                              weights aligned with ``traj`` for SMC.
      - ``ess_pair``        : ``(mean, min)`` per-step ESS for SMC;
                              ``(nan, nan)`` for IS.
    """
    from main import create_masked_input

    method = args.cdm_neg_sample_method
    total_steps = args.steps
    reward_chunk = args.chunk_m_size

    assert method in ("is", "smc"), (
        f"cdm_neg_sample_method='{method}' not supported. "
        "Expected one of: is, smc"
    )

    if method == "is":
        init_seq = create_masked_input(
            tokenizer=denoiser.dplm.tokenizer,
            seq_length=args.seq_length,
            num_seqs=num_seqs,
            device=args.device,
        )

        _, traj_states = _sample_with_traj(
            sampler, init_seq, total_steps, args.sampling_strategy,
            chunk_size=args.cdm_train_chunk_size,
        )

        x0 = traj_states[-1]
        if reward_chunk is not None and reward_chunk < x0.shape[0]:
            parts = []
            for c0 in range(0, x0.shape[0], reward_chunk):
                c1 = min(c0 + reward_chunk, x0.shape[0])
                parts.append(reward_fn(x0[c0:c1]).to(torch.float32))
            rewards = torch.cat(parts, dim=0)
        else:
            rewards = reward_fn(x0).to(torch.float32)

        return traj_states, rewards, None, (float("nan"), float("nan"))

    # ── SMC branch: twist-guided neg collection ──
    assert denoiser.head is not None, (
        "cdm_neg_sample_method='smc' requires an attached twist head"
    )

    traj_states, _, weights_per_step, extras = _run_smc_dplm(
        args, sampler, denoiser, None, num_seqs,
        score_type="twist",
        smc_mode=args.cdm_neg_smc_mode,
        alpha=alpha,
        terminal_reward=False,
    )

    # True reward on final state, for logging only.
    x0 = traj_states[-1]
    if reward_chunk is not None and reward_chunk < x0.shape[0]:
        parts = []
        for c0 in range(0, x0.shape[0], reward_chunk):
            c1 = min(c0 + reward_chunk, x0.shape[0])
            parts.append(reward_fn(x0[c0:c1]).to(torch.float32))
        rewards = torch.cat(parts, dim=0)
    else:
        rewards = reward_fn(x0).to(torch.float32)

    return traj_states, rewards, weights_per_step, ess_summary(extras["ess_steps"])


# ══════════════════════════════════════════════════════════════
#  Pos / Neg loss functions
# ══════════════════════════════════════════════════════════════


def compute_pos_loss_dplm(args, denoiser, sampler, pos, t_idx, alpha):
    """Positive-phase CDM loss term at diffusion step ``t_idx``.

    Branches on ``args.cdm_pos_sample_method``:
      - ``is`` / ``smc`` / ``tsmc`` : forward-noise ``traj[-1]`` to
                                       ``t_idx``, weight by terminal IS
                                       weights (``weights_per_step[-1]``).
      - ``asmc`` : consume ``traj[t_idx]`` directly, weight by the
                   per-step IS weights (``weights_per_step[t_idx]``).
                   With ESS-based resampling these are only uniform at
                   steps that actually resampled.
    """
    method = args.cdm_pos_sample_method
    total_steps = args.steps
    fwd_step = total_steps - t_idx

    if method in ("is", "smc", "tsmc"):
        x0 = pos.traj[-1].detach()
        x_t = _forward_noise_dplm(x0, fwd_step, total_steps, denoiser)
        log_psi_pos = denoiser(x_t, return_logits=False, return_value=True)  # [B]
        W_bar = pos.weights_per_step[-1].detach()                            # [B]
        ess = _ess_from_weights(W_bar)
        return (W_bar * log_psi_pos).sum(), ess

    elif method == "asmc":
        x_t = pos.traj[t_idx].detach()
        log_psi_pos = denoiser(x_t, return_logits=False, return_value=True)  # [B]
        W_bar = pos.weights_per_step[t_idx].detach()                         # [B]
        ess = _ess_from_weights(W_bar)
        return (W_bar * log_psi_pos).sum(), ess

    raise ValueError(f"Unknown cdm_pos_sample_method '{method}'")


def compute_neg_loss_dplm(
    args, denoiser, traj, t_idx, weights_per_step=None,
    head_weights=None,
):
    """Negative-phase CDM loss term at diffusion step ``t_idx``.

    ``traj`` is a per-step list of ``[B, L]`` states from
    ``_collect_neg_samples_dplm``.  The neg phase always consumes the
    intermediate ``traj[t_idx]`` (it never forward-noises x_0).

    Two heads can be applied on top of the (fixed) denoiser backbone:
      - ``denoiser.head``  computes ``log_psi_neg`` — the gradient-
                           carrying term (flows to the trainable head).
      - ``head_weights``   optional nn.Module.  When given, computes the
                           IS weights ``softmax(log psi)`` (detached).
                           Assumed to have the same architecture type
                           (linear/mlp vs transformer) as the live head,
                           so it consumes the same backbone output.  The
                           backbone runs once and both heads see the same
                           hidden state.  ``None`` reuses the live head
                           (previous behaviour).

    When ``weights_per_step`` is ``None`` (IS collection, q = p_ref),
    the p_ref -> pi^theta importance weight is psi^theta itself;
    self-normalise with ``softmax(log psi^theta(x_t))``.

    When ``weights_per_step`` is provided (SMC collection), pick the
    entry at ``t_idx`` — particles already carry their own IS weights
    and ``head_weights`` is unused.
    """
    x_t = traj[t_idx].detach()

    if (
        weights_per_step is None
        and head_weights is not None
        and head_weights is not denoiser.head
    ):
        # One shared backbone pass, two heads applied to the same hidden
        # state.  Backbone params are frozen, so autograd only needs to
        # track the live head for `log_psi_neg`.
        net_out = denoiser.dplm.forward(input_ids=x_t)
        h = net_out["last_hidden_state"].float()                         # [B, L, D]
        if denoiser.head_type != "transformer":
            h = h.mean(dim=1)                                            # [B, D]
        log_psi_neg = denoiser.head(h).squeeze(-1)                       # [B], grad
        with torch.no_grad():
            log_w = head_weights(h.detach()).squeeze(-1).detach()        # [B], no grad
        weights = F.softmax(log_w.float(), dim=0)
    else:
        log_psi_neg = denoiser(x_t, return_logits=False, return_value=True)  # [B]
        if weights_per_step is None:
            log_w = log_psi_neg.detach()
            weights = F.softmax(log_w.float(), dim=0)                    # [B]
        else:
            weights = weights_per_step[t_idx].detach()

    ess = _ess_from_weights(weights)
    return (weights * log_psi_neg).sum(), ess
