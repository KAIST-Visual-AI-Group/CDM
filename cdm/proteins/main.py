"""
Protein designability with DPLM-2 (joint amino-acid + structure tokens).

Every method is the same SMC loop (Eq. 5-9 of the paper):
    for t = T to 1:
        x_{t-1} ~ q(x_{t-1} | x_t)                 [propose, q = base model]
        log w  = psi_{t-1}(x_{t-1}) - psi_t(x_t)   [weight]
        a_k ~ Cat(w_1, ..., w_K)                   [resample, when ESS < ess_threshold * K]

What changes between the code paths is only how psi is obtained:
    smc : psi is the reward twist of Eq. (7), estimated with M x0-predictions per step.
    cdm : psi_theta is a trained twist head (CDM training is contrastive twist learning).
          With twist_ckpt=... the same loop runs with the trained head loaded.
    K=1 : no resampling happens at all, which is the unguided base model.

Usage:
  python -m cdm.proteins.main --config-name smc
  python -m cdm.proteins.main --config-name cdm
  python -m cdm.proteins.main --config-name cdm twist_ckpt=<path>.pt
"""

import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
sys.path.insert(0, _this_dir)
sys.path.insert(0, _project_root)

from cdm.utils import seed_everything
from samplers import DPLMSampler
from dplm_denoiser import DPLMDenoiser

from utils import save_results
from twist_model import (
    load_twist_checkpoint,
    make_twist_net,
    save_twist_checkpoint,
)


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

_LOG_FILE = None


def setup_logging(log_path):
    global _LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _LOG_FILE = open(log_path, "a")


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    if _LOG_FILE is not None:
        print(*args, **kwargs, file=_LOG_FILE, flush=True)


def init_wandb(args, method):
    """Initialize a wandb run, or return None when disabled / unavailable.

    Controlled by `args.disable_wandb` (default True). When enabled, reads
    optional `wandb_project`, `wandb_entity`, `wandb_name` from args.
    """
    if getattr(args, "disable_wandb", True):
        return None
    try:
        import wandb
    except ImportError:
        print_log("[!] wandb not installed; skipping wandb logging")
        return None

    run_name = getattr(args, "wandb_name", None)
    if not run_name:
        run_name = f"{method}_{os.path.basename(args.save_path)}"
    run = wandb.init(
        project=getattr(args, "wandb_project", None) or "discrete_smc_proteins",
        entity=getattr(args, "wandb_entity", None),
        name=run_name,
        config=OmegaConf.to_container(args, resolve=True),
        dir=args.save_path,
        reinit=True,
    )
    print_log(f"[*] wandb run: {run.name} ({run.url})")
    return run


def wandb_log(run, metrics, step=None):
    """Log a dict of metrics to wandb. No-op when run is None."""
    if run is None:
        return
    run.log(metrics, step=step)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


# ══════════════════════════════════════════════════════════════
#  Protein-specific tokenizer / input helpers
# ══════════════════════════════════════════════════════════════


def get_modality_type(input_ids, tokenizer):
    pad_id = tokenizer._token_to_id["<pad>"]
    L_total = input_ids.shape[1]
    # DPLM2 fixed layout: [struct_part, aa_part]
    # Each part is exactly half the total length (seq_length + 2)
    modality_type = torch.zeros_like(input_ids)
    modality_type[:, L_total // 2:] = 1  # Second half is AA
    # Mark padding
    input_mask = input_ids.ne(pad_id)
    modality_type[~input_mask] = 2
    return modality_type


def get_non_special_symbol_mask(output_tokens, tokenizer, partial_masks=None):
    pad_id = tokenizer._token_to_id["<pad>"]
    aa_bos_id = tokenizer._token_to_id["<cls_aa>"]
    aa_eos_id = tokenizer._token_to_id["<eos_aa>"]
    struct_bos_id = tokenizer._token_to_id["<cls_struct>"]
    struct_eos_id = tokenizer._token_to_id["<eos_struct>"]

    non_special_symbol_mask = (
        output_tokens.ne(pad_id)
        & output_tokens.ne(aa_bos_id)
        & output_tokens.ne(aa_eos_id)
        & output_tokens.ne(struct_bos_id)
        & output_tokens.ne(struct_eos_id)
    )
    if partial_masks is not None:
        non_special_symbol_mask &= ~partial_masks
    return non_special_symbol_mask


def get_special_tokens(tokenizer):
    return [
        tokenizer._token_to_id["<cls_aa>"],
        tokenizer._token_to_id["<eos_aa>"],
        tokenizer._token_to_id["<mask_aa>"],
        tokenizer._token_to_id["<cls_struct>"],
        tokenizer._token_to_id["<eos_struct>"],
        tokenizer._token_to_id["<mask_struct>"],
        tokenizer._token_to_id["<pad>"],
        tokenizer._token_to_id["<unk_struct>"],
        tokenizer._token_to_id["<unk_aa>"],
        tokenizer._token_to_id["X"],
        tokenizer._token_to_id["B"],
        tokenizer._token_to_id["U"],
        tokenizer._token_to_id["Z"],
        tokenizer._token_to_id["O"],
    ]


def create_init_seq(length, tokenizer):
    seq_struct = tokenizer.all_tokens[50] * length
    seq_aa = tokenizer.aa_mask_token * length
    seq_struct = (
        tokenizer.struct_cls_token
        + seq_struct
        + tokenizer.struct_eos_token
    )
    seq_aa = tokenizer.aa_cls_token + seq_aa + tokenizer.aa_eos_token
    return (seq_struct, seq_aa)


def create_masked_input(tokenizer, seq_length, num_seqs, device, partial_masks=None):
    """Create fully masked AA + struct sequences for DPLM2 co-generation."""
    input_data_struct_tokens, input_data_aatype = [], []
    for _ in range(num_seqs):
        seq_struct, seq_aa = create_init_seq(seq_length, tokenizer)
        input_data_struct_tokens.append(seq_struct)
        input_data_aatype.append(seq_aa)

    batch_struct = tokenizer.batch_encode_plus(
        input_data_struct_tokens,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    batch_aatype = tokenizer.batch_encode_plus(
        input_data_aatype,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    input_tokens = torch.concat(
        [batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1
    ).to(device)

    output_mask = get_non_special_symbol_mask(
        input_tokens, tokenizer, partial_masks=partial_masks
    )
    type_ids = get_modality_type(input_tokens, tokenizer)
    aa_position = type_ids.eq(1) & output_mask
    struct_position = type_ids.eq(0) & output_mask

    aa_mask_id = tokenizer._token_to_id["<mask_aa>"]
    struct_mask_id = tokenizer._token_to_id["<mask_struct>"]

    output_tokens = input_tokens.masked_fill(aa_position, aa_mask_id)
    output_tokens = output_tokens.masked_fill(struct_position, struct_mask_id)

    return output_tokens.to(device)


def load_reward_fn(reward_name, tokenizer, struct_tokenizer, device):
    if reward_name != "esmfold":
        raise ValueError(f"Unknown reward: {reward_name}")
    from reward import FoldReward
    return FoldReward(
        tokenizer=tokenizer, struct_tokenizer=struct_tokenizer, device=device,
    )


# ══════════════════════════════════════════════════════════════
#  Value estimation for SMC weights
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _score(denoiser, sampler, x, step_idx, args, reward_fn, score=None,
           chunk_size=None, logits=None, return_logits=False):
    """Estimate the value V_t(x) / alpha at state x for the SMC weights.

    score modes:
        twist   — denoiser(x, return_value=True)             [requires trained head]
        x0_pred — logsumexp(r/alpha, M samples) - log M

    Optional overrides:
        score      : override args.score (CDM training uses its own
                     `args.twist_score` knob, distinct from the
                     inference-time `args.score`).
        chunk_size : when not None, the M Monte-Carlo samples in the
                     x0_pred branch are processed in groups
                     of `chunk_size`, bounding the largest tensor flowing
                     through the denoiser to roughly [B * chunk_size, ...].
                     None = process all M at once.
        logits     : precomputed mask-corrected logits [B, L, V] from a
                     prior `denoiser(x)` call.  When supplied the x0_pred
                     branch skips its own backbone forward pass.
        return_logits : if True *and* score is ``x0_pred``, return
                     ``(value, logits)`` so the caller can cache the
                     logits for the next proposal step.
    """
    if score is None:
        score = args.score
    M = max(1, int(args.M))

    aa_mask_id = denoiser.aa_mask_id
    struct_mask_id = denoiser.struct_mask_id

    if score == "twist":
        assert denoiser.head is not None, (
            "score='twist' requires a twist head attached to the denoiser"
        )
        value_denoiser = denoiser

        chunk_s = getattr(args, "chunk_sample_size", None)
        if chunk_s is not None and chunk_s < x.shape[0]:
            v_chunks = []
            for i in range(0, x.shape[0], chunk_s):
                v_chunks.append(
                    value_denoiser(
                        x[i:i + chunk_s], return_logits=False, return_value=True,
                    )
                )
            return torch.cat(v_chunks, dim=0).to(torch.float32)
        else:
            twist_val = value_denoiser(x, return_logits=False, return_value=True)
            return twist_val.to(torch.float32)

    elif score == "x0_pred":
        B = x.shape[0]

        # Single shared backbone pass yields the logits for all M MC samples.
        if logits is None:
            logits = denoiser(x)                            # [B, L, V]
        p_x0 = F.softmax(logits.float(), dim=-1)
        mask_index = (x == aa_mask_id) | (x == struct_mask_id)

        m_chunk_size = chunk_size if chunk_size is not None else M

        reward_chunks = []
        for m0 in range(0, M, m_chunk_size):
            m_chunk = min(m_chunk_size, M - m0)
            p_x0_chunk = p_x0.repeat_interleave(m_chunk, dim=0)     # [B*m_chunk, L, V]
            x_chunk = x.repeat_interleave(m_chunk, dim=0)            # [B*m_chunk, L]
            mask_chunk = mask_index.repeat_interleave(m_chunk, dim=0)

            Bm, L, V = p_x0_chunk.shape
            x0_hat = torch.multinomial(
                p_x0_chunk.reshape(Bm * L, V), 1
            ).reshape(Bm, L)
            x0_hat = torch.where(mask_chunk, x0_hat, x_chunk)

            rewards_chunk = reward_fn(x0_hat).to(torch.float32)      # [B*m_chunk]
            reward_chunks.append(rewards_chunk.reshape(B, m_chunk))

        rewards = torch.cat(reward_chunks, dim=1)                    # [B, M]
        value = torch.logsumexp(rewards / args.kl_weight, dim=-1) - math.log(M)
        if return_logits:
            return value, logits
        return value

    raise ValueError(f"Unsupported score mode: {score}")


# ══════════════════════════════════════════════════════════════
#  Sampling
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def sample(denoiser, sampler, batch_size, args, reward_fn):
    """Unified sampling loop for all methods.

    All methods share the same per-step proposal loop.  Scoring and
    resampling are toggled by the config:

        K=1  : no scoring, no resampling (the unguided base model).
        smc  : K>1, scoring via the reward twist (x0_pred), resampling.
        cdm  : K>1, scoring via the trained twist head, resampling.

    Returns (x, ess_per_step) where x is [batch_size, L] and
    ess_per_step is a list of floats (empty when scoring is off).
    """
    method = args.method
    device = denoiser.device
    seq_length = args.seq_length
    steps = args.steps
    K = args.K if args.K is not None else 1
    total = batch_size * K

    # ── Initialize: fully masked sequences ──
    x = create_masked_input(
        tokenizer=denoiser.dplm.tokenizer,
        seq_length=seq_length,
        num_seqs=total,
        device=device,
    )

    ess_per_step = []
    fuse_twist = args.score == "twist"
    cache_logits = args.score in ("twist", "x0_pred") and K > 1
    do_score = args.score is not None and K > 1

    logits_cache = None
    v_cache = None
    # Accumulated log-weights for adaptive resampling.  Each batch row
    # accumulates incremental log-weights until that row resamples, at
    # which point only its accumulator resets to zero.
    log_w_accum = torch.zeros(batch_size, K, device=device)
    chunk_s = getattr(args, "chunk_sample_size", None)

    # ── Per-step proposal (+ optional scoring / resampling) ──
    for i in tqdm(range(steps), desc=f"Sampling ({method})"):
        # ── Propose ──
        if chunk_s is not None and chunk_s < total:
            x_next_chunks = []
            for c0 in range(0, total, chunk_s):
                c1 = min(c0 + chunk_s, total)
                lc = logits_cache[c0:c1] if logits_cache is not None else None
                x_next_chunks.append(
                    sampler._propose_step(
                        x[c0:c1], i, steps, args.sampling_strategy, logits=lc,
                    )
                )
            x_next = torch.cat(x_next_chunks, dim=0)
        else:
            x_next = sampler._propose_step(
                x, i, steps, args.sampling_strategy, logits=logits_cache,
            )

        # ── Score ──
        if do_score:
            if fuse_twist:
                if chunk_s is not None and chunk_s < total:
                    # Twist: single fused backbone+head pass for logits + value.
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
                # x0_pred: _score already calls denoiser(x_next) for logits
                # internally — return them so we can reuse them next step.
                v_next, logits_next = _score(
                    denoiser, sampler, x_next, i + 1, args, reward_fn,
                    chunk_size=args.chunk_m_size, return_logits=True,
                )
            else:
                logits_next = None
                v_next = _score(
                    denoiser, sampler, x_next, i + 1, args, reward_fn,
                    chunk_size=args.chunk_m_size,
                )

            if v_cache is None:
                # First step: all K particles are identical (fully masked),
                # so v_curr is the same constant for every particle.  Any
                # constant cancels out in w / w.sum(), so skip the forward pass.
                v_curr = torch.ones(total, device=device)
            else:
                v_curr = v_cache

            # Incremental log-weight for this step.
            log_w_inc = (v_next - v_curr).reshape(batch_size, K)

            # Accumulate across non-resampling steps.
            log_w_accum = log_w_accum + log_w_inc

            log_w = log_w_accum - log_w_accum.max(dim=1, keepdim=True)[0]
            w = torch.exp(log_w)
            w = w / w.sum(dim=1, keepdim=True)

            # ESS = 1 / sum(w_i^2)
            ess_per_batch = 1.0 / (w ** 2).sum(dim=1)
            ess = ess_per_batch.mean().item()
            ess_per_step.append(ess)

            # ── Independently resample batches below the ESS threshold ──
            should_resample = ess_per_batch < args.ess_threshold * K

            if should_resample.any():
                # construct resampling mask
                indices = torch.arange(K, device=device)[None, :].expand(
                    batch_size, -1,
                ).clone()
                indices[should_resample] = torch.multinomial(
                    w[should_resample], K, replacement=True,
                )
                batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, K)

                x_next = x_next.reshape(batch_size, K, -1)
                x_next = x_next[batch_idx, indices].reshape(total, -1)

                v_cache = v_next.reshape(batch_size, K)
                v_cache = v_cache[batch_idx, indices].reshape(total)

                if cache_logits:
                    _, L_dim, V_dim = logits_next.shape
                    logits_next = logits_next.reshape(batch_size, K, L_dim, V_dim)
                    logits_cache = logits_next[batch_idx, indices].reshape(
                        total, L_dim, V_dim,
                    )

                # Reset log weights of resampled batches
                log_w_accum[should_resample] = 0.0
            else:
                v_cache = v_next
                if cache_logits:
                    logits_cache = logits_next

        x = x_next

    # ── Select best particle per batch element ──
    if K <= 1:
        return x, ess_per_step

    # Return the highest-reward particle of each group.
    chunk_k = args.chunk_b_size if args.chunk_b_size is not None else K
    x_bk = x.reshape(batch_size, K, -1)
    score_chunks = []
    for k0 in range(0, K, chunk_k):
        k1 = min(k0 + chunk_k, K)
        xk = x_bk[:, k0:k1].reshape(batch_size * (k1 - k0), -1)
        scores_xk = reward_fn(xk).detach()
        score_chunks.append(scores_xk.reshape(batch_size, k1 - k0))
    scores_2d = torch.cat(score_chunks, dim=1)            # [batch_size, K]
    best = scores_2d.argmax(dim=1)
    x = x.reshape(batch_size, K, -1)
    x = x[torch.arange(batch_size, device=device), best]
    return x, ess_per_step


# ══════════════════════════════════════════════════════════════
#  LR schedule
# ══════════════════════════════════════════════════════════════


def _make_linear_decay_scheduler(optimizer, total_steps, decay_start_frac):
    """LambdaLR that holds LR constant then linearly decays to 0,
    beginning at `decay_start_frac` of total training steps.
    """
    decay_start = int(total_steps * decay_start_frac)

    def _lr_lambda(step):
        if step <= decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ══════════════════════════════════════════════════════════════
#  Training: CDM
# ══════════════════════════════════════════════════════════════


def train_cdm(args, denoiser, sampler, reward_fn):
    """Train a twist head with the contrastive twist learning (CDM) objective.

    The CDM gradient (Eq. 27) decomposes into positive and negative phases:
        -grad_theta L_CDM(theta) = sum_t ( E_{p*(x_t)}[grad log psi_t^theta(x_t)]
                                          - E_{pi_t^theta(x_t)}[grad log psi_t^theta(x_t)] )

    Mirrors ``cdm.texts.main.train_cdm``, adapted for DPLM2:
        - Unconditional generation (no prompts).
        - Dual-modality (struct + AA) forward noising.
        - Sample collectors and loss functions live in ``cdm_utils``.
    """
    from cdm_utils import (
        CDMPosBuffer,
        EMA,
        _collect_pos_samples_dplm,
        _collect_neg_samples_dplm,
        compute_pos_loss_dplm,
        compute_neg_loss_dplm,
        make_linear_decay_scheduler,
    )

    device = args.device

    # ── Build optimizer over the twist head (already attached by caller) ──
    assert denoiser.head is not None, (
        "train_cdm expects a twist head already attached via `make_twist_net`"
    )
    head = denoiser.head
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, head.parameters()),
        lr=args.twist_lr,
        weight_decay=args.twist_weight_decay,
    )

    # ── EMA ──
    # `cdm_ema_decay` may be a single float or a list of floats.  When a
    # list is given, one EMA is tracked per decay and each is checkpointed
    # separately (e.g. `twist_best_ema0.999.pt`, `twist_best_ema0.99.pt`).
    # The first decay in the list is used for EMA-based sampling.
    raw_decay = args.cdm_ema_decay
    decays = list(raw_decay) if isinstance(raw_decay, (list, ListConfig)) else [raw_decay]
    emas = (
        [EMA(head, decay=d) for d in decays]
        if (args.cdm_save_ema or args.cdm_ema_for_sampling) and decays
        else []
    )

    def _ema_suffix(decay):
        """Filename suffix for one EMA decay, e.g. 0.999 -> '_ema0.999'."""
        return f"_ema{decay:g}" if len(emas) > 1 else "_ema"

    def _save_all_emas(filename_stem, epoch, loss):
        """Swap each EMA's shadow into denoiser.head in turn and checkpoint it."""
        if not emas or not args.cdm_save_ema:
            return
        for ema_obj, d in zip(emas, decays):
            denoiser.head = ema_obj.shadow
            save_twist_checkpoint(
                denoiser=denoiser, args=args,
                epoch=epoch, loss=loss,
                filename=f"{filename_stem}{_ema_suffix(d)}.pt",
                vocab_size=vocab_size,
            )
        denoiser.head = head

    def _swap_to_ema(force_ema=False):
        """Temporarily install EMA shadow as the live head for sampling."""
        if emas and (args.cdm_ema_for_sampling or force_ema):
            denoiser.head = emas[0].shadow

    def _swap_to_live():
        """Restore the trainable head after sampling."""
        denoiser.head = head

    # ── LR scheduler ──
    total_train_steps = args.twist_epochs * args.twist_steps_per_epoch
    scheduler, warmup_steps = make_linear_decay_scheduler(
        optimizer, total_train_steps, args.twist_lr_decay_start_frac,
    )

    # ── Logging ──
    vocab_size = len(denoiser.dplm.tokenizer)
    total_params = sum(p.numel() for p in head.parameters()) / 1e6
    trainable_params_num = sum(
        p.numel() for p in head.parameters() if p.requires_grad
    ) / 1e6
    print_log(f"[*] CDM total head params:     {total_params:.2f}M")
    print_log(f"[*] CDM trainable head params: {trainable_params_num:.2f}M")
    print_log(
        f"[*] Training CDM | lr={args.twist_lr} | "
        f"LR decay from step {warmup_steps}/{total_train_steps} | "
        f"alpha={args.kl_weight:.2f} | "
        f"EMA={'on' if emas else 'off'}"
        f"{f' (decays={decays}, for_sampling={args.cdm_ema_for_sampling})' if emas else ''}"
    )

    wandb_run = init_wandb(args, "cdm")
    best_loss = float("inf")
    best_head_sd = None
    global_step = 0
    checkpoint_log = []
    pos_buffer = CDMPosBuffer(args)
    avg_loss = 0.0

    # ── Active head for the neg-loss IS weights ──
    # Maintained as an EMA of the live head across epochs:
    #   new = decay * old + (1 - decay) * live
    # decay = 0  → fresh snapshot at each epoch (old behaviour).
    # decay = 1  → frozen at the initial head for the whole run.
    active_head_ema = float(getattr(args, "cdm_active_head_ema", 0.0))
    epoch_start_head = copy.deepcopy(head).eval()
    for _p in epoch_start_head.parameters():
        _p.requires_grad = False

    pbar = tqdm(
        range(args.twist_epochs), desc="CDM", leave=True, dynamic_ncols=True,
    )
    for epoch in pbar:
        alpha = args.kl_weight

        # ── Refill positive buffer once per epoch ──
        # Use EMA shadow for sampling if cdm_ema_for_sampling is set.
        head.eval()
        _swap_to_ema()
        pos_buffer.clear()
        pos_buffer.fill(
            sampler, denoiser, reward_fn,
            num_seqs=args.cdm_buffer_size,
            collect_fn=_collect_pos_samples_dplm,
            alpha=alpha,
        )
        _swap_to_live()
        head.train()

        pos_smc_ess_mean = pos_buffer.smc_ess_mean
        pos_smc_ess_min = pos_buffer.smc_ess_min
        pos_reward_mean = float(pos_buffer.data.rewards.mean().item())

        # Refresh `epoch_start_head` as an EMA of the live head so the
        # neg-branch IS weights don't shift mid-epoch, while still
        # tracking training progress across epochs.  At epoch 0 this is
        # a no-op (active head == live head from the pre-loop init).
        with torch.no_grad():
            for p_snap, p_live in zip(
                epoch_start_head.parameters(), head.parameters(),
            ):
                p_snap.data.mul_(active_head_ema).add_(
                    p_live.data, alpha=1.0 - active_head_ema,
                )

        losses = []
        neg_rewards = []
        pos_final_ess_list = []
        neg_final_ess_list = []
        neg_smc_ess_mean_list = []
        neg_smc_ess_min_list = []

        for _ in range(args.twist_steps_per_epoch):
            # ── 1. Sample pos minibatch + collect neg samples ──
            head.eval()
            _swap_to_ema()
            with torch.no_grad():
                pos_batch = pos_buffer.sample(args.twist_batch_size, device)
                traj_neg, rewards_neg, weights_neg_per_step, neg_smc_ess = _collect_neg_samples_dplm(
                    args, sampler, denoiser, reward_fn,
                    num_seqs=args.twist_batch_size,
                    alpha=alpha,
                )
                if args.cdm_neg_sample_method == "smc":
                    neg_smc_ess_mean_list.append(neg_smc_ess[0])
                    neg_smc_ess_min_list.append(neg_smc_ess[1])
            _swap_to_live()
            head.train()

            # ── 2. Loss computation over diffusion steps ──
            optimizer.zero_grad()
            total_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0

            num_t_steps = args.steps + 1
            t_schedule = getattr(args, "twist_t_schedule", "random")
            # When either `twist_t_sampling=random` or a non-random
            # schedule is specified, we pick exactly one t_idx per
            # training step (instead of iterating over all of them).
            single_t_mode = args.twist_t_sampling == "random" or t_schedule != "random"

            for t_idx in range(num_t_steps):
                if single_t_mode:
                    if t_schedule == "random":
                        t_idx = int(
                            torch.randint(1, num_t_steps, (1,), device=device).item()
                        )
                    elif t_schedule in ("decreasing", "increasing"):
                        n_sched = int(args.twist_t_schedule_steps)
                        frac = min(1.0, (global_step % n_sched) / max(1, n_sched - 1))
                        if t_schedule == "decreasing":
                            # noisiest (num_t_steps-1) -> cleanest (0)
                            t_idx = int(round((1.0 - frac) * (num_t_steps - 1)))
                        else:
                            # cleanest (0) -> noisiest (num_t_steps-1)
                            t_idx = int(round(frac * (num_t_steps - 1)))
                        t_idx = max(1, min(num_t_steps - 1, t_idx))
                    else:
                        raise ValueError(
                            f"Unknown twist_t_schedule '{t_schedule}'. "
                            "Expected one of: random, decreasing, increasing."
                        )

                pos_term, pos_final_ess = compute_pos_loss_dplm(
                    args, denoiser, sampler, pos_batch, t_idx, alpha,
                )
                # Neg loss: `log_psi_neg` goes through the live head
                # (so gradient reaches the trainable params), while the
                # IS weights are computed from the frozen epoch-start
                # head.  Both share a single backbone forward.
                neg_term, neg_final_ess = compute_neg_loss_dplm(
                    args, denoiser, traj_neg, t_idx,
                    weights_per_step=weights_neg_per_step,
                    head_weights=epoch_start_head,
                )
                pos_final_ess_list.append(pos_final_ess)
                neg_final_ess_list.append(neg_final_ess)

                # Surrogate: grad_theta(neg - pos) = grad_theta L_CDM
                divide_factor = 1.0 if single_t_mode else num_t_steps
                step_loss = (neg_term - pos_term) / divide_factor
                step_loss.backward()
                total_loss += step_loss.item()
                total_pos_loss += pos_term.item() / divide_factor
                total_neg_loss += neg_term.item() / divide_factor

                if single_t_mode:
                    break
                
                

            # ── 3. Gradient clipping + optimizer step + EMA update ──
            trainable_params_list = [p for p in head.parameters() if p.requires_grad]
            if args.twist_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params_list, args.twist_clip_grad_norm,
                ).item()
            else:
                grads = [p.grad for p in trainable_params_list if p.grad is not None]
                grad_norm = (
                    torch.sqrt(sum((g ** 2).sum() for g in grads)).item()
                    if grads else 0.0
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if emas:
                for ema_obj, d in zip(emas, decays):
                    if args.cdm_ema_schedule == "uniform":
                        decay_rate = d
                    elif args.cdm_ema_schedule == "nft":
                        ramp_end = int(args.twist_epochs * args.cdm_ema_nft_start_frac)
                        decay_rate = min(
                            (epoch + 1) / max(ramp_end, 1) * d, d,
                        )
                    else:
                        decay_rate = d
                    ema_obj.update(head, decay=decay_rate)

            losses.append(total_loss)
            neg_rewards.append(rewards_neg.mean().item())
            global_step += 1

            train_log = {
                "train/loss": float(total_loss),
                "train/pos_loss": float(total_pos_loss),
                "train/neg_loss": float(total_neg_loss),
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/neg_reward": float(rewards_neg.mean().item()),
                "train/alpha": float(alpha),
                "train/epoch": epoch,
            }
            if single_t_mode:
                train_log["train/t_idx"] = int(t_idx)
                train_log["train/t_idx_frac"] = t_idx / max(1, num_t_steps - 1)
            wandb_log(wandb_run, train_log, step=global_step)

            # ── Step-based light reward eval ──
            eval_every_steps = getattr(args, "cdm_eval_every_steps", None) or 0
            if eval_every_steps > 0 and global_step % eval_every_steps == 0:
                head.eval()
                _swap_to_ema(force_ema=True)
                eval_batches = getattr(args, "cdm_eval_batches", 2)
                eval_batch_size = getattr(args, "cdm_eval_batch_size", 4)
                eval_rewards = []
                for _ in range(eval_batches):
                    x_eval, _ = sample(
                        denoiser, sampler, eval_batch_size, args, reward_fn,
                    )
                    eval_rewards.extend(reward_fn(x_eval).cpu().float().tolist())
                _swap_to_live()
                head.train()
                eval_reward_mean = float(np.mean(eval_rewards))
                wandb_log(wandb_run, {
                    "eval/reward_mean": eval_reward_mean,
                    "eval/n": len(eval_rewards),
                }, step=global_step)
                print_log(
                    f"  [eval-step] step={global_step} | n={len(eval_rewards)} | "
                    f"reward_mean={eval_reward_mean:.4f}"
                )

            # ── Step-based intermediate save ──
            # Save the live head + each EMA shadow under a step-based
            # filename so these intermediate checkpoints don't collide
            # with the epoch-based ones.
            save_every_steps = getattr(args, "cdm_save_every_steps", None) or 0
            if save_every_steps > 0 and global_step % save_every_steps == 0:
                save_twist_checkpoint(
                    denoiser=denoiser, args=args,
                    epoch=epoch, loss=float(total_loss),
                    filename=f"twist_step_{global_step}.pt",
                    vocab_size=vocab_size,
                )
                _save_all_emas(
                    f"twist_step_{global_step}", epoch, float(total_loss),
                )

        avg_loss = float(np.mean(losses))
        avg_neg_reward = float(np.mean(neg_rewards))
        pos_final_ess_mean = float(np.mean(pos_final_ess_list)) if pos_final_ess_list else float("nan")
        neg_final_ess_mean = float(np.mean(neg_final_ess_list)) if neg_final_ess_list else float("nan")

        epoch_metrics = {
            "epoch/avg_loss": avg_loss,
            "epoch/best_loss": best_loss,
            "epoch/avg_neg_reward": avg_neg_reward,
            "epoch/pos_reward_mean": float(pos_reward_mean),
            "epoch/alpha": float(alpha),
            "epoch/pos_final_ess": pos_final_ess_mean,
            "epoch/neg_final_ess": neg_final_ess_mean,
            "epoch/index": epoch + 1,
        }
        if args.cdm_pos_sample_method in ("smc", "asmc", "tsmc"):
            epoch_metrics["epoch/pos_smc_ess_mean"] = float(pos_smc_ess_mean)
            epoch_metrics["epoch/pos_smc_ess_min"] = float(pos_smc_ess_min)
        if args.cdm_neg_sample_method == "smc" and neg_smc_ess_mean_list:
            epoch_metrics["epoch/neg_smc_ess_mean"] = float(np.mean(neg_smc_ess_mean_list))
            epoch_metrics["epoch/neg_smc_ess_min"] = float(np.mean(neg_smc_ess_min_list))
        wandb_log(wandb_run, epoch_metrics, step=global_step)

        if (epoch + 1) % max(1, int(args.twist_log_every)) == 0:
            lr_str = (
                f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else f"{args.twist_lr:.2e}"
            )
            pbar.set_postfix(loss=f"{avg_loss:.4f}", neg_r=f"{avg_neg_reward:.4f}")
            log_parts = [
                f"  Epoch {epoch+1}/{args.twist_epochs}",
                f"Loss: {avg_loss:.6f}",
                f"Pos Reward: {pos_reward_mean:.6f}",
                f"Neg Reward: {avg_neg_reward:.6f}",
                f"alpha: {alpha:.3f}",
                f"lr: {lr_str}",
            ]
            if args.cdm_pos_sample_method in ("smc", "asmc", "tsmc"):
                log_parts.append(
                    f"pos_smc_ess: {pos_smc_ess_mean:.3f}/{pos_smc_ess_min:.3f}"
                )
            if args.cdm_neg_sample_method == "smc" and neg_smc_ess_mean_list:
                neg_smc_ess_mean = float(np.mean(neg_smc_ess_mean_list))
                neg_smc_ess_min = float(np.mean(neg_smc_ess_min_list))
                log_parts.append(
                    f"neg_smc_ess: {neg_smc_ess_mean:.3f}/{neg_smc_ess_min:.3f}"
                )
            log_parts.append(f"pos_final_ess: {pos_final_ess_mean:.3f}")
            log_parts.append(f"neg_final_ess: {neg_final_ess_mean:.3f}")
            print_log(" | ".join(log_parts))
            checkpoint_log.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "best_loss": best_loss,
            })
            with open(
                os.path.join(args.save_path, "checkpoint_log.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(checkpoint_log, f, indent=2)

        if args.twist_epochs_intermediate_eval and \
           (epoch + 1) % args.twist_epochs_intermediate_eval == 0:
            save_twist_checkpoint(
                denoiser=denoiser, args=args,
                epoch=epoch, loss=float(avg_loss),
                filename=f"twist_epoch_{epoch+1}.pt",
                vocab_size=vocab_size,
            )
            _save_all_emas(f"twist_epoch_{epoch+1}", epoch, float(avg_loss))

        # ── Light reward-based eval of the current head ──
        eval_every = getattr(args, "cdm_eval_every", None) or 0
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            head.eval()
            _swap_to_ema(force_ema=True)
            eval_batches = getattr(args, "cdm_eval_batches", 2)
            eval_batch_size = getattr(args, "cdm_eval_batch_size", 4)
            eval_rewards = []
            for _ in range(eval_batches):
                x_eval, _ = sample(
                    denoiser, sampler, eval_batch_size, args, reward_fn,
                )
                eval_rewards.extend(reward_fn(x_eval).cpu().float().tolist())
            _swap_to_live()
            head.train()
            eval_reward_mean = float(np.mean(eval_rewards))
            wandb_log(wandb_run, {
                "eval/reward_mean": eval_reward_mean,
                "eval/n": len(eval_rewards),
            }, step=global_step)
            print_log(
                f"  [eval] Epoch {epoch+1} | n={len(eval_rewards)} | "
                f"reward_mean={eval_reward_mean:.4f}"
            )
            if checkpoint_log and checkpoint_log[-1]["epoch"] == epoch + 1:
                checkpoint_log[-1]["eval_reward_mean"] = eval_reward_mean
                with open(
                    os.path.join(args.save_path, "checkpoint_log.json"),
                    "w", encoding="utf-8",
                ) as f:
                    json.dump(checkpoint_log, f, indent=2)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_twist_checkpoint(
                denoiser=denoiser, args=args,
                epoch=epoch, loss=float(avg_loss),
                filename="twist_best.pt",
                vocab_size=vocab_size,
            )
            _save_all_emas("twist_best", epoch, float(avg_loss))
            # Snapshot the head that was used for sampling (first EMA or live).
            head_to_save = emas[0].shadow if emas else head
            best_head_sd = {
                k: v.detach().cpu().clone()
                for k, v in head_to_save.state_dict().items()
            }

    # ── Final save ──
    save_twist_checkpoint(
        denoiser=denoiser, args=args,
        epoch=args.twist_epochs - 1,
        loss=float(avg_loss),
        filename="twist_final.pt",
        vocab_size=vocab_size,
    )
    _save_all_emas("twist_final", args.twist_epochs - 1, float(avg_loss))

    # Restore best weights into the live head for downstream SMC inference.
    if best_head_sd is not None:
        head.load_state_dict(best_head_sd)
    head.eval()
    denoiser.head = head
    print_log(f"[*] Best CDM loss: {best_loss:.6f}")
    if wandb_run is not None:
        wandb_run.finish()
    return denoiser


# ══════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════


def evaluate(args, denoiser, sampler, reward_fn, tag):
    """Generate proteins and report the given (-scRMSD) and heldout (scTM) rewards."""
    all_sequences, all_rewards, all_heldout = [], [], []
    all_sample_times, all_ess_traces = [], []
    generated = 0

    with tqdm(total=args.num_sample_batches, desc="Evaluating", leave=False) as pbar:
        while generated < args.num_sample_batches:
            bs = min(args.batch_size, args.num_sample_batches - generated)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            x, ess_trace = sample(denoiser, sampler, bs, args, reward_fn)
            if ess_trace:
                all_ess_traces.append(ess_trace)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            all_sample_times.append((elapsed, bs))

            save_results(
                outputs={"output_tokens": x},
                save_dir=f"{args.save_path}/outputs",
                task="co_generation",
                tokenizer=denoiser.dplm.tokenizer,
                struct_tokenizer=denoiser.dplm.struct_tokenizer,
                headers=None,
                save_pdb=True,
                continue_write=False,
            )

            rewards, sctm = reward_fn.crmsd_and_sctm(x)

            decoded = denoiser.dplm.tokenizer.batch_decode(x, skip_special_tokens=True)
            all_sequences.extend("".join(seq.split(" ")) for seq in decoded)
            all_rewards.extend(rewards.cpu().float().tolist())
            all_heldout.extend(sctm.cpu().float().tolist())

            generated += bs
            pbar.update(bs)

    with open(os.path.join(args.save_path, "generated.fasta"), "w") as f:
        for idx, (seq, rew, tm) in enumerate(zip(all_sequences, all_rewards, all_heldout)):
            f.write(f">SEQ_{idx}_L={args.seq_length}_reward={rew:.4f}_scTM={tm:.4f}\n{seq}\n")

    if all_ess_traces:
        ess_arr = np.array(all_ess_traces)
        with open(os.path.join(args.save_path, "ess.json"), "w") as f:
            json.dump({"K": int(args.K), "mean_per_step": ess_arr.mean(axis=0).tolist(),
                       "std_per_step": ess_arr.std(axis=0).tolist()}, f, indent=2)

    total_time = sum(t for t, _ in all_sample_times)
    total_samples = sum(b for _, b in all_sample_times)

    result = dict(app="protein", method=args.method, K=args.K, M=args.M, seed=args.seed,
                  given_reward=round(float(np.mean(all_rewards)), 4),
                  heldout_reward=round(float(np.nanmean(all_heldout)), 4),
                  sec_per_sample=round(total_time / max(total_samples, 1), 4),
                  n_samples=len(all_rewards), twist_ckpt=args.twist_ckpt)
    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print_log("[RESULT] " + " ".join(f"{k}={v}" for k, v in result.items()))
    return result


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


@hydra.main(config_path="configs", config_name="smc", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)

    train_cdm_twist = cfg.method == "cdm" and cfg.twist_ckpt is None
    if train_cdm_twist:
        tag = f"cdm_train"
    elif cfg.method == "cdm":
        # the twist stem keeps two runs that differ only in twist_ckpt from colliding
        tag = f"cdm_K{cfg.K}_{os.path.splitext(os.path.basename(cfg.twist_ckpt))[0]}"
    else:
        tag = f"{cfg.method}_K{cfg.K}_M{cfg.M}"
    cfg.tag = tag
    cfg.save_path = os.path.join(cfg.save_path, tag)
    os.makedirs(cfg.save_path, exist_ok=True)
    setup_logging(os.path.join(cfg.save_path, "run.log"))

    print_log(f"[*] Method: {cfg.method} | K: {cfg.K} | Score: {cfg.score}")
    denoiser = DPLMDenoiser(device=cfg.device)
    tokenizer = denoiser.dplm.tokenizer
    struct_tokenizer = denoiser.dplm.struct_tokenizer
    freeze_model(denoiser.dplm)

    sampler = DPLMSampler(denoiser=denoiser, steps=cfg.steps, temperature=cfg.temperature)

    print_log(f"[*] Loading reward: {cfg.reward_name} (alpha={cfg.kl_weight})")
    reward_fn = load_reward_fn(cfg.reward_name, tokenizer, struct_tokenizer, cfg.device)

    with open(os.path.join(cfg.save_path, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # The twist head shares the backbone with the LM forward path, so one backbone pass
    # yields both the proposal logits and the scalar twist value.
    if cfg.method == "cdm":
        if cfg.twist_ckpt is not None:
            print_log(f"[*] Loading twist from '{cfg.twist_ckpt}'")
            load_twist_checkpoint(cfg.twist_ckpt, device=cfg.device, denoiser=denoiser)
        else:
            print_log("[*] No twist_ckpt provided — attaching a fresh head and training it")
            make_twist_net(cfg, denoiser)
            train_start = time.time()
            train_cdm(cfg, denoiser, sampler, reward_fn)
            print_log(f"CDM training time: {time.time() - train_start:.1f}s")
        denoiser.head.eval()

    evaluate(cfg, denoiser, sampler, reward_fn, tag)


if __name__ == "__main__":
    main()
