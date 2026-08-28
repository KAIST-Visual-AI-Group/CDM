"""
Diffusion LLM alignment with LLaDA-8B-Instruct.

Every method is the same SMC loop (Eq. 5-9 of the paper):
    for t = T to 1:
        x_{t-1} ~ q(x_{t-1} | x_t)                 [propose, q = base model]
        log w  = psi_{t-1}(x_{t-1}) - psi_t(x_t)   [weight]
        a_k ~ Cat(w_1, ..., w_K)                   [resample, when ESS < threshold]

What changes between the code paths is only how psi is obtained:
    smc : psi is the reward twist of Eq. (7), estimated with M x0-predictions per step.
    cdm : psi_theta is a trained twist head (CDM training is contrastive twist learning).
          With twist_ckpt=... the same loop runs with the trained head loaded.
    K=1 : no resampling happens at all, which is the unguided base model.

Prompts come from RewardBench (80/20 train/validation split). Given reward: Skywork-Reward
Llama-3.1-8B. Heldout reward: ArmoRM-Llama3-8B.

Usage (4 GPUs):
  torchrun --nnodes=1 --nproc_per_node=4 -m cdm.texts.main --config-name smc distributed.enabled=true
  torchrun --nnodes=1 --nproc_per_node=4 -m cdm.texts.main --config-name cdm distributed.enabled=true
  torchrun --nnodes=1 --nproc_per_node=4 -m cdm.texts.main --config-name cdm distributed.enabled=true twist_ckpt=<path>.pt
"""

import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
sys.path.insert(0, _this_dir)
sys.path.insert(0, _project_root)
sys.modules.setdefault("cdm.texts.main", sys.modules[__name__])

from cdm.utils import seed_everything
from cdm.texts.eval import QwenPPL
from cdm.texts.llada_denoiser import LLaDADenoiser
from cdm.texts.rewards import (
    PROMPT_CONDITIONED_REWARDS as _REWARDS_PROMPT_CONDITIONED,
    evaluate_generation,
)
from cdm.texts.samplers import DiffusionSampler, add_gumbel_noise
from cdm.texts.text_dataset.rewardbench_dataloader import (
    create_rewardbench_prompt_dataloader,
)
from cdm.texts.twist_model import (
    load_twist_checkpoint,
    make_twist_net,
    save_twist_checkpoint,
    unwrap_head,
)

from cdm.texts.utils import get_input_prompt, make_linear_decay_scheduler
from cdm.texts.dist_utils import (
    init_distributed_mode,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    cleanup,
    all_reduce_mean,
    all_gather_list,
)
from cdm.texts.cdm_utils import (
    prepare_cdm_config_and_tag,
    _collect_pos_samples,
    _collect_neg_samples,
    compute_pos_loss,
    compute_neg_loss
)
from cdm.texts.cdm_buffer import CDMPosBuffer
# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

_LOG_FILE = None
PROMPT_CONDITIONED_REWARDS = _REWARDS_PROMPT_CONDITIONED
INTERMEDIATE_EVAL_CATEGORIES = ("chat", "chat-hard", "reasoning", "safety")


def setup_logging(log_path):
    global _LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _LOG_FILE = open(log_path, "a", encoding="utf-8")


def close_logging():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    if _LOG_FILE is not None:
        print(*args, **kwargs, file=_LOG_FILE, flush=True)


def _wandb_log(data, step=None):
    if wandb is None or wandb.run is None:
        return
    wandb.log(data, step=step)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def _build_train_prompt_loader(args, batch_size, distributed=False, seed=None):
    """Build a prompt DataLoader for training.

    When distributed=True, uses DistributedSampler to ensure each rank sees
    different prompts every epoch.  ``batch_size`` is the **global** batch
    size; it is divided by world_size so each rank processes its share.
    Caller must call ``loader.sampler.set_epoch(epoch)`` each epoch for
    proper shuffling.
    """
    from torch.utils.data import DataLoader, DistributedSampler

    seed = args.seed if seed is None else seed
    if distributed:
        batch_size = max(1, batch_size // get_world_size())

    assert args.reward_name in PROMPT_CONDITIONED_REWARDS

    if distributed:
        from cdm.texts.text_dataset.rewardbench_dataloader import (
            load_rewardbench_records,
            RewardBenchPromptDataset,
        )
        records = load_rewardbench_records(args.train_rewardbench_jsonl)
        dataset = RewardBenchPromptDataset(records)
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        prompt_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
            collate_fn=lambda batch: batch, drop_last=(str(getattr(args, "method", "")) == "cdm"),
        )
    else:
        prompt_loader = create_rewardbench_prompt_dataloader(
            jsonl_path=args.train_rewardbench_jsonl,
            batch_size=batch_size,
            shuffle=True,
        )
    print_log(
        f"[*] Prompt source: RewardBench jsonl={args.train_rewardbench_jsonl} "
        f"(batch_size={batch_size}, distributed={distributed})"
    )
    return prompt_loader


def _next_prompt_batch(prompt_loader, prompt_iter):
    if prompt_loader is None:
        return None, None, prompt_iter

    try:
        batch_payload = next(prompt_iter)
    except StopIteration:
        prompt_iter = iter(prompt_loader)
        batch_payload = next(prompt_iter)

    if len(batch_payload) == 0:
        return [], None, prompt_iter

    if isinstance(batch_payload[0], dict):
        prompt_texts = [x["prompt"] for x in batch_payload]
        prompt_meta = list(batch_payload)
    else:
        prompt_texts = list(batch_payload)
        prompt_meta = None
    return prompt_texts, prompt_meta, prompt_iter


def _canonical_intermediate_category(category):
    if category is None:
        return None
    normalized = str(category).strip().lower()
    if normalized == "reason":
        return "reasoning"
    return normalized



# ══════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _score(sampler, x, step_idx, query_sz, args, prompt_texts=None,
           score=None, chunk_size=None, attention_mask=None, logits=None,
           return_logits=False, cache_hidden=False, alpha_override=None):
    """Estimate the value V_t(x) / alpha at state x for the SMC weights.

    Optional overrides:
        score      : override args.score (CDM training uses its own
                     `args.twist_score` knob, distinct from the
                     inference-time `args.score`).
        chunk_size : when not None, the M Monte-Carlo samples in the
                     x0_pred branch are processed in groups
                     of `chunk_size`, bounding the largest tensor flowing
                     through the denoiser to roughly [B * chunk_size, ...].
                     None = process all M at once (legacy behavior).
        logits     : pre-computed logits [B, L, V] from a prior forward pass.
                     When provided, the x0_pred branch skips its own backbone
                     call and reuses these logits instead.
        return_logits : when True, return a ``(score, logits)`` tuple so the
                     caller can cache logits for a subsequent proposal step.
    """
    if score is None:
        score = args.score
    M = max(1, int(args.M))
    alpha = float(alpha_override) if alpha_override is not None else float(args.kl_weight)

    if score == "twist":
        assert sampler.denoiser.head is not None, (
            "score='twist' requires a twist head attached to the denoiser"
        )
        if return_logits:
            logits_out, twist_val = sampler.denoiser(
                x,
                attention_mask=attention_mask,
                return_logits=True,
                return_value=True,
            )
            return twist_val.to(torch.float32), logits_out
        twist_val = sampler.denoiser(
            x,
            attention_mask=attention_mask,
            return_logits=False,
            return_value=True,
        )
        print_log(f"twist output : {twist_val}")
        return twist_val.to(torch.float32)

    elif score == "x0_pred":
        B = x.shape[0]
        mask_index = (x == sampler.mask_token)

        if cache_hidden:
            hidden = sampler.denoiser(x, attention_mask=attention_mask, return_logits=False, return_hidden=True)
        else:
            if logits is None:
                logits = sampler.denoiser(
                    x,
                    attention_mask=attention_mask,
                )  # [B, L, V] — single shared backbone pass

        m_chunk_size = chunk_size if chunk_size is not None else M

        reward_chunks = []
        for m0 in range(0, M, m_chunk_size):
            m_chunk = min(m_chunk_size, M - m0)

            if m_chunk == 1:
                x_chunk = x
                mask_chunk = mask_index
                if cache_hidden:
                    hidden_chunk = hidden
                else:
                    logits_chunk = logits
            else:
                x_chunk = x.repeat_interleave(m_chunk, dim=0)
                mask_chunk = mask_index.repeat_interleave(m_chunk, dim=0)
                if cache_hidden:
                    hidden_chunk = hidden.repeat_interleave(m_chunk, dim=0)
                else:
                    logits_chunk = logits.repeat_interleave(m_chunk, dim=0)

            total_in_m = hidden_chunk.shape[0] if cache_hidden else logits_chunk.shape[0]
            b_chunk_size = args.chunk_b_size if args.chunk_b_size is not None else total_in_m
            x0_hat_parts = []
            for b0 in range(0, total_in_m, b_chunk_size):
                this_b = min(b_chunk_size, total_in_m - b0)
                if cache_hidden:
                    hidden_b = hidden_chunk[b0:b0 + this_b]
                    logits_b = sampler.denoiser.apply_lm_head(hidden_b)
                else:
                    logits_b = logits_chunk[b0:b0 + this_b]
                noisy = add_gumbel_noise(logits_b, temperature=sampler.temperature)
                x0_hat_parts.append(torch.argmax(noisy, dim=-1))
                del noisy, logits_b
            x0_hat = torch.cat(x0_hat_parts, dim=0)  # [B*m_chunk, L]
            if cache_hidden:
                del hidden_chunk
            else:
                del logits_chunk
            del x0_hat_parts

            x0_hat = torch.where(mask_chunk, x0_hat, x_chunk)
            del x_chunk, mask_chunk

            rewards_chunk = evaluate_generation(
                x0_hat[:, query_sz:],
                sampler.tokenizer,
                args,
                prompt_texts=prompt_texts,
            )

            rewards_chunk = torch.tensor(
                rewards_chunk,
                device=x.device,
                dtype=torch.float32,
            ).reshape(B, m_chunk)
            reward_chunks.append(rewards_chunk)

        rewards_m = torch.cat(reward_chunks, dim=1)  # [B, M]
        score_val = torch.logsumexp(rewards_m / alpha, dim=1) - math.log(M)
        if return_logits:
            return (score_val, hidden) if cache_hidden else (score_val, logits)
        return score_val

    raise ValueError(f"Unsupported score mode: {score}")


# ══════════════════════════════════════════════════════════════
#  LLaDA proposal step
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _propose_llada_step(
    sampler, x, num_transfer_tokens, step_idx, remasking, chunk_size=None,
    attention_mask=None, cached_logits=None, denoise_chunk_size=None,
    cached_hidden=None,
):
    """Single LLaDA reverse step.

    When ``chunk_size`` is set, the batch is processed in groups of at most
    ``chunk_size`` rows so the largest tensor flowing through the denoiser is
    bounded — useful when running SMC over wide particle pools.

    When ``cached_logits`` is provided ([B, L, V]), the backbone forward pass
    is skipped and the cached logits are used directly.

    When ``denoise_chunk_size`` is set (and ``cached_logits`` is *not*
    provided), the denoiser forward pass is pre-computed in chunks of
    ``denoise_chunk_size`` — which can be larger than ``chunk_size`` to
    improve GPU utilisation — and the rest of the step runs with the smaller
    ``chunk_size`` using the pre-computed logits.
    """
    B = x.shape[0]
    cs = chunk_size if chunk_size is not None else B
    use_hidden_cache = cached_hidden is not None
    # Pre-compute logits with a (potentially larger) denoise chunk size.
    _locally_cached = False
    if (not use_hidden_cache) and cached_logits is None and denoise_chunk_size is not None:
        dcs = denoise_chunk_size
        logit_chunks = []
        for b0 in range(0, B, dcs):
            b_end = min(b0 + dcs, B)
            am_b = None if attention_mask is None else attention_mask[b0:b_end]
            logit_chunks.append(sampler.denoiser(x[b0:b_end], attention_mask=am_b))
        cached_logits = torch.cat(logit_chunks, dim=0)
        del logit_chunks
        _locally_cached = True

    out_chunks = []



    for b0 in range(0, B, cs):

        b_end = min(b0 + cs, B)
        x_b = x[b0:b_end]
        ntt_b = num_transfer_tokens[b0:b_end]
        attention_mask_b = None if attention_mask is None else attention_mask[b0:b_end]

        mask_index = (x_b == sampler.mask_token)

        if use_hidden_cache:
            hidden_b = cached_hidden[b0:b_end]
            logits = sampler.denoiser.apply_lm_head(hidden_b)
        elif cached_logits is not None:
            logits = cached_logits[b0:b_end]
        else:
            logits = sampler.denoiser(x_b, attention_mask=attention_mask_b)
        logits_with_noise = add_gumbel_noise(logits, temperature=sampler.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "low_conf_noisy":
            p = F.log_softmax(logits_with_noise.log().to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(f"Unknown remasking strategy: {remasking}")
        del logits, logits_with_noise

        x0 = torch.where(mask_index, x0, x_b)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            k = int(ntt_b[j, step_idx].item())
            if k <= 0:
                continue
            _, select_index = torch.topk(confidence[j], k=k)
            transfer_index[j, select_index] = True

        x_next_b = x_b.clone()
        x_next_b[transfer_index] = x0[transfer_index]
        out_chunks.append(x_next_b)

    if _locally_cached:
        del cached_logits

    return torch.cat(out_chunks, dim=0)


# ══════════════════════════════════════════════════════════════
#  Sampling
# ══════════════════════════════════════════════════════════════


_ESS_LOG_PERCENTAGES = (0, 10, 25, 50, 75, 100)
_ESS_RESAMPLE_THRESHOLD = 0.5


def _ess_checkpoint_steps(num_steps):
    """Map configured logging percentages to reverse-step counts."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return {
        percent: int(round(num_steps * percent / 100.0))
        for percent in _ESS_LOG_PERCENTAGES
    }








def _summarize_ess_diagnostics(diagnostics, K):
    """Aggregate sampler ESS diagnostics over all prompt trajectories."""
    values_by_percent = {
        percent: [] for percent in _ESS_LOG_PERCENTAGES if percent != 0
    }
    resample_counts = []
    total_resample_opportunities = 0

    for batch_diag in diagnostics:
        batch_values = batch_diag["normalized_ess"]
        for percent in values_by_percent:
            values_by_percent[percent].extend(
                float(value) for value in batch_values[str(percent)]
            )

        batch_resample_counts = [
            int(value) for value in batch_diag["resample_counts"]
        ]
        resample_counts.extend(batch_resample_counts)
        total_resample_opportunities += (
            int(batch_diag["resample_opportunities_per_trajectory"])
            * len(batch_resample_counts)
        )

    num_trajectories = len(resample_counts)
    normalized_ess_mean = {
        str(percent): (float(np.mean(values)) if values else None)
        for percent, values in values_by_percent.items()
    }
    raw_ess_mean = {
        percent: (value * K if value is not None else None)
        for percent, value in normalized_ess_mean.items()
    }
    num_resamples = int(sum(resample_counts))
    resamples_per_trajectory = (
        num_resamples / num_trajectories if num_trajectories else 0.0
    )
    resampling_ratio = (
        num_resamples / total_resample_opportunities
        if total_resample_opportunities
        else 0.0
    )

    checkpoint_steps = (
        diagnostics[0].get("checkpoint_steps") if diagnostics else None
    )
    return {
        "metric": "normalized_ess",
        "definition": "ESS/K = 1 / (K * sum_i(w_i^2))",
        "K": int(K),
        "resample_threshold": _ESS_RESAMPLE_THRESHOLD,
        "percentages": list(_ESS_LOG_PERCENTAGES),
        "checkpoint_steps": checkpoint_steps,
        "normalized_ess_mean": normalized_ess_mean,
        "raw_ess_mean": raw_ess_mean,
        "num_trajectories": num_trajectories,
        "resamples_per_trajectory": resamples_per_trajectory,
        "num_resamples": num_resamples,
        "resampling_ratio": resampling_ratio,
        "total_resample_opportunities": total_resample_opportunities,
    }


def _format_ess_table(summary):
    """Format the requested test-split ESS and resampling summary table."""
    headers = [
        *(f"{percent}%" for percent in _ESS_LOG_PERCENTAGES),
        "resample/traj",
        "N(resample)",
    ]
    ess_cells = ["--"]
    for percent in _ESS_LOG_PERCENTAGES[1:]:
        value = summary["normalized_ess_mean"][str(percent)]
        ess_cells.append("--" if value is None else f"{value:.4f}")
    ess_cells.extend([
        f"{summary['resamples_per_trajectory']:.4f}",
        str(summary["num_resamples"]),
    ])

    lines = [
        (
            f"[*] Test-split ESS (normalized by K={summary['K']}; "
            f"higher is better; resample if ESS/K < "
            f"{summary['resample_threshold']:.2f})"
        ),
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(ess_cells) + " |",
        (
            "[*] Resampling ratio: "
            f"{summary['resampling_ratio']:.4f} "
            f"({summary['num_resamples']}/"
            f"{summary['total_resample_opportunities']})"
        ),
    ]
    return "\n".join(lines)


def _write_ess_summary(eval_path, ess_path, summary):
    with open(ess_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    output_block = _format_ess_table(summary)
    print_log(output_block)
    with open(eval_path, "a", encoding="utf-8") as f:
        f.write(output_block + "\n\n")

    wandb_payload = {
        "eval/ess_resamples_per_trajectory": summary["resamples_per_trajectory"],
        "eval/ess_num_resamples": summary["num_resamples"],
        "eval/ess_resampling_ratio": summary["resampling_ratio"],
        "eval/ess_resample_threshold": summary["resample_threshold"],
    }
    for percent, value in summary["normalized_ess_mean"].items():
        if value is not None:
            wandb_payload[f"eval/ess_normalized_{percent}pct"] = value
    _wandb_log(wandb_payload)


@torch.no_grad()
def sample(
    sampler,
    init_seq,
    args,
    query_sz,
    prompt_texts=None,
    attention_mask=None,
    diagnostics=None,
):
    """Unified SMC loop; K=1 reduces to unguided LLaDA sampling."""
    batch_size = init_seq.shape[0]

    if args.method == "base" or args.K <= 1:
        return sampler.sample(
            init_seq=init_seq,
            batch_size=batch_size,
            remasking=args.remasking,
            return_traj=False, 
            stop_t=None,
            attention_mask=attention_mask,
        )

    # ── SMC with resampling ──
    K = args.K
    seq_len = init_seq.shape[1]
    total = batch_size * K

    x = init_seq.repeat_interleave(K, dim=0).to(sampler.device)
    attention_mask_expanded = None
    if attention_mask is not None:
        attention_mask_expanded = attention_mask.repeat_interleave(K, dim=0)

    mask_index = (x == sampler.mask_token)
    num_transfer_tokens = sampler.get_num_transfer_tokens(mask_index, sampler.steps)

    v_cache = None
    logits_cache = None
    log_w_accum = torch.zeros(
        batch_size, K, device=sampler.device, dtype=torch.float32
    )
    checkpoint_steps = None
    percent_by_step = {}
    ess_by_percent = {}
    resample_counts = None
    if diagnostics is not None:
        checkpoint_steps = _ess_checkpoint_steps(sampler.steps)
        percent_by_step = {
            step: percent
            for percent, step in checkpoint_steps.items()
            if 0 < percent < 100
        }
        resample_counts = torch.zeros(
            batch_size, device=sampler.device, dtype=torch.int64
        )

    for i in tqdm(range(sampler.steps), desc=f"SMC ({args.score})"):
        x_next = _propose_llada_step(
            sampler=sampler,
            x=x,
            num_transfer_tokens=num_transfer_tokens,
            step_idx=i,
            remasking=args.remasking,
            chunk_size=args.chunk_b_size,
            attention_mask=attention_mask_expanded,
            cached_logits=logits_cache,
            denoise_chunk_size=getattr(args, 'denoise_chunk_size', None),
        )

        if i == sampler.steps - 1:
            x = x_next
            break

        v_next, logits_cache = _score(
            sampler=sampler,
            x=x_next,
            step_idx=i + 1,
            query_sz=query_sz,
            args=args,
            prompt_texts=prompt_texts,
            chunk_size=args.chunk_m_size,
            attention_mask=attention_mask_expanded,
            return_logits=True,
        )

        if v_cache is None:
            v_curr = torch.ones_like(v_next)
        else:
            v_curr = v_cache

        log_w_increment = (v_next - v_curr).reshape(batch_size, K)
        log_w_accum = log_w_accum + log_w_increment
        log_w = log_w_accum - log_w_accum.max(dim=1, keepdim=True).values
        w = torch.softmax(log_w, dim=1)
        print_log("importance weight: ", w)
        print_log(f"[*] Step {i+1}/{sampler.steps}: ESS/K = {1.0 / (K * w.float().pow(2).sum(dim=1)).mean():.4f}")
        ess = 1.0 / (K * w.float().pow(2).sum(dim=1))
        need_resample = ess < _ESS_RESAMPLE_THRESHOLD
        indices = torch.arange(K, device=w.device, dtype=torch.long).unsqueeze(0).expand(batch_size, K).clone()
        if need_resample.any():
            indices[need_resample] = torch.multinomial(
                w[need_resample],
                num_samples=K,
                replacement=True,
            )
        progress_step = i + 1
        if diagnostics is not None and progress_step in percent_by_step:
            percent = percent_by_step[progress_step]
            ess_by_percent[percent] = ess.detach().cpu().tolist()

        if resample_counts is not None:
            resample_counts += need_resample.to(torch.int64)
        batch_idx = torch.arange(batch_size, device=sampler.device)[:, None].expand(-1, K)

        x_next = x_next.reshape(batch_size, K, seq_len)
        x_next = x_next[batch_idx, indices].reshape(total, seq_len)

        v_cache = v_next.reshape(batch_size, K)
        v_cache = v_cache[batch_idx, indices].reshape(total)
        print_log("twist values: ", v_cache)

        # Resample cached logits alongside particles
        if logits_cache is not None:
            logits_cache = logits_cache.reshape(batch_size, K, seq_len, -1)
            logits_cache = logits_cache[batch_idx, indices].reshape(total, seq_len, logits_cache.shape[-1])

        # Keep accumulated weights for rows that skipped resampling; reset
        # only the prompt-level particle systems that were resampled.
        log_w_accum = log_w_accum.clone()
        log_w_accum[need_resample] = 0.0
        x = x_next


    rewards = evaluate_generation(
        x[:, query_sz:],
        sampler.tokenizer,
        args,
        prompt_texts=prompt_texts,
    )

    rewards_tensor = torch.tensor(rewards, device=sampler.device, dtype=torch.float32)
    terminal_log_w = (
        rewards_tensor / args.kl_weight - v_cache
    ).reshape(batch_size, K)
    log_w_last = log_w_accum + terminal_log_w
    w_last = torch.softmax(log_w_last, dim=1)
    ess_last = 1.0 / (K * w_last.float().pow(2).sum(dim=1))
    need_resample_last = ess_last < _ESS_RESAMPLE_THRESHOLD
    indices = torch.arange(K, device=w_last.device, dtype=torch.long).unsqueeze(0).expand(batch_size, K).clone()
    if need_resample_last.any():
        indices[need_resample_last] = torch.multinomial(
            w_last[need_resample_last],
            num_samples=K,
            replacement=True,
        )

    if diagnostics is not None:
        ess_by_percent[100] = ess_last.detach().cpu().tolist()

    if resample_counts is not None:
        resample_counts += need_resample_last.to(torch.int64)
    batch_idx = torch.arange(batch_size, device=sampler.device)[:, None].expand(-1, K)

    x = x.reshape(batch_size, K, seq_len)
    x = x[batch_idx, indices].reshape(total, seq_len)

    rewards_2d = rewards_tensor.reshape(batch_size, K)[batch_idx, indices]

    best = rewards_2d.argmax(dim=1)

    x = x.reshape(batch_size, K, seq_len)
    x = x[torch.arange(batch_size, device=sampler.device), best]
    print_log(ess_by_percent)

    if diagnostics is not None:
        missing_percentages = [
            percent
            for percent in _ESS_LOG_PERCENTAGES[1:]
            if percent not in ess_by_percent
        ]
        if missing_percentages:
            raise RuntimeError(
                "Missing ESS checkpoints for percentages: "
                f"{missing_percentages}"
            )
        diagnostics.update({
            "checkpoint_steps": {
                str(percent): checkpoint_steps[percent]
                for percent in _ESS_LOG_PERCENTAGES
            },
            "normalized_ess": {
                str(percent): ess_by_percent[percent]
                for percent in _ESS_LOG_PERCENTAGES[1:]
            },
            "resample_counts": resample_counts.detach().cpu().tolist(),
            "resample_opportunities_per_trajectory": int(sampler.steps),
        })

    return x


# ══════════════════════════════════════════════════════════════
#  CDM training
# ══════════════════════════════════════════════════════════════


def train_cdm(args, model, denoiser, sampler, ppl_model):
    WS = get_world_size()
    assert sampler.steps == args.gen_length, f"Sampler steps ({sampler.steps}) must match generation length ({args.gen_length})"
    assert args.twist_batch_size % WS == 0, f"Expected twist_batch_size ({args.twist_batch_size}) to be divisible by world size ({WS})"
    assert args.cdm_buffer_size % WS == 0, f"Expected cdm_buffer_size ({args.cdm_buffer_size}) to be divisible by world size ({WS})"
    assert args.cdm_pos_prompt_sample_size % WS == 0, f"Expected cdm_pos_prompt_sample_size ({args.cdm_pos_prompt_sample_size}) to be divisible by world size ({WS})"
    assert args.cdm_buffer_size % args.cdm_pos_prompt_sample_size == 0, f"Expected cdm_buffer_size ({args.cdm_buffer_size}) to be divisible by cdm_pos_prompt_sample_size ({args.cdm_pos_prompt_sample_size})"
    _run_smc_size = getattr(args, "run_smc_size", None)
    if _run_smc_size is not None and int(_run_smc_size) > 0:
        assert int(_run_smc_size) % WS == 0, f"Expected run_smc_size ({_run_smc_size}) to be divisible by world size ({WS})"
        assert args.cdm_pos_prompt_sample_size % int(_run_smc_size) == 0, f"Expected cdm_pos_prompt_sample_size ({args.cdm_pos_prompt_sample_size}) to be divisible by run_smc_size ({_run_smc_size})"
    if args.match_pos_neg_prompt:
        assert args.twist_batch_size % args.cdm_pos_prompt_sample_size == 0, f"match_pos_neg_prompt requires twist_batch_size ({args.twist_batch_size}) divisible by cdm_pos_prompt_sample_size ({args.cdm_pos_prompt_sample_size})"
        if not args.cdm_pos_keep_all_smc:
            assert args.cdm_buffer_size == args.twist_batch_size, f"match_pos_neg_prompt with cdm_pos_keep_all_smc=False requires cdm_buffer_size ({args.cdm_buffer_size}) == twist_batch_size ({args.twist_batch_size})"
    elif args.cdm_neg_sample_method == "multi_is":
        assert args.twist_batch_size % args.cdm_neg_prompt_sample_size == 0, "Check Neg Multi IS prompt sample size"
        assert args.cdm_neg_prompt_sample_size % WS == 0, f"cdm_neg_prompt_sample_size ({args.cdm_neg_prompt_sample_size}) must be divisible by world size ({WS})"

    if is_main_process():
        print("="*100)
        print(args.tag)
    distributed = getattr(args, "distributed_active", False)
    make_twist_net(args, denoiser)
    if distributed:
        denoiser.head = torch.nn.parallel.DistributedDataParallel(denoiser.head, device_ids=[args.gpu], output_device=args.gpu)
    head = denoiser.head
    head.train()
    model.eval()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, head.parameters()), lr=args.twist_lr, weight_decay=args.twist_weight_decay)
    total_steps = args.twist_epochs * args.twist_steps_per_epoch
    scheduler, _ = make_linear_decay_scheduler(optimizer, total_steps, args.twist_lr_decay_start_frac) if args.twist_lr_decay_start_frac is not None else (None, None)

    ema_list = list(args.ema_decay)
    use_emas = bool(args.save_ema or args.cdm_ema_for_sampling)
    if use_emas:
        denoiser.attach_ema_heads(ema_list)

    total_params = sum(p.numel() for p in head.parameters()) / 1e6
    trainable_params_num = sum(p.numel() for p in head.parameters() if p.requires_grad) / 1e6

    if is_main_process():
        print_log(f"[*] Total parameters in twist head: {total_params:.2f}M")
        print_log(f"[*] Trainable parameters in twist head: {trainable_params_num:.2f}M")

    pos_prompt_loader = _build_train_prompt_loader(args, batch_size=args.cdm_pos_prompt_sample_size, distributed=distributed, seed=args.seed)
    pos_prompt_iter = iter(pos_prompt_loader)

    if args.match_pos_neg_prompt:
        neg_prompt_loader = None
        neg_prompt_iter = None
    else:
        neg_prompt_loader = _build_train_prompt_loader(args, batch_size=args.cdm_neg_prompt_sample_size, distributed=distributed, seed=args.seed+1)
        neg_prompt_iter = iter(neg_prompt_loader)

    pos_batches_per_epoch = len(pos_prompt_loader)
    if args.match_pos_neg_prompt:
        assert args.twist_steps_per_epoch <= pos_batches_per_epoch, "Reduce steps per epoch or revise the code"
    else:
        neg_batches_per_epoch = len(neg_prompt_loader)
        assert args.twist_steps_per_epoch <= pos_batches_per_epoch and args.twist_steps_per_epoch <= neg_batches_per_epoch, "Reduce steps per epoch or revise the code"

    best_loss = 1e9
    best_head_sd = None
    best_ema_sds = None
    best_epoch = -1
    global_step = 0
    checkpoint_log = []
    full_eval_every = int(args.full_eval_interval) if getattr(args, "full_eval_interval", None) is not None else 0

    pos_buffer = CDMPosBuffer(args)
    pbar = tqdm(range(args.twist_epochs), desc="CDM", total=args.twist_epochs, leave=True, dynamic_ncols=True, disable=not is_main_process())
    for epoch in pbar:
        if distributed and hasattr(pos_prompt_loader, "sampler") and hasattr(pos_prompt_loader.sampler, "set_epoch"):
            pos_prompt_loader.sampler.set_epoch(epoch)
            pos_prompt_iter = iter(pos_prompt_loader)
            if not args.match_pos_neg_prompt:
                neg_prompt_loader.sampler.set_epoch(epoch)
                neg_prompt_iter = iter(neg_prompt_loader)
        else:
            assert not distributed, "Distributed training with a prompt loader requires a sampler with set_epoch support"

        pos_prompt_texts, _, pos_prompt_iter = _next_prompt_batch(pos_prompt_loader, pos_prompt_iter)
        _, pos_init_seq, pos_query_sz, pos_attention_mask = get_input_prompt(args, denoiser, pos_prompt_texts)

        head.eval()
        pos_buffer.clear()
        pos_buffer.fill(sampler, pos_init_seq, pos_query_sz, pos_prompt_texts, pos_attention_mask, epoch, _collect_pos_samples)

        losses = list()
        for step_in_epoch in tqdm(range(args.twist_steps_per_epoch), desc="Train Steps", total=args.twist_steps_per_epoch, leave=False, dynamic_ncols=True, disable=not is_main_process()):
            if args.match_pos_neg_prompt:
                neg_init_seq = pos_init_seq
                neg_attention_mask = pos_attention_mask
            else:
                neg_prompt_texts, _, neg_prompt_iter = _next_prompt_batch(neg_prompt_loader, neg_prompt_iter)
                _, neg_init_seq, neg_query_sz, neg_attention_mask = get_input_prompt(args, denoiser, neg_prompt_texts)

            _random_sampled_t = torch.randint(1, sampler.steps, (1,), device=args.device)
            if distributed:
                torch.distributed.broadcast(_random_sampled_t, src=0)
            random_sampled_t = int(_random_sampled_t.item())
            with torch.no_grad():
                head.eval()
                twist_bs_per_rank = args.twist_batch_size // WS
                samples_pos, rewards_pos, weights_pos, prompt_texts_pos, attn_mask_pos = pos_buffer.sample(twist_bs_per_rank, args.device)
                samples_neg = _collect_neg_samples(args, sampler, neg_init_seq, neg_attention_mask, random_sampled_t)

            head.train()
            optimizer.zero_grad()
            pos_ess_vals, neg_ess_vals = list(), list()

            pos_term = compute_pos_loss(args, denoiser, sampler, samples_pos, weights_pos, attn_mask_pos, pos_query_sz, random_sampled_t, epoch)
            neg_term, neg_ess = compute_neg_loss(args, denoiser, samples_neg, neg_attention_mask, ema_for_sampling=args.cdm_ema_for_sampling, ema_idx=0)

            pos_scale = WS if (distributed and args.global_softmax and not args.do_pos_multi_smc) else 1
            neg_scale = WS if (distributed and args.global_softmax and args.cdm_neg_sample_method != "multi_is") else 1

            step_loss = (neg_scale * neg_term) - (pos_scale * pos_term)
            step_loss.backward()

            pos_ess = pos_buffer.final_is_ess
            if neg_ess is not None:
                neg_ess_vals.append(neg_ess)
            if pos_ess is not None:
                pos_ess_vals.append(pos_ess)

            trainable_params = [p for p in head.parameters() if p.requires_grad]
            if args.twist_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(args.twist_clip_grad_norm)).item()
            else:
                grad_norm = torch.nn.utils.get_total_norm([p.grad for p in trainable_params if p.grad is not None]).item()

            optimizer.step()
            scheduler.step() if scheduler is not None else None

            if use_emas:
                denoiser.update_ema_heads()
            losses.append(step_loss.item())

            if wandb.run is not None and is_main_process():
                log_dict = {"cdm/loss": step_loss.item(), "cdm/grad_norm": grad_norm}

                if neg_ess_vals:
                    log_dict["cdm/neg_last_weight_ess"] = np.mean(neg_ess_vals)
                if pos_ess_vals:
                    log_dict["cdm/pos_last_weight_ess"] = np.mean(pos_ess_vals)

                if pos_buffer.smc_ess_mean is not None:
                    log_dict["cdm/pos_smc_ess_mean"] = pos_buffer.smc_ess_mean
                    log_dict["cdm/pos_smc_ess_min"] = pos_buffer.smc_ess_min
                    log_dict["cdm/pos_resampled_ratio"] = pos_buffer.ratio_resampled

                log_dict["cdm/pos_reward_mean"] = pos_buffer.reward_mean
                log_dict["cdm/pos_reward_min"] = pos_buffer.reward_min
                log_dict["cdm/pos_reward_max"] = pos_buffer.reward_max
                log_dict["cdm/pos_unique_per_prompt"] = pos_buffer.unique_per_prompt

                log_dict["cdm/Neg Term"] = neg_term.item()
                log_dict["cdm/Pos Term"] = pos_term.item()
                wandb.log(log_dict, step=global_step)
            global_step += 1

        avg = np.mean(losses)
        if distributed:
            avg = all_reduce_mean(avg)

        if (epoch + 1) % args.twist_log_every == 0 and is_main_process():
            checkpoint_log.append({"epoch": epoch + 1, "loss": avg, "best_loss": best_loss})
            with open(os.path.join(args.save_path, "checkpoint_log.json"), "w", encoding="utf-8") as f:
                json.dump(checkpoint_log, f, indent=2)

        if (epoch + 1) % args.model_save_interval == 0:
            if is_main_process():
                save_twist_checkpoint(denoiser, args, epoch, avg, f"twist_epoch_{epoch+1}.pt", model.config.vocab_size)
                if args.save_ema:
                    for idx, ema_decay_rate in enumerate(ema_list):
                        save_twist_checkpoint(denoiser, args, epoch, avg, f"twist_epoch_{epoch+1}_ema{ema_decay_rate}.pt", model.config.vocab_size, head_override=denoiser.ema_heads[idx])
            barrier()

        if full_eval_every > 0 and (epoch + 1) % full_eval_every == 0:
            barrier()
            if (epoch + 1) % args.model_save_interval != 0:
                if is_main_process():
                    save_twist_checkpoint(denoiser, args, epoch, avg, f"twist_epoch_{epoch+1}.pt", model.config.vocab_size)
            barrier()
            head.eval()
            evaluate(args, denoiser, sampler, ppl_model, postfix=f"epoch_{epoch+1}", print_log_bool=False)
            barrier()
            head.train()

        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch + 1
            if epoch >= 15:
                best_head_sd = {k: v.detach().clone() for k, v in unwrap_head(head).state_dict().items()}
                if use_emas:
                    best_ema_sds = [{k: v.detach().clone() for k, v in eh.state_dict().items()} for eh in denoiser.ema_heads]
    if is_main_process():
        save_twist_checkpoint(denoiser, args, epoch, avg, f"twist_final.pt", model.config.vocab_size)
        if args.save_ema:
            for idx, ema_decay_rate in enumerate(ema_list):
                save_twist_checkpoint(denoiser, args, epoch, avg, f"twist_final_ema{ema_decay_rate}.pt", model.config.vocab_size, head_override=denoiser.ema_heads[idx])

        if best_head_sd is not None:
            final_head_sd = {k: v.detach().clone() for k, v in unwrap_head(head).state_dict().items()}
            unwrap_head(head).load_state_dict(best_head_sd)
            save_twist_checkpoint(denoiser, args, best_epoch - 1, best_loss, f"twist_best_epoch_{best_epoch}.pt", model.config.vocab_size)
            unwrap_head(head).load_state_dict(final_head_sd)
            if args.save_ema and best_ema_sds is not None:
                pass

    barrier()

    if distributed:
        denoiser.head = unwrap_head(denoiser.head)
        head = denoiser.head
    head.eval()
    return best_epoch


# ══════════════════════════════════════════════════════════════
#  Output helpers
# ══════════════════════════════════════════════════════════════


def write_batch_outputs(
    eval_path,
    jsonl_path,
    raw_prompts,
    out_decoded,
    rewards,
    ppls,
    res_dict,
    prompt_meta=None,
    gen_times=None,
):
    with open(eval_path, "a", encoding="utf-8") as f_eval, open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
        for i in range(len(out_decoded)):
            meta = None if prompt_meta is None else prompt_meta[i]
            meta_block = ""
            if meta is not None:
                meta_block = (
                    f"[*] Category: {meta.get('category')}\n"
                    f"[*] Subset: {meta.get('subset')}\n"
                    f"[*] Prompt ID: {meta.get('id')}\n"
                )
            gen_time_val = None
            if gen_times is not None and i < len(gen_times):
                gen_time_val = gen_times[i]
            gen_time_block = ""
            if gen_time_val is not None:
                gen_time_block = f"[*] Gen Time (s): \n{float(gen_time_val):.6f}\n"
            output_block = (
                f"[*] Query: \n{raw_prompts[i]}\n"
                f"{meta_block}"
                f"[*] Response: \n{out_decoded[i]}\n"
                f"[*] Reward (Given): \n{rewards[i]}\n"
                f"[*] Reward (Holdout): \n{rewards[i]}\n"
                f"{gen_time_block}"
                f"[*] PPL: \n{ppls[i]}\n"
                "-------------------------------- \n"
            )
            print(output_block)
            f_eval.write(output_block + "\n")

            generation_record = {
                "query": raw_prompts[i],
                "response": out_decoded[i],
                "reward": float(rewards[i]),
                "ppl": float(ppls[i]),
            }
            if gen_time_val is not None:
                generation_record["gen_time_sec"] = float(gen_time_val)
            if meta is not None:
                generation_record["category"] = meta.get("category")
                generation_record["subset"] = meta.get("subset")
                generation_record["prompt_id"] = meta.get("id")
            f_jsonl.write(json.dumps(generation_record, ensure_ascii=False) + "\n")

            res_dict["reward"].append(float(rewards[i]))
            res_dict["ppl"].append(float(ppls[i]))


def write_summary(eval_path, avg_reward, avg_ppl=None):
    output_block = f"[*] Average Reward: \n{avg_reward}\n"
    if avg_ppl is not None:
        output_block += f"[*] Average PPL: \n{avg_ppl}\n"
    output_block += "-------------------------------- \n"
    print(output_block)
    with open(eval_path, "a", encoding="utf-8") as f:
        f.write(output_block + "\n")


def write_group_summary(eval_path, title, grouped_scores):
    lines = [f"[*] {title}"]
    for key in sorted(grouped_scores.keys()):
        scores = grouped_scores[key]
        mean_score = sum(scores) / max(1, len(scores))
        lines.append(f"    - {key}: {mean_score:.6f} (n={len(scores)})")
    lines.append("-------------------------------- ")
    output_block = "\n".join(lines)
    print(output_block)
    with open(eval_path, "a", encoding="utf-8") as f:
        f.write(output_block + "\n")


# ══════════════════════════════════════════════════════════════
#  Run (evaluation loop)
# ══════════════════════════════════════════════════════════════


def _heldout_reward(args, generation_records):
    """Mean ArmoRM score over already-generated (query, response) pairs."""
    from cdm.texts.rewards import armorm_score

    queries = [r["query"] for r in generation_records]
    responses = [r["response"] for r in generation_records]
    scores = armorm_score(
        texts=responses, prompt_texts=queries, args=args,
        batch_size=int(args.reward_eval_batch_size),
    )[0]
    for rec, score in zip(generation_records, scores):
        rec["heldout_reward"] = float(score)
    return float(np.mean(scores))


def evaluate(args, denoiser, sampler, ppl_model=None, postfix=None, print_log_bool=True):
    distributed = getattr(args, "distributed_active", False)
    rank = get_rank()
    world_size = get_world_size()

    if postfix is None:
        eval_path = os.path.join(args.save_path, "eval.txt")
        jsonl_path = os.path.join(args.save_path, "generations.jsonl")
        eval_desc = "EVAL"
    else:
        eval_path = os.path.join(args.save_path, f"eval_{postfix}.txt")
        jsonl_path = os.path.join(args.save_path, "generations", f"generations_{postfix}.jsonl")
        eval_desc = "EVAL: "+str(postfix)
    res_dict = defaultdict(list)
    category_reward = defaultdict(list)
    subset_reward = defaultdict(list)
    collect_ess = bool(
        args.method == "cdm"
        and args.score == "twist"
        and getattr(args, "twist_ckpt", None)
    )
    local_ess_diagnostics = []

    print_log(f"[*] Method: {args.method}") if print_log_bool else None
    print_log(f"[*] Reward: {args.reward_name} ({args.reward_label})") if print_log_bool else None
    print_log(f"[*] Save dir: {args.save_path}") if print_log_bool else None

    eval_ppl = bool(getattr(args, "eval_ppl", True))
    if eval_ppl and ppl_model is None:
        ppl_model = QwenPPL(task="text")
    eval_ppl = eval_ppl and ppl_model is not None

    # Load all records and partition across GPUs for distributed eval
    from cdm.texts.text_dataset.rewardbench_dataloader import load_rewardbench_records
    all_records = load_rewardbench_records(args.eval_rewardbench_jsonl)
    if args.max_batches is not None:
        max_records = int(args.max_batches) * int(args.batch_size)
        all_records = all_records[:max_records]

    # Partition records across ranks (each rank gets a contiguous shard)
    if distributed:
        per_rank = len(all_records) // world_size
        start_idx = rank * per_rank
        # Last rank takes any remainder
        end_idx = len(all_records) if rank == world_size - 1 else start_idx + per_rank
        local_records = all_records[start_idx:end_idx]
        print_log(f"[*] Rank {rank}: evaluating records [{start_idx}:{end_idx}] ({len(local_records)} samples)") if print_log_bool else None
    else:
        local_records = all_records

    print_log(f"[*] Prompt source: RewardBench jsonl={args.eval_rewardbench_jsonl}") if print_log_bool and is_main_process() else None

    # Process local partition in batches
    use_rewardbench = args.reward_name in PROMPT_CONDITIONED_REWARDS
    bs = int(args.batch_size)
    local_generation_records = []
    for batch_start in tqdm(range(0, len(local_records), bs), desc=eval_desc, leave=False, dynamic_ncols=True, disable=not is_main_process()):
        batch_payload = local_records[batch_start:batch_start + bs]
        if len(batch_payload) == 0:
            break

        if use_rewardbench:
            prompt_meta = list(batch_payload)
            raw_prompts = [x["prompt"] for x in prompt_meta]

        _, init_seq, query_sz, attention_mask = get_input_prompt(args, denoiser, raw_prompts)
        batch_ess_diagnostics = {} if collect_ess else None
        t_gen_start = time.time()
        out_full = sample(
            sampler=sampler,
            init_seq=init_seq,
            args=args,
            query_sz=query_sz,
            prompt_texts=raw_prompts,
            attention_mask=attention_mask,
            diagnostics=batch_ess_diagnostics,
        )
        gen_times = [(time.time() - t_gen_start) / len(raw_prompts)] * len(raw_prompts)
        if collect_ess:
            local_ess_diagnostics.append(batch_ess_diagnostics)

        # NOTE: query_size needs to be same across batch.
        out = out_full[:, query_sz:]

        rewards = evaluate_generation(
            out,
            denoiser.tokenizer,
            args,
            prompt_texts=raw_prompts,
        )
        out_decoded = denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)
        ppls = ppl_model(out_decoded) if eval_ppl else None

        # Collect results locally
        for i in range(len(out_decoded)):
            res_dict["reward"].append(float(rewards[i]))
            if eval_ppl:
                res_dict["ppl"].append(float(ppls[i]))
            meta = prompt_meta[i] if use_rewardbench else None
            rec = {
                "query": raw_prompts[i],
                "response": out_decoded[i],
                "reward": float(rewards[i]),
            }
            if eval_ppl:
                rec["ppl"] = float(ppls[i])
            if gen_times is not None and i < len(gen_times):
                rec["gen_time_sec"] = float(gen_times[i])
            if meta is not None:
                rec["category"] = meta.get("category")
                rec["subset"] = meta.get("subset")
                rec["prompt_id"] = meta.get("id")
            local_generation_records.append(rec)

        for m, r in zip(prompt_meta, rewards):
            cat = m.get("category")
            subset = m.get("subset")
            if cat is not None:
                category_reward[str(cat)].append(float(r))
            if subset is not None:
                subset_reward[str(subset)].append(float(r))

    # Gather results from all ranks
    barrier()
    if distributed:
        all_rewards = all_gather_list(res_dict["reward"])
        all_generation_records = all_gather_list(local_generation_records)
        all_ess_diagnostics = (
            all_gather_list(local_ess_diagnostics) if collect_ess else []
        )
        # Gather category/subset dicts
        local_cat_items = [(k, v) for k, v in category_reward.items()]
        local_sub_items = [(k, v) for k, v in subset_reward.items()]
        all_cat_items = all_gather_list(local_cat_items)
        all_sub_items = all_gather_list(local_sub_items)
        # Rebuild on all ranks (needed for rank 0 summary)
        category_reward = defaultdict(list)
        subset_reward = defaultdict(list)
        for k, v in all_cat_items:
            category_reward[k].extend(v)
        for k, v in all_sub_items:
            subset_reward[k].extend(v)
        res_dict["reward"] = all_rewards
        if eval_ppl:
            res_dict["ppl"] = all_gather_list(res_dict["ppl"])
    else:
        all_generation_records = local_generation_records
        all_ess_diagnostics = local_ess_diagnostics

    # Only rank 0 writes outputs and summary
    if is_main_process():
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for rec in all_generation_records:
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        avg_reward = sum(res_dict["reward"]) / max(1, len(res_dict["reward"]))
        avg_ppl = (sum(res_dict["ppl"]) / max(1, len(res_dict["ppl"]))) if eval_ppl else None

        write_summary(eval_path, avg_reward, avg_ppl)
        print_log(f"[*] Average reward: {avg_reward:.6f}") if print_log_bool else None
        if eval_ppl:
            print_log(f"[*] Average ppl: {avg_ppl:.6f}") if print_log_bool else None
        print_log("[*] Reward mean by category:") if print_log_bool else None
        for category in sorted(category_reward.keys()):
            scores = category_reward[category]
            mean_score = sum(scores) / max(1, len(scores))
            print_log(f"    - {category}: {mean_score:.6f} (n={len(scores)})") if print_log_bool else None
        print_log("[*] Reward mean by subset:") if print_log_bool else None
        for subset_key in sorted(subset_reward.keys()):
            scores = subset_reward[subset_key]
            mean_score = sum(scores) / max(1, len(scores))
            print_log(f"    - {subset_key}: {mean_score:.6f} (n={len(scores)})") if print_log_bool else None
        write_group_summary(eval_path, "Reward mean by category", category_reward)
        write_group_summary(eval_path, "Reward mean by subset", subset_reward)
        if collect_ess:
            ess_summary = _summarize_ess_diagnostics(
                all_ess_diagnostics,
                K=int(args.K),
            )
            ess_filename = (
                "ess.json" if postfix is None else f"ess_{postfix}.json"
            )
            _write_ess_summary(
                eval_path=eval_path,
                ess_path=os.path.join(args.save_path, ess_filename),
                summary=ess_summary,
            )
        # Heldout reward: ArmoRM over the same generations, never used for scaling.
        heldout = _heldout_reward(args, all_generation_records)
        gen_time_vals = [r["gen_time_sec"] for r in all_generation_records if "gen_time_sec" in r]
        result = dict(app="dllm", method=args.method, K=args.K, M=args.M, seed=args.seed,
                      given_reward=round(float(avg_reward), 4),
                      heldout_reward=round(float(heldout), 6),
                      sec_per_sample=round(float(np.mean(gen_time_vals)), 4) if gen_time_vals else None,
                      n_samples=len(all_generation_records),
                      twist_ckpt=args.twist_ckpt)
        results_name = "results.json" if postfix is None else f"results_{postfix}.json"
        with open(os.path.join(args.save_path, results_name), "w") as f:
            json.dump(result, f, indent=2)
        print_log("[RESULT] " + " ".join(f"{k}={v}" for k, v in result.items()))

        print_log(f"[*] Saved to {args.save_path}") if print_log_bool else None


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════


@hydra.main(config_path="configs", config_name="smc", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Initialize distributed mode (no-op if distributed.enabled=false)
    init_distributed_mode(cfg)

    cfg.device = cfg.get("device", "cuda") if not getattr(cfg, "distributed_active", False) else cfg.device
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    seed_everything(cfg.seed + get_rank())  # Different seed per rank for diversity

    train_cdm_twist = cfg.method == "cdm" and cfg.get("twist_ckpt") is None
    if train_cdm_twist:
        tag = prepare_cdm_config_and_tag(cfg)
    elif cfg.method == "cdm":
        # the twist stem keeps two runs that differ only in twist_ckpt from colliding
        tag = f"cdm_K{cfg.K}_{os.path.splitext(os.path.basename(cfg.twist_ckpt))[0]}"
    else:
        tag = f"{cfg.method}_K{cfg.K}_M{cfg.M}"

    cfg.tag = tag
    cfg.save_path = os.path.join(cfg.save_path, tag)
    if is_main_process():
        os.makedirs(cfg.save_path, exist_ok=True)
        with open(os.path.join(cfg.save_path, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        os.makedirs(os.path.join(cfg.save_path, "generations"), exist_ok=True)
    barrier()

    # Setup logging — all ranks write to run.log (rank 0) or run_rankN.log.
    if is_main_process():
        setup_logging(os.path.join(cfg.save_path, "run.log"))
    else:
        setup_logging(os.path.join(cfg.save_path, f"run_rank{get_rank()}.log"))
    if is_main_process() and not bool(cfg.get("disable_wandb", False)):
        if wandb is None:
            print_log("[!] wandb is not installed; continuing without wandb logging")
        else:
            wandb_project = str(cfg.get("wandb_project", "cdm"))
            wandb_name = str(cfg.get("wandb_name") or cfg.tag)
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                dir=cfg.save_path,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print_log(f"[*] wandb enabled | project={wandb_project} | name={wandb_name}")

    try:
        # Build denoiser + sampler once; shared by training and evaluation.
        # The twist head (if any) is attached directly to `llada_denoiser` —
        # there is no separate twist_net to thread through the call sites.
        llada_denoiser = LLaDADenoiser(device=cfg.device, chunk_b_size=cfg.chunk_b_size, call_chunk_size=cfg.call_chunk_size)
        llada_denoiser._set_length(cfg.gen_length)
        freeze_model(llada_denoiser.model)

        sampler = DiffusionSampler(
            llada_denoiser,
            steps=cfg.num_timesteps,
            temperature=cfg.temperature,
        )

        cdm_best_epoch = None
        cdm_ppl_model = None
        # The twist head is attached directly to `llada_denoiser`; the backbone is shared, so
        # one forward pass yields both the proposal logits and the scalar twist value.
        if cfg.method == "cdm":
            if not train_cdm_twist:
                print_log(f"[*] Loading twist from '{cfg.twist_ckpt}'")
                load_twist_checkpoint(cfg.twist_ckpt, device=cfg.device, denoiser=llada_denoiser)
            else:
                print_log("[*] No twist_ckpt provided — training the CDM twist")
                start = time.time()
                cdm_ppl_model = QwenPPL(task="text") if bool(getattr(cfg, "eval_ppl", False)) else None
                cdm_best_epoch = train_cdm(cfg, llada_denoiser.model, llada_denoiser, sampler, cdm_ppl_model)
                train_time = time.time() - start
                print_log(f"[*] CDM training time: {int(train_time // 3600)}h "
                          f"{int((train_time % 3600) // 60)}m") if is_main_process() else None
            llada_denoiser.head.eval()

        # Evaluation
        barrier()
        start = time.time()
        if train_cdm_twist:
            evaluate(cfg, llada_denoiser, sampler, cdm_ppl_model, postfix="final", print_log_bool=False)
            if cdm_best_epoch is not None:
                best_ckpt_path = os.path.join(cfg.save_path, f"twist_best_epoch_{cdm_best_epoch}.pt")
                if os.path.exists(best_ckpt_path):
                    best_payload = torch.load(best_ckpt_path, map_location=cfg.device, weights_only=False)
                    unwrap_head(llada_denoiser.head).load_state_dict(best_payload["twist_net"], strict=True)
                    llada_denoiser.head.eval()
                    evaluate(cfg, llada_denoiser, sampler, cdm_ppl_model, postfix=f"best_{cdm_best_epoch}", print_log_bool=False)
        else:
            evaluate(cfg, llada_denoiser, sampler)
        elapsed = time.time() - start
        print_log(f"Elapsed: {elapsed:.1f}s")
    except Exception:
        print_log(f"[!] Fatal error on rank {get_rank()}:\n{traceback.format_exc()}")
        raise
    finally:
        if is_main_process() and wandb is not None and wandb.run is not None:
            wandb.finish()
        close_logging()
        cleanup()


if __name__ == "__main__":
    main()
