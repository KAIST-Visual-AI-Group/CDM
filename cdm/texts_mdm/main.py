"""
Toxic text generation with an MDLM base model.

Every method is the same SMC loop (Eq. 5-9 of the paper):
    for t = T to 1:
        x_{t-1} ~ q(x_{t-1} | x_t)                 [propose, q = base model]
        log w  = psi_{t-1}(x_{t-1}) - psi_t(x_t)   [weight]
        a_k ~ Cat(w_1, ..., w_K)                   [resample]

What changes between the code paths is only how psi is obtained:
    smc : psi is the reward twist of Eq. (7), estimated with M x0-predictions per step.
    cdm : psi_theta is a trained twist head (CDM training is contrastive twist learning).
          With twist_ckpt=... the same loop runs with the trained head loaded.
    K=1 : no resampling happens at all, which is the unguided base model.

Usage:
  python -m cdm.texts_mdm.main --config-name smc
  python -m cdm.texts_mdm.main --config-name cdm
  python -m cdm.texts_mdm.main --config-name cdm twist_ckpt=<path>.pt
"""

import copy
import itertools
import json
import math
import os
import sys
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# The vendored MDLM modules (diffusion/dataloader/models/noise_schedule/utils) import each other by
# bare module name, and the released MDLM checkpoint is unpickled against those names.
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

import dataloader as mdlm_dataloader
import diffusion

from cdm.texts_mdm.cdm_buffer import CDMPosBuffer
from cdm.texts_mdm.rewards import evaluate_generation, heldout_reward
from cdm.texts_mdm.samplers import MDLMSampler, _sample_categorical
from cdm.texts_mdm.twist_model import MergedFrozenBackboneTwist
from cdm.texts_mdm.utils import EMA, ess_normalized, ess_summary, make_linear_decay_scheduler
from cdm.utils import seed_everything

_LOG_FILE = None


def setup_logging(log_path):
    global _LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _LOG_FILE = open(log_path, "a")


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    if _LOG_FILE is not None:
        print(*args, **kwargs, file=_LOG_FILE, flush=True)


def load_prompts(prompt_file):
    if prompt_file is None:
        return [None]
    with open(prompt_file, "r") as f:
        return [json.loads(line)["context_string"] for line in f]


def tokenize_prompt(prompt_text, tokenizer, device):
    if prompt_text is None:
        return None, 0
    tokens = tokenizer(prompt_text, return_tensors="pt", padding=False)
    prompt_ids = tokens["input_ids"][0, :-1].to(device)
    return prompt_ids, len(prompt_ids)


def load_mdlm_model(config, device="cuda"):
    tokenizer = mdlm_dataloader.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(config.eval.checkpoint_path, tokenizer=tokenizer, config=config, weights_only=False).to(device)
    if model.ema:
        model.ema.copy_to(itertools.chain(model.backbone.parameters(), model.noise.parameters()))
        model.ema = None
    return model, tokenizer


def _save_twist(twist_net, args, vocab_size, epoch, loss, filename):
    payload = {
        "twist_net": twist_net.head.state_dict(),
        "config": dict(
            vocab_size=vocab_size,
            head_type=twist_net.head_type,
            mlp_n_layers=twist_net.mlp_n_layers,
            mlp_hidden_size=twist_net.mlp_hidden_size,
        ),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(payload, os.path.join(args.save_path, "ckpt", filename))


def make_twist(args, model, head_type="mlp", mlp_n_layers=None, mlp_hidden_size=None):
    return MergedFrozenBackboneTwist(
        backbone=model.backbone,
        sigma_processor=model._process_sigma,
        mask_index=model.mask_index,
        neg_infinity=model.neg_infinity,
        head_type=head_type,
        mlp_n_layers=args.twist_head_mlp_n_layers if mlp_n_layers is None else mlp_n_layers,
        mlp_hidden_size=args.twist_head_mlp_hidden_size if mlp_hidden_size is None else mlp_hidden_size,
    ).to(args.device)


def prepare_twist_training(args, model):
    twist_net = make_twist(args, model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, twist_net.parameters()), lr=args.twist_lr, weight_decay=args.twist_weight_decay)
    return twist_net, optimizer


def load_twist_checkpoint(args, model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    twist_net = make_twist(args, model, head_type=cfg["head_type"], mlp_n_layers=cfg["mlp_n_layers"], mlp_hidden_size=cfg["mlp_hidden_size"])
    twist_net.head.load_state_dict(ckpt["twist_net"])
    twist_net.eval()
    return twist_net


@torch.no_grad()
def _run_smc(args, base_model, sampler, score_fn, tokenizer, batch_size=None, cdm_smc_chunk_size=None,
             stop_t=None, do_not_return_rewards=False, do_not_return_x0=False, ess_threshold=1.0):
    B = args.twist_batch_size if batch_size is None else batch_size
    chunk_size = args.twist_batch_size if cdm_smc_chunk_size is None else cdm_smc_chunk_size
    gumbel_chunk_size = args.get("gumbel_chunk_size", None)
    total = B
    device = args.device
    seq_len = args.seq_len

    x = sampler.mask_index * torch.ones(total, seq_len, dtype=torch.int64, device=device)
    timesteps, dt = sampler._get_timesteps()

    p_x0_cache = None
    v_cache = None
    log_w_acc = torch.zeros(total, device=device)

    ess_steps = list()
    num_resampled = torch.zeros(B, device=device)
    num_runnned_steps = 0

    for i in range(sampler.num_steps + 1):
        t = timesteps[i] * torch.ones(total, 1, device=device)
        v_curr = torch.ones(total, device=device) if v_cache is None else v_cache

        if i == sampler.num_steps or i == stop_t:
            break

        x_chunks = list()
        for cur_idx in range(0, total, chunk_size):
            cur_chunk_size = min(chunk_size, total - cur_idx)
            x_chunk = x[cur_idx:cur_idx + cur_chunk_size]
            t_chunk = t[cur_idx:cur_idx + cur_chunk_size]
            p_x0_cache_chunk = p_x0_cache[cur_idx:cur_idx + cur_chunk_size] if p_x0_cache is not None else None

            x_chunk_res, _ = sampler._ddpm_cache_step(base_model, x_chunk, t_chunk, dt, p_x0_cache_chunk, gumbel_chunk_size=gumbel_chunk_size)
            x_chunks.append(x_chunk_res)
        x = torch.cat(x_chunks, dim=0)

        t_next = timesteps[i + 1] * torch.ones(total, 1, device=device)
        sigma_next, _ = sampler.noise(t_next)
        if sigma_next.ndim > 1:
            sigma_next = sigma_next.squeeze(-1)

        v_chunks, p_x0_list = list(), list()
        for cur_idx in range(0, total, chunk_size):
            cur_chunk_size = min(chunk_size, total - cur_idx)
            x_chunk = x[cur_idx:cur_idx + cur_chunk_size]
            sigma_chunk = sigma_next[cur_idx:cur_idx + cur_chunk_size]
            v_chunk, p_x0s = score_fn(x_chunk, sigma_chunk)
            v_chunks.append(v_chunk)
            p_x0_list.append(p_x0s)
        v_next = torch.cat(v_chunks, dim=0)
        p_x0_cache = torch.cat(p_x0_list, dim=0)

        log_w_acc = log_w_acc + (v_next - v_curr)

        ess = ess_normalized(log_w_acc, dim=0)
        ess_steps.append(ess)

        if ess < ess_threshold:
            num_resampled += 1
            w = torch.softmax(log_w_acc, dim=0)
            indices = torch.multinomial(w, total, replacement=True)
            x = x[indices]
            v_cache = v_next[indices]
            p_x0_cache = p_x0_cache[indices]
            log_w_acc = torch.zeros(total, device=device)
        else:
            v_cache = v_next

        num_runnned_steps += 1

    if not do_not_return_x0 or not do_not_return_rewards:
        x0_chunks = list()
        for cur_idx in range(0, total, chunk_size):
            cur_chunk_size = min(chunk_size, total - cur_idx)
            x_chunk = x[cur_idx:cur_idx + cur_chunk_size]
            x0_chunk = sampler._noise_removal(base_model, x_chunk)
            x0_chunks.append(x0_chunk)
        x0_all = torch.cat(x0_chunks, dim=0)
    else:
        x0_all = None

    rewards_all = None
    if not do_not_return_rewards:
        rewards_all = evaluate_generation(x0_all[:, :args.gen_length], tokenizer, args)
        rewards_all = torch.tensor(rewards_all, device=device, dtype=torch.float32)

    if num_runnned_steps == 0:
        ratio_resampled = 0.0
    else:
        ratio_resampled = num_resampled / num_runnned_steps

    return (rewards_all, v_curr, log_w_acc), x0_all, (ess_steps, ratio_resampled)


def _ess_normalized(weights):
    N = weights.shape[0]
    return (1.0 / (N * weights.float().pow(2).sum())).item()


@torch.no_grad()
def _collect_pos_samples(args, sampler, model, tokenizer, batch_size=None, cdm_smc_chunk_size=None):
    B = args.twist_batch_size if batch_size is None else batch_size
    alpha = args.kl_weight

    def score_fn(x, sigma):
        return _score(model, sampler, x, sigma, tokenizer, args, twist_net=None, score_type="x0_pred",
                      M=args.twist_M, chunk_size=args.twist_x0_pred_batch_size)

    rewards_all_and_v_curr, x0_all, extras_ = _run_smc(args, model, sampler, score_fn, tokenizer, batch_size=B,
                                                       cdm_smc_chunk_size=cdm_smc_chunk_size, ess_threshold=args.pos_ess_threshold)

    rewards_all, v_curr, log_w_acc = rewards_all_and_v_curr
    ess_steps, ratio_resampled = extras_
    ess_pair = ess_summary(ess_steps)

    samples = x0_all
    log_W = (rewards_all / alpha - v_curr + log_w_acc).detach()
    W_bar = F.softmax(log_W.float(), dim=0).detach()
    return samples, W_bar, (ess_pair, ratio_resampled)


def compute_pos_loss(model, samples, W_bar, twist_net, t, sigma_t):
    x0 = samples.detach()
    move_chance = t
    x_t_pos = model.q_xt(x0, move_chance)
    log_psi_pos = twist_net(x_t_pos, sigma_t, get_only_value=True)
    pos_term = (W_bar * log_psi_pos).sum()
    return pos_term, _ess_normalized(W_bar)


@torch.no_grad()
def _collect_neg_samples(args, base_model, sampler, stop_t=None):
    traj_states = sampler.sample(base_model, args.twist_batch_size, args.seq_len, return_traj=True, stop_t=stop_t)[1]
    return traj_states, (None, None), None


def compute_neg_loss(traj, idx, twist_net, sigma_t, ema_model=None):
    log_psi_neg_ema = ema_model(traj[idx].detach(), sigma_t, get_only_value=True).clone().detach()
    log_psi_neg = twist_net(traj[idx].detach(), sigma_t, get_only_value=True)
    weights = F.softmax(log_psi_neg_ema.float(), dim=0)
    neg_term = (weights * log_psi_neg).sum()
    return neg_term, _ess_normalized(weights)


def train_cdm(args, model, tokenizer, sampler):
    print_log("=" * 100)
    device = args.device
    twist_net, optimizer = prepare_twist_training(args, model)

    ema_list = list(args.ema_decay)
    emas = list()
    for ema_decay in ema_list:
        emas.append(EMA(twist_net, decay=ema_decay))
    ema = emas[0]

    twist_net_old = ema.shadow

    total_steps = args.twist_epochs * args.twist_steps_per_epoch
    scheduler, _ = make_linear_decay_scheduler(optimizer, total_steps, args.twist_lr_decay_start_frac)

    total_params = sum(p.numel() for p in twist_net.parameters()) / 1e6
    trainable_params_num = sum(p.numel() for p in twist_net.parameters() if p.requires_grad) / 1e6
    print_log(f"Total Parameters: {total_params:,} M")
    print_log(f"Trainable Parameters: {trainable_params_num:,} M")

    best_loss = 1e9
    best_model = None
    global_step = 0
    checkpoint_log = []
    best_epoch = -1

    timesteps, _ = sampler._get_timesteps()
    pos_buffer = CDMPosBuffer(args)

    pbar = tqdm(range(args.twist_epochs), desc="CDM", total=args.twist_epochs, leave=True, dynamic_ncols=True)
    for epoch in pbar:
        # One epoch == one positive-buffer refresh; twist_steps_per_epoch is n_update in the paper.
        twist_net.eval()
        pos_buffer.clear()
        pos_buffer.fill(sampler, model, tokenizer, collect_fn=_collect_pos_samples)
        twist_net.train()

        pos_smc_ess_mean = pos_buffer.smc_ess_mean
        pos_smc_ess_min = pos_buffer.smc_ess_min
        pos_resampled_ratio = pos_buffer.ratio_resampled

        losses = []
        for step_idx in tqdm(range(args.twist_steps_per_epoch), desc="Steps", total=args.twist_steps_per_epoch, leave=False):
            twist_net.eval()
            random_sampled_t = int(torch.randint(1, args.num_timesteps + 1, (1,), device=args.device).item())
            with torch.no_grad():
                samples_pos, W_bar = pos_buffer.sample(args.twist_batch_size, device)
                traj_neg, neg_smc_ess, neg_resampled_ratio = _collect_neg_samples(args, model, sampler, stop_t=random_sampled_t)
                neg_smc_ess_mean, neg_smc_ess_min = neg_smc_ess

            twist_net.train()
            optimizer.zero_grad()
            total_loss = 0.0
            neg_ess_vals = list()
            pos_ess_vals = list()

            t_idx = random_sampled_t
            t = timesteps[t_idx] * torch.ones(args.twist_batch_size, 1, device=device)
            sigma_t, _ = sampler.noise(t)
            if sigma_t.ndim > 1:
                sigma_t = sigma_t.squeeze(-1)

            pos_term, pos_ess = compute_pos_loss(model, samples_pos, W_bar, twist_net, t, sigma_t)
            neg_term, neg_ess = compute_neg_loss(traj_neg, t_idx, twist_net, sigma_t, twist_net_old)
            if neg_ess is not None:
                neg_ess_vals.append(neg_ess)
            if pos_ess is not None:
                pos_ess_vals.append(pos_ess)
            step_loss = neg_term - pos_term

            step_loss.backward()
            total_loss += step_loss.item()

            trainable_params = [p for p in twist_net.parameters() if p.requires_grad]
            if args.twist_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(args.twist_clip_grad_norm)).item()
            else:
                grad_norm = torch.nn.utils.get_total_norm([p.grad for p in trainable_params if p.grad is not None]).item()

            optimizer.step()
            scheduler.step() if scheduler is not None else None
            for ema_, cur_ema_decay_rate in zip(emas, ema_list):
                ema_.update(twist_net, decay=cur_ema_decay_rate)

            losses.append(total_loss)
            if wandb.run is not None:
                log_dict = {"cdm/loss": total_loss, "cdm/grad_norm": grad_norm}
                if neg_ess_vals:
                    log_dict["cdm/neg_last_weight_ess"] = np.mean(neg_ess_vals)
                if pos_ess_vals:
                    log_dict["cdm/pos_last_weight_ess"] = np.mean(pos_ess_vals)

                if neg_smc_ess_min is not None:
                    log_dict["cdm/neg_smc_ess_mean"] = neg_smc_ess_mean
                    log_dict["cdm/neg_smc_ess_min"] = neg_smc_ess_min
                    log_dict["cdm/neg_resampled_ratio"] = neg_resampled_ratio

                if pos_smc_ess_mean is not None:
                    log_dict["cdm/pos_smc_ess_mean"] = pos_smc_ess_mean
                    log_dict["cdm/pos_smc_ess_min"] = pos_smc_ess_min
                    log_dict["cdm/pos_resampled_ratio"] = pos_resampled_ratio

                log_dict["cdm/Neg Term"] = neg_term.item()
                log_dict["cdm/Pos Term"] = pos_term.item()
                wandb.log(log_dict, step=global_step)
            global_step += 1

        avg = np.mean(losses)
        if (epoch + 1) % args.twist_log_every == 0:
            checkpoint_log.append({"epoch": epoch + 1, "loss": avg, "best_loss": best_loss})
            with open(os.path.join(args.save_path, "checkpoint_log.json"), "w") as f:
                json.dump(checkpoint_log, f, indent=2)
            _save_twist(twist_net, args, model.vocab_size, epoch, avg, f"twist_epoch_{epoch+1}.pt")

        if avg < best_loss:
            best_loss = avg
            _save_twist(twist_net, args, model.vocab_size, epoch, best_loss, "twist_best.pt")
            best_model = copy.deepcopy(twist_net_old).to(device)
            best_epoch = epoch + 1

    _save_twist(twist_net, args, model.vocab_size, args.twist_epochs - 1, avg, "twist_final.pt")
    return best_model, best_epoch


def _score(model, sampler, x, sigma, tokenizer, args, twist_net, score_type=None, M=None, chunk_size=None):
    score_type = score_type if score_type is not None else args.score
    M = M if M is not None else args.M
    chunk_size = chunk_size if chunk_size is not None else args.get("m_chunk_size", M)
    alpha = args.kl_weight

    if score_type == "twist":
        logits, score = twist_net(x, sigma, get_also_value=True)
        p_x0 = logits.exp()
        return score, p_x0

    elif score_type == "x0_pred":
        B = x.shape[0]
        p_x0 = model.forward(x, sigma).exp()

        reward_chunks = []
        for m0 in range(0, M, chunk_size):
            m_chunk = min(chunk_size, M - m0)
            p_x0_chunk = p_x0.repeat_interleave(m_chunk, dim=0)
            x_chunk = x.repeat_interleave(m_chunk, dim=0)
            x0_hat = _sample_categorical(p_x0_chunk)
            x0_hat = torch.where(x_chunk == sampler.mask_index, x0_hat, x_chunk)

            rewards_chunk = evaluate_generation(x0_hat[:, : args.gen_length], tokenizer, args)
            rewards_chunk = torch.tensor(rewards_chunk, device=x.device, dtype=torch.float32).reshape(B, m_chunk)
            reward_chunks.append(rewards_chunk)

        rewards = torch.cat(reward_chunks, dim=1)
        rewards = torch.logsumexp(rewards / alpha, dim=-1) - math.log(M)
        return rewards, p_x0

    raise ValueError(f"Unknown score type: {score_type}")


@torch.no_grad()
def sample(model, sampler, batch_size, seq_len, K, tokenizer, args, prompt_ids=None, twist_net=None):
    device = sampler.device
    mask_index = sampler.mask_index
    noise = sampler.noise

    prompt_len = len(prompt_ids) if prompt_ids is not None else 0
    total = batch_size * K

    x = mask_index * torch.ones(total, seq_len, dtype=torch.int64, device=device)
    if prompt_ids is not None:
        x[:, :prompt_len] = prompt_ids

    timesteps, dt = sampler._get_timesteps()
    v_cache = None
    p_x0_next = None
    log_w_accum = torch.zeros(batch_size, K, device=device)
    ess_per_step = []

    for i in tqdm(range(sampler.num_steps), desc="SMC Sampling", leave=False):
        t = timesteps[i] * torch.ones(total, 1, device=device)
        sigma_t, _ = noise(t)
        t_val = t.squeeze(-1)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)

        # 1. Transition probabilities of the proposal q (= base model).
        if p_x0_next is None:
            log_q_x0 = model.forward(x, sigma_t)
            p_x0 = log_q_x0.exp()
        else:
            p_x0 = p_x0_next

        mc_t = t_val[:, None, None]
        mc_s = (t_val - dt)[:, None, None]
        q_xs = p_x0 * (mc_t - mc_s)
        q_xs[:, :, mask_index] = mc_s[:, :, 0]

        # 2. Propose x_{t-1} ~ q(.|x_t).
        _x = _sample_categorical(q_xs)
        copy_flag = (x != mask_index).to(x.dtype)
        x_next = copy_flag * x + (1 - copy_flag) * _x
        if prompt_ids is not None:
            x_next[:, :prompt_len] = prompt_ids

        # 3. Weight and resample. K=1 leaves this out, which is the unguided base model.
        if K > 1:
            t_next_val = timesteps[i + 1] if (i + 1) < len(timesteps) else sampler.eps
            sigma_next = noise(t_next_val * torch.ones(total, 1, device=device))[0]
            if sigma_next.ndim > 1:
                sigma_next = sigma_next.squeeze(-1)

            v_next, p_x0_next = _score(model, sampler, x_next, sigma_next, tokenizer, args, twist_net)
            v_curr = torch.ones_like(v_next) if v_cache is None else v_cache

            # Weights accumulate across steps and reset only when we resample; without this the
            # incremental weight would be dropped on every step we skip.
            log_w_accum = log_w_accum + (v_next - v_curr).reshape(batch_size, K)

            log_w = log_w_accum - log_w_accum.max(dim=1, keepdim=True)[0]
            w = torch.exp(log_w)
            w = w / w.sum(dim=1, keepdim=True)

            # Compute ESS independently per batch element, retaining the
            # batch mean for the existing diagnostic trace.
            ess_per_batch = 1.0 / (w ** 2).sum(dim=1)
            ess = ess_per_batch.mean().item()
            ess_per_step.append(ess)

            should_resample = ess_per_batch < args.ess_threshold * K

            if should_resample.any():
                # Rows above the threshold keep their particles in place;
                # sample ancestors only for rows below the threshold.
                indices = torch.arange(K, device=device)[None, :].expand(
                    batch_size, -1,
                ).clone()
                indices[should_resample] = torch.multinomial(
                    w[should_resample], K, replacement=True,
                )
                batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, K)

                x_next = x_next.reshape(batch_size, K, seq_len)
                x_next = x_next[batch_idx, indices].reshape(total, seq_len)

                v_cache = v_next.reshape(batch_size, K)
                v_cache = v_cache[batch_idx, indices].reshape(total)

                cat_dim = p_x0_next.shape[-1]
                p_x0_next = p_x0_next.reshape(batch_size, K, seq_len, cat_dim)
                p_x0_next = p_x0_next[batch_idx, indices].reshape(total, seq_len, cat_dim)

                # Only resampled rows return to uniform accumulated weights.
                log_w_accum[should_resample] = 0.0
            else:
                v_cache = v_next
        x = x_next

    x = sampler._noise_removal(model, x)
    if prompt_ids is not None:
        x[:, :prompt_len] = prompt_ids

    # Return the highest-reward particle of each group.
    if K > 1:
        rewards = evaluate_generation(x[:, : args.gen_length], tokenizer, args)
        rewards_2d = torch.tensor(rewards, device=device).reshape(batch_size, K)
        best = rewards_2d.argmax(dim=1)
        x = x.reshape(batch_size, K, seq_len)
        x = x[torch.arange(batch_size, device=device), best]
    return x


def evaluate(args, model, tokenizer, sampler, tag, twist_net=None):
    model.eval()
    prompts = load_prompts(args.prompt_file)

    original_gen_length = args.gen_length
    all_results, all_rewards = list(), list()

    start = time.time()
    outter_pbar = tqdm(enumerate(prompts), desc="Prompts", total=len(prompts), leave=False)
    for pi, prompt_text in outter_pbar:
        prompt_ids, prompt_len = tokenize_prompt(prompt_text, tokenizer, args.device)
        args.gen_length = min(args.reward_trim_length + prompt_len, args.seq_len) if prompt_ids is not None else original_gen_length
        p_rewards, p_conts, generated = list(), list(), 0

        with tqdm(total=args.num_sample_batches, desc=f"Prompt {pi+1}/{len(prompts)}", leave=False) as pbar:
            while generated < args.num_sample_batches:
                bs = min(args.batch_size, args.num_sample_batches - generated)
                x = sample(model, sampler, bs, args.seq_len, args.K, tokenizer, args, prompt_ids, twist_net=twist_net)
                rewards = evaluate_generation(x[:, : args.gen_length], tokenizer, args)

                for j in range(x.shape[0]):
                    start_idx = prompt_len if prompt_ids is not None else 0
                    p_conts.append(tokenizer.decode(x[j, start_idx:], skip_special_tokens=True))
                    p_rewards.append(rewards[j])

                generated += bs
                pbar.update(bs)

        all_rewards.extend(p_rewards)
        all_results.append(dict(context_string=prompt_text or "", string=p_conts, rewards=p_rewards))

        r = np.array(p_rewards)
        outter_pbar.set_postfix(Avg=f"{r.mean():.4f}")
    elapsed = time.time() - start

    gen_path = os.path.join(args.save_path, "generations", f"generations_{tag}.jsonl")
    with open(gen_path, "w") as f:
        for e in all_results:
            f.write(json.dumps(e) + "\n")

    n_samples = len(all_rewards)
    given = float(np.mean(all_rewards))
    heldout = heldout_reward([f"{e['context_string']}{s}" for e in all_results for s in e["string"]], device=args.device)
    sec_per_sample = elapsed / n_samples

    result = dict(app="toxicity", method=args.method, K=args.K, M=args.M, seed=args.seed,
                  given_reward=round(given, 4), heldout_reward=round(heldout, 4),
                  sec_per_sample=round(sec_per_sample, 4), n_samples=n_samples,
                  twist_ckpt=args.twist_ckpt)
    with open(os.path.join(args.save_path, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print_log("[RESULT] " + " ".join(f"{k}={v}" for k, v in result.items()))
    return result


@hydra.main(config_path="configs", config_name="smc", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("gen_length") is None:
        cfg.gen_length = cfg.seq_len
    cfg.model.length = cfg.seq_len
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
    os.makedirs(os.path.join(cfg.save_path, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_path, "generations"), exist_ok=True)
    setup_logging(os.path.join(cfg.save_path, "run.log"))

    model, tokenizer = load_mdlm_model(cfg, device=cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    sampler = MDLMSampler(model, num_steps=cfg.num_timesteps)
    twist_net = None

    if not cfg.get("disable_wandb", True):
        wandb.init(project=cfg.get("wandb_name") or "cdm", name=tag, config=OmegaConf.to_container(cfg, resolve=True))

    if train_cdm_twist:
        train_start = time.time()
        twist_net, best_epoch = train_cdm(cfg, model, tokenizer, sampler)
        print_log(f"CDM training time: {time.time() - train_start:.1f}s (best epoch {best_epoch})")
        twist_net = load_twist_checkpoint(cfg, model, os.path.join(cfg.save_path, "ckpt", "twist_best.pt"))
    elif cfg.method == "cdm":
        twist_net = load_twist_checkpoint(cfg, model, cfg.twist_ckpt)
        print_log(f"Loaded twist from {cfg.twist_ckpt}")

    with open(os.path.join(cfg.save_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    evaluate(cfg, model, tokenizer, sampler, tag, twist_net=twist_net)


if __name__ == "__main__":
    main()
