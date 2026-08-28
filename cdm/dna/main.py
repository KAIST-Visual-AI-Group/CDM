"""
Regulatory DNA sequence design with a CNN-parameterized MDLM.

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
  python -m cdm.dna.main --config-name smc
  python -m cdm.dna.main --config-name cdm
  python -m cdm.dna.main --config-name cdm twist_ckpt=<path>.pt
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
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import cdm.dna.dataloader as dna_dataloader
import cdm.dna.diffusion as diffusion
from cdm.dna.cdm_utils import (
    CDMPosBuffer,
    _collect_neg_samples,
    compute_neg_loss,
    compute_pos_loss,
)
from cdm.dna.rewards import _get_gosai_oracle, evaluate_generation
from cdm.dna.samplers import MDLMSampler, _sample_categorical
from cdm.dna.twist_model import FrozenBackboneTwist
from cdm.dna.utils import EMA, make_linear_decay_scheduler
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


def load_dna_model(config, device="cuda"):
    """Load the pre-trained DNA diffusion model.

    weights_only=False because the duo checkpoint pickles OmegaConf and tokenizer objects; the
    dataloader module is aliased so pickle can resolve ``dataloader.MPRATokenizer``.
    """
    sys.modules.setdefault('dataloader', dna_dataloader)

    tokenizer = dna_dataloader.get_tokenizer(config)
    model = diffusion.Diffusion(config=config, tokenizer=tokenizer)
    ckpt = torch.load(config.eval.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    if "ema" in ckpt and model.ema is not None:
        model.ema.load_state_dict(ckpt["ema"])
    model = model.to(device)
    if model.ema:
        model.ema.copy_to(
            itertools.chain(model.backbone.parameters(), model.noise.parameters())
        )
        model.ema = None
    return model, tokenizer


def make_twist(args, model, head_type=None, mlp_n_layers=None, mlp_hidden_size=None, n_heads=None):
    return FrozenBackboneTwist(
        backbone=model.backbone,
        sigma_processor=model._process_sigma,
        head_type=args.twist_head_type if head_type is None else head_type,
        mlp_n_layers=args.twist_head_mlp_n_layers if mlp_n_layers is None else mlp_n_layers,
        mlp_hidden_size=args.twist_head_mlp_hidden_size if mlp_hidden_size is None else mlp_hidden_size,
        n_heads=args.twist_head_n_heads if n_heads is None else n_heads,
    ).to(args.device)


def _save_twist(twist_net, args, vocab_size, epoch, loss, filename):
    payload = {
        "twist_net": twist_net.head.state_dict(),
        "config": dict(
            vocab_size=vocab_size,
            head_type=twist_net.head_type,
            mlp_n_layers=twist_net.mlp_n_layers,
            mlp_hidden_size=twist_net.mlp_hidden_size,
            n_heads=twist_net.n_heads,
        ),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(payload, os.path.join(args.save_path, "ckpt", filename))


def load_twist_checkpoint(args, model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    twist_net = make_twist(
        args, model,
        head_type=cfg["head_type"], mlp_n_layers=cfg["mlp_n_layers"],
        mlp_hidden_size=cfg["mlp_hidden_size"], n_heads=cfg["n_heads"],
    )
    twist_net.head.load_state_dict(ckpt["twist_net"])
    twist_net.eval()
    return twist_net


def prepare_twist_training(args, model):
    twist_net = make_twist(args, model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, twist_net.parameters()),
        lr=args.twist_lr,
        weight_decay=args.twist_weight_decay,
    )
    return twist_net, optimizer


def train_cdm(args, model, tokenizer, sampler):
    """Train psi via CDM.

    Positive samples come from a per-epoch buffer filled by bootstrapped-value SMC; negative
    samples come from an IS (or twist-guided SMC) chain in which the true reward never enters
    the weights. An EMA of the twist net is refreshed at the end of every epoch and used for
    all pos/neg sampling during the next one.
    """
    device = args.device

    twist_net, optimizer = prepare_twist_training(args, model)
    ema = EMA(twist_net, decay=args.cdm_ema_decay)

    total_steps = args.twist_epochs * args.twist_steps_per_epoch
    scheduler, _ = make_linear_decay_scheduler(optimizer, total_steps, args.twist_lr_decay_start_frac)

    total_params = sum(p.numel() for p in twist_net.parameters()) / 1e6
    trainable_params_num = sum(p.numel() for p in twist_net.parameters() if p.requires_grad) / 1e6
    print_log(f"Total Parameters: {total_params:,} M")
    print_log(f"Trainable Parameters: {trainable_params_num:,} M")

    best_neg_reward = -1e9
    best_model = None
    global_step = 0
    checkpoint_log = []
    timesteps, _ = sampler._get_timesteps()
    pos_buffer = CDMPosBuffer(args)
    avg = float("inf")

    for epoch in tqdm(range(args.twist_epochs), desc="CDM"):
        # Refill the positive buffer once per epoch; scoring uses the EMA head.
        twist_net.eval()
        pos_buffer.clear()
        pos_buffer.fill(sampler, model, tokenizer, args.kl_weight, twist_net=ema.shadow)
        twist_net.train()

        pos_smc_ess_mean = pos_buffer.smc_ess_mean
        pos_smc_ess_min = pos_buffer.smc_ess_min

        losses, neg_rewards = [], []
        pos_final_ess_list, neg_final_ess_list = [], []

        for _ in range(args.twist_steps_per_epoch):
            twist_net.eval()
            with torch.no_grad():
                pos_batch = pos_buffer.sample(args.twist_batch_size, device)
                traj_neg, rewards_neg, weights_neg_per_step, _ = _collect_neg_samples(
                    args, sampler, model, ema.shadow, tokenizer,
                    num_seqs=args.twist_batch_size,
                )
            twist_net.train()

            optimizer.zero_grad()
            t_idx = int(torch.randint(0, args.num_timesteps + 1, (1,), device=device).item())

            batch_t = pos_batch.rewards.shape[0]
            t = timesteps[t_idx] * torch.ones(batch_t, 1, device=device)
            _, sigma_t = sampler._get_alpha_sigma(t)

            pos_term, pos_final_ess = compute_pos_loss(args, model, pos_batch, twist_net, sigma_t, t_idx)

            # Neg: the gradient flows through the live twist_net; the IS weights come from the
            # frozen EMA head so the negative phase is not self-referential.
            batch_n = traj_neg[0].shape[0]
            t_n = timesteps[t_idx] * torch.ones(batch_n, 1, device=device)
            _, sigma_t_n = sampler._get_alpha_sigma(t_n)
            neg_term, neg_final_ess = compute_neg_loss(
                traj_neg, twist_net, sigma_t_n, t_idx,
                weights_per_step=weights_neg_per_step,
                head_weights=ema.shadow,
            )
            pos_final_ess_list.append(pos_final_ess)
            neg_final_ess_list.append(neg_final_ess)

            step_loss = neg_term - pos_term
            step_loss.backward()
            total_loss = step_loss.item()

            trainable_params = [p for p in twist_net.parameters() if p.requires_grad]
            if args.twist_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.twist_clip_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in trainable_params if p.grad is not None]
                ).item()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(total_loss)
            neg_rewards.append(rewards_neg.mean().item())

            if wandb.run is not None:
                wandb.log({
                    "cdm/loss": total_loss,
                    "cdm/reward_pos_mean": pos_batch.rewards.mean().item(),
                    "cdm/reward_neg_mean": rewards_neg.mean().item(),
                    "cdm/grad_norm": grad_norm,
                    "cdm/pos_smc_ess_mean": pos_smc_ess_mean,
                    "cdm/pos_smc_ess_min": pos_smc_ess_min,
                }, step=global_step)
            global_step += 1

        # EMA refresh at the end of each epoch, not per step.
        ema.update(twist_net)

        avg = np.mean(losses)
        avg_neg_reward = np.mean(neg_rewards)

        if (epoch + 1) % args.twist_log_every == 0:
            print_log(
                f"  Epoch {epoch+1}/{args.twist_epochs} | Loss: {avg:.6f} | "
                f"Neg Reward: {avg_neg_reward:.6f} | "
                f"pos_final_ess: {float(np.mean(pos_final_ess_list)):.3f} | "
                f"neg_final_ess: {float(np.mean(neg_final_ess_list)):.3f} | "
                f"pos_smc_ess: {pos_smc_ess_mean:.3f}/{pos_smc_ess_min:.3f}"
            )
            checkpoint_log.append({"epoch": epoch + 1, "loss": float(avg),
                                   "neg_reward": float(avg_neg_reward)})
            with open(os.path.join(args.save_path, "checkpoint_log.json"), "w") as f:
                json.dump(checkpoint_log, f, indent=2)
            _save_twist(twist_net, args, model.vocab_size, epoch, avg, f"twist_epoch_{epoch+1}.pt")
            _save_twist(ema.shadow, args, model.vocab_size, epoch, avg, f"twist_epoch_{epoch+1}_ema.pt")

        if avg_neg_reward > best_neg_reward:
            best_neg_reward = avg_neg_reward
            _save_twist(twist_net, args, model.vocab_size, epoch, best_neg_reward, "twist_best.pt")
            _save_twist(ema.shadow, args, model.vocab_size, epoch, best_neg_reward, "twist_best_ema.pt")
            best_model = copy.deepcopy(twist_net).to(device)

    _save_twist(twist_net, args, model.vocab_size, args.twist_epochs - 1, avg, "twist_final.pt")
    _save_twist(ema.shadow, args, model.vocab_size, args.twist_epochs - 1, avg, "twist_final_ema.pt")
    print_log(f"[*] Best CDM neg reward: {best_neg_reward:.6f}")
    return best_model


def _score(model, sampler, x, sigma, tokenizer, args, twist_net):
    """Score particles for SMC reweighting.

    Returns ``(value, log_p_x0_or_None)``. ``log_p_x0`` is returned by the ``x0_pred`` path and
    cached by the caller as the next step's proposal kernel, since the next iteration's
    (x, sigma_t) equals this iteration's (x_next, sigma_next).
    """
    B = x.shape[0]
    b_chunk = args.b_chunk_size or B

    def _sigma_slice(b0, b1):
        if torch.is_tensor(sigma) and sigma.dim() > 0 and sigma.shape[0] == B:
            return sigma[b0:b1]
        return sigma

    if args.score == "twist":
        assert twist_net is not None
        parts = []
        for b0 in range(0, B, b_chunk):
            b1 = min(b0 + b_chunk, B)
            parts.append(twist_net(x[b0:b1], _sigma_slice(b0, b1)))
        return torch.cat(parts, dim=0), None

    elif args.score == "x0_pred":
        M = args.M
        log_p_x0_parts = []
        for b0 in range(0, B, b_chunk):
            b1 = min(b0 + b_chunk, B)
            log_p_x0_parts.append(model.forward(x[b0:b1], _sigma_slice(b0, b1)))
        log_p_x0 = torch.cat(log_p_x0_parts, dim=0)
        p_x0 = log_p_x0.exp()

        p_x0_all = p_x0.repeat_interleave(M, dim=0)
        x_all = x.repeat_interleave(M, dim=0)
        x0_hat = _sample_categorical(p_x0_all)
        x0_hat = torch.where(x_all == sampler.mask_index, x0_hat, x_all)

        rewards_list = evaluate_generation(x0_hat[:, :args.gen_length], tokenizer, args)
        rewards = torch.tensor(rewards_list, device=x.device, dtype=torch.float32).reshape(B, M)
        rewards = torch.logsumexp(rewards / args.kl_weight, dim=-1) - math.log(M)
        return rewards, log_p_x0

    raise ValueError(f"Unknown score: {args.score}")


@torch.no_grad()
def sample(model, sampler, batch_size, seq_len, K, tokenizer, args, twist_net=None):
    device = sampler.device
    mask_index = sampler.mask_index
    total = batch_size * K

    x = mask_index * torch.ones(total, seq_len, dtype=torch.int64, device=device)

    timesteps, dt = sampler._get_timesteps()
    v_cache = None
    logits_cache = None  # cached log p(x0|x_t) from the scoring pass

    # Accumulated log-weights for adaptive resampling.  Each batch row
    # accumulates incremental log-weights until that row resamples, at
    # which point only its accumulator resets to zero.
    log_w_accum = torch.zeros(batch_size, K, device=device)
    ess_per_step = list()

    # With a trained twist the backbone is shared, so one fused pass yields both the logits for
    # the next denoising step and the twist value for resampling.
    fuse_twist = args.score == "twist" and twist_net is not None and K > 1

    for i in tqdm(range(sampler.num_steps), desc=f"{args.method} Sampling", leave=False):
        t = timesteps[i] * torch.ones(total, 1, device=device)
        alpha_t, sigma_t = sampler._get_alpha_sigma(t)
        t_s = (t.squeeze(-1) - dt)
        alpha_s, _ = sampler._get_alpha_sigma(t_s)

        # 1. log p(x0 | x_t), reusing the cache when the scoring pass already produced it.
        if logits_cache is not None:
            log_q_x0 = logits_cache
        else:
            log_q_x0 = model.forward(x, sigma_t.unsqueeze(-1))
        p_x0 = log_q_x0.exp()

        # 2. Propose x_{t-1} ~ q(.|x_t).
        q_xs = p_x0 * (alpha_s - alpha_t)[:, None, None]
        q_xs[:, :, mask_index] = (1 - alpha_s)[:, None]

        _x = _sample_categorical(q_xs)
        copy_flag = (x != mask_index).to(x.dtype)
        x_next = copy_flag * x + (1 - copy_flag) * _x

        # 3. Weight and resample. K=1 leaves this out, which is the unguided base model.
        if K > 1:
            t_next_val = timesteps[i + 1] if (i + 1) < len(timesteps) else sampler.eps
            _, sigma_next = sampler._get_alpha_sigma(t_next_val * torch.ones(total, device=device))

            if fuse_twist:
                log_p_next, v_next = model.forward_fused(
                    x_next, sigma_next.unsqueeze(-1), twist_net=twist_net,
                )
                v_next = v_next.to(torch.float32)
            else:
                v_next, log_p_next = _score(model, sampler, x_next, sigma_next, tokenizer, args, twist_net)

            # At t=1 every particle is fully masked, so any constant v_curr cancels in the
            # softmax; skip the redundant forward pass.
            v_curr = v_cache if v_cache is not None else torch.ones(total, device=device, dtype=torch.float32)

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

                if log_p_next is not None:
                    _, L_dim, V_dim = log_p_next.shape
                    log_p_next = log_p_next.reshape(batch_size, K, L_dim, V_dim)
                    logits_cache = log_p_next[batch_idx, indices].reshape(total, L_dim, V_dim)
                else:
                    logits_cache = None

                # Only resampled rows return to uniform accumulated weights.
                log_w_accum[should_resample] = 0.0
            else:
                v_cache = v_next
                logits_cache = log_p_next
        else:
            v_cache = None
            logits_cache = None

        x = x_next

    x = sampler._noise_removal(model, x)

    # Return the highest-reward particle of each group.
    if K > 1:
        rewards = evaluate_generation(x[:, :args.gen_length], tokenizer, args)
        rewards_2d = torch.tensor(rewards, device=device).reshape(batch_size, K)
        best = rewards_2d.argmax(dim=1)
        x = x.reshape(batch_size, K, seq_len)
        x = x[torch.arange(batch_size, device=device), best]
    return x, ess_per_step


def evaluate(args, model, tokenizer, sampler, tag, twist_net=None):
    model.eval()
    all_rewards, all_held_out_rewards, all_seqs = list(), list(), list()
    sample_times, all_ess_traces = list(), list()

    # Heldout reward: a second Enformer trained on the validation split, never used for scaling.
    held_out_oracle = _get_gosai_oracle(mode='eval')

    generated = 0
    with tqdm(total=args.num_sample_batches, desc="Eval", leave=False) as pbar:
        while generated < args.num_sample_batches:
            bs = min(args.batch_size, args.num_sample_batches - generated)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x, ess_trace = sample(model, sampler, bs, args.seq_len, args.K, tokenizer, args, twist_net=twist_net)
            if ess_trace:
                all_ess_traces.append(ess_trace)
            torch.cuda.synchronize()
            sample_times.append((time.perf_counter() - t0, bs))

            rewards = evaluate_generation(x[:, :args.gen_length], tokenizer, args)
            held_out_rewards = evaluate_generation(
                x[:, :args.gen_length], tokenizer, args, oracle_model=held_out_oracle,
            )

            all_seqs.extend(tokenizer.batch_decode(x[:, :args.gen_length]))
            all_rewards.extend(rewards)
            all_held_out_rewards.extend(held_out_rewards)
            generated += bs
            pbar.update(bs)

    with open(os.path.join(args.save_path, "generations", f"generations_{tag}.jsonl"), "w") as f:
        for seq, r, r_ho in zip(all_seqs, all_rewards, all_held_out_rewards):
            f.write(json.dumps({"seq": seq, "reward": r, "held_out_reward": r_ho}) + "\n")

    if all_ess_traces:
        ess_arr = np.array(all_ess_traces)
        with open(os.path.join(args.save_path, "ess.json"), "w") as f:
            json.dump({"K": int(args.K), "mean_per_step": ess_arr.mean(axis=0).tolist(),
                       "std_per_step": ess_arr.std(axis=0).tolist()}, f, indent=2)

    total_sample_time = sum(t for t, _ in sample_times)
    total_samples = sum(b for _, b in sample_times)

    result = dict(app="dna", method=args.method, K=args.K, M=args.M, seed=args.seed,
                  given_reward=round(float(np.mean(all_rewards)), 4),
                  heldout_reward=round(float(np.mean(all_held_out_rewards)), 4),
                  sec_per_sample=round(total_sample_time / max(total_samples, 1), 4),
                  n_samples=len(all_rewards), twist_ckpt=args.twist_ckpt)
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

    print_log(f"[*] Method: {cfg.method} | K: {cfg.K} | Score: {cfg.score}")
    model, tokenizer = load_dna_model(cfg, device=cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    sampler = MDLMSampler(model, num_steps=cfg.num_timesteps)
    twist_net = None

    if not cfg.get("disable_wandb", True):
        wandb.init(project=cfg.get("wandb_name") or "cdm", name=tag,
                   config=OmegaConf.to_container(cfg, resolve=True))

    if train_cdm_twist:
        train_start = time.time()
        twist_net = train_cdm(cfg, model, tokenizer, sampler).eval()
        print_log(f"CDM training time: {time.time() - train_start:.1f}s")
    elif cfg.method == "cdm":
        twist_net = load_twist_checkpoint(cfg, model, cfg.twist_ckpt)
        print_log(f"Loaded twist from {cfg.twist_ckpt}")

    with open(os.path.join(cfg.save_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    evaluate(cfg, model, tokenizer, sampler, tag, twist_net=twist_net)


if __name__ == "__main__":
    main()
