import numpy as np
import torch

from cdm.texts.dist_utils import get_world_size, global_softmax




def prepare_cdm_config_and_tag(cfg):
    """Normalise the interdependent CDM-training knobs and return the run tag."""
    if not cfg.do_pos_multi_smc:
        cfg.cdm_pos_prompt_sample_size = cfg.cdm_buffer_size

    if cfg.match_pos_neg_prompt:
        # Paired estimator: reuse the epoch's positive prompts for the negative phase.
        cfg.cdm_neg_prompt_sample_size = cfg.cdm_pos_prompt_sample_size
    if cfg.cdm_neg_sample_method == "is":
        cfg.cdm_neg_prompt_sample_size = cfg.twist_batch_size
    elif cfg.cdm_neg_sample_method != "multi_is":
        raise NotImplementedError(f"Unknown cdm_neg_sample_method '{cfg.cdm_neg_sample_method}'")

    return f"cdm_train"


def _ess_normalized(weights):
    N = weights.shape[0]
    return (1.0 / (N * weights.float().pow(2).sum().clamp(min=1e-12))).item()

@torch.no_grad()
def run_smc(args, sampler, score_fn, x, query_sz, prompt_texts, attention_mask, **kwargs):
    ws = get_world_size()
    B_in = x.shape[0]

    run_smc_size = getattr(args, "run_smc_size", None)
    if run_smc_size is None or int(run_smc_size) <= 0:
        return _run_smc(args, sampler, score_fn, x, query_sz, prompt_texts, attention_mask, **kwargs)

    chunk_b = int(run_smc_size) // ws
    assert chunk_b > 0, f"run_smc_size ({run_smc_size}) must be >= world size ({ws})"
    assert B_in % chunk_b == 0, f"Per-rank input size ({B_in}) must be divisible by run_smc_size//ws ({chunk_b})"
    num_chunks = B_in // chunk_b

    if num_chunks == 1:
        return _run_smc(args, sampler, score_fn, x, query_sz, prompt_texts, attention_mask, **kwargs)

    all_traj_states, all_traj_weights, all_ess_steps = None, None, None
    all_rewards, all_v_curr, all_log_w_acc, all_x, all_prompt_texts_resampled, all_attention_mask_resampled, all_ratio_resampled = list(), list(), list(), list(), list(), list(), list()

    rewards_present = True
    for c in range(num_chunks):
        s, e = c * chunk_b, (c + 1) * chunk_b
        traj_pair, rwlw, x_out, ptr_out, am_out, exras = _run_smc(args, sampler, score_fn, x[s:e], query_sz, prompt_texts[s:e], attention_mask[s:e], **kwargs)
        traj_states_c, traj_weights_c = traj_pair
        rewards_c, v_curr_c, log_w_acc_c = rwlw
        ess_steps_c, ratio_c = exras

        if all_traj_states is None:
            all_traj_states = [[] for _ in traj_states_c]
            all_traj_weights = [[] for _ in traj_weights_c]
            all_ess_steps = [[] for _ in ess_steps_c]
        for i in range(len(traj_states_c)):
            all_traj_states[i].append(traj_states_c[i])
            all_traj_weights[i].append(traj_weights_c[i])
        for i in range(len(ess_steps_c)):
            all_ess_steps[i].append(ess_steps_c[i])

        if rewards_c is None:
            rewards_present = False
        else:
            all_rewards.append(rewards_c)
        all_v_curr.append(v_curr_c)
        all_log_w_acc.append(log_w_acc_c)
        all_x.append(x_out)
        all_prompt_texts_resampled.extend(ptr_out)
        all_attention_mask_resampled.append(am_out)
        all_ratio_resampled.append(ratio_c)

    traj_states = [torch.cat(ts, dim=0) for ts in all_traj_states] if all_traj_states else []
    traj_weights = [torch.cat(tw, dim=0) for tw in all_traj_weights] if all_traj_weights else []
    ess_steps = [float(np.mean(es)) for es in all_ess_steps] if all_ess_steps else []
    rewards = torch.cat(all_rewards, dim=0) if rewards_present else None
    v_curr = torch.cat(all_v_curr, dim=0)
    log_w_acc = torch.cat(all_log_w_acc, dim=0)
    x_cat = torch.cat(all_x, dim=0)
    attention_mask_cat = torch.cat(all_attention_mask_resampled, dim=0)
    ratio_resampled_avg = float(sum(all_ratio_resampled) / len(all_ratio_resampled)) if all_ratio_resampled else 0.0

    return (traj_states, traj_weights), (rewards, v_curr, log_w_acc), x_cat, all_prompt_texts_resampled, attention_mask_cat, (ess_steps, ratio_resampled_avg)


def _run_smc(args, sampler, score_fn, x, query_sz, prompt_texts, attention_mask, stop_t=None, do_not_return_rewards=False,
             ess_threshold=1.1, return_pre_post="none", do_multi_smc=False, cache_hidden=False):
    # >>> NOTE ON SMC SETTINGS
    """
    For POSITIVE SMC
    do_not_return_rewards: False
    stop_t: None
    -> traj will not be used where only final x0 and calculated rewards will be used

    For NEGATIVE SMC
    do_not_return_rewards: True
    stop_t: [0, steps-1]
    -> x0 will not be used and final reward should not be calculated since Neg samples cannot be at the end of the trajectory.
    -> Currently it returns whole trajectory, but it can be changed to return only the final states at stop_t if needed.
    """
    # <<< NOTE ON SMC SETTINGS
    from cdm.texts.main import (
        _propose_llada_step,
        evaluate_generation
    )
    if do_multi_smc:
        B = x.shape[0]
        K = args.multi_smc_K                        # particles per SMC
        S = args.cdm_buffer_size // args.cdm_pos_prompt_sample_size   # particles to draw per SMC
    else:
        B = x.shape[0]  # single SMC: B = per-rank prompt count from loader
        K = 1

    total = B * K
    traj_states, traj_weights = list(), list()
    log_w_acc = torch.zeros(B, device=args.device)

    if return_pre_post == "pre":
        if not do_multi_smc:
            traj_states.append(x.clone())
            traj_weights.append(log_w_acc.clone())
        else:
            traj_states.append(x.repeat_interleave(S, dim=0))
            traj_weights.append(log_w_acc.repeat_interleave(S, dim=0))

    if do_multi_smc:
        x = x.repeat_interleave(K, dim=0)
        log_w_acc = log_w_acc.repeat_interleave(K, dim=0)

    mask_index = (x == sampler.mask_token)
    num_transfer_tokens = sampler.get_num_transfer_tokens(mask_index, sampler.steps)
    ess_steps = list()
    num_resampled = 0 if not do_multi_smc else torch.zeros(B, device=args.device)
    num_runned_steps = 0
    v_cache = None
    cache = None
    stop_t = sampler.steps if stop_t is None else stop_t

    if not do_multi_smc:
        attention_mask_resampled = attention_mask
        prompt_texts_resampled = prompt_texts
    else:
        attention_mask_resampled = attention_mask.repeat_interleave(K, dim=0)
        prompt_texts_resampled = [p for p in prompt_texts for _ in range(K)]

    if cache_hidden:
        cache = sampler.denoiser(x, attention_mask=attention_mask_resampled, return_logits=False, return_hidden=True)

    for i in range(sampler.steps + 1):
        v_curr = v_cache if v_cache is not None else torch.ones(total, device=args.device)
        if return_pre_post == "post":
            if not do_multi_smc:
                traj_states.append(x.clone())
                traj_weights.append(log_w_acc.clone())
            else:
                log_w_mat_post = log_w_acc.view(B, K)
                w_mat_post = torch.softmax(log_w_mat_post, dim=1)
                traj_idx_post = torch.multinomial(w_mat_post, num_samples=S, replacement=True)
                flat_post_idx = (traj_idx_post + torch.arange(B, device=args.device)[:, None] * K).view(-1)
                traj_states.append(x[flat_post_idx].clone())
                traj_weights.append(torch.zeros(B*S, device=args.device))

        if i == stop_t:
            break

        if cache_hidden:
            assert cache is not None, "caching hidden state always have cache"
            x = _propose_llada_step(sampler, x, num_transfer_tokens, i, args.remasking, args.chunk_b_size, attention_mask_resampled, cached_logits=None, cached_hidden=cache)
        else:
            x = _propose_llada_step(sampler, x, num_transfer_tokens, i, args.remasking, args.chunk_b_size, attention_mask_resampled, cached_logits=cache, cached_hidden=None)
        if i == sampler.steps - 1:
            continue

        v_next, cache = score_fn(x, prompt_texts_resampled, attention_mask_resampled)
        log_w_acc += (v_next - v_curr)

        if not do_multi_smc:
            w = torch.softmax(log_w_acc, dim=0)
            ess = ((1.0 / w.pow(2).sum(dim=0).clamp(min=1e-12)) / B).item()
            ess_steps.append(ess)

            if return_pre_post == "pre":
                traj_states.append(x.clone())
                traj_weights.append(log_w_acc.clone())

            if ess < ess_threshold:
                num_resampled += 1
                if getattr(args, "pos_resample_method", "multinomial") == "ssp":
                    from cdm.texts.resampling import ssp_resample
                    indices = ssp_resample(w, B)
                else:
                    indices = torch.multinomial(w, B, replacement=True)
                x = x[indices]
                v_cache = v_next[indices]
                cache = cache[indices]
                attention_mask_resampled = attention_mask_resampled[indices]
                indices_cpu = indices.cpu().tolist()
                prompt_texts_resampled = [prompt_texts_resampled[j] for j in indices_cpu]
                log_w_acc = torch.zeros(B, device=args.device)
            else:
                v_cache = v_next

        else:
            log_w_mat = log_w_acc.view(B, K)
            w_mat = torch.softmax(log_w_mat, dim=1)
            ess_per_smc = 1.0 / w_mat.pow(2).sum(dim=1).clamp(min=1e-12)
            ess_per_smc_norm = ess_per_smc / K
            ess_steps.append(ess_per_smc_norm.mean().item())

            if return_pre_post == "pre":
                traj_idx = torch.multinomial(w_mat, num_samples=S, replacement=True)
                flat_traj_idx = (traj_idx + torch.arange(B, device=args.device)[:, None] * K).view(-1)
                traj_states.append(x[flat_traj_idx].clone())
                traj_weights.append(torch.zeros(B*S, device=args.device))

            need_resample_mask = ess_per_smc_norm < ess_threshold
            num_resampled = num_resampled + need_resample_mask.float()
            if need_resample_mask.any():
                row_idx = torch.arange(K, device=args.device).unsqueeze(0).expand(B, K).contiguous()
                if getattr(args, "pos_resample_method", "multinomial") == "ssp":
                    from cdm.texts.resampling import ssp_resample_batch
                    resampled_idx = ssp_resample_batch(w_mat, K)
                else:
                    resampled_idx = torch.multinomial(w_mat, num_samples=K, replacement=True)
                idx_mat = torch.where(need_resample_mask.unsqueeze(1), resampled_idx, row_idx)
                base = torch.arange(B, device=args.device).unsqueeze(1) * K
                flat_idx = (idx_mat + base).view(-1)

                x = x[flat_idx]
                v_cache = v_next[flat_idx]
                cache = cache[flat_idx]

                new_log_w_mat = log_w_mat.clone()
                new_log_w_mat[need_resample_mask] = 0.0
                log_w_acc = new_log_w_mat.view(-1)
            else:
                v_cache = v_next

        num_runned_steps += 1

    rewards = None
    if not do_not_return_rewards:
        rewards = evaluate_generation(x[:, query_sz:], sampler.tokenizer, args, prompt_texts=prompt_texts_resampled)
        rewards = torch.tensor(rewards, device=args.device, dtype=torch.float32)

    if do_multi_smc:
        attention_mask_resampled = attention_mask.repeat_interleave(S, dim=0)
        prompt_texts_resampled = [p for p in prompt_texts for _ in range(S)]

    if num_runned_steps == 0:
        ratio_of_resampled = 0.0
    elif do_multi_smc:
        ratio_of_resampled = (num_resampled / num_runned_steps).mean().item()
    else:
        ratio_of_resampled = num_resampled / num_runned_steps

    return (traj_states, traj_weights), (rewards, v_curr, log_w_acc), x, prompt_texts_resampled, attention_mask_resampled, (ess_steps, ratio_of_resampled)


@torch.no_grad()
def _collect_pos_samples(args, sampler, init_seq, query_sz, prompt_texts, attention_mask, cur_epoch=None):
    from cdm.texts.main import (
        _score
    )
    method = args.cdm_pos_sample_method
    do_multi_smc = args.do_pos_multi_smc
    if method == "stmc":
        switch_to_tsmc_epoch = int(args.cdm_stmc_frac * args.twist_epochs)
        method = "smc" if cur_epoch < switch_to_tsmc_epoch else "twist_smc"

    pos_alpha = float(args.pos_alpha) if getattr(args, "pos_alpha", None) is not None else float(args.kl_weight)

    if method == "is":
        assert False, "Not implemented yet"
    elif method in ("smc", "twist_smc"):
        score_type = "twist" if method == "twist_smc" else "x0_pred"
        def score_fn(x, prompts, attn_mask):
            return _score(sampler, x, None, query_sz, args, prompts, score_type, args.chunk_m_size, attn_mask, logits=None, return_logits=True, cache_hidden=args.cache_hidden, alpha_override=pos_alpha)
    else:
        raise ValueError(f"Unsupported cdm_pos_sample_method: {args.cdm_pos_sample_method}")

    _, rewards_v_curr_log_w, x0, prompt_texts_resampled, attention_mask_resampled, exras = run_smc(args, sampler, score_fn, init_seq, query_sz, prompt_texts,
                                                                                                   attention_mask, ess_threshold=args.pos_ess_threshold, return_pre_post="none",
                                                                                                   do_multi_smc=do_multi_smc, cache_hidden=args.cache_hidden)
    rewards, v_curr, log_w_acc = rewards_v_curr_log_w
    ess_steps, ratio_of_resampled = exras
    ess_pair = (float(np.mean(ess_steps)), float(np.min(ess_steps))) if len(ess_steps) > 0 else (float("nan"), float("nan"))

    reward_mean = float(rewards.mean().item())
    reward_min = float(rewards.min().item())
    reward_max = float(rewards.max().item())
    if do_multi_smc:
        _ws_diag = get_world_size()
        _B_diag = args.cdm_pos_prompt_sample_size // _ws_diag
        _K_diag = args.multi_smc_K
        responses_diag = x0[:, query_sz:].view(_B_diag, _K_diag, -1)
        unique_counts = [int(torch.unique(responses_diag[b], dim=0).shape[0]) for b in range(_B_diag)]
        unique_per_prompt = float(np.mean(unique_counts))
    else:
        unique_per_prompt = 1.0
    reward_stats = (reward_mean, reward_min, reward_max)

    if do_multi_smc:
        ws = get_world_size()
        B = args.cdm_pos_prompt_sample_size // ws
        K = args.multi_smc_K

        norm_mode = getattr(args, "cdm_pos_reward_norm", "none")
        r_mat = rewards.view(B, K)
        if norm_mode == "per_prompt":
            scale = float(getattr(args, "cdm_pos_reward_norm_scale", 1.0))
            r_centered = r_mat - r_mat.mean(dim=1, keepdim=True)
            r_std = r_mat.std(dim=1, keepdim=True).clamp(min=1e-6)
            r_target = (r_centered / r_std) * scale
        elif norm_mode == "none":
            r_target = r_mat / pos_alpha
        else:
            raise ValueError(f"Unknown cdm_pos_reward_norm: '{norm_mode}'")

        log_W = r_target + (-v_curr + log_w_acc).view(B, K)
        w_mat = torch.softmax(log_W, dim=1)
        ess_per_smc = (1.0 / w_mat.pow(2).sum(dim=1).clamp(min=1e-12)) / K
        ess_mean = ess_per_smc.mean().item()
        final_is_ess = ess_mean

        if args.cdm_pos_keep_all_smc:
            W_bar = w_mat
            attention_mask_resampled = attention_mask.repeat_interleave(K, dim=0)
            prompt_texts_resampled = [p for p in prompt_texts for _ in range(K)]
        else:
            S = (args.cdm_buffer_size // ws) // B
            if args.cdm_pos_uniform_subsample:
                assert S <= K, f"cdm_pos_uniform_subsample requires S ({S}) <= K ({K})"
                rand_scores = torch.rand(B, K, device=sampler.device)
                _, sel_idx = torch.topk(rand_scores, S, dim=1, largest=False)
                flat_sel_idx = (sel_idx + torch.arange(B, device=sampler.device)[:, None] * K).view(-1)
                x0 = x0[flat_sel_idx]
                rewards = rewards[flat_sel_idx]
                w_sel = torch.gather(w_mat, 1, sel_idx) * (K / S) / B
                W_bar = w_sel.reshape(-1).contiguous()
            else:
                if getattr(args, "pos_resample_method", "multinomial") == "ssp":
                    from cdm.texts.resampling import ssp_resample_batch
                    sel_idx = ssp_resample_batch(w_mat, S)
                else:
                    sel_idx = torch.multinomial(w_mat, num_samples=S, replacement=True)
                flat_sel_idx = (sel_idx + torch.arange(B, device=sampler.device)[:, None] * K).view(-1)
                x0 = x0[flat_sel_idx]
                rewards = rewards[flat_sel_idx]
                W_bar = torch.ones(B*S, device=args.device) / (B*S)

    else:
        norm_mode = getattr(args, "cdm_pos_reward_norm", "none")
        if norm_mode != "none":
            raise NotImplementedError(f"cdm_pos_reward_norm='{norm_mode}' requires do_pos_multi_smc=True (per-prompt normalization needs K>1 particles per prompt).")
        final_log_w = (rewards / pos_alpha - v_curr + log_w_acc)
        if args.global_softmax:
            W_bar, final_is_ess = global_softmax(final_log_w, dim=0, return_ess=True)
            W_bar = W_bar.detach()
        else:
            W_bar = torch.softmax(final_log_w, dim=0).detach()
            final_is_ess = _ess_normalized(W_bar)
    return x0, rewards, W_bar, prompt_texts_resampled, attention_mask_resampled, (ess_pair, ratio_of_resampled, final_is_ess, reward_stats, unique_per_prompt)


@torch.no_grad()
def _collect_neg_samples(args, sampler, init_seq, attention_mask, stop_t):
    from cdm.texts.main import (
        _propose_llada_step
    )
    if args.match_pos_neg_prompt:
        S = args.twist_batch_size // args.cdm_pos_prompt_sample_size
    elif args.cdm_neg_sample_method == "multi_is":
        S = args.twist_batch_size // args.cdm_neg_prompt_sample_size
    elif args.cdm_neg_sample_method == "is":
        S = 1
    else:
        raise ValueError(f"Unsupported cdm_neg_sample_method: {args.cdm_neg_sample_method}")

    if S > 1:
        init_seq = init_seq.repeat_interleave(S, dim=0)
        attention_mask = attention_mask.repeat_interleave(S, dim=0)

    mask_index = (init_seq == sampler.mask_token)
    num_transfer_tokens = sampler.get_num_transfer_tokens(mask_index, sampler.steps)
    x_t = init_seq
    for i in range(stop_t):
        x_t = _propose_llada_step(sampler, x_t, num_transfer_tokens, i, args.remasking, args.chunk_b_size, attention_mask)
    return x_t

def compute_pos_loss(args, denoiser, sampler, samples, W_bar, attn_mask, query_sz, t, cur_epoch):
    method = args.cdm_pos_sample_method
    if method == "stmc":
        switch_to_tsmc_epoch = int(args.cdm_stmc_frac * args.twist_epochs)
        method = "smc" if cur_epoch < switch_to_tsmc_epoch else "twist_smc"

    if method == "is":
        assert False, "Not implemented yet"
    elif method in ("smc", "twist_smc"):
        x0 = samples.detach()
        B = x0.shape[0]
        gen_length = x0.shape[1] - query_sz
        assert gen_length == args.gen_length, f"Expected gen_length to be {args.gen_length} but got {gen_length}"
        num_to_mask = gen_length - int(t)

        rand_scores = torch.rand(B, gen_length, device=x0.device)
        _, mask_idx_in_resp = torch.topk(rand_scores, num_to_mask, dim=1, largest=False)
        mask_idx_abs = mask_idx_in_resp + query_sz

        x_t_pos = x0.clone()
        batch_idx = torch.arange(B, device=x0.device)[:, None].expand(-1, num_to_mask)
        x_t_pos[batch_idx, mask_idx_abs] = sampler.mask_token

        log_psi = denoiser(x_t_pos.detach(), attention_mask=attn_mask, return_logits=False, return_value=True)
        pos_term = (W_bar * log_psi).sum()
        return pos_term

def compute_neg_loss(args, denoiser, samples, attn_mask, ema_for_sampling=False, ema_idx=0):
    if args.match_pos_neg_prompt:
        ws = get_world_size()
        B = args.cdm_pos_prompt_sample_size // ws
        S = samples.shape[0] // B
        attn_mask = attn_mask.repeat_interleave(S, dim=0)
    elif args.cdm_neg_sample_method == "multi_is":
        S = args.twist_batch_size // args.cdm_neg_prompt_sample_size
        B = samples.shape[0] // S
        attn_mask = attn_mask.repeat_interleave(S, dim=0)

    x = samples.detach()
    if ema_for_sampling:
        log_psi_neg, log_psi_weight = denoiser(x, attention_mask=attn_mask, return_logits=False,
                                                return_value=True, return_ema_value=True, ema_idx=ema_idx)
        log_psi_weight = log_psi_weight.clone().detach()
    else:
        log_psi_neg = denoiser(x, attention_mask=attn_mask, return_logits=False, return_value=True)
        log_psi_weight = log_psi_neg.clone().detach()

    if args.cdm_neg_sample_method == "is":
        if args.global_softmax:
            weights, neg_ess = global_softmax(log_psi_weight.float(), dim=0, return_ess=True)
        else:
            weights = torch.softmax(log_psi_weight.float(), dim=0)
            neg_ess = _ess_normalized(weights)
        neg_term = (weights * log_psi_neg).sum()

    elif args.cdm_neg_sample_method == "multi_is":
        weights = torch.softmax(log_psi_weight.float().reshape(B,S), dim=1)
        neg_esss = (1.0 / (S * weights.float().pow(2).sum(dim=-1).clamp(min=1e-12)))
        neg_ess = neg_esss.mean().item()
        weights_ = weights.reshape(-1) / B
        neg_term = (weights_ * log_psi_neg).sum()
    return neg_term, neg_ess



