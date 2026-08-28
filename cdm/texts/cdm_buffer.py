import torch

from cdm.texts.dist_utils import get_world_size


class CDMPosBuffer:
    def __init__(self, args):
        self.args = args
        self.samples = None
        self.rewards = None
        self.W_bar = None
        self.prompt_texts = None
        self.attention_mask = None
        self.ratio_resampled = None
        self.smc_ess_mean = float("nan")
        self.smc_ess_min = float("nan")
        self.final_is_ess = float("nan")
        self.reward_mean = float("nan")
        self.reward_min = float("nan")
        self.reward_max = float("nan")
        self.unique_per_prompt = float("nan")
        # args.cdm_buffer_size and args.cdm_pos_prompt_sample_size are
        # GLOBAL config values; convert to per-rank counts here so that
        # asserts and indexing match the per-rank slice each rank holds.
        self.ws = get_world_size()
        self.cdm_buffer_size = args.cdm_buffer_size // self.ws

        if not args.do_pos_multi_smc:
            assert args.cdm_pos_prompt_sample_size == args.cdm_buffer_size, f"Expected 'cdm_pos_prompt_sample_size' to be equal to 'cdm_buffer_size' when doing single SMC!!"

        self.keep_all_smc = args.cdm_pos_keep_all_smc
        if self.keep_all_smc:
            assert args.do_pos_multi_smc, "cdm_pos_keep_all_smc=True requires do_pos_multi_smc=True"
            B = args.cdm_pos_prompt_sample_size // self.ws
            self.expected_buffer_size = B * args.multi_smc_K
        else:
            self.expected_buffer_size = self.cdm_buffer_size

    def __len__(self):
        return 0 if self.rewards is None else self.rewards.shape[0]

    def clear(self):
        self.samples = None
        self.rewards = None
        self.W_bar = None
        self.prompt_texts = None
        self.attention_mask = None
        self.ratio_resampled = None
        self.smc_ess_mean = float("nan")
        self.smc_ess_min = float("nan")
        self.final_is_ess = float("nan")
        self.reward_mean = float("nan")
        self.reward_min = float("nan")
        self.reward_max = float("nan")
        self.unique_per_prompt = float("nan")

    @torch.no_grad()
    def fill(self, sampler, init_seq, query_sz, prompt_texts, attention_mask, epoch, collect_fn):
        args = self.args
        self.samples, self.rewards, self.W_bar, self.prompt_texts, self.attention_mask, extras = collect_fn(args, sampler, init_seq, query_sz, prompt_texts, attention_mask, epoch)
        ess_pairs, self.ratio_resampled, self.final_is_ess, reward_stats, self.unique_per_prompt = extras
        self.smc_ess_mean = ess_pairs[0]
        self.smc_ess_min = ess_pairs[1]
        self.reward_mean, self.reward_min, self.reward_max = reward_stats
        assert self.rewards.shape[0] == self.expected_buffer_size, f"Expected rewards shape[0] to be {self.expected_buffer_size}, but got {self.rewards.shape[0]}"

    def sample(self, batch_size, device):
        if self.keep_all_smc:
            B, K = self.W_bar.shape
            S = batch_size // B
            if self.args.cdm_pos_uniform_subsample:
                assert S <= K, f"cdm_pos_uniform_subsample requires S ({S}) <= K ({K})"
                rand_scores = torch.rand(B, K, device=device)
                _, sel_idx = torch.topk(rand_scores, S, dim=1, largest=False)
                w_sel = torch.gather(self.W_bar, 1, sel_idx) * (K / S) / B
                weights = w_sel.reshape(-1).contiguous()
            else:
                if getattr(self.args, "pos_resample_method", "multinomial") == "ssp":
                    from cdm.texts.resampling import ssp_resample_batch
                    sel_idx = ssp_resample_batch(self.W_bar, S)
                else:
                    sel_idx = torch.multinomial(self.W_bar, num_samples=S, replacement=True)
                weights = torch.full((batch_size,), 1.0 / batch_size, device=device)

            flat_sel_idx = (sel_idx + torch.arange(B, device=device)[:, None] * K).view(-1)
            if isinstance(self.samples, list):
                samples = [s[flat_sel_idx] for s in self.samples]
            else:
                samples = self.samples[flat_sel_idx]
            rewards = self.rewards[flat_sel_idx]
            flat_sel_cpu = flat_sel_idx.cpu().tolist()
            prompt_texts = [self.prompt_texts[i] for i in flat_sel_cpu] if self.prompt_texts is not None else None
            attn_mask = self.attention_mask[flat_sel_idx] if self.attention_mask is not None else None
            return samples, rewards, weights, prompt_texts, attn_mask
        
        else:
            if batch_size == self.cdm_buffer_size:
                if isinstance(self.samples, list):
                    samples = [s.clone() for s in self.samples]
                else:
                    samples = self.samples.clone()
                rewards = self.rewards.clone()
                if self.W_bar is None:
                    weights = None
                elif self.args.cdm_pos_random_sampling_method == "uniform":
                    if isinstance(self.W_bar, list):
                        weights = [w_.clone() for w_ in self.W_bar]
                    else:
                        weights = self.W_bar.clone()
                else:
                    raise NotImplementedError(f"Unknown cdm_pos_random_sampling_method: {self.args.cdm_pos_random_sampling_method}")
                prompt_texts = list(self.prompt_texts) if self.prompt_texts is not None else None
                attention_mask = self.attention_mask.clone() if self.attention_mask is not None else None
                return samples, rewards, weights, prompt_texts, attention_mask

            idx = torch.randperm(self.cdm_buffer_size, device=device)[:batch_size]
            if isinstance(self.samples, list):
                samples = [s[idx] for s in self.samples]
            else:
                samples = self.samples[idx]

            rewards = self.rewards[idx]
            if self.W_bar is None:
                weights = None
            elif self.args.cdm_pos_random_sampling_method == "uniform":
                if isinstance(self.W_bar, list):
                    weights = [w_[idx] * (self.cdm_buffer_size / batch_size) for w_ in self.W_bar]
                else:
                    weights = self.W_bar[idx] * (self.cdm_buffer_size / batch_size)
            else:
                raise NotImplementedError(f"Unknown cdm_pos_random_sampling_method: {self.args.cdm_pos_random_sampling_method}")

            idx_cpu = idx.cpu().tolist()
            prompt_texts = [self.prompt_texts[i] for i in idx_cpu] if self.prompt_texts is not None else None
            attention_mask = self.attention_mask[idx] if self.attention_mask is not None else None
            return samples, rewards, weights, prompt_texts, attention_mask