import torch


class CDMPosBuffer:
    def __init__(self, args):
        self.args = args
        self.samples = None
        self.extra = None
        self.smc_ess_mean = float("nan")
        self.smc_ess_min = float("nan")

    def __len__(self):
        return 0 if self.rewards is None else self.rewards.shape[0]

    def clear(self):
        self.samples = None
        self.extra = None
        self.smc_ess_mean = float("nan")
        self.smc_ess_min = float("nan")

    @torch.no_grad()
    def fill(self, sampler, model, tokenizer, collect_fn):
        args = self.args
        self.cdm_buffer_size = args.cdm_pos_batch_size
        chunk_size = args.cdm_smc_chunk_size

        self.samples, self.W_bar, extras = collect_fn(args, sampler, model, tokenizer, batch_size=self.cdm_buffer_size, cdm_smc_chunk_size=chunk_size)

        ess_pairs, ratio_resampled = extras
        self.smc_ess_mean = ess_pairs[0]
        self.smc_ess_min = ess_pairs[1]
        self.ratio_resampled = ratio_resampled

    def sample(self, batch_size, device):
        n = self.samples.shape[0]
        idx = torch.randperm(n, device=device)[:batch_size]
        samples = self.samples[idx]
        if self.W_bar is None:
            weights = None
        else:
            weights = self.W_bar[idx] * (self.cdm_buffer_size / batch_size)
        return samples, weights
