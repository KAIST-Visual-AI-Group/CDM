"""LLaDA sampler: masked-diffusion ancestral sampling with confidence-based unmasking."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.

    Uses in-place ops to reduce peak GPU memory from 5× to 2× the
    float64 tensor size (i.e. from ~16 GB to ~6.5 GB for typical
    training batches).
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    logits.sub_(logits.amax(dim=-1, keepdim=True))
    noise = torch.rand_like(logits, dtype=torch.float64)
    # In-place: noise = (-log(noise))^temperature
    noise.clamp_(min=torch.finfo(torch.float64).tiny)
    noise.log_().neg_()
    if temperature != 1.0:
        noise.pow_(temperature)
    # In-place: logits = exp(logits) / noise
    logits.exp_()
    logits.div_(noise)
    del noise
    return logits


class DiffusionSampler():

    def __init__(self, denoiser, steps=10, temperature=1.0):
        self.denoiser = denoiser
        self.device = denoiser.device

        if hasattr(denoiser, 'tokenizer'):
            self.tokenizer = denoiser.tokenizer
        else:
            self.tokenizer = None

        if hasattr(denoiser, 'log_prob_table'):
            self.log_prob_target = denoiser.log_prob_table
        else:
            self.log_prob_target = None

        self.steps = steps
        self.temperature = temperature

        # if denoiser has a length, set it, otherwise, set to None (and must be initialized with input_seq)
        if hasattr(self.denoiser, 'length'):
            self.length = self.denoiser.length
        else:
            self.length = None
        self.mask_token = self.denoiser.mask_token

    # sampling done with linear noise schedule alpha_t for now (default with LLADA)
    def get_num_transfer_tokens(self, mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens

    @torch.no_grad()
    def sample(
        self,
        init_seq = None,
        batch_size = 10,
        cfg_scale = 0.,
        remasking='low_confidence',
        return_traj = False,
        stop_t=None,
        attention_mask=None,
    ):

        '''
            init_seq: A tensor of shape (1, L).
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            stop_t: If provided, run only the first `stop_t` reverse steps and
                return the (still-partially-masked) intermediate `x_t`. Defaults
                to `self.steps` (full denoising).
            return_traj: When True, also returns per-step trajectories. The
                returned ``x_traj`` has length ``steps + 1`` (or ``stop_t + 1``
                on early return): entry 0 is the initial state (prompt + mask
                tail) and entry ``-1`` is the final clean x_0 
        '''
        if stop_t is None:
            stop_t = self.steps
        if stop_t < 0 or stop_t > self.steps:
            raise ValueError(f"stop_t must be in [0, {self.steps}], got {stop_t}")
        #x = torch.full((batch_size, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)
        #x[:, :init_seq.shape[1]] = init_seq.clone()

        if init_seq is not None:
            x = init_seq.clone().to(self.denoiser.device)
            
            self.length = init_seq.shape[-1]
            
            batch_size = init_seq.shape[0] #override 
        else:
            if self.length is None:
                raise ValueError("self.length is None and no init_seq provided. Either provide init_seq or initialize denoiser with a length attribute.")
            x = torch.full((batch_size, self.length), self.mask_token, dtype=torch.long).to(self.denoiser.device)

        prompt_index = (x != self.mask_token)

        mask_index = (x == self.mask_token)
        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, self.steps)

        if return_traj:
            x0_traj = []
            # Seed x_traj with the initial state so entry 0 = init and entry -1
            # = final clean x_0 (length steps + 1 when stop_t is None).
            x_traj = [x.clone()]

        for i in tqdm(range(self.steps), 'Sampling'):

            if stop_t == i:
                if return_traj:
                    return x, x0_traj, x_traj
                return x

            mask_index = (x == self.mask_token)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = self.mask_token
                x_ = torch.cat([x, un_x], dim=0) # concat along batch dim
                attention_mask_ = None
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = self.denoiser(x_, attention_mask=attention_mask_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = self.denoiser(x, attention_mask=attention_mask) # b, l, v

            logits_with_noise = add_gumbel_noise(logits, temperature = self.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
         
            if return_traj:
                x0_traj.append(x0.clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l

            elif remasking == 'low_conf_noisy':
                p = F.log_softmax(logits_with_noise.log().to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                    
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

            if return_traj:
                x_traj.append(x.clone())

        if return_traj:
            return x, x0_traj, x_traj
        
        return x
