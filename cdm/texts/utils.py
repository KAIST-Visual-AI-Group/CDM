
import numpy as np
import torch


def get_input_prompt(args, llada_denoiser, system_prompts: list):
    """
    Takes a list of different system prompts, applies the chat template, 
    tokenizes with official batched-inference settings, and adds mask tokens.
    """
    formatted_inputs = []
    
    # Apply chat template to each prompt individually
    for prompt in system_prompts:
        m = [{"role": "user", "content": prompt}]
        user_input = llada_denoiser.tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        formatted_inputs.append(user_input)

    # Match the official LLaDA batched inference path:
    # - left padding
    # - chat template rendered to string first
    # - add_special_tokens=False on the already-rendered prompt
    tokenized = llada_denoiser.tokenizer(
        formatted_inputs,
        add_special_tokens=False,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"].to(args.device)
    prompt_attention_mask = tokenized["attention_mask"].to(args.device)
    query_sz = input_ids.shape[1]

    mask_pad_input_ids = llada_denoiser.add_mask_tokens(input_ids)
    gen_length = mask_pad_input_ids.shape[1] - query_sz
    tail_attention_mask = torch.ones(
        (input_ids.shape[0], gen_length),
        dtype=prompt_attention_mask.dtype,
        device=args.device,
    )
    full_attention_mask = torch.cat(
        [prompt_attention_mask, tail_attention_mask],
        dim=1,
    )

    return input_ids, mask_pad_input_ids, query_sz, full_attention_mask


# ══════════════════════════════════════════════════════════════
#  Training utilities for CDM
# ══════════════════════════════════════════════════════════════


def make_linear_decay_scheduler(optimizer, total_steps, decay_start_frac):
  decay_start = int(total_steps * decay_start_frac)
  def _lr_lambda(step):
      if step <= decay_start:
          return 1.0
      return max(0.0, (total_steps - step) / (total_steps - decay_start))
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
  return scheduler, decay_start


def ess_normalized(log_w, dim):
  """Normalized ESS in [1/N, 1] from unnormalized log weights along `dim`.
  ESS = (Σ w_i)^2 / Σ w_i^2; per-slice ESS averaged over remaining dims."""
  n = log_w.shape[dim]
  w = torch.softmax(log_w, dim=dim)
  ess = 1.0 / w.pow(2).sum(dim=dim).clamp(min=1e-12)
  return (ess / n).mean().item()


def ess_summary(ess_steps):
  """Reduce a list of per-step ESS floats to (mean, min). Empty → (nan, nan)."""
  if not ess_steps:
    return float("nan"), float("nan")
  return float(np.mean(ess_steps)), float(np.min(ess_steps))
