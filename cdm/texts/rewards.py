# Reward models for dLLM alignment.
#
# Given reward   : Skywork-Reward-Llama-3.1-8B preference score.
# Heldout reward : ArmoRM-Llama3-8B score, never used for scaling.
import importlib.util
from multiprocessing.managers import BaseManager

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODELS = {}
_REMOTE_MANAGER = None
PROMPT_CONDITIONED_REWARDS = {"armorm", "skywork"}


def _requires_prompt_texts(reward_name):
    return reward_name in PROMPT_CONDITIONED_REWARDS


def _align_prompt_texts(prompt_texts, output_count, reward_name="unknown"):
    if prompt_texts is None:
        return None
    if isinstance(prompt_texts, str):
        prompt_texts = [prompt_texts]
    else:
        prompt_texts = list(prompt_texts)
    if len(prompt_texts) == 0:
        raise ValueError(f"prompt_texts must be non-empty when reward_name={reward_name}")
    if len(prompt_texts) == output_count:
        return prompt_texts
    if output_count % len(prompt_texts) != 0:
        raise ValueError(
            f"prompt_texts length ({len(prompt_texts)}) must divide output length ({output_count}) "
            f"for reward_name={reward_name}"
        )
    repeat_factor = output_count // len(prompt_texts)
    return [p for p in prompt_texts for _ in range(repeat_factor)]


def _get_model_device(model):
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_armorm_model(args):
    model_id = args.armorm_model_id
    revision = args.armorm_revision
    cache_key = f"armorm::{model_id}::{revision}"
    if cache_key not in MODELS:
        device_map = args.armorm_device_map
        trust_remote_code = args.armorm_trust_remote_code
        torch_dtype = args.armorm_torch_dtype

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        use_device_map = device_map not in (None, "", "none", "None")
        if use_device_map and not has_accelerate:
            print(
                "[*] accelerate not found; disabling armorm_device_map and "
                "falling back to standard model loading."
            )
            use_device_map = False

        target_device = args.device

        model_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": torch_dtype}
        tokenizer_kwargs = {"use_fast": True}
        if revision is not None:
            model_kwargs["revision"] = revision
            tokenizer_kwargs["revision"] = revision
        if use_device_map:
            model_kwargs["device_map"] = device_map

        model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs).eval()
        if not use_device_map:
            model = model.to(target_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        MODELS[cache_key] = {"model": model, "tokenizer": tokenizer}

    return MODELS[cache_key]["model"], MODELS[cache_key]["tokenizer"]




def _get_skywork_model(args):
    model_id = args.skywork_model_id
    revision = args.skywork_revision
    cache_key = f"skywork::{model_id}::{revision}"
    if cache_key not in MODELS:
        device_map = args.skywork_device_map
        trust_remote_code = args.skywork_trust_remote_code
        torch_dtype = args.skywork_torch_dtype

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        use_device_map = device_map not in (None, "", "none", "None")
        if use_device_map and not has_accelerate:
            print(
                "[*] accelerate not found; disabling skywork_device_map and "
                "falling back to standard model loading."
            )
            use_device_map = False

        target_device = args.device

        model_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": torch_dtype}
        tokenizer_kwargs = {"use_fast": True}
        if revision is not None:
            model_kwargs["revision"] = revision
            tokenizer_kwargs["revision"] = revision
        if use_device_map:
            model_kwargs["device_map"] = device_map

        model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs).eval()
        if not use_device_map:
            model = model.to(target_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        MODELS[cache_key] = {"model": model, "tokenizer": tokenizer}

    return MODELS[cache_key]["model"], MODELS[cache_key]["tokenizer"]


def armorm_score(*, texts, prompt_texts, args, batch_size=8):
    aligned_prompts = _align_prompt_texts(prompt_texts, len(texts), reward_name="armorm")
    model, tokenizer = _get_armorm_model(args)
    device = _get_model_device(model)
    truncation = args.armorm_truncation
    max_length = args.armorm_max_length

    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_responses = texts[i : i + batch_size]
        batch_prompts = aligned_prompts[i : i + batch_size]
        rendered = []
        for prompt, response in zip(batch_prompts, batch_responses):
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            rendered.append(text)

        tokenized = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=truncation,
            max_length=max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.no_grad():
            output = model(**tokenized)
            if hasattr(output, "score"):
                batch_scores = output.score
            elif hasattr(output, "scores"):
                batch_scores = output.scores
            elif hasattr(output, "logits"):
                batch_scores = output.logits.squeeze(-1)
            else:
                raise ValueError("ArmoRM output does not contain score/logits field")
        batch_scores = batch_scores.reshape(-1).detach().cpu().to(torch.float32).tolist()
        all_scores.extend([float(s) for s in batch_scores])

    unreduced = [[s] for s in all_scores]
    return all_scores, unreduced


def skywork_score(*, texts, prompt_texts, args, batch_size=8):
    aligned_prompts = _align_prompt_texts(prompt_texts, len(texts), reward_name="skywork")
    model, tokenizer = _get_skywork_model(args)
    device = _get_model_device(model)
    truncation = args.skywork_truncation
    max_length = args.skywork_max_length

    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_responses = texts[i : i + batch_size]
        batch_prompts = aligned_prompts[i : i + batch_size]
        rendered = []
        for prompt, response in zip(batch_prompts, batch_responses):
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            rendered.append(text)

        tokenized = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=truncation,
            max_length=max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.no_grad():
            output = model(**tokenized)
            if hasattr(output, "score"):
                batch_scores = output.score
            elif hasattr(output, "scores"):
                batch_scores = output.scores
            elif hasattr(output, "logits"):
                batch_scores = output.logits.squeeze(-1)
            else:
                raise ValueError("Skywork output does not contain score/logits field")
        batch_scores = batch_scores.reshape(-1).detach().cpu().to(torch.float32).tolist()
        all_scores.extend([float(s) for s in batch_scores])

    unreduced = [[s] for s in all_scores]
    return all_scores, unreduced





def _get_remote_manager(args):
    """Connect (or reconnect) to a running `cdm.texts.reward_server`."""
    global _REMOTE_MANAGER
    if _REMOTE_MANAGER is None:
        class _RewardClientManager(BaseManager):
            pass
        _RewardClientManager.register("process_reward")

        addr = getattr(args, "reward_server_addr", "localhost")
        port = int(getattr(args, "reward_server_port", 5000))
        authkey = getattr(args, "reward_server_authkey", "reward_secret")

        mgr = _RewardClientManager(
            address=(addr, port),
            authkey=authkey.encode() if isinstance(authkey, str) else authkey,
        )
        mgr.connect()
        _REMOTE_MANAGER = mgr
        print(f"[rewards] Connected to reward server at {addr}:{port}")
    return _REMOTE_MANAGER


def _remote_compute_reward_scores(texts, reward_name, reward_label, args, prompt_texts=None):
    """Send texts to the remote reward server and return scores."""
    mgr = _get_remote_manager(args)
    proxy = mgr.process_reward({
        "texts": texts,
        "prompt_texts": prompt_texts,
        "reward_name": reward_name,
        "reward_label": reward_label,
        "batch_size": args.reward_eval_batch_size,
    })
    # BaseManager hands back a proxy; unwrap it to a real dict.
    result = proxy._getvalue() if hasattr(proxy, "_getvalue") else proxy
    return result["scores"]


def compute_reward_scores(texts, reward_name, reward_label, args, prompt_texts=None):
    """Dispatcher for specific reward model calls."""
    # The 8B reward models can be served from another GPU to free memory here.
    if getattr(args, "reward_server_enabled", False):
        return _remote_compute_reward_scores(texts, reward_name, reward_label, args, prompt_texts)

    if prompt_texts is None:
        raise ValueError(f"reward_name={reward_name} requires prompt_texts for prompt+answer scoring")
    scorer = {"armorm": armorm_score, "skywork": skywork_score}.get(reward_name)
    if scorer is None:
        raise ValueError(f"Unknown reward function: {reward_name}")
    return scorer(
        texts=texts,
        prompt_texts=prompt_texts,
        args=args,
        batch_size=args.reward_eval_batch_size,
    )[0]


def evaluate_generation(output_ids, tokenizer, args, prompt_texts=None):
    """
    Decodes and computes rewards for generated sequences.
    Handles batching internally to prevent OOM in the reward model.
    """
    # 1. Truncate to desired reward length
    assert output_ids.shape[1] == args.gen_length, f"output_ids.shape[1] {output_ids.shape[1]} != args.gen_length {args.gen_length}"
    sequences = output_ids[:, :args.gen_length]

    # 2. Decode all sequences at once (Tokenizer is usually fast enough)
    decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    all_scores = []

    # 3. Compute rewards in chunks
    if args.reward_eval_batch_size is None:
        reward_eval_batch_size = args.batch_size
    else:
        reward_eval_batch_size = args.reward_eval_batch_size

    reward_name = args.reward_name
    aligned_prompts = None
    if _requires_prompt_texts(reward_name):
        if prompt_texts is None:
            raise ValueError(f"reward_name={reward_name} requires prompt_texts for prompt+answer scoring")
        aligned_prompts = _align_prompt_texts(prompt_texts, len(decoded_texts), reward_name=reward_name)

    for i in range(0, len(decoded_texts), reward_eval_batch_size):
        batch_texts = decoded_texts[i : i + reward_eval_batch_size]
        batch_prompts = None if aligned_prompts is None else aligned_prompts[i : i + reward_eval_batch_size]
        batch_scores = compute_reward_scores(
            texts=batch_texts,
            reward_name=args.reward_name,
            reward_label=args.reward_label,
            args=args,
            prompt_texts=batch_prompts,
        )
        all_scores.extend(batch_scores)

    return all_scores
