"""
Reward model server — offloads heavy reward models (ArmoRM, Skywork)
to a separate GPU and serves reward computation via BaseManager IPC.

Usage:
    # On the machine/GPU where you want to run the reward model:
    python -m cdm.texts.reward_server --gpu 1 --port 5000 --reward_name skywork

    # Then in your training config, set:
    #   reward_server_enabled: true
    #   reward_server_port: 5000
"""

from multiprocessing.managers import BaseManager
import argparse
import time
import torch


class RewardServerManager(BaseManager):
    pass


def process_reward(input_dict):
    """Called remotely by the client. Receives texts and returns scores."""
    start_time = time.time()

    texts = input_dict["texts"]
    prompt_texts = input_dict.get("prompt_texts")
    reward_name = input_dict.get("reward_name", _server_args.reward_name)
    reward_label = input_dict.get("reward_label", _server_args.reward_label)
    batch_size = input_dict.get("batch_size", _server_args.reward_eval_batch_size)

    old_bs = _server_args.reward_eval_batch_size
    _server_args.reward_eval_batch_size = batch_size
    try:
        scores = _compute_reward_scores(
            texts, reward_name, reward_label, _server_args, prompt_texts
        )
    finally:
        _server_args.reward_eval_batch_size = old_bs

    elapsed = time.time() - start_time
    print(f"[reward_server] {reward_name} | {len(texts)} texts | {elapsed:.3f}s")

    return {"scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward model server")

    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA device index for the reward model")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to listen on")
    parser.add_argument("--authkey", type=str, default="reward_secret",
                        help="Shared secret for BaseManager auth")

    # Reward config (mirrors base.yaml keys)
    parser.add_argument("--reward_name", type=str, default="skywork")
    parser.add_argument("--reward_label", type=str, default="positive")
    parser.add_argument("--reward_eval_batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)

    # ArmoRM
    parser.add_argument("--armorm_model_id", type=str,
                        default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--armorm_revision", type=str, default=None)
    parser.add_argument("--armorm_max_length", type=int, default=4096)
    parser.add_argument("--armorm_truncation", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--armorm_torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--armorm_device_map", type=str, default=None)
    parser.add_argument("--armorm_trust_remote_code", action=argparse.BooleanOptionalAction,
                        default=True)

    # Skywork
    parser.add_argument("--skywork_model_id", type=str,
                        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    parser.add_argument("--skywork_revision", type=str, default=None)
    parser.add_argument("--skywork_max_length", type=int, default=4096)
    parser.add_argument("--skywork_truncation", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--skywork_torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--skywork_device_map", type=str, default=None)
    parser.add_argument("--skywork_trust_remote_code", action=argparse.BooleanOptionalAction,
                        default=False)


    args = parser.parse_args()

    # Set device to the specified GPU
    args.device = f"cuda:{args.gpu}"

    # Import reward logic (does NOT load models yet — lazy loading on first call)
    from cdm.texts.rewards import compute_reward_scores as _compute_reward_scores

    _server_args = args

    # Warm up: trigger model loading so the first client call isn't slow
    print(f"[reward_server] Warming up '{args.reward_name}' on {args.device} ...")
    try:
        warmup_texts = ["warmup"]
        warmup_prompts = ["warmup"] if args.reward_name in ("armorm", "skywork") else None
        _compute_reward_scores(
            warmup_texts, args.reward_name, args.reward_label, args,
            prompt_texts=warmup_prompts,
        )
        print(f"[reward_server] Warm-up complete.")
    except Exception as e:
        print(f"[reward_server] Warm-up failed (non-fatal): {e}")

    # Register and start server
    RewardServerManager.register("process_reward", callable=process_reward)

    manager = RewardServerManager(
        address=("0.0.0.0", args.port),
        authkey=args.authkey.encode(),
    )
    server = manager.get_server()
    print(f"[reward_server] Listening on port {args.port} (authkey='{args.authkey}')")
    server.serve_forever()
