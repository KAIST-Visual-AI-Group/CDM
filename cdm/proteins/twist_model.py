"""
Twist head construction / checkpointing for the DPLM2 discrete diffusion
alignment pipeline.

The twist function psi^theta(x_t) -> scalar lives as an `nn.Module` head
attached directly to a `DPLMDenoiser` (see `DPLMDenoiser.attach_twist_head`).
The backbone is shared with the LM forward path, so a single backbone forward
pass yields both the LM logits AND the scalar twist value — there is no
separate twist network and no duplicated backbone in memory.

The helpers in this module operate on a denoiser-with-attached-head:

    denoiser = DPLMDenoiser(...)
    make_twist_net(args, denoiser)              # attaches the head in place
    save_twist_checkpoint(denoiser, args, ...)  # serializes denoiser.head
    load_twist_checkpoint(path, ..., denoiser)  # attaches & loads head
"""

import os

import torch


_TWIST_ARCHES = ("frozen_linear", "frozen_mlp", "frozen_transformer")
_ARCH_TO_HEAD_TYPE = {
    "frozen_linear": "linear",
    "frozen_mlp": "mlp",
    "frozen_transformer": "transformer",
}


def make_twist_net(args, denoiser):
    """Attach a trainable twist head to `denoiser` and return it.

    The denoiser is mutated in place; the return value is just a
    convenience so callers can write `denoiser = make_twist_net(args, denoiser)`.
    """
    twist_arch = args.twist_arch
    if twist_arch not in _TWIST_ARCHES:
        raise NotImplementedError(
            f"twist_arch='{twist_arch}' not implemented for DPLM2"
        )

    denoiser.attach_twist_head(
        head_type=_ARCH_TO_HEAD_TYPE[twist_arch],
        mlp_n_layers=args.twist_head_mlp_n_layers,
        mlp_hidden_size=args.twist_head_mlp_hidden_size,
        transformer_n_layers=getattr(args, "twist_head_transformer_n_layers", 2),
        transformer_n_heads=getattr(args, "twist_head_transformer_n_heads", 8),
        transformer_ff_dim=getattr(args, "twist_head_transformer_ff_dim", 1024),
    )
    return denoiser


def save_twist_checkpoint(denoiser, args, epoch, loss, filename, vocab_size=None):
    """Persist `denoiser.head` + the architecture config needed to rebuild it."""
    if denoiser.head is None:
        raise RuntimeError("denoiser has no twist head attached; nothing to save")

    twist_arch = args.twist_arch
    if twist_arch not in _TWIST_ARCHES:
        raise NotImplementedError(
            f"Saving twist_arch='{twist_arch}' not implemented for DPLM2"
        )

    payload = {
        "twist_net": denoiser.head.state_dict(),
        "config": dict(
            twist_arch=twist_arch,
            vocab_size=vocab_size,
            head_type=denoiser.head_type,
            mlp_n_layers=denoiser.mlp_n_layers,
            mlp_hidden_size=denoiser.mlp_hidden_size,
            transformer_n_layers=denoiser.transformer_n_layers,
            transformer_n_heads=denoiser.transformer_n_heads,
            transformer_ff_dim=denoiser.transformer_ff_dim,
        ),
        "epoch": epoch,
        "loss": loss,
    }

    os.makedirs(args.save_path, exist_ok=True)
    torch.save(payload, os.path.join(args.save_path, filename))


def load_twist_checkpoint(ckpt_path, device, denoiser):
    """Attach a twist head to `denoiser` and load its weights from disk."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    twist_arch = cfg["twist_arch"]

    if twist_arch not in _TWIST_ARCHES:
        raise NotImplementedError(
            f"Loading twist_arch='{twist_arch}' not implemented for DPLM2"
        )

    denoiser.attach_twist_head(
        head_type=cfg["head_type"],
        mlp_n_layers=cfg["mlp_n_layers"],
        mlp_hidden_size=cfg["mlp_hidden_size"],
        transformer_n_layers=cfg.get("transformer_n_layers", 2),
        transformer_n_heads=cfg.get("transformer_n_heads", 8),
        transformer_ff_dim=cfg.get("transformer_ff_dim", 1024),
    )
    denoiser.head.load_state_dict(ckpt["twist_net"])
    denoiser.head.eval()
    return denoiser
