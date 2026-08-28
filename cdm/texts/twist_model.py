"""
Twist head construction / checkpointing for the LLaDA discrete diffusion
alignment pipeline.

The twist function ψ^θ(x_t) → scalar lives as an `nn.Module` head attached
directly to a `LLaDADenoiser` (see `LLaDADenoiser.attach_twist_head`). The
backbone is shared with the LM forward path, so a single backbone forward
pass yields both the LM logits AND the scalar twist value — there is no
separate twist network and no duplicated 8B-parameter model in memory.

The helpers in this module operate on a denoiser-with-attached-head:

    denoiser = LLaDADenoiser(...)
    make_twist_net(args, denoiser)              # attaches the head in place
    save_twist_checkpoint(denoiser, args, ...)  # serializes denoiser.head
    load_twist_checkpoint(path, ..., denoiser)  # attaches & loads head
"""

import os

import torch


def unwrap_head(head):
    if isinstance(head, torch.nn.parallel.DistributedDataParallel):
        return head.module
    return head


def make_twist_net(args, denoiser):
    """Attach a trainable twist head to `denoiser` and return it.

    The denoiser is mutated in place; the return value is just a
    convenience so callers can write `denoiser = make_twist_net(args, denoiser)`.
    """
    twist_arch = args.twist_arch
    if twist_arch in ("frozen_linear", "frozen_mlp"):
        head_type = "linear" if twist_arch == "frozen_linear" else "mlp"
        denoiser.attach_twist_head(
            head_type=head_type,
            mlp_n_layers=args.twist_head_mlp_n_layers,
            mlp_hidden_size=args.twist_head_mlp_hidden_size,
        )
        return denoiser
    if twist_arch == "frozen_transformer":
        denoiser.attach_twist_head(
            head_type="transformer",
            tf_n_layers=args.twist_head_tf_n_layers,
            tf_n_heads=args.twist_head_tf_n_heads,
            tf_hidden_size=args.twist_head_tf_hidden_size,
            tf_ffn_size=args.twist_head_tf_ffn_size,
            tf_dropout=args.twist_head_tf_dropout,
        )
        return denoiser
    raise NotImplementedError(f"twist_arch='{twist_arch}' not implemented for LLaDA")


def save_twist_checkpoint(denoiser, args, epoch, loss, filename, vocab_size=None, head_override=None):
    """Persist `denoiser.head` + the architecture config needed to rebuild it.

    The payload mirrors the `texts_mdm` format so that downstream tooling can
    consume LLaDA and MDM twist checkpoints with the same loader logic.

    If `head_override` is supplied, that module's state_dict is saved
    instead of `denoiser.head`. The architecture config is still pulled
    from `denoiser` (the override is expected to mirror the live head's
    architecture — e.g. an EMA copy).
    """
    if head_override is not None:
        head_module = head_override
    else:
        if denoiser.head is None:
            raise RuntimeError("denoiser has no twist head attached; nothing to save")
        head_module = unwrap_head(denoiser.head)

    twist_arch = args.twist_arch
    if twist_arch in ("frozen_linear", "frozen_mlp"):
        payload = {
            "twist_net": head_module.state_dict(),
            "config": dict(
                twist_arch=twist_arch,
                vocab_size=vocab_size,
                head_type=denoiser.head_type,
                mlp_n_layers=denoiser.mlp_n_layers,
                mlp_hidden_size=denoiser.mlp_hidden_size,
                            ),
            "epoch": epoch,
            "loss": loss,
        }
    elif twist_arch == "frozen_transformer":
        payload = {
            "twist_net": head_module.state_dict(),
            "config": dict(
                twist_arch=twist_arch,
                vocab_size=vocab_size,
                head_type=denoiser.head_type,
                tf_n_layers=denoiser.tf_n_layers,
                tf_n_heads=denoiser.tf_n_heads,
                tf_hidden_size=denoiser.tf_hidden_size,
                tf_ffn_size=denoiser.tf_ffn_size,
                tf_dropout=denoiser.tf_dropout,
                            ),
            "epoch": epoch,
            "loss": loss,
        }
    else:
        raise NotImplementedError(f"Saving twist_arch='{twist_arch}' not implemented for LLaDA")

    torch.save(payload, os.path.join(args.save_path, filename))


def load_twist_checkpoint(ckpt_path, device, denoiser):
    """Attach a twist head to `denoiser` and load its weights from disk."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    twist_arch = cfg["twist_arch"]

    if twist_arch in ("frozen_linear", "frozen_mlp"):
        denoiser.attach_twist_head(
            head_type=cfg["head_type"],
            mlp_n_layers=cfg["mlp_n_layers"],
            mlp_hidden_size=cfg["mlp_hidden_size"],
        )
        denoiser.head.load_state_dict(ckpt["twist_net"])
    elif twist_arch == "frozen_transformer":
        denoiser.attach_twist_head(
            head_type=cfg["head_type"],
            tf_n_layers=cfg["tf_n_layers"],
            tf_n_heads=cfg["tf_n_heads"],
            tf_hidden_size=cfg["tf_hidden_size"],
            tf_ffn_size=cfg["tf_ffn_size"],
            tf_dropout=cfg["tf_dropout"],
        )
        # strict=False: older checkpoints lack timestep_proj weights;
        # the zero-init in TransformerTwistHead makes this safe.
        denoiser.head.load_state_dict(ckpt["twist_net"], strict=False)
    else:
        raise NotImplementedError(f"Loading twist_arch='{twist_arch}' not implemented for LLaDA")

    denoiser.head.eval()
    return denoiser
