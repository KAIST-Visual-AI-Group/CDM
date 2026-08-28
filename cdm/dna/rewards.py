"""
DNA reward functions for SMC guided generation.

Uses the Gosai oracle model to predict HepG2 gene expression from DNA sequences.
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

_ORACLE_CACHE = {}

# Local cache for the Enformer "human_state_dict" artifact pulled from wandb
# by grelu.  After first successful fetch we write the file here so later
# runs can load it without contacting wandb at all.
_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
_ENFORMER_CACHE_PATH = _CKPT_DIR / "human_state_dict.h5"


class _LocalArtifact:
    """Stub that mimics a wandb Artifact.download() using a local file."""

    def __init__(self, src_path: Path, dst_name: str = "human.h5"):
        self._src_path = src_path
        self._dst_name = dst_name

    def download(self, root):
        shutil.copy(self._src_path, Path(root) / self._dst_name)


def _patch_grelu_artifact_fetch():
    """Route grelu's wandb artifact fetches through a local cache.

    The Enformer backbone's ``__init__`` downloads a ~940 MB state dict from
    wandb on every instantiation.  The Lightning ``load_from_checkpoint`` call
    then overwrites those weights anyway, so the download is pure waste — but
    it also introduces a hard runtime dependency on the wandb API.  Here we:

    * on the first successful fetch, copy the downloaded file into
      ``_ENFORMER_CACHE_PATH``;
    * on every subsequent fetch, short-circuit wandb entirely and return a
      local-file stub with the same ``download(root)`` interface.
    """
    import grelu.resources as _gres

    # Already patched (e.g. from a prior oracle load in the same process).
    if getattr(_gres.get_artifact, "_cdm_patched", False):
        return

    _gres._check_wandb = lambda host=None: None

    _orig_get_artifact = _gres.get_artifact

    def _patched_get_artifact(name, project=None, host=None, alias="latest"):
        if name == "human_state_dict" and _ENFORMER_CACHE_PATH.exists():
            return _LocalArtifact(_ENFORMER_CACHE_PATH)

        art = _orig_get_artifact(name, project=project, host=host, alias=alias)

        if name == "human_state_dict":
            _ENFORMER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory() as d:
                art.download(d)
                shutil.copy(Path(d) / "human.h5", _ENFORMER_CACHE_PATH)
            return _LocalArtifact(_ENFORMER_CACHE_PATH)

        return art

    _patched_get_artifact._cdm_patched = True
    _gres.get_artifact = _patched_get_artifact


def _load_grelu_oracle(ckpt_path, device='cuda'):
    """Load a grelu ``LightningModel`` oracle, tolerant of grelu version drift.

    The DRAKES oracle checkpoints store ``model_params`` / ``train_params`` / ``data_params``
    under ``hyper_parameters``, but grelu >= 1.1's ``on_load_checkpoint`` reads a top-level
    ``checkpoint["data_params"]`` that these checkpoints lack. Rebuild the module from
    ``hyper_parameters`` and load the weights directly, which is equivalent for inference.
    """
    from grelu.lightning import LightningModel

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ck.get("hyper_parameters", {})
    if "data_params" not in ck and {"model_params", "train_params"} <= set(hp):
        model = LightningModel(model_params=hp["model_params"], train_params=hp["train_params"])
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        if missing or unexpected:
            import warnings
            warnings.warn(f"grelu oracle load: {len(missing)} missing / "
                          f"{len(unexpected)} unexpected state_dict keys "
                          f"(grelu version drift). First missing: {missing[:3]}")
        return model.to(device)
    return LightningModel.load_from_checkpoint(ckpt_path, map_location=device)


def _get_gosai_oracle(mode='train'):
    """Load and cache the Gosai oracle model."""
    if mode not in _ORACLE_CACHE:
        _patch_grelu_artifact_fetch()

        if mode == 'train':
            ckpt = str(_CKPT_DIR / "reward_oracle_ft.ckpt")
        elif mode == 'eval':
            ckpt = str(_CKPT_DIR / "reward_oracle_eval.ckpt")
        else:
            raise ValueError(f"Unknown oracle mode: {mode}")
        model = _load_grelu_oracle(ckpt, device='cuda')
        model.train_params['logger'] = None
        model.eval()
        _ORACLE_CACHE[mode] = model
    return _ORACLE_CACHE[mode]


def _predict_hepg2_tokens(tokens, oracle_model=None):
    """GPU-only HepG2 prediction from a token-id tensor.

    Args:
        tokens: [B, L] int tensor on the oracle's device.  Values in
                {0..4} (mask = 4) are clamped to {0..3}, collapsing the
                mask channel to T — matches the legacy
                ``np.clip(token_ids_np, 0, 3)`` behaviour before
                detokenization.
        oracle_model: optional pre-loaded oracle.

    Returns:
        [B] tensor of HepG2 predictions on the oracle's device.
    """
    if oracle_model is None:
        oracle_model = _get_gosai_oracle(mode='train')
    oracle_model.eval()
    tokens = tokens.long().clamp(0, 3)
    onehot = F.one_hot(tokens, num_classes=4).float()
    preds = oracle_model(onehot.transpose(1, 2))
    if preds.ndim > 2:
        preds = preds.squeeze(2)
    return preds[:, 0]  # HepG2 column


def _predict_hepg2(seqs_str, oracle_model=None):
    """
    Predict HepG2 expression for a list of DNA strings.

    Args:
        seqs_str: list of DNA strings (e.g., ["ACGTACGT...", ...])
        oracle_model: optional pre-loaded oracle

    Returns:
        numpy array of shape [n_seqs] with HepG2 predictions (column 0).
    """
    from cdm.dna.dataloader import batch_dna_tokenize
    tokens = torch.as_tensor(batch_dna_tokenize(seqs_str)).long().cuda()
    preds = _predict_hepg2_tokens(tokens, oracle_model=oracle_model)
    return preds.detach().cpu().float().numpy()


def _predict_hepg2_diff(probs, oracle_model=None):
    """Differentiable counterpart of ``_predict_hepg2``.

    Args:
        probs: [B, L, V] continuous one-hot / probability tensor over
            (ACGT + mask). Gradients flow through this tensor.
        oracle_model: optional pre-loaded Gosai oracle. If None, the
            cached training oracle is used.

    The slicing matches ``_predict_hepg2``:
        ``probs[..., :4]`` drops the mask channel because the oracle was
        trained on 4-channel ACGT one-hot input.
        ``preds[:, 0]`` selects the HepG2 track from the multi-task
        Enformer head (cols 1/2 are K562/SK-N-SH).

    Returns:
        [B] HepG2 prediction tensor.
    """
    if oracle_model is None:
        oracle_model = _get_gosai_oracle(mode='train')
    oracle_model.eval()
    preds = oracle_model(probs[..., :4].transpose(1, 2))
    if preds.ndim > 2:
        preds = preds.squeeze(2)
    return preds[:, 0]


def evaluate_generation(token_ids, tokenizer, args, oracle_model=None):
    """
    Evaluate generated DNA sequences using the Gosai HepG2 oracle.

    Args:
        token_ids: [B, L] int64 tensor of token indices (0-3 for ACGT, 4 for mask)
        tokenizer: GosaiTokenizer instance
        args: config namespace (needs reward_name, etc.)
        oracle_model: optional pre-loaded oracle model

    Returns:
        list of float rewards, length B.
    """
    reward_name = args.reward_name
    if reward_name != 'hepg2':
        raise ValueError(f"Unknown reward_name: {reward_name}")

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.as_tensor(np.asarray(token_ids))
    if not token_ids.is_cuda:
        token_ids = token_ids.cuda()

    B = token_ids.shape[0]
    if B == 0:
        return []

    chunk_size = args.m_chunk_size or B
    chunks = []
    for i in range(0, B, chunk_size):
        chunks.append(
            _predict_hepg2_tokens(token_ids[i:i + chunk_size], oracle_model=oracle_model)
        )
    rewards = torch.cat(chunks, dim=0)
    return rewards.detach().cpu().float().tolist()
