import os
import sys
import math

import torch
import torch.nn as nn

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from dplm.generate_dplm import initialize_generation


def _sinusoidal_positional_encoding(length, d_model, device, dtype):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Returns a [length, d_model] tensor where
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerHead(nn.Module):
    """A small transformer encoder + mean-pool + linear scalar head.

    Takes the backbone's last hidden state [B, L, D], adds a sinusoidal
    positional encoding, refines with a small stack of self-attention
    layers, mean-pools across the sequence, and projects to a scalar
    value psi^theta(x_t). The final linear is zero-initialized so the
    head starts outputting 0.
    """

    def __init__(self, d_model, n_layers, n_heads, ff_dim, dropout=0.0,
                 max_len=4096):
        super().__init__()
        self.d_model = d_model
        # Precompute positional encodings up to max_len; register as a
        # non-persistent buffer so it moves with `.to(device)` but isn't
        # serialized (can be regenerated from d_model on load).
        pe = _sinusoidal_positional_encoding(
            max_len, d_model, device=torch.device("cpu"), dtype=torch.float32,
        )
        self.register_buffer("pe", pe, persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h):                       # h: [B, L, D]
        L = h.shape[1]
        if L > self.pe.shape[0]:
            # Fallback: compute on-the-fly for sequences longer than max_len.
            pe = _sinusoidal_positional_encoding(
                L, self.d_model, device=h.device, dtype=h.dtype,
            )
        else:
            pe = self.pe[:L].to(h.dtype)
        h = h + pe                              # broadcast over batch
        h = self.encoder(h)                     # [B, L, D]
        h = h.mean(dim=1)                       # [B, D]
        return self.proj(h)                     # [B, 1]


class DPLM1Denoiser:

    def __init__(self, device = 'cuda'):
        self.device = device
        from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM

        self.dplm = DPLM.from_pretrained("airkingbd/dplm_650m").to(device)

        self.mask_token = self.dplm.mask_id
        self.length = None
        self.name = "DPLM1Denoiser"
        self.tokenizer = self.dplm.tokenizer


    def __call__(self, input_seq):
        # Implement the denoising process using the DPLM model
        net_out = self.dplm.net(
            input_ids=input_seq,
        )

        logits = net_out["logits"]

        # logits have a value for the mask token as well (not just clean data)
        logits[..., self.dplm.mask_id] = -math.inf
        logits[..., self.dplm.x_id] = -math.inf
        logits[..., self.dplm.pad_id] = -math.inf
        logits[..., self.dplm.bos_id] = -math.inf
        logits[..., self.dplm.eos_id] = -math.inf

        # cut out logit value for mask id (which is last one according to dplm vanilla)
        if self.dplm.mask_id == logits.shape[-1] - 1:
            logits = logits[..., :self.dplm.mask_id]

        # logits over clean (non-mask) data (incl special tokens)
        return logits 
    

class DPLMDenoiser:

    def __init__(self, device = 'cuda'):
        self.device = device

        from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2

        self.dplm = DPLM2.from_pretrained("airkingbd/dplm2_650m").to(device)

        self.aa_mask_id = self.dplm.aa_mask_id
        self.struct_mask_id = self.dplm.struct_mask_id
        self.length = None
        self.name = "DPLM2Denoiser"
        self.tokenizer = self.dplm.tokenizer

        # Optional twist head — attached lazily via `attach_twist_head`.
        # When present, `__call__(..., return_value=True)` projects the
        # backbone's last hidden state to a scalar value psi^theta(x_t)
        # used by the CDM twist. The backbone is shared with
        # the LM forward path, so a single backbone pass yields both
        # logits and the value.
        self.head = None
        self.head_type = None
        self.mlp_n_layers = None
        self.mlp_hidden_size = None
        self.transformer_n_layers = None
        self.transformer_n_heads = None
        self.transformer_ff_dim = None

    def attach_twist_head(
        self, head_type="linear",
        mlp_n_layers=3, mlp_hidden_size=256,
        transformer_n_layers=2, transformer_n_heads=8, transformer_ff_dim=1024,
    ):
        """Install a trainable scalar reward head on top of the backbone.

        Backbone parameters are not touched (they are kept frozen by callers
        that train this head). Returns `self` so the call chains.

        head_type:
            "linear"      — single linear layer on mean-pooled hidden state
            "mlp"         — N-layer MLP on mean-pooled hidden state
            "transformer" — small transformer encoder applied to the full
                            sequence, then mean-pool + linear projection
        """
        hidden_size = self.dplm.net.config.hidden_size

        if head_type == "linear":
            head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        elif head_type == "mlp":
            layers = []
            in_dim = hidden_size
            for _ in range(mlp_n_layers - 1):
                lin = nn.Linear(in_dim, mlp_hidden_size)
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.ReLU()])
                in_dim = mlp_hidden_size
            final = nn.Linear(in_dim, 1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            head = nn.Sequential(*layers)
        elif head_type == "transformer":
            head = TransformerHead(
                d_model=hidden_size,
                n_layers=transformer_n_layers,
                n_heads=transformer_n_heads,
                ff_dim=transformer_ff_dim,
            )
        else:
            raise ValueError(
                f"Unknown head_type '{head_type}'. "
                f"Choose 'linear', 'mlp', or 'transformer'."
            )

        self.head = head.to(self.device)
        self.head_type = head_type
        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.transformer_n_layers = transformer_n_layers
        self.transformer_n_heads = transformer_n_heads
        self.transformer_ff_dim = transformer_ff_dim
        return self

    def _mask_logits(self, logits, input_seq):
        """Mask out invalid tokens per modality and special tokens."""
        logits = logits.log_softmax(dim=-1)

        # Mask logits by modality: AA positions can only predict AA tokens (id < 33),
        # struct positions can only predict struct tokens (id >= 33)
        type_ids = self.dplm.get_modality_type(input_seq)
        non_special = self.dplm.get_non_special_symbol_mask(input_seq)

        aa_position = (type_ids == self.dplm.aa_type) & non_special
        struct_position = (type_ids == self.dplm.struct_type) & non_special

        indices_aa = torch.where(aa_position)
        indices_struct = torch.where(struct_position)

        if indices_aa[0].numel() > 0:
            logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
        if indices_struct[0].numel() > 0:
            logits[indices_struct[0], indices_struct[1], :33] = -math.inf

        # Mask special tokens so they are never predicted
        logits[..., self.dplm.special_token_list] = -math.inf
        return logits

    def __call__(self, input_seq, return_logits=True, return_value=False):
        """Single backbone forward pass returning logits, value, or both.

        Args:
            input_seq: [B, L] int64 token ids (with mask tokens at noised
                positions).
            return_logits: include the LM logits in the output.
            return_value:  include the scalar twist value psi^theta(x_t).
                Requires `attach_twist_head` to have been called first.

        Returns:
            - logits          if only `return_logits` is True   (default)
            - value           if only `return_value`  is True
            - (logits, value) if both flags are True
        """
        if not return_logits and not return_value:
            raise ValueError("At least one of return_logits/return_value must be True")
        if return_value and self.head is None:
            raise RuntimeError(
                "twist head not attached; call `attach_twist_head(...)` first"
            )

        net_out = self.dplm.forward(input_ids=input_seq)

        if return_logits:
            logits = self._mask_logits(net_out["logits"], input_seq)

        if return_value:
            # The transformer head consumes the full sequence and pools
            # internally; the linear/mlp heads consume a mean-pooled
            # hidden state.
            h = net_out["last_hidden_state"].float()                # [B, L, D]
            if self.head_type != "transformer":
                h = h.mean(dim=1)                                   # [B, D]
            value = self.head(h).squeeze(-1)                        # [B]

        if return_logits and return_value:
            return logits, value
        if return_logits:
            return logits
        return value
