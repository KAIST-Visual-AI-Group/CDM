"""
Twist head architectures for the DNA MDLM (CNN backbone).

The head reads the frozen backbone's last hidden state and returns a scalar value.
The paper's DNA setting uses ``positional_mlp`` (MLP+PE): sinusoidal position embeddings are
added to the per-token features before a shared MLP, because the CNN backbone carries no
explicit positional information.
"""

import torch
import torch.nn as nn


class PositionalMLPHead(nn.Module):
    """Position-aware per-position MLP + pool to scalar.

    Adds a fixed sinusoidal positional embedding to each hidden vector
    *before* the per-position MLP, so the MLP sees a position-tagged
    input. Each position produces its own score [B, L] that is reduced
    to a scalar via mean (default) or sum — so each position can say
    "this motif is here at THIS location, it's worth X".
    """

    def __init__(self, hidden_dim, seq_len, mlp_hidden_size, n_layers, reduce="mean"):
        super().__init__()
        assert reduce in ("mean", "sum"), f"Unknown reduce: {reduce}"
        self.reduce = reduce

        # Fixed sinusoidal PE — no learnable params, lets the MLP see position.
        self.register_buffer(
            "pos_embed", _sinusoidal_position_embedding(seq_len, hidden_dim),
            persistent=False,
        )

        layers = []
        in_dim = hidden_dim
        for _ in range(n_layers - 1):
            lin = nn.Linear(in_dim, mlp_hidden_size)
            nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.ReLU()])
            in_dim = mlp_hidden_size
        final = nn.Linear(in_dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):
        # h: [B, L, hidden_dim]
        L = h.shape[1]
        h = h + self.pos_embed[:L]               # inject position into each token
        per_pos = self.mlp(h).squeeze(-1)        # [B, L]
        if self.reduce == "mean":
            return per_pos.mean(dim=-1)
        return per_pos.sum(dim=-1)


def _sinusoidal_position_embedding(seq_len: int, hidden_dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal positional embedding.

    PE[pos, 2i]   = sin(pos / 10000^(2i / D))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / D))

    Returns a [seq_len, hidden_dim] tensor (no learnable params).
    """
    import math
    pe = torch.zeros(seq_len, hidden_dim)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_dim)
    )                                                                       # [D/2]
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerHead(nn.Module):
    """Transformer encoder head: PE + self-attention → mean-pool → scalar.

    Uses standard PyTorch TransformerEncoder. Lets each position attend
    to others, so the head can capture long-range interactions (e.g.,
    motif co-occurrence) that MLP heads cannot.

    Pipeline:
        h + sinusoidal_PE  →  [encoder × n_layers]  →  mean over L  →  Linear → scalar
    """

    def __init__(
        self,
        hidden_dim,
        seq_len,
        n_layers,
        n_heads,
        ffn_dim,
        dropout=0.0,
        out_head_type: str = "linear",
        out_mlp_n_layers: int = 2,
        out_mlp_hidden_size: int = None,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )
        assert out_head_type in ("linear", "mlp"), (
            f"out_head_type must be 'linear' or 'mlp', got '{out_head_type}'"
        )
        self.register_buffer(
            "pos_embed", _sinusoidal_position_embedding(seq_len, hidden_dim),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final projection: linear (default) or MLP → scalar.
        if out_head_type == "linear":
            self.out = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            mlp_hidden = out_mlp_hidden_size if out_mlp_hidden_size is not None else hidden_dim
            layers = []
            in_dim = hidden_dim
            for _ in range(out_mlp_n_layers - 1):
                lin = nn.Linear(in_dim, mlp_hidden)
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.GELU()])
                in_dim = mlp_hidden
            final = nn.Linear(in_dim, 1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.out = nn.Sequential(*layers)

    def forward(self, h):
        # h: [B, L, hidden_dim]
        L = h.shape[1]
        h = h + self.pos_embed[:L]
        h = self.encoder(h)          # [B, L, hidden_dim]
        h = h.mean(dim=1)            # [B, hidden_dim]
        return self.out(h).squeeze(-1)


class PosEmbedMeanPoolHead(nn.Module):
    """Add a sinusoidal positional embedding, then mean-pool, then MLP -> scalar.

    Simpler than PositionalMLPHead: each position's hidden vector is tagged
    with a fixed (non-learnable) sinusoidal embedding before mean-pooling,
    so the pooled representation retains positional information that vanilla
    mean-pool discards. A small MLP then maps the pooled vector to a scalar.
    """

    def __init__(self, hidden_dim, seq_len, mlp_hidden_size, n_layers, head_type="mlp"):
        super().__init__()
        # Fixed sinusoidal embedding — buffer (no grad), moves with .to(device)
        self.register_buffer(
            "pos_embed", _sinusoidal_position_embedding(seq_len, hidden_dim),
            persistent=False,
        )

        if head_type == "linear":
            head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.head = head
        else:
            layers = []
            in_dim = hidden_dim
            for _ in range(n_layers - 1):
                lin = nn.Linear(in_dim, mlp_hidden_size)
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.ReLU()])
                in_dim = mlp_hidden_size
            final = nn.Linear(in_dim, 1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.head = nn.Sequential(*layers)

    def forward(self, h):
        # h: [B, L, hidden_dim]
        L = h.shape[1]
        h = h + self.pos_embed[:L]   # broadcast over batch
        h = h.mean(dim=1)            # [B, hidden_dim]
        return self.head(h).squeeze(-1)


def _build_twist_head(
    head_type,
    hidden_size,
    mlp_n_layers,
    mlp_hidden_size,
    seq_len=None,
    n_heads=None,
):
    """Create a twist head module.

    head_type:
        linear           — mean-pool over L, then Linear → scalar
        mlp              — mean-pool over L, then MLP → scalar
        positional_mlp   — add sinusoidal PE → per-position MLP → mean over L
        pos_embed        — add sinusoidal PE → mean-pool → MLP → scalar
        transformer      — add sinusoidal PE → encoder → mean-pool → Linear → scalar
        transformer_mlp  — add sinusoidal PE → encoder → mean-pool → MLP → scalar
                           (encoder: mlp_n_layers blocks, mlp_hidden_size ffn dim,
                            n_heads attention heads;
                            final MLP: 2 layers with hidden = encoder hidden_dim)
    """
    if head_type == "linear":
        head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return head
    if head_type == "mlp":
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
        return nn.Sequential(*layers)
    if head_type == "positional_mlp":
        assert seq_len is not None, "positional_mlp head requires seq_len"
        return PositionalMLPHead(hidden_size, seq_len, mlp_hidden_size, mlp_n_layers)
    if head_type == "pos_embed":
        assert seq_len is not None, "pos_embed head requires seq_len"
        return PosEmbedMeanPoolHead(hidden_size, seq_len, mlp_hidden_size, mlp_n_layers)
    if head_type in ("transformer", "transformer_mlp"):
        assert seq_len is not None, f"{head_type} head requires seq_len"
        assert n_heads is not None, f"{head_type} head requires n_heads"
        return TransformerHead(
            hidden_dim=hidden_size,
            seq_len=seq_len,
            n_layers=mlp_n_layers,
            n_heads=n_heads,
            ffn_dim=mlp_hidden_size,
            out_head_type="mlp" if head_type == "transformer_mlp" else "linear",
            out_mlp_n_layers=2,
            out_mlp_hidden_size=hidden_size,
        )
    raise ValueError(f"Unknown head_type '{head_type}'.")


_FULL_HIDDEN_HEADS = ("positional_mlp", "pos_embed", "transformer", "transformer_mlp")


class FrozenBackboneTwist(nn.Module):
    """Twist using frozen CNN backbone hidden states + trainable head."""

    needs_all_hidden = False

    def __init__(
        self,
        backbone,
        sigma_processor,
        head_type: str = "linear",
        mlp_n_layers: int = 3,
        mlp_hidden_size: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        hidden_size = backbone.config.hidden_dim

        self.backbone = backbone
        self.sigma_processor = sigma_processor

        for p in self.backbone.parameters():
            p.requires_grad = False

        seq_len = getattr(backbone.config, 'length', None)
        self.head = _build_twist_head(
            head_type, hidden_size, mlp_n_layers, mlp_hidden_size,
            seq_len=seq_len, n_heads=n_heads,
        )
        self.head_type = head_type
        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.n_heads = n_heads

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the head to hidden states [B, L, D] and return [B].

        Dispatches pooling based on head_type so the fused forward path
        can share the same logic without re-running the backbone.
        """
        h = h.float()
        if self.head_type in _FULL_HIDDEN_HEADS:
            return self.head(h)
        return self.head(h.mean(dim=1)).squeeze(-1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_processor(sigma)
        with torch.no_grad():
            h = self.backbone.get_hidden_states(x, sigma)  # [B, L, hidden_dim]
        return self._apply_head(h)
