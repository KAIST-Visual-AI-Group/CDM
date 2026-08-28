"""
Twist function psi^theta_t(x_t) -> scalar.

The twist head shares the denoising backbone: one forward pass produces the hidden states from
which both the logit head (the proposal) and the scalar twist head are read off, so evaluating
the twist during SMC costs almost nothing on top of ordinary sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MergedFrozenBackboneTwist(nn.Module):
    def __init__(
        self,
        backbone,
        sigma_processor,
        mask_index,
        neg_infinity,
        head_type: str = "linear",
        mlp_n_layers: int = 3,
        mlp_hidden_size: int = 256,
    ):
        super().__init__()
        hidden_size = backbone.config.model.hidden_size

        self.backbone = backbone
        self.sigma_processor = sigma_processor
        self.mask_index = mask_index
        self.neg_infinity = neg_infinity

        for p in self.backbone.parameters():
            p.requires_grad = False

        if head_type == "linear":
            head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.head = head

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
            self.head = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown head_type '{head_type}'. Choose 'linear' or 'mlp'.")

        self.head_type = head_type
        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, get_also_value=False, get_only_value=False):
        sigma = self.sigma_processor(sigma)
        bb = self.backbone

        with torch.no_grad():
            h = bb.vocab_embed(x)                        # [B, L, D]
            c = F.silu(bb.sigma_map(sigma))              # [B, cond_dim]
            rotary_cos_sin = bb.rotary_emb(h)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block in bb.blocks:
                    h = block(h, rotary_cos_sin, c, seqlens=None)
                logits = bb.output_layer(h, c) if not get_only_value else None
            logits = self._subs_parameterization(logits, x) if not get_only_value else None

        if get_only_value:
            value = self.head(h.float().mean(dim=1)).squeeze(-1)  # [B]
            return value

        if get_also_value:
            value = self.head(h.float().mean(dim=1)).squeeze(-1)  # [B]
            return logits, value
        else:
            return logits
