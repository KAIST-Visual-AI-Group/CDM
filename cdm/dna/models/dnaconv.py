"""
CNN backbone for DNA discrete diffusion.

Adapted from: https://github.com/HannesStark/dirichlet-flow-matching
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class CNNModel(nn.Module):
    """
    Dilated CNN for DNA sequence diffusion.

    Args:
        config: model config with hidden_dim, num_cnn_stacks, dropout, clean_data,
                cls_free_guidance attributes.
        alphabet_size: number of nucleotide tokens (typically 4 for ACGT).
        num_cls: number of classes (for optional classifier-free guidance).
        classifier: if True, build a classifier head instead of denoiser.
    """

    def __init__(self, config, alphabet_size, num_cls=2, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.config = config
        self.classifier = classifier
        self.num_cls = num_cls

        if self.config.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=config.hidden_dim)
        else:
            inp_size = self.alphabet_size
            self.linear = nn.Conv1d(inp_size, config.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(
                GaussianFourierProjection(embed_dim=config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )

        self.num_layers = 5 * config.num_cnn_stacks
        self.convs = [
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        self.convs = nn.ModuleList(
            [copy.deepcopy(layer) for layer in self.convs for _ in range(config.num_cnn_stacks)]
        )
        self.time_layers = nn.ModuleList(
            [Dense(config.hidden_dim, config.hidden_dim) for _ in range(self.num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(self.num_layers)]
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim if classifier else self.alphabet_size,
                kernel_size=1,
            ),
        )
        self.dropout = nn.Dropout(config.dropout)

        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, self.num_cls),
            )

        if self.config.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(
                num_embeddings=self.num_cls + 1, embedding_dim=config.hidden_dim
            )
            self.cls_layers = nn.ModuleList(
                [Dense(config.hidden_dim, config.hidden_dim) for _ in range(self.num_layers)]
            )

    def forward(self, seq, t, cls=None, return_embedding=False):
        # Support both integer indices and one-hot input
        if not (seq.ndim > 2 and seq.shape[-1] == self.alphabet_size):
            seq = F.one_hot(seq, num_classes=self.alphabet_size).float()

        if self.config.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.config.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.config.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.config.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h

        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)

        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat

    def forward_fused(self, seq, t, cls=None, return_all_hidden=False):
        """
        Single backbone pass returning both logits and hidden states.

        Returns: (logits [B, L, alphabet_size], hidden)
            hidden = [B, L, hidden_dim] (default), or
                    [num_layers + 1, B, L, hidden_dim] when return_all_hidden=True
                    (index 0 is post-input projection; index i+1 is output of block i).
        """
        if not (seq.ndim > 2 and seq.shape[-1] == self.alphabet_size):
            seq = F.one_hot(seq, num_classes=self.alphabet_size).float()

        if self.config.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.config.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        hidden_states = [feat.permute(0, 2, 1)] if return_all_hidden else None

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.config.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.config.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
            if return_all_hidden:
                hidden_states.append(feat.permute(0, 2, 1))

        logits = self.final_conv(feat)
        logits = logits.permute(0, 2, 1)    # [B, L, alphabet_size]
        if return_all_hidden:
            return logits, torch.stack(hidden_states, dim=0)  # [num_layers + 1, B, L, hidden_dim]
        return logits, feat.permute(0, 2, 1)

    def get_hidden_states(self, seq, t, cls=None):
        """
        Extract hidden states before the final conv layer.
        Used by twist models to get intermediate representations.

        Returns: [B, L, hidden_dim]
        """
        if not (seq.ndim > 2 and seq.shape[-1] == self.alphabet_size):
            seq = F.one_hot(seq, num_classes=self.alphabet_size).float()

        if self.config.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.config.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.config.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.config.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h

        return feat.permute(0, 2, 1)  # [B, L, hidden_dim]
