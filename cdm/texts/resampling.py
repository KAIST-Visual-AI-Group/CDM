"""SSP (Srinivasan Sampling Process) resampling for SMC.

`ssp` is verbatim from Chopin's `particles` library
(Gerber, Chopin & Whiteley 2019, Ann. Statist. 47(4): 2236-2260).
Two thin torch wrappers convert [N]/[B,N] torch weight tensors to LongTensor indices.
"""
import numpy as np
import torch
from numba import jit
from numpy import random


@jit(nopython=True)
def ssp(W, M):
    """SSP resampling. Verbatim from Gerber, Chopin & Whiteley (2019)."""
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = MW - nr_children
    u = random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1.0 - xi[i])
        delta_j = min(xi[i], 1.0 - xi[j])
        sum_delta = delta_i + delta_j
        pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
        if u[k] < pj:
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1.0 - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    if np.sum(nr_children) != M:
        raise ValueError("ssp resampling: wrong size for output")
    return np.arange(N).repeat(nr_children)


def ssp_resample(weights, M):
    """1-D SSP. weights: torch [N] (non-negative). Returns LongTensor [M]."""
    w_np = weights.detach().cpu().numpy().astype(np.float64)
    w_np = w_np / w_np.sum()
    idx = ssp(w_np, int(M))
    return torch.from_numpy(idx.astype(np.int64)).to(weights.device)


def ssp_resample_batch(weights_mat, M):
    """Per-row SSP. weights_mat: torch [B, N]. Returns LongTensor [B, M]."""
    B = weights_mat.shape[0]
    out = torch.empty(B, M, dtype=torch.long, device=weights_mat.device)
    for b in range(B):
        out[b] = ssp_resample(weights_mat[b], M)
    return out
