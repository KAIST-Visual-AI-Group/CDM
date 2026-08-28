"""Noise schedules for discrete diffusion (DNA).

Copied from texts_mdm/noise_schedule.py — same continuous-time schedules.
"""

import abc

import torch
import torch.nn as nn

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_noise(config, dtype=torch.float32):
    if config.noise.type == 'geometric':
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    elif config.noise.type == 'loglinear':
        return LogLinearNoise()
    elif config.noise.type == 'cosine':
        return CosineNoise()
    elif config.noise.type == 'linear':
        return Linear(config.noise.sigma_min, config.noise.sigma_max, dtype)
    else:
        raise ValueError(f'{config.noise.type} is not a valid noise')


class Noise(abc.ABC, nn.Module):
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        pass


class CosineNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
        scale = torch.pi / 2
        return scale * sin / (cos + self.eps)

    def total_noise(self, t):
        cos = torch.cos(t * torch.pi / 2)
        return -torch.log(self.eps + (1 - self.eps) * cos)


class Linear(Noise):
    def __init__(self, sigma_min=0, sigma_max=10, dtype=torch.float32):
        super().__init__()
        self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
        self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

    def rate_noise(self, t):
        return self.sigma_max - self.sigma_min

    def total_noise(self, t):
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)


class GeometricNoise(Noise):
    def __init__(self, sigma_min=1e-3, sigma_max=1):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (
            self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
    """Log-linear noise schedule: 1 - 1/e^(n(t)) interpolates 0 to ~1."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)
