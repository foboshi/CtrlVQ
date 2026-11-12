

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, int32
import logging

logger = logging.getLogger(__name__)


def round_ste(z: Tensor) -> Tensor:
    
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(nn.Module):
    

    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        scale: Optional[float] = None
    ):
        super().__init__()

        
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        
        
        
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        
        self.dim = dim if dim is not None else effective_codebook_dim

        
        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections else nn.Identity()
        )
        self.has_projections = has_projections

        
        self.codebook_size = self._levels.prod().item()

        
        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook,
                              persistent=False)

        logger.info(
            f"FSQ 初始化: levels={levels}, codebook_size={self.codebook_size}, "
            f"dim={self.dim}, num_codebooks={num_codebooks}"
        )

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor, use_gumbel=False, temperature=1.0) -> Tensor:
        
        bounded = self.bound(z)
        half_width = self._levels // 2

        if use_gumbel and self.training:
            
            
            half_width_expanded = half_width.float()
            if half_width_expanded.ndim == 0:
                half_width_expanded = half_width_expanded.unsqueeze(0)
            logits = bounded / half_width_expanded
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            soft_probs = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
            
            
            quantized = (soft_probs * bounded).sum(dim=-1, keepdim=True)
        else:
            
            quantized = round_ste(bounded)

        
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor,
                          project_out: bool = True) -> Tensor:
        
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.num_codebooks > 1:
            codes = codes.reshape(*codes.shape[:-2], -1)

        if project_out:
            codes = self.project_out(codes)

        return codes

    def forward(self, z: Tensor, use_gumbel=False, temperature=1.0):
        
        
        original_shape = z.shape
        need_reshape = z.ndim == 2

        if need_reshape:
            z = z.unsqueeze(1)  

        assert z.shape[-1] == self.dim, \
            f"期望维度 {self.dim}，但输入维度为 {z.shape[-1]}"

        
        z = self.project_in(z)

        
        if self.num_codebooks > 1:
            z = z.reshape(*z.shape[:-1], self.num_codebooks, self.codebook_dim)

        
        codes = self.quantize(z, use_gumbel=use_gumbel, temperature=temperature)
        indices = self.codes_to_indices(codes)

        
        if self.num_codebooks > 1:
            codes = codes.reshape(*codes.shape[:-2], -1)

        
        out = self.project_out(codes)

        if need_reshape:
            out = out.squeeze(1)  
            indices = indices.squeeze(1)  

        return out, indices

    def get_codebook_usage(self, indices: Tensor) -> float:
        
        unique_indices = torch.unique(indices)
        return unique_indices.numel() / self.codebook_size


class FSQAutoEncoder(nn.Module):
    

    def __init__(self, in_dim: int, latent_dim: int,
                 levels: List[int], num_codebooks: int = 1):
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim

        
        
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1),
        )

        
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
        )

        
        self.fsq = FSQ(
            levels=levels,
            dim=latent_dim,
            num_codebooks=num_codebooks,
        )

        
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, in_dim),
        )

        logger.info(
            f"FSQAutoEncoder 初始化: in_dim={in_dim}, latent_dim={latent_dim}, "
            f"levels={levels}, codebook_size={self.fsq.codebook_size}"
        )

    def encode(self, x: Tensor, use_adapter=True):
        
        z = self.encoder(x)
        
        if use_adapter:
            z = self.adapter(z)
        z_q, indices = self.fsq(z)
        return z_q, indices, z

    def decode(self, z_q: Tensor):
        
        return self.decoder(z_q)

    def forward(self, x: Tensor, use_adapter=True, use_gumbel=False, temperature=1.0):
        
        z = self.encoder(x)
        
        if use_adapter:
            z = self.adapter(z)

        
        z_q, indices = self.fsq(z, use_gumbel=use_gumbel, temperature=temperature)
        x_recon = self.decode(z_q)
        return x_recon, indices, z, z_q

    @torch.no_grad()
    def get_codes(self, x: Tensor):
        
        self.eval()
        z = self.encoder(x)
        _, indices = self.fsq(z)
        return indices
