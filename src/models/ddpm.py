"""Denoising Diffusion Probabilistic Model (Ho et al., 2020).

Forward:  q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)
Training: L_simple = E_{t, x_0, eps} || eps - eps_theta(x_t, t) ||^2
Sampling: ancestral reverse process with sigma_t^2 = beta_t, t = T-1 ... 0.

eps_theta is a compact U-Net for 28x28 inputs (28 -> 14 -> 7, attention at 7x7).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


# ---------------------------------------------------------------- U-Net
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)   # (b, hw, hw)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """28x28 (64ch) -> 14x14 (128ch) -> 7x7 (128ch) + attention bottleneck, with skip connections."""

    def __init__(self, base=C.DDPM_BASE_CH, mults=(1, 2, 2), num_res=2):
        super().__init__()
        tdim = base * 4
        self.base = base
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, base, 3, padding=1)

        skip_ch, cur = [base], base
        self.downs = nn.ModuleList()
        for i, m in enumerate(mults):
            for _ in range(num_res):
                self.downs.append(ResBlock(cur, base * m, tdim))
                cur = base * m
                skip_ch.append(cur)
            if i != len(mults) - 1:
                self.downs.append(Downsample(cur))
                skip_ch.append(cur)

        self.mid1 = ResBlock(cur, cur, tdim)
        self.mid_attn = SelfAttention(cur)
        self.mid2 = ResBlock(cur, cur, tdim)

        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res + 1):
                self.ups.append(ResBlock(cur + skip_ch.pop(), base * m, tdim))
                cur = base * m
            if i != 0:
                self.ups.append(Upsample(cur))
        self.out = nn.Sequential(nn.GroupNorm(8, cur), nn.SiLU(), nn.Conv2d(cur, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.inc(x)
        hs = [h]
        for layer in self.downs:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, hs.pop()], 1), temb)
            else:
                h = layer(h)
        return self.out(h)


# ---------------------------------------------------------------- diffusion
class GaussianDiffusion(nn.Module):
    def __init__(self, model, T=C.DDPM_T, beta=C.DDPM_BETA):
        super().__init__()
        self.model = model
        self.T = T
        betas = torch.linspace(beta[0], beta[1], T, dtype=torch.float64)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        reg = lambda name, v: self.register_buffer(name, v.float())
        reg("betas", betas)
        reg("sqrt_abar", abar.sqrt())
        reg("sqrt_1m_abar", (1 - abar).sqrt())
        reg("coef_eps", betas / (1 - abar).sqrt())     # beta_t / sqrt(1 - abar_t)
        reg("sqrt_recip_alpha", 1.0 / alphas.sqrt())
        reg("sigma", betas.sqrt())

    def q_sample(self, x0, t, noise):
        return self.sqrt_abar[t][:, None, None, None] * x0 + self.sqrt_1m_abar[t][:, None, None, None] * noise

    def compute_loss(self, x0):
        """x0 in [-1, 1]."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, n, device, generator=None, keep_steps=None):
        """Returns x_0 in [0, 1]; with keep_steps also a dict {t: x_t in [0,1]}."""
        x = torch.randn(n, *C.IMG_SHAPE, device=device, generator=generator)
        traj = {}
        if keep_steps is not None and self.T in keep_steps:
            traj[self.T] = ((x + 1) / 2).clamp(0, 1).cpu()
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self.model(x, tt)
            mean = self.sqrt_recip_alpha[t] * (x - self.coef_eps[t] * eps)
            if t > 0:
                z = torch.randn(x.shape, device=device, generator=generator)
                x = mean + self.sigma[t] * z
            else:
                x = mean
            if keep_steps is not None and t in keep_steps:
                traj[t] = ((x + 1) / 2).clamp(0, 1).cpu()
        x0 = ((x + 1) / 2).clamp(0, 1)
        return (x0, traj) if keep_steps is not None else x0


class EMA:
    def __init__(self, model, decay=C.EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def build_ddpm():
    return GaussianDiffusion(UNet())
