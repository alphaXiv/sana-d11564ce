"""Reduced-scale SANA-Video-2.0-style video DiT.

Implements, per the paper's disclosed design (reconstructed, weights unreleased):
  - gated bidirectional ReLU linear attention (SANA LiteLAReLURope style: kernel
    then RoPE, normalizer from un-rotated features) with sigmoid output gate
  - gated softmax anchors interleaved 3:1 (every 4th layer) in the hybrid
  - convolution-free SwiGLU FFN, adaLN-single timestep+class modulation
  - Block Attention Residuals (AttnRes): span-8 blocks, shared routing query per
    sublayer type across depth, timestep-independent router over sources
    {initial embedding, completed block summaries, running partial sum}
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants as C


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            x = x * self.weight.float()
        return x.to(dt)


def build_rope(T, H, W, device):
    """Factorized 3D RoPE freqs, complex (N, head_dim/2)."""
    freqs = []
    for n, d in zip((T, H, W), C.ROPE_DIMS):
        half = d // 2
        inv = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32, device=device) / half))
        t = torch.arange(n, dtype=torch.float32, device=device)
        f = torch.outer(t, inv)  # (n, half)
        freqs.append(torch.polar(torch.ones_like(f), f))
    ft, fh, fw = freqs
    ft = ft[:, None, None, :].expand(T, H, W, -1)
    fh = fh[None, :, None, :].expand(T, H, W, -1)
    fw = fw[None, None, :, :].expand(T, H, W, -1)
    return torch.cat([ft, fh, fw], dim=-1).reshape(T * H * W, -1)  # (N, 32) complex


def apply_rope(x, freqs):
    # x: (B, h, N, d) float32
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    out = torch.view_as_real(xc * freqs).flatten(-2)
    return out.type_as(x)


class LinearAttention(nn.Module):
    """Gated bidirectional ReLU linear attention (SANA style)."""

    kind = "linear"

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.gate = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.eps = 1e-15
        self.probe = None  # optionally dict to receive state matrices

    def forward(self, x, freqs):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # fp32 for numerical stability (SANA fp32_attention)
        q, k, v = q.float(), k.float(), v.float()
        q, k = F.relu(q), F.relu(k)
        q_rot, k_rot = apply_rope(q, freqs), apply_rope(k, freqs)
        # normalizer from un-rotated nonneg features (SANA LiteLAReLURope)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + self.eps)  # (B,h,N)
        state = torch.einsum("bhnd,bhne->bhde", k_rot, v)  # (B,h,d,d)
        out = torch.einsum("bhnd,bhde->bhne", q_rot, state) * z.unsqueeze(-1)
        if self.probe is not None:
            self.probe["state"] = state.detach()
            self.probe["out"] = out.detach()
        out = out.to(x.dtype).transpose(1, 2).reshape(B, N, -1)
        out = out * torch.sigmoid(self.gate(x))
        return self.proj(out)


class SoftmaxAttention(nn.Module):
    """Gated full softmax attention with RoPE."""

    kind = "softmax"

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.gate = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.probe = None

    def forward(self, x, freqs):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, freqs).to(v.dtype), apply_rope(k, freqs).to(v.dtype)
        out = F.scaled_dot_product_attention(q, k, v)
        if self.probe is not None:
            self.probe["out"] = out.detach().float()
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = out * torch.sigmoid(self.gate(x))
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class AttnResRouter(nn.Module):
    """Timestep-independent router with shared queries across depth (one per sublayer type)."""

    def __init__(self, dim):
        super().__init__()
        self.q = nn.ParameterDict(
            {"attn": nn.Parameter(torch.randn(dim) * 0.02), "ffn": nn.Parameter(torch.randn(dim) * 0.02)}
        )
        self.alpha = nn.ParameterDict(
            {"attn": nn.Parameter(torch.tensor(0.1)), "ffn": nn.Parameter(torch.tensor(0.1))}
        )
        self.norm = RMSNorm(dim, affine=False)
        self.scale = dim**-0.5

    def forward(self, kind, sources):
        # sources: list of (B,N,C)
        s = torch.stack(sources, dim=2)  # (B,N,S,C)
        sn = self.norm(s)
        logits = torch.einsum("bnsc,c->bns", sn.float(), self.q[kind].float()) * self.scale
        w = logits.softmax(dim=-1).to(s.dtype)  # (B,N,S)
        routed = torch.einsum("bns,bnsc->bnc", w, sn)
        return self.alpha[kind] * routed, w


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, attn_kind):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = (LinearAttention if attn_kind == "linear" else SoftmaxAttention)(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = SwiGLU(dim, C.FFN_DIM)
        self.layer_mod = nn.Parameter(torch.zeros(6 * dim))  # adaLN-single per-layer offset


class VideoDiT(nn.Module):
    def __init__(self, attn="linear", attnres=False, depth=C.DEPTH, dim=C.DIM, heads=C.HEADS):
        super().__init__()
        self.attnres = attnres
        self.depth = depth
        self.dim = dim
        pt, ph, pw = C.PATCH
        self.patch = nn.Conv3d(3, dim, kernel_size=C.PATCH, stride=C.PATCH)
        self.y_emb = nn.Embedding(C.NUM_CLASSES + 1, dim)  # last index = null (CFG dropout)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.global_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        kinds = []
        for i in range(depth):
            if attn == "softmax":
                kinds.append("softmax")
            elif attn == "hybrid":
                kinds.append("softmax" if i % C.SOFTMAX_EVERY == C.SOFTMAX_EVERY - 1 else "linear")
            else:
                kinds.append("linear")
        self.kinds = kinds
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, k) for k in kinds)
        self.router = AttnResRouter(dim) if attnres else None
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, pt * ph * pw * 3)
        self.rope_cache = {}
        self.probe_routing = None  # set to list to collect routing weights
        self.ablate_completed = False
        self._init()

    def _init(self):
        nn.init.zeros_(self.global_mod[1].weight)
        nn.init.zeros_(self.global_mod[1].bias)
        nn.init.zeros_(self.final_mod.weight)
        nn.init.zeros_(self.final_mod.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.y_emb.weight, std=0.02)

    def freqs_for(self, T, H, W, device):
        key = (T, H, W, str(device))
        if key not in self.rope_cache:
            self.rope_cache[key] = build_rope(T, H, W, device)
        return self.rope_cache[key]

    @staticmethod
    def t_embed(t, dim=256):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * 1000 * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, t, y):
        # x: (B, 3, F, H, W) in [-1,1]; t: (B,) in (0,1); y: (B,) long
        B = x.shape[0]
        pt, ph, pw = C.PATCH
        Ft, Hh, Ww = x.shape[2] // pt, x.shape[3] // ph, x.shape[4] // pw
        tokens = self.patch(x).flatten(2).transpose(1, 2)  # (B,N,C)
        freqs = self.freqs_for(Ft, Hh, Ww, x.device)
        cond = self.t_mlp(self.t_embed(t).to(tokens.dtype)) + self.y_emb(y)
        base_mod = self.global_mod(cond)  # (B, 6C)

        e0 = tokens
        xh = tokens
        partial = None
        completed = []
        if self.probe_routing is not None:
            self.probe_routing.clear()
        for i, blk in enumerate(self.blocks):
            block_entry = i % C.BLOCK_SPAN == 0
            if self.attnres and block_entry:
                if partial is not None:
                    completed.append(partial)
                partial = None
            sa, ga, sm, gm = None, None, None, None
            mod = (base_mod + blk.layer_mod).chunk(6, dim=-1)
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod

            def routed_for(kind):
                if not self.attnres:
                    return 0
                sources = [e0]
                if not (self.ablate_completed and block_entry):
                    sources += completed
                if partial is not None:
                    sources.append(partial)
                r, w = self.router(kind, sources)
                if self.probe_routing is not None:
                    self.probe_routing.append(
                        {"layer": i, "kind": kind, "n_completed": len(completed),
                         "has_partial": partial is not None,
                         "w": w.detach().float().mean(dim=(0, 1)).cpu()}
                    )
                return r

            h = modulate(blk.norm1(xh + routed_for("attn")), shift_a, scale_a)
            inc = gate_a.unsqueeze(1) * blk.attn(h, freqs)
            xh = xh + inc
            partial = inc if partial is None else partial + inc
            h = modulate(blk.norm2(xh + routed_for("ffn")), shift_f, scale_f)
            inc = gate_f.unsqueeze(1) * blk.ffn(h)
            xh = xh + inc
            partial = partial + inc

        shift, scale = self.final_mod(F.silu(cond)).chunk(2, dim=-1)
        out = self.out(modulate(self.norm_f(xh), shift, scale))  # (B,N,p*3)
        out = out.view(B, Ft, Hh, Ww, pt, ph, pw, 3)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, 3, Ft * pt, Hh * ph, Ww * pw)
        return out


def build_model(cfg):
    return VideoDiT(attn=cfg.ATTN, attnres=cfg.ATTNRES)
