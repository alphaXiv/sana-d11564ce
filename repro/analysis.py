"""Mechanistic probes: linear-attention state effective rank, token-feature rank,
AttnRes routing mass, and completed-block-source ablation."""

import json

import torch

from . import constants as C
from .model import LinearAttention


def effective_rank(mat):
    """exp(entropy of normalized singular values); mat (..., m, n) float32."""
    s = torch.linalg.svdvals(mat.float())
    p = s / (s.sum(dim=-1, keepdim=True) + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    return ent.exp()


@torch.no_grad()
def rank_probe(model, batch, t_values, deep_from=16, quick=False):
    """Per-layer linear-attention state erank + token-feature erank at given noise levels."""
    x0, y = batch
    device = next(model.parameters()).device
    x0, y = x0.to(device), y.to(device)
    results = []
    lin_layers = [i for i, k in enumerate(model.kinds) if k == "linear"]
    for tv in t_values:
        g = torch.Generator(device="cpu").manual_seed(1234)
        noise = torch.randn(x0.shape, generator=g).to(device)
        t = torch.full((x0.shape[0],), tv, device=device)
        xt = (1 - tv) * x0 + tv * noise
        probes = {}
        for i in lin_layers:
            model.blocks[i].attn.probe = probes.setdefault(i, {})
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            model(xt, t, y)
        per_layer = {}
        for i, p in probes.items():
            st_rank = effective_rank(p["state"]).mean().item()  # mean over batch, heads
            entry = {"state_erank": st_rank}
            if not quick:
                o = p["out"]  # (B,h,N,d)
                tok = o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1)
                entry["token_erank"] = effective_rank(tok).mean().item()
            per_layer[i] = entry
            model.blocks[i].attn.probe = None
        deep = [v["state_erank"] for i, v in per_layer.items() if i >= deep_from]
        results.append(
            {
                "t": tv,
                "deep_state_erank_mean": sum(deep) / len(deep),
                "per_layer": per_layer,
            }
        )
    return results


@torch.no_grad()
def routing_probe(model, batch, t_values):
    """Mean routing mass per source category, per layer and sublayer type."""
    if not model.attnres:
        return None
    x0, y = batch
    device = next(model.parameters()).device
    x0, y = x0.to(device), y.to(device)
    out = []
    for tv in t_values:
        g = torch.Generator(device="cpu").manual_seed(1234)
        noise = torch.randn(x0.shape, generator=g).to(device)
        t = torch.full((x0.shape[0],), tv, device=device)
        xt = (1 - tv) * x0 + tv * noise
        model.probe_routing = []
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            model(xt, t, y)
        entries = []
        for e in model.probe_routing:
            w = e["w"].tolist()  # ordered: [e0] + completed... + [partial?]
            n_c = e["n_completed"]
            rec = {
                "layer": e["layer"],
                "kind": e["kind"],
                "w_init": w[0],
                "w_completed": w[1 : 1 + n_c],
                "w_partial": w[1 + n_c] if e["has_partial"] else None,
            }
            entries.append(rec)
        model.probe_routing = None
        out.append({"t": tv, "entries": entries})
    return out


@torch.no_grad()
def ablation_probe(model, batch, t_values, deep_from=16):
    """Rank with completed-block sources removed at block-entry layers vs normal."""
    if not model.attnres:
        return None
    res = {}
    for mode in ["normal", "ablate_completed"]:
        model.ablate_completed = mode == "ablate_completed"
        r = rank_probe(model, batch, t_values, deep_from=deep_from)
        # also token rank at block-entry layer outputs (linear layers at block entries: 8, 16)
        res[mode] = r
    model.ablate_completed = False
    return res


def log_metric(kind, obj):
    obj = {"kind": kind, **obj}
    print(f"ORXMETRIC {json.dumps(obj)}", flush=True)
