"""FVD evaluation from saved checkpoints (EMA weights) for all four variants.

Loads /shared/sana2/ckpt/<run>/latest.pt, samples 1024 videos with the same
paired noise seeds as training-time eval, computes Frechet distance on I3D
features (contiguous-input fix) with r3d_18 fallback, and logs sample grids.
"""

import os

import numpy as np
import torch

from . import constants as C
from . import data
from .analysis import log_metric
from .model import VideoDiT
from .train import log_sample_grid, sample_videos

RUNS = [
    ("pure-linear-s0", "linear", False),
    ("hybrid25-s0", "hybrid", False),
    ("softmax-s0", "softmax", False),
    ("hybrid25-attnres-s0", "hybrid", True),
]


def main():
    device = torch.device("cuda")
    val_clips, val_labels = data.load_split("val")
    from .fvd import extract_features, frechet_distance, get_extractor

    extractor, name = get_extractor(device)
    # verify extractor works on a small real batch; fall back if not
    try:
        extract_features(val_clips[:4], extractor, name, device, bs=4)
    except Exception as e:
        print(f"[fvdeval] {name} failed at inference ({e}); falling back to r3d18", flush=True)
        from .fvd import _load_r3d

        extractor, name = _load_r3d(device)
    log_metric("fvd_extractor", {"extractor": name})
    real_feats = extract_features(val_clips, extractor, name, device)
    gen_labels = val_labels[: C.FVD_NUM_GEN]

    for run, attn, attnres in RUNS:
        ckpt_path = os.path.join(C.CKPT_DIR, run, "latest.pt")
        if not os.path.exists(ckpt_path):
            log_metric("fvd_skip", {"run": run, "reason": "no checkpoint"})
            continue
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = VideoDiT(attn=attn, attnres=attnres).to(device)
        model.load_state_dict(sd["ema"])
        model.eval()
        step = sd.get("step", -1)
        gen = sample_videos(model, gen_labels, device)
        log_sample_grid(gen, f"samples_{run}")
        gen_feats = extract_features(gen, extractor, name, device)
        fvd = frechet_distance(real_feats, gen_feats)
        log_metric(
            "fvd",
            {"run": run, "fvd": fvd, "extractor": name, "step": step, "n_real": len(val_clips), "n_gen": len(gen)},
        )
        del model
        torch.cuda.empty_cache()
    log_metric("run_complete", {"run": "fvdeval"})


if __name__ == "__main__":
    main()
