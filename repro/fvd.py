"""FVD: Frechet distance on I3D features (standard StyleGAN-V torchscript I3D).

Fallback: torchvision r3d_18 penultimate features (reported as FVD* if used).
Videos: uint8 (B, T, H, W, 3).
"""

import numpy as np
import torch


def _load_i3d(device):
    from huggingface_hub import hf_hub_download

    from . import constants as C

    path = hf_hub_download("flateon/FVD-I3D-torchscript", "i3d_torchscript.pt", cache_dir=C.HF_CACHE)
    return torch.jit.load(path).eval().to(device), "i3d"


def _load_r3d(device):
    from torchvision.models.video import R3D_18_Weights, r3d_18

    m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).eval().to(device)
    m.fc = torch.nn.Identity()
    return m, "r3d18"


def get_extractor(device):
    try:
        return _load_i3d(device)
    except Exception as e:
        print(f"[fvd] I3D load failed ({e}); falling back to r3d18", flush=True)
        return _load_r3d(device)


@torch.no_grad()
def extract_features(videos, model, name, device, bs=32):
    feats = []
    for i in range(0, len(videos), bs):
        v = torch.from_numpy(videos[i : i + bs]).to(device)
        v = v.permute(0, 4, 1, 2, 3).contiguous().float()  # (B,3,T,H,W); i3d jit needs contiguous
        if name == "i3d":
            f = model(v, rescale=True, resize=True, return_features=True)
        else:
            import torch.nn.functional as F

            v = v / 255.0
            mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
            v = (v - mean) / std
            B, Cc, T, H, W = v.shape
            v = v.transpose(1, 2).reshape(B * T, Cc, H, W)
            v = F.interpolate(v, size=(112, 112), mode="bilinear", align_corners=False)
            v = v.view(B, T, Cc, 112, 112).transpose(1, 2)
            f = model(v)
        feats.append(f.cpu().float().numpy())
    return np.concatenate(feats)


def frechet_distance(f1, f2):
    from scipy import linalg

    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def compute_fvd(real_videos, gen_videos, device):
    model, name = get_extractor(device)
    fr = extract_features(real_videos, model, name, device)
    fg = extract_features(gen_videos, model, name, device)
    return frechet_distance(fr, fg), name
