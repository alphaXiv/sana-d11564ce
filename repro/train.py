"""Flow-matching trainer (DDP) with timestep-stratified held-out loss, EMA,
end-of-run generation eval (FVD), latency microbench, and mechanistic probes.

Evidence channel is stdout: ORXMETRIC {json} lines.
"""

import base64
import copy
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import config as cfg
from . import constants as C
from . import data
from .analysis import ablation_probe, log_metric, rank_probe, routing_probe
from .model import build_model


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def to_float(clips):
    # uint8 (B,T,H,W,3) -> float (B,3,T,H,W) in [-1,1]
    x = clips.permute(0, 4, 1, 2, 3).float() / 127.5 - 1.0
    return x


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips, labels):
        self.clips, self.labels = clips, labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(self.clips[i])), int(self.labels[i])


@torch.no_grad()
def stratified_val(model, val_clips, val_labels, device, max_clips=C.VAL_LOSS_CLIPS, bs=64):
    model.eval()
    n = min(max_clips, len(val_clips))
    bucket_losses = {}
    for bi, tv in enumerate(C.VAL_BUCKETS):
        losses = []
        for s in range(0, n, bs):
            clips = torch.from_numpy(val_clips[s : s + bs]).to(device)
            y = torch.from_numpy(val_labels[s : s + bs]).to(device)
            x0 = to_float(clips)
            g = torch.Generator(device="cpu").manual_seed(bi * 100003 + s)
            noise = torch.randn(x0.shape, generator=g).to(device)
            t = torch.full((x0.shape[0],), tv, device=device)
            xt = (1 - tv) * x0 + tv * noise
            target = noise - x0
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(xt, t, y)
            losses.append(F.mse_loss(pred.float(), target).item())
        bucket_losses[f"{tv:.2f}"] = float(np.mean(losses))
    model.train()
    mean = float(np.mean(list(bucket_losses.values())))
    return mean, bucket_losses


@torch.no_grad()
def sample_videos(model, labels, device, steps=C.SAMPLE_STEPS, bs=C.SAMPLE_BS, seed0=777):
    """Euler sampler t=1 -> 0. Returns uint8 (N,T,H,W,3). Paired noise via seeds."""
    model.eval()
    out = []
    for s in range(0, len(labels), bs):
        y = torch.from_numpy(labels[s : s + bs]).to(device)
        b = len(y)
        g = torch.Generator(device="cpu").manual_seed(seed0 + s)
        x = torch.randn((b, 3, C.FRAMES, C.RES, C.RES), generator=g).to(device)
        ts = torch.linspace(1.0, 0.0, steps + 1)
        for i in range(steps):
            t = torch.full((b,), float(ts[i]), device=device)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                v = model(x, t, y)
            x = x - (float(ts[i]) - float(ts[i + 1])) * v.float()
        vid = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        out.append(vid.permute(0, 2, 3, 4, 1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def log_sample_grid(videos, tag, n_vids=8, n_frames=8):
    try:
        from PIL import Image

        idx = np.linspace(0, videos.shape[1] - 1, n_frames).astype(int)
        rows = [np.concatenate([v[i] for i in idx], axis=1) for v in videos[:n_vids]]
        grid = np.concatenate(rows, axis=0)
        buf = io.BytesIO()
        Image.fromarray(grid).save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        print(f"ORXPNG {tag} {b64}", flush=True)
    except Exception as e:
        print(f"[warn] sample grid failed: {e}", flush=True)


@torch.no_grad()
def latency_microbench(model, device):
    """Quick forward-latency check at the training shape (batch 1)."""
    model.eval()
    x = torch.randn(1, 3, C.FRAMES, C.RES, C.RES, device=device)
    t = torch.full((1,), 0.5, device=device)
    y = torch.zeros(1, dtype=torch.long, device=device)
    times = []
    with torch.autocast("cuda", torch.bfloat16):
        for i in range(8):
            torch.cuda.synchronize()
            t0 = time.time()
            model(x, t, y)
            torch.cuda.synchronize()
            if i >= 3:
                times.append(time.time() - t0)
    model.train()
    return float(np.mean(times) * 1000)


def save_ckpt(path, model, ema, opt, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step},
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def main():
    data.ensure_data()  # before dist init: prep can exceed NCCL timeouts
    if "RANK" in os.environ:
        from datetime import timedelta

        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.SEED * 1000 + rank)
    np.random.seed(cfg.SEED * 1000 + rank)

    train_clips, train_labels = data.load_split("train")
    val_clips, val_labels = data.load_split("val")
    if is_main():
        log_metric("data", {"train_clips": len(train_clips), "val_clips": len(val_clips)})

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main():
        log_metric(
            "arch",
            {
                "run": cfg.RUN_NAME,
                "attn": cfg.ATTN,
                "attnres": cfg.ATTNRES,
                "seed": cfg.SEED,
                "params": n_params,
                "kinds": model.kinds,
                "depth": C.DEPTH,
                "dim": C.DIM,
                "tokens": (C.FRAMES // C.PATCH[0]) * (C.RES // C.PATCH[1]) ** 2,
                "world": world,
            },
        )

    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    raw = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    opt = torch.optim.AdamW(raw.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY, betas=C.BETAS)

    ds = ClipDataset(train_clips, train_labels)
    sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=cfg.SEED) if world > 1 else None
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=C.BS_PER_GPU,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # resume if checkpoint exists (job restart safety)
    ckpt_path = os.path.join(C.CKPT_DIR, cfg.RUN_NAME, "latest.pt")
    step = 0
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        raw.load_state_dict(sd["model"])
        ema.load_state_dict(sd["ema"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
        if is_main():
            log_metric("resume", {"step": step})

    # fixed probe batch (held-out)
    probe_batch = (
        to_float(torch.from_numpy(val_clips[:16])),
        torch.from_numpy(val_labels[:16]),
    )

    train_start = time.time()
    epoch = step * world * C.BS_PER_GPU // max(1, len(ds))
    model.train()
    t_log = time.time()
    loss_acc = []
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for clips, y in loader:
            lr = C.LR * min(1.0, (step + 1) / C.WARMUP)
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            clips, y = clips.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x0 = to_float(clips)
            # CFG label dropout
            drop = torch.rand(y.shape, device=device) < C.CLASS_DROP
            y = torch.where(drop, torch.full_like(y, C.NUM_CLASSES), y)
            # logit-normal timestep density
            t = torch.sigmoid(torch.randn(x0.shape[0], device=device))
            noise = torch.randn_like(x0)
            tt = t.view(-1, 1, 1, 1, 1)
            xt = (1 - tt) * x0 + tt * noise
            target = noise - x0
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(xt, t, y)
            loss = F.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), C.GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                d = C.EMA_DECAY
                for pe, pm in zip(ema.parameters(), raw.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), raw.buffers()):
                    be.copy_(bm)
            loss_acc.append(loss.item())
            step += 1

            if step % C.LOG_EVERY == 0 and is_main():
                dt = time.time() - t_log
                sps = C.LOG_EVERY / dt
                log_metric(
                    "train",
                    {
                        "step": step,
                        "loss": float(np.mean(loss_acc)),
                        "steps_per_s": sps,
                        "clips_per_s": sps * C.BS_PER_GPU * world,
                        "elapsed_s": time.time() - train_start,
                    },
                )
                t_log = time.time()
                loss_acc = []

            if step % C.VAL_EVERY == 0 and is_main():
                mean, buckets = stratified_val(raw, val_clips, val_labels, device)
                log_metric("val", {"step": step, "val_loss": mean, "buckets": buckets})

            if step % C.RANK_PROBE_EVERY == 0 and is_main() and any(k == "linear" for k in raw.kinds):
                r = rank_probe(raw, probe_batch, [0.999, 0.5], quick=True)
                log_metric(
                    "rank_quick",
                    {"step": step, "res": [{"t": e["t"], "deep": e["deep_state_erank_mean"]} for e in r]},
                )
                raw.train()

            if step % C.CKPT_EVERY == 0 and is_main():
                save_ckpt(ckpt_path, raw, ema, opt, step)

            if time.time() - train_start > C.TRAIN_TIME_S or step >= C.MAX_STEPS:
                done = True
                break
        epoch += 1

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()  # free non-main ranks before long single-GPU evals
    if rank == 0:
        save_ckpt(ckpt_path, raw, ema, opt, step)
        log_metric("train_done", {"step": step, "train_wall_s": time.time() - train_start})

        # ---- final evals (rank 0 only) ----
        mean, buckets = stratified_val(raw, val_clips, val_labels, device)
        log_metric("val_final", {"step": step, "val_loss": mean, "buckets": buckets, "weights": "raw"})
        mean_e, buckets_e = stratified_val(ema, val_clips, val_labels, device)
        log_metric("val_final", {"step": step, "val_loss": mean_e, "buckets": buckets_e, "weights": "ema"})

        ms = latency_microbench(raw, device)
        log_metric("train_shape_latency_ms", {"ms": ms})

        # generation + FVD (EMA weights, paired noise across variants)
        gen_labels = val_labels[: C.FVD_NUM_GEN]
        t0 = time.time()
        gen = sample_videos(ema, gen_labels, device)
        log_metric("sampling_done", {"n": len(gen), "wall_s": time.time() - t0})
        log_sample_grid(gen, f"samples_{cfg.RUN_NAME}")
        try:
            from .fvd import compute_fvd

            fvd_val, extractor = compute_fvd(val_clips, gen, device)
            log_metric("fvd", {"fvd": fvd_val, "extractor": extractor, "n_real": len(val_clips), "n_gen": len(gen)})
        except Exception as e:
            log_metric("fvd_error", {"error": str(e)})

        # mechanistic probes
        if any(k == "linear" for k in raw.kinds):
            r = rank_probe(raw, probe_batch, [0.999, 0.75, 0.5, 0.25, 0.05])
            log_metric("rank_full", {"step": step, "res": r})
        if raw.attnres:
            rp = routing_probe(raw, probe_batch, [0.999, 0.5, 0.05])
            log_metric("routing", {"step": step, "res": rp})
            ab = ablation_probe(raw, probe_batch, [0.999, 0.5])
            log_metric(
                "ablation",
                {
                    "step": step,
                    "normal": [{"t": e["t"], "deep": e["deep_state_erank_mean"], "per_layer": e["per_layer"]} for e in ab["normal"]],
                    "ablate_completed": [
                        {"t": e["t"], "deep": e["deep_state_erank_mean"], "per_layer": e["per_layer"]} for e in ab["ablate_completed"]
                    ],
                },
            )
        log_metric("run_complete", {"run": cfg.RUN_NAME, "step": step})


if __name__ == "__main__":
    main()
