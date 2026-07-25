"""Long-sequence forward-latency scaling benchmark (single GPU).

Latency is weight-independent: random-init models, bf16 autocast, batch 1,
frames 16..512 at 64x64 (2048..65536 tokens). All four architectures.
"""

import time

import numpy as np
import torch

from . import constants as C
from .analysis import log_metric
from .model import VideoDiT


@torch.no_grad()
def bench_arch(name, attn, attnres, device):
    model = VideoDiT(attn=attn, attnres=attnres).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    for frames in C.BENCH_FRAMES:
        tokens = (frames // C.PATCH[0]) * (C.RES // C.PATCH[1]) * (C.RES // C.PATCH[2])
        try:
            x = torch.randn(1, 3, frames, C.RES, C.RES, device=device)
            t = torch.full((1,), 0.5, device=device)
            y = torch.zeros(1, dtype=torch.long, device=device)
            torch.cuda.reset_peak_memory_stats()
            times = []
            with torch.autocast("cuda", torch.bfloat16):
                for i in range(C.BENCH_WARMUP + C.BENCH_ITERS):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    model(x, t, y)
                    torch.cuda.synchronize()
                    if i >= C.BENCH_WARMUP:
                        times.append(time.time() - t0)
            log_metric(
                "bench",
                {
                    "arch": name,
                    "params": n_params,
                    "frames": frames,
                    "tokens": tokens,
                    "ms_mean": float(np.mean(times) * 1000),
                    "ms_std": float(np.std(times) * 1000),
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                },
            )
        except torch.cuda.OutOfMemoryError:
            log_metric("bench", {"arch": name, "frames": frames, "tokens": tokens, "oom": True})
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()


def main():
    device = torch.device("cuda")
    log_metric("bench_env", {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
    for name, attn, attnres in [
        ("pure-linear", "linear", False),
        ("hybrid-25", "hybrid", False),
        ("hybrid-25-attnres", "hybrid", True),
        ("full-softmax", "softmax", False),
    ]:
        bench_arch(name, attn, attnres, device)
    log_metric("run_complete", {"run": "bench"})


if __name__ == "__main__":
    main()
