import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Reproducing SANA-Video 2.0's hybrid attention at reduced scale

    [SANA-Video 2.0 (arXiv 2607.21553)](https://arxiv.org/abs/2607.21553) builds a video
    Diffusion Transformer whose attention is mostly **O(N) linear attention**, with every
    4th layer a full **softmax "anchor"** (25%), plus **Attention Residuals (AttnRes)**
    that route representations from completed 8-layer blocks to later layers.

    We tested the paper's two central claims from scratch at reduced scale — four 190M-parameter
    video DiTs on UCF-101 (16x64x64 pixel clips, 2048 tokens), identical except for their
    attention layers, trained with matched wall-clock budgets on a Kubernetes cluster of
    NVIDIA RTX PRO 6000 Blackwell GPUs (peak 16 concurrent, ~11.7h elapsed).

    **All result data below is embedded in this notebook** — logged during the Kubernetes runs —
    so nothing needs to be re-run. The full report lives at
    [`reports/sana-video-2-repro/report.md`](https://github.com/alphaXiv/sana-d11564ce/blob/main/reports/sana-video-2-repro/report.md).
    """)
    return


@app.cell
def _():
    # Result data logged by the Kubernetes runs (ORXMETRIC lines), embedded verbatim.
    VAL = {"pure-linear": [[2000, 0.10658], [4000, 0.08995], [6000, 0.08332], [8000, 0.07854], [10000, 0.07663], [12000, 0.07468], [14000, 0.07305], [16000, 0.0723], [18000, 0.07154], [20000, 0.07105], [22000, 0.07067], [24000, 0.0732], [26000, 0.06971], [28000, 0.07017]], "hybrid25": [[2000, 0.07923], [4000, 0.06752], [6000, 0.0634], [8000, 0.06043], [10000, 0.05934], [12000, 0.05779], [14000, 0.05781], [16000, 0.0573], [18000, 0.057], [20000, 0.05708], [22000, 0.05668], [24000, 0.05665], [26000, 0.05633], [28000, 0.05653], [30000, 0.05702]], "softmax": [[2000, 0.07819], [4000, 0.06542], [6000, 0.06187], [8000, 0.05921], [10000, 0.05787], [12000, 0.05664], [14000, 0.05617], [16000, 0.05589], [18000, 0.05525], [20000, 0.05531], [22000, 0.05511], [24000, 0.05569], [26000, 0.05446], [28000, 0.05546], [30000, 0.05499]], "hybrid25-attnres": [[2000, 0.07996], [4000, 0.06587], [6000, 0.06282], [8000, 0.05991], [10000, 0.05885], [12000, 0.05737], [14000, 0.05734], [16000, 0.05695]], "pure-linear-s1": [[2000, 0.10859], [4000, 0.09183], [6000, 0.08266], [8000, 0.0784], [10000, 0.07546], [12000, 0.07425], [14000, 0.07546], [16000, 0.07282]], "hybrid25-s1": [[2000, 0.07991], [4000, 0.06744], [6000, 0.06382], [8000, 0.06126], [10000, 0.05881], [12000, 0.05861], [14000, 0.05791], [16000, 0.05795]], "softmax-s1": [[2000, 0.081], [4000, 0.06776], [6000, 0.06203], [8000, 0.06047], [10000, 0.05731], [12000, 0.05713], [14000, 0.05616], [16000, 0.05633]], "hybrid25-attnres-s1": [[2000, 0.07867], [4000, 0.06689], [6000, 0.06263], [8000, 0.06061]]}
    BUCKETS = {"pure-linear": {"0.05": 0.1858, "0.15": 0.07867, "0.25": 0.05539, "0.35": 0.04597, "0.45": 0.0419, "0.55": 0.04084, "0.65": 0.04249, "0.75": 0.04761, "0.85": 0.06015, "0.95": 0.10133}, "softmax": {"0.05": 0.13828, "0.15": 0.053, "0.25": 0.03724, "0.35": 0.03177, "0.45": 0.03018, "0.55": 0.03095, "0.65": 0.03413, "0.75": 0.04074, "0.85": 0.0544, "0.95": 0.09843}, "hybrid25-attnres": {"0.05": 0.14294, "0.15": 0.05634, "0.25": 0.03972, "0.35": 0.03383, "0.45": 0.03192, "0.55": 0.03247, "0.65": 0.03563, "0.75": 0.0418, "0.85": 0.05564, "0.95": 0.09911}, "pure-linear-s1": {"0.05": 0.18866, "0.15": 0.08244, "0.25": 0.05826, "0.35": 0.04835, "0.45": 0.04378, "0.55": 0.0425, "0.65": 0.04354, "0.75": 0.04824, "0.85": 0.05979, "0.95": 0.10136}, "hybrid25-s1": {"0.05": 0.14442, "0.15": 0.0575, "0.25": 0.04048, "0.35": 0.03433, "0.45": 0.03235, "0.55": 0.03272, "0.65": 0.03555, "0.75": 0.04169, "0.85": 0.0553, "0.95": 0.09956}, "softmax-s1": {"0.05": 0.14265, "0.15": 0.05508, "0.25": 0.03847, "0.35": 0.03263, "0.45": 0.03073, "0.55": 0.03129, "0.65": 0.03427, "0.75": 0.04055, "0.85": 0.05437, "0.95": 0.09858}}
    BENCH = [{"arch": "pure-linear", "tokens": 2048, "ms": 16.0}, {"arch": "pure-linear", "tokens": 4096, "ms": 21.8}, {"arch": "pure-linear", "tokens": 8192, "ms": 38.4}, {"arch": "pure-linear", "tokens": 16384, "ms": 84.2}, {"arch": "pure-linear", "tokens": 32768, "ms": 199.7}, {"arch": "pure-linear", "tokens": 65536, "ms": 425.8}, {"arch": "hybrid-25", "tokens": 2048, "ms": 16.0}, {"arch": "hybrid-25", "tokens": 4096, "ms": 22.2}, {"arch": "hybrid-25", "tokens": 8192, "ms": 40.1}, {"arch": "hybrid-25", "tokens": 16384, "ms": 92.4}, {"arch": "hybrid-25", "tokens": 32768, "ms": 239.7}, {"arch": "hybrid-25", "tokens": 65536, "ms": 602.9}, {"arch": "hybrid-25-attnres", "tokens": 2048, "ms": 22.0}, {"arch": "hybrid-25-attnres", "tokens": 4096, "ms": 31.5}, {"arch": "hybrid-25-attnres", "tokens": 8192, "ms": 62.2}, {"arch": "hybrid-25-attnres", "tokens": 16384, "ms": 151.1}, {"arch": "hybrid-25-attnres", "tokens": 32768, "ms": 368.4}, {"arch": "hybrid-25-attnres", "tokens": 65536, "ms": 858.8}, {"arch": "full-softmax", "tokens": 2048, "ms": 14.8}, {"arch": "full-softmax", "tokens": 4096, "ms": 22.2}, {"arch": "full-softmax", "tokens": 8192, "ms": 46.0}, {"arch": "full-softmax", "tokens": 16384, "ms": 119.5}, {"arch": "full-softmax", "tokens": 32768, "ms": 368.6}, {"arch": "full-softmax", "tokens": 65536, "ms": 1169.8}]
    FVD = {"seed0": {"pure-linear": 1091.9, "hybrid25": 575.8, "softmax": 513.6, "hybrid25-attnres": 643.1}, "seed1": {"pure-linear": 1288.0, "hybrid25": 690.4, "softmax": 654.6}}
    ROUTING = [{"t": 0.999, "layer": 0, "init": 1.0, "completed": 0, "partial": 0}, {"t": 0.999, "layer": 1, "init": 0.474, "completed": 0, "partial": 0.526}, {"t": 0.999, "layer": 2, "init": 0.457, "completed": 0, "partial": 0.543}, {"t": 0.999, "layer": 3, "init": 0.449, "completed": 0, "partial": 0.551}, {"t": 0.999, "layer": 4, "init": 0.466, "completed": 0, "partial": 0.534}, {"t": 0.999, "layer": 5, "init": 0.466, "completed": 0, "partial": 0.534}, {"t": 0.999, "layer": 6, "init": 0.474, "completed": 0, "partial": 0.526}, {"t": 0.999, "layer": 7, "init": 0.497, "completed": 0, "partial": 0.503}, {"t": 0.999, "layer": 8, "init": 0.602, "completed": 0.398, "partial": 0}, {"t": 0.999, "layer": 9, "init": 0.149, "completed": 0.099, "partial": 0.752}, {"t": 0.999, "layer": 10, "init": 0.121, "completed": 0.08, "partial": 0.799}, {"t": 0.999, "layer": 11, "init": 0.138, "completed": 0.091, "partial": 0.771}, {"t": 0.999, "layer": 12, "init": 0.161, "completed": 0.106, "partial": 0.733}, {"t": 0.999, "layer": 13, "init": 0.148, "completed": 0.098, "partial": 0.754}, {"t": 0.999, "layer": 14, "init": 0.145, "completed": 0.096, "partial": 0.759}, {"t": 0.999, "layer": 15, "init": 0.21, "completed": 0.139, "partial": 0.652}, {"t": 0.999, "layer": 16, "init": 0.332, "completed": 0.668, "partial": 0}, {"t": 0.999, "layer": 17, "init": 0.121, "completed": 0.245, "partial": 0.634}, {"t": 0.999, "layer": 18, "init": 0.098, "completed": 0.199, "partial": 0.703}, {"t": 0.999, "layer": 19, "init": 0.116, "completed": 0.235, "partial": 0.649}, {"t": 0.999, "layer": 20, "init": 0.119, "completed": 0.239, "partial": 0.642}, {"t": 0.999, "layer": 21, "init": 0.118, "completed": 0.238, "partial": 0.643}, {"t": 0.999, "layer": 22, "init": 0.119, "completed": 0.24, "partial": 0.641}, {"t": 0.999, "layer": 23, "init": 0.125, "completed": 0.251, "partial": 0.624}, {"t": 0.5, "layer": 0, "init": 1.0, "completed": 0, "partial": 0}, {"t": 0.5, "layer": 1, "init": 0.328, "completed": 0, "partial": 0.672}, {"t": 0.5, "layer": 2, "init": 0.315, "completed": 0, "partial": 0.685}, {"t": 0.5, "layer": 3, "init": 0.305, "completed": 0, "partial": 0.695}, {"t": 0.5, "layer": 4, "init": 0.328, "completed": 0, "partial": 0.672}, {"t": 0.5, "layer": 5, "init": 0.335, "completed": 0, "partial": 0.665}, {"t": 0.5, "layer": 6, "init": 0.344, "completed": 0, "partial": 0.656}, {"t": 0.5, "layer": 7, "init": 0.358, "completed": 0, "partial": 0.642}, {"t": 0.5, "layer": 8, "init": 0.469, "completed": 0.531, "partial": 0}, {"t": 0.5, "layer": 9, "init": 0.13, "completed": 0.151, "partial": 0.719}, {"t": 0.5, "layer": 10, "init": 0.103, "completed": 0.12, "partial": 0.777}, {"t": 0.5, "layer": 11, "init": 0.111, "completed": 0.129, "partial": 0.76}, {"t": 0.5, "layer": 12, "init": 0.127, "completed": 0.147, "partial": 0.726}, {"t": 0.5, "layer": 13, "init": 0.12, "completed": 0.139, "partial": 0.741}, {"t": 0.5, "layer": 14, "init": 0.123, "completed": 0.142, "partial": 0.735}, {"t": 0.5, "layer": 15, "init": 0.147, "completed": 0.17, "partial": 0.683}, {"t": 0.5, "layer": 16, "init": 0.248, "completed": 0.752, "partial": 0}, {"t": 0.5, "layer": 17, "init": 0.103, "completed": 0.32, "partial": 0.577}, {"t": 0.5, "layer": 18, "init": 0.085, "completed": 0.264, "partial": 0.651}, {"t": 0.5, "layer": 19, "init": 0.1, "completed": 0.306, "partial": 0.594}, {"t": 0.5, "layer": 20, "init": 0.098, "completed": 0.301, "partial": 0.601}, {"t": 0.5, "layer": 21, "init": 0.097, "completed": 0.296, "partial": 0.607}, {"t": 0.5, "layer": 22, "init": 0.096, "completed": 0.293, "partial": 0.611}, {"t": 0.5, "layer": 23, "init": 0.096, "completed": 0.294, "partial": 0.61}, {"t": 0.05, "layer": 0, "init": 1.0, "completed": 0, "partial": 0}, {"t": 0.05, "layer": 1, "init": 0.253, "completed": 0, "partial": 0.747}, {"t": 0.05, "layer": 2, "init": 0.24, "completed": 0, "partial": 0.76}, {"t": 0.05, "layer": 3, "init": 0.243, "completed": 0, "partial": 0.757}, {"t": 0.05, "layer": 4, "init": 0.284, "completed": 0, "partial": 0.716}, {"t": 0.05, "layer": 5, "init": 0.283, "completed": 0, "partial": 0.717}, {"t": 0.05, "layer": 6, "init": 0.287, "completed": 0, "partial": 0.713}, {"t": 0.05, "layer": 7, "init": 0.294, "completed": 0, "partial": 0.706}, {"t": 0.05, "layer": 8, "init": 0.349, "completed": 0.651, "partial": 0}, {"t": 0.05, "layer": 9, "init": 0.152, "completed": 0.28, "partial": 0.568}, {"t": 0.05, "layer": 10, "init": 0.143, "completed": 0.263, "partial": 0.594}, {"t": 0.05, "layer": 11, "init": 0.144, "completed": 0.264, "partial": 0.592}, {"t": 0.05, "layer": 12, "init": 0.147, "completed": 0.269, "partial": 0.584}, {"t": 0.05, "layer": 13, "init": 0.144, "completed": 0.264, "partial": 0.592}, {"t": 0.05, "layer": 14, "init": 0.147, "completed": 0.269, "partial": 0.584}, {"t": 0.05, "layer": 15, "init": 0.16, "completed": 0.293, "partial": 0.547}, {"t": 0.05, "layer": 16, "init": 0.201, "completed": 0.799, "partial": 0}, {"t": 0.05, "layer": 17, "init": 0.121, "completed": 0.476, "partial": 0.403}, {"t": 0.05, "layer": 18, "init": 0.111, "completed": 0.439, "partial": 0.45}, {"t": 0.05, "layer": 19, "init": 0.117, "completed": 0.461, "partial": 0.422}, {"t": 0.05, "layer": 20, "init": 0.116, "completed": 0.456, "partial": 0.428}, {"t": 0.05, "layer": 21, "init": 0.115, "completed": 0.454, "partial": 0.431}, {"t": 0.05, "layer": 22, "init": 0.116, "completed": 0.457, "partial": 0.427}, {"t": 0.05, "layer": 23, "init": 0.12, "completed": 0.475, "partial": 0.404}]
    return BENCH, BUCKETS, FVD, ROUTING, VAL


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim 1a — held-out denoising loss

    Held-out flow-matching loss (mean over 10 noise buckets, 2048 UCF-101 test clips with
    fixed noise), evaluated every 2000 steps. The pure-linear model plateaus well above the
    hybrid and softmax models, which are nearly indistinguishable — at both seeds.
    """)
    return


@app.cell
def _(VAL):
    import matplotlib.pyplot as plt

    COLORS = {
        "pure-linear": "#2a78d6",
        "hybrid25": "#1baf7a",
        "hybrid25-attnres": "#4a3aa7",
        "softmax": "#e34948",
    }
    LABELS = {
        "pure-linear": "Pure linear",
        "hybrid25": "Hybrid 25% softmax",
        "hybrid25-attnres": "Hybrid + AttnRes",
        "softmax": "Full softmax",
    }
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _ax, _sfx, _title in [
        (_axes[0], "", "Seed 0 (5.8h budget)"),
        (_axes[1], "-s1", "Seed 1 (3.25h budget)"),
    ]:
        for _name in ["pure-linear", "hybrid25", "hybrid25-attnres", "softmax"]:
            _key = _name + _sfx
            if _key not in VAL or not VAL[_key]:
                continue
            _xs, _ys = zip(*VAL[_key])
            _ax.plot(_xs, _ys, "-o", ms=3, lw=1.8, color=COLORS[_name], label=LABELS[_name])
        _ax.set_xlabel("Training step")
        _ax.set_ylabel("Held-out loss")
        _ax.set_title(_title, fontsize=10)
        _ax.grid(alpha=0.25, lw=0.5)
        _ax.legend(fontsize=8, frameon=False)
    _fig.tight_layout()
    _fig
    return COLORS, LABELS, plt


@app.cell
def _(BUCKETS, COLORS, LABELS, mo, plt):
    _fig2, _ax2 = plt.subplots(figsize=(6.5, 4))
    for _name2, _run in [
        ("pure-linear", "pure-linear"),
        ("softmax", "softmax"),
        ("hybrid25-attnres", "hybrid25-attnres"),
    ]:
        if _run not in BUCKETS:
            continue
        _ts = sorted(float(k) for k in BUCKETS[_run])
        _ax2.plot(
            _ts,
            [BUCKETS[_run][f"{t:.2f}"] for t in _ts],
            "-o",
            ms=4,
            lw=1.8,
            color=COLORS[_name2],
            label=LABELS[_name2],
        )
    _ax2.set_xlabel("Noise level t (1 = pure noise)")
    _ax2.set_ylabel("Held-out loss in bucket")
    _ax2.set_title("Final timestep-stratified held-out loss (seed 0)", fontsize=10)
    _ax2.grid(alpha=0.25, lw=0.5)
    _ax2.legend(fontsize=8.5, frameon=False)
    mo.vstack([_fig2, mo.md("The gap over pure-linear holds at every noise level of the diffusion process.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim 1b — generation quality (FVD)

    Frechet Video Distance with the standard StyleGAN-V I3D features: 1024 generated videos
    (40-step Euler, EMA weights, paired noise seeds across variants) against the full
    7562-clip UCF-101 test split. The hybrid closes ~89-94% of the linear-to-softmax gap.
    Absolute values are high (small pixel-space models, hours of training) — the relative
    ordering is the evidence.
    """)
    return


@app.cell
def _(FVD, plt):
    import numpy as np

    _names = ["pure-linear", "hybrid25", "hybrid25-attnres", "softmax"]
    _fig3, _ax3 = plt.subplots(figsize=(7, 4))
    _x = np.arange(len(_names))
    _s0 = [FVD["seed0"].get(n) for n in _names]
    _s1 = [FVD["seed1"].get(n) for n in _names]
    _ax3.bar(_x - 0.18, [v or 0 for v in _s0], width=0.34, color="#2a78d6", label="seed 0 (~30k steps; AttnRes 17k)")
    _ax3.bar(_x + 0.18, [v or 0 for v in _s1], width=0.34, color="#9fc5ec", label="seed 1 (~17k steps)")
    for _xi, _v in zip(_x - 0.18, _s0):
        if _v:
            _ax3.text(_xi, _v, f"{_v:.0f}", ha="center", va="bottom", fontsize=8)
    for _xi, _v in zip(_x + 0.18, _s1):
        if _v:
            _ax3.text(_xi, _v, f"{_v:.0f}", ha="center", va="bottom", fontsize=8)
    _ax3.set_xticks(_x, ["Pure linear", "Hybrid 25%", "Hybrid+AttnRes", "Full softmax"])
    _ax3.set_ylabel("FVD vs UCF-101 test split (lower is better)")
    _ax3.grid(alpha=0.25, lw=0.5, axis="y")
    _ax3.legend(fontsize=8.5, frameon=False)
    _fig3
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim 1c — long-sequence scaling

    Forward latency of the 190M DiT (bf16, batch 1, eager PyTorch, one RTX PRO 6000 Blackwell)
    from 2,048 to 65,536 tokens. Pure linear grows near-linearly; full softmax shows its
    quadratic term; the hybrid's speedup over softmax **grows with sequence length**,
    reaching 1.94x at 65k tokens — the trend the paper reports across resolutions.
    """)
    return


@app.cell
def _(BENCH, plt):
    _archs = {
        "pure-linear": ("#2a78d6", "Pure linear"),
        "hybrid-25": ("#1baf7a", "Hybrid 25% softmax"),
        "hybrid-25-attnres": ("#4a3aa7", "Hybrid + AttnRes"),
        "full-softmax": ("#e34948", "Full softmax"),
    }
    _fig4, _axes4 = plt.subplots(1, 2, figsize=(11, 4))
    _sm = {b["tokens"]: b["ms"] for b in BENCH if b["arch"] == "full-softmax"}
    for _a, (_c, _l) in _archs.items():
        _pts = sorted((b["tokens"], b["ms"]) for b in BENCH if b["arch"] == _a)
        _tk, _ms = zip(*_pts)
        _axes4[0].plot(_tk, _ms, "-o", ms=4, lw=1.8, color=_c, label=_l)
        if _a != "full-softmax":
            _axes4[1].plot(_tk, [_sm[t] / m for t, m in _pts], "-o", ms=4, lw=1.8, color=_c, label=_l)
    _axes4[0].set_xscale("log", base=2)
    _axes4[0].set_yscale("log")
    _axes4[0].set_xlabel("Sequence length (tokens)")
    _axes4[0].set_ylabel("Forward latency (ms)")
    _axes4[0].legend(fontsize=8, frameon=False)
    _axes4[0].grid(alpha=0.25, lw=0.5)
    _axes4[1].axhline(1.0, color="#888", lw=1, ls="--")
    _axes4[1].set_xscale("log", base=2)
    _axes4[1].set_xlabel("Sequence length (tokens)")
    _axes4[1].set_ylabel("Speedup over full softmax (x)")
    _axes4[1].legend(fontsize=8, frameon=False)
    _axes4[1].grid(alpha=0.25, lw=0.5)
    _fig4.tight_layout()
    _fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim 2 — AttnRes routing reuse

    The AttnRes router mixes, at every layer, three kinds of sources: the initial token
    embedding, summaries of *completed* 8-layer blocks, and the running partial sum of the
    current block. Below: the trained router's attention-branch routing mass per layer at
    maximum noise. At block-entry layers (8, 16 — dotted) the partial sum resets, and mass
    shifts sharply onto completed-block summaries; in the deepest block only ~12-14% of mass
    remains on the initial embedding.
    """)
    return


@app.cell
def _(ROUTING, plt):
    _entries = [r for r in ROUTING if r["t"] > 0.9]
    _layers = [e["layer"] for e in _entries]
    _init = [e["init"] for e in _entries]
    _comp = [e["completed"] for e in _entries]
    _part = [e["partial"] for e in _entries]
    _fig5, _ax5 = plt.subplots(figsize=(8, 4))
    _ax5.stackplot(
        _layers,
        _init,
        _comp,
        _part,
        colors=["#9aa4b2", "#4a3aa7", "#1baf7a"],
        labels=["initial embedding", "completed blocks", "partial sum (current block)"],
        alpha=0.9,
    )
    for _be in (8, 16):
        _ax5.axvline(_be, color="white", lw=1, ls=":")
    _ax5.set_xlabel("Layer")
    _ax5.set_ylabel("Routing weight share")
    _ax5.set_title("AttnRes attention-branch routing mass (t=0.999, trained model)", fontsize=10)
    _ax5.set_xlim(0, max(_layers))
    _ax5.set_ylim(0, 1)
    _ax5.legend(fontsize=8, frameon=False, loc="lower left")
    _fig5
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Claim | Assessment |
    |---|---|
    | Hybrid improves held-out loss over pure-linear | **Reproduced** (both seeds; ~86% of the gap to softmax closed) |
    | Hybrid improves generation quality (FVD) | **Reproduced** (closes 89-94% of the linear-to-softmax gap) |
    | Hybrid keeps better long-sequence scaling than softmax | **Reproduced** (speedup grows to 1.94x at 65k tokens) |
    | AttnRes reuses completed-block representations | **Reproduced** (29-50% deep-layer routing mass on completed blocks) |
    | AttnRes raises deep-layer effective rank | **Partially** (+20% at matched early step; gap closes late in training) |
    | AttnRes keeps the latency advantage | **Partially** (1.36x faster than softmax at 65k, but +42% over plain hybrid in our unfused implementation) |

    Caveats: 190M pixel-space models on UCF-101 vs the paper's 5B latent models on a
    proprietary corpus; eager PyTorch latency constants; the AttnRes router is a
    reconstruction from the paper's prose. Full details, sample grids, and limitations:
    [report](https://github.com/alphaXiv/sana-d11564ce/blob/main/reports/sana-video-2-repro/report.md).
    """)
    return


if __name__ == "__main__":
    app.run()
