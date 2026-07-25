"""Shared constants for the SANA-Video 2.0 reduced-scale reproduction.

Identical on every experiment branch; per-variant knobs live in repro/config.py.
"""

# Paths (shared PVC mounted at /shared on k8s; overridden implicitly by absence locally)
CACHE = "/shared/sana2"
DATA_DIR = f"{CACHE}/data"
CKPT_DIR = f"{CACHE}/ckpt"
HF_CACHE = f"{CACHE}/hf"

# Data
RES = 64
FRAMES = 16
FRAME_STRIDE = 2  # sample every 2nd frame (~12.5 fps from 25 fps UCF-101)
TRAIN_CLIPS_PER_VIDEO = 4
VAL_CLIPS_PER_VIDEO = 2
NUM_CLASSES = 101
DATA_VERSION = "v1"

# Model
PATCH = (2, 4, 4)  # (t, h, w) -> 8 x 16 x 16 = 2048 tokens
DEPTH = 24
DIM = 768
HEADS = 12
FFN_DIM = 2048
BLOCK_SPAN = 8  # AttnRes block span S
SOFTMAX_EVERY = 4  # hybrid: layers i with i % 4 == 3 are softmax anchors (25%)
CLASS_DROP = 0.1
ROPE_DIMS = (16, 24, 24)  # per-axis rope dims over head_dim=64

# Training
BS_PER_GPU = 16
LR = 2e-4
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
WARMUP = 1000
GRAD_CLIP = 1.0
MAX_STEPS = 1_000_000  # effectively unbounded; time budget governs
TRAIN_TIME_S = 21000  # ~5.8h of pure training, then final evals
EMA_DECAY = 0.9995
LOG_EVERY = 100
VAL_EVERY = 2000
RANK_PROBE_EVERY = 5000
CKPT_EVERY = 5000

# Validation / generation
VAL_LOSS_CLIPS = 2048
VAL_BUCKETS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
FVD_NUM_GEN = 1024
SAMPLE_STEPS = 40
SAMPLE_BS = 128

# Latency benchmark
BENCH_FRAMES = [16, 32, 64, 128, 256, 512]
BENCH_ITERS = 10
BENCH_WARMUP = 3
