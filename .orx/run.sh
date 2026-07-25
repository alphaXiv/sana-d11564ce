#!/bin/bash
set -euo pipefail
echo "=== preflight ==="
date -u
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"
df -h /shared || true
pip install -q --no-input av huggingface_hub pillow scipy einops 2>&1 | tail -1 || pip install -q av huggingface_hub pillow scipy einops
export HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=8
MODE=$(python -c "from repro import config; print(config.MODE)")
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "MODE=$MODE NGPU=$NGPU"
if [ "$MODE" = "bench" ]; then
  python -m repro.run
else
  torchrun --standalone --nproc_per_node="$NGPU" -m repro.run
fi
echo "=== run.sh finished ==="
