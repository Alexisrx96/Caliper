#!/usr/bin/env bash
# LCE environment bootstrap — idempotent. Foundation spec §6.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_DIR="models"

echo "== [1/4] Baseline audit =="
date -Is
uname -r
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
else
  echo "WARNING: nvidia-smi not found — CUDA build will fail" >&2
fi
if ! command -v nvcc >/dev/null && [ ! -x /opt/cuda/bin/nvcc ]; then
  echo "WARNING: nvcc not found. Install CUDA toolkit first: sudo pacman -S cuda" >&2
fi
free -h | head -2

echo "== [2/4] uv =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== [3/4] Dependencies (CUDA compile of llama-cpp-python: 10-20 min) =="
export PATH="/opt/cuda/bin:$PATH"   # Arch installs nvcc here
CMAKE_ARGS="-DGGML_CUDA=on" uv sync

echo "== [4/4] Model download (skipped if present) =="
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Model already present: $MODEL_DIR/$MODEL_FILE"
else
  uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$MODEL_REPO', filename='$MODEL_FILE', local_dir='$MODEL_DIR')
"
fi

echo "== Verify CUDA offload support =="
uv run python -c "
from llama_cpp import llama_supports_gpu_offload
assert llama_supports_gpu_offload(), 'llama-cpp-python built WITHOUT GPU support'
print('GPU offload: OK')
"
echo "Setup complete."

# Troubleshooting:
# If uv reused a cached CPU-only wheel of llama-cpp-python, force a rebuild:
#   CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python
