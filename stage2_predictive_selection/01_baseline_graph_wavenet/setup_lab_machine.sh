#!/usr/bin/env bash
# Setup for running Express_Links_Step1_Graph_WaveNet.ipynb unchanged on a Linux GPU lab machine.
#
# What it does:
#   1. Checks the NVIDIA GPU.
#   2. Creates a Python >= 3.11 environment (the notebook needs 3.11+ for hashlib.file_digest).
#   3. Installs PyTorch (CUDA build), numpy, pandas, scipy, tables, gdown and ipykernel.
#   4. Registers a Jupyter kernel called "Python (express-links)".
#   5. Makes /content usable, because the notebook writes to /content/... (Colab paths).
#      /content becomes a link to a folder in your home directory, so nothing in the code changes.
#   6. Checks internet access to GitHub and the Hugging Face mirror, which the notebook downloads from.
#   7. Runs a final import + GPU test inside the new environment.
#
# Usage:
#   bash setup_lab.sh
# Optional overrides:
#   ENV_DIR=$HOME/envs/express-links       where the environment goes
#   WORK_DIR=$HOME/express_links_content   where /content will really point (needs ~2 GB free)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu118   use this if the GPU driver is old (CUDA 11.x)

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/express-links}"
WORK_DIR="${WORK_DIR:-$HOME/express_links_content}"
KERNEL_NAME="express-links"
TORCH_INDEX="${TORCH_INDEX:-}"

say() { printf '\n== %s\n' "$*"; }

say "1. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
  echo "(If another user's job is already filling this GPU, training will be slower or run out of memory.)"
else
  echo "WARNING: nvidia-smi not found. The notebook needs an NVIDIA GPU for PROFILE=\"full\"."
fi

say "2. Python environment at $ENV_DIR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  PY=""
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PY="$(command -v "$candidate")"
      break
    fi
  done
  mkdir -p "$(dirname "$ENV_DIR")"
  if [ -n "$PY" ] && "$PY" -m venv "$ENV_DIR" 2>/dev/null; then
    echo "Created venv with $("$PY" --version)"
  elif command -v conda >/dev/null 2>&1; then
    rm -rf "$ENV_DIR"
    conda create -y -p "$ENV_DIR" python=3.11
  else
    echo "ERROR: need Python >= 3.11 (with venv) or conda. Ask the lab admin, or install Miniconda in your home folder."
    exit 1
  fi
else
  echo "Environment already exists, reusing it."
fi
ENV_PY="$ENV_DIR/bin/python"
"$ENV_PY" --version

say "3. Packages"
"$ENV_PY" -m pip install --upgrade pip
if [ -n "$TORCH_INDEX" ]; then
  "$ENV_PY" -m pip install torch --index-url "$TORCH_INDEX"
else
  "$ENV_PY" -m pip install torch
fi
"$ENV_PY" -m pip install numpy pandas scipy tables "gdown==5.2.0" ipykernel

say "4. Jupyter kernel"
"$ENV_PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

say "5. /content folder"
mkdir -p "$WORK_DIR"
if [ -d /content ] && [ -w /content ]; then
  echo "/content already exists and is writable: $(readlink -f /content)"
elif [ ! -e /content ] && ln -s "$WORK_DIR" /content 2>/dev/null; then
  echo "/content -> $WORK_DIR"
elif [ ! -e /content ] && sudo -n true 2>/dev/null; then
  sudo ln -s "$WORK_DIR" /content
  echo "/content -> $WORK_DIR (created with sudo)"
elif [ ! -e /content ] && [ -t 0 ] && command -v sudo >/dev/null 2>&1 && sudo ln -s "$WORK_DIR" /content; then
  echo "/content -> $WORK_DIR (created with sudo)"
else
  echo "WARNING: could not create a writable /content (no admin rights, or /content belongs to someone else)."
  echo "Ask the lab admin to run:   sudo ln -s $WORK_DIR /content"
  echo "Otherwise the notebook will stop at the configuration cell with a PermissionError."
fi

say "6. Internet access needed by the notebook"
"$ENV_PY" - <<'EOF'
import urllib.request
checks = {
    "GitHub (author code + graphs)": "https://raw.githubusercontent.com/nnzhan/Graph-WaveNet/6b162e80c59a1d494809252eca055cff93dc66b1/LICENSE",
    "Hugging Face mirror (speed data)": "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip",
}
for label, url in checks.items():
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=20) as response:
            print(f"OK    {label} ({response.status})")
    except Exception as exc:
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
print("If the mirror fails, copy metr-la.h5 and pems-bay.h5 to the machine and set MANUAL_DATA_DIR in the notebook.")
EOF

say "7. Final check inside the environment"
"$ENV_PY" - <<'EOF'
import hashlib, sys
import numpy, pandas, scipy, tables, gdown, torch
assert sys.version_info >= (3, 11), "Python 3.11+ required"
assert hasattr(hashlib, "file_digest")
print("python", sys.version.split()[0], "| torch", torch.__version__, "| numpy", numpy.__version__,
      "| pandas", pandas.__version__, "| scipy", scipy.__version__, "| tables", tables.__version__)
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    print("CUDA OK:", torch.cuda.get_device_name(0), "| test matmul sum", float((x @ x).sum()))
else:
    print("WARNING: torch cannot see a GPU. Check the driver (nvidia-smi). If the driver is older than CUDA 12,")
    print("rerun with: TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash setup_lab.sh")
EOF

say "Done"
echo "In Jupyter, open the notebook and pick the kernel: Python ($KERNEL_NAME)"
echo "Outputs will be saved in: $WORK_DIR/express_links_output"
