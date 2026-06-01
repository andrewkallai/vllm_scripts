#!/usr/bin/env bash

# Exit on any error and print each command as it runs
set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

# Path to the Python interpreter (must be 3.10–3.12 for ROCm builds)
PYTHON_BIN="/home/users/andrewka/sw/python/Python-3.12.13_install/bin/python3.12"

# ROCm version for the PyTorch index URL
# Valid options: rocm7.0, rocm7.2, etc.
ROCM_PYTORCH_URL="https://download.pytorch.org/whl/rocm7.2"

# vLLM clone settings
VLLM_VERSION="deepseek-v4_rocm"
VLLM_BRANCH="v0.21.0"
#VLLM_BRANCH="main"
VLLM_ROOT="${HOME}/sw/vLLM"
VENV_DIR="${VLLM_ROOT}/vllm_envs/${VLLM_VERSION}_vllm_rocm"
VLLM_DIR="${VLLM_ROOT}/vllm-${VLLM_BRANCH}"
VLLM_REPO="https://github.com/vllm-project/vllm.git"

# GPU architecture to target (run `rocminfo | grep gfx` to find yours)
# Common values: gfx90a (MI200), gfx942 (MI300), gfx950 (MI350)
PYTORCH_ROCM_ARCH="gfx942"

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
echo "Creating virtual environment using $PYTHON_BIN..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install uv
echo "Upgrading pip and installing uv..."
pip install --upgrade pip uv

# ---------------------------------------------------------------------------
# 2. Install prerequisites: ninja, cmake, wheel, pybind11
# ---------------------------------------------------------------------------
module purge
module load rocm/7.2.0
module load gcc/13.2
module load binutils/2.34
module load cuda/13.0
module list
echo "Installing build prerequisites..."
uv pip install ninja cmake wheel pybind11 Cython setuptools

# ---------------------------------------------------------------------------
# 3. Install PyTorch for ROCm
# ---------------------------------------------------------------------------
echo "Installing PyTorch for ROCm..."
uv pip install torch torchvision torchao --index-url "${ROCM_PYTORCH_URL}"

# ---------------------------------------------------------------------------
# 4. Build & install LLVM+MLIR from FlyDSL (ROCm FlyDSL — Flexible LaYout DSL)
# ---------------------------------------------------------------------------
# echo "Building FlyDSL (v0.1.1.dev409)..."
# FLYDSL_ROOT="/tmp/flydsl"
# TARGET_COMMIT="cf3ab625dfc81ebf997fc17bde77b387eb1efe6d"
# if [ -d "$FLYDSL_ROOT" ]; then
#     echo "FlyDSL directory already exists. Updating..."
#     cd "$FLYDSL_ROOT"
#     git fetch --depth 512 origin "$TARGET_COMMIT"
#     git checkout "$TARGET_COMMIT"
# else
#     git clone --depth 512 https://github.com/ROCm/FlyDSL.git "$FLYDSL_ROOT"
#     cd "$FLYDSL_ROOT"
#     git checkout "$TARGET_COMMIT"
# fi
# "$FLYDSL_ROOT/scripts/build_llvm.sh" -j"$(($(nproc)-2))"
# # "$FLYDSL_ROOT/scripts/build.sh" -j"$(($(nproc)-2))"
# # uv pip install -e "$FLYDSL_ROOT"
# ## rm -rf "$FLYDSL_ROOT"


# # ---------------------------------------------------------------------------
# # 5. Build & install Triton for ROCm
# # ---------------------------------------------------------------------------
# export CMAKE_PREFIX_PATH=/tmp/llvm-project/build-flydsl
# echo "Building Triton for ROCm..."
# uv pip install ninja cmake wheel pybind11
# uv pip uninstall -y triton
# #if [ -d "/tmp/triton-rocm-latest" ]; then
# if [ -d "/tmp/triton-rocm" ]; then
#     echo "Triton directory already exists."
# else
#     git clone https://github.com/ROCm/triton.git /tmp/triton-rocm
#     #git clone https://github.com/triton-lang/triton.git /tmp/triton-rocm-latest
# fi
# #cd /tmp/triton-rocm-latest
# cd /tmp/triton-rocm
# # git pull origin main
# git checkout ba5c1517
# git cherry-pick 555d04f
# git cherry-pick dd998b6
# #uv pip install -r python/requirements.txt # build-time dependencies
# if [ ! -f setup.py ]; then cd python; fi
# python3 setup.py bdist_wheel --dist-dir=dist
# #python3 setup.py install
# #uv cache clean
# #uv pip install -e .
# cd /tmp
# ## rm -rf /tmp/triton-rocm-latest

# # ---------------------------------------------------------------------------
# # 6. Build & install Flash Attention for ROCm (Triton backend)
# #    https://github.com/Dao-AILab/flash-attention#triton-backend
# # ---------------------------------------------------------------------------
# echo "Building Flash Attention for ROCm (Triton backend)..."
# FA_ROOT="/tmp/flash-attention"
# if [ -d "$FA_ROOT" ]; then
#     echo "Flash Attention directory already exists."
# else
#     git clone https://github.com/Dao-AILab/flash-attention.git "$FA_ROOT"
# fi
# cd "$FA_ROOT"
# git checkout 0e60e394
# git submodule update --init
# FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE" FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" uv pip install --no-build-isolation .

# # ---------------------------------------------------------------------------
# # 7. Clone vLLM repository
# # ---------------------------------------------------------------------------
echo "Cloning vLLM repository (branch $VLLM_BRANCH) into $VLLM_DIR..."
mkdir -p "$VLLM_ROOT"
if [ -d "$VLLM_DIR" ]; then
    echo "vLLM directory already exists, skipping clone."
else
    git clone --branch "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_DIR"
fi
cd "$VLLM_DIR"
git pull origin "$VLLM_BRANCH"

# ---------------------------------------------------------------------------
# 8. Install vLLM ROCm build dependencies
# ---------------------------------------------------------------------------
echo "Installing vLLM ROCm build dependencies..."
#uv pip install /opt/rocm/share/amd_smi
uv pip install /home/users/andrewka/my_repos/mlgo_mlir/amd_smi
uv pip install --upgrade numba scipy huggingface-hub setuptools_scm
uv pip install -r requirements/rocm.txt

# ---------------------------------------------------------------------------
# 9. Build vLLM from source for ROCm
# ---------------------------------------------------------------------------
echo "Building vLLM for ROCm (arch=$PYTORCH_ROCM_ARCH)..."
export PYTORCH_ROCM_ARCH="$PYTORCH_ROCM_ARCH"
python setup.py develop

# ---------------------------------------------------------------------------
# 10. Build AMD Quark from source for ROCm
# ---------------------------------------------------------------------------
# if [ -d "/tmp/Quark" ]; then
#     echo "Quark directory already exists, skipping clone."
# else
#     git clone --recursive https://github.com/AMD/Quark /tmp/Quark
# fi
# cd /tmp/Quark

# # [Optional] run git submodule if you are updating an existing Quark repository
# git submodule sync
# git submodule update --init --recursive
# uv pip install .

echo ""
echo "============================================"
echo "vLLM ROCm build complete!"
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "============================================"
