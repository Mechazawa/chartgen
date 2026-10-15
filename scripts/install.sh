#!/usr/bin/env bash
# Set up chartgen: create .venv, install the matching torch build and every
# dependency, then run the self-checks.
#
#   scripts/install.sh                   # CUDA build if nvidia-smi works, else CPU
#   scripts/install.sh --cpu             # force the CPU build
#   scripts/install.sh --cuda-tag cu128  # a newer CUDA build for newer drivers
#   scripts/install.sh --training        # also install the training stack
#
# Set PYTHON to pick the interpreter (default: python3).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

use_cuda=auto
# cu128/cu129 builds of torch 2.9.1 exist but want newer drivers, and cu124
# has no 2.9.1 build at all.
cuda_tag=cu126
training=0

while (($#)); do
    case "$1" in
        --cpu) use_cuda=no ;;
        --cuda) use_cuda=yes ;;
        --cuda-tag) cuda_tag="${2:?--cuda-tag needs a value, for example cu128}"; shift ;;
        --training) training=1 ;;
        -h | --help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

python="${PYTHON:-python3}"
if ! version="$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"; then
    echo "No Python found at '$python'. Install Python 3.10-3.14 or set PYTHON." >&2
    exit 1
fi
if [[ ! "$version" =~ ^3\.(1[0-4])$ ]]; then
    echo "Python $version is not supported. Install Python 3.10-3.14 or set PYTHON." >&2
    exit 1
fi
if ! "$python" -c 'import tkinter' 2>/dev/null; then
    echo "Python $version has no tkinter, which the desktop app needs." >&2
    echo "Install it with your package manager (Arch: tk, Debian/Ubuntu: python3-tk, Fedora: python3-tkinter)." >&2
    exit 1
fi
echo "python       $python ($version)"

# Optional: EasyChartGenerator (MIT) backs the `--reducer easygen` option and
# ChartFormats (CC0) is the .chart spec, kept for reference.
declare -A vendor_repos=(
    [vendor/EasyChartGenerator]=https://github.com/Eerovil/EasyChartGenerator.git
    [vendor/ChartFormats]=https://github.com/TheNathannator/GuitarGame_ChartFormats.git
)
for path in "${!vendor_repos[@]}"; do
    [[ -d "$path" ]] && continue
    git clone --quiet --depth 1 "${vendor_repos[$path]}" "$path" \
        || echo "vendor       $path skipped (clone failed, optional)"
done

has_nvidia=no
nvidia-smi >/dev/null 2>&1 && has_nvidia=yes
[[ "$use_cuda" == auto ]] && use_cuda="$has_nvidia"
if [[ "$use_cuda" == yes ]]; then
    index="https://download.pytorch.org/whl/$cuda_tag"
    build="CUDA ($cuda_tag)"
else
    index="https://download.pytorch.org/whl/cpu"
    build="CPU build"
fi
echo "gpu          NVIDIA detected: $has_nvidia -> $build"

[[ -d .venv ]] || "$python" -m venv .venv
py="$root/.venv/bin/python"
"$py" -m pip install --quiet --upgrade pip

echo "installing torch (a few minutes) ..."
if ! "$py" -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url "$index"; then
    echo "torch install failed. For a CUDA driver mismatch, retry with --cuda-tag cu128 or --cpu." >&2
    exit 1
fi

echo "installing everything else ..."
"$py" -m pip install -r requirements.txt

# The basic-pitch wheel asks for a TensorFlow that does not exist for modern
# Python, so pip refuses it. Its ONNX backend needs only onnxruntime, which
# requirements.txt already installs.
echo "installing basic-pitch ..."
"$py" -m pip install basic-pitch==0.4.0 --no-deps

# madmom (the beat tracker) has no release that builds on current Python. Its
# git master does, against the numpy already installed, once cython is there.
echo "installing madmom ..."
"$py" -m pip install cython==3.3.0
"$py" -m pip install --no-build-isolation --no-deps \
    "madmom @ git+https://github.com/CPJKU/madmom@27f032e8947204902c675e5e341a3faf5dc86dae"

if ((training)); then
    echo "installing training stack ..."
    "$py" -m pip install pytorch-lightning==2.5.6 lightning==2.5.5 \
        hydra-core==1.3.2 omegaconf==2.3.0 torchmetrics==1.8.2 wandb==0.22.2 \
        tensorboard==2.21.0
fi

echo "--- verifying ---"
"$py" - <<'EOF'
import sys, torch, tkinter
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__)
print('tkinter      ok')
if torch.cuda.is_available():
    print('gpu         ', torch.cuda.get_device_name(0))
else:
    print('gpu          none: generating charts works, training does not')
EOF
"$py" -m tests.test_chartgen

echo "Ready. Start the app with scripts/chartgen.sh."
echo "The first charts download the separation and lyric models (about 2.5 GB) automatically."
