#!/bin/bash
# Download MeanFlow / EDM2 models to model_checkpoints/
# Run from DMD2_OT: bash scripts/download_meanflow_models.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p model_checkpoints

EDM2_ROOT="https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"

# ---------------------------------------------------------------------------
# ImageNet 64x64 (pixel space) - teacher (S) and student (XS)
# ---------------------------------------------------------------------------
echo "Downloading EDM2 ImageNet 64x64 models..."
wget -q --show-progress -O model_checkpoints/edm2-img64-s-teacher.pkl \
    "${EDM2_ROOT}/edm2-img64-s-1073741-0.045.pkl"
wget -q --show-progress -O model_checkpoints/edm2-img64-xs-student.pkl \
    "${EDM2_ROOT}/edm2-img64-xs-0134217-0.110.pkl"

# ---------------------------------------------------------------------------
# ImageNet 512x512 (latent space, 4ch) - teacher and student
# ---------------------------------------------------------------------------
echo "Downloading EDM2 ImageNet 512x512 models..."
wget -q --show-progress -O model_checkpoints/edm2-img512-s-teacher.pkl \
    "${EDM2_ROOT}/edm2-img512-s-2147483-0.070.pkl"
wget -q --show-progress -O model_checkpoints/edm2-img512-xs-student.pkl \
    "${EDM2_ROOT}/edm2-img512-xs-0134217-0.125.pkl"

echo "Done. Models saved to model_checkpoints/"
echo ""
echo "NOTE: ImageNet 256x256 MeanFlow models are NOT publicly released by NVlabs."
echo "      For 256 resolution, you need to:"
echo "      1. Train a MeanFlow teacher on ImageNet 256 (Re-MeanFlow repo), or"
echo "      2. Obtain checkpoints from the Re-MeanFlow / EDM2 authors."
echo ""
echo "For 64x64 comparison, use:"
echo "  TEACHER_CKPT=model_checkpoints/edm2-img64-s-teacher.pkl"
echo "  STUDENT_CKPT=model_checkpoints/edm2-img64-xs-student.pkl"
echo "  (Also set RESOLUTION=64 in compare_teacher_meanflow.py)"
