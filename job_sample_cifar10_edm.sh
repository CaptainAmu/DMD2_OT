#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=sample_cifar10_edm
#SBATCH --output=logs/%j_sample_cifar10_edm_out.txt
#SBATCH --error=logs/%j_sample_cifar10_edm_err.txt
#SBATCH --time=00:30:00
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs cifar10_edm_samples

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
TEACHER_PATH=model_checkpoints/edm-cifar10-32x32-cond-vp.pkl
OUT_DIR=cifar10_edm_samples

srun -u $PYTHON -m demo.cifar10_edm_sample \
    --teacher_path "$TEACHER_PATH" \
    --out_dir "$OUT_DIR" \
    --num_images 64 \
    --steps 18 \
    --seed 0

