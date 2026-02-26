#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=evaluate
#SBATCH --output=logs/%j_eval_out.txt
#SBATCH --error=logs/%j_eval_err.txt
#SBATCH --time=00:30:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python

srun -u $PYTHON -u demo/evaluate.py \
    --comparison_dir ./comparison_output \
    --master_seed 42 \
    --num_images 30 \
    --clip_model ViT-B/32
