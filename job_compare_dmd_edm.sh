#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=compare_dmd_edm
#SBATCH --output=logs/%j_compare_out.txt
#SBATCH --error=logs/%j_compare_err.txt
#SBATCH --time=06:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs comparison_output

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
DMD_CKPT=model_checkpoints/pytorch_model.bin
EDM_CKPT=model_checkpoints/edm-imagenet-64x64-cond-adm.pkl

# ---- Control S_churn here ----
S_CHURN=0

srun -u $PYTHON -u demo/compare_dmd_edm.py \
    --dmd_checkpoint $DMD_CKPT \
    --edm_checkpoint $EDM_CKPT \
    --output_dir ./comparison_output \
    --num_images 30 \
    --edm_steps 256 \
    --master_seed 42 \
    --S_churn $S_CHURN
