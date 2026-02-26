#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=sample_dmd_edm
#SBATCH --output=logs/%j_sample_out.txt
#SBATCH --error=logs/%j_sample_err.txt
#SBATCH --time=01:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs sample_output

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
DMD_CKPT=model_checkpoints/pytorch_model.bin
EDM_CKPT=model_checkpoints/edm-imagenet-64x64-cond-adm.pkl

srun -u $PYTHON -u demo/sample_dmd_edm.py \
    --dmd_checkpoint $DMD_CKPT \
    --edm_checkpoint $EDM_CKPT \
    --output_dir ./sample_output \
    --class_index 207 \
    --seed 42 \
    --edm_steps 256
