#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=compare_sweep
#SBATCH --output=logs/%j_sweep_out.txt
#SBATCH --error=logs/%j_sweep_err.txt
#SBATCH --time=06:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
DMD_CKPT=model_checkpoints/pytorch_model.bin
EDM_CKPT=model_checkpoints/edm-imagenet-64x64-cond-adm.pkl

for S_CHURN in 0 1 5 10 40; do
    OUTPUT_DIR=./comparison_output/s_churn_${S_CHURN}
    mkdir -p "$OUTPUT_DIR"

    echo "========== S_churn = ${S_CHURN} =========="
    $PYTHON -u demo/compare_dmd_edm.py \
        --dmd_checkpoint $DMD_CKPT \
        --edm_checkpoint $EDM_CKPT \
        --output_dir "$OUTPUT_DIR" \
        --num_images 30 \
        --edm_steps 256 \
        --master_seed 421 \
        --S_churn $S_CHURN
    echo "========== Done S_churn = ${S_CHURN} =========="
done

echo "All 5 experiments complete."
