#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tsne_edm_1_vs_N
#SBATCH --output=logs/%j_tsne1N_out.txt
#SBATCH --error=logs/%j_tsne1N_err.txt
#SBATCH --time=3:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs comparison_output/tsne_edm_1_vs_N

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
EDM_CKPT=model_checkpoints/edm-imagenet-64x64-cond-adm.pkl
DMD_CKPT=model_checkpoints/pytorch_model.bin

# Target class (e.g. 207 = golden retriever).
CLASS_INDEX=207

# Number of shared noise seeds (pairs).
NUM_SEEDS=100

# Master RNG seed for generating the per-sample noise seeds.
MASTER_SEED=42

# Multi-step EDM sampling parameter (N).
STEPS_N=256

OUTPUT_DIR=./comparison_output/tsne_edm_1_vs_N

echo "Running EDM 1-step / EDM N-step / DMD t-SNE analysis (same noise seeds)..."
echo "  Class index: ${CLASS_INDEX}"
echo "  Num seeds: ${NUM_SEEDS}"
echo "  Master seed: ${MASTER_SEED}"
echo "  N-step: ${STEPS_N}"
echo "  Output: edm_1_images / edm_N_images / dmd_images"

srun -u $PYTHON -u demo/tsne_edm_1_vs_N_imagenet.py \
    --edm_checkpoint $EDM_CKPT \
    --dmd_checkpoint $DMD_CKPT \
    --class_index $CLASS_INDEX \
    --num_seeds $NUM_SEEDS \
    --master_seed $MASTER_SEED \
    --steps_N $STEPS_N \
    --output_dir "$OUTPUT_DIR"

echo "Done. Check ${OUTPUT_DIR}/${CLASS_INDEX}_*/ for results."

