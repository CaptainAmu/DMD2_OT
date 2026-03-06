#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=edm_vs_cm
#SBATCH --output=logs/%j_edm_cm_out.txt
#SBATCH --error=logs/%j_edm_cm_err.txt
#SBATCH --time=6:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs comparison_output/edm_vs_cm

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
EDM_CKPT=model_checkpoints/edm-imagenet-64x64-cond-adm.pkl

# Target class (default: 248 = husky)
CLASS_INDEX=248

# Number of images per model (shared noise seeds)
NUM_SEEDS=500

# Master RNG seed for per-sample noise seeds
MASTER_SEED=42

# EDM sampling parameters
EDM_STEPS=256
S_CHURN=0

OUTPUT_DIR=./comparison_output/edm_vs_cm

echo "Running EDM 1-step / EDM N-step / CM generation (same noise seeds)..."
echo "  Class index: ${CLASS_INDEX}"
echo "  Num seeds: ${NUM_SEEDS}"
echo "  Master seed: ${MASTER_SEED}"
echo "  EDM N-step: ${EDM_STEPS}, S_churn=${S_CHURN}"
echo "  Output: ${OUTPUT_DIR} (edm_1_images / edm_N_images / cm_images)"

srun -u $PYTHON -u demo/generate_edm_vs_cm.py \
    --edm_checkpoint $EDM_CKPT \
    --class_index $CLASS_INDEX \
    --num_seeds $NUM_SEEDS \
    --master_seed $MASTER_SEED \
    --edm_steps $EDM_STEPS \
    --S_churn $S_CHURN \
    --output_dir "$OUTPUT_DIR"

echo "Done. Check ${OUTPUT_DIR}/${CLASS_INDEX}_*/edm_1_images, edm_N_images, cm_images"
